"""Tests for the planner's structured routing guidance.

The planner system prompt pairs a hard rule — *"Only propose tools that appear
in the provided tool catalog"* — with guidance naming roughly fifteen specific
tools.  When the two disagree the prompt contradicts itself, and the planner
has been observed resolving the contradiction by discarding the guidance: the
catalog once advertised ``core_dump_analysis`` while the guidance said
``atlas.core_dump_analysis``, so an explicit request to analyse a core dump was
answered with a log analysis instead.

Nothing detected that.  These tests do, in three layers:

* :class:`TestGuidanceMatchesCatalog` pins every guidance-named tool against
  the catalog the planner is actually shown, so the drift above cannot recur
  silently.
* :class:`TestRuleToolsMatchText` pins each rule's declared ``tools`` against
  the names appearing literally in its ``text``, so editing one without the
  other goes red rather than quietly disabling the filter for that rule.
* :class:`TestRenderRoutingGuidance` pins the filtering predicate itself,
  including the mutations it would be easiest to introduce.
"""
from __future__ import annotations

import pytest

from bamboo.tools.planner import (
    RoutingRule,
    _ATLAS_ROUTING_RULES,
    _CGSIM_ROUTING_RULES,
    _collect_tool_catalog,
    _render_routing_guidance,
    build_planner_system_prompt,
    get_plan_json_schema,
    routing_rules_for_plugin,
)


def _catalog_names(namespaces: list[str] | None = None) -> frozenset[str]:
    """Return the wire names in the planner's tool catalog.

    Args:
        namespaces: Optional namespace filter, passed through to
            :func:`~bamboo.tools.planner._collect_tool_catalog`.

    Returns:
        FrozenSet[str]: Catalog tool names.
    """
    return frozenset(str(entry["name"]) for entry in _collect_tool_catalog(namespaces=namespaces))


def _rule_named_tools(rules: tuple[RoutingRule, ...]) -> frozenset[str]:
    """Return every tool named across a rule table.

    Args:
        rules: Routing rules.

    Returns:
        FrozenSet[str]: Union of the rules' declared tool sets.
    """
    return frozenset().union(*(rule.tools for rule in rules)) if rules else frozenset()


def _legitimately_absent() -> frozenset[str]:
    """Return guidance-named tools whose absence is explained by the environment.

    Three of the built-in PanDA tools register only when ``duckdb`` is
    importable, and the ATLAS plugin's entry points reach the catalog only when
    that package is installed.  The quality gate runs the suite both with and
    without the optional dependencies, so the catalog assertions must tolerate
    those absences without weakening into a no-op: each name is excused
    individually, and only on evidence that its backing dependency or package
    is genuinely missing.

    Returns:
        FrozenSet[str]: Names excused from the catalog assertions in this
        environment.  Empty on a fully-provisioned install.
    """
    excused: set[str] = set()

    try:
        from bamboo import core as _core
    except Exception:  # pragma: no cover - core is importable in every env we run
        return frozenset()

    if not getattr(_core, "_JOBS_QUERY_AVAILABLE", False):
        excused.add("panda_jobs_query")
    if not getattr(_core, "_HARVESTER_WORKERS_AVAILABLE", False):
        excused.add("panda_harvester_workers")
    if not getattr(_core, "_PANDA_SERVER_HEALTH_AVAILABLE", False):
        excused.add("panda_server_health")

    # The ATLAS plugin either reaches the catalog or it does not; excuse its
    # tools only when no atlas.* name made it in at all, so a plugin that is
    # installed but has lost a single entry point still fails.
    if not any(name.startswith("atlas.") for name in _catalog_names()):
        excused.update(name for name in _rule_named_tools(_ATLAS_ROUTING_RULES) if name.startswith("atlas."))

    return frozenset(excused)


class TestGuidanceMatchesCatalog:
    """Guidance may only name tools the planner can actually propose."""

    def test_every_atlas_guidance_tool_is_in_the_catalog(self) -> None:
        """Each ATLAS rule names only catalogued tools.

        This is the regression test for the ``core_dump_analysis`` /
        ``atlas.core_dump_analysis`` drift.  A stale ``.egg-info`` shadowing a
        plugin's entry points reproduces that drift exactly, so this assertion
        also catches a half-installed environment.
        """
        named = _rule_named_tools(_ATLAS_ROUTING_RULES)
        missing = named - _catalog_names(namespaces=["atlas"]) - _legitimately_absent()
        assert not missing, (
            f"Routing guidance names tools absent from the planner catalog: {sorted(missing)}. "
            "The prompt would instruct the planner to propose a tool its own hard rules forbid. "
            "If a plugin entry point was recently added, reinstall the plugin: a stale "
            ".egg-info shadows pyproject.toml's entry points."
        )

    def test_core_dump_analysis_is_catalogued_under_its_wire_name(self) -> None:
        """``atlas.core_dump_analysis`` is present, and the bare name is not.

        Named explicitly rather than left to the set assertion above because
        this one pair is the documented failure, and because the bare name
        appearing *instead* would satisfy a weaker containment check while
        reintroducing the original bug.
        """
        if "atlas.core_dump_analysis" in _legitimately_absent():
            pytest.skip("ATLAS plugin not installed in this environment")
        catalog = _catalog_names(namespaces=["atlas"])
        assert "atlas.core_dump_analysis" in catalog
        assert "core_dump_analysis" not in catalog

    def test_cgsim_guidance_tools_are_namespaced(self) -> None:
        """Every CGSim rule names tools under the ``cgsim.`` namespace.

        The CGSim catalog is only populated when that plugin is installed, so
        the catalog itself cannot be asserted against in every environment.
        The namespace prefix can be, and a missing prefix is the same class of
        bug as the ATLAS drift.
        """
        named = _rule_named_tools(_CGSIM_ROUTING_RULES)
        assert named
        assert all(name.startswith("cgsim.") for name in named), sorted(named)

    def test_cgsim_guidance_names_no_panda_tools(self) -> None:
        """Plugin isolation: the CGSim prompt leaks no PanDA vocabulary."""
        named = _rule_named_tools(_CGSIM_ROUTING_RULES)
        assert not any(name.startswith(("panda_", "atlas.")) for name in named)

    def test_pinned_fallback_tools_are_catalogued(self) -> None:
        """The universal fallback route's two tools are always present.

        ``panda_doc_search`` and ``panda_doc_bm25`` back the "for ALL other
        questions" clause.  They are unconditional in ``bamboo.core.TOOLS``, so
        unlike the DuckDB-backed tools they have no excused absence.
        """
        catalog = _catalog_names(namespaces=["atlas"])
        assert {"panda_doc_search", "panda_doc_bm25"} <= catalog


class TestRuleToolsMatchText:
    """A rule's declared tools must match the names its text actually uses."""

    @pytest.mark.parametrize("plugin_id", ["atlas", "cgsim"])
    def test_declared_tools_appear_in_the_text(self, plugin_id: str) -> None:
        """Every declared tool is named literally in the clause.

        A declared tool absent from the text over-constrains the filter: the
        clause would be withheld for a tool it never mentions.
        """
        for rule in routing_rules_for_plugin(plugin_id):
            for tool in rule.tools:
                assert tool in rule.text, f"{tool!r} declared but not named in: {rule.text[:80]!r}"

    def test_atlas_text_names_no_undeclared_catalog_tool(self) -> None:
        """No ATLAS clause names a catalogued tool it failed to declare.

        This is the direction that matters.  An undeclared tool is invisible to
        the filter, so the clause survives retrieval dropping that tool and the
        prompt contradicts itself again — the precise failure these rules
        exist to prevent.

        Names are matched longest-first so that declaring the qualified name
        (``atlas.pilot_source_analysis``) is not reported as a stray match on a
        shorter catalog name contained within it.
        """
        catalog = sorted(_catalog_names(namespaces=["atlas"]), key=len, reverse=True)
        for rule in _ATLAS_ROUTING_RULES:
            remaining = rule.text
            for name in catalog:
                if name in remaining:
                    assert name in rule.tools, (
                        f"{name!r} is named in a routing clause but not declared in its tools: "
                        f"{rule.text[:80]!r}"
                    )
                    remaining = remaining.replace(name, "")


class TestRenderRoutingGuidance:
    """The filtering predicate, including its easiest mutations."""

    def test_none_emits_every_clause(self) -> None:
        """``None`` means "do not filter", not "filter against nothing"."""
        rendered = _render_routing_guidance(_ATLAS_ROUTING_RULES, None)
        for rule in _ATLAS_ROUTING_RULES:
            assert rule.text in rendered

    def test_empty_set_emits_nothing(self) -> None:
        """An empty catalog withholds every clause.

        Distinguishes ``None`` from ``frozenset()``.  A predicate written as
        ``if not available_tools`` would collapse the two and emit everything
        here, which is the fail-open shape this codebase keeps rediscovering.
        """
        assert _render_routing_guidance(_ATLAS_ROUTING_RULES, frozenset()) == ""

    def test_full_catalog_matches_unfiltered(self) -> None:
        """Passing the complete name set is identical to not filtering."""
        named = _rule_named_tools(_ATLAS_ROUTING_RULES)
        assert _render_routing_guidance(_ATLAS_ROUTING_RULES, named) == _render_routing_guidance(
            _ATLAS_ROUTING_RULES, None
        )

    def test_partial_overlap_withholds_the_clause(self) -> None:
        """A clause needs *all* its tools, not any of them.

        Mutation guard: ``rule.tools & available_tools`` would keep this clause
        and emit guidance telling the planner to call a tool that is not in the
        catalog.
        """
        rules = (RoutingRule(frozenset({"a", "b"}), "- use a AND b together."),)
        assert _render_routing_guidance(rules, frozenset({"a"})) == ""
        assert _render_routing_guidance(rules, frozenset({"a", "b"})) == "- use a AND b together.\n"

    def test_superset_catalog_keeps_the_clause(self) -> None:
        """Extra catalog entries beyond a clause's tools do not withhold it.

        Mutation guard: an equality test rather than a subset test would
        withhold every clause on a catalog larger than its own tool set, which
        is every real catalog.
        """
        rules = (RoutingRule(frozenset({"a"}), "- use a."),)
        assert _render_routing_guidance(rules, frozenset({"a", "b", "c"})) == "- use a.\n"

    def test_order_is_preserved(self) -> None:
        """Surviving clauses keep their precedence order.

        Rule precedence is load-bearing in this prompt — the broad fallback
        clause must stay last — so a filter that reorders is a behaviour
        change even when it keeps every clause.
        """
        rules = (
            RoutingRule(frozenset({"a"}), "- first."),
            RoutingRule(frozenset({"b"}), "- second."),
            RoutingRule(frozenset({"c"}), "- third."),
        )
        assert _render_routing_guidance(rules, frozenset({"a", "c"})) == "- first.\n- third.\n"

    def test_every_clause_ends_with_a_newline(self) -> None:
        """Rendering terminates each clause, including the last.

        The prompt builder appends a blank line after the guidance block; a
        missing terminator would run the final clause into the JSON Schema
        header.
        """
        rendered = _render_routing_guidance(_ATLAS_ROUTING_RULES, None)
        assert rendered.endswith("\n")
        assert "\n\n" not in rendered.replace("\n  IMPORTANT", "")


class TestSystemPromptAssembly:
    """The assembled prompt keeps its shape under filtering."""

    def test_filtered_prompt_omits_the_withheld_clause(self) -> None:
        """Dropping a tool drops the clause naming it, and nothing else."""
        schema = get_plan_json_schema()
        named = _rule_named_tools(_ATLAS_ROUTING_RULES)
        full = build_planner_system_prompt(schema, plugin_id="atlas", available_tools=named)
        reduced = build_planner_system_prompt(
            schema, plugin_id="atlas", available_tools=named - {"panda_queue_info"}
        )
        assert "panda_queue_info" in full
        assert "panda_queue_info" not in reduced
        assert "Only propose tools that appear in the provided tool catalog." in reduced
        assert "JSON Schema (must match exactly):" in reduced

    def test_default_is_unfiltered(self) -> None:
        """Omitting *available_tools* emits the full guidance.

        Pins the call-site default: this refactor must not change the prompt
        any caller currently receives.
        """
        schema = get_plan_json_schema()
        assert build_planner_system_prompt(schema, plugin_id="atlas") == build_planner_system_prompt(
            schema, plugin_id="atlas", available_tools=None
        )

    @pytest.mark.parametrize("plugin_id", ["", "atlas", "epic", "verarubin"])
    def test_unknown_plugins_fall_back_to_atlas(self, plugin_id: str) -> None:
        """Dispatch is unchanged: only ``cgsim`` diverges."""
        schema = get_plan_json_schema()
        assert build_planner_system_prompt(schema, plugin_id=plugin_id) == build_planner_system_prompt(
            schema, plugin_id="atlas"
        )

    def test_routing_rules_for_plugin_mirrors_prompt_dispatch(self) -> None:
        """The helper the tests use agrees with the builder they test."""
        assert routing_rules_for_plugin("cgsim") is _CGSIM_ROUTING_RULES
        for plugin_id in ("", "atlas", "epic", "verarubin"):
            assert routing_rules_for_plugin(plugin_id) is _ATLAS_ROUTING_RULES
