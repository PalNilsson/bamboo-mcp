"""Tests for the ``BAMBOO_TOOL_PROFILE`` mechanism.

Covers :mod:`bamboo.tools._tool_profiles` and its two consumers — the wire
surface in ``bamboo.core.list_tools`` and the planner catalog in
``bamboo.tools.planner._collect_tool_catalog``.

Both consumers are exercised through the real code paths rather than through a
local re-implementation of the rule.  ``tests/test_plugin_tool_filter.py``
tests the sibling namespace filter by copying its logic into the test file,
which cannot catch drift in ``core.py``; that pattern is deliberately not
repeated here.

Reaching ``list_tools`` takes two pieces of setup.  It is a closure registered
by ``create_server`` on a ``Server`` that ``conftest`` has replaced with a
``MagicMock``, so the decorator returns a mock and the function is not
reachable from the returned app — it is captured from the decorator call
instead.  Its return shape then depends on ``inspect.isclass`` checks against
``Tool`` and ``ListToolsResult``, both of which ``conftest`` sets to
``MagicMock``; replacing them with *instances* makes both checks false and
``list_tools`` returns the plain-dict branch, where the filtering is directly
observable.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import bamboo.core as core
import bamboo.tools.planner as planner_mod
from bamboo.tools import _tool_profiles as tp


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an unset variable and an empty warning memo.

    Args:
        monkeypatch: Pytest environment patcher.

    Returns:
        None.
    """
    monkeypatch.delenv(tp.ENV_VAR, raising=False)
    tp.reset_profile_warnings()


class _FakeTool:
    """Minimal tool object exposing a fixed definition.

    Attributes:
        _defn: The definition returned by :meth:`get_definition`.
    """

    def __init__(self, defn: dict[str, Any]) -> None:
        """Store the definition to advertise.

        Args:
            defn: Tool definition dict.
        """
        self._defn = defn

    def get_definition(self) -> dict[str, Any]:
        """Return the stored definition.

        Returns:
            The definition dict supplied at construction.
        """
        return self._defn


def _defn(name: str, profiles: Any = None) -> dict[str, Any]:
    """Build a tool definition, optionally carrying a ``profiles`` key.

    Args:
        name: Tool name.
        profiles: Value for the ``profiles`` key.  Omitted entirely when
            ``None``, which is the profile-agnostic case, distinct from an
            explicit empty list.

    Returns:
        A tool definition dict.
    """
    out: dict[str, Any] = {
        "name": name,
        "description": f"{name} description",
        "inputSchema": {"type": "object", "properties": {}},
    }
    if profiles is not None:
        out[tp.DEFINITION_KEY] = profiles
    return out


async def _wire_names(tools: dict[str, Any], ep_defs: list[dict[str, Any]]) -> list[str]:
    """Run the real ``list_tools`` closure and return the names it advertises.

    Args:
        tools: Replacement for ``bamboo.core.TOOLS``.
        ep_defs: Replacement return value for entry-point discovery.

    Returns:
        The ``name`` of every advertised definition, in listing order.
    """
    captured: list[Any] = []

    def _capture_decorator() -> Any:
        def _decorate(fn: Any) -> Any:
            captured.append(fn)
            return fn
        return _decorate

    app = MagicMock()
    app.list_tools = _capture_decorator

    with patch.object(core, "Server", return_value=app), \
            patch.dict(core.TOOLS, tools, clear=True), \
            patch.object(core, "_load_entrypoint_tool_definitions", return_value=ep_defs), \
            patch.object(core, "Tool", MagicMock()), \
            patch.object(core, "ListToolsResult", MagicMock()):
        core.create_server()
        assert captured, "list_tools was not registered"
        result = await captured[0]()

    return [d["name"] for d in result]


# --------------------------------------------------------------------------
# active_profile / expand_profile
# --------------------------------------------------------------------------


class TestActiveProfile:
    """Resolution of the environment variable to a server profile."""

    def test_unset_defaults_to_orchestrated(self) -> None:
        """An unset variable selects the default without complaint."""
        assert tp.active_profile() == tp.PROFILE_ORCHESTRATED

    @pytest.mark.parametrize("value", ["orchestrated", "primitive", "both"])
    def test_each_recognised_value_is_returned(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every documented value round-trips.

        Args:
            value: A recognised server profile.
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv(tp.ENV_VAR, value)
        assert tp.active_profile() == value

    @pytest.mark.parametrize("value", ["  PRIMITIVE  ", "Both", "\tprimitive\n"])
    def test_case_and_whitespace_are_tolerated(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shell quoting accidents do not change the profile.

        ``list_tools`` already normalises ``ASKPANDA_PLUGIN`` the same way.

        Args:
            value: A recognised value with stray case or whitespace.
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv(tp.ENV_VAR, value)
        assert tp.active_profile() == value.strip().lower()

    def test_empty_string_defaults_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``BAMBOO_TOOL_PROFILE=`` is "unset", not "invalid".

        Args:
            monkeypatch: Pytest environment patcher.
            caplog: Pytest log capture.
        """
        monkeypatch.setenv(tp.ENV_VAR, "   ")
        with caplog.at_level(logging.WARNING):
            assert tp.active_profile() == tp.PROFILE_ORCHESTRATED
        assert caplog.records == []

    def test_unrecognised_value_warns_and_defaults(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo must be visible in the log, not silently ignored.

        Args:
            monkeypatch: Pytest environment patcher.
            caplog: Pytest log capture.
        """
        monkeypatch.setenv(tp.ENV_VAR, "primitve")
        with caplog.at_level(logging.WARNING):
            assert tp.active_profile() == tp.PROFILE_ORCHESTRATED
        assert len(caplog.records) == 1
        assert "primitve" in caplog.records[0].getMessage()

    def test_unrecognised_value_warns_only_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The memo exists so a misconfigured server does not flood its log.

        ``active_profile`` runs once per ``tools/list`` and once per plan.

        Args:
            monkeypatch: Pytest environment patcher.
            caplog: Pytest log capture.
        """
        monkeypatch.setenv(tp.ENV_VAR, "compound")
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                tp.active_profile()
        assert len(caplog.records) == 1

    def test_reset_profile_warnings_restores_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The memo is clearable, or tests would leak state into each other.

        Args:
            monkeypatch: Pytest environment patcher.
            caplog: Pytest log capture.
        """
        monkeypatch.setenv(tp.ENV_VAR, "compound")
        with caplog.at_level(logging.WARNING):
            tp.active_profile()
            tp.reset_profile_warnings()
            tp.active_profile()
        assert len(caplog.records) == 2


class TestExpandProfile:
    """Mapping from a server profile to the definition profiles it activates."""

    def test_orchestrated_activates_only_orchestrated(self) -> None:
        """The default profile must not activate primitives."""
        assert tp.expand_profile(tp.PROFILE_ORCHESTRATED) == frozenset(
            {tp.PROFILE_ORCHESTRATED}
        )

    def test_primitive_activates_only_primitive(self) -> None:
        """Primitive mode must not also advertise the compound surface."""
        assert tp.expand_profile(tp.PROFILE_PRIMITIVE) == frozenset(
            {tp.PROFILE_PRIMITIVE}
        )

    def test_both_activates_every_tool_profile(self) -> None:
        """``both`` is the union, and is the only value that is.

        This is the property that distinguishes ``both`` from the other two; if
        it collapsed to a single profile the mode would be redundant.
        """
        assert tp.expand_profile(tp.PROFILE_BOTH) == tp.TOOL_PROFILES
        assert len(tp.TOOL_PROFILES) == 2

    def test_both_is_not_a_tool_profile(self) -> None:
        """A definition may never name ``both``; it describes a server."""
        assert tp.PROFILE_BOTH in tp.SERVER_PROFILES
        assert tp.PROFILE_BOTH not in tp.TOOL_PROFILES

    def test_unknown_name_expands_as_the_default(self) -> None:
        """Defence in depth: validation lives in ``active_profile``."""
        assert tp.expand_profile("nonsense") == tp.expand_profile(tp.DEFAULT_PROFILE)


# --------------------------------------------------------------------------
# definition_profiles / is_advertised
# --------------------------------------------------------------------------


class TestDefinitionProfiles:
    """Reading the ``profiles`` key off a tool definition."""

    def test_declared_profiles_are_returned(self) -> None:
        """The ordinary case: a definition naming one profile."""
        assert tp.definition_profiles(_defn("t", ["primitive"])) == frozenset(
            {tp.PROFILE_PRIMITIVE}
        )

    def test_members_are_normalised(self) -> None:
        """Case and whitespace in a definition are tolerated as in the env."""
        assert tp.definition_profiles(_defn("t", [" Primitive "])) == frozenset(
            {tp.PROFILE_PRIMITIVE}
        )

    @pytest.mark.parametrize(
        "profiles",
        [
            pytest.param(None, id="key-absent"),
            pytest.param([], id="empty-list"),
            pytest.param("primitive", id="bare-string"),
            pytest.param(["primitve", "compound"], id="all-members-unknown"),
            pytest.param(17, id="not-a-sequence"),
            pytest.param([None, 3], id="non-string-members"),
        ],
    )
    def test_unusable_values_are_profile_agnostic(self, profiles: Any) -> None:
        """All four routes to "names no profile" collapse to the empty set.

        The bare-string case is the subtle one: ``"primitive"`` is iterable, so
        an iterability test would decompose it into characters.  It is rejected
        by not being one of the accepted sequence types instead.

        Args:
            profiles: A ``profiles`` value that names nothing usable.
        """
        assert tp.definition_profiles(_defn("t", profiles)) == frozenset()

    def test_unknown_members_are_dropped_but_known_ones_kept(self) -> None:
        """A partly-mistyped list keeps whatever was spelled correctly."""
        assert tp.definition_profiles(
            _defn("t", ["primitve", "orchestrated"])
        ) == frozenset({tp.PROFILE_ORCHESTRATED})

    def test_non_dict_definition_is_tolerated(self) -> None:
        """One malformed plugin must not break discovery for the rest."""
        assert tp.definition_profiles("not a definition") == frozenset()
        assert tp.definition_profiles(None) == frozenset()


class TestIsAdvertised:
    """The advertising predicate itself."""

    def test_agnostic_definition_is_advertised_under_every_profile(self) -> None:
        """No ``profiles`` key means the tool is on every surface."""
        defn = _defn("panda_job_status")
        for profile in tp.SERVER_PROFILES:
            assert tp.is_advertised(defn, tp.expand_profile(profile))

    def test_declared_definition_is_withheld_when_inactive(self) -> None:
        """A primitive is not advertised in orchestrated mode."""
        defn = _defn("atlas.log.fetch_text", ["primitive"])
        assert not tp.is_advertised(defn, tp.expand_profile(tp.PROFILE_ORCHESTRATED))

    def test_declared_definition_is_advertised_when_active(self) -> None:
        """The same primitive is advertised in primitive mode."""
        defn = _defn("atlas.log.fetch_text", ["primitive"])
        assert tp.is_advertised(defn, tp.expand_profile(tp.PROFILE_PRIMITIVE))

    def test_both_advertises_the_compound_and_the_primitive(self) -> None:
        """``both`` is what lets an agent choose between the two surfaces."""
        active = tp.expand_profile(tp.PROFILE_BOTH)
        assert tp.is_advertised(_defn("panda_log_analysis", ["orchestrated"]), active)
        assert tp.is_advertised(_defn("atlas.log.fetch_text", ["primitive"]), active)

    def test_partial_overlap_is_enough(self) -> None:
        """Naming several profiles advertises under any one of them."""
        defn = _defn("t", ["orchestrated", "primitive"])
        assert tp.is_advertised(defn, frozenset({tp.PROFILE_PRIMITIVE}))

    def test_empty_active_set_still_advertises_agnostic_tools(self) -> None:
        """Fail-open holds even against a degenerate active set.

        The alternative rule — treating "names nothing" as "advertised
        nowhere" — would empty the entire catalog here rather than only the
        tools that opted in.
        """
        assert tp.is_advertised(_defn("t"), frozenset())
        assert not tp.is_advertised(_defn("t", ["primitive"]), frozenset())


# --------------------------------------------------------------------------
# Consumer: the wire surface
# --------------------------------------------------------------------------


class TestListToolsAppliesTheProfile:
    """``bamboo.core.list_tools`` honours ``BAMBOO_TOOL_PROFILE``."""

    _BUILTINS = {
        "bamboo_health": _FakeTool(_defn("bamboo_health")),
        "panda_log_analysis": _FakeTool(_defn("panda_log_analysis", ["orchestrated"])),
    }
    _EP_DEFS = [
        _defn("atlas.core_dump_analysis"),
        _defn("atlas.log.fetch_text", ["primitive"]),
        _defn("epic.doc_search"),
    ]

    @pytest.mark.asyncio
    async def test_orchestrated_hides_primitives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default surface is exactly today's surface.

        Args:
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv("ASKPANDA_PLUGIN", "atlas")
        monkeypatch.setenv(tp.ENV_VAR, "orchestrated")
        names = await _wire_names(self._BUILTINS, self._EP_DEFS)
        assert names == [
            "bamboo_health",
            "panda_log_analysis",
            "atlas.core_dump_analysis",
        ]

    @pytest.mark.asyncio
    async def test_primitive_hides_the_monolith(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Primitive mode swaps one surface for the other, keeping agnostics.

        Args:
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv("ASKPANDA_PLUGIN", "atlas")
        monkeypatch.setenv(tp.ENV_VAR, "primitive")
        names = await _wire_names(self._BUILTINS, self._EP_DEFS)
        assert names == [
            "bamboo_health",
            "atlas.core_dump_analysis",
            "atlas.log.fetch_text",
        ]

    @pytest.mark.asyncio
    async def test_both_advertises_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In ``both`` mode the namespace filter is the only one left.

        Args:
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv("ASKPANDA_PLUGIN", "atlas")
        monkeypatch.setenv(tp.ENV_VAR, "both")
        names = await _wire_names(self._BUILTINS, self._EP_DEFS)
        assert names == [
            "bamboo_health",
            "panda_log_analysis",
            "atlas.core_dump_analysis",
            "atlas.log.fetch_text",
        ]

    @pytest.mark.asyncio
    async def test_profiles_key_never_reaches_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The filter reads ``profiles``; the projection then drops it.

        Args:
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv("ASKPANDA_PLUGIN", "atlas")
        monkeypatch.setenv(tp.ENV_VAR, "both")

        captured: list[Any] = []

        def _capture_decorator() -> Any:
            def _decorate(fn: Any) -> Any:
                captured.append(fn)
                return fn
            return _decorate

        app = MagicMock()
        app.list_tools = _capture_decorator

        ep_patch = patch.object(
            core, "_load_entrypoint_tool_definitions", return_value=self._EP_DEFS
        )
        with patch.object(core, "Server", return_value=app), \
                patch.dict(core.TOOLS, self._BUILTINS, clear=True), \
                ep_patch, \
                patch.object(core, "Tool", MagicMock()), \
                patch.object(core, "ListToolsResult", MagicMock()):
            core.create_server()
            result = await captured[0]()

        assert all(tp.DEFINITION_KEY not in d for d in result)

    @pytest.mark.asyncio
    async def test_the_profile_adds_only_primitives_to_the_real_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The profile widens the surface; it never narrows it.

        Asserted against the real ``TOOLS`` registry and real entry points.
        Until B4 this test read that *no* definition opted into a profile, and
        it was written to fail the moment one did rather than be silently
        satisfied — which is what ``atlas.log.plan_fetch`` made it do.  The
        invariant it now pins is the one that matters going forward: the
        orchestrated surface is a subset of the other two, and everything the
        other two add is a primitive.

        The orchestrated surface must stay *exactly* what it was, because it
        is what Bamboo's own planner and every existing client see.  It would
        narrow if ``panda_log_analysis`` were given ``profiles:
        ["orchestrated"]`` — harmless — but also if a typo gave it
        ``["primitve"]``, which fails open to profile-agnostic by design
        (D-7), so this test would not catch that.  ``both`` is the union and
        must therefore equal ``primitive`` here, since nothing declares
        ``orchestrated``.

        Args:
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv("ASKPANDA_PLUGIN", "atlas")
        surfaces: dict[str, list[str]] = {}
        for profile in ("orchestrated", "primitive", "both"):
            monkeypatch.setenv(tp.ENV_VAR, profile)
            surfaces[profile] = await _wire_names(
                dict(core.TOOLS), core._load_entrypoint_tool_definitions()
            )

        assert surfaces["orchestrated"], "expected a non-empty tool surface"
        orchestrated = set(surfaces["orchestrated"])
        primitive = set(surfaces["primitive"])

        assert orchestrated <= primitive
        assert primitive == set(surfaces["both"])

        # Everything the primitive profile adds is a log primitive, and the
        # compound tool remains available under every profile until B6 gives
        # it a profile of its own.
        added = primitive - orchestrated
        assert all(name.startswith("atlas.log.") for name in added), added
        for profile in ("orchestrated", "primitive", "both"):
            assert "panda_log_analysis" in surfaces[profile]


# --------------------------------------------------------------------------
# Consumer: the planner catalog
# --------------------------------------------------------------------------


class TestPlannerCatalogExcludesPrimitives:
    """``_collect_tool_catalog`` is pinned to the orchestrated surface."""

    _BUILTINS = {
        "panda_log_analysis": _FakeTool(_defn("panda_log_analysis", ["orchestrated"])),
        "panda_job_status": _FakeTool(_defn("panda_job_status")),
        "atlas_log_fetch_text": _FakeTool(_defn("atlas.log.fetch_text", ["primitive"])),
    }

    def _catalog_names(self) -> list[str]:
        """Collect catalog names with the fake registry and no entry points.

        Returns:
            Tool names the planner would be shown.
        """
        with patch.dict(core.TOOLS, self._BUILTINS, clear=True), \
                patch.object(planner_mod, "wire_tool_definitions", return_value=[]):
            return [e["name"] for e in planner_mod._collect_tool_catalog()]

    @pytest.mark.parametrize("profile", ["orchestrated", "primitive", "both"])
    def test_primitives_are_never_catalogued(
        self, profile: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalog ignores the environment entirely.

        Following ``BAMBOO_TOOL_PROFILE`` here would drop
        ``panda_log_analysis`` from the catalog under ``primitive`` while
        ``bamboo_answer`` stayed advertised and callable, breaking every
        planner-routed question.  It is also what keeps primitives out of the
        Track A retrieval index unconditionally.

        Args:
            profile: Server profile to set before collecting.
            monkeypatch: Pytest environment patcher.
        """
        monkeypatch.setenv(tp.ENV_VAR, profile)
        names = self._catalog_names()
        assert "atlas.log.fetch_text" not in names
        assert "panda_log_analysis" in names
        assert "panda_job_status" in names

    def test_entry_point_primitives_are_excluded_too(self) -> None:
        """Both catalog sources are filtered, not just the built-in one."""
        ep_defs = [
            _defn("atlas.core_dump_analysis"),
            _defn("atlas.log.plan_fetch", ["primitive"]),
        ]
        wire_patch = patch.object(
            planner_mod, "wire_tool_definitions", return_value=ep_defs
        )
        with patch.dict(core.TOOLS, {}, clear=True), wire_patch:
            names = [
                e["name"]
                for e in planner_mod._collect_tool_catalog(namespaces=["atlas"])
            ]
        assert names == ["atlas.core_dump_analysis"]

    def test_real_catalog_is_unchanged(self) -> None:
        """B2 is inert on the planner side as well as the wire."""
        assert planner_mod._collect_tool_catalog(namespaces=["atlas"])


class TestRealPrimitiveIsGated:
    """Guards against the *installed* ``atlas.log.plan_fetch`` entry point.

    The rest of this module drives the rule with synthetic definitions, which
    proves the mechanism but not the wiring.  These two tests read the real
    registry, so a definition that lost its ``profiles`` key, or an entry point
    whose dotted name stopped resolving, fails here rather than silently
    widening the planner catalog.

    Both skip when the plugin is not installed: entry points are unavailable
    in a source-only checkout, which is the pre-existing container caveat that
    also affects ``tests/test_tool_name_canon.py``.
    """

    _NAME = "atlas.log.plan_fetch"

    def _real_definition(self) -> dict[str, Any]:
        """Return the installed primitive's wire definition.

        Returns:
            The definition dict.
        """
        from bamboo.tools._tool_names import wire_tool_definitions

        for defn in wire_tool_definitions():
            if defn.get("name") == self._NAME:
                return defn
        pytest.skip(f"{self._NAME} entry point is not installed")
        raise AssertionError("unreachable")

    def test_the_real_primitive_declares_the_primitive_profile(self) -> None:
        """The definition must restrict itself, or the gate does nothing."""
        defn = self._real_definition()
        assert tp.definition_profiles(defn) == frozenset({tp.PROFILE_PRIMITIVE})

    def test_the_real_primitive_is_gated_by_profile(self) -> None:
        """Advertised under ``primitive`` and ``both``, withheld under ``orchestrated``."""
        defn = self._real_definition()
        assert not tp.is_advertised(defn, tp.expand_profile(tp.PROFILE_ORCHESTRATED))
        assert tp.is_advertised(defn, tp.expand_profile(tp.PROFILE_PRIMITIVE))
        assert tp.is_advertised(defn, tp.expand_profile(tp.PROFILE_BOTH))

    def test_the_real_primitive_never_reaches_the_planner_catalog(self) -> None:
        """The catalog is pinned to ``orchestrated``, so the primitive is absent."""
        self._real_definition()  # skip early when not installed
        names = [
            entry["name"]
            for entry in planner_mod._collect_tool_catalog(namespaces=["atlas"])
        ]
        assert self._NAME not in names
        assert "panda_log_analysis" in names
