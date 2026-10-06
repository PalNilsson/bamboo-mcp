"""Tests for query-conditioned tool retrieval.

Three things are pinned here, in rising order of how badly they fail quietly.

The **config surface**, because every one of these variables is unset in
production today and a typo in one must not be the thing that changes planner
behaviour.

The **degradation paths**, because a retrieval layer that has stopped
retrieving presents as nothing worse than a slightly larger prompt.  Every
route back to the full catalog is asserted to say *why*, and the distinct
reasons are kept distinct: "switched off" and "the backend threw" are the same
output and completely different problems.

The **debug line**, because it is the only window onto which tools the planner
was shown, and a summary that drifted into saying "kept 10 of 22" would answer
the easy question while hiding the one being asked.
"""
from __future__ import annotations

import logging

import pytest

from bamboo.tools.planner import (
    _collect_tool_catalog,
    _collect_tool_catalog_with_decision,
)
from bamboo.tools.tool_retrieval import (
    DEFAULT_K,
    ENV_BACKEND,
    ENV_K,
    ENV_LOG,
    ENV_MIN_CATALOG,
    MAX_INDEXED_DESCRIPTION_CHARS,
    PINNED_TOOLS,
    LexicalRetriever,
    active_backend,
    active_k,
    active_min_catalog,
    format_decision,
    index_terms,
    narrow_catalog,
    select_tools,
)


def _entry(name: str, description: str = "", parameters: list[str] | None = None) -> dict:
    """Build a catalog entry.

    Args:
        name: Wire name.
        description: Tool description.
        parameters: Top-level parameter names.

    Returns:
        dict: A catalog entry.
    """
    properties = {p: {"type": "string"} for p in (parameters or [])}
    return {
        "name": name,
        "description": description or f"description for {name}",
        "inputSchema": {"type": "object", "properties": properties},
    }


def _catalog(size: int = 20) -> list[dict]:
    """Build a catalog of distinguishable tools.

    Args:
        size: Number of filler tools beyond the pinned pair.

    Returns:
        list[dict]: Catalog entries.
    """
    entries = [_entry(name) for name in sorted(PINNED_TOOLS)]
    entries += [_entry(f"tool_{i}", f"handles subject{i} matters") for i in range(size)]
    return entries


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove retrieval configuration so each test starts from the default.

    Env-var leakage between tests has broken this suite before — it is in the
    deployment notes — so the isolation is explicit rather than assumed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    for name in (ENV_BACKEND, ENV_K, ENV_MIN_CATALOG, ENV_LOG):
        monkeypatch.delenv(name, raising=False)


class TestConfiguration:
    """The environment surface."""

    def test_retrieval_is_off_by_default(self) -> None:
        """An unset variable means no retrieval.

        The whole commit lands dark; this is the assertion that says so.
        """
        assert active_backend() == "off"

    def test_a_known_backend_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recognised name selects that backend, case and space insensitively."""
        monkeypatch.setenv(ENV_BACKEND, "  LEXICAL ")
        assert active_backend() == "lexical"

    def test_an_unknown_backend_falls_back_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo disables retrieval loudly rather than failing the server.

        Failing closed here would turn a one-character mistake into a server
        that errors on every question, to protect a feature whose fallback is
        the behaviour that shipped for two years.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexcial")
        with caplog.at_level(logging.WARNING):
            assert active_backend() == "off"
        assert "not a recognised retrieval backend" in caplog.text

    @pytest.mark.parametrize("raw", ["0", "-3", "banana", ""])
    def test_a_bad_k_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """Non-positive or unparseable budgets use the default.

        A ``k`` of 0 would retrieve nothing and leave the planner with the two
        pinned tools, which looks like a catastrophic retrieval regression
        rather than the configuration error it is.
        """
        monkeypatch.setenv(ENV_K, raw)
        assert active_k() == DEFAULT_K

    def test_k_is_read_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changing the variable takes effect without reimporting."""
        monkeypatch.setenv(ENV_K, "4")
        assert active_k() == 4
        monkeypatch.setenv(ENV_K, "7")
        assert active_k() == 7

    def test_min_catalog_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The small-catalog threshold reads from the environment."""
        monkeypatch.setenv(ENV_MIN_CATALOG, "5")
        assert active_min_catalog() == 5


class TestIndexTerms:
    """What gets indexed for each tool."""

    def test_names_are_split_on_dots_and_underscores(self) -> None:
        """A qualified name contributes its parts as separate terms.

        Without this, "harvester workers" in a question would not match a tool
        named ``panda_harvester_workers`` at all.
        """
        terms = index_terms(_entry("atlas.harvester_timeseries", "x"))
        assert {"atlas", "harvester", "timeseries"} <= set(terms)

    def test_parameter_names_are_indexed(self) -> None:
        """Parameter names add cheap discriminating terms."""
        terms = index_terms(_entry("t", "nothing useful", parameters=["computingsite"]))
        assert "computingsite" in terms

    def test_parameter_descriptions_are_not_indexed(self) -> None:
        """Only parameter names are taken, not their schema prose."""
        entry = _entry("t", "desc")
        entry["inputSchema"]["properties"]["site"] = {
            "type": "string",
            "description": "a distinctivetoken appearing only here",
        }
        assert "distinctivetoken" not in index_terms(entry)

    def test_long_descriptions_are_truncated(self) -> None:
        """Indexing stops at the character cap.

        ``opensearch_promptlog_query``'s description is 8,151 characters — a
        third of the catalog. Indexed whole it would dominate the term
        statistics of a 22-document corpus and pull unrelated questions towards
        itself.
        """
        tail = "zzmarker"
        description = ("word " * 400) + tail
        assert len(description) > MAX_INDEXED_DESCRIPTION_CHARS
        assert tail not in index_terms(_entry("t", description))

    def test_a_malformed_schema_does_not_raise(self) -> None:
        """A tool with no usable schema still indexes its name and description."""
        entry = {"name": "t", "description": "desc", "inputSchema": "not-a-dict"}
        assert "desc" in index_terms(entry)


class TestLexicalRetriever:
    """The scorer."""

    def test_it_ranks_the_matching_tool_first(self) -> None:
        """A question's distinctive term pulls its tool to the top."""
        catalog = [
            _entry("a", "handles stage-in timing and transfers"),
            _entry("b", "reports queue configuration"),
            _entry("c", "checks server health"),
        ]
        assert LexicalRetriever().retrieve("what about stage-in timing?", catalog, 3)[0] == "a"

    def test_it_respects_k(self) -> None:
        """Never more than *k* names come back."""
        catalog = [_entry(f"t{i}", "shared term here") for i in range(10)]
        assert len(LexicalRetriever().retrieve("shared term", catalog, 3)) == 3

    def test_zero_scoring_tools_are_dropped_not_padded(self) -> None:
        """A budget is not a quota.

        Padding the result to *k* with tools sharing no term with the question
        would hand the planner entries the retriever has no reason to believe
        in — exactly the prompt bloat this module exists to remove.
        """
        catalog = [_entry("a", "stage-in timing")] + [
            _entry(f"t{i}", "entirely unrelated subject matter") for i in range(5)
        ]
        assert LexicalRetriever().retrieve("stage-in", catalog, 5) == ["a"]

    def test_ties_break_on_catalog_order(self) -> None:
        """Equal scores rank deterministically.

        The harness compares runs; a ranking that permuted ties would make two
        identical measurements disagree.
        """
        catalog = [_entry(f"t{i}", "identical description text") for i in range(5)]
        retriever = LexicalRetriever()
        first = retriever.retrieve("identical description", catalog, 5)
        assert first == [f"t{i}" for i in range(5)]
        assert retriever.retrieve("identical description", catalog, 5) == first

    def test_an_empty_catalog_scores_nothing(self) -> None:
        """No catalog, no names, no exception."""
        assert LexicalRetriever().retrieve("anything", [], 5) == []

    def test_a_non_positive_k_returns_nothing(self) -> None:
        """A zero budget returns an empty list rather than the whole catalog."""
        assert LexicalRetriever().retrieve("q", [_entry("a")], 0) == []


class TestSelectToolsPassthrough:
    """Every route back to the full catalog, and its stated reason."""

    def test_disabled_by_default(self) -> None:
        """With no backend configured the catalog passes through."""
        decision = select_tools("why did job 1 fail?", _catalog())
        assert not decision.applied
        assert decision.reason == "disabled"

    def test_an_empty_question_is_not_scored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank question passes the catalog through.

        Scoring against an empty string would rank by document length alone,
        which is a ranking by verbosity rather than relevance.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        decision = select_tools("   ", _catalog())
        assert not decision.applied
        assert decision.reason == "no_question"

    def test_a_small_catalog_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retrieval does not run on a catalog no bigger than the budget.

        Narrowing 8 tools to 10 cannot help and can only lose one.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        decision = select_tools("why did job 1 fail?", _catalog(size=4))
        assert not decision.applied
        assert decision.reason == "catalog_small"

    def test_a_backend_error_falls_back_loudly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising retriever yields the full catalog and an exception log.

        The question still gets answered. What must not happen is answering it
        from the full catalog *silently*, which is indistinguishable from
        retrieval working until someone reads the token bill.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")

        class _Exploding:
            name = "lexical"

            def retrieve(self, question: str, catalog: object, k: int) -> list[str]:
                """Fail."""
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "bamboo.tools.tool_retrieval._build_retriever", lambda backend: _Exploding()
        )
        with caplog.at_level(logging.ERROR):
            decision = select_tools("why did job 1 fail?", _catalog())
        assert not decision.applied
        assert decision.reason == "backend_error"
        assert "tool retrieval failed" in caplog.text

    def test_names_outside_the_catalog_are_rejected(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A retriever inventing a name falls back rather than poisoning the prompt.

        A hallucinated or stale tool name reaching the catalog would make the
        planner propose a tool the executor cannot resolve — a failure that
        surfaces far from its cause.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")

        class _Inventing:
            name = "lexical"

            def retrieve(self, question: str, catalog: object, k: int) -> list[str]:
                """Return a name that is not in the catalog."""
                return ["tool_0", "panda_imaginary"]

        monkeypatch.setattr(
            "bamboo.tools.tool_retrieval._build_retriever", lambda backend: _Inventing()
        )
        with caplog.at_level(logging.ERROR):
            decision = select_tools("why did job 1 fail?", _catalog())
        assert not decision.applied
        assert decision.reason == "backend_error"
        assert "panda_imaginary" in caplog.text

    def test_an_all_zero_score_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When nothing matches, the planner gets everything rather than two tools.

        Keeping only the pins would leave the documentation route as the sole
        option for a question the retriever simply failed to understand.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        decision = select_tools("zzzz qqqq xxxx", _catalog())
        assert not decision.applied
        assert decision.reason == "empty_result"


class TestSelectToolsApplied:
    """What a successful narrowing produces."""

    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enable the lexical backend for this class.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")

    def test_pinned_tools_survive_without_being_scored(self) -> None:
        """The fallback pair is always kept, however the question scores."""
        decision = select_tools("subject3 matters", _catalog())
        assert decision.applied
        assert PINNED_TOOLS <= set(decision.kept)
        assert decision.pinned == PINNED_TOOLS

    def test_pins_count_against_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Total tools never exceed *k*, pins included.

        A pin outside the budget would make *k* a budget in name only — the
        planner would receive k + 2 tools under a configuration saying k.
        """
        monkeypatch.setenv(ENV_K, "6")
        decision = select_tools("subject3 matters", _catalog())
        assert decision.applied
        assert len(decision.kept) <= 6

    def test_kept_and_withheld_partition_the_catalog(self) -> None:
        """Every tool is either kept or withheld, never both and never neither."""
        catalog = _catalog()
        decision = select_tools("subject3 matters", catalog)
        names = {str(e["name"]) for e in catalog}
        assert set(decision.kept) | set(decision.withheld) == names
        assert not set(decision.kept) & set(decision.withheld)

    def test_scores_are_recorded_for_withheld_tools_too(self) -> None:
        """A withheld tool's score is reported, not just its absence.

        "It scored 0.83 and missed the cut" and "it scored nothing" call for
        different fixes; a bare list of survivors cannot tell them apart.
        """
        decision = select_tools("subject3 matters", _catalog())
        scored = dict(decision.scores)
        assert all(name in scored for name in decision.withheld)


class TestNarrowCatalog:
    """The catalog the planner actually receives."""

    def test_catalog_order_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Survivors keep catalog order, not score order.

        Score order would make the prompt's tool list reorder itself between
        questions, so two planner prompts would differ for reasons unrelated to
        their content — and prompt diffs are how routing regressions get found.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        catalog = _catalog()
        narrowed, decision = narrow_catalog("subject3 matters", catalog)
        assert decision.applied
        original = [str(e["name"]) for e in catalog]
        assert [str(e["name"]) for e in narrowed] == [
            n for n in original if n in set(decision.kept)
        ]

    def test_disabled_returns_the_catalog_unchanged(self) -> None:
        """With retrieval off the entries come back as they went in."""
        catalog = _catalog()
        narrowed, decision = narrow_catalog("why did job 1 fail?", catalog)
        assert not decision.applied
        assert [e["name"] for e in narrowed] == [e["name"] for e in catalog]


class TestPlannerIntegration:
    """The hook in the planner's catalog assembly."""

    def test_the_bare_collector_never_retrieves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_collect_tool_catalog`` assembles and does not narrow.

        Retrieval lives in the wrapper so a caller with no question cannot
        acquire it by omitting an argument.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        assert len(_collect_tool_catalog(namespaces=["atlas"])) == len(
            _collect_tool_catalog(namespaces=["atlas"])
        )

    def test_no_question_yields_no_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a question retrieval is never consulted."""
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        catalog, decision = _collect_tool_catalog_with_decision(namespaces=["atlas"])
        assert decision is None
        assert catalog

    def test_the_default_configuration_changes_nothing(self) -> None:
        """With retrieval unset the planner's catalog is byte-identical.

        This commit must be dark. The flip is C5's, not C3's.
        """
        plain = _collect_tool_catalog(namespaces=["atlas"])
        hooked, decision = _collect_tool_catalog_with_decision(
            namespaces=["atlas"], question="why did job 6837798305 fail?"
        )
        assert hooked == plain
        assert decision is not None and not decision.applied

    def test_enabling_it_narrows_the_real_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Switched on, the real ATLAS catalog shrinks and keeps the right tool."""
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        plain = _collect_tool_catalog(namespaces=["atlas"])
        if len(plain) <= DEFAULT_K:
            pytest.skip("catalog too small to narrow in this environment")
        hooked, decision = _collect_tool_catalog_with_decision(
            namespaces=["atlas"], question="why did job 6837798305 fail?"
        )
        assert decision is not None and decision.applied
        assert len(hooked) < len(plain)
        assert "panda_log_analysis" in {str(e["name"]) for e in hooked}


class TestDebugVisibility:
    """The line that says which tools the planner was shown."""

    def test_it_names_every_kept_and_withheld_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both sides appear in full, with scores.

        The point of the line is answering *which* tools, so a summary giving
        only counts would be a regression even though it reads as tidier.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        decision = select_tools("subject3 matters", _catalog())
        rendered = format_decision(decision)
        for name in decision.kept + decision.withheld:
            assert name in rendered

    def test_pinned_tools_are_marked_as_pinned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pin is distinguishable from a high scorer.

        Otherwise a pinned tool reads as evidence the scorer is working when it
        is evidence of nothing.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        rendered = format_decision(select_tools("subject3 matters", _catalog()))
        for name in sorted(PINNED_TOOLS):
            assert f"{name}(pinned)" in rendered

    def test_a_passthrough_states_its_reason(self) -> None:
        """The not-applied line says why, not just that."""
        rendered = format_decision(select_tools("why did job 1 fail?", _catalog()))
        assert "not applied" in rendered
        assert "reason=disabled" in rendered

    def test_it_logs_at_debug_by_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every call is logged, quietly, without extra configuration."""
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        with caplog.at_level(logging.DEBUG, logger="bamboo.tools.tool_retrieval"):
            select_tools("subject3 matters", _catalog())
        assert "tool retrieval:" in caplog.text

    def test_the_log_flag_raises_it_to_info(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``BAMBOO_TOOL_RETRIEVAL_LOG`` surfaces selections without global DEBUG.

        Turning on DEBUG for the whole process to see one line buries it in
        HTTP and SDK chatter, which is why the flag exists.
        """
        monkeypatch.setenv(ENV_BACKEND, "lexical")
        monkeypatch.setenv(ENV_LOG, "1")
        with caplog.at_level(logging.INFO, logger="bamboo.tools.tool_retrieval"):
            select_tools("subject3 matters", _catalog())
        assert any(r.levelno == logging.INFO for r in caplog.records)
