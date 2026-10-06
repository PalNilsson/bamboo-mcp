"""Tests for the labelled tool-selection corpus and its evaluation harness.

A corpus is only as good as its coupling to the thing it labels.  A corpus
naming a tool the catalog does not have measures nothing; a corpus silently
missing a tool measures nothing *about that tool*, which is worse, because the
aggregate still looks healthy. Both are checked here.

The harness tests pin the metrics rather than the retriever, because no
retriever exists yet.  They matter most now: the baseline these metrics
produce is the reference the first real retriever will be judged against, and
a baseline computed by a subtly wrong harness would quietly set the bar in the
wrong place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bamboo.evaluation.tool_retrieval import (
    Corpus,
    CorpusCase,
    NullRetriever,
    ToolRetriever,
    evaluate,
    load_corpus,
)
from bamboo.tools.planner import (
    RoutingRule,
    _ATLAS_ROUTING_RULES,
    _collect_tool_catalog,
)

CORPUS_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "tool_selection_corpus.json"

#: Mirrors decision T-4 and ``scripts/eval_tool_retrieval.py``.
PINNED_TOOLS = frozenset({"panda_doc_search", "panda_doc_bm25"})

#: Minimum labelled questions per catalogue tool.  Three is not a statistical
#: claim — it is the point below which a single wording quirk decides whether a
#: tool looks retrievable.
MIN_CASES_PER_TOOL = 3


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """Return the labelled corpus.

    Returns:
        Corpus: Parsed corpus.
    """
    return load_corpus(CORPUS_PATH)


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    """Return the ATLAS planner catalog.

    Returns:
        List[dict]: Catalog entries.
    """
    return list(_collect_tool_catalog(namespaces=["atlas"]))


def _catalog_names(catalog: list[dict]) -> frozenset[str]:
    """Return catalog wire names.

    Args:
        catalog: Catalog entries.

    Returns:
        FrozenSet[str]: Tool names.
    """
    return frozenset(str(entry["name"]) for entry in catalog)


class TestCorpusIntegrity:
    """The corpus file itself is well-formed."""

    def test_it_parses(self, corpus: Corpus) -> None:
        """The corpus loads and is not empty."""
        assert len(corpus.cases) >= 100

    def test_case_ids_are_unique(self, corpus: Corpus) -> None:
        """No two cases share an id."""
        ids = [case.case_id for case in corpus.cases]
        assert len(ids) == len(set(ids))

    def test_questions_are_unique(self, corpus: Corpus) -> None:
        """No question appears twice.

        A duplicated question silently double-weights whatever it tests, and
        the duplicate is almost always a copy-paste while extending a section.
        """
        questions = [case.question.strip().lower() for case in corpus.cases]
        duplicates = {q for q in questions if questions.count(q) > 1}
        assert not duplicates, sorted(duplicates)

    def test_sources_are_declared(self, corpus: Corpus) -> None:
        """Every case says where it came from."""
        assert all(case.source in {"cheatsheet", "authored"} for case in corpus.cases)

    def test_hard_cases_explain_themselves(self, corpus: Corpus) -> None:
        """A case flagged *hard* says which tool it is confusable with.

        The flag drives a separately-reported metric, so an unexplained flag
        is a number nobody can act on.
        """
        for case in corpus.cases:
            if case.hard:
                assert case.notes, f"{case.case_id} is flagged hard with no note"

    def test_a_meaningful_fraction_is_hard(self, corpus: Corpus) -> None:
        """Enough confusable cases exist for the hard-subset metric to mean something.

        Below roughly a fifth, ``recall@k (hard)`` moves in large jumps on
        single cases and stops being readable as a trend.
        """
        hard = sum(1 for case in corpus.cases if case.hard)
        assert hard >= len(corpus.cases) // 5


class TestCorpusMatchesCatalog:
    """The corpus labels tools that actually exist, and covers them all."""

    def test_every_expected_tool_is_in_the_catalog(
        self, corpus: Corpus, catalog: list[dict]
    ) -> None:
        """No case expects a tool the planner cannot propose.

        This is the corpus-side counterpart of the guidance drift that
        ``tests/test_planner_routing_rules.py`` pins.  ``docs/question-cheatsheet.md``
        currently documents a ``panda_prompt_log`` tool that has never existed
        under that name — the real tool is ``opensearch_promptlog_query`` — so
        transcribing the cheat sheet without this check would have imported the
        error straight into the corpus.
        """
        unknown = corpus.expected_tools() - _catalog_names(catalog)
        assert not unknown, (
            f"Corpus expects tools absent from the catalog: {sorted(unknown)}. "
            "Either the name is wrong or the plugin is not installed."
        )

    def test_every_catalog_tool_is_covered_or_exempt(
        self, corpus: Corpus, catalog: list[dict]
    ) -> None:
        """Each catalog tool has cases, or a written reason it needs none.

        Forces the decision at the moment a tool is added rather than leaving a
        blind spot that the aggregate recall figure conceals.
        """
        covered = corpus.expected_tools()
        exempt = frozenset(corpus.coverage_exempt)
        uncovered = _catalog_names(catalog) - covered - exempt
        assert not uncovered, (
            f"Catalog tools with no corpus cases and no exemption: {sorted(uncovered)}. "
            "Add cases, or add the tool to coverage_exempt with a reason."
        )

    def test_exemptions_name_real_tools(self, corpus: Corpus, catalog: list[dict]) -> None:
        """No exemption excuses a tool that is not in the catalog.

        A stale exemption is how the previous test quietly stops covering a
        renamed tool.
        """
        stale = frozenset(corpus.coverage_exempt) - _catalog_names(catalog)
        assert not stale, sorted(stale)

    def test_exemptions_have_reasons(self, corpus: Corpus) -> None:
        """Each exemption records why the tool needs no cases."""
        for tool, reason in corpus.coverage_exempt.items():
            assert reason.strip(), f"{tool} is exempt with no reason given"

    def test_no_tool_is_both_covered_and_exempt(self, corpus: Corpus) -> None:
        """A tool is either exercised or excused, never both."""
        overlap = corpus.expected_tools() & frozenset(corpus.coverage_exempt)
        assert not overlap, sorted(overlap)

    def test_each_covered_tool_has_enough_cases(self, corpus: Corpus) -> None:
        """Every covered tool clears the minimum case count."""
        counts: dict[str, int] = {}
        for case in corpus.cases:
            for tool in case.expected_tools:
                counts[tool] = counts.get(tool, 0) + 1
        thin = {t: n for t, n in counts.items() if n < MIN_CASES_PER_TOOL}
        assert not thin, f"Tools with fewer than {MIN_CASES_PER_TOOL} cases: {thin}"

    def test_pinned_tools_are_exercised_as_a_pair(self, corpus: Corpus) -> None:
        """The fallback tools are only ever expected together.

        They back one route.  A case expecting just one of them would reward a
        retriever for splitting a pair that must not be split.
        """
        for case in corpus.cases:
            overlap = case.expected_tools & PINNED_TOOLS
            assert overlap in (frozenset(), PINNED_TOOLS), case.case_id


class TestLoadCorpus:
    """Malformed corpora are rejected rather than silently shrunk."""

    def _write(self, tmp_path: Path, cases: list[dict]) -> Path:
        """Write a throwaway corpus file.

        Args:
            tmp_path: Pytest temporary directory.
            cases: Case dicts.

        Returns:
            Path: The written file.
        """
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        return path

    def test_a_case_without_expected_tools_is_rejected(self, tmp_path: Path) -> None:
        """An unlabelled case raises rather than loading.

        It would otherwise score as a hit against every retriever, inflating
        recall by exactly the number of cases nobody finished labelling.
        """
        path = self._write(tmp_path, [{"id": "x", "question": "q", "expected_tools": []}])
        with pytest.raises(ValueError, match="no expected tool"):
            load_corpus(path)

    def test_a_case_without_a_question_is_rejected(self, tmp_path: Path) -> None:
        """An empty question raises."""
        path = self._write(tmp_path, [{"id": "x", "question": "  ", "expected_tools": ["a"]}])
        with pytest.raises(ValueError, match="no question"):
            load_corpus(path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        """Two cases with one id raise."""
        case = {"id": "x", "question": "q", "expected_tools": ["a"]}
        path = self._write(tmp_path, [case, dict(case, question="other")])
        with pytest.raises(ValueError, match="duplicate"):
            load_corpus(path)


class TestNullBaseline:
    """The no-retrieval baseline, which every candidate is measured against."""

    def test_baseline_recall_is_perfect(self, corpus: Corpus, catalog: list[dict]) -> None:
        """Returning the whole catalog retrieves every expected tool.

        If this is not 1.000 the corpus is wrong, not the retriever — which is
        why it is asserted rather than assumed.
        """
        report = evaluate(
            NullRetriever(), corpus, catalog, k=10,
            pinned=PINNED_TOOLS, routing_rules=_ATLAS_ROUTING_RULES,
        )
        assert report.recall() == 1.0
        assert report.recall(hard_only=True) == 1.0
        assert report.guidance_coverage() == 1.0
        assert report.mean_payload_fraction() == 1.0

    def test_baseline_is_reported_as_over_budget(
        self, corpus: Corpus, catalog: list[dict]
    ) -> None:
        """The baseline ignores *k*, and the harness says so rather than hiding it.

        Mutation guard for the harness's own first bug: truncating the
        retriever's output to *k* inside ``evaluate`` turned this baseline into
        "the first *k* tools in catalog order" and produced a recall of 0.383,
        which would have become the reference every later measurement was
        compared against.
        """
        report = evaluate(NullRetriever(), corpus, catalog, k=10, pinned=PINNED_TOOLS)
        assert report.over_budget_cases() == len(corpus.cases)
        assert all(len(r.retrieved) == len(catalog) for r in report.results)


class TestEvaluateMetrics:
    """The metrics themselves, on constructed inputs."""

    def _catalog(self, names: list[str]) -> list[dict]:
        """Build a minimal catalog.

        Args:
            names: Tool names.

        Returns:
            List[dict]: Catalog entries.
        """
        return [{"name": n, "description": f"desc {n}", "inputSchema": {}} for n in names]

    def _corpus(self, expected: list[list[str]], hard: bool = False) -> Corpus:
        """Build a corpus with one case per expected-tool set.

        Args:
            expected: Expected tool sets.
            hard: Whether to flag the cases hard.

        Returns:
            Corpus: The corpus.
        """
        return Corpus(
            cases=tuple(
                CorpusCase(f"c{i}", f"question {i}", frozenset(tools), "authored", hard)
                for i, tools in enumerate(expected)
            )
        )

    def _fixed(self, names: list[str]) -> ToolRetriever:
        """Return a retriever that always returns *names*.

        Args:
            names: Names to return.

        Returns:
            ToolRetriever: The stub.
        """

        class _Fixed:
            name = "fixed"

            def retrieve(self, question: str, catalog: object, k: int) -> list[str]:
                """Return the fixed list."""
                return list(names)

        return _Fixed()

    def test_recall_requires_every_expected_tool(self) -> None:
        """Half a co-occurrence pair scores zero, not a half.

        Mutation guard: an intersection test rather than a subset test would
        score this 1.0 and declare a broken plan a success.
        """
        catalog = self._catalog(["a", "b", "c"])
        corpus = self._corpus([["a", "b"]])
        report = evaluate(self._fixed(["a"]), corpus, catalog, k=3)
        assert report.recall() == 0.0
        assert report.results[0].missing == frozenset({"b"})

    def test_pins_are_added_and_count_against_k(self) -> None:
        """Pinned tools appear without being scored, and consume budget.

        With k=3 and two pins the retriever is asked for 1, so a pin that did
        not count would quietly hand the planner four tools under a budget of
        three.
        """
        catalog = self._catalog(["p1", "p2", "a", "b"])
        corpus = self._corpus([["p1", "a"]])
        seen: list[int] = []

        class _Recording:
            name = "recording"

            def retrieve(self, question: str, catalog: object, k: int) -> list[str]:
                """Record the budget and return one tool."""
                seen.append(k)
                return ["a"]

        report = evaluate(
            _Recording(), corpus, catalog, k=3, pinned=frozenset({"p1", "p2"})
        )
        assert seen == [1]
        assert set(report.results[0].retrieved) == {"p1", "p2", "a"}
        assert report.recall() == 1.0
        assert report.over_budget_cases() == 0

    def test_a_pin_absent_from_the_catalog_is_ignored(self) -> None:
        """Pinning a tool that is not catalogued does not consume budget.

        The DuckDB-backed tools vanish from the catalog on a host without the
        dependency; charging budget for a tool nobody can call would shrink the
        real budget on exactly those hosts.
        """
        catalog = self._catalog(["a", "b"])
        corpus = self._corpus([["a"]])
        seen: list[int] = []

        class _Recording:
            name = "recording"

            def retrieve(self, question: str, catalog: object, k: int) -> list[str]:
                """Record the budget and return one tool."""
                seen.append(k)
                return ["a"]

        evaluate(_Recording(), corpus, catalog, k=2, pinned=frozenset({"ghost"}))
        assert seen == [2]

    def test_worst_rank_not_best_rank(self) -> None:
        """A pair is ranked by its worse member.

        A pair at ranks 0 and 4 survives only a budget that reaches rank 4, so
        reporting the better rank would overstate how safely it fits.
        """
        catalog = self._catalog(["a", "b", "c", "d", "e"])
        corpus = self._corpus([["a", "e"]])
        report = evaluate(self._fixed(["a", "b", "c", "d", "e"]), corpus, catalog, k=5)
        assert report.results[0].worst_rank == 4

    def test_payload_fraction_tracks_the_retrieved_subset(self) -> None:
        """Payload is measured on what was retrieved, not on *k*."""
        catalog = self._catalog(["a", "b", "c", "d"])
        corpus = self._corpus([["a"]])
        report = evaluate(self._fixed(["a", "b"]), corpus, catalog, k=4)
        assert 0.0 < report.mean_payload_fraction() < 1.0

    def test_guidance_coverage_drops_when_a_clause_is_withheld(self) -> None:
        """Keeping a tool but losing its clause is counted as a miss.

        The retrieval-specific failure recall cannot see: ``b`` is retrieved,
        so recall is perfect, but the only clause naming it also names ``c``,
        which was not — so the planner is handed ``b`` with no instruction
        about when to use it.
        """
        catalog = self._catalog(["a", "b", "c"])
        corpus = self._corpus([["b"]])
        rules = (
            RoutingRule(frozenset({"a"}), "- use a."),
            RoutingRule(frozenset({"b", "c"}), "- use b together with c."),
        )
        report = evaluate(self._fixed(["a", "b"]), corpus, catalog, k=3, routing_rules=rules)
        assert report.recall() == 1.0
        assert report.guidance_coverage() == 0.0

    def test_guidance_coverage_ignores_out_of_scope_cases(self) -> None:
        """A tool no clause mentions is excluded, not scored as a pass.

        Counting it as a pass would drift the figure towards 1.0 using cases
        the metric says nothing about — ``cric_query`` and the OpenSearch tools
        are in the catalog but named by no routing clause.
        """
        catalog = self._catalog(["a", "z"])
        corpus = self._corpus([["z"]])
        rules = (RoutingRule(frozenset({"a"}), "- use a."),)
        report = evaluate(self._fixed(["z"]), corpus, catalog, k=2, routing_rules=rules)
        assert report.results[0].guidance_covered is None
        assert report.guidance_coverage() == 1.0

    def test_hard_recall_is_reported_separately(self) -> None:
        """The hard subset is scored on its own, not folded into the aggregate."""
        catalog = self._catalog(["a", "b"])
        easy = CorpusCase("e1", "easy", frozenset({"a"}), "authored", hard=False)
        hard = CorpusCase("h1", "hard", frozenset({"b"}), "authored", hard=True)
        corpus = Corpus(cases=(easy, hard))
        report = evaluate(self._fixed(["a"]), corpus, catalog, k=2)
        assert report.recall() == 0.5
        assert report.recall(hard_only=True) == 0.0

    def test_empty_corpus_does_not_divide_by_zero(self) -> None:
        """Aggregates on an empty corpus are 1.0, not an exception."""
        report = evaluate(self._fixed([]), Corpus(cases=()), self._catalog(["a"]), k=1)
        assert report.recall() == 1.0
        assert report.guidance_coverage() == 1.0
        assert report.mean_payload_fraction() == 1.0
