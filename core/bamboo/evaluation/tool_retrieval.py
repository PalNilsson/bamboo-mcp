"""Evaluation harness for query-conditioned tool retrieval.

The planner is shown the whole tool catalog on every question.  Retrieval
narrows that to the tools a question plausibly needs, which trades a smaller
prompt for the risk of withholding the tool the planner actually required.
That trade cannot be argued, only measured, and it has to be measured against
the behaviour it replaces — so this module evaluates *any* retriever,
including the null one that returns everything and reproduces today's
behaviour exactly.

Three numbers, because one is not enough:

* **Recall@k.** Did the retrieved set contain every tool a correct plan needs?
  Strict by design: a question routed to two tools scores zero when only one
  survives, since half of a co-occurrence pair is a broken plan, not a partial
  one.
* **Rule coverage.** Did the routing guidance that survives filtering still
  name those tools?  Retrieval can keep the right tool in the catalog while
  dropping the clause that tells the planner when to use it, which is a
  silent, retrieval-specific regression no recall figure reveals.
* **Payload bytes.** What the narrowing actually bought, as a fraction of the
  unfiltered catalog — the only reason to accept any recall loss at all.

Every metric is additionally reported over the subset of cases flagged *hard*:
questions paired with a tool that is easy to confuse for a neighbouring one
(``panda_queue_info`` against ``cric_query``, live counts against historical
aggregates).  Aggregate recall is dominated by easy cases and will look fine
while the confusable ones rot.

Nothing here imports a retrieval backend.  :class:`ToolRetriever` is a
structural protocol, so a backend satisfies it by shape without importing this
module, and the null baseline below needs no backend at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ToolRetriever(Protocol):
    """A scorer that narrows a tool catalog to the tools a question needs."""

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

        Args:
            question: The user's question, verbatim.
            catalog: Catalog entries as the planner would receive them; each
                has at least ``name``, ``description`` and ``inputSchema``.
            k: Maximum number of names to return.  A retriever may return
                fewer, never more.

        Returns:
            List[str]: Tool names drawn from *catalog*, ordered by descending
            relevance.
        """
        ...


class NullRetriever:
    """Returns the whole catalog, reproducing the pre-retrieval planner.

    The baseline every candidate is measured against.  Recall is 1.0 by
    construction and payload cost is 100%, which is the point: it fixes both
    ends of the trade so a candidate's numbers mean something.

    *k* is deliberately ignored rather than honoured, because truncating the
    catalog to its first *k* entries in catalog order would be a retriever —
    an arbitrarily bad one — and not the behaviour being used as a reference.
    """

    name = "null"

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return every catalog name, in catalog order.

        Args:
            question: Ignored.
            catalog: Catalog entries.
            k: Ignored; see the class docstring.

        Returns:
            List[str]: Every name in *catalog*.
        """
        return [str(entry["name"]) for entry in catalog]


@dataclass(frozen=True)
class CorpusCase:
    """One labelled question.

    Attributes:
        case_id: Stable identifier, used to name a case in a report.
        question: The question, as a user would type it.
        expected_tools: Every tool a correct plan must propose.  More than one
            means a co-occurrence requirement, not a choice.
        source: ``"cheatsheet"`` for a question lifted verbatim from
            ``docs/question-cheatsheet.md``, ``"authored"`` for one written for
            this corpus.
        hard: Whether this question is paired with a deliberately confusable
            tool.
        notes: Why the case exists, where that is not obvious.
    """

    case_id: str
    question: str
    expected_tools: frozenset[str]
    source: str
    hard: bool = False
    notes: str = ""


@dataclass(frozen=True)
class Corpus:
    """A labelled corpus with its coverage exemptions.

    Attributes:
        cases: The labelled questions.
        coverage_exempt: Tool name to the reason it needs no cases — tools
            invoked by the interface rather than asked for in words.  Recorded
            with the data rather than in a test so that adding a tool forces a
            decision: write cases, or write down why not.
        description: What the corpus is for.
    """

    cases: tuple[CorpusCase, ...]
    coverage_exempt: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    def expected_tools(self) -> frozenset[str]:
        """Return every tool named by any case.

        Returns:
            FrozenSet[str]: Union of all cases' expected tools.
        """
        if not self.cases:
            return frozenset()
        return frozenset().union(*(case.expected_tools for case in self.cases))


def load_corpus(path: Path) -> Corpus:
    """Load a labelled corpus from JSON.

    Args:
        path: Path to the corpus file.

    Returns:
        Corpus: The parsed corpus.

    Raises:
        ValueError: If a case lacks an id, a question or an expected tool, or
            if two cases share an id.  A case with no expected tool would
            silently score as a free pass on every retriever, so it is rejected
            rather than skipped.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[CorpusCase] = []
    seen: set[str] = set()
    for entry in raw.get("cases", []):
        case_id = str(entry.get("id", "")).strip()
        question = str(entry.get("question", "")).strip()
        expected = frozenset(str(t) for t in entry.get("expected_tools", []))
        if not case_id:
            raise ValueError(f"corpus case without an id: {entry!r}")
        if case_id in seen:
            raise ValueError(f"duplicate corpus case id: {case_id}")
        if not question:
            raise ValueError(f"corpus case {case_id} has no question")
        if not expected:
            raise ValueError(f"corpus case {case_id} names no expected tool")
        seen.add(case_id)
        cases.append(
            CorpusCase(
                case_id=case_id,
                question=question,
                expected_tools=expected,
                source=str(entry.get("source", "authored")),
                hard=bool(entry.get("hard", False)),
                notes=str(entry.get("notes", "")),
            )
        )
    return Corpus(
        cases=tuple(cases),
        coverage_exempt=dict(raw.get("coverage_exempt", {})),
        description=str(raw.get("description", "")),
    )


@dataclass(frozen=True)
class CaseResult:
    """What a retriever did on one case.

    Attributes:
        case: The case evaluated.
        retrieved: Names returned, best match first.
        hit: Whether every expected tool was retrieved.
        missing: Expected tools that were not retrieved.
        worst_rank: Zero-based rank of the last-placed expected tool, or
            ``None`` when one is missing.  The *worst* rank rather than the
            best, because a plan needs all of them: a pair at ranks 0 and 9 is
            only as safe as its rank-9 member.
        guidance_covered: Whether the surviving routing guidance still names
            every expected tool, or ``None`` when no clause names them and the
            question is therefore outside the guidance's scope.
        payload_bytes: Serialised size of the retrieved catalog subset.
        over_budget: How many tools beyond *k* were returned, pins included;
            0 when the budget was respected.
    """

    case: CorpusCase
    retrieved: tuple[str, ...]
    hit: bool
    missing: frozenset[str]
    worst_rank: int | None
    guidance_covered: bool | None
    payload_bytes: int
    over_budget: int = 0


@dataclass(frozen=True)
class Report:
    """Aggregate results for one retriever at one *k*.

    Attributes:
        retriever: Name of the retriever evaluated.
        k: The *k* it was asked for.
        results: Per-case results, in corpus order.
        full_payload_bytes: Serialised size of the unfiltered catalog, the
            denominator for payload savings.
    """

    retriever: str
    k: int
    results: tuple[CaseResult, ...]
    full_payload_bytes: int

    def recall(self, hard_only: bool = False) -> float:
        """Return the fraction of cases where every expected tool was retrieved.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            float: Recall in [0, 1]; 1.0 when no case qualifies, since a
            retriever cannot be blamed for a subset that does not exist.
        """
        chosen = [r for r in self.results if r.case.hard or not hard_only]
        if not chosen:
            return 1.0
        return sum(1 for r in chosen if r.hit) / len(chosen)

    def guidance_coverage(self) -> float:
        """Return the fraction of in-scope cases whose guidance survived.

        Cases no routing clause names are excluded rather than counted as
        passes, which would dilute the figure towards 1.0 with questions the
        metric says nothing about.

        Returns:
            float: Coverage in [0, 1]; 1.0 when no case is in scope.
        """
        chosen = [r for r in self.results if r.guidance_covered is not None]
        if not chosen:
            return 1.0
        return sum(1 for r in chosen if r.guidance_covered) / len(chosen)

    def mean_payload_fraction(self) -> float:
        """Return mean retrieved payload size as a fraction of the full catalog.

        Returns:
            float: Mean fraction; 1.0 when the catalog is empty or no case ran.
        """
        if not self.results or self.full_payload_bytes <= 0:
            return 1.0
        return sum(r.payload_bytes for r in self.results) / (
            len(self.results) * self.full_payload_bytes
        )

    def over_budget_cases(self) -> int:
        """Return how many cases returned more tools than *k* allowed.

        The harness measures the budget rather than enforcing it.  Truncating
        a retriever's output to *k* here would hide a retriever that ignores
        its budget, and would reduce :class:`NullRetriever` to "the first *k*
        entries in catalog order" — a retriever, and a bad one, rather than
        the unfiltered reference it exists to provide.  The baseline is
        therefore expected to be over budget on every case; a candidate
        retriever is not.

        Returns:
            int: Number of cases that exceeded *k*.
        """
        return sum(1 for r in self.results if r.over_budget)

    def failures(self, hard_only: bool = False) -> tuple[CaseResult, ...]:
        """Return cases that missed at least one expected tool.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Tuple[CaseResult, ...]: Failing cases, in corpus order.
        """
        return tuple(
            r for r in self.results if not r.hit and (r.case.hard or not hard_only)
        )


def _payload_bytes(catalog: Sequence[Mapping[str, Any]], names: Sequence[str]) -> int:
    """Return the serialised size of the named subset of a catalog.

    Args:
        catalog: Full catalog entries.
        names: Names to include.

    Returns:
        int: Length of the JSON serialisation, in characters.  A proxy for
        prompt cost that avoids depending on a tokeniser, and one that moves
        in step with tokens closely enough for a ratio.
    """
    wanted = set(names)
    subset = [entry for entry in catalog if str(entry["name"]) in wanted]
    return len(json.dumps(subset, ensure_ascii=False))


def _guidance_names(rendered: str, tools: frozenset[str]) -> bool | None:
    """Report whether rendered guidance names every tool in *tools*.

    Args:
        rendered: Rendered routing guidance.
        tools: Tools the case expects.

    Returns:
        Optional[bool]: ``True`` or ``False`` when at least one of *tools* is
        named by the unfiltered guidance, ``None`` when none is — the question
        is then outside the guidance's scope and the metric does not apply.
    """
    return all(tool in rendered for tool in tools)


def evaluate(
    retriever: ToolRetriever,
    corpus: Corpus,
    catalog: Sequence[Mapping[str, Any]],
    k: int,
    pinned: frozenset[str] = frozenset(),
    routing_rules: Sequence[Any] = (),
) -> Report:
    """Evaluate a retriever over a corpus against a catalog.

    Args:
        retriever: The retriever under test.
        corpus: Labelled questions.
        catalog: Catalog entries the planner would otherwise receive whole.
        k: Maximum tools the planner should end up with, *including* pinned
            ones.  Pins that did not count against *k* would let a caller claim
            a budget it does not keep.
        pinned: Tools always present regardless of score — the universal
            fallback route, which has no graceful degradation if dropped.
        routing_rules: ``RoutingRule``-shaped objects with ``tools`` and
            ``text``, used for the guidance-coverage metric.  Empty disables it.

    Returns:
        Report: Per-case results and aggregates.
    """
    catalog_names = [str(entry["name"]) for entry in catalog]
    available_pins = frozenset(pinned) & frozenset(catalog_names)
    full_bytes = _payload_bytes(catalog, catalog_names)
    budget = max(0, k - len(available_pins))

    unfiltered_guidance = "\n".join(str(rule.text) for rule in routing_rules)

    results: list[CaseResult] = []
    for case in corpus.cases:
        scored = retriever.retrieve(case.question, catalog, budget)
        ordered = [name for name in scored if name not in available_pins]
        retrieved = tuple(sorted(available_pins)) + tuple(ordered)
        retrieved_set = frozenset(retrieved)

        missing = case.expected_tools - retrieved_set
        if missing:
            worst_rank: int | None = None
        else:
            worst_rank = max(retrieved.index(tool) for tool in case.expected_tools)

        guidance_covered: bool | None = None
        if routing_rules and any(tool in unfiltered_guidance for tool in case.expected_tools):
            kept = [
                str(rule.text)
                for rule in routing_rules
                if frozenset(rule.tools) <= retrieved_set
            ]
            guidance_covered = _guidance_names("\n".join(kept), case.expected_tools)

        results.append(
            CaseResult(
                case=case,
                retrieved=retrieved,
                hit=not missing,
                missing=missing,
                worst_rank=worst_rank,
                guidance_covered=guidance_covered,
                payload_bytes=_payload_bytes(catalog, retrieved),
                over_budget=max(0, len(retrieved) - k),
            )
        )

    return Report(
        retriever=getattr(retriever, "name", type(retriever).__name__),
        k=k,
        results=tuple(results),
        full_payload_bytes=full_bytes,
    )
