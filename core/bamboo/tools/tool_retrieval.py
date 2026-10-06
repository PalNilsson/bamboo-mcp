"""Query-conditioned narrowing of the planner's tool catalog.

The planner is shown every tool on every question: 22 definitions, roughly
29 kB of JSON, of which one tool is a third.  Selection accuracy degrades as
that list grows, and the cost is paid per call regardless of what was asked.
This module scores the catalog against the question and hands the planner only
the tools it plausibly needs.

**Lexical, not embedding, as the first backend.**  The corpus is ~22 short
strings of rare domain jargon — ``pilottiming``, ``HS06``, ``cmtconfig``,
``netzone``, ``queuedata``.  Exact term matching is close to its best case
there and a small general-purpose embedding model close to its worst, since
those tokens are precisely the ones it has least signal for.  BM25 also loads
nothing, costs microseconds on a corpus this size, adds no dependency to the
planner path, and cannot fail on a vector-dimension mismatch.
:class:`ToolRetriever` keeps the choice reversible: an embedding backend is a
second implementation of the same protocol, and
``scripts/eval_tool_retrieval.py`` decides between them on recall rather than
on taste.

**On by default, as ``lexical``, since the harness said so.**  Measured over
120 labelled questions at k=10: recall 0.992, 0.983 on the deliberately
confusable subset, guidance coverage 1.000, and 39% of the former prompt.  The
embedding backend scored worse on both recall and payload, losing precisely the
jargon cases — a literal source path, ``MCORE``, ``queue time`` — that carry
most of this catalog's meaning.  ``BAMBOO_TOOL_RETRIEVAL=off`` remains the kill
switch, and restores the previous behaviour exactly.

**Degradation is loud.**  A retriever that raises, or returns names outside the
catalog, falls back to the full catalog and says so in the log and the trace.
Silent fail-open is this codebase's most frequently rediscovered root cause,
and a retrieval layer that quietly stops retrieving would present as nothing
worse than a larger prompt.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)

#: Selects the retrieval backend.  ``off`` (the default) disables retrieval
#: entirely; ``lexical`` enables :class:`LexicalRetriever`; ``embedding`` and
#: ``hybrid`` enable the backends in
#: :mod:`bamboo.tools._tool_retrieval_embedding`, both of which need an
#: embedding model and fall back loudly when none is installed.
ENV_BACKEND = "BAMBOO_TOOL_RETRIEVAL"

#: Tool budget, pinned tools included.
ENV_K = "BAMBOO_TOOL_RETRIEVAL_K"

#: Catalogs at or below this size are passed through unnarrowed.
ENV_MIN_CATALOG = "BAMBOO_TOOL_RETRIEVAL_MIN_CATALOG"

#: Truthy raises the per-question selection line from DEBUG to INFO.
ENV_LOG = "BAMBOO_TOOL_RETRIEVAL_LOG"

DEFAULT_BACKEND = "lexical"
DEFAULT_K = 10
DEFAULT_MIN_CATALOG = 12

#: Tools never scored and never withheld.  They back the "for ALL other
#: questions" route, which is the planner's only fallback; dropping either
#: leaves no graceful degradation, just a planner with no catch-all.  They
#: count against *k* — a pin outside the budget would be a budget in name only.
PINNED_TOOLS: frozenset[str] = frozenset({"panda_doc_search", "panda_doc_bm25"})

#: Description characters indexed per tool.  ``opensearch_promptlog_query``'s
#: description is 8,151 characters, a third of the whole catalog; indexed whole
#: it would dominate the term statistics of a 22-document corpus and drag
#: unrelated questions towards itself.  The opening sentences carry the
#: tool's purpose; the remainder is a field reference.
MAX_INDEXED_DESCRIPTION_CHARS = 800

_BACKENDS = ("off", "lexical", "embedding", "hybrid")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WARNED: set[str] = set()

# BM25 parameters.  The Robertson/Spärck Jones defaults; no tuning is claimed,
# and tuning them without the harness would be guessing.
_BM25_K1 = 1.5
_BM25_B = 0.75


class ToolRetriever(Protocol):
    """A scorer that narrows a tool catalog to the tools a question needs.

    Structurally identical to the protocol in
    :mod:`bamboo.evaluation.tool_retrieval`, deliberately rather than by
    import: the evaluation harness must stay runnable without importing the
    production tool stack, and a backend satisfies both by shape.
    """

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

        Args:
            question: The user's question, verbatim.
            catalog: Catalog entries, each with at least ``name``,
                ``description`` and ``inputSchema``.
            k: Maximum names to return.  Fewer is allowed, more is not.

        Returns:
            List[str]: Names drawn from *catalog*, most relevant first.
        """
        ...


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase alphanumeric terms.

    Punctuation that separates words in tool names — the ``.`` in
    ``atlas.job_stats`` and the ``_`` in ``panda_harvester_workers`` — is a
    separator here, so a question saying "harvester workers" matches a tool
    whose name says ``harvester_workers``.

    Args:
        text: Arbitrary text.

    Returns:
        List[str]: Terms, in order, with duplicates kept.
    """
    return _TOKEN_RE.findall(text.lower())


def _parameter_names(entry: Mapping[str, Any]) -> list[str]:
    """Return a tool's top-level parameter names.

    Parameter *names* are indexed; their descriptions and types are not.  Names
    are short, discriminating and cheap — ``site``, ``queue``, ``job_id`` tell a
    question-matcher a great deal — while schema descriptions would multiply
    the indexed text for little added signal.

    Args:
        entry: A catalog entry.

    Returns:
        List[str]: Parameter names, or an empty list for a malformed schema.
    """
    schema = entry.get("inputSchema")
    if not isinstance(schema, Mapping):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return [str(key) for key in properties]


def index_terms(entry: Mapping[str, Any]) -> list[str]:
    """Return the terms indexed for one tool.

    Args:
        entry: A catalog entry.

    Returns:
        List[str]: Terms from the tool's name, its truncated description and
        its parameter names.
    """
    name = str(entry.get("name", ""))
    description = str(entry.get("description", ""))[:MAX_INDEXED_DESCRIPTION_CHARS]
    terms = _tokenise(name) + _tokenise(description)
    for parameter in _parameter_names(entry):
        terms.extend(_tokenise(parameter))
    return terms


class LexicalRetriever:
    """BM25 over tool names, truncated descriptions and parameter names.

    Builds its index per call.  On a 22-document corpus that is tens of
    microseconds, so caching would trade a measurable correctness risk — a
    stale index after a plugin reload — for an unmeasurable saving.  A corpus
    large enough to need caching is a corpus large enough to justify measuring
    the cache first.
    """

    name = "lexical"

    def score(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, float]]:
        """Score every catalog entry against a question.

        Args:
            question: The user's question.
            catalog: Catalog entries.

        Returns:
            List[Tuple[str, float]]: ``(name, score)`` pairs, highest first.
            Ties are broken by catalog order, so the ranking is deterministic
            for a given catalog — a property the tests and the harness both
            depend on.
        """
        documents = [index_terms(entry) for entry in catalog]
        names = [str(entry.get("name", "")) for entry in catalog]
        if not documents:
            return []

        lengths = [len(doc) for doc in documents]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        total_docs = len(documents)

        frequencies: list[dict[str, int]] = []
        document_count: dict[str, int] = {}
        for doc in documents:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            frequencies.append(counts)
            for term in counts:
                document_count[term] = document_count.get(term, 0) + 1

        query_terms = _tokenise(question)
        scored: list[tuple[str, float]] = []
        for position, counts in enumerate(frequencies):
            length = lengths[position] or 1
            total = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                containing = document_count.get(term, 0)
                # Robertson/Spärck Jones IDF with the +1 that keeps a term
                # present in every document at a small positive weight rather
                # than a negative one.  A term in all 22 tools is uninformative,
                # not evidence against the tools that contain it.
                idf = math.log(1.0 + (total_docs - containing + 0.5) / (containing + 0.5))
                denominator = frequency + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * length / (avg_length or 1.0)
                )
                total += idf * (frequency * (_BM25_K1 + 1.0)) / denominator
            scored.append((names[position], total))

        order = {name: position for position, name in enumerate(names)}
        scored.sort(key=lambda pair: (-pair[1], order[pair[0]]))
        return scored

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

        Tools scoring zero — sharing no term with the question — are dropped
        rather than used to pad the result up to *k*.  Padding would hand the
        planner tools the retriever has no reason to believe in, which is the
        prompt bloat this module exists to remove.

        Args:
            question: The user's question.
            catalog: Catalog entries.
            k: Maximum names to return.

        Returns:
            List[str]: Names, most relevant first.
        """
        if k <= 0:
            return []
        return [name for name, value in self.score(question, catalog) if value > 0.0][:k]


@dataclass(frozen=True)
class RetrievalDecision:
    """What retrieval did for one question, and why.

    Carries the reason in every case, applied or not, because "retrieval did
    nothing" has several causes that need telling apart when a selection looks
    wrong: switched off, catalog too small, no question to score, backend
    failed.

    Attributes:
        applied: Whether the catalog was actually narrowed.
        reason: Short machine-readable cause (``ok``, ``disabled``,
            ``catalog_small``, ``no_question``, ``backend_unavailable``,
            ``backend_error``, ``empty_result``).
        backend: Backend name, or ``"off"``.
        k: Tool budget in force, pins included.
        kept: Surviving tool names, pinned first, then by descending score.
        withheld: Names removed from the catalog.
        scores: ``(name, score)`` for every scored tool, highest first.  Empty
            when nothing was scored.
        pinned: Pins present in the catalog and exempt from scoring.
    """

    applied: bool
    reason: str
    backend: str
    k: int
    kept: tuple[str, ...]
    withheld: tuple[str, ...] = ()
    scores: tuple[tuple[str, float], ...] = ()
    pinned: frozenset[str] = field(default_factory=frozenset)


def active_backend() -> str:
    """Return the retrieval backend named by the environment.

    Read at call time rather than at import, matching
    :mod:`bamboo.tools._tool_profiles`, so the variable stays testable without
    reimporting the module.

    Returns:
        str: One of :data:`_BACKENDS`.  Unset yields :data:`DEFAULT_BACKEND`
        silently; an unrecognised value yields it after a warning logged once
        per distinct value.  Raising would turn a configuration typo into a
        server that fails on every question — and failing *closed* here would
        be worse still, since the fallback is simply the behaviour that shipped
        for the last two years.
    """
    raw = os.getenv(ENV_BACKEND, "")
    normalised = raw.strip().lower()
    if not normalised:
        return DEFAULT_BACKEND
    if normalised not in _BACKENDS:
        if raw not in _WARNED:
            _WARNED.add(raw)
            logger.warning(
                "%s=%r is not a recognised retrieval backend (expected one of %s); "
                "falling back to %r.",
                ENV_BACKEND,
                raw,
                ", ".join(_BACKENDS),
                DEFAULT_BACKEND,
            )
        return DEFAULT_BACKEND
    return normalised


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value used when unset, unparseable or non-positive.

    Returns:
        int: The configured value, or *default* after a warning logged once per
        distinct bad value.
    """
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        key = f"{name}={raw}"
        if key not in _WARNED:
            _WARNED.add(key)
            logger.warning(
                "%s=%r is not a positive integer; falling back to %d.", name, raw, default
            )
        return default
    return value


def active_k() -> int:
    """Return the configured tool budget.

    Returns:
        int: Budget including pinned tools.
    """
    return _positive_int(ENV_K, DEFAULT_K)


def active_min_catalog() -> int:
    """Return the catalog size below which retrieval is skipped.

    Returns:
        int: Minimum catalog size for retrieval to apply.
    """
    return _positive_int(ENV_MIN_CATALOG, DEFAULT_MIN_CATALOG)


def _encoder_unavailable() -> type[BaseException]:
    """Return the exception type signalling a missing embedding model.

    Resolved lazily so that importing this module never imports the embedding
    module.  Falls back to a type that cannot be raised when the embedding
    module is itself unimportable, which leaves the generic handler to deal
    with it rather than letting the lookup become the failure.

    Returns:
        type[BaseException]: The exception class to treat as "no model".
    """
    try:
        from bamboo.tools._tool_retrieval_embedding import (  # noqa: PLC0415
            EncoderUnavailable,
        )

        return EncoderUnavailable
    except Exception:  # pragma: no cover - only when the module itself is broken
        class _Unreachable(BaseException):
            """Never raised."""

        return _Unreachable


def _build_retriever(backend: str) -> ToolRetriever | None:
    """Instantiate a backend by name.

    Args:
        backend: Backend name.

    Returns:
        Optional[ToolRetriever]: The retriever, or ``None`` for ``off`` and
        anything unrecognised.
    """
    if backend == "lexical":
        return LexicalRetriever()
    if backend in ("embedding", "hybrid"):
        # Imported here, not at module level: the embedding module is only a
        # few hundred lines of pure Python, but importing it is the first step
        # on a path that ends in loading an ONNX runtime, and a process that
        # never plans must not take that step.
        from bamboo.tools._tool_retrieval_embedding import (  # noqa: PLC0415
            EmbeddingRetriever,
            HybridRetriever,
        )

        return EmbeddingRetriever() if backend == "embedding" else HybridRetriever()
    return None


def format_decision(decision: RetrievalDecision) -> str:
    """Render a decision as one human-readable line.

    The line every test run and every ``/tracing`` inspection is read from, so
    it names every surviving tool with its score and every withheld one.  A
    summary that said only "kept 10 of 22" would answer the easy question and
    hide the one actually being asked, which is always *which ten*.

    Args:
        decision: The decision to render.

    Returns:
        str: A single line, no trailing newline.
    """
    if not decision.applied:
        return (
            f"tool retrieval: not applied (reason={decision.reason}, "
            f"backend={decision.backend}, catalog={len(decision.kept)} tools)"
        )
    lookup = dict(decision.scores)
    kept_parts = [
        f"{name}(pinned)" if name in decision.pinned else f"{name}={lookup.get(name, 0.0):.2f}"
        for name in decision.kept
    ]
    withheld_parts = [f"{name}={lookup.get(name, 0.0):.2f}" for name in decision.withheld]
    return (
        f"tool retrieval: backend={decision.backend} k={decision.k} "
        f"kept {len(decision.kept)}/{len(decision.kept) + len(decision.withheld)} "
        f"[{', '.join(kept_parts)}] "
        f"withheld [{', '.join(withheld_parts) or 'none'}]"
    )


def _log_decision(decision: RetrievalDecision) -> None:
    """Log a decision, and mirror it into the trace stream.

    Args:
        decision: The decision to report.
    """
    level = logging.INFO if os.getenv(ENV_LOG, "").strip() else logging.DEBUG
    logger.log(level, "%s", format_decision(decision))

    try:
        from bamboo.tracing import EVENT_RETRIEVAL, emit_sync  # noqa: PLC0415

        emit_sync(
            EVENT_RETRIEVAL,
            tool="tool_retrieval",
            applied=decision.applied,
            reason=decision.reason,
            backend=decision.backend,
            k=decision.k,
            kept=list(decision.kept),
            withheld=list(decision.withheld),
            scores=[[name, round(value, 4)] for name, value in decision.scores],
        )
    except Exception:  # pragma: no cover - tracing must never break a question
        logger.debug("tool retrieval: trace emission failed", exc_info=True)


def select_tools(
    question: str,
    catalog: Sequence[Mapping[str, Any]],
    pinned: frozenset[str] = PINNED_TOOLS,
) -> RetrievalDecision:
    """Decide which tools the planner should see for one question.

    Args:
        question: The user's question.  Empty disables retrieval for this call;
            there is nothing to score against, and scoring against an empty
            string would rank by document length alone.
        catalog: Catalog entries.
        pinned: Tools exempt from scoring, counted against *k*.

    Returns:
        RetrievalDecision: The decision, already logged and traced.
    """
    names = tuple(str(entry.get("name", "")) for entry in catalog)
    backend = active_backend()

    def _passthrough(reason: str) -> RetrievalDecision:
        """Build a no-narrowing decision.

        Args:
            reason: Why retrieval did not apply.

        Returns:
            RetrievalDecision: A decision keeping the whole catalog.
        """
        decision = RetrievalDecision(
            applied=False, reason=reason, backend=backend, k=active_k(), kept=names
        )
        _log_decision(decision)
        return decision

    if backend == "off":
        return _passthrough("disabled")
    if not question.strip():
        return _passthrough("no_question")

    k = active_k()
    if len(catalog) <= max(active_min_catalog(), k):
        return _passthrough("catalog_small")

    retriever = _build_retriever(backend)
    if retriever is None:  # pragma: no cover - active_backend already filtered
        return _passthrough("disabled")

    available_pins = frozenset(pinned) & frozenset(names)
    budget = max(0, k - len(available_pins))
    scorable = [entry for entry in catalog if str(entry.get("name", "")) not in available_pins]

    try:
        ranked = retriever.retrieve(question, scorable, budget)
        scores = tuple(
            getattr(retriever, "score")(question, scorable)
            if hasattr(retriever, "score")
            else ()
        )
    except _encoder_unavailable() as exc:
        # An absent optional dependency, not a bug. Reported at WARNING and
        # with its own reason, because "install requirements-rag.txt" and
        # "the retriever is broken" need different responses and would
        # otherwise be the same line in the log.
        logger.warning(
            "tool retrieval backend %r has no embedding model (%s); "
            "falling back to the full catalog.",
            backend,
            exc,
        )
        return _passthrough("backend_unavailable")
    except Exception:
        # Loud, not silent: the question still gets answered from the full
        # catalog, but a retrieval layer that has quietly stopped retrieving
        # presents as nothing worse than a larger prompt, and would stay broken
        # for as long as nobody read the token bill.
        logger.exception(
            "tool retrieval failed (backend=%s); falling back to the full catalog.", backend
        )
        return _passthrough("backend_error")

    unknown = [name for name in ranked if name not in names]
    if unknown:
        logger.error(
            "tool retrieval returned names absent from the catalog: %s; "
            "falling back to the full catalog.",
            sorted(unknown),
        )
        return _passthrough("backend_error")

    if not ranked:
        # Every tool scored zero.  Keeping the pins alone would leave the
        # planner with only the documentation route, so pass the catalog
        # through and say why.
        return _passthrough("empty_result")

    kept_set = available_pins | set(ranked)
    kept = tuple(sorted(available_pins)) + tuple(ranked)
    withheld = tuple(name for name in names if name not in kept_set)

    decision = RetrievalDecision(
        applied=True,
        reason="ok",
        backend=backend,
        k=k,
        kept=kept,
        withheld=withheld,
        scores=scores,
        pinned=available_pins,
    )
    _log_decision(decision)
    return decision


def narrow_catalog(
    question: str,
    catalog: Sequence[Mapping[str, Any]],
    pinned: frozenset[str] = PINNED_TOOLS,
) -> tuple[list[dict[str, Any]], RetrievalDecision]:
    """Narrow a catalog to the tools a question needs.

    Args:
        question: The user's question.
        catalog: Catalog entries.
        pinned: Tools exempt from scoring.

    Returns:
        Tuple[List[Dict[str, Any]], RetrievalDecision]: The surviving entries
        in catalog order, and the decision that produced them.  Catalog order
        is kept rather than score order so that the prompt's tool list does not
        reorder itself between questions, which would make two planner prompts
        differ for reasons unrelated to their content.
    """
    decision = select_tools(question, catalog, pinned=pinned)
    if not decision.applied:
        return [dict(entry) for entry in catalog], decision
    kept = set(decision.kept)
    return [dict(entry) for entry in catalog if str(entry.get("name", "")) in kept], decision
