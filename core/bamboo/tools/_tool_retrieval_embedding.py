"""Embedding and hybrid backends for tool retrieval.

The lexical backend misses a specific shape of question: one whose wording
shares no term with the description of the tool that answers it.  *"Which
questions received the lowest ratings last month?"* needs
``opensearch_promptlog_query``, whose description talks about prompt logs,
sessions and token costs — correct, and lexically disjoint.  That is the gap an
embedding model exists to close.

**Which model.**  The same one the document corpus already uses:
``all-MiniLM-L6-v2``, reached through ChromaDB's default embedding function,
which is installed wherever the RAG tools work and cached on the deployment
hosts.  A second model would mean a second download, a second cache and a
second thing to be wrong about, for a corpus of 22 strings.

**No on-disk precomputed index, and the plan said there would be one.**  The
reasoning that motivated it does not survive contact with the query path: every
question must be encoded at query time, so the model has to be loaded whatever
happens.  A precomputed index would save encoding 22 short documents — tens of
milliseconds — while leaving the actual cost, the one-to-three second model
load, exactly where it was.  What does help is an in-process vector cache keyed
by a hash of the catalog's indexed text, which costs a dict and is invalidated
correctly when a plugin changes.  See :class:`EmbeddingRetriever`.

**Nothing is loaded at import.**  The encoder is resolved on first use, so a
process that never plans — ``_mcp_caller``'s one-shot subprocesses, the test
suite, ``python -m bamboo.server`` at start-up — pays nothing.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Mapping, Protocol, Sequence

from bamboo.tools.tool_retrieval import LexicalRetriever, _positive_int, index_terms

logger = logging.getLogger(__name__)

#: Overrides :data:`RRF_K`.  Exposed so the tuning below can be settled by
#: measurement rather than argument.
ENV_RRF_K = "BAMBOO_TOOL_RETRIEVAL_RRF_K"

#: Reciprocal-rank-fusion constant.
#:
#: The literature's default is 60, and on this catalog 60 is wrong.  With *N*
#: scorable tools, a tool ranked first by one backend scores ``1/(K+1)`` while a
#: tool ranked last by *both* scores ``2/(K+N)``.  At K=60 and N=20 those are
#: 0.0164 and 0.0250: **every** tool both backends rank beats **every** tool only
#: one ranks, whatever the ranks.  Fusion degenerates into "the intersection,
#: then the leftovers", and the single-source find that hybrid retrieval exists
#: to rescue is exactly what it cannot rescue — which is what the harness
#: observed, with hybrid missing the one case the embedding backend alone
#: solved.
#:
#: 60 assumes rankings over thousands of documents.  A sweep confirmed the
#: prediction exactly: on a 22-tool catalog hybrid scored 0.992 at K=10, 20 and
#: 60, and 1.000 at K=3 and K=5.  5 is the default.
#:
#: Treat that 1.000 with suspicion rather than pride.  K was chosen on the same
#: 120 cases it is reported against, the gain is a single case, and it is the
#: case the sweep went looking for — a fit, not a validated improvement.  The
#: same case was subsequently fixed with no model at all, by writing the
#: offending tool's description so that its own vocabulary appears in the
#: indexed window.
RRF_K = 5


def active_rrf_k() -> int:
    """Return the configured rank-fusion constant.

    Returns:
        int: :data:`RRF_K` unless ``BAMBOO_TOOL_RETRIEVAL_RRF_K`` overrides it.
    """
    return _positive_int(ENV_RRF_K, RRF_K)


class EncoderUnavailable(RuntimeError):
    """Raised when no embedding model can be reached.

    Distinct from a model that loads and then misbehaves: this one means the
    optional dependency is absent or the model is not cached, which is an
    install-time condition and must read differently in the log from a runtime
    failure.
    """


class Encoder(Protocol):
    """Turns texts into vectors."""

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode a batch of texts.

        Args:
            texts: Texts to encode.

        Returns:
            Sequence[Sequence[float]]: One vector per text, in order.
        """
        ...


def _probe(encoder: Encoder) -> Encoder:
    """Return *encoder* once it has produced a usable vector.

    Constructing an embedding function is not evidence that it works.
    ChromaDB's ``DefaultEmbeddingFunction()`` constructs without touching the
    model and downloads it on first call, so a host that cannot reach the model
    cache raises a bare ``ValueError`` from deep inside the download path —
    after the caller has already concluded the encoder is fine.  Probing moves
    that failure to resolution time, where it can be reported as what it is.

    The probe costs one encode of one short string, which is dominated by the
    model load it forces, and that load was going to happen on the next call
    anyway.

    Args:
        encoder: A candidate embedding function.

    Returns:
        Encoder: The same encoder.

    Raises:
        Exception: Whatever the encoder raised, for the caller to wrap.
    """
    vectors = encoder(["probe"])
    if not vectors or not len(vectors[0]):
        raise ValueError("encoder returned an empty vector for a non-empty input")
    return encoder


def resolve_encoder() -> Encoder:
    """Return an embedding function, preferring the one the doc corpus uses.

    Each candidate is probed before being accepted, so this either returns an
    encoder that has demonstrably produced a vector or raises.

    Returns:
        Encoder: A callable taking a list of texts and returning vectors.

    Raises:
        EncoderUnavailable: If neither ChromaDB's default embedding function
            nor ``sentence-transformers`` yields a working model.  Both are
            optional extras, and the model itself may be absent from the cache
            on a host with no outbound network; the planner must keep working
            in every one of those cases.
    """
    errors: list[str] = []

    try:
        from chromadb.utils import embedding_functions  # noqa: PLC0415

        return _probe(embedding_functions.DefaultEmbeddingFunction())  # type: ignore[arg-type]
    except Exception as exc:
        errors.append(f"chromadb default embedding function: {exc!r}")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _encode(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            """Encode with sentence-transformers.

            Args:
                texts: Texts to encode.

            Returns:
                Sequence[Sequence[float]]: Vectors.
            """
            return [list(map(float, vector)) for vector in model.encode(list(texts))]

        return _probe(_encode)
    except Exception as exc:
        errors.append(f"sentence-transformers: {exc!r}")

    raise EncoderUnavailable(
        "no embedding model available (" + "; ".join(errors) + "). "
        "Install requirements-rag.txt and warm the model cache, or use "
        "BAMBOO_TOOL_RETRIEVAL=lexical."
    )


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale a vector to unit length.

    Normalising once at index time turns every later cosine into a dot product,
    which matters only because it removes a per-query square root from a loop
    that runs over the whole catalog.

    Args:
        vector: The vector.

    Returns:
        Tuple[float, ...]: The unit vector, or the original when its norm is
        zero — a degenerate embedding that would otherwise divide by zero.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(float(value) for value in vector)
    return tuple(float(value) / norm for value in vector)


def catalog_fingerprint(catalog: Sequence[Mapping[str, Any]]) -> str:
    """Return a hash of the text a catalog contributes to the index.

    Hashes the *indexed* terms rather than the raw entries, so a change that
    does not affect retrieval — a reordered JSON schema, a tweak past the
    description truncation point — does not invalidate the cache.

    Args:
        catalog: Catalog entries.

    Returns:
        str: A hex digest.
    """
    digest = hashlib.sha256()
    for entry in catalog:
        digest.update(str(entry.get("name", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(" ".join(index_terms(entry)).encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def indexed_text(entry: Mapping[str, Any]) -> str:
    """Return the text embedded for one tool.

    Uses the same terms the lexical backend indexes — name split on separators,
    truncated description, parameter names — so the two backends see the same
    corpus and a comparison between them measures the scorer rather than two
    different preprocessing choices.

    Args:
        entry: A catalog entry.

    Returns:
        str: Space-joined terms.
    """
    return " ".join(index_terms(entry))


class EmbeddingRetriever:
    """Cosine similarity over ``all-MiniLM-L6-v2`` vectors.

    Holds one cached index, keyed by :func:`catalog_fingerprint`.  A single
    entry is enough: the catalog is rebuilt identically on every planner call,
    so the cache hits from the second question onward and is dropped the moment
    a plugin changes the catalog's indexed text.
    """

    name = "embedding"

    def __init__(self, encoder: Encoder | None = None) -> None:
        """Initialise the retriever.

        Args:
            encoder: Embedding function to use.  ``None`` resolves one lazily
                on first use, so constructing a retriever — which happens
                whenever the backend is merely *named* — loads no model.
        """
        self._encoder = encoder
        self._fingerprint: str = ""
        self._vectors: dict[str, tuple[float, ...]] = {}

    def _get_encoder(self) -> Encoder:
        """Return the encoder, resolving it on first use.

        Returns:
            Encoder: The embedding function.

        Raises:
            EncoderUnavailable: If no model can be reached.
        """
        if self._encoder is None:
            self._encoder = resolve_encoder()
        return self._encoder

    def _index(self, catalog: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, ...]]:
        """Return unit vectors for a catalog, encoding only when it has changed.

        Args:
            catalog: Catalog entries.

        Returns:
            Dict[str, Tuple[float, ...]]: Tool name to unit vector.
        """
        fingerprint = catalog_fingerprint(catalog)
        if fingerprint == self._fingerprint and self._vectors:
            return self._vectors

        texts = [indexed_text(entry) for entry in catalog]
        names = [str(entry.get("name", "")) for entry in catalog]
        vectors = self._get_encoder()(texts) if texts else []
        self._vectors = {
            name: _normalise(vector) for name, vector in zip(names, vectors)
        }
        self._fingerprint = fingerprint
        return self._vectors

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
            List[Tuple[str, float]]: ``(name, cosine)`` pairs, highest first,
            ties broken by catalog order for determinism.

        Raises:
            ValueError: If a tool vector and the query vector have different
                dimensions.  A hard error, not a warning: mismatched dimensions
                mean the index and the query came from different models, and
                every score computed from them would be meaningless rather than
                merely degraded.
        """
        vectors = self._index(catalog)
        if not vectors:
            return []

        query = _normalise(self._get_encoder()([question])[0])
        names = [str(entry.get("name", "")) for entry in catalog]
        order = {name: position for position, name in enumerate(names)}

        scored: list[tuple[str, float]] = []
        for name in names:
            vector = vectors.get(name)
            if vector is None:  # pragma: no cover - index covers the catalog
                continue
            if len(vector) != len(query):
                raise ValueError(
                    f"embedding dimension mismatch for {name!r}: index has "
                    f"{len(vector)}, query has {len(query)}. The tool index and "
                    "the query were produced by different models."
                )
            scored.append((name, sum(a * b for a, b in zip(vector, query))))

        scored.sort(key=lambda pair: (-pair[1], order[pair[0]]))
        return scored

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

        Unlike the lexical backend there is no zero-score cut: cosine gives
        every tool a defensible similarity, so truncating at *k* is the only
        filter and the budget is always spent in full.

        Args:
            question: The user's question.
            catalog: Catalog entries.
            k: Maximum names to return.

        Returns:
            List[str]: Names, most relevant first.
        """
        if k <= 0:
            return []
        return [name for name, _ in self.score(question, catalog)][:k]


class HybridRetriever:
    """Reciprocal rank fusion of the lexical and embedding backends.

    Exists because the two fail on different questions.  Lexical is strong on
    the rare jargon that carries most of this catalog's meaning and blind to
    paraphrase; the embedding model is the reverse.  RRF combines rankings
    rather than scores, so it needs no calibration between a BM25 score and a
    cosine — which is the part that usually goes wrong when the two are mixed
    by weighted sum.

    The harness's verdict so far is that it does not beat lexical alone on this
    catalog; see :data:`RRF_K` for why the original tuning made that outcome
    structurally certain, and what to re-measure.
    """

    name = "hybrid"

    def __init__(self, encoder: Encoder | None = None) -> None:
        """Initialise the retriever.

        Args:
            encoder: Embedding function passed to the embedding half; ``None``
                resolves lazily.
        """
        self._lexical = LexicalRetriever()
        self._embedding = EmbeddingRetriever(encoder=encoder)

    def score(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, float]]:
        """Score every catalog entry by fused rank.

        Args:
            question: The user's question.
            catalog: Catalog entries.

        Returns:
            List[Tuple[str, float]]: ``(name, rrf_score)`` pairs, highest
            first.  Scores are rank-derived and not comparable with either
            backend's own scale.
        """
        names = [str(entry.get("name", "")) for entry in catalog]
        order = {name: position for position, name in enumerate(names)}

        fused: dict[str, float] = {name: 0.0 for name in names}
        rrf_k = active_rrf_k()

        # Only positively-scoring lexical entries contribute a rank. A tool
        # sharing no term with the question has not been ranked by BM25 at all,
        # and giving it a tail rank would be inventing evidence.
        lexical = [
            name for name, value in self._lexical.score(question, catalog) if value > 0.0
        ]
        for rank, name in enumerate(lexical):
            fused[name] += 1.0 / (rrf_k + rank + 1)

        for rank, (name, _) in enumerate(self._embedding.score(question, catalog)):
            fused[name] += 1.0 / (rrf_k + rank + 1)

        scored = [(name, fused[name]) for name in names]
        scored.sort(key=lambda pair: (-pair[1], order[pair[0]]))
        return scored

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

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
