"""Tests for the embedding and hybrid retrieval backends.

No model is loaded.  Every test supplies a deterministic stub encoder, which is
not a compromise: what needs pinning here is the machinery around the model —
lazy resolution, cache invalidation, dimension checking, rank fusion — and a
real model would make those tests slow, network-dependent and non-deterministic
without testing any of them better.  Whether MiniLM beats BM25 on this corpus is
a question for ``scripts/eval_tool_retrieval.py``, not for a unit test.

The encoder's own contract — that a thing which constructs can still be
unusable — is pinned separately in :class:`TestResolveEncoder`, because that is
exactly the failure this module was first written to miss.
"""
from __future__ import annotations

import math

import pytest

from bamboo.tools._tool_retrieval_embedding import (
    RRF_K,
    EmbeddingRetriever,
    EncoderUnavailable,
    HybridRetriever,
    _normalise,
    catalog_fingerprint,
    indexed_text,
    resolve_encoder,
)
from bamboo.tools.tool_retrieval import (
    ENV_BACKEND,
    MAX_INDEXED_DESCRIPTION_CHARS,
    index_terms,
    select_tools,
)

#: Stub embedding space.  Each text becomes a count vector over these tokens,
#: so cosine behaves like weighted term overlap and expectations can be
#: reasoned about by hand.
VOCABULARY = ("alpha", "beta", "gamma", "delta")


class StubEncoder:
    """Deterministic encoder over :data:`VOCABULARY`.

    Attributes:
        calls: Texts passed to each invocation, in order, so tests can assert
            how often encoding actually happened.
    """

    def __init__(
        self,
        dimension: int = len(VOCABULARY),
        synonyms: dict[str, str] | None = None,
    ) -> None:
        """Initialise the stub.

        Args:
            dimension: Length of the vectors produced.  Values other than the
                vocabulary size exist so dimension-mismatch handling can be
                exercised.
            synonyms: Surface word to vocabulary token.  Models the one
                property that distinguishes an embedding from term matching —
                two different words landing in the same place — which is the
                only reason the embedding backend exists.
        """
        self.dimension = dimension
        self.synonyms = synonyms or {}
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        """Encode texts as counts over the vocabulary.

        Args:
            texts: Texts to encode.

        Returns:
            list[list[float]]: One vector per text.
        """
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            for surface, token in self.synonyms.items():
                lowered = lowered.replace(surface, token)
            vector = [float(lowered.count(token)) for token in VOCABULARY]
            vector = (vector + [0.0] * self.dimension)[: self.dimension]
            if not any(vector):
                vector[0] = 0.001
            vectors.append(vector)
        return vectors


def _entry(name: str, description: str) -> dict:
    """Build a catalog entry.

    Args:
        name: Wire name.
        description: Description text.

    Returns:
        dict: A catalog entry.
    """
    return {"name": name, "description": description, "inputSchema": {"type": "object"}}


@pytest.fixture
def catalog() -> list[dict]:
    """Return a small catalog spanning the stub vocabulary.

    Returns:
        list[dict]: Catalog entries.
    """
    return [
        _entry("tool_a", "alpha alpha concerns"),
        _entry("tool_b", "beta concerns"),
        _entry("tool_c", "gamma concerns"),
        _entry("tool_d", "delta concerns"),
    ]


class TestNormalise:
    """Vector normalisation."""

    def test_it_produces_a_unit_vector(self) -> None:
        """A normalised vector has length one."""
        vector = _normalise([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)

    def test_a_zero_vector_survives(self) -> None:
        """An all-zero embedding does not divide by zero.

        A degenerate vector should give a zero similarity to everything, not
        take the planner down.
        """
        assert _normalise([0.0, 0.0]) == (0.0, 0.0)


class TestFingerprint:
    """Cache keying."""

    def test_it_is_stable_for_identical_catalogs(self, catalog: list[dict]) -> None:
        """The same catalog hashes the same way twice."""
        assert catalog_fingerprint(catalog) == catalog_fingerprint(list(catalog))

    def test_it_changes_when_indexed_text_changes(self, catalog: list[dict]) -> None:
        """An edit inside the indexed window invalidates the cache."""
        before = catalog_fingerprint(catalog)
        changed = [dict(e) for e in catalog]
        changed[0]["description"] = "beta concerns instead"
        assert catalog_fingerprint(changed) != before

    def test_it_ignores_text_past_the_truncation_point(self, catalog: list[dict]) -> None:
        """An edit beyond the indexed window does not invalidate the cache.

        The fingerprint hashes indexed terms rather than raw entries, so a
        change retrieval cannot see does not force a re-encode of the catalog.
        """
        padded = [dict(e) for e in catalog]
        padded[0]["description"] = "alpha alpha concerns " + "x" * (
            MAX_INDEXED_DESCRIPTION_CHARS * 2
        )
        before = catalog_fingerprint(padded)
        padded[0]["description"] += "tail-change-well-past-the-cap"
        assert catalog_fingerprint(padded) == before

    def test_renaming_a_tool_changes_it(self, catalog: list[dict]) -> None:
        """Names are part of the key, not only descriptions."""
        renamed = [dict(e) for e in catalog]
        renamed[0]["name"] = "tool_z"
        assert catalog_fingerprint(renamed) != catalog_fingerprint(catalog)


class TestIndexedText:
    """What the embedding backend encodes."""

    def test_it_matches_the_lexical_corpus(self, catalog: list[dict]) -> None:
        """Both backends see the same text.

        Otherwise a comparison between them would measure two preprocessing
        choices rather than two scorers.
        """
        for entry in catalog:
            assert indexed_text(entry) == " ".join(index_terms(entry))


class TestEmbeddingRetriever:
    """The embedding backend."""

    def test_construction_loads_no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Building a retriever never resolves an encoder.

        ``_build_retriever`` constructs one whenever the backend is merely
        *named*, including on a passthrough; if that loaded an ONNX runtime,
        every planner call on a misconfigured host would pay seconds for
        nothing.
        """
        monkeypatch.setattr(
            "bamboo.tools._tool_retrieval_embedding.resolve_encoder",
            lambda: (_ for _ in ()).throw(AssertionError("resolved too early")),
        )
        EmbeddingRetriever()

    def test_it_ranks_the_closest_tool_first(self, catalog: list[dict]) -> None:
        """A question in the stub space retrieves its matching tool."""
        retriever = EmbeddingRetriever(encoder=StubEncoder())
        assert retriever.retrieve("gamma", catalog, 4)[0] == "tool_c"

    def test_it_respects_k(self, catalog: list[dict]) -> None:
        """Never more than *k* names come back."""
        retriever = EmbeddingRetriever(encoder=StubEncoder())
        assert len(retriever.retrieve("alpha", catalog, 2)) == 2

    def test_a_non_positive_k_returns_nothing(self, catalog: list[dict]) -> None:
        """A zero budget returns nothing rather than everything."""
        assert EmbeddingRetriever(encoder=StubEncoder()).retrieve("alpha", catalog, 0) == []

    def test_the_catalog_is_encoded_once(self, catalog: list[dict]) -> None:
        """Repeat questions re-encode the query but not the catalog.

        One index encode plus one query encode each; three questions therefore
        make four calls, not six.
        """
        encoder = StubEncoder()
        retriever = EmbeddingRetriever(encoder=encoder)
        for question in ("alpha", "beta", "gamma"):
            retriever.retrieve(question, catalog, 2)
        assert len(encoder.calls) == 4
        assert len(encoder.calls[0]) == len(catalog)

    def test_a_changed_catalog_is_re_encoded(self, catalog: list[dict]) -> None:
        """A plugin changing the catalog invalidates the cached vectors.

        A cache that survived this would serve vectors for tools that no longer
        exist, silently.
        """
        encoder = StubEncoder()
        retriever = EmbeddingRetriever(encoder=encoder)
        retriever.retrieve("alpha", catalog, 2)
        extended = catalog + [_entry("tool_e", "alpha beta")]
        retriever.retrieve("alpha", extended, 2)
        index_encodes = [call for call in encoder.calls if len(call) > 1]
        assert len(index_encodes) == 2

    def test_ties_break_on_catalog_order(self) -> None:
        """Equal similarities rank deterministically."""
        catalog = [_entry(f"tool_{i}", "alpha concerns") for i in range(4)]
        retriever = EmbeddingRetriever(encoder=StubEncoder())
        assert retriever.retrieve("alpha", catalog, 4) == [f"tool_{i}" for i in range(4)]

    def test_an_empty_catalog_scores_nothing(self) -> None:
        """No catalog, no scores, no exception."""
        assert EmbeddingRetriever(encoder=StubEncoder()).score("alpha", []) == []

    def test_a_dimension_mismatch_is_a_hard_error(self, catalog: list[dict]) -> None:
        """Mixed dimensions raise rather than scoring nonsense.

        Mismatched dimensions mean the index and the query came from different
        models, so every score computed from them is meaningless rather than
        merely degraded — the one case in this module that must not fail open.
        """
        encoder = StubEncoder()
        retriever = EmbeddingRetriever(encoder=encoder)
        retriever.retrieve("alpha", catalog, 2)
        retriever._encoder = StubEncoder(dimension=8)  # type: ignore[assignment]
        with pytest.raises(ValueError, match="dimension mismatch"):
            retriever.score("alpha", catalog)


class TestHybridRetriever:
    """Rank fusion of the two backends."""

    def test_it_recovers_a_tool_lexical_alone_would_miss(self) -> None:
        """A paraphrased question still reaches its tool.

        The motivating case: the question shares no term with the description
        of the tool that answers it, so BM25 scores it zero and never ranks it.
        The embedding half supplies the rank, and fusion keeps it.
        """
        catalog = [
            _entry("promptlog", "session token cost records"),
            _entry("other_1", "beta unrelated"),
            _entry("other_2", "gamma unrelated"),
        ]
        question = "lowest ratings last month"
        # "ratings" and "session" are different words that mean nearby things;
        # BM25 cannot see that and the encoder can, which is the whole case.
        encoder = StubEncoder(synonyms={"ratings": "alpha", "session": "alpha"})
        hybrid = HybridRetriever(encoder=encoder)

        lexically_matched = [
            name for name, value in hybrid._lexical.score(question, catalog) if value > 0.0
        ]
        assert "promptlog" not in lexically_matched
        assert "promptlog" in hybrid.retrieve(question, catalog, 2)

    def test_it_keeps_a_tool_both_backends_agree_on_first(self) -> None:
        """Agreement outranks a single backend's enthusiasm."""
        catalog = [
            _entry("agreed", "alpha concerns alpha"),
            _entry("lexical_only", "concerns concerns"),
            _entry("filler", "delta"),
        ]
        ranked = HybridRetriever(encoder=StubEncoder()).retrieve("alpha concerns", catalog, 3)
        assert ranked[0] == "agreed"

    def test_zero_scoring_lexical_tools_contribute_no_rank(self) -> None:
        """BM25 ranks only what it scored.

        Giving every unmatched tool a tail rank would invent evidence: the
        lexical backend has not ranked them, it has declined to.
        """
        catalog = [_entry("a", "alpha"), _entry("b", "beta")]
        hybrid = HybridRetriever(encoder=StubEncoder())
        scores = dict(hybrid.score("alpha", catalog))
        # 'b' is lexically unmatched, so its only contribution is the embedding
        # rank; it cannot exceed one reciprocal-rank term.
        assert scores["b"] <= 1.0 / (RRF_K + 1)

    def test_a_non_positive_k_returns_nothing(self) -> None:
        """A zero budget returns nothing."""
        catalog = [_entry("a", "alpha")]
        assert HybridRetriever(encoder=StubEncoder()).retrieve("alpha", catalog, 0) == []


class TestResolveEncoder:
    """Encoder resolution, and the failure it originally missed."""

    def test_an_encoder_that_constructs_but_fails_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Construction is not evidence of a working model.

        ChromaDB's DefaultEmbeddingFunction constructs without touching the
        model and downloads it on first call, so a host that cannot reach the
        cache raised a bare ValueError from inside the download path — after
        the caller had already accepted the encoder. Probing moves that failure
        to resolution, where it is reportable.
        """

        class _Broken:
            def __call__(self, texts):
                """Fail on use."""
                raise ValueError("corrupted download")

        monkeypatch.setitem(
            __import__("sys").modules,
            "chromadb.utils",
            type("M", (), {"embedding_functions": type("E", (), {
                "DefaultEmbeddingFunction": staticmethod(_Broken)
            })})(),
        )
        with pytest.raises(EncoderUnavailable) as excinfo:
            resolve_encoder()
        assert "corrupted download" in str(excinfo.value)

    def test_the_error_names_both_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure says what was tried and how to proceed."""
        with pytest.raises(EncoderUnavailable) as excinfo:
            monkeypatch.setitem(__import__("sys").modules, "chromadb.utils", None)
            monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
            resolve_encoder()
        message = str(excinfo.value)
        assert "chromadb" in message and "sentence-transformers" in message
        assert "BAMBOO_TOOL_RETRIEVAL=lexical" in message


class TestPolicyIntegration:
    """How a missing model reaches the planner."""

    def test_a_missing_model_is_its_own_reason(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An absent model passes the catalog through as ``backend_unavailable``.

        Distinct from ``backend_error`` on purpose: "install
        requirements-rag.txt" and "the retriever is broken" need different
        responses, and would otherwise be the same line in the log.
        """
        import logging

        monkeypatch.setenv(ENV_BACKEND, "embedding")
        monkeypatch.setattr(
            "bamboo.tools._tool_retrieval_embedding.resolve_encoder",
            lambda: (_ for _ in ()).throw(EncoderUnavailable("no model here")),
        )
        catalog = [_entry(f"tool_{i}", f"subject{i}") for i in range(20)]
        with caplog.at_level(logging.WARNING):
            decision = select_tools("subject3", catalog)
        assert not decision.applied
        assert decision.reason == "backend_unavailable"
        assert "no embedding model" in caplog.text or "no model here" in caplog.text
