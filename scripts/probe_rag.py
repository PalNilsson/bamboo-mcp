#!/usr/bin/env python3
"""Smoke-test the Bamboo RAG corpus (ChromaDB vector store + BM25 index).

Runs a small set of canonical queries against both the vector-search
(``panda_doc_search``) and keyword-search (``panda_doc_bm25``) tools and
verifies that each query returns at least one relevant result.  A fresh or
broken corpus will fail these checks, surfacing the problem before the live
system is queried by a user.

Exit codes
----------
0   All checks passed.
1   One or more checks failed (corpus empty, missing hits, or connection error).
2   Dependency or configuration error (chromadb not installed, path missing).

Usage::

    python scripts/probe_rag.py
    python scripts/probe_rag.py --path ~/data/chroma_db --collection bamboo_docs
    python scripts/probe_rag.py --verbose
    python scripts/probe_rag.py --query "looping job" --query "pilot"

Arguments
---------
--path PATH
    ChromaDB directory.  Defaults to ``$BAMBOO_CHROMA_PATH`` then ``./chroma_db``.
--collection COLLECTION
    Collection name.  Defaults to ``$BAMBOO_CHROMA_COLLECTION`` then ``bamboo_docs``.
--top-k N
    Results to request per query (default: 3).
--min-score FLOAT
    Minimum cosine similarity score [0–1] for a result to be considered a hit
    (default: 0.3).  Lower values are very permissive; raise to 0.5+ to be strict.
--verbose
    Print the top snippet for every query result.
--query TEXT
    Add an extra query to the standard suite.  May be repeated.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_CHROMA_PATH = "./chroma_db"
_DEFAULT_CHROMA_COLLECTION = "bamboo_docs"
_DEFAULT_TOP_K = 3
_DEFAULT_MIN_SCORE = 0.3
_SNIPPET_WIDTH = 100

# ---------------------------------------------------------------------------
# Canonical query suite
# Each tuple is (query_text, description_of_what_should_be_found).
# Add entries here whenever new mandatory topics are ingested into the corpus.
# ---------------------------------------------------------------------------
_STANDARD_QUERIES: list[tuple[str, str]] = [
    ("PanDA workload management system",
     "Overview of PanDA — must always be present"),
    ("looping job algorithm",
     "Looping job detection — a core PanDA concept"),
    ("pilot job submission",
     "Pilot framework — fundamental to PanDA operation"),
    ("task retry failed jobs",
     "Task retry mechanism — part of PanDA workflow management"),
    ("Harvester worker agent",
     "Harvester — the PanDA edge-service component"),
    ("ATLAS experiment distributed computing",
     "ATLAS/PanDA integration context"),
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result for a single query against one backend.

    Attributes:
        query: The search query string.
        backend: Either ``"vector"`` or ``"bm25"``.
        hits: Number of results returned above the minimum score threshold.
        top_score: Highest similarity score among returned results, or 0.0.
        top_snippet: First 200 chars of the top result document, or empty string.
        error: Error message if the query failed, otherwise empty string.
        returned: Results the backend returned before the score threshold was
            applied.  Reported separately because the two zeros mean opposite
            things: nothing retrieved is an empty or unreachable corpus, while
            results retrieved and all filtered out is a scoring problem.
        top_distance: Smallest raw distance the vector backend returned, or
            ``None``.  The one number needed to interpret a vector failure,
            and the one the probe used to discard.
    """

    query: str
    backend: str
    hits: int = 0
    top_score: float = 0.0
    top_snippet: str = ""
    error: str = ""
    returned: int = 0
    top_distance: float | None = None

    @property
    def passed(self) -> bool:
        """Return True when the query returned at least one qualifying hit."""
        return not self.error and self.hits > 0


@dataclass
class SuiteResult:
    """Aggregated results across all queries and backends.

    Attributes:
        results: Individual QueryResult objects.
        collection_count: Total documents in the collection, or -1 on error.
        collection_name: Name of the collection that was queried.
    """

    results: list[QueryResult] = field(default_factory=list)
    collection_count: int = -1
    collection_name: str = ""

    @property
    def passed(self) -> bool:
        """Return True when every individual query passed."""
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        """Return the number of passing queries."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Return the number of failing queries."""
        return sum(1 for r in self.results if not r.passed)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _resolve_physical_name(chroma_path: str, collection_name: str) -> str:
    """Resolve a logical collection name to its live physical slot name.

    Reads the routing sidecar written by bamboo-mcp-services.  Falls back to
    the logical name when the sidecar is absent or the bamboo package is not
    available.

    Args:
        chroma_path: Path to the ChromaDB persistent directory.
        collection_name: Logical collection name (e.g. ``"atlas_docs"``).

    Returns:
        Physical collection name (e.g. ``"atlas_docs__a"``), or
        *collection_name* unchanged if no routing sidecar is found.
    """
    try:
        from bamboo.tools._chroma_routing import resolve_collection  # noqa: PLC0415
        return resolve_collection(chroma_path, collection_name)
    except ImportError:
        return collection_name


def _open_collection(
    chroma_path: str,
    collection_name: str,
) -> tuple[Any, Any]:
    """Open the ChromaDB persistent client and return the target collection.

    Args:
        chroma_path: Path to the ChromaDB persistent directory.
        collection_name: Name of the collection to open.

    Returns:
        Tuple of ``(client, collection)``.

    Raises:
        SystemExit: On missing dependency, missing path, or connection error.
    """
    try:
        from bamboo.tools._sqlite_compat import ensure_sqlite_compat  # noqa: PLC0415
        if not ensure_sqlite_compat():
            print(
                "error: system SQLite is too old for ChromaDB.  "
                "Install pysqlite3-binary and retry.",
                file=sys.stderr,
            )
            sys.exit(2)
    except ImportError:
        pass  # Running outside the bamboo package — skip compat check.

    try:
        import chromadb  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        print(
            "error: chromadb is not installed.  Run: pip install chromadb",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.exists(chroma_path):
        print(
            f"error: ChromaDB path not found: '{chroma_path}'\n"
            "       Set --path or BAMBOO_CHROMA_PATH to the correct directory.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        client = chromadb.PersistentClient(path=chroma_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to open ChromaDB at '{chroma_path}': {exc}", file=sys.stderr)
        sys.exit(2)

    # Resolve the logical collection name to the current live physical slot via
    # the routing sidecar written by bamboo-mcp-services document-monitor-agent.
    physical_name = _resolve_physical_name(chroma_path, collection_name)

    if physical_name != collection_name:
        print(
            f"  Resolved slot  : {physical_name}  "
            f"(via collection_routing.json)"
        )

    try:
        collection = client.get_collection(name=physical_name)
    except Exception as exc:  # noqa: BLE001
        _report_collection_not_found(client, physical_name, exc)

    return client, collection


def _report_collection_not_found(
    client: Any, physical_name: str, exc: Exception
) -> None:
    """Print a diagnostic message when a collection cannot be opened and exit.

    Args:
        client: Open ChromaDB PersistentClient (used to list available names).
        physical_name: The physical collection name that was not found.
        exc: The exception raised by ``client.get_collection``.
    """
    print(f"error: collection '{physical_name}' not found: {exc}", file=sys.stderr)
    try:
        available = [c.name for c in client.list_collections()]
    except Exception:  # noqa: BLE001
        available = []
    if available:
        print("\nAvailable collections in this store:", file=sys.stderr)
        for name in sorted(available):
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nFix: check collection_routing.json in your ChromaDB directory, "
            "or set BAMBOO_CHROMA_COLLECTION to the logical name used during ingestion.",
            file=sys.stderr,
        )
    else:
        print(
            "No collections found — the store may be empty or the path may be wrong.\n"
            "Run inspect_chroma.py for a full diagnosis.",
            file=sys.stderr,
        )
    sys.exit(2)


def _run_vector_query(
    collection: Any,
    query: str,
    top_k: int,
    min_score: float,
) -> QueryResult:
    """Run a vector (embedding) query against the ChromaDB collection.

    Args:
        collection: Open ChromaDB Collection object.
        query: Natural-language query string.
        top_k: Maximum number of results to request.
        min_score: Minimum cosine similarity score for a result to count as a hit.

    Returns:
        QueryResult with hit count, top score, and top snippet.
    """
    try:
        raw = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        return QueryResult(query=query, backend="vector", error=str(exc))

    documents: list[str] = (raw.get("documents") or [[]])[0]
    distances: list[float] = (raw.get("distances") or [[]])[0]

    hits = 0
    top_score = 0.0
    top_snippet = ""
    top_distance: float | None = None

    for doc, dist in zip(documents, distances):
        # score = 1 - distance is only meaningful for a cosine-space
        # collection. ChromaDB's default space is squared L2, where distances
        # are unbounded and routinely exceed 1.0, so every score clamps to 0.0
        # and every query reports zero hits with "score=n/a" — a total failure
        # indistinguishable from an empty corpus. Keeping the raw distance is
        # what tells those apart.
        score = max(0.0, 1.0 - dist)
        if score >= min_score:
            hits += 1
        if top_distance is None or dist < top_distance:
            top_distance = dist
            top_snippet = doc[:200]
        if score > top_score:
            top_score = score

    return QueryResult(
        query=query,
        backend="vector",
        hits=hits,
        top_score=top_score,
        top_snippet=top_snippet,
        returned=len(documents),
        top_distance=top_distance,
    )


def _run_bm25_query(
    collection: Any,
    query: str,
    top_k: int,
) -> QueryResult:
    """Run a BM25 keyword query using ChromaDB's ``where_document`` filter.

    ChromaDB does not have a native BM25 backend; this simulates the keyword
    search by splitting the query into tokens and filtering for documents that
    contain at least one significant token.  The BM25 tool in production uses
    a similar approach; this is sufficient as a corpus health check.

    Args:
        collection: Open ChromaDB Collection object.
        query: Keyword query string.
        top_k: Maximum number of results to retrieve.

    Returns:
        QueryResult with hit count and top snippet.
    """
    # Use the most distinctive word in the query for the contains-filter.
    # Stop-words and short tokens are skipped.
    _STOP = {"the", "a", "an", "of", "for", "to", "in", "is", "are", "and",
             "or", "how", "does", "what", "with", "using", "by", "at", "on"}
    tokens = [
        t.strip(".,?!") for t in query.lower().split()
        if len(t) > 3 and t not in _STOP
    ]
    if not tokens:
        tokens = query.split()[:1]

    # Try tokens from most to least distinctive until we get results.
    for token in sorted(tokens, key=len, reverse=True):
        try:
            raw = collection.get(
                where_document={"$contains": token},
                limit=top_k,
                include=["documents"],
            )
        except Exception as exc:  # noqa: BLE001
            return QueryResult(query=query, backend="bm25", error=str(exc))

        docs: list[str] = raw.get("documents") or []
        if docs:
            return QueryResult(
                query=query,
                backend="bm25",
                hits=len(docs),
                top_score=1.0,
                top_snippet=docs[0][:200],
            )

    return QueryResult(query=query, backend="bm25", hits=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _status(result: QueryResult) -> str:
    """Return a short pass/fail status string for a QueryResult.

    Args:
        result: The query result to format.

    Returns:
        ``"PASS"`` or ``"FAIL"`` with leading spacing for alignment.
    """
    return "PASS" if result.passed else "FAIL"


def _print_suite_report(
    suite: SuiteResult,
    queries_with_desc: list[tuple[str, str]],
    verbose: bool,
) -> None:
    """Print a human-readable report for the full suite.

    Args:
        suite: Aggregated SuiteResult.
        queries_with_desc: List of (query, description) pairs in suite order.
        verbose: When True, print the top snippet for each result.
    """
    col_name = suite.collection_name
    doc_count = suite.collection_count

    print(f"\nCorpus : {col_name}  ({doc_count} documents)")
    print(f"Checks : {suite.n_passed} passed, {suite.n_failed} failed\n")

    desc_map = {q: d for q, d in queries_with_desc}

    # Group results by query for compact display.
    by_query: dict[str, list[QueryResult]] = {}
    for r in suite.results:
        by_query.setdefault(r.query, []).append(r)

    for query, results in by_query.items():
        desc = desc_map.get(query, "")
        any_fail = any(not r.passed for r in results)
        marker = "✗" if any_fail else "✓"
        print(f"  {marker}  {query!r}")
        if desc:
            print(f"     ({desc})")
        for r in results:
            score_str = f"score={r.top_score:.2f}" if r.top_score else "score=n/a"
            hits_str = f"hits={r.hits}"
            status = _status(r)
            err_str = f"  error: {r.error}" if r.error else ""
            extra = ""
            if r.backend == "vector" and not r.error:
                dist_str = (
                    f"{r.top_distance:.4f}" if r.top_distance is not None else "n/a"
                )
                extra = f"  returned={r.returned}  best_distance={dist_str}"
            print(
                f"     [{r.backend:6}]  {status}  {hits_str}  {score_str}{extra}{err_str}"
            )
            if verbose and r.top_snippet:
                wrapped = textwrap.fill(
                    r.top_snippet, width=_SNIPPET_WIDTH,
                    initial_indent="             ",
                    subsequent_indent="             ",
                )
                print(wrapped)
        print()

    if suite.passed:
        print("Result: ALL CHECKS PASSED — corpus looks healthy.")
    else:
        print("Result: CHECKS FAILED — see above for details.")
        print()
        print("Common causes:")
        print("  • Documents have not been ingested yet  →  run the ingestion script")
        print("  • Wrong BAMBOO_CHROMA_PATH or BAMBOO_CHROMA_COLLECTION  →  check env vars")
        print("  • Corpus was cleared or rebuilt without re-ingesting  →  re-ingest")
        print("  • run inspect_chroma.py for a detailed view of the store")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Smoke-test the Bamboo RAG corpus (vector + BM25).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        default=os.getenv("BAMBOO_CHROMA_PATH", _DEFAULT_CHROMA_PATH),
        help=(
            f"ChromaDB directory "
            f"(default: $BAMBOO_CHROMA_PATH or '{_DEFAULT_CHROMA_PATH}')"
        ),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("BAMBOO_CHROMA_COLLECTION", _DEFAULT_CHROMA_COLLECTION),
        help=(
            f"Collection name "
            f"(default: $BAMBOO_CHROMA_COLLECTION or '{_DEFAULT_CHROMA_COLLECTION}')"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        metavar="N",
        help=f"Results per query (default: {_DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=_DEFAULT_MIN_SCORE,
        metavar="FLOAT",
        help=f"Minimum similarity score for a hit (default: {_DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the top snippet for each result",
    )
    parser.add_argument(
        "--query", "-q",
        action="append",
        default=[],
        metavar="TEXT",
        dest="extra_queries",
        help="Add an extra query to the standard suite (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the RAG smoke-test suite and exit with an appropriate code.

    Raises:
        SystemExit: Always — exit code 0 on success, 1 on test failure,
            2 on configuration/dependency error.
    """
    args = _parse_args()

    chroma_path: str = args.path
    collection_name: str = args.collection
    top_k: int = max(1, args.top_k)
    min_score: float = max(0.0, min(1.0, args.min_score))
    verbose: bool = args.verbose

    # Build the full query suite.
    queries_with_desc: list[tuple[str, str]] = list(_STANDARD_QUERIES)
    for extra in args.extra_queries:
        queries_with_desc.append((extra, "user-supplied query"))

    print("Bamboo RAG smoke-test")
    print(f"  ChromaDB path : {os.path.abspath(chroma_path)}")
    print(f"  Collection    : {collection_name}")
    print(f"  Queries       : {len(queries_with_desc)}")
    print(f"  top_k={top_k}  min_score={min_score}")

    _client, collection = _open_collection(chroma_path, collection_name)

    # _open_collection printed the resolved slot name if it differed; use the
    # actual collection name for the report header.
    resolved_name = collection.name

    try:
        doc_count = collection.count()
    except Exception:  # noqa: BLE001
        doc_count = -1

    if doc_count == 0:
        print(
            "\nerror: collection is empty — ingest documents before running this check.",
            file=sys.stderr,
        )
        sys.exit(1)

    suite = SuiteResult(collection_count=doc_count, collection_name=resolved_name)

    for query, _desc in queries_with_desc:
        suite.results.append(
            _run_vector_query(collection, query, top_k, min_score)
        )
        suite.results.append(
            _run_bm25_query(collection, query, top_k)
        )

    _print_suite_report(suite, queries_with_desc, verbose)

    sys.exit(0 if suite.passed else 1)


if __name__ == "__main__":
    main()
