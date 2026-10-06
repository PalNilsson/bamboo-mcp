#!/usr/bin/env python3
"""Measure tool-retrieval quality against the labelled corpus.

Run with no arguments to record the baseline — the null retriever, which
returns the whole catalog and so reproduces the planner's current behaviour
exactly.  Its numbers are the reference every candidate retriever is compared
against: recall 1.000 and payload 100%, by construction.

Typical use::

    python scripts/eval_tool_retrieval.py                 # baseline
    python scripts/eval_tool_retrieval.py --k 8 --k 10 --k 12
    python scripts/eval_tool_retrieval.py --json > baseline.json

Exits non-zero when ``--min-recall`` is given and not met, so the same command
serves as a gate once a real retriever exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "core", REPO_ROOT / "packages" / "askpanda_atlas"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bamboo.evaluation.tool_retrieval import (  # noqa: E402
    NullRetriever,
    Report,
    ToolRetriever,
    evaluate,
    load_corpus,
)
from bamboo.tools.planner import (  # noqa: E402
    _collect_tool_catalog,
    routing_rules_for_plugin,
)

DEFAULT_CORPUS = REPO_ROOT / "tests" / "data" / "tool_selection_corpus.json"

#: Tools exempt from retrieval because the universal fallback route has no
#: graceful degradation without them.  Mirrors decision T-4; a retriever never
#: sees them and they count against the *k* budget.
PINNED_TOOLS = frozenset({"panda_doc_search", "panda_doc_bm25"})

RETRIEVERS: dict[str, Any] = {"null": NullRetriever}


def _build_retriever(name: str) -> ToolRetriever:
    """Instantiate a registered retriever by name.

    Args:
        name: Key in :data:`RETRIEVERS`.

    Returns:
        ToolRetriever: A ready retriever.

    Raises:
        SystemExit: If *name* is not registered, listing what is.
    """
    try:
        return RETRIEVERS[name]()
    except KeyError:
        raise SystemExit(
            f"unknown retriever {name!r}; available: {', '.join(sorted(RETRIEVERS))}"
        )


def _format_report(report: Report, show_failures: int) -> str:
    """Render a report as plain text.

    Args:
        report: The report to render.
        show_failures: Maximum failing cases to list; 0 lists none.

    Returns:
        str: The rendered report.
    """
    lines = [
        f"retriever={report.retriever}  k={report.k}  cases={len(report.results)}",
        f"  recall@k            {report.recall():.3f}",
        f"  recall@k (hard)     {report.recall(hard_only=True):.3f}",
        f"  guidance coverage   {report.guidance_coverage():.3f}",
        f"  payload fraction    {report.mean_payload_fraction():.3f}"
        f"  (full catalog = {report.full_payload_bytes:,} chars)",
        f"  over budget         {report.over_budget_cases()}/{len(report.results)} cases",
    ]
    failures = report.failures()
    if failures and show_failures:
        lines.append(f"  failures ({len(failures)}):")
        for result in failures[:show_failures]:
            flag = " [hard]" if result.case.hard else ""
            lines.append(
                f"    {result.case.case_id}{flag} missing="
                f"{sorted(result.missing)}  {result.case.question[:64]!r}"
            )
        if len(failures) > show_failures:
            lines.append(f"    ... and {len(failures) - show_failures} more")
    return "\n".join(lines)


def _report_to_dict(report: Report) -> dict[str, Any]:
    """Convert a report to a JSON-serialisable summary.

    Args:
        report: The report to convert.

    Returns:
        Dict[str, Any]: Aggregates plus the identifiers of failing cases.
    """
    return {
        "retriever": report.retriever,
        "k": report.k,
        "cases": len(report.results),
        "recall_at_k": round(report.recall(), 4),
        "recall_at_k_hard": round(report.recall(hard_only=True), 4),
        "guidance_coverage": round(report.guidance_coverage(), 4),
        "payload_fraction": round(report.mean_payload_fraction(), 4),
        "over_budget_cases": report.over_budget_cases(),
        "full_payload_chars": report.full_payload_bytes,
        "failures": [
            {
                "id": r.case.case_id,
                "question": r.case.question,
                "missing": sorted(r.missing),
                "hard": r.case.hard,
            }
            for r in report.failures()
        ],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--retriever",
        default="null",
        choices=sorted(RETRIEVERS),
        help="Retriever to evaluate (default: the no-retrieval baseline).",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        help="Tool budget, including pinned tools. Repeatable. Default: 10.",
    )
    parser.add_argument("--namespace", default="atlas")
    parser.add_argument("--plugin-id", default="atlas")
    parser.add_argument(
        "--no-pins",
        action="store_true",
        help="Do not pin the fallback documentation tools (measures their value).",
    )
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Exit non-zero if recall@k falls below this at any k.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and print a report.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        int: Process exit status.
    """
    args = _parse_args(argv)
    corpus = load_corpus(args.corpus)
    catalog = _collect_tool_catalog(namespaces=[args.namespace] if args.namespace else None)
    rules = routing_rules_for_plugin(args.plugin_id)
    pinned = frozenset() if args.no_pins else PINNED_TOOLS

    reports = [
        evaluate(
            _build_retriever(args.retriever),
            corpus,
            catalog,
            k=k,
            pinned=pinned,
            routing_rules=rules,
        )
        for k in (args.k or [10])
    ]

    if args.json:
        print(json.dumps({"reports": [_report_to_dict(r) for r in reports]}, indent=2))
    else:
        print(f"catalog: {len(catalog)} tools, namespace={args.namespace!r}")
        print(f"corpus:  {len(corpus.cases)} cases from {args.corpus}")
        print(f"pinned:  {sorted(pinned) or 'none'}")
        print()
        for report in reports:
            print(_format_report(report, args.show_failures))
            print()

    if args.min_recall is not None:
        worst = min(r.recall() for r in reports)
        if worst < args.min_recall:
            print(
                f"FAIL: recall@k {worst:.3f} below threshold {args.min_recall:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
