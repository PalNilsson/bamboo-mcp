r"""Command-line front end for the job-analysis composer.

Everything here is presentation and process concerns: argument parsing,
endpoint resolution, output rendering, exit codes.  The analysis itself is
:mod:`interfaces.agent.job_agent.composer`, which knows nothing about
argparse, stdout or the MCP client class — so this module is the only place
that imports a transport.

Usage::

    python -m interfaces.agent.job_agent --panda-id 6799893074 \\
        --host aipanda033.cern.ch --port 8000

    # several jobs over one session
    python -m interfaces.agent.job_agent --panda-id 679989307 --panda-id 679989308

    # evidence only, no LLM call, one JSON object per line
    python -m interfaces.agent.job_agent --panda-id 6799893074 \\
        --no-synthesis --format jsonl

    # job IDs piped in
    cut -f1 failed_jobs.tsv | python -m interfaces.agent.job_agent --format jsonl

Environment variables
---------------------
``BAMBOO_MCP_HTTP_URL``  Default endpoint, used when neither ``--url`` nor
                         ``--host``/``--port`` is given.
``BAMBOO_MCP_TOKEN``     Bearer token for an authenticated endpoint.

Exit codes
----------
0  Every job was analysed.
1  Could not connect to the server, or the arguments were unusable.
2  At least one job failed to analyse, or a primitive is not registered.
3  No job failed, but at least one had no log content to read.

An error outranks a missing log: exit 2 means something went wrong, exit 3
means everything worked and one of the jobs simply had nothing to show.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Iterator
from typing import Any, Callable

from interfaces.agent.job_agent.composer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    OUTCOME_ERROR,
    OUTCOME_NO_LOGS,
    JobAgentError,
    JobAnalysisResult,
    analyse_jobs,
    missing_primitives,
)

logger = logging.getLogger(__name__)

EXIT_OK: int = 0
EXIT_CONNECT: int = 1
EXIT_ERROR: int = 2
EXIT_NO_LOGS: int = 3

DEFAULT_HOST: str = "localhost"
DEFAULT_PORT: int = 8000
DEFAULT_HTTP_URL: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/mcp"

FORMATS: tuple[str, ...] = ("text", "markdown", "json", "jsonl")

_RULE: str = "═" * 62


# ---------------------------------------------------------------------------
# Endpoint and job IDs
# ---------------------------------------------------------------------------

def resolve_url(url: str | None, host: str | None, port: int | None) -> str:
    """Resolve the MCP endpoint from the three ways of naming it.

    ``--url`` wins because it is the only form that can express TLS or a
    non-standard path; ``--host``/``--port`` is the ergonomic form; the
    environment is the fallback so an operator can set the endpoint once.

    Args:
        url: Full endpoint URL, or ``None``.
        host: Host name, or ``None``.
        port: Port number, or ``None``.

    Returns:
        The endpoint URL to connect to.
    """
    if url:
        return url
    if host is None and port is None:
        return os.getenv("BAMBOO_MCP_HTTP_URL", DEFAULT_HTTP_URL)
    return f"http://{host or DEFAULT_HOST}:{port if port is not None else DEFAULT_PORT}/mcp"


def read_job_ids(raw: list[str] | None, stdin_text: str | None) -> list[int]:
    """Parse the job IDs to analyse, preserving order and dropping repeats.

    Args:
        raw: Values collected from repeated ``--panda-id`` flags, or ``None``.
        stdin_text: Piped input to fall back on, or ``None`` when stdin is a
            terminal and must not be read.

    Returns:
        The job IDs, de-duplicated with first-occurrence order kept — a
        duplicate would otherwise cost a second full analysis for an
        identical answer.

    Raises:
        ValueError: When a value is not an integer, naming the offender.
    """
    values: list[str] = list(raw or [])
    if not values and stdin_text:
        values = stdin_text.split()

    seen: set[int] = set()
    job_ids: list[int] = []
    for value in values:
        token = value.strip().lstrip("#")
        if not token:
            continue
        try:
            job_id = int(token)
        except ValueError as exc:
            raise ValueError(f"not a PanDA job ID: {value!r}") from exc
        if job_id not in seen:
            seen.add(job_id)
            job_ids.append(job_id)
    return job_ids


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def make_progress_printer(quiet: bool) -> "Callable[[str], None] | None":
    """Return a callback writing one overwriting status line to stderr.

    Status goes to stderr so that ``--format jsonl`` stays pipeable while the
    operator still sees which file is being downloaded.

    Args:
        quiet: When ``True``, returns ``None`` and nothing is printed.

    Returns:
        The callback, or ``None``.
    """
    if quiet:
        return None

    state: dict[str, int] = {"last_len": 0}

    def _print(message: str) -> None:
        """Write one status line, erasing the previous one.

        Args:
            message: Status text from the composer.
        """
        cols = min(shutil.get_terminal_size(fallback=(120, 24)).columns - 2, 120)
        display = message[:cols]
        sys.stderr.write("\r" + display.ljust(state["last_len"]))
        sys.stderr.flush()
        state["last_len"] = len(display)

    return _print


@contextlib.contextmanager
def quiet_connect_logging(log_level: str) -> "Iterator[None]":
    """Suppress the MCP client's connect-rollback traceback at the default level.

    A failed ``connect()`` makes ``mcp_client._aclose_quietly`` log the
    rollback at WARNING with the full anyio exception group attached — forty
    lines of traceback whose only content is "connection refused", which this
    CLI then reports in one line anyway.  Mistyping a port is the most common
    way to reach it, so the default level hides it.

    Scoped to the connect call and conditional on the level, so anything the
    client has to say during the analysis itself is untouched and
    ``--log-level INFO`` brings the traceback back.

    Args:
        log_level: The ``--log-level`` value.  Only ``WARNING``, the default,
            triggers the suppression; an explicitly chosen level is honoured.

    Yields:
        None.
    """
    logger_name = "interfaces.shared.mcp_client"
    client_logger = logging.getLogger(logger_name)
    if log_level != "WARNING":
        yield
        return

    previous = client_logger.level
    client_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        client_logger.setLevel(previous)


def _clear_progress(quiet: bool) -> None:
    """Drop to a clean line after the last progress update.

    Args:
        quiet: When ``True``, nothing was printed and nothing is cleared.
    """
    if not quiet:
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _field_lines(result: JobAnalysisResult) -> list[str]:
    """Render the fact block shown above the answer.

    Args:
        result: A completed analysis.

    Returns:
        Aligned ``label : value`` lines, omitting fields the job has no value
        for.
    """
    meta = result.metadata
    code = meta.get("piloterrorcode")
    diag = str(meta.get("piloterrordiag") or "")
    attempt = meta.get("attemptnr")
    maximum = meta.get("maxattempt")

    pairs: list[tuple[str, str]] = []
    if meta.get("computingsite"):
        pairs.append(("Site", str(meta["computingsite"])))
    if meta.get("jobstatus"):
        status = str(meta["jobstatus"])
        if meta.get("jobsubstatus"):
            status = f"{status} / {meta['jobsubstatus']}"
        pairs.append(("Status", status))
    if code or diag:
        pairs.append(("Pilot error", f"{code or 0} — {diag or '(no diagnosis text)'}"))
    if meta.get("atlasrelease"):
        pairs.append(("Release", str(meta["atlasrelease"])))
    if meta.get("jeditaskid"):
        task = str(meta["jeditaskid"])
        if attempt:
            task = f"{task}  (attempt {attempt}{f'/{maximum}' if maximum else ''})"
        pairs.append(("Task", task))
    if result.pilot_version:
        pairs.append(("Pilot version", result.pilot_version))
    if result.strategy:
        pairs.append(("Strategy", result.strategy))
    pairs.append(("Files read", ", ".join(result.fetch_order) or "none"))
    if result.monitor_url:
        pairs.append(("Monitor", result.monitor_url))

    width = max((len(label) for label, _ in pairs), default=0)
    return [f"{label.ljust(width)} : {value}" for label, value in pairs]


def format_text(result: JobAnalysisResult, *, verbose: bool) -> str:
    """Render one result as a human-readable block.

    Args:
        result: A completed analysis.
        verbose: Also show accumulated notes, the per-file URLs and the raw
            log excerpt.

    Returns:
        The rendered block, without a trailing newline.
    """
    if result.outcome == OUTCOME_ERROR:
        return "\n".join([
            _RULE,
            f"Job {result.job_id} — ANALYSIS FAILED",
            _RULE,
            result.error or "unknown error",
        ])

    lines: list[str] = [
        _RULE,
        f"Job {result.job_id} — {result.failure_type}"
        + ("  (no log content)" if result.outcome == OUTCOME_NO_LOGS else ""),
        _RULE,
    ]
    lines.extend(_field_lines(result))

    if verbose:
        for entry in result.fetched:
            url = str(entry.get("url") or "")
            if url:
                lines.append(f"  {entry.get('filename')}: {url}")
        for note in result.notes:
            lines.append(f"  note: {note}")
        excerpt = str((result.verdict.get("context") or {}).get("excerpt") or "")
        if excerpt:
            lines.extend(["", "--- log excerpt ---", excerpt, "--- end excerpt ---"])

    if result.answer_markdown:
        lines.extend(["", result.answer_markdown])

    lines.append("")
    lines.append(
        f"(tools={result.tool_calls}, llm={result.llm_calls}, "
        f"{result.elapsed_s:.2f}s)"
    )
    return "\n".join(lines)


def format_markdown(result: JobAnalysisResult) -> str:
    """Render one result as the prose answer alone.

    The form to paste into a ticket or hand back to a caller.  Falls back to
    a one-line summary when synthesis was off or failed, so the output is
    never empty.

    Args:
        result: A completed analysis.

    Returns:
        Markdown text.
    """
    if result.outcome == OUTCOME_ERROR:
        return f"**Job {result.job_id}** — analysis failed: {result.error}"
    if result.answer_markdown:
        header = f"## Job {result.job_id} — {result.failure_type}"
        if result.monitor_url:
            header += f"\n\n[BigPanDA]({result.monitor_url})"
        return f"{header}\n\n{result.answer_markdown}"
    return (
        f"**Job {result.job_id}** — classified as `{result.failure_type}`; "
        f"files read: {', '.join(result.fetch_order) or 'none'}."
    )


def render(results: list[JobAnalysisResult], fmt: str, *, verbose: bool) -> str:
    """Render every result in the requested format.

    ``json`` emits a bare object for a single job and an array for several,
    which is what a caller analysing one job expects; ``jsonl`` is always one
    object per line and is the predictable shape for a pipeline.

    Args:
        results: Completed analyses, in job order.
        fmt: One of :data:`FORMATS`.
        verbose: Passed through to the text renderer.

    Returns:
        The rendered output, without a trailing newline.
    """
    if fmt == "jsonl":
        return "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in results)
    if fmt == "json":
        payload: Any = (
            results[0].to_dict() if len(results) == 1 else [r.to_dict() for r in results]
        )
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if fmt == "markdown":
        return "\n\n---\n\n".join(format_markdown(r) for r in results)
    return "\n\n".join(format_text(r, verbose=verbose) for r in results)


def exit_code(results: list[JobAnalysisResult]) -> int:
    """Reduce per-job outcomes to one process exit code.

    An error outranks a missing log: a run where one job broke and another had
    no logs reports the breakage, because that is the one needing attention.

    Args:
        results: Completed analyses.

    Returns:
        :data:`EXIT_ERROR`, :data:`EXIT_NO_LOGS` or :data:`EXIT_OK`.
    """
    if any(r.outcome == OUTCOME_ERROR for r in results):
        return EXIT_ERROR
    if any(r.outcome == OUTCOME_NO_LOGS for r in results):
        return EXIT_NO_LOGS
    return EXIT_OK


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace, job_ids: list[int]) -> int:
    """Connect once, analyse every job, render and return the exit code.

    One ``asyncio.run`` and one session for the whole batch.  That is not
    only cheaper than a session per job — anyio binds a cancel scope to the
    task that opened it, so ``connect()`` and ``aclose()`` must run in the
    same task or the event-loop thread wedges on shutdown.

    ``interfaces.shared.mcp_client`` is imported here rather than at module
    scope so that the parser, the renderers and the exit-code rule stay
    importable — and testable — without the MCP SDK and its ``httpx``
    dependency installed.  It is the same deferral the primitives use for
    ``bamboo.tools.base``.

    Args:
        args: Parsed arguments.
        job_ids: Jobs to analyse, in order.

    Returns:
        Process exit code.
    """
    try:
        from interfaces.shared.mcp_client import (  # noqa: PLC0415 - see docstring
            MCPAsyncClient,
            MCPServerConfig,
        )
    except ImportError as exc:
        # Deferring the import is what makes this reachable at all; without
        # the handler it would surface as a traceback from inside asyncio.run.
        print(
            f"[ERROR] The MCP client is unavailable: {exc}\n"
            f"        Install the runtime dependencies with:\n"
            f"            pip install -r requirements.txt",
            file=sys.stderr,
        )
        return EXIT_CONNECT

    headers: dict[str, str] | None = None
    if args.token:
        headers = {"Authorization": f"Bearer {args.token}"}

    cfg = MCPServerConfig(
        transport="http",
        http_url=resolve_url(args.url, args.host, args.port),
        http_headers=headers,
    )

    client = MCPAsyncClient(cfg)
    try:
        with quiet_connect_logging(args.log_level):
            await client.connect()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Could not connect to {cfg.http_url}: {exc}", file=sys.stderr)
        return EXIT_CONNECT

    try:
        missing = await missing_primitives(client)
        if missing and not args.quiet:
            print(
                f"[warn] {cfg.http_url} does not advertise {', '.join(missing)} — "
                f"continuing, since BAMBOO_TOOL_PROFILE gates advertising, not "
                f"dispatch. Set BAMBOO_TOOL_PROFILE=both on the server to list them.",
                file=sys.stderr,
            )

        results = await analyse_jobs(
            client,
            job_ids,
            timeout=args.timeout,
            synthesise=not args.no_synthesis,
            with_listing=args.with_listing,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            progress=make_progress_printer(args.quiet),
        )
    except JobAgentError as exc:
        _clear_progress(args.quiet)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        await client.aclose()

    _clear_progress(args.quiet)
    print(render(results, args.format, verbose=args.verbose))
    return exit_code(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="python -m interfaces.agent.job_agent",
        description=(
            "Analyse a failed PanDA job by composing Bamboo's atlas.log.* "
            "primitives against a running Bamboo MCP server.\n\n"
            "Deterministic: every decision on the diagnosis path is a "
            "server-side rule. The only LLM call is the final synthesis, and "
            "--no-synthesis removes that one too."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--panda-id",
        action="append",
        metavar="ID",
        help=(
            "PanDA job ID to analyse. Repeat for several jobs, which are then "
            "analysed in order over one session. Omit to read whitespace- or "
            "newline-separated IDs from stdin."
        ),
    )

    p.add_argument(
        "--host",
        default=None,
        help=f"MCP server host (default: {DEFAULT_HOST}).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"MCP server port (default: {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help=(
            "Full MCP endpoint, overriding --host/--port. Use this for TLS or "
            "a non-standard path. Defaults to BAMBOO_MCP_HTTP_URL when no "
            "host or port is given."
        ),
    )
    p.add_argument(
        "--token",
        default=os.getenv("BAMBOO_MCP_TOKEN", ""),
        metavar="TOKEN",
        help="Bearer token for an authenticated endpoint (or BAMBOO_MCP_TOKEN).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="HTTP timeout passed to each primitive (server default: 60).",
    )

    p.add_argument(
        "--format",
        choices=list(FORMATS),
        default="text",
        help=(
            "Output format (default: text). 'json' emits an object for one job "
            "and an array for several; 'jsonl' always emits one object per line."
        ),
    )
    p.add_argument(
        "--no-synthesis",
        action="store_true",
        default=False,
        help=(
            "Skip the final LLM call and report evidence only. Zero LLM calls, "
            "fully deterministic output."
        ),
    )
    p.add_argument(
        "--with-listing",
        action="store_true",
        default=False,
        help=(
            "Also call atlas.log.list_files. Off the diagnosis path; answers "
            "'what did this job produce'."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Token budget for the synthesis call (default: {DEFAULT_MAX_TOKENS}).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        metavar="F",
        help=f"Sampling temperature for synthesis (default: {DEFAULT_TEMPERATURE}).",
    )

    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Include notes, per-file URLs and the raw log excerpt in text output.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress and warnings on stderr.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the analysis.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code; see the module docstring for the meanings.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    stdin_text: str | None = None
    if not args.panda_id and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()

    try:
        job_ids = read_job_ids(args.panda_id, stdin_text)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_CONNECT

    if not job_ids:
        parser.print_help()
        print(
            "\nError: give at least one --panda-id, or pipe job IDs via stdin.",
            file=sys.stderr,
        )
        return EXIT_CONNECT

    return asyncio.run(_run(args, job_ids))


if __name__ == "__main__":
    raise SystemExit(main())
