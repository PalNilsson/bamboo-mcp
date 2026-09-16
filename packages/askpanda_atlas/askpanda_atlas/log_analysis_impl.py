"""ATLAS PanDA job log analysis tool — canonical implementation.

Fetches job metadata and pilot log from BigPanDA, extracts a relevant
context window using known log message patterns, classifies the failure,
and returns structured evidence suitable for LLM summarisation.

This is experiment-specific logic and belongs in the plugin package, not
in bamboo core.  Core provides only a thin shim that re-exports this tool.

The only bamboo dependency (``bamboo.tools.base``) is imported lazily
inside ``PandaLogAnalysisTool.call()`` so that the pure diagnostic
functions (``classify_failure``, ``extract_log_excerpt``, etc.) remain
importable even when bamboo core is not installed.

Interface
---------
- ``panda_log_analysis_tool.get_definition()`` — MCP tool definition
- ``await panda_log_analysis_tool.call(arguments)`` — returns
  ``list[MCPContent]`` whose ``text`` field is a JSON-serialised dict
  with ``evidence`` and ``text`` keys.

Extraction strategy
-------------------
Excerpt extraction is *traceback-first*.  Before consulting any error-code
lookup table the log is scanned for a real Python traceback using the format
invariants in :mod:`askpanda_atlas._traceback_parse`; when one is found the
excerpt is built around it and the exception is also returned as structured
evidence (``exception_type``, ``exception_message``, ``deepest_pilot_frame``).

Only when no traceback exists does extraction fall back to the previous
behaviour: the ``_PILOT_CODE_PATTERNS`` anchor for the job's pilot error code,
then ``piloterrordiag`` as a literal regex, then the tail of the log.  Those
fallbacks are unreliable because ``piloterrordiag`` is written by a different
pilot code path than the log record, so the wordings frequently differ — pilot
error code 1310 reports ``"Exception caught during payload execution"`` in
metadata while the log record reads ``"execute payloads caught an exception
(cannot recover): timed out, Traceback ..."``.  When the anchor missed, the
excerpt used to degrade to the tail of the log, which for a failed job is
stage-out and log-archiving boilerplate that contains no diagnostic content.

Evidence keys
-------------
job_id, monitor_url, jobstatus, jobsubstatus, computingsite, cloud,
atlasrelease, jeditaskid, attemptnr, maxattempt, piloterrorcode,
piloterrordiag, exeerrorcode, exeerrordiag, taskbuffererrorcode,
taskbuffererrordiag, ddmerrorcode, ddmerrordiag, starttime, endtime,
duration, failure_type, log_url, log_excerpt, log_available,
traceback_available, exception_type, exception_message, exception_frames,
deepest_pilot_frame, traceback_count, pilot_version, code_analysis_offer_md,
core_dump_probe_state, core_dump_available, core_dump_candidates,
core_dump_total_bytes, core_dump_offer_md.

Core-dump probe
---------------
The job-log listing fetched for the log downloads is also read for core files
(``core.<pid>``), so a looping-job kill can be followed by a core-dump
analysis without a second request.  The probe distinguishes a usable core from
one truncated mid-write and from a looping-job kill that produced no core at
all, because those mean different things to the person reading the answer.
``core_dump_offer_md`` is emitted only for a looping-job kill with a usable
core; the evidence keys are populated for every failure type so that an
explicit request works regardless of how the job failed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from askpanda_atlas._fallback_http import get_base_url
from askpanda_atlas._traceback_parse import (
    ExceptionInfo,
    TracebackBlock,
    coerce_bool,
    coerce_int,
    coerce_optional_str,
    coerce_str,
    find_primary_exception,
    parse_pilot_version,
    parse_pilot_version_from_pilotid,
    truncate_traceback,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core-dump probe
# ---------------------------------------------------------------------------

# Whether a core-dump analysis tool is registered for this experiment, and so
# whether an analysis follow-up may be offered to the user.  The probe itself
# is experiment-neutral — it only reads the job's file listing — but the
# analysis tool reconstructs the job's ATLAS release container, which is
# ATLAS-specific.  Offering analysis where no tool is registered would produce
# an offer that cannot be accepted, so the ePIC mirror sets this to False (see
# tests/plugin_mirror_spec.py).  The evidence keys are still populated for
# both, since "does this job have a core dump" is a useful answer either way.
_CORE_DUMP_ANALYSIS_AVAILABLE: bool = True

# Pilot error code emitted when the pilot's looping-job detector kills a
# payload that has stopped writing output.  The pilot requests a core dump at
# that point, so this is the failure mode where analysis is worth offering
# proactively; an explicit request still works for any failure type.
_LOOPING_JOB_PILOT_CODE: int = 1150

# The pilot names core files core.<pid>.  Anchored on both ends so that
# unrelated names containing "core" (core_dump_config.txt, .corefile) cannot
# match.
_CORE_FILE_RE: re.Pattern[str] = re.compile(r"^core\.\d+$")

# Probe outcomes, in the order the probe considers them.
CORE_DUMP_PRESENT: str = "present"
CORE_DUMP_TRUNCATED: str = "truncated"
CORE_DUMP_TIMED_OUT: str = "timed_out"
CORE_DUMP_ABSENT: str = "absent"
CORE_DUMP_NOT_PROBED: str = "not_probed"

# ---------------------------------------------------------------------------
# Failure classification patterns
# ---------------------------------------------------------------------------

# Each entry: (category_name, [keywords to search in combined error text])
# Order matters — first match wins.
# stageout_timeout MUST appear before stagein_timeout: the piloterrordiag for
# stage-out timeouts (code 1152) begins with "File transfer timed out during
# stage-out", which also matches the stagein_timeout keyword "file transfer
# timed out".  The more specific stage-out check must win.
_FAILURE_PATTERNS: list[tuple[str, list[str]]] = [
    ("reassigned_by_jedi", ["reassigned by jedi", "toreassign"]),
    ("stageout_timeout", [
        "file transfer timed out during stage-out",
        "timed out during stage-out",
        "timeout during stage-out",
    ]),
    ("stagein_timeout", ["file transfer timed out", "timeout during stage-in", "cp_timeout"]),
    ("timeout", ["timeout", "timed out", "walltime", "cpu time exceeded", "tobekilled"]),
    ("segfault", ["segmentation fault", "sigsegv", "signal 11"]),
    ("disk_full", ["no space left", "disk quota", "disk full", "work directory.*too large"]),
    ("memory", ["out of memory", "oom killer", "memory limit", "job has exceeded the memory"]),
    ("network", ["connection refused", "network unreachable", "dns failure", "socket error"]),
    ("input_missing", ["no such file", "file not found", "input file missing"]),
    ("stagein_failed", ["failed to stage-in", "stage-in failed", "piloterrorcode.*1099"]),
    # Pilot infrastructure errors: UID not found in process table scan during CPU monitoring.
    # Must appear before payload_error — the WARNING log contains "exception" and would
    # otherwise be misclassified as a user payload failure.
    ("pilot_monitoring_error", ["getpwuid", "uid not found", "list_processes_and_threads"]),
    # Release / container setup failures detected in setup.stdout before payload runs.
    # Must appear before payload_error so the setup log excerpt wins over the empty
    # payload stdout that accompanies these jobs.
    ("setup_release_not_found", ["no matched release is found", "!!!error!!!"]),
    ("payload_error", ["athena", "traceback", "exception", "abort", "core dump"]),
    ("pilot_error", ["piloterrorcode"]),
]

# Failure categories produced only by _classify_from_exception (never by the
# substring table above).  Listed here for documentation and for consumers that
# need to enumerate the full category vocabulary.
#
# transform_download_timeout / transform_download_failed
#     The pilot failed while downloading the job's transformation script (e.g.
#     runGen) over HTTP, inside get_analysis_trf -> download_transform ->
#     download_file.  The payload never started, so any diag text mentioning
#     "payload execution" (pilot error 1310 does) is misleading.
# pilot_exception
#     An unrecognised exception raised inside pilot3 code.  Preferred over
#     payload_error, which wrongly implies the user's payload was at fault.
_EXCEPTION_ONLY_CATEGORIES: frozenset[str] = frozenset({
    "transform_download_timeout",
    "transform_download_failed",
    "pilot_exception",
})

# Map pilot error codes to a search string likely to appear near the failure
# in the log, used to anchor the context-window extraction.
#
# This table is now a *fallback*: extract_log_excerpt first looks for a real
# Python traceback (see _traceback_parse), which needs no per-code entry.  The
# table still helps for failures the pilot reports without raising an exception
# (stage-in timeouts, looping-job kills, memory limits), where there is no
# traceback to anchor on.
_PILOT_CODE_PATTERNS: dict[int, str] = {
    1099: "Failed to stage-in file",
    1104: r"work directory .* is too large",
    1150: "pilot has decided to kill looping job",
    1151: "File transfer timed out",
    1201: "caught signal",
    1235: "job has exceeded the memory limit",
    1305: "",          # payload failure — use tail of payload.stdout instead
    # 1310: exception raised while the pilot was preparing or running the
    # payload.  The metadata diag ("Exception caught during payload execution")
    # does not appear in the log; the log record reads "execute payloads caught
    # an exception".  Normally the traceback-first path handles this code, so
    # this anchor only matters for the rare 1310 job whose log lost its
    # traceback (truncated upload, killed pilot).
    1310: "caught an exception",
    1324: "Service not available",
    # 1354: UID not found when scanning the process table for CPU monitoring.
    # This is a pilot infrastructure error, not a user payload failure.
    1354: "getpwuid",
}

# Pilot error codes whose diagnostic content *follows* the anchor line
# (e.g. a WARNING header followed by a multi-line traceback).
# For these codes _extract_context_window_with_trailing is used instead of
# _extract_context_window so the traceback body is included in the excerpt.
_TRAILING_CONTEXT_CODES: frozenset[int] = frozenset({1354})

# Number of preceding lines to include when a pattern match is found
_CONTEXT_LINES: int = 40
# Number of lines to continue collecting *after* the match line for error
# codes whose relevant content follows the anchor (e.g. Python tracebacks
# that start with a WARNING line and then span several subsequent lines).
_TRAILING_LINES: int = 30
# For pilotlog.txt fallback: number of tail lines when no pattern matches
_PAYLOAD_TAIL_LINES: int = 300
# Maximum log excerpt length sent to the LLM (characters).
# Raised from 6000: tracebacks that pass through the CVMFS standard library are
# long (the runGen transform-download timeout traceback is ~4 kB on its own),
# and a 6000-char budget left too little room for surrounding pilot context.
_MAX_EXCERPT_CHARS: int = 8000
# Characters reserved for payload.stderr within the excerpt budget.
# Guarantees the stderr traceback is always included even when stdout is long.
_STDERR_RESERVED_CHARS: int = 2000
# Characters reserved for the primary traceback within the excerpt budget.
# The traceback is the highest-value content in the excerpt, so it is allocated
# first and the surrounding context gets the remainder — never the other way
# round.
_TRACEBACK_RESERVED_CHARS: int = 5000
# Lines of log context collected immediately after the traceback.  The pilot
# usually logs the resulting error code and state transition here.
_TRACEBACK_TRAILING_LINES: int = 10
# Character cap on that trailing context, so it cannot crowd out the preceding
# context that explains what the pilot was attempting.
_TRACEBACK_TRAILING_CHARS: int = 600
# Retained for backwards compatibility only.  The stderr reservation is now
# applied by the caller that actually appends payload.stderr
# (_fetch_logs_payload), which passes the reduced budget into
# extract_failure_context.  A direct extract_log_excerpt call has no stderr to
# append and therefore gets the full budget.
#
# Char-based (not line-based) slicing is still used for the no-traceback payload
# tail: it guarantees recency regardless of line length, whereas a line count on
# verbose logs pushes the ERROR lines out of the budget.
_STDOUT_CHAR_TAIL: int = _MAX_EXCERPT_CHARS - _STDERR_RESERVED_CHARS


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch_metadata(job_id: int, base_url: str, timeout: int) -> dict[str, Any] | None:
    """Fetch job metadata JSON from BigPanDA, using the in-process TTL cache.

    Results are cached for :data:`~askpanda_atlas._cache.METADATA_TTL`
    seconds (60 s) so follow-up questions within the same session do not
    trigger redundant HTTP requests.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON dict (with ``job``, ``files``, ``dsfiles`` keys) or
        ``None`` on failure.
    """
    from askpanda_atlas._cache import cached_fetch_jsonish  # type: ignore[import]

    url = f"{base_url}/job?pandaid={job_id}&json"
    status, _ctype, _text, payload = cached_fetch_jsonish(url, timeout)
    if status < 200 or status >= 300 or payload is None:
        logger.warning("Metadata fetch failed for job %d: HTTP %d", job_id, status)
        return None
    return payload


def _log_file_url(job_id: int, filename: str, base_url: str) -> str:
    """Build the filebrowser URL for one of a job's log files.

    The query string is load-bearing and must not be reordered or extended:
    the ``&json`` form is served unauthenticated, while the bare form
    redirects to the CERN SSO login page.  This was written out inline at
    five call sites, one of which (:func:`_fetch_log_text`) actually
    *downloads* while the other four only record a URL for the evidence
    bundle, so a divergence between them would have produced links that do
    not point at the file that was read.

    Args:
        job_id: PanDA job ID.
        filename: Log filename, relative to the job directory (e.g.
            ``setup.stdout``).
        base_url: BigPanDA base URL.

    Returns:
        Fully-qualified filebrowser URL for the named file.
    """
    return f"{base_url}/filebrowser/?pandaid={job_id}&json&filename={filename}"


def _fetch_log_text(job_id: int, filename: str, base_url: str, timeout: int) -> str | None:
    """Download a pilot or payload log file, using the in-process cache.

    Log files are immutable once written, so hits are cached for the
    lifetime of the process via :func:`~askpanda_atlas._cache.cached_fetch_log`
    (TTL = ``math.inf``).  A log that has been downloaded once is never
    re-fetched.

    Args:
        job_id: PanDA job ID.
        filename: Log filename to fetch (e.g. ``pilotlog.txt``).
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Full log text as a string, or ``None`` if the file is not found
        or the download fails.
    """
    from askpanda_atlas._cache import cached_fetch_log  # type: ignore[import]

    url = _log_file_url(job_id, filename, base_url)
    logger.info("Fetching log (cache-aware): %s", url)
    return cached_fetch_log(url, timeout)


# ---------------------------------------------------------------------------
# Setup log error detection
# ---------------------------------------------------------------------------

# Patterns that, when found in setup.stdout, indicate a fatal setup error that
# is the definitive root cause of the job failure.  When any of these match,
# payload.stdout and payload.stderr are not downloaded — they will be empty
# (the payload never ran) and attempting to fetch them wastes time and budget.
# Order does not matter; all are checked.
_SETUP_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"!!!error!!!", re.IGNORECASE),
    re.compile(r"no matched release is found", re.IGNORECASE),
    re.compile(r"asetup.*failed", re.IGNORECASE),
    re.compile(r"error.*release.*not.*found", re.IGNORECASE),
]


def _setup_log_has_error(setup_text: str) -> bool:
    """Return True if setup.stdout contains a recognisable fatal setup error.

    Checks ``setup_text`` against :data:`_SETUP_ERROR_PATTERNS`.  A match
    means the Athena/Apptainer environment setup failed before the payload
    could start, making payload.stdout and payload.stderr empty and
    irrelevant.

    Args:
        setup_text: Full content of the setup.stdout log file.

    Returns:
        ``True`` if any setup-error pattern matches; ``False`` otherwise.
    """
    for pattern in _SETUP_ERROR_PATTERNS:
        if pattern.search(setup_text):
            return True
    return False


def _listing_entries(payload: Any) -> list[Any]:
    """Extract the raw file-entry list from a filebrowser JSON payload.

    Args:
        payload: Parsed JSON body, either a bare list of entries or a dict
            wrapping one under ``files``/``data``/``results``.

    Returns:
        The raw entry list, or an empty list when no recognisable list is
        present.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Some BigPanDA versions wrap the list under a "files" key.
        for key in ("files", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _normalise_listing_entry(entry: Any) -> dict[str, Any] | None:
    """Normalise one filebrowser entry into a path-keyed record.

    The ``dirname`` field carries the path and ``name`` carries only the
    basename, so the relative path has to be reassembled.  Entries with no
    usable name are dropped.

    Note that the entry's own ``media_link`` is deliberately not carried
    forward: BigPanDA omits the separator between ``dirname`` and ``name``
    for nested files, producing links such as ``.../workDirin.txt``.  Media
    URLs are constructed from the job's log GUID instead.

    Args:
        entry: One element of the listing's ``files`` array.

    Returns:
        Dict with ``relative_path``, ``name``, ``dirname``, ``size_bytes``
        and ``modification`` keys, or ``None`` when the entry is unusable.
    """
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name") or entry.get("filename") or "")
    if not name:
        return None
    dirname = str(entry.get("dirname") or "").strip("/")
    try:
        size_bytes = int(entry.get("size") or entry.get("fsize") or 0)
    except (ValueError, TypeError):
        size_bytes = 0
    return {
        "relative_path": f"{dirname}/{name}" if dirname else name,
        "name": name,
        "dirname": dirname,
        "size_bytes": size_bytes,
        "modification": str(entry.get("modification") or ""),
    }


def _fetch_file_listing(
    job_id: int,
    base_url: str,
    timeout: int,
) -> list[dict[str, Any]] | None:
    """Fetch the recursive filebrowser listing for a job.

    Calls ``/filebrowser/?pandaid={job_id}&json`` (no ``filename=`` param).
    The ``&json`` form is served unauthenticated; the bare form redirects to
    the CERN SSO login page, so the query string must not be altered.

    The listing is fully recursive and each entry's path lives in ``dirname``
    rather than in ``name``, so entries are normalised to a single
    ``relative_path`` key by :func:`_normalise_listing_entry`.

    The listing's ``error`` field is advisory whenever the file array is
    populated.  A large job log makes BigPanDA report a warning such as
    ``"slow_downloading:The size of job log tarball is quite big (119.36MB)."``
    alongside a complete and correct listing, so treating a non-empty
    ``error`` as fatal would reject usable responses.  It is only fatal when
    no entries came back.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        List of normalised entry dicts, or ``None`` when the listing could
        not be fetched or came back empty with an error set.  Callers must
        treat ``None`` as "unknown", not as "no files".
    """
    from askpanda_atlas._cache import cached_fetch_jsonish  # type: ignore[import]

    url = f"{base_url}/filebrowser/?pandaid={job_id}&json"
    logger.info("Fetching file listing: %s", url)
    status, _ctype, _text, payload = cached_fetch_jsonish(url, timeout)
    if status < 200 or status >= 300 or payload is None:
        logger.warning("File listing fetch failed for job %d: HTTP %d", job_id, status)
        return None

    listing_error = ""
    if isinstance(payload, dict):
        listing_error = str(payload.get("error") or "")

    entries = _listing_entries(payload)
    normalised = [
        record for record in (_normalise_listing_entry(entry) for entry in entries)
        if record is not None
    ]

    if not normalised:
        if listing_error:
            logger.warning(
                "File listing for job %d returned no files and reported: %s",
                job_id, listing_error,
            )
            return None
        logger.debug("File listing for job %d is empty or unrecognised format", job_id)
        return []

    if listing_error:
        # Advisory only — the entries are usable.
        logger.info(
            "File listing for job %d reported a warning alongside %d entries: %s",
            job_id, len(normalised), listing_error,
        )
    return normalised


def _fetch_file_index(
    job_id: int,
    base_url: str,
    timeout: int,
) -> dict[str, int] | None:
    """Fetch the filebrowser listing for a job, returning file sizes by path.

    Thin wrapper over :func:`_fetch_file_listing` for callers that only need
    sizes, so they can skip zero-length files before attempting a download.

    The mapping is keyed on the **full relative path**, not on the basename.
    Keying on the basename alone silently collapsed distinct files: a single
    job tarball can contain ``output.root`` five times under five different
    ``dirname`` values, and ``setup.sh`` both at the job root and under
    ``workDir/usr/...``, in which case the last entry parsed won.  Top-level
    files are unaffected because their relative path *is* their basename;
    use :func:`_top_level_file_index` when a basename lookup is what is
    wanted.

    A ``None`` return means the listing could not be fetched; callers must
    treat this as "assume all files are non-empty" (fail-open) so that an
    unavailable listing does not suppress log downloads.

    BigPanDA's filebrowser JSON listing uses the following structure::

        {"error": "", "files": [
            {"name": "pilotlog.txt", "size": 123456, "dirname": ""},
            {"name": "in.txt", "size": 2359, "dirname": "/workDir"},
        ]}

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Dict mapping relative path to size in bytes, or ``None`` on failure.
    """
    listing = _fetch_file_listing(job_id, base_url, timeout)
    if listing is None:
        return None
    return {record["relative_path"]: int(record["size_bytes"]) for record in listing}


def _top_level_file_index(file_index: dict[str, int] | None) -> dict[str, int] | None:
    """Restrict a path-keyed size index to entries at the job-directory root.

    Exists so that basename lookups stay unambiguous.  Every helper that
    consults the index asks about a canonical top-level log
    (``setup.stdout``, ``payload.stdout``, ``payload.stderr``,
    ``pilotlog.txt``), and a nested file that happens to share one of those
    names must not be able to answer for it.

    Args:
        file_index: Path-keyed mapping from :func:`_fetch_file_index`, or
            ``None`` when the listing is unavailable.

    Returns:
        Mapping of basename to size for root-level entries only, or ``None``
        when *file_index* is ``None`` (propagating the fail-open signal).
    """
    if file_index is None:
        return None
    return {path: size for path, size in file_index.items() if "/" not in path}


def _file_is_nonempty(file_index: dict[str, int] | None, filename: str) -> bool:
    """Return True if *filename* should be downloaded based on the file index.

    Uses a fail-open policy: if *file_index* is ``None`` (the directory
    listing could not be fetched), the file is considered non-empty and
    the download proceeds.  This preserves existing behaviour when the
    index endpoint is unavailable.

    The lookup is an exact match on the index key, which is a relative path.
    Callers passing a bare filename should pass an index restricted by
    :func:`_top_level_file_index` so a nested namesake cannot answer for a
    root-level file.

    Args:
        file_index: Mapping of ``{relative_path: size_bytes}`` from
            :func:`_fetch_file_index`, or ``None`` if the index is
            unavailable.
        filename: The log filename to check (e.g. ``"setup.stdout"``).

    Returns:
        ``True`` if the file should be downloaded; ``False`` if it is
        confirmed to have zero bytes and can safely be skipped.
    """
    if file_index is None:
        return True  # fail-open: index unavailable, assume non-empty
    size = file_index.get(filename)
    if size is None:
        # File not listed at all — may not exist yet; attempt download anyway
        # to get a definitive 404 rather than silently skipping.
        return True
    return size > 0


# ---------------------------------------------------------------------------
# Payload log noise stripping
# ---------------------------------------------------------------------------

# Matches PanDA pilot "=== ls in <dir> ===" directory listing sections.
# These appear in payload.stdout between the application error output and the
# "==== Result ====" footer, consuming budget without diagnostic value.
_LS_SECTION_RE: re.Pattern[str] = re.compile(
    r"\n=== ls in [^\n]+=+\n"
    r"(?:(?:total \d+|[dlrwxs\-]{10})[^\n]*\n)*",
    re.MULTILINE,
)


def _strip_payload_noise(text: str) -> str:
    """Remove PanDA pilot boilerplate sections from payload stdout text.

    Strips ``=== ls in <dir> ===`` directory listing blocks that the PanDA
    pilot appends between the application output and the result footer.  These
    sections are structurally identifiable (heading + lines starting with
    permission bits or ``total N``) and consume character budget without
    contributing diagnostic information.

    Args:
        text: Raw payload stdout text.

    Returns:
        Text with ls listing sections removed and excess blank lines collapsed.
    """
    text = _LS_SECTION_RE.sub("\n", text)
    # Collapse runs of 3+ blank lines left by the removed sections.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Context window extraction
# ---------------------------------------------------------------------------

def _extract_context_window(log_text: str, pattern: str, n_lines: int) -> str:
    """Extract lines from a log up to and including the first pattern match.

    Maintains a rolling buffer of ``n_lines`` and returns it when the
    compiled pattern is found, exactly as AskPanDA's
    ``extract_preceding_lines_streaming`` does.

    Args:
        log_text: Full log content as a string.
        pattern: Regular expression to search for.
        n_lines: Number of lines to include before (and including) the
            match line.

    Returns:
        Extracted context string, or an empty string if the pattern is
        not found.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    buffer: deque[str] = deque(maxlen=n_lines)
    for line in log_text.splitlines(keepends=True):
        buffer.append(line)
        if compiled.search(line):
            return "".join(buffer)
    return ""


def _extract_context_window_with_trailing(
    log_text: str,
    pattern: str,
    n_before: int,
    n_trailing: int,
) -> str:
    """Extract lines around a pattern match, including lines that follow it.

    Like :func:`_extract_context_window` but continues collecting up to
    ``n_trailing`` lines after the match.  Collection stops early if a blank
    line is encountered after the match (blank lines reliably signal the end
    of a Python traceback block in pilot logs).

    Use this for error codes whose diagnostic content *follows* the anchor
    line (e.g. a WARNING header line followed by a multi-line traceback).

    Args:
        log_text: Full log content as a string.
        pattern: Regular expression to search for.
        n_before: Lines to include before (and including) the match line.
        n_trailing: Maximum additional lines to collect after the match.

    Returns:
        Extracted context string, or an empty string if the pattern is
        not found.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    buffer: deque[str] = deque(maxlen=n_before)
    lines = log_text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        buffer.append(line)
        if compiled.search(line):
            # Collect up to n_trailing further lines, stopping on a blank line.
            trailing: list[str] = []
            for subsequent in lines[i + 1:i + 1 + n_trailing]:
                if subsequent.strip() == "":
                    trailing.append(subsequent)
                    break
                trailing.append(subsequent)
            return "".join(buffer) + "".join(trailing)
    return ""


def _extract_tail(log_text: str, n_lines: int) -> str:
    """Return the last ``n_lines`` lines of a log.

    Args:
        log_text: Full log content.
        n_lines: Number of trailing lines to return.

    Returns:
        Last ``n_lines`` lines joined as a single string.
    """
    lines = log_text.splitlines(keepends=True)
    return "".join(lines[-n_lines:])


def _select_log_filename(job: dict[str, Any]) -> str:
    """Choose the primary log file to download based on job metadata.

    Pilot error code 1305 indicates a user payload failure; in that case
    ``payload.stdout`` is the primary log.  ``payload.stderr`` is also
    fetched separately and appended to the excerpt when available, since
    some failures (e.g. Python tracebacks, segfaults) only appear there.
    All other failures are diagnosed from ``pilotlog.txt``.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.

    Returns:
        Primary log filename string (``"pilotlog.txt"`` or ``"payload.stdout"``).
    """
    pilot_error_code = job.get("piloterrorcode") or 0
    try:
        code = int(pilot_error_code)
    except (ValueError, TypeError):
        code = 0
    return "payload.stdout" if code == 1305 else "pilotlog.txt"


# Matches an `ls -l` long-format entry, e.g.
#   -rw-r--r--. 1 atlasprd000 atlasprd 28218 Aug 17 10:37 remote_open.stderr
# and the "total 72" header that precedes such a block (with or without a pilot
# log record prefix in front of it).
#
# These lines are stripped from the context around a traceback because they are
# never diagnostic and are actively harmful: given a directory listing and no
# real error, an LLM will infer a cause from which files exist and how large
# they are.  That is exactly how job 7261310898 was misdiagnosed as a "remote
# file open failure" — the only signal in the excerpt was that
# remote_open.stderr was 28 kB.
_LISTING_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[-dlbcps][rwxsStT-]{9}[.+]?\s+\d+\s+\S+\s+\S+\s+\d+\s"),
    re.compile(r"^total \d+\s*$"),
    re.compile(r"\|\s*(?:list_work_dir|print_executable)\s*\|"),
    re.compile(r"^\s*$"),
)


def _strip_directory_listing(text: str) -> str:
    """Remove directory-listing and command-echo lines from log context.

    Applied to the context lines surrounding a traceback, never to the traceback
    itself.  See :data:`_LISTING_LINE_RES` for why these lines are worse than
    merely useless.

    Args:
        text: Block of log context lines.

    Returns:
        The same block with listing lines, ``total N`` headers, pilot
        ``list_work_dir``/``print_executable`` records and blank lines removed.
    """
    kept = [
        line for line in text.splitlines()
        if not any(pattern.search(line) for pattern in _LISTING_LINE_RES)
    ]
    return "\n".join(kept)


def _trim_partial_first_line(text: str) -> str:
    """Drop a leading partial line left behind by character-based slicing.

    Slicing a block of log text to a character budget usually cuts through the
    middle of a line.  A half line of pilot log is noise at best and misleading
    at worst, so it is removed.

    Args:
        text: Text whose first line may be a fragment.

    Returns:
        *text* without its first line, or an empty string when *text* is a
        single fragment with no newline.
    """
    if "\n" not in text:
        return ""
    return text.split("\n", 1)[1]


def _build_traceback_excerpt(
    log_text: str,
    block: TracebackBlock,
    max_chars: int,
) -> str:
    """Assemble an excerpt centred on a traceback, within a character budget.

    Allocation order is deliberate: the traceback first (up to
    ``_TRACEBACK_RESERVED_CHARS``), then a short trailing window, then as much
    preceding context as the remaining budget allows.  The preceding context is
    trimmed from its *start* so the lines nearest the traceback survive.

    Args:
        log_text: Full log content the traceback was found in.
        block: The selected traceback block.
        max_chars: Total character budget for the excerpt.

    Returns:
        Excerpt string of at most *max_chars* characters, in log order
        (preceding context, traceback, trailing context).
    """
    tb_text = truncate_traceback(block.text, min(_TRACEBACK_RESERVED_CHARS, max_chars))
    remaining = max_chars - len(tb_text)
    if remaining <= 0:
        return tb_text

    lines = log_text.splitlines()

    trailing = _strip_directory_listing(
        "\n".join(lines[block.end_line:block.end_line + _TRACEBACK_TRAILING_LINES])
    )
    trailing = trailing[:min(_TRACEBACK_TRAILING_CHARS, remaining)]
    remaining -= len(trailing)

    context = ""
    if remaining > 0:
        first = max(0, block.start_line - _CONTEXT_LINES)
        context = _strip_directory_listing("\n".join(lines[first:block.start_line]))
        if len(context) > remaining:
            context = _trim_partial_first_line(context[-remaining:])

    parts = [part for part in (context, tb_text, trailing) if part]
    return "\n".join(parts)


@dataclass
class FailureContext:
    """Excerpt plus structured exception data extracted from a job's logs.

    Attributes:
        excerpt: The log excerpt to send to the LLM.
        exception: Parsed exception when a Python traceback was found in the
            log, otherwise ``None``.
        traceback_count: Number of distinct tracebacks found in the log.  A
            value above 1 means :func:`select_primary_traceback` discarded
            alternatives, which is worth surfacing in evidence.
    """

    excerpt: str = ""
    exception: ExceptionInfo | None = None
    traceback_count: int = 0

    def to_state(self) -> dict[str, Any]:
        """Return a round-trippable representation of the context.

        See :meth:`ExceptionInfo.to_state` for why this exists alongside the
        ``as_dict`` evidence projections rather than replacing them.

        Returns:
            Dict keyed by the dataclass field names, accepted by
            :meth:`from_state`.  ``exception`` is a nested state dict, or
            ``None`` when no traceback was found — a distinction the reader
            must preserve, since ``None`` means "no traceback in this log"
            while an empty dict would mean "a traceback with no detail".
        """
        return {
            "excerpt": self.excerpt,
            "exception": self.exception.to_state() if self.exception else None,
            "traceback_count": self.traceback_count,
        }

    @classmethod
    def from_state(cls, state: Any) -> FailureContext:
        """Rebuild a context from :meth:`to_state` output.

        Args:
            state: Mapping as produced by :meth:`to_state`; see
                ``Frame.from_state`` for why this is typed ``Any``.
                Unknown keys are
                ignored and missing keys fall back to the field defaults.

        Returns:
            The reconstructed :class:`FailureContext`.  ``exception`` is
            ``None`` unless the state carries a mapping for it, so both a
            missing key and an explicit ``null`` restore the no-traceback case.
        """
        if not isinstance(state, Mapping):
            return cls()

        raw_exception: Any = state.get("exception")
        exception: ExceptionInfo | None = None
        if isinstance(raw_exception, Mapping):
            exception = ExceptionInfo.from_state(
                cast("Mapping[str, Any]", raw_exception)
            )

        return cls(
            excerpt=coerce_str(state.get("excerpt")),
            exception=exception,
            traceback_count=coerce_int(state.get("traceback_count")),
        )


def extract_failure_context(
    log_text: str,
    log_filename: str,
    pilot_error_code: int,
    pilot_error_diag: str,
    max_chars: int = _MAX_EXCERPT_CHARS,
) -> FailureContext:
    """Extract the most relevant section of a log, plus any parsed exception.

    Traceback-first: the log is scanned for Python tracebacks and, when one is
    found, the excerpt is built around it via :func:`_build_traceback_excerpt`
    and the exception is parsed into structured form.  This path applies to
    *every* log type — ``pilotlog.txt``, ``payload.stdout``, ``payload.stderr``
    and ``setup.stdout`` — because the traceback format is identical in all of
    them and Athena payload failures benefit as much as pilot failures.

    Only when no traceback is present does the previous behaviour apply: for
    payload logs a character tail, and for ``pilotlog.txt`` the
    ``_PILOT_CODE_PATTERNS`` anchor, then ``piloterrordiag`` as a literal
    regex, then the log tail.

    Args:
        log_text: Full log content as a string.
        log_filename: Name of the log file (used to detect payload logs).
        pilot_error_code: Numeric pilot error code from job metadata.
        pilot_error_diag: Textual pilot error diagnosis from job metadata.
        max_chars: Character budget for the excerpt.  Callers that must reserve
            part of the overall budget for another file (e.g. appending
            ``payload.stderr``) pass the reduced figure here so truncation
            stays traceback-aware rather than happening via a later slice.

    Returns:
        Populated :class:`FailureContext`.  ``excerpt`` is an empty string when
        no relevant section could be found.
    """
    if not log_text:
        return FailureContext()

    is_payload = "payload" in log_filename

    if is_payload or pilot_error_code == 1305:
        # Strip pilot boilerplate (ls listings) first so neither the traceback
        # search nor the char tail spends budget on directory output.
        log_text = _strip_payload_noise(log_text)

    exception, block, count = find_primary_exception(log_text)
    if exception is not None and block is not None:
        logger.info(
            "Traceback-anchored excerpt for %s: %s (%d traceback(s) in log)",
            log_filename,
            exception.exc_type or "unparsed exception",
            count,
        )
        return FailureContext(
            excerpt=_build_traceback_excerpt(log_text, block, max_chars),
            exception=exception,
            traceback_count=count,
        )

    if is_payload or pilot_error_code == 1305:
        # Char-based tail: errors appear at the very end of payload.stdout;
        # a line-count tail on verbose logs would cut them off.
        return FailureContext(excerpt=log_text[-max_chars:])

    search_pattern = _PILOT_CODE_PATTERNS.get(pilot_error_code)
    if search_pattern is None:
        # Unknown code: use first 40 chars of piloterrordiag as pattern.
        # Unreliable (see module docstring) but retained as a last resort.
        search_pattern = re.escape(pilot_error_diag[:40]) if pilot_error_diag else ""

    excerpt = ""
    if search_pattern:
        if pilot_error_code in _TRAILING_CONTEXT_CODES:
            excerpt = _extract_context_window_with_trailing(
                log_text, search_pattern, _CONTEXT_LINES, _TRAILING_LINES
            )
        else:
            excerpt = _extract_context_window(log_text, search_pattern, _CONTEXT_LINES)

    # If no match found, fall back to the tail.
    # WARNING-level so production logs surface gaps in _PILOT_CODE_PATTERNS.
    if not excerpt:
        logger.warning(
            "No traceback found and pattern %r did not match for pilot error "
            "code %d; falling back to tail extraction. The excerpt may contain "
            "only stage-out boilerplate. Consider adding this code to "
            "_PILOT_CODE_PATTERNS.",
            search_pattern,
            pilot_error_code,
        )
        excerpt = _extract_tail(log_text, _CONTEXT_LINES)

    return FailureContext(excerpt=excerpt[:max_chars] if excerpt else "")


def extract_log_excerpt(
    log_text: str,
    log_filename: str,
    pilot_error_code: int,
    pilot_error_diag: str,
    max_chars: int = _MAX_EXCERPT_CHARS,
) -> str:
    """Extract the most relevant section of a log file for LLM analysis.

    Thin wrapper over :func:`extract_failure_context` that returns only the
    excerpt, preserving the original signature for existing callers and tests.

    Args:
        log_text: Full log content as a string.
        log_filename: Name of the log file (used to detect payload logs).
        pilot_error_code: Numeric pilot error code from job metadata.
        pilot_error_diag: Textual pilot error diagnosis from job metadata.
        max_chars: Character budget for the excerpt.

    Returns:
        Extracted context window, at most *max_chars* characters.  Empty
        string if no relevant section is found.
    """
    return extract_failure_context(
        log_text, log_filename, pilot_error_code, pilot_error_diag, max_chars,
    ).excerpt


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# Exception types that mean "the operation did not complete in time".
_TIMEOUT_EXC_TYPES: frozenset[str] = frozenset({
    "TimeoutError", "timeout", "ReadTimeout", "ReadTimeoutError",
    "ConnectTimeout", "ConnectTimeoutError", "SocketTimeout",
})

# Exception types that mean "the network or peer failed", excluding timeouts.
_NETWORK_EXC_TYPES: frozenset[str] = frozenset({
    "ConnectionError", "ConnectionResetError", "ConnectionRefusedError",
    "ConnectionAbortedError", "BrokenPipeError", "URLError", "HTTPError",
    "SSLError", "SSLEOFError", "CertificateError", "gaierror", "herror",
})

# Pilot functions that identify the transformation-script download path:
#   get_payload_command -> get_analysis_trf -> download_transform -> download_file
# A failure anywhere under get_analysis_trf/download_transform means the payload
# command could not even be assembled, so the payload never ran.
_TRANSFORM_DOWNLOAD_FUNCS: frozenset[str] = frozenset({
    "get_analysis_trf", "download_transform",
})

# Signals for the pilot CPU-monitoring UID lookup failure (pilot error 1354).
# Preserved as its own category because bamboo_answer and planner routing
# reference "pilot_monitoring_error" by name.
_MONITORING_FUNCS: frozenset[str] = frozenset({
    "list_processes_and_threads", "get_process_info",
})
_MONITORING_MESSAGE_SIGNALS: tuple[str, ...] = ("getpwuid", "uid not found")


# Keywords for the metadata-only pre-check in classify_failure.  Kept in sync
# with the "reassigned_by_jedi" entry of _FAILURE_PATTERNS.
_REASSIGNED_KEYWORDS: tuple[str, ...] = ("reassigned by jedi", "toreassign")


def _build_search_text(job: dict[str, Any], log_excerpt: str) -> str:
    """Build the lower-cased search string used by substring classification.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.
        log_excerpt: Extracted context window, or an empty string to search
            job metadata only.

    Returns:
        Single lower-cased string joining the error diagnosis fields and the
        excerpt.
    """
    return " ".join([
        str(job.get("taskbuffererrordiag") or ""),
        str(job.get("piloterrordiag") or ""),
        str(job.get("exeerrordiag") or ""),
        str(job.get("jobsubstatus") or ""),
        str(job.get("commandtopilot") or ""),
        log_excerpt,
    ]).lower()


def _classify_from_exception(exception: ExceptionInfo) -> str | None:
    """Classify a failure from a parsed exception rather than substring search.

    Preferred over :data:`_FAILURE_PATTERNS` because it reasons about the
    exception type and the pilot call chain instead of matching substrings
    anywhere in the excerpt.  Substring matching over a low-quality excerpt
    produces confident nonsense: job 7261310898 was classified ``"timeout"``
    because the excerpt happened to contain ``"using timeout=90 s"`` from the
    pilot's log-archiving ``tar`` command, not because anything timed out.

    Args:
        exception: Parsed exception from :mod:`askpanda_atlas._traceback_parse`.

    Returns:
        A failure category string, or ``None`` when the exception is not
        recognised *and* contains no pilot frames — in which case the caller
        should fall through to the substring table (payload tracebacks are
        classified there as ``payload_error``).
    """
    exc_type = exception.exc_type
    message = exception.message.lower()
    funcs = {frame.func for frame in exception.pilot_frames}
    paths = {frame.pilot_path for frame in exception.pilot_frames}

    is_timeout = exc_type in _TIMEOUT_EXC_TYPES or "timed out" in message
    in_transform_download = bool(funcs & _TRANSFORM_DOWNLOAD_FUNCS)

    # Order matters: the transform-download checks are more specific than the
    # bare timeout/network checks and must win.
    if in_transform_download:
        return "transform_download_timeout" if is_timeout else "transform_download_failed"

    if funcs & _MONITORING_FUNCS or any(sig in message for sig in _MONITORING_MESSAGE_SIGNALS):
        return "pilot_monitoring_error"
    if any(path.endswith("psutils.py") for path in paths):
        return "pilot_monitoring_error"

    if exc_type == "MemoryError":
        return "memory"
    if "no space left" in message or "disk quota" in message:
        return "disk_full"
    if exc_type in ("FileNotFoundError", "IsADirectoryError"):
        return "input_missing"
    if is_timeout:
        return "timeout"
    if exc_type in _NETWORK_EXC_TYPES:
        return "network"

    # Unrecognised exception, but it was raised inside pilot code: report it as
    # a pilot exception rather than letting the substring table label it
    # "payload_error", which wrongly blames the user's payload.
    if exception.pilot_frames:
        return "pilot_exception"

    return None


def classify_failure(
    job: dict[str, Any],
    log_excerpt: str,
    exception: ExceptionInfo | None = None,
) -> str:
    """Classify a job failure from job metadata, log excerpt and exception.

    When *exception* is supplied, :func:`_classify_from_exception` is consulted
    first; it is far more reliable than substring matching.  The
    ``_FAILURE_PATTERNS`` table is used only when no exception was parsed or
    the exception is unrecognised and originated outside pilot code.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.
        log_excerpt: Extracted context window from the pilot log.
        exception: Parsed exception from the log, when one was found.

    Returns:
        A short failure category string (e.g. ``"stagein_timeout"``).
        Falls back to ``"unknown"`` if nothing matches.
    """
    # Metadata-level signals that outrank any log content: a job reassigned by
    # JEDI never really "failed" in the pilot, so an exception in its log (if
    # any) is incidental.
    metadata_search = _build_search_text(job, "")
    if any(kw in metadata_search for kw in _REASSIGNED_KEYWORDS):
        return "reassigned_by_jedi"

    if exception is not None:
        category = _classify_from_exception(exception)
        if category is not None:
            return category

    search = _build_search_text(job, log_excerpt)
    for category, keywords in _FAILURE_PATTERNS:
        if any(kw in search for kw in keywords):
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Synchronous fetch-and-analyse (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

@dataclass
class _LogFetchResult:
    """Collected outputs from one of the log-fetch helper functions.

    Bundles the six mutable values that the two fetch helpers must return to
    ``fetch_and_analyse`` so the helpers can be extracted into their own
    functions without needing to return an unwieldy tuple.

    Attributes:
        log_excerpt: Extracted context window for LLM analysis.
        log_url: Filebrowser URL of the primary log file, or ``None``.
        log_available: True if any usable log content was obtained.
        stderr_url: Filebrowser URL of ``payload.stderr``, or ``None``.
        setup_log_url: Filebrowser URL of ``setup.stdout``, or ``None``.
        setup_log_excerpt: Full (budget-capped) content of ``setup.stdout``,
            or ``None`` if it was not fetched or contained no error.
        exception: Parsed exception when a Python traceback was found in any of
            the fetched logs, otherwise ``None``.
        traceback_count: Number of distinct tracebacks found in the log the
            exception came from.
        pilot_version: Pilot release version parsed from the pilot log (e.g.
            ``"3.14.0.22"``), or an empty string when unavailable.  Used to pin
            GitHub source fetches to the tag the job actually ran.
    """

    log_excerpt: str = ""
    log_url: str | None = None
    log_available: bool = False
    stderr_url: str | None = None
    setup_log_url: str | None = None
    setup_log_excerpt: str | None = field(default=None)
    exception: ExceptionInfo | None = None
    traceback_count: int = 0
    pilot_version: str = ""

    def to_state(self) -> dict[str, Any]:
        """Return a round-trippable representation of the fetch result.

        The four URL fields and ``setup_log_excerpt`` are optional strings
        where ``None`` and ``""`` differ: ``None`` means the file was never
        fetched, ``""`` would claim it was fetched and found empty.  The codec
        preserves that distinction rather than normalising it away.

        Returns:
            Dict keyed by the dataclass field names, accepted by
            :meth:`from_state`.
        """
        return {
            "log_excerpt": self.log_excerpt,
            "log_url": self.log_url,
            "log_available": self.log_available,
            "stderr_url": self.stderr_url,
            "setup_log_url": self.setup_log_url,
            "setup_log_excerpt": self.setup_log_excerpt,
            "exception": self.exception.to_state() if self.exception else None,
            "traceback_count": self.traceback_count,
            "pilot_version": self.pilot_version,
        }

    @classmethod
    def from_state(cls, state: Any) -> _LogFetchResult:
        """Rebuild a fetch result from :meth:`to_state` output.

        Args:
            state: Mapping as produced by :meth:`to_state`; see
                ``Frame.from_state`` for why this is typed ``Any``.
                Unknown keys are
                ignored and missing keys fall back to the field defaults.

        Returns:
            The reconstructed :class:`_LogFetchResult`.
        """
        if not isinstance(state, Mapping):
            return cls()

        raw_exception: Any = state.get("exception")
        exception: ExceptionInfo | None = None
        if isinstance(raw_exception, Mapping):
            exception = ExceptionInfo.from_state(
                cast("Mapping[str, Any]", raw_exception)
            )

        return cls(
            log_excerpt=coerce_str(state.get("log_excerpt")),
            log_url=coerce_optional_str(state.get("log_url")),
            log_available=coerce_bool(state.get("log_available")),
            stderr_url=coerce_optional_str(state.get("stderr_url")),
            setup_log_url=coerce_optional_str(state.get("setup_log_url")),
            setup_log_excerpt=coerce_optional_str(state.get("setup_log_excerpt")),
            exception=exception,
            traceback_count=coerce_int(state.get("traceback_count")),
            pilot_version=coerce_str(state.get("pilot_version")),
        )


def _fetch_logs_payload(
    job_id: int,
    pilot_error_code: int,
    pilot_error_diag: str,
    file_index: dict[str, int] | None,
    base_url: str,
    timeout: int,
) -> _LogFetchResult:
    """Fetch and excerpt logs for pilot error code 1305 (payload failure).

    Checks ``setup.stdout`` first.  If it contains a recognisable fatal setup
    error the function returns immediately with that content as the excerpt
    and does not attempt ``payload.stdout`` or ``payload.stderr`` — the
    payload never ran so those files will be empty.

    If ``setup.stdout`` is absent, empty, or error-free the function falls
    through to the ``payload.stdout`` → ``payload.stderr`` path, skipping any
    file that the file-size index confirms to be zero-length.

    Args:
        job_id: PanDA job ID.
        pilot_error_code: Numeric pilot error code (always 1305 for this path).
        pilot_error_diag: Textual pilot error diagnosis from job metadata.
        file_index: Mapping of ``{filename: size_bytes}`` from
            :func:`_fetch_file_index`, or ``None`` (fail-open).
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Populated :class:`_LogFetchResult` instance.
    """
    result = _LogFetchResult()

    # --- setup.stdout first ---
    setup_fetched = False
    if _file_is_nonempty(file_index, "setup.stdout"):
        result.setup_log_url = _log_file_url(job_id, "setup.stdout", base_url)
        setup_text = _fetch_log_text(job_id, "setup.stdout", base_url, timeout)
        if setup_text:
            setup_fetched = True
            if _setup_log_has_error(setup_text):
                setup_ctx = extract_failure_context(
                    setup_text, "setup.stdout", pilot_error_code,
                    pilot_error_diag, _MAX_EXCERPT_CHARS,
                )
                # Setup errors are usually shell output rather than tracebacks;
                # when no traceback is present keep the whole (capped) file so
                # the asetup/release diagnostics are not cropped by an anchor.
                result.setup_log_excerpt = (
                    setup_ctx.excerpt if setup_ctx.exception
                    else setup_text[:_MAX_EXCERPT_CHARS]
                )
                result.log_excerpt = result.setup_log_excerpt
                result.exception = setup_ctx.exception
                result.traceback_count = setup_ctx.traceback_count
                result.log_available = True
                logger.info(
                    "Setup error found in setup.stdout for job %d; "
                    "skipping payload.stdout and payload.stderr.",
                    job_id,
                )
                return result
    else:
        logger.info("setup.stdout is zero-length for job %d; skipping.", job_id)

    # --- Fall through to payload logs ---
    log_filename = "payload.stdout"
    result.log_url = _log_file_url(job_id, log_filename, base_url)
    log_text: str | None = None
    if _file_is_nonempty(file_index, log_filename):
        log_text = _fetch_log_text(job_id, log_filename, base_url, timeout)
    else:
        logger.info("payload.stdout is zero-length for job %d; skipping.", job_id)

    stderr_text: str | None = None
    if _file_is_nonempty(file_index, "payload.stderr"):
        result.stderr_url = _log_file_url(job_id, "payload.stderr", base_url)
        stderr_text = _fetch_log_text(job_id, "payload.stderr", base_url, timeout)
    else:
        logger.info("payload.stderr is zero-length for job %d; skipping.", job_id)

    if log_text or stderr_text:
        result.log_available = True
        # Pass the reduced budget into extraction rather than slicing the
        # result afterwards: a post-hoc slice can cut a traceback and discard
        # the terminal exception line, which is the whole point of the excerpt.
        stdout_budget = _MAX_EXCERPT_CHARS - _STDERR_RESERVED_CHARS
        stdout_ctx = extract_failure_context(
            log_text or "", log_filename,
            pilot_error_code, pilot_error_diag, stdout_budget,
        )
        stderr_ctx = FailureContext()
        if stderr_text:
            stderr_ctx = extract_failure_context(
                stderr_text, "payload.stderr",
                pilot_error_code, pilot_error_diag, _STDERR_RESERVED_CHARS,
            )
            result.log_excerpt = (
                stdout_ctx.excerpt
                + "\n\n--- payload.stderr ---\n"
                + stderr_ctx.excerpt
            )
        else:
            result.log_excerpt = stdout_ctx.excerpt

        # Prefer the stderr traceback: Python tracebacks and segfault reports
        # are written to stderr, so when both files contain one, stderr holds
        # the exception that actually terminated the payload.
        chosen = stderr_ctx if stderr_ctx.exception else stdout_ctx
        result.exception = chosen.exception
        result.traceback_count = chosen.traceback_count
    elif setup_fetched:
        # setup.stdout was fetched but contained no recognised error; use it
        # as the excerpt so the LLM still has environment context.
        result.log_available = True
        result.log_excerpt = result.setup_log_excerpt or ""
    else:
        logger.info(
            "No usable logs for job %d; proceeding with metadata only.", job_id
        )

    return result


def _fetch_logs_pilotlog(
    job: dict[str, Any],
    job_id: int,
    pilot_error_code: int,
    pilot_error_diag: str,
    file_index: dict[str, int] | None,
    base_url: str,
    timeout: int,
) -> _LogFetchResult:
    """Fetch and excerpt the pilot log for all non-1305 error codes.

    Selects either ``pilotlog.txt`` or ``payload.stdout`` via
    :func:`_select_log_filename`, skips the file if the size index confirms
    zero bytes, then extracts the relevant context window.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.
        job_id: PanDA job ID.
        pilot_error_code: Numeric pilot error code from job metadata.
        pilot_error_diag: Textual pilot error diagnosis from job metadata.
        file_index: Mapping of ``{filename: size_bytes}`` from
            :func:`_fetch_file_index`, or ``None`` (fail-open).
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Populated :class:`_LogFetchResult` instance.
    """
    result = _LogFetchResult()
    log_filename = _select_log_filename(job)
    result.log_url = _log_file_url(job_id, log_filename, base_url)

    log_text: str | None = None
    if _file_is_nonempty(file_index, log_filename):
        log_text = _fetch_log_text(job_id, log_filename, base_url, timeout)
    else:
        logger.info("%s is zero-length for job %d; skipping.", log_filename, job_id)

    if log_text:
        result.log_available = True
        context = extract_failure_context(
            log_text, log_filename, pilot_error_code, pilot_error_diag,
            _MAX_EXCERPT_CHARS,
        )
        result.log_excerpt = context.excerpt
        result.exception = context.exception
        result.traceback_count = context.traceback_count
        # Parse the version from the *full* log: the pilot reports it during
        # start-up, so an excerpt taken from the failure point will not have it.
        result.pilot_version = parse_pilot_version(log_text)
    else:
        logger.info(
            "Log unavailable for job %d; proceeding with metadata only.", job_id
        )

    return result


def _build_exception_evidence(
    exception: ExceptionInfo | None,
    traceback_count: int,
) -> dict[str, Any]:
    """Build the exception-related evidence keys.

    Promoting the exception to first-class evidence keys means the synthesis LLM
    does not have to locate it inside ``log_excerpt``, and gives the follow-up
    ``pilot_source_analysis`` route a reliable signal to gate on.

    Args:
        exception: Parsed exception, or ``None`` when the log had no traceback.
        traceback_count: Number of tracebacks found in the log.

    Returns:
        Dict of evidence keys.  All keys are always present (``None``/``False``
        when there is no exception) so downstream consumers can rely on the
        shape rather than probing with ``in``.
    """
    if exception is None:
        return {
            "traceback_available": False,
            "exception_type": None,
            "exception_message": None,
            "exception_frames": None,
            "deepest_pilot_frame": None,
            "traceback_count": 0,
        }

    parsed = exception.as_dict()
    return {
        "traceback_available": True,
        "exception_type": parsed["type"] or None,
        "exception_message": parsed["message"] or None,
        "exception_frames": parsed["frames"],
        "deepest_pilot_frame": parsed["deepest_pilot_frame"],
        "traceback_count": traceback_count,
    }


def _build_code_analysis_offer(exception: ExceptionInfo | None) -> str:
    """Build the Markdown follow-up offer for pilot source code analysis.

    Args:
        exception: Parsed exception, or ``None`` when the log had no traceback.

    Returns:
        Markdown string naming the pilot frame the traceback originates in and
        inviting a source-level follow-up, or an empty string when there is no
        pilot frame to analyse.
    """
    if exception is None:
        return ""
    deepest = exception.deepest_pilot_frame
    if deepest is None:
        return ""
    return (
        f"\n\nThe exception was raised in pilot code at "
        f"`{deepest.pilot_path}:{deepest.lineno}` (`{deepest.func}`). "
        f"Ask me to show the pilot source for a code-level diagnosis."
    )


def _format_bytes(size_bytes: int) -> str:
    """Render a byte count in decimal units for human-facing text.

    Decimal (1000-based) units are used deliberately so the figure matches
    what the job monitor reports for the same file.

    Args:
        size_bytes: Size in bytes; negative values are treated as zero.

    Returns:
        A short string such as ``"225 B"``, ``"444 kB"`` or ``"1.1 GB"``.
    """
    size = max(0, int(size_bytes))
    if size < 1000:
        return f"{size} B"
    scaled = float(size)
    for unit in ("kB", "MB", "GB", "TB"):
        scaled /= 1000.0
        if scaled < 1000.0 or unit == "TB":
            return f"{scaled:.1f} {unit}"
    return f"{scaled:.1f} TB"  # pragma: no cover - unreachable, loop returns


def _find_core_dump_candidates(
    file_listing: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Select the core files from a normalised job-log listing.

    Args:
        file_listing: Normalised records from :func:`_fetch_file_listing`, or
            ``None`` when the listing is unavailable.

    Returns:
        Matching records ordered largest first, each with ``name``,
        ``dirname``, ``size_bytes`` and ``modification`` keys.  Empty when the
        listing is unavailable or contains no core file.
    """
    if not file_listing:
        return []
    candidates = [
        {
            "name": record["name"],
            "dirname": record["dirname"],
            "size_bytes": int(record["size_bytes"]),
            "modification": record["modification"],
        }
        for record in file_listing
        if _CORE_FILE_RE.match(str(record.get("name") or ""))
    ]
    candidates.sort(key=lambda item: (-item["size_bytes"], item["name"]))
    return candidates


def _classify_core_dump_probe(
    file_listing: list[dict[str, Any]] | None,
    candidates: list[dict[str, Any]],
    pilot_error_code: int,
) -> str:
    """Classify what the job log says about core-dump availability.

    The distinction between the negative outcomes matters to the user: a
    zero-length core means the kernel was still writing it when the process
    was killed, whereas no core at all after a looping-job kill usually means
    core creation did not finish in the time the pilot allowed.  Reporting
    both as "no core dump" would hide a real difference in what went wrong.

    Args:
        file_listing: Normalised listing records, or ``None`` when the listing
            was not fetched or could not be fetched.
        candidates: Core files found by :func:`_find_core_dump_candidates`.
        pilot_error_code: The job's pilot error code.

    Returns:
        One of :data:`CORE_DUMP_PRESENT`, :data:`CORE_DUMP_TRUNCATED`,
        :data:`CORE_DUMP_TIMED_OUT`, :data:`CORE_DUMP_ABSENT` or
        :data:`CORE_DUMP_NOT_PROBED`.
    """
    if file_listing is None:
        return CORE_DUMP_NOT_PROBED
    if candidates:
        if any(item["size_bytes"] > 0 for item in candidates):
            return CORE_DUMP_PRESENT
        return CORE_DUMP_TRUNCATED
    if pilot_error_code == _LOOPING_JOB_PILOT_CODE:
        return CORE_DUMP_TIMED_OUT
    return CORE_DUMP_ABSENT


def _build_core_dump_offer(
    probe_state: str,
    candidates: list[dict[str, Any]],
    pilot_error_code: int,
) -> str:
    """Build the Markdown follow-up offer for core-dump analysis.

    Deliberately narrow: offered only for a looping-job kill with a usable
    core, following the ``_build_code_analysis_offer`` precedent that a
    proactive offer must be built deterministically here rather than left to
    the synthesis LLM.  An explicit request remains available for any failure
    type, so narrowing the proactive offer costs no capability.

    Args:
        probe_state: Outcome from :func:`_classify_core_dump_probe`.
        candidates: Core files found by :func:`_find_core_dump_candidates`.
        pilot_error_code: The job's pilot error code.

    Returns:
        Markdown offer string, or an empty string when no offer should be
        made.
    """
    if not _CORE_DUMP_ANALYSIS_AVAILABLE:
        return ""
    if probe_state != CORE_DUMP_PRESENT:
        return ""
    if pilot_error_code != _LOOPING_JOB_PILOT_CODE:
        return ""
    usable = [item for item in candidates if item["size_bytes"] > 0]
    if not usable:
        return ""
    largest = usable[0]
    extra = ""
    if len(usable) > 1:
        others = len(usable) - 1
        extra = f" ({others} further core file{'s' if others > 1 else ''} present)"
    return (
        f"\n\nA core dump is present in the job log: `{largest['name']}` "
        f"({_format_bytes(largest['size_bytes'])}){extra}. I can fetch it and "
        f"run gdb inside the matching ATLAS release container — usually under a "
        f"minute. Analyse it?"
    )


def _build_core_dump_evidence(
    file_listing: list[dict[str, Any]] | None,
    pilot_error_code: int,
) -> dict[str, Any]:
    """Build the core-dump evidence keys from the job-log listing.

    No network access beyond the listing already fetched for the log
    downloads, no disk access and no gdb: this is a pure read of the listing's
    own ``size`` and ``modification`` fields.

    Args:
        file_listing: Normalised listing records, or ``None`` when the listing
            was not fetched or could not be fetched.
        pilot_error_code: The job's pilot error code.

    Returns:
        Dict of evidence keys.  All keys are always present so downstream
        consumers can rely on the shape.  ``core_dump_available`` is ``None``
        rather than ``False`` when nothing was probed, so "not looked at" is
        never reported as "not there".
    """
    candidates = _find_core_dump_candidates(file_listing)
    probe_state = _classify_core_dump_probe(
        file_listing, candidates, pilot_error_code
    )
    available: bool | None = (
        None if probe_state == CORE_DUMP_NOT_PROBED
        else probe_state == CORE_DUMP_PRESENT
    )
    return {
        "core_dump_probe_state": probe_state,
        "core_dump_available": available,
        "core_dump_candidates": candidates,
        "core_dump_total_bytes": sum(item["size_bytes"] for item in candidates),
        "core_dump_offer_md": _build_core_dump_offer(
            probe_state, candidates, pilot_error_code
        ),
    }


def fetch_and_analyse(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
    """Fetch metadata and logs, extract context window, classify failure.

    Intentionally synchronous so it can be offloaded to a thread pool via
    ``asyncio.to_thread``, keeping the async event loop unblocked during
    network I/O.

    For pilot error code 1305 the download order is: ``setup.stdout`` first
    (see :func:`_fetch_logs_payload`); for all other codes ``pilotlog.txt``
    is used (see :func:`_fetch_logs_pilotlog`).  Both helpers skip any file
    confirmed to be zero-length by the filebrowser file-size index.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL (from environment or default).
        timeout: HTTP timeout in seconds for each request.

    Returns:
        Dict with ``evidence`` and ``text`` keys suitable for
        JSON serialisation and return as MCP content.
    """
    monitor_url = f"{base_url}/job?pandaid={job_id}"
    base_evidence: dict[str, Any] = {
        "job_id": job_id,
        "monitor_url": monitor_url,
    }

    # --- Step 1: Fetch metadata ---
    payload = _fetch_metadata(job_id, base_url, timeout)
    if payload is None:
        base_evidence["error"] = "Failed to fetch job metadata from BigPanDA"
        return {
            "evidence": base_evidence,
            "text": f"Could not retrieve metadata for job {job_id}.",
        }

    job: dict[str, Any] = payload.get("job") or {}
    if not job:
        base_evidence["not_found"] = True
        return {
            "evidence": base_evidence,
            "text": f"Job {job_id} was not found in BigPanDA.",
        }

    jobstatus = str(job.get("jobstatus") or "")
    pilot_error_code: int = 0
    try:
        pilot_error_code = int(job.get("piloterrorcode") or 0)
    except (ValueError, TypeError):
        pass
    pilot_error_diag: str = str(job.get("piloterrordiag") or "")

    # --- Step 2: Download logs (only for failed/holding/cancelled jobs) ---
    fetch_result = _LogFetchResult()

    file_listing: list[dict[str, Any]] | None = None

    if jobstatus in ("failed", "holding", "cancelled"):
        # Fetch the recursive listing once.  It feeds both the zero-length
        # skip decisions below and the core-dump probe further down, so no
        # extra request is made for the probe.  Returns None on failure;
        # helpers treat None as fail-open.
        file_listing = _fetch_file_listing(job_id, base_url, timeout)
        file_index = (
            None if file_listing is None
            else {
                record["relative_path"]: int(record["size_bytes"])
                for record in file_listing
            }
        )
        # The fetch helpers only ever ask about canonical root-level logs, so
        # they get the basename-safe view: a nested namesake such as
        # ``workDir/usr/.../setup.sh`` must not answer for a root-level file.
        top_level_index = _top_level_file_index(file_index)

        if pilot_error_code == 1305:
            fetch_result = _fetch_logs_payload(
                job_id, pilot_error_code, pilot_error_diag,
                top_level_index, base_url, timeout,
            )
        else:
            fetch_result = _fetch_logs_pilotlog(
                job, job_id, pilot_error_code, pilot_error_diag,
                top_level_index, base_url, timeout,
            )

    log_excerpt = fetch_result.log_excerpt
    log_url = fetch_result.log_url
    log_available = fetch_result.log_available
    stderr_url = fetch_result.stderr_url
    setup_log_url = fetch_result.setup_log_url
    setup_log_excerpt = fetch_result.setup_log_excerpt
    exception = fetch_result.exception

    # Fall back to the pilotid metadata field when the pilot log was not
    # downloaded (e.g. pilot error 1305 reads payload.stdout instead).
    pilot_version = fetch_result.pilot_version or parse_pilot_version_from_pilotid(
        str(job.get("pilotid") or "")
    )

    # --- Step 3: Classify failure ---
    failure_type = classify_failure(job, log_excerpt, exception)

    # --- Step 4: Build evidence dict ---
    evidence: dict[str, Any] = {
        **base_evidence,
        "jobstatus": jobstatus,
        "jobsubstatus": job.get("jobsubstatus"),
        "computingsite": job.get("computingsite"),
        "cloud": job.get("cloud"),
        "atlasrelease": job.get("atlasrelease"),
        "jeditaskid": job.get("jeditaskid"),
        "attemptnr": job.get("attemptnr"),
        "maxattempt": job.get("maxattempt"),
        "transformation": job.get("transformation"),
        "piloterrorcode": pilot_error_code,
        "piloterrordiag": pilot_error_diag,
        "exeerrorcode": job.get("exeerrorcode"),
        "exeerrordiag": job.get("exeerrordiag"),
        "taskbuffererrorcode": job.get("taskbuffererrorcode"),
        "taskbuffererrordiag": job.get("taskbuffererrordiag"),
        "ddmerrorcode": job.get("ddmerrorcode"),
        "ddmerrordiag": job.get("ddmerrordiag"),
        "starttime": job.get("starttime"),
        "endtime": job.get("endtime"),
        "duration": job.get("duration"),
        "failure_type": failure_type,
        "log_url": log_url,
        "stderr_url": stderr_url,
        "setup_log_url": setup_log_url,
        "setup_log_excerpt": setup_log_excerpt,
        "log_available": log_available,
        "log_excerpt": log_excerpt or None,
        "pilot_version": pilot_version or None,
    }

    evidence.update(_build_exception_evidence(exception, fetch_result.traceback_count))
    # Core-dump probe.  Reuses the listing already fetched above, so a job in a
    # non-terminal state (where no listing is fetched) reports "not probed"
    # rather than paying for an extra request to assert an obvious negative.
    evidence.update(_build_core_dump_evidence(file_listing, pilot_error_code))

    summary = f"Job {job_id} ({jobstatus}): failure type '{failure_type}'."
    if exception is not None and exception.exc_type:
        # The parsed exception is more trustworthy than piloterrordiag, which is
        # a summary written elsewhere in the pilot and can contradict the log.
        summary += f" Exception: {exception.exc_type}: {exception.message[:120]}."
        deepest = exception.deepest_pilot_frame
        if deepest is not None:
            summary += (
                f" Raised in pilot code at {deepest.pilot_path}:{deepest.lineno}"
                f" ({deepest.func})."
            )
    elif job.get("taskbuffererrordiag"):
        summary += f" Task buffer: {job['taskbuffererrordiag']}."
    elif pilot_error_diag:
        summary += f" Pilot: {pilot_error_diag[:120]}."

    # Build a pre-formatted Markdown links block stored in evidence so that
    # bamboo_executor can append it verbatim after LLM synthesis, bypassing
    # the LLM entirely and guaranteeing real URLs reach the TUI.
    link_lines: list[str] = [f"- [BigPanDA Monitor]({monitor_url})"]
    if setup_log_url:
        link_lines.append(f"- [Setup Log]({setup_log_url})")
    if log_url:
        log_label = (
            "Payload Log"
            if log_available and pilot_error_code == 1305
            else "Pilot Log"
            if log_available
            else "Pilot Log (may not be available yet)"
        )
        link_lines.append(f"- [{log_label}]({log_url})")
    if stderr_url:
        link_lines.append(f"- [Payload stderr]({stderr_url})")
    evidence["links_md"] = "\n\nLinks:\n" + "\n".join(link_lines)

    # Deterministic follow-up offer, appended verbatim by bamboo_executor so the
    # LLM cannot garble the frame reference.  Only offered when the traceback
    # actually reaches pilot3 code, since pilot_source_analysis has nothing to
    # fetch for a pure payload traceback.
    evidence["code_analysis_offer_md"] = _build_code_analysis_offer(exception)

    return {"evidence": evidence, "text": summary}


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for panda_log_analysis.

    Returns:
        Dict with ``name``, ``description``, ``inputSchema``,
        ``examples``, and ``tags`` keys.
    """
    return {
        "name": "panda_log_analysis",
        "description": (
            "Diagnose why a specific PanDA job failed. Downloads the job's "
            "pilot log and error metadata from BigPanDA, extracts the "
            "relevant failure context, and classifies the error "
            "(e.g. stage-in timeout, segfault, memory error, network issue, "
            "payload failure, JEDI reassignment). Use when the question asks "
            "why a job failed, what the error was, or what action to take."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid) to analyse.",
                },
                "query": {
                    "type": "string",
                    "description": "Original user query (optional).",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context (site, task ID, release, etc.).",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "examples": [
            {"job_id": 6799893074, "query": "Why did job 6799893074 fail?"},
        ],
        "tags": ["atlas", "panda", "bigpanda", "job", "log", "failure", "diagnosis"],
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class PandaLogAnalysisTool:
    """MCP tool for downloading and analysing PanDA job failure logs.

    Fetches job metadata and pilot/payload logs directly from BigPanDA,
    extracts a failure context window using pilot error code patterns,
    classifies the failure, and returns structured evidence for LLM
    summarisation.
    """

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> list[Any]:
        """Fetch logs and return structured failure analysis.

        ``bamboo.tools.base`` is imported here (deferred) so the rest of
        this module remains importable when bamboo core is not installed.
        All blocking HTTP calls are offloaded to a thread pool via
        ``asyncio.to_thread`` so the async event loop is not blocked.

        The result is a one-element ``list[MCPContent]`` whose ``text``
        field contains the JSON-serialised evidence dict.  Callers that
        need the raw evidence should parse
        ``json.loads(result[0]["text"])``.  This keeps the tool compliant
        with the MCP narrow-waist contract.

        Args:
            arguments: Dict with required ``job_id`` (int) and optional
                ``query`` (str) and ``context`` (str).

        Returns:
            One-element MCP content list containing the JSON-serialised
            evidence and text summary.
        """
        from bamboo.tools.base import text_content  # deferred — see module docstring

        def _err(payload: dict[str, Any]) -> list[Any]:
            return text_content(json.dumps(payload))

        if not isinstance(arguments, dict):
            return _err({
                "evidence": {
                    "error": "arguments must be a dict",
                    "provided": repr(arguments),
                },
            })

        job_id = arguments.get("job_id")
        if job_id is None:
            return _err({"evidence": {"error": "missing job_id", "provided": str(arguments)}})

        try:
            job_id_int = int(job_id)
        except (ValueError, TypeError):
            return _err({
                "evidence": {
                    "error": "job_id must be an integer",
                    "provided": str(arguments),
                },
            })

        timeout: int = 60
        try:
            timeout = int(arguments.get("timeout") or 60)
        except (ValueError, TypeError):
            pass

        base_url = get_base_url()

        try:
            result = await asyncio.to_thread(
                fetch_and_analyse, job_id_int, base_url, timeout
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error analysing job %d", job_id_int)
            return _err({
                "evidence": {
                    "job_id": job_id_int,
                    "monitor_url": f"{base_url}/job?pandaid={job_id_int}",
                    "error": repr(exc),
                },
                "text": f"Unexpected error while analysing job {job_id_int}: {exc}",
            })

        return text_content(json.dumps(result))


panda_log_analysis_tool = PandaLogAnalysisTool()

__all__ = [
    "FailureContext",
    "PandaLogAnalysisTool",
    "classify_failure",
    "extract_failure_context",
    "extract_log_excerpt",
    "fetch_and_analyse",
    "get_definition",
    "panda_log_analysis_tool",
    "_build_code_analysis_offer",
    "_build_exception_evidence",
    "_classify_from_exception",
    "_fetch_file_index",
    "_file_is_nonempty",
    "_log_file_url",
    "_select_log_filename",
    "_setup_log_has_error",
    "_top_level_file_index",
]
