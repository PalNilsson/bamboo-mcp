#!/usr/bin/env python3
"""Analyze a core dump with gdb and explain it in plain language using an LLM.

Backs the ``atlas.core_dump_analysis`` MCP tool and remains usable as a
standalone CLI (see ``scripts/analyze_core_dump.py`` and its README).  It is
deliberately split into two layers:

1.  An *evidence layer* (:func:`collect_evidence`) that drives ``gdb`` in batch
    mode and normalises, de-duplicates, redacts and truncates the output into a
    JSON-serialisable dictionary. This layer has no LLM awareness at all.
2.  A *synthesis layer* (:func:`analyze_with_llm`) that hands that dictionary to
    an LLM and asks for an operator-readable explanation.  The provider is not
    fixed; see :func:`resolve_llm_backend`.

Bamboo runs layer 1 out of process with ``--no-llm --json`` and performs
synthesis itself, so that model configuration, credentials, prompt logging and
tracing stay in one place.  Running gdb in a subprocess also isolates the
server from a gdb hang or OOM on a multi-gigabyte core.

Any caller that synthesises for itself must still apply
:func:`reconcile_llm_analysis` to the model's structured reply.  That guard is
what stops EventLoop completion markers being read as evidence that a looping
PanDA job finished normally, and dropping it reproduces a misdiagnosis this
analyzer previously shipped.  Bamboo therefore imports it rather than
reimplementing it: :func:`build_system_prompt`, :func:`build_user_prompt`,
:func:`extract_json_object`, :func:`core_evidence_from_dict` and
:func:`reconcile_llm_analysis` are the intended embedding surface.
:func:`render_report` is not — it is the CLI's own fixed-width presentation.

Typical usage::

    python -m askpanda_atlas._core_dump_analyzer core.123456
    python scripts/analyze_core_dump.py core.123456 --mode hang
    python scripts/analyze_core_dump.py core.123456 --no-llm --json evidence.json
    python scripts/analyze_core_dump.py core.123456 --llm-backend anthropic

Note on the executable: gdb needs the ELF binary that was running, not the
script. For an ``athena.py`` job that is the Python interpreter, and it must be
the same build the job used (normally from CVMFS). The script tries to recover
that path automatically from the core's NT_FILE note before falling back to
``--exe``.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

__version__ = "0.3.0"

#: Version of the ``--json`` payload contract. Bumped only when the shape of
#: that payload changes incompatibly, independently of :data:`__version__`.
#: Consumers reading the subprocess output must key on this rather than on the
#: tool version, since the two move for different reasons.
EVIDENCE_SCHEMA_VERSION: int = 1

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_FRAMES = 40
DEFAULT_MAX_THREAD_GROUPS = 25
DEFAULT_MAX_TARGETED_THREADS = 3
DEFAULT_MAX_JOB_LOG_FILES = 12
DEFAULT_MAX_JOB_LOG_MATCHES = 60
DEFAULT_MAX_JOB_LOG_BYTES = 20 * 1024 * 1024
DEFAULT_JOB_LOG_TAIL_LINES = 20
DEFAULT_HANG_WORKDIR_LOG_RECENCY_S = 2 * 60 * 60
DEFAULT_GDB_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 4000
DEFAULT_MAX_EVIDENCE_CHARS = 50_000
DEFAULT_CONTAINER_TIMEOUT = 1800
DEFAULT_ATLAS_LOCAL_ROOT_BASE = "/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
DEFAULT_ATLAS_PLATFORM = "el9"

#: Seconds between "still running" heartbeat messages during a gdb phase.
#: Only printed with --verbose; a phase producing no output for this long is
#: exactly the "is it frozen?" situation the heartbeat exists to answer.
DEFAULT_HEARTBEAT_INTERVAL: float = 15.0

#: Core size, in MiB, above which a one-time slow-analysis warning is printed.
#: Set --large-core-warning-mib 0 to disable. gdb reloads the whole core once
#: per phase (see README on the deliberate per-phase-subprocess design), so
#: wall-clock time scales with both core size and phase count.
DEFAULT_LARGE_CORE_WARNING_MIB = 1024

#: Rough characters-per-token ratio used only for a human-readable size log
#: line before the API call. Not an accurate tokenizer; do not use for billing.
CHARS_PER_TOKEN_ESTIMATE = 4

#: Hard multiplier on --max-evidence-chars applied as a last-resort cap on the
#: full rendered prompt just before the API call. enforce_global_budget()
#: should always bring evidence under --max-evidence-chars first; this exists
#: as defense-in-depth so a future evidence field can never bypass the budget
#: and send an unbounded (and unboundedly expensive) prompt.
HARD_CAP_MULTIPLIER = 2

#: Per-section character budgets applied before the global budget.
SECTION_LIMITS: dict[str, int] = {
    "backtrace": 12_000,
    "registers": 3_000,
    "args": 4_000,
    "locals": 8_000,
    "frame": 2_000,
    "python_backtrace": 8_000,
    "python_source": 3_000,
    "thread_group": 6_000,
    "targeted_frame": 1_500,
    "targeted_args": 2_000,
    "targeted_locals": 3_000,
    "job_log_line": 800,
}

#: Emitted by gdb's ``echo`` between commands so sections can be split exactly
#: rather than guessed at with boundary regexes.
SECTION_MARKER = "@@BAMBOO_SECTION:{name}@@"
_MARKER_RE = re.compile(r"^@@BAMBOO_SECTION:([a-z0-9_]+)@@\s*$", re.M)

#: Signals that indicate a genuine fault rather than a deliberate core dump.
CRASH_SIGNALS = frozenset({"SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL", "SIGSYS", "SIGTRAP"})

#: Signals typically seen when a supervisor snapshots or kills a looping job.
HANG_SIGNALS = frozenset({"SIGQUIT", "SIGABRT", "SIGTERM", "SIGKILL", "SIGUSR1", "SIGUSR2", "SIGINT"})

#: Earliest possible gdb initialization. AnalysisBase exports PYTHONHOME/PYTHONPATH
#: for its Python 3.13 runtime, while EL9 gdb embeds Python 3.9. This setting,
#: together with sanitising those environment variables at process launch, keeps
#: gdb's embedded Python from trying to import the wrong standard library.
GDB_EARLY_INIT_COMMANDS: tuple[str, ...] = (
    "set python ignore-environment on",
)

#: gdb settings applied before the core is loaded. Errors here are harmless on
#: older gdb builds (the command is simply undefined) and are captured in stderr.
GDB_INIT_COMMANDS: tuple[str, ...] = (
    "set confirm off",
    "set pagination off",
    "set height 0",
    "set width 0",
    "set backtrace past-main on",
    "set print frame-arguments scalars",
    # Keep frame lines short for large C/C++ containers. Note that language
    # pretty-printers (notably libpython's) apply their own internal cap instead.
    "set print elements 40",
    "set print repeats 8",
    # debuginfod prompts block on stdin in EL9 gdb and would hang the batch run.
    "set debuginfod enabled off",
    # Required for libpython-gdb.py to auto-load from CVMFS/LCG paths (py-bt).
    "set auto-load safe-path /",
)

#: Patterns scrubbed from any gdb text before it leaves this process.
REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_KEY]"),
    (re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S), "[REDACTED_CERT]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED_JWT]"),
    (re.compile(r"/tmp/x509up_u\d+\w*"), "/tmp/[REDACTED_PROXY]"),
    (re.compile(r"\b(\w*(?:TOKEN|PASSWORD|SECRET|APIKEY|API_KEY))\s*=\s*\S+", re.I), r"\1=[REDACTED]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), "[REDACTED_API_KEY]"),
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class GdbPhaseResult:
    """Outcome of one batched gdb invocation.

    Attributes:
        name: Short identifier for the phase, e.g. ``"metadata"``.
        commands: The gdb commands that were executed, in order.
        stdout: Captured standard output.
        stderr: Captured standard error.
        sections: Per-command output, keyed by section name.
        returncode: Process exit code, or ``-1`` if the phase timed out.
        timed_out: Whether the phase exceeded its timeout.
        duration_s: Wall-clock duration in seconds, rounded to two decimals.
    """

    name: str
    commands: list[str]
    sections: dict[str, str] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    duration_s: float = 0.0


@dataclass
class ThreadGroup:
    """A set of threads sharing an identical (address-normalised) backtrace.

    Attributes:
        count: Number of threads with this backtrace.
        thread_ids: gdb thread numbers belonging to the group (may be truncated).
        names: Distinct thread names observed in the group.
        backtrace: One representative backtrace for the group.
        idle: Backwards-compatible flag indicating a genuinely benign idle wait.
        state: ``"active"``, ``"blocked"``, or ``"idle"``. A thread blocked on
            a synchronization primitive while executing meaningful shutdown, I/O,
            or lock-acquisition code is ``"blocked"`` rather than ``"idle"``.
    """

    count: int
    thread_ids: list[str]
    names: list[str]
    backtrace: str
    idle: bool = False
    state: str = "active"


@dataclass
class CoreEvidence:
    """Structured, LLM-ready evidence extracted from a core dump.

    Attributes:
        core_file: Path, size and modification time of the core file.
        executable: Resolved executable path plus how it was resolved.
        gdb: gdb executable path and reported version.
        signal: Terminating signal name, if gdb reported one.
        mode: Either ``"crash"`` or ``"hang"``, after auto-detection.
        mode_source: Whether the mode was supplied or inferred, and from what.
        generated_by: The ``Core was generated by`` command line, if present.
        thread_count: Total number of threads found in the core.
        warnings: Human-readable warnings about degraded evidence quality.
        primary_thread: Backtrace, args, locals, registers of the faulting thread.
        thread_groups: De-duplicated backtraces across all threads.
        targeted_threads: Focused frame/args/locals evidence for selected non-idle threads.
        python: Python-level backtrace from ``py-bt``, if available.
        job_logs: Bounded PanDA/payload log evidence correlated with the captured state.
        process_identity: Conservative identification of whether the captured process is the payload, prmon, or unknown.
        diagnosis: Conservative machine-readable deterministic diagnosis for downstream tools.
        shared_libraries: Summary of loaded libraries and missing symbols.
        phases: Raw metadata about each gdb invocation.
        truncated_sections: Names of sections shortened to fit the budget.
    """

    core_file: dict[str, Any] = field(default_factory=dict)
    executable: dict[str, Any] = field(default_factory=dict)
    gdb: dict[str, Any] = field(default_factory=dict)
    signal: str | None = None
    mode: str = "auto"
    mode_source: str = ""
    generated_by: str | None = None
    thread_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    primary_thread: dict[str, str] = field(default_factory=dict)
    thread_groups: list[ThreadGroup] = field(default_factory=list)
    targeted_threads: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    job_logs: dict[str, Any] = field(default_factory=dict)
    process_identity: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    python: dict[str, Any] = field(default_factory=dict)
    shared_libraries: dict[str, Any] = field(default_factory=dict)
    build_ids: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    gdb_metadata: dict[str, Any] = field(default_factory=dict)
    phases: list[dict[str, Any]] = field(default_factory=list)
    truncated_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the evidence.

        Returns:
            A dictionary with dataclass members expanded recursively.
        """
        payload = asdict(self)
        payload["thread_groups"] = [asdict(group) for group in self.thread_groups]
        return payload


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def redact(text: str, enabled: bool = True) -> str:
    """Strip credentials and other secrets from gdb output.

    Args:
        text: Raw text to scrub.
        enabled: When ``False``, the text is returned unchanged.

    Returns:
        The scrubbed text.
    """
    if not enabled or not text:
        return text
    for pattern, replacement in REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


#: Default separator inserted between the retained head and tail of truncated
#: text. Shared with _shrink_text_field() so it can compute the true minimum
#: length truncate() can produce for a given floor (limit + len(marker)),
#: rather than looping forever comparing against a length it can never reach.
TRUNCATION_MARKER = "\n... [truncated] ..."


def truncate(text: str, limit: int, marker: str = TRUNCATION_MARKER) -> tuple[str, bool]:
    """Shorten text to a character budget, keeping head and tail.

    The head is favoured because the top stack frames matter most, but the tail
    is retained so that ``main`` and thread entry points remain visible.

    Args:
        text: Text to shorten.
        limit: Maximum number of characters to keep, excluding the marker.
        marker: Separator inserted between the retained head and tail.

    Returns:
        A tuple of the possibly-shortened text and whether truncation occurred.
    """
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.75)
    tail = limit - head
    return f"{text[:head]}{marker}{text[-tail:]}", True


def clean_gdb_noise(text: str) -> str:
    """Remove repetitive gdb boilerplate that carries no diagnostic value.

    Args:
        text: Raw gdb output.

    Returns:
        The output with download progress, licence banners and duplicate
        "Missing separate debuginfos" hints removed.
    """
    drop_prefixes = (
        "GNU gdb",
        "Copyright (C)",
        "License GPLv",
        "This is free software",
        "There is NO WARRANTY",
        "Type \"show copying\"",
        "Type \"show warranty\"",
        "This GDB was configured",
        "For bug reporting instructions",
        "Find the GDB manual",
        "For help, type \"help\"",
        "Type \"apropos word\"",
        "Reading symbols from",
        "[Thread debugging using",
        "Using host libthread_db",
        "Downloading",
        "[New LWP",
        "[New Thread",
        "[Current thread is",
        "warning: Memory read failed for corefile section",
    )
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(drop_prefixes)]
    collapsed: list[str] = []
    for line in lines:
        if collapsed and not line.strip() and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def normalise_frame_line(line: str) -> str:
    """Reduce a backtrace line to an address-independent signature.

    Args:
        line: A single ``#N  0x... in func (...) at file:line`` frame line.

    Returns:
        A normalised string suitable for grouping identical stacks.
    """
    line = re.sub(r"^#\d+\s+", "", line.strip())
    line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
    line = re.sub(r"\s+", " ", line)
    return line


# --------------------------------------------------------------------------- #
# gdb driving
# --------------------------------------------------------------------------- #


def split_sections(text: str) -> dict[str, str]:
    """Split gdb output on the section markers emitted between commands.

    Everything before the first marker is gdb's load banner and is discarded.

    Args:
        text: Raw gdb stdout containing section markers.

    Returns:
        A mapping of section name to that command's cleaned output.
    """
    matches = list(_MARKER_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = clean_gdb_noise(text[match.end():end])
    return sections


def find_gdb(explicit: str | None) -> str:
    """Locate the gdb executable.

    Args:
        explicit: A user-supplied path, or ``None`` to search ``PATH``.

    Returns:
        Path to a usable gdb executable.

    Raises:
        FileNotFoundError: If gdb cannot be found.
    """
    candidate = explicit or shutil.which("gdb")
    if not candidate or not Path(candidate).exists():
        raise FileNotFoundError(
            "gdb not found. Install it (dnf install gdb) or pass --gdb /path/to/gdb."
        )
    return candidate


def gdb_subprocess_env() -> dict[str, str]:
    """Return a copy of the environment safe for gdb's embedded Python.

    AnalysisBase commonly exports ``PYTHONHOME`` and ``PYTHONPATH`` for its
    Python runtime. EL9 gdb embeds a different Python version, so inheriting
    those variables can make gdb fail before it processes any commands. Other
    release variables, especially ``PATH`` and ``LD_LIBRARY_PATH``, are kept.
    """
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def gdb_version(gdb_path: str) -> str:
    """Return the first line of ``gdb --version`` using a sanitized environment.

    Args:
        gdb_path: Path to the gdb executable.

    Returns:
        The version banner, or ``"unknown"`` if it could not be read.
    """
    try:
        proc = subprocess.run(
            [gdb_path, "--version"], capture_output=True, text=True, timeout=30, check=False,
            env=gdb_subprocess_env(),
        )
        return proc.stdout.splitlines()[0].strip() if proc.stdout else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def _report_heartbeat(name: str, started: float, stop_event: threading.Event,
                      interval: float) -> None:
    """Print periodic "still running" messages while a gdb phase is blocked.

    ``subprocess.run(capture_output=True)`` buffers all of a phase's output
    until the process exits, so nothing is printed while gdb is working no
    matter how long that takes. On a large core, a single phase reloading and
    walking the whole core can easily run past a minute; this background
    thread is the only thing that distinguishes "still working" from "frozen"
    during that window.

    Args:
        name: Short identifier for the phase being watched.
        started: ``time.monotonic()`` value captured when the phase started.
        stop_event: Set by the caller once the phase has finished, so this
            loop exits promptly instead of waiting out its final interval.
        interval: Seconds between heartbeat messages.
    """
    while not stop_event.wait(interval):
        elapsed = time.monotonic() - started
        print(f"[*] gdb phase '{name}' still running ({elapsed:.0f}s elapsed)...", file=sys.stderr)


def run_gdb_phase(
    gdb_path: str,
    core_path: Path,
    exe_path: str | None,
    name: str,
    commands: Sequence[tuple[str, str]],
    timeout: int,
    progress: bool = True,
    detail: bool = False,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    extra_init: Sequence[str] = (),
) -> GdbPhaseResult:
    """Run one batch of gdb commands against the core file.

    Each phase is a separate process so that a single hanging command cannot
    take down the whole analysis.

    Args:
        gdb_path: Path to the gdb executable.
        core_path: Path to the core dump file.
        exe_path: Path to the matching ELF executable, or ``None``.
        name: Short identifier for this phase.
        commands: ``(section_name, gdb_command)`` pairs to execute in order.
        timeout: Per-phase timeout in seconds.
        progress: Whether to print a start/finish line for this phase.
        detail: Whether to also print periodic heartbeat messages while the
            phase is running. Has no effect if ``progress`` is ``False``.
        heartbeat_interval: Seconds between heartbeat messages when ``detail``
            is enabled.
        extra_init: Additional settings applied before the core is loaded.
            They must run as ``-iex``: gdb resolves separate debug files when
            an objfile is first read, so a ``set debug-file-directory`` issued
            afterwards has no effect on anything already loaded.

    Returns:
        A :class:`GdbPhaseResult` describing the invocation.
    """
    argv: list[str] = [gdb_path, "-q", "-nx", "-batch"]
    for setting in GDB_EARLY_INIT_COMMANDS:
        argv += ["-eiex", setting]
    for setting in GDB_INIT_COMMANDS:
        argv += ["-iex", setting]
    for setting in extra_init:
        argv += ["-iex", setting]
    if exe_path:
        argv.append(exe_path)
    argv += ["-c", str(core_path)]
    for section, command in commands:
        argv += ["-ex", f"echo \\n{SECTION_MARKER.format(name=section)}\\n", "-ex", command]

    if progress:
        print(f"[*] gdb phase '{name}' starting...", file=sys.stderr)

    started = time.monotonic()
    stop_event = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if progress and detail:
        heartbeat_thread = threading.Thread(
            target=_report_heartbeat, args=(name, started, stop_event, heartbeat_interval), daemon=True,
        )
        heartbeat_thread.start()

    timed_out = False
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False, env=gdb_subprocess_env()
        )
        stdout = proc.stdout or ""
        result = GdbPhaseResult(
            name=name,
            commands=[cmd for _, cmd in commands],
            sections=split_sections(stdout),
            stdout=stdout,
            stderr=proc.stderr or "",
            returncode=proc.returncode,
            duration_s=round(time.monotonic() - started, 2),
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        result = GdbPhaseResult(
            name=name,
            commands=[cmd for _, cmd in commands],
            stdout=exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=f"gdb phase '{name}' timed out after {timeout}s",
            returncode=-1,
            timed_out=True,
            duration_s=round(time.monotonic() - started, 2),
        )
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    if progress:
        status = f"timed out after {result.duration_s:.1f}s" if timed_out else f"completed in {result.duration_s:.1f}s"
        print(f"[*] gdb phase '{name}' {status}", file=sys.stderr)
    return result


# --------------------------------------------------------------------------- #
# Executable resolution
# --------------------------------------------------------------------------- #


def executable_from_auxv(
    gdb_path: str,
    core_path: Path,
    timeout: int,
    progress: bool = True,
    detail: bool = False,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
) -> str | None:
    """Recover the executable path from the core's ``AT_EXECFN`` auxiliary vector entry.

    This is the most portable source: gdb can read it from a bare core with no
    executable loaded, and it does not depend on ``readelf`` being able to decode
    64-bit notes. It records the path exactly as passed to ``execve``.

    Args:
        gdb_path: Path to the gdb executable.
        core_path: Path to the core dump file.
        timeout: gdb timeout in seconds.
        progress: Whether to print a start/finish line for the gdb probe.
        detail: Whether to also print heartbeat messages during the probe.
        heartbeat_interval: Seconds between heartbeat messages when ``detail``
            is enabled.

    Returns:
        The recorded executable path, or ``None`` if it could not be read.
    """
    result = run_gdb_phase(
        gdb_path, core_path, None, "auxv", [("auxv", "info auxv")], timeout,
        progress=progress, detail=detail, heartbeat_interval=heartbeat_interval,
    )
    match = re.search(r"AT_EXECFN\s+File name of executable\s+0x[0-9a-fA-F]+\s+\"(.+?)\"", result.stdout)
    return match.group(1) if match else None


def executable_from_nt_file(core_path: Path) -> str | None:
    """Recover the executable path from the core file's NT_FILE note.

    This is the most reliable source: the kernel records absolute paths for all
    file-backed mappings, and the first mapping at page offset zero is the main
    executable. It survives cases where ``argv[0]`` was relative or truncated,
    which matters for CVMFS-hosted Python interpreters running ``athena.py``.

    Args:
        core_path: Path to the core dump file.

    Returns:
        The absolute executable path, or ``None`` if it could not be recovered.
    """
    readelf = shutil.which("eu-readelf") or shutil.which("readelf")
    if not readelf:
        return None
    try:
        proc = subprocess.run(
            [readelf, "-n", str(core_path)], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None

    lines = proc.stdout.splitlines()
    in_nt_file = False
    triple = re.compile(r"^\s+0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+\s+(?:0x)?([0-9a-fA-F]+)\s*$")
    for index, line in enumerate(lines):
        if "NT_FILE" in line:
            in_nt_file = True
            continue
        if not in_nt_file:
            continue
        if line.strip().startswith("Owner") or "NT_" in line:
            break
        match = triple.match(line)
        if match and int(match.group(1), 16) == 0 and index + 1 < len(lines):
            candidate = lines[index + 1].strip()
            if candidate.startswith("/"):
                return candidate
    return None


def parse_generated_by(text: str) -> str | None:
    """Extract the ``Core was generated by`` command line from gdb output.

    Args:
        text: gdb output to search.

    Returns:
        The recorded command line, or ``None`` if absent.
    """
    match = re.search(r"Core was generated by [`'\"](.*?)['\"]\.", text, re.S)
    return match.group(1).strip() if match else None


def _argv0_from_command_line(command_line: str | None) -> str | None:
    """Extract argv[0] from a recorded command line.

    Args:
        command_line: The ``Core was generated by`` string, if any.

    Returns:
        The first token, or ``None`` if there is none.
    """
    parts = (command_line or "").split()
    return parts[0] if parts else None


def _existing_path(candidate: str | None) -> tuple[str | None, bool]:
    """Resolve a recorded executable path to something present on this host.

    An absolute recorded path is treated as a build identity and must exist
    exactly. Searching ``PATH`` for its basename is deliberately *not* attempted:
    if a core references ``/cvmfs/.../bin/python`` and CVMFS is not mounted,
    silently substituting the system interpreter would give gdb a different build
    and yield plausible but wrong symbols. Only bare names and relative paths,
    which the OS would itself have resolved via ``PATH``, are searched for.

    Args:
        candidate: A path recorded in the core, possibly relative or stale.

    Returns:
        A tuple of the resolved path (or ``None``) and whether resolution
        involved a search that the caller should warn about.
    """
    if not candidate:
        return None, False
    path = Path(candidate)
    if path.is_file():
        return str(path.resolve()), False
    if path.is_absolute():
        return None, False
    local = Path.cwd() / path.name
    if local.is_file():
        return str(local.resolve()), True
    found = shutil.which(path.name)
    return (found, True) if found else (None, False)


def resolve_executable(gdb_path: str, core_path: Path, explicit: str | None,
                       probe_output: str, timeout: int, progress: bool = True,
                       detail: bool = False,
                       heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL) -> dict[str, Any]:
    """Determine which ELF binary gdb should load alongside the core.

    Resolution order is ``--exe``, then ``AT_EXECFN`` from the auxiliary vector,
    then the core's NT_FILE note, then the recorded command line. Failed automatic
    candidates are recorded as attempts but do not become user-facing warnings if
    a later candidate resolves successfully. This avoids stale warnings such as a
    truncated ``AT_EXECFN`` path surviving after command-line resolution succeeds.
    """
    persistent_notes: list[str] = []
    failed_notes: list[str] = []
    attempts: list[dict[str, Any]] = []

    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.suffix == ".py":
            persistent_notes.append(
                f"--exe pointed at a Python script ({explicit}). gdb needs the interpreter ELF binary, "
                "not the script; ignoring it and attempting automatic resolution."
            )
            attempts.append({"source": "--exe", "recorded": explicit, "resolved": False, "reason": "python-script"})
        elif not explicit_path.is_file():
            persistent_notes.append(f"--exe path does not exist: {explicit}")
            attempts.append({"source": "--exe", "recorded": explicit, "resolved": False, "reason": "missing"})
        else:
            resolved = str(explicit_path.resolve())
            attempts.append({"source": "--exe", "recorded": explicit, "resolved": True, "path": resolved})
            return {"path": resolved, "resolved": True, "source": "--exe",
                    "recorded": None, "notes": persistent_notes, "attempts": attempts}

    candidates: list[tuple[str, str | None]] = [
        ("AT_EXECFN", executable_from_auxv(
            gdb_path, core_path, timeout, progress=progress, detail=detail, heartbeat_interval=heartbeat_interval)),
        ("NT_FILE", executable_from_nt_file(core_path)),
        ("command-line", _argv0_from_command_line(parse_generated_by(probe_output))),
    ]
    for source, recorded in candidates:
        if not recorded:
            attempts.append({"source": source, "recorded": None, "resolved": False, "reason": "not-found-in-core"})
            continue
        resolved, searched = _existing_path(recorded)
        if resolved:
            attempts.append({"source": source, "recorded": recorded, "resolved": True,
                             "path": resolved, "searched": searched})
            notes = list(persistent_notes)
            if searched:
                notes.append(
                    f"Executable recorded as '{recorded}' was not found directly and was matched to "
                    f"'{resolved}' by search. Verify it is the same build; a mismatched binary yields "
                    "plausible but wrong symbols."
                )
            return {"path": resolved, "resolved": True, "source": source,
                    "recorded": recorded, "notes": notes, "attempts": attempts}
        attempts.append({"source": source, "recorded": recorded, "resolved": False, "reason": "missing"})
        failed_notes.append(
            f"The core references executable '{recorded}' ({source}), which is not present on this host. "
            "No substitute was used, because a different build would produce misleading symbols. "
            "Re-run where that path is available (for ATLAS jobs, with the matching CVMFS release mounted), "
            "or pass the correct binary with --exe."
        )

    notes = persistent_notes + failed_notes
    notes.append("No executable could be resolved. Backtraces will be unsymbolised and largely uninterpretable.")
    return {"path": None, "resolved": False, "source": "none", "recorded": None,
            "notes": notes, "attempts": attempts}


CRITICAL_BUILD_ID_BASENAMES = frozenset({"libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2"})


def collect_runtime_environment() -> dict[str, Any]:
    """Collect a small deterministic description of the current analysis OS."""
    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    except OSError:
        pass

    glibc = "unknown"
    ldd = shutil.which("ldd")
    if ldd:
        try:
            proc = subprocess.run([ldd, "--version"], capture_output=True, text=True, timeout=15, check=False)
            first = (proc.stdout or proc.stderr or "").splitlines()
            if first:
                glibc = first[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "execution_backend": "local",
        "os": os_release.get("PRETTY_NAME") or os_release.get("NAME") or "unknown",
        "os_id": os_release.get("ID"),
        "os_version": os_release.get("VERSION_ID"),
        "glibc": glibc,
    }


def parse_eu_unstrip_modules(text: str) -> list[dict[str, str]]:
    """Parse ``eu-unstrip -n --core`` module lines into compact records."""
    modules: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or "+0x" not in parts[0] or "@0x" not in parts[1]:
            continue
        build_id = parts[1].split("@", 1)[0]
        path = next((part for part in parts[2:] if part.startswith("/")), "")
        name = Path(path).name if path else parts[-1]
        modules.append({"build_id": build_id, "path": path, "name": name, "mapping": parts[0]})
    return modules


def file_build_id(path: str) -> str | None:
    """Read an ELF Build ID from a file using an available readelf implementation."""
    if not path or not Path(path).is_file():
        return None
    tool = shutil.which("eu-readelf") or shutil.which("readelf")
    if not tool:
        return None
    try:
        proc = subprocess.run([tool, "-n", path], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", (proc.stdout or "") + "\n" + (proc.stderr or ""))
    return match.group(1).lower() if match else None


def collect_build_id_evidence(core_path: Path, exe_path: str | None) -> tuple[dict[str, Any], str]:
    """Collect core module Build IDs and compare key files on the analysis host."""
    tool = shutil.which("eu-unstrip")
    if not tool:
        return {"available": False, "reason": "eu-unstrip not found"}, ""
    try:
        proc = subprocess.run(
            [tool, "-n", "--core", str(core_path)], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"eu-unstrip failed: {exc}"}, ""
    raw = (proc.stdout or "") + (proc.stderr or "")
    modules = parse_eu_unstrip_modules(proc.stdout or "")
    selected: list[dict[str, Any]] = []
    exe_resolved = str(Path(exe_path).resolve()) if exe_path and Path(exe_path).is_file() else exe_path
    for module in modules:
        path = module.get("path", "")
        name = module.get("name", "")
        is_executable = bool(
            exe_path and path
            and (path == exe_path or (exe_resolved and str(Path(path).resolve()) == exe_resolved))
        )
        if name not in CRITICAL_BUILD_ID_BASENAMES and not is_executable:
            continue
        disk_id = file_build_id(path)
        core_id = module["build_id"].lower()
        selected.append({
            "name": name,
            "path": path,
            "role": "executable" if is_executable else "system-library",
            "core_build_id": core_id,
            "file_build_id": disk_id,
            "file_present": bool(path and Path(path).is_file()),
            "match": (disk_id == core_id) if disk_id else None,
        })
    mismatches = [item for item in selected if item.get("match") is False]
    unavailable = [item for item in selected if item.get("match") is None]
    return {
        "available": proc.returncode == 0 or bool(modules),
        "tool": tool,
        "module_count": len(modules),
        "checked": selected,
        "mismatch_count": len(mismatches),
        "unverified_count": len(unavailable),
        "coverage": "verified" if selected and not mismatches and not unavailable else ("partial" if selected else "unverified"),
        "raw_excerpt": raw[:2000] if len(modules) < 4 else "",
    }, raw


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #


def parse_signal(text: str) -> str | None:
    """Find the terminating signal reported by gdb.

    Args:
        text: gdb output to search.

    Returns:
        The signal name such as ``"SIGSEGV"``, or ``None``.
    """
    match = re.search(r"(?:Program terminated with signal|It stopped with signal)\s+(SIG[A-Z0-9]+)", text)
    return match.group(1) if match else None


def parse_thread_count(text: str) -> int | None:
    """Count threads from ``info threads`` output.

    Args:
        text: gdb output containing an ``info threads`` table.

    Returns:
        The number of threads, or ``None`` if the table was not found.
    """
    count = len(re.findall(r"^[\s*]+\d+\s+(?:Thread|LWP|process|Process)\b", text, re.M))
    return count or None


def collect_warnings(text: str) -> list[str]:
    """Detect conditions that degrade the quality of the evidence.

    Args:
        text: Combined gdb stdout and stderr.

    Returns:
        A list of human-readable warnings.
    """
    warnings: list[str] = []
    checks = (
        ("is truncated", "The core file is truncated (likely a `ulimit -c` cap). Deep frames may be missing or wrong."),
        ("core file may not match", "gdb reports the core may not match the executable. Symbols may be misleading."),
        ("no debugging symbols found", "The executable was built or shipped without debug symbols."),
        ("Missing separate debuginfo", "Separate debuginfo packages are missing for one or more libraries."),
        ("Cannot access memory", "Parts of the process memory are unreadable in this core."),
    )
    for needle, message in checks:
        if needle in text and message not in warnings:
            warnings.append(message)
    return warnings


def _classify_thread_stack(backtrace: str) -> str:
    """Classify a thread stack as active, blocked, or genuinely idle.

    A top-level futex/condition-variable wait does *not* by itself mean a thread
    is uninteresting. In hang cores, the thread we care about is often blocked
    in exactly such a primitive while a deeper frame shows a shutdown handshake,
    mutex acquisition, timeout handler, or other meaningful operation.

    Returns:
        ``"active"`` when the top frames are not a known wait, ``"blocked"``
        when they are waiting in a meaningful blocking context, otherwise
        ``"idle"`` for a benign parked worker.
    """
    wait_markers = (
        "pthread_cond_wait", "pthread_cond_timedwait", "__futex_abstimed_wait",
        "epoll_wait", "poll (", "ppoll", "select (", "nanosleep", "sem_wait",
        "sigwait", "accept (", "read (", "recvmsg", "XrdSysCondVar::Wait",
    )
    frames = [line for line in backtrace.splitlines() if line.lstrip().startswith("#")]
    head = "\n".join(frames[:3])
    if not any(marker in head for marker in wait_markers):
        return "active"

    # These contexts make a blocking primitive diagnostically meaningful rather
    # than a benign worker wait. The list intentionally mixes generic lock/exit
    # patterns with XRootD shutdown/timeout operations seen in ATLAS jobs.
    blocking_context_markers = (
        "pthread_mutex_lock", "__lll_lock_wait", "std::mutex::lock",
        "::Lock(", "::SendCmd(", "::Stop(", "::Finalize(",
        "::ShutdownEvents(", "::ForceDisconnect(", "::ForceError(",
        "::OnReadTimeout(", "Py_Exit (", "__run_exit_handlers",
    )
    if any(marker in backtrace for marker in blocking_context_markers):
        return "blocked"
    return "idle"


def _is_idle_stack(backtrace: str) -> bool:
    """Return whether a stack is a genuinely benign parked-worker wait.

    Kept as a small compatibility wrapper for callers/tests that used the
    original boolean classifier.
    """
    return _classify_thread_stack(backtrace) == "idle"


def _thread_context_frame(backtrace: str) -> str:
    """Return the most useful single frame for a compact thread summary."""
    frames = [line.strip() for line in backtrace.splitlines() if line.lstrip().startswith("#")]
    if not frames:
        return "?"
    preferred = (
        "::OnReadTimeout(", "::ForceDisconnect(", "::ShutdownEvents(",
        "::StreamMutex::Lock(", "::SendCmd(", "::Stop(", "::Finalize(",
        "Py_Exit (", "pthread_mutex_lock", "__lll_lock_wait",
    )
    for line in frames:
        if any(marker in line for marker in preferred):
            return line
    generic = (
        "__futex_abstimed_wait", "pthread_cond_wait", "XrdSysCondVar::Wait",
        "start_thread", "clone3",
    )
    for line in frames[1:]:
        if not any(marker in line for marker in generic):
            return line
    return frames[0]


def _frame_number(frame_line: str) -> int | None:
    """Extract a numeric gdb frame index from a rendered ``#N`` frame line."""
    match = re.match(r"^#(\d+)\b", frame_line.strip())
    return int(match.group(1)) if match else None


def select_targeted_threads(thread_groups: list[ThreadGroup], max_targets: int) -> list[dict[str, Any]]:
    """Select representative non-idle threads for focused frame inspection.

    One representative is chosen from each interesting thread group.  The
    context-frame heuristic is the same one used by the compact report, so the
    detailed evidence explains exactly the frame the operator sees highlighted.
    """
    if max_targets <= 0:
        return []
    targets: list[dict[str, Any]] = []
    for group in thread_groups:
        if group.state == "idle" or not group.thread_ids:
            continue
        context = _thread_context_frame(group.backtrace)
        frame_no = _frame_number(context)
        if frame_no is None:
            continue
        targets.append({
            "thread_id": group.thread_ids[0],
            "state": group.state,
            "frame": frame_no,
            "context": context,
        })
        if len(targets) >= max_targets:
            break
    return targets


def _build_targeted_phase(targets: list[dict[str, Any]], include_locals: bool) -> list[tuple[str, str]]:
    """Build one batched gdb phase for the selected thread/frame pairs."""
    commands: list[tuple[str, str]] = []
    for index, target in enumerate(targets, start=1):
        prefix = f"target_{index}"
        commands.extend([
            (f"{prefix}_thread", f"thread {target['thread_id']}"),
            (f"{prefix}_frame_select", f"frame {target['frame']}"),
            (f"{prefix}_frame", "info frame"),
            (f"{prefix}_args", "info args"),
        ])
        if include_locals:
            commands.append((f"{prefix}_locals", "info locals"))
    return commands


def summarise_targeted_threads(targets: list[dict[str, Any]], sections: dict[str, str],
                               redact_enabled: bool) -> list[dict[str, Any]]:
    """Attach bounded ``info frame/args/locals`` output to targeted thread metadata.

    ``info sharedlibrary`` saying ``Yes`` only means that GDB read symbols for
    the DSO; an optimized function can still lack usable argument/local DWARF.
    Record that distinction explicitly so the report does not imply that the
    whole library is unsymbolized when only frame-local detail is unavailable.
    """
    summaries: list[dict[str, Any]] = []
    unavailable_marker = "No symbol table info available."
    for index, target in enumerate(targets, start=1):
        prefix = f"target_{index}"
        item = dict(target)
        detail_available = False
        for key, limit_name in (("frame_info", "targeted_frame"),
                                ("args", "targeted_args"),
                                ("locals", "targeted_locals")):
            section_key = f"{prefix}_{'frame' if key == 'frame_info' else key}"
            body = sections.get(section_key, "").strip()
            if body:
                item[key], _ = truncate(redact(body, redact_enabled), SECTION_LIMITS[limit_name])
                if key in {"args", "locals"} and unavailable_marker not in body:
                    detail_available = True
        item["frame_details_available"] = detail_available
        summaries.append(item)
    return summaries


JOB_LOG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("termination", re.compile(
        r"\b(SystemExit|SIGTERM|SIGQUIT|SIGKILL|killed|kill signal|payload.*(?:exit|finished)|exit code|walltime|looping job)\b",
        re.I,
    )),
    # Runtime I/O evidence only.  Generic mentions such as `lsetup xrootd` or
    # `root://` entries in a catalog describe configuration/input locations,
    # not an observed XRootD failure in the payload.
    ("xrootd", re.compile(
        r"(?:\bXrd(?:Cl|Sys)::|\bread timeout\b|\boperation expired\b|\bforce(?:d)? disconnect\b|"
        r"\bXRootD\b.*\b(?:error|timeout|fail(?:ed|ure)?)\b)",
        re.I,
    )),
    # Severity words are intentionally case-sensitive here.  Payload text can
    # legitimately contain phrases such as "without any error state set";
    # treating every lower-case word "error" as a log severity creates false
    # positives.  Exception/traceback markers remain case-insensitive.
    ("error", re.compile(
        r"(?:^|[\s|])(?:FATAL|ERROR)(?=[\s:|]|$)|(?:^|[\s|])(?:Fatal|Error):|(?i:\b(?:exception|traceback)\b)",
    )),
    ("completion", re.compile(
        r"\bworker finished successfully\b|\bcurrent job status:\s*\d+\s+success,\s*0\s+failure|"
        r"\bMoving the analysis root file\b|\brenaming .*output\.root\b",
        re.I,
    )),
    ("progress", re.compile(
        r"\b(events? processed|processed .*events?|accepted \d+ out of \d+ events|"
        r"finali[sz](?:e|ing|ation)|closing .*file|output .*file|write .*output)\b",
        re.I,
    )),
)


def _log_role(path: Path, job_dir: Path) -> str:
    """Return a stable evidence role for a discovered payload/job log."""
    try:
        rel = path.relative_to(job_dir)
    except ValueError:
        rel = path
    name = path.name.lower()
    if len(rel.parts) == 1 and name == "payload.stdout":
        return "payload-stdout"
    if len(rel.parts) == 1 and name == "payload.stderr":
        return "payload-stderr"
    if len(rel.parts) == 1 and name == "pilotlog.txt":
        return "pilot"
    if rel.parts and rel.parts[0] == "workDir":
        return "workdir-log"
    if "payload" in name:
        return "payload-log"
    return "other"


def _job_log_rank(path: Path, job_dir: Path, core_mtime: float | None = None) -> tuple[int, float, str]:
    """Rank payload streams and recent workDir logs ahead of incidental files."""
    role = _log_role(path, job_dir)
    role_rank = {
        "payload-stdout": 0,
        "payload-stderr": 0,
        "payload-log": 1,
        "workdir-log": 2,
        "pilot": 3,
        "other": 9,
    }.get(role, 9)
    recency = float("inf")
    if core_mtime is not None:
        try:
            recency = abs(path.stat().st_mtime - core_mtime)
        except OSError:
            pass
    return (role_rank, recency, str(path))


def _looks_like_log_file(path: Path) -> bool:
    """Conservatively identify runtime log artifacts by name/suffix.

    A bare ``.txt`` suffix is not enough: PanDA work directories commonly
    contain input lists, path/configuration files, and other static text.
    Arbitrarily named text logs can still be supplied explicitly with
    ``--job-log``.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".log", ".out", ".err", ".stdout", ".stderr"}:
        return True
    return any(token in name for token in ("log", "stdout", "stderr", "trace", "debug", "report"))


#: Filename prefixes this analyzer itself writes into a job directory. They are
#: excluded from discovery so a previous run's artifacts are never mistaken for
#: payload evidence.
GENERATED_LOG_PREFIXES: tuple[str, ...] = ("core-analysis", ".core_dump_analyzer_")


def _explicit_job_logs(job_dir: Path, explicit: Sequence[str]) -> list[Path]:
    """Resolve explicitly requested job logs against the job directory.

    Args:
        job_dir: PanDA job directory used to resolve relative paths.
        explicit: Raw ``--job-log`` values.

    Returns:
        Resolved paths for the values that exist as files.
    """
    resolved: list[Path] = []
    for raw in explicit:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = job_dir / path
        if path.is_file():
            resolved.append(path.resolve())
    return resolved


def _payload_stream_logs(job_dir: Path) -> list[Path]:
    """Collect the canonical payload streams at the job-directory root.

    Args:
        job_dir: PanDA job directory.

    Returns:
        Resolved paths for ``payload.stdout``/``payload.stderr`` and any other
        log-like ``payload*`` file at the job root.
    """
    found: list[Path] = []
    for name in ("payload.stdout", "payload.stderr"):
        path = job_dir / name
        if path.is_file():
            found.append(path.resolve())
    for path in job_dir.glob("payload*"):
        if path.is_file() and _looks_like_log_file(path):
            found.append(path.resolve())
    return found


def _latest_payload_mtime(job_dir: Path) -> float | None:
    """Return the newest modification time across the non-empty payload streams.

    This anchors the hang-mode recency window.  It is deliberately the payload
    streams rather than the core's own mtime: for a looping job the payload has
    by definition been silent for a long time before the core is captured, so a
    window measured from the core would discard the very workDir logs that were
    active when the payload stopped.

    Empty streams are ignored — a zero-length ``payload.stderr`` carries no
    activity information and its mtime would move the window arbitrarily.

    Args:
        job_dir: PanDA job directory.

    Returns:
        The newest payload-stream mtime, or ``None`` when neither stream exists
        with content.
    """
    mtimes: list[float] = []
    for payload_name in ("payload.stdout", "payload.stderr"):
        payload_path = job_dir / payload_name
        try:
            if payload_path.is_file() and payload_path.stat().st_size > 0:
                mtimes.append(payload_path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes, default=None)


def _workdir_log_is_relevant(path: Path, rel_work: Path, failure_mode: str,
                             latest_payload_mtime: float | None) -> bool:
    """Decide whether a log-like file below ``workDir`` is payload evidence.

    Args:
        path: Candidate file.
        rel_work: The candidate's path relative to ``workDir``.
        failure_mode: Resolved analysis mode; ``"hang"`` enables the recency
            window.
        latest_payload_mtime: Anchor for the recency window, or ``None``.

    Returns:
        ``True`` when the file should be scanned.
    """
    # The unpacked user release can live below workDir/usr and may contain
    # thousands of build/configuration .txt files. Those are not runtime logs
    # and must not crowd payload-created files out of the bounded set.
    if rel_work.parts and rel_work.parts[0] == "usr":
        return False
    if path.name.lower().startswith(GENERATED_LOG_PREFIXES):
        return False
    if failure_mode == "hang" and latest_payload_mtime is not None:
        try:
            cutoff = latest_payload_mtime - DEFAULT_HANG_WORKDIR_LOG_RECENCY_S
            if path.stat().st_mtime < cutoff:
                return False
        except OSError:
            return False
    return True


def _workdir_job_logs(job_dir: Path, failure_mode: str) -> list[Path]:
    """Collect log-like payload artifacts below ``workDir``.

    Searched recursively but only log-like text artifacts are retained; build
    products and payload data files are intentionally ignored.  For hang
    analysis, files that were already stale when the payload fell silent are
    dropped too: a job tarball can contain old reference/test logs copied into
    ``workDir``, and their ERROR lines are not evidence about this execution.

    Args:
        job_dir: PanDA job directory.
        failure_mode: Resolved analysis mode.

    Returns:
        Resolved paths for the retained files.
    """
    work_dir = job_dir / "workDir"
    if not work_dir.is_dir():
        return []
    latest_payload_mtime = _latest_payload_mtime(job_dir)
    found: list[Path] = []
    for path in work_dir.rglob("*"):
        if not path.is_file() or not _looks_like_log_file(path):
            continue
        try:
            rel_work = path.relative_to(work_dir)
        except ValueError:
            continue
        if _workdir_log_is_relevant(path, rel_work, failure_mode, latest_payload_mtime):
            found.append(path.resolve())
    return found


def discover_job_logs(job_dir: Path, explicit: Sequence[str] | None = None,
                      max_files: int = DEFAULT_MAX_JOB_LOG_FILES,
                      failure_mode: str = "auto",
                      core_mtime: float | None = None) -> list[Path]:
    """Discover bounded payload-centric logs for a core-analysis failure mode.

    For looping/hang jobs the pilot's own log is deliberately excluded from
    automatic discovery: pilot termination records describe what the pilot did
    *after* deciding the payload was looping, not what the payload was doing
    before the core was captured.  The primary automatic sources are the
    payload stdout/stderr streams plus user/payload-generated log-like files
    below ``workDir``.  ``--job-log`` remains an explicit escape hatch for any
    other file, including ``pilotlog.txt``.

    Args:
        job_dir: PanDA job directory to search.
        explicit: Explicit ``--job-log`` values; when supplied, automatic
            discovery is skipped entirely.
        max_files: Upper bound on the number of returned paths.
        failure_mode: Resolved analysis mode (``"hang"``, ``"crash"`` or
            ``"auto"``).
        core_mtime: Core-capture time used to rank candidates by recency.

    Returns:
        Ranked, de-duplicated paths, truncated to *max_files*.
    """
    if explicit:
        candidates = _explicit_job_logs(job_dir, explicit)
    else:
        candidates = _payload_stream_logs(job_dir)
        candidates += _workdir_job_logs(job_dir, failure_mode)
        # Pilot evidence can still be useful for non-looping failures, but not
        # for an explicitly diagnosed hang/loop.
        if failure_mode != "hang":
            pilot = job_dir / "pilotlog.txt"
            if pilot.is_file():
                candidates.append(pilot.resolve())

    unique = list(dict.fromkeys(candidates))
    unique.sort(key=lambda path: _job_log_rank(path, job_dir, core_mtime))
    return unique[:max(0, max_files)]


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration compactly for evidence-only reports."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes:02d}m" if hours else f"{minutes}m")
    parts.append(f"{secs:02d}s" if hours or minutes else f"{secs}s")
    return " ".join(parts)


#: Roles whose end-of-output tail is preserved independently of keyword matching.
_TAIL_ROLES: frozenset[str] = frozenset(
    {"payload-stdout", "payload-stderr", "payload-log", "workdir-log"}
)


def _read_job_log_window(
    path: Path, max_bytes: int,
) -> tuple[os.stat_result, int, bytes] | None:
    """Read the trailing scan window of a job log and the line offset it starts at.

    Large logs are searched in a tail window because the last payload activity
    before a loop is usually the most useful.  The skipped prefix is still read
    so that reported line numbers refer to the real file, not to the window.

    Args:
        path: Log file to read.
        max_bytes: Size of the trailing window in bytes.

    Returns:
        Tuple of ``(stat_result, line_base, scanned_bytes)``, or ``None`` when
        the file could not be stat'ed or read.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    window_start = max(0, stat.st_size - max_bytes)
    line_base = 0
    try:
        with path.open("rb") as handle:
            remaining = window_start
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                line_base += chunk.count(b"\n")
                remaining -= len(chunk)
            if window_start:
                partial = handle.readline()
                line_base += 1
                scanned = handle.read(max(0, max_bytes - len(partial)))
            else:
                scanned = handle.read(max_bytes)
    except OSError:
        return None
    return stat, line_base, scanned


def _job_log_meta(path: Path, job_dir: Path, stat: os.stat_result,
                  scanned_bytes: int, max_bytes: int,
                  core_mtime: float | None) -> dict[str, Any]:
    """Build the per-file evidence metadata record for a scanned job log.

    Args:
        path: Log file.
        job_dir: Job directory, used to derive the relative path.
        stat: Stat result for *path*.
        scanned_bytes: Number of bytes actually read.
        max_bytes: Scan-window size, used to decide whether the read was
            truncated.
        core_mtime: Core-capture time, or ``None``.

    Returns:
        Metadata dict including ``mtime_delta_from_core_s`` when *core_mtime*
        is known.  That delta is what makes the payload-silence observation
        possible, so the file's original modification time must be preserved
        by whatever put it on disk.
    """
    try:
        rel = str(path.relative_to(job_dir))
    except ValueError:
        rel = path.name
    window_start = max(0, stat.st_size - max_bytes)
    meta: dict[str, Any] = {
        "path": str(path),
        "relative_path": rel,
        "role": _log_role(path, job_dir),
        "size_bytes": stat.st_size,
        "scanned_bytes": scanned_bytes,
        "window": "tail" if window_start else "full",
        "truncated": bool(window_start),
        "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    }
    if core_mtime is not None:
        meta["mtime_delta_from_core_s"] = round(stat.st_mtime - core_mtime, 3)
    return meta


def _job_log_tail(text: str, line_base: int, role: str, tail_lines: int,
                  redact_enabled: bool) -> list[dict[str, Any]]:
    """Collect the last non-empty lines of a runtime log.

    Preserved independently of keyword matching: for looping jobs the true end
    of output is often more diagnostic than the latest periodic
    "Processed N events" line, because shutdown/finalize messages can follow
    the last progress counter.

    Args:
        text: Decoded scan window.
        line_base: Line number the window starts at.
        role: Evidence role of the file.
        tail_lines: Number of trailing non-empty lines to keep; 0 disables.
        redact_enabled: Whether to scrub credentials from the retained lines.

    Returns:
        List of ``{"line", "text"}`` records, empty when disabled or when the
        role carries no runtime output.
    """
    if tail_lines <= 0 or role not in _TAIL_ROLES:
        return []
    tail: deque[dict[str, Any]] = deque(maxlen=tail_lines)
    for relative_line, line in enumerate(text.splitlines(), start=1):
        clean = line.strip()
        if not clean:
            continue
        bounded, _ = truncate(redact(clean, redact_enabled), SECTION_LIMITS["job_log_line"])
        tail.append({"line": line_base + relative_line, "text": bounded})
    return list(tail)


def _job_log_matches(text: str, line_base: int, path: Path, rel: str, role: str,
                     per_file_limit: int,
                     redact_enabled: bool) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Match a scan window against the job-log patterns, keeping the most recent.

    Retention is bounded per file so one noisy log cannot evict evidence from
    every other payload-generated log.  The returned counts describe everything
    that matched, including records the bound discarded, so callers can report
    that the limit was reached rather than silently under-reporting.

    Args:
        text: Decoded scan window.
        line_base: Line number the window starts at.
        path: Log file, recorded on each match.
        rel: Path of *path* relative to the job directory.
        role: Evidence role of the file.
        per_file_limit: Maximum retained matches for this file.
        redact_enabled: Whether to scrub credentials from retained lines.

    Returns:
        Tuple of ``(retained_matches, found_counts_by_category, total_found)``.
    """
    recent: deque[dict[str, Any]] = deque(maxlen=per_file_limit)
    found_counts: dict[str, int] = {}
    total_found = 0
    for relative_line, line in enumerate(text.splitlines(), start=1):
        clean = line.strip()
        if not clean:
            continue
        for category, pattern in JOB_LOG_PATTERNS:
            if not pattern.search(clean):
                continue
            bounded, _ = truncate(
                redact(clean, redact_enabled), SECTION_LIMITS["job_log_line"]
            )
            recent.append({
                "file": str(path),
                "relative_file": rel,
                "role": role,
                "line": line_base + relative_line,
                "category": category,
                "text": bounded,
            })
            found_counts[category] = found_counts.get(category, 0) + 1
            total_found += 1
            break
    return list(recent), found_counts, total_found


def collect_job_log_evidence(job_dir: Path, explicit: Sequence[str] | None = None,
                             max_files: int = DEFAULT_MAX_JOB_LOG_FILES,
                             max_matches: int = DEFAULT_MAX_JOB_LOG_MATCHES,
                             max_bytes: int = DEFAULT_MAX_JOB_LOG_BYTES,
                             tail_lines: int = DEFAULT_JOB_LOG_TAIL_LINES,
                             redact_enabled: bool = True,
                             core_mtime: float | None = None,
                             failure_mode: str = "auto") -> dict[str, Any]:
    """Extract bounded payload/runtime evidence near the captured core state.

    Hang-mode collection is payload-centric: canonical payload stdout/stderr
    and log-like files under ``workDir`` are scanned automatically, while the
    pilot log is excluded unless supplied explicitly.  Large logs are searched
    in a tail window because the last payload activity before a loop is usually
    the most useful.  Matches are bounded per file so one noisy log cannot
    evict evidence from all other payload-generated logs.
    """
    files = discover_job_logs(
        job_dir, explicit=explicit, max_files=max_files,
        failure_mode=failure_mode, core_mtime=core_mtime,
    )
    result: dict[str, Any] = {
        "available": bool(files),
        "job_dir": str(job_dir),
        "profile": "payload-centric" if failure_mode == "hang" else "general",
        "pilotlog_default_excluded": failure_mode == "hang" and not explicit,
        "files": [],
        "matches": [],
        "category_counts": {},
        "category_counts_found": {},
        "tail_lines_per_file": max(0, tail_lines),
    }
    if failure_mode == "hang" and not explicit:
        result["workdir_recency_window_s"] = DEFAULT_HANG_WORKDIR_LOG_RECENCY_S
    if not files or max_matches <= 0:
        result["match_limit_reached"] = False
        return result

    per_file_limit = max(4, (max_matches + len(files) - 1) // len(files))
    all_matches: list[dict[str, Any]] = []
    found_counts: dict[str, int] = {}
    total_found = 0

    for path in files:
        read = _read_job_log_window(path, max_bytes)
        if read is None:
            continue
        stat, line_base, scanned = read

        meta = _job_log_meta(path, job_dir, stat, len(scanned), max_bytes, core_mtime)
        result["files"].append(meta)

        text = scanned.decode("utf-8", errors="replace")
        role = str(meta["role"])

        tail = _job_log_tail(text, line_base, role, tail_lines, redact_enabled)
        if tail:
            meta["tail"] = tail

        matches, file_found_counts, file_total_found = _job_log_matches(
            text, line_base, path, str(meta["relative_path"]), role,
            per_file_limit, redact_enabled,
        )
        for category, count in file_found_counts.items():
            found_counts[category] = found_counts.get(category, 0) + count
        total_found += file_total_found
        all_matches.extend(matches)

    retained = all_matches[:max_matches]
    retained_counts: dict[str, int] = {}
    for item in retained:
        category = str(item.get("category", "other"))
        retained_counts[category] = retained_counts.get(category, 0) + 1
    result["matches"] = retained
    result["category_counts"] = retained_counts
    result["category_counts_found"] = found_counts
    result["matched_lines_found"] = total_found
    result["match_limit_reached"] = total_found > len(retained)

    # Filesystem modification time is valuable deterministic evidence for a
    # looping job even when the payload log itself has no timestamps.  Record
    # the latest observed payload-stream write before the core and the most
    # recent retained progress line from that stream.
    payload_files = [
        meta for meta in result["files"]
        if meta.get("role") in {"payload-stdout", "payload-stderr", "payload-log"}
        and meta.get("size_bytes", 0) > 0
        and isinstance(meta.get("mtime_delta_from_core_s"), (int, float))
        and meta["mtime_delta_from_core_s"] <= 0
    ]
    if payload_files:
        latest = max(payload_files, key=lambda meta: meta["mtime_delta_from_core_s"])
        silence_s = abs(float(latest["mtime_delta_from_core_s"]))
        progress_matches = [
            item for item in retained
            if item.get("category") == "progress"
            and item.get("role") in {"payload-stdout", "payload-stderr", "payload-log"}
        ]
        latest_progress = max(progress_matches, key=lambda item: int(item.get("line", 0)), default=None)
        activity: dict[str, Any] = {
            "latest_payload_file": latest.get("relative_path", Path(str(latest.get("path", ""))).name),
            "last_write_before_core_s": round(silence_s, 3),
            "last_write_before_core_human": _format_duration(silence_s),
        }
        latest_tail = latest.get("tail")
        if isinstance(latest_tail, list) and latest_tail:
            activity["last_nonempty_line"] = latest_tail[-1]
            activity["tail"] = latest_tail
        if latest_progress:
            activity["latest_progress"] = latest_progress
        result["payload_activity"] = activity

    return result


def derive_payload_log_observations(job_logs: dict[str, Any], primary_backtrace: str) -> list[str]:
    """Derive conservative completion/shutdown observations from payload tails.

    The payload tail is more reliable than filename heuristics for deciding
    whether a looping job was still in event processing.  These observations
    intentionally describe ordering/state only; they do not claim which XRootD
    lock or timeout caused the shutdown hang.
    """
    activity = job_logs.get("payload_activity", {}) if isinstance(job_logs, dict) else {}
    tail = activity.get("tail", []) if isinstance(activity, dict) else []
    tail_text = "\n".join(str(item.get("text", "")) for item in tail if isinstance(item, dict))
    observations: list[str] = []

    worker_success = "worker finished successfully" in tail_text
    job_success = bool(re.search(r"current job status:\s*\d+\s+success,\s*0\s+failure", tail_text, re.I))
    output_postprocessing = bool(re.search(
        r"Moving the analysis root file|Moving .*hist-output|renaming .*output\.root", tail_text, re.I
    ))
    clean_python_exit = "Py_Exit (sts=0)" in primary_backtrace
    xrootd_finalize = "XrdCl::DefaultEnv::Finalize" in primary_backtrace

    if worker_success and job_success:
        observations.append(
            "Payload EventLoop reported that the worker finished successfully and its internal status was 1 success / 0 failure. "
            "This is EventLoop-level completion evidence only; it does not mean the payload process or PanDA job completed normally."
        )
    if output_postprocessing:
        last_line = activity.get("last_nonempty_line")
        suffix = ""
        if isinstance(last_line, dict) and last_line.get("text"):
            suffix = f"; the last payload line was: {last_line['text']}"
        observations.append(
            "Payload output continued into post-processing/output-file handling after EventLoop completion" + suffix + "."
        )
    if worker_success and clean_python_exit and xrootd_finalize:
        observations.append(
            "Combined payload and core evidence places the captured hang after the EventLoop reported successful event processing, "
            "during process shutdown/XRootD finalization rather than inside the EventLoop; the payload did not complete normal process exit."
        )
    return observations


def derive_process_identity(evidence: CoreEvidence) -> dict[str, Any]:
    """Identify the process captured by the core without trusting gdb symbols alone.

    The ``Core was generated by`` command line is core metadata and therefore
    remains useful even when gdb warns that a supplied executable may not match.
    Stack-shape signals are used as corroboration, not as the sole identity source.
    """
    command = evidence.generated_by or ""
    primary = evidence.primary_thread.get("backtrace", "")
    all_stacks = "\n".join(group.backtrace for group in evidence.thread_groups)
    stack_text = primary + "\n" + all_stacks

    signals = {
        "generated_by": command or None,
        "command_mentions_prmon": bool(re.search(r"(?:^|[/\s])prmon(?:\s|$)", command, re.I)),
        "command_looks_like_payload": bool(re.search(
            r"EWRun\.py|eventloop|/srv/workDir/usr/|/InstallArea/.*/bin/.*Run\.py", command, re.I
        )),
        "python_runtime_stack": "Py_Exit" in stack_text or "Py_RunMain" in stack_text,
        "root_runtime_stack": "TROOT::" in stack_text or "TNetXNGFile::" in stack_text,
        "xrootd_runtime_stack": "XrdCl::" in stack_text or "XrdSys::" in stack_text,
        "stack_mentions_prmon": bool(re.search(r"\bprmon\b", stack_text, re.I)),
    }

    if signals["command_mentions_prmon"]:
        return {
            "kind": "prmon",
            "confidence": "high",
            "signals": signals,
            "reason": "The core-recorded command line identifies prmon.",
        }
    if signals["command_looks_like_payload"]:
        corroborated = signals["python_runtime_stack"] and (
            signals["root_runtime_stack"] or signals["xrootd_runtime_stack"]
        )
        return {
            "kind": "payload",
            "confidence": "high" if corroborated else "medium",
            "signals": signals,
            "reason": (
                "The core-recorded command line identifies the payload and the captured runtime stack is consistent with it."
                if corroborated else
                "The core-recorded command line identifies the payload, but stack corroboration is limited."
            ),
        }
    if signals["stack_mentions_prmon"]:
        return {
            "kind": "prmon",
            "confidence": "medium",
            "signals": signals,
            "reason": "The captured stack mentions prmon, but the core-recorded command line is inconclusive.",
        }
    return {
        "kind": "unknown",
        "confidence": "low",
        "signals": signals,
        "reason": "The available core metadata does not identify the captured process reliably.",
    }


def derive_symbol_evidence_quality(evidence: CoreEvidence) -> dict[str, Any]:
    """Summarise whether symbol/build identity is verified strongly enough for diagnosis."""
    checked = evidence.build_ids.get("checked", []) if isinstance(evidence.build_ids, dict) else []
    mismatch_count = int(evidence.build_ids.get("mismatch_count", 0) or 0) if isinstance(evidence.build_ids, dict) else 0
    module_count = int(evidence.build_ids.get("module_count", 0) or 0) if isinstance(evidence.build_ids, dict) else 0
    match_warning = any("core may not match the executable" in warning.lower() for warning in evidence.warnings)
    verified = len(checked) > 0 and mismatch_count == 0 and not match_warning
    if mismatch_count:
        level = "low"
    elif match_warning and not checked:
        level = "degraded"
    elif verified:
        level = "verified"
    else:
        level = "partial"
    return {
        "level": level,
        "gdb_executable_match_warning": match_warning,
        "key_build_ids_checked": len(checked),
        "build_id_mismatch_count": mismatch_count,
        "eu_unstrip_module_count": module_count,
        "verified": verified,
    }


def _diagnosis_signals(evidence: CoreEvidence,
                       process_identity: dict[str, Any]) -> dict[str, Any]:
    """Extract the deterministic stack and payload-log signals used for classification.

    Every entry is a literal presence test on captured text, so the signal set
    is auditable: a reader can check any one of them against the raw gdb
    output.  Nothing here interprets the signals.

    Args:
        evidence: The assembled evidence.
        process_identity: Resolved process identity for the captured core.

    Returns:
        Flat mapping of signal name to boolean, plus ``captured_process`` and,
        when known, ``payload_silence_before_core_s``.
    """
    primary = evidence.primary_thread.get("backtrace", "")
    all_stacks = "\n".join(group.backtrace for group in evidence.thread_groups)
    activity = _payload_activity(evidence)
    tail = activity.get("tail", []) if isinstance(activity, dict) else []
    tail_text = "\n".join(str(item.get("text", "")) for item in tail if isinstance(item, dict))

    signals: dict[str, Any] = {
        "clean_python_exit": "Py_Exit (sts=0)" in primary,
        "xrootd_finalization": "XrdCl::DefaultEnv::Finalize" in primary,
        "xrootd_poller_stop_wait": (
            "XrdSys::IOEvents::Poller::SendCmd" in primary
            and "XrdSys::IOEvents::Poller::Stop" in primary
        ),
        "root_close_files": "TROOT::CloseFiles" in primary,
        "xrootd_remote_file_close": (
            "TNetXNGFile::Close" in primary
            and "XrdCl::File::Close" in primary
            and "XrdCl::FileStateHandler::Close" in primary
        ),
        "xrootd_close_stream_mutex_wait": (
            "XrdCl::StreamMutex::Lock" in primary
            and "XrdCl::Stream::Send" in primary
        ),
        "xrootd_shutdown_events": "XrdCl::PollerBuiltIn::ShutdownEvents" in all_stacks,
        "xrootd_socket_fault": (
            "XrdCl::AsyncSocketHandler::OnFault" in all_stacks
            or "XrdCl::Stream::OnError" in all_stacks
        ),
        "xrootd_read_timeout_force_disconnect": (
            "XrdCl::Stream::OnReadTimeout" in all_stacks
            and "ForceDisconnect" in all_stacks
        ),
        "xrootd_stream_mutex_wait": (
            "XrdCl::StreamMutex::Lock" in all_stacks
            and "XrdCl::Stream::Tick" in all_stacks
        ),
        "eventloop_worker_success": "worker finished successfully" in tail_text,
        "eventloop_status_success": bool(re.search(
            r"current job status:\s*\d+\s+success,\s*0\s+failure", tail_text, re.I
        )),
        "output_postprocessing": bool(re.search(
            r"Moving the analysis root file|Moving .*hist-output|renaming .*output\.root", tail_text, re.I
        )),
        "captured_process": process_identity.get("kind", "unknown"),
    }
    silence = activity.get("last_write_before_core_s") if isinstance(activity, dict) else None
    if isinstance(silence, (int, float)):
        signals["payload_silence_before_core_s"] = round(float(silence), 3)
    return signals


def _payload_activity(evidence: CoreEvidence) -> dict[str, Any]:
    """Return the payload-activity sub-dict of the job-log evidence.

    Args:
        evidence: The assembled evidence.

    Returns:
        The ``payload_activity`` mapping, or an empty dict when absent.
    """
    if not isinstance(evidence.job_logs, dict):
        return {}
    activity = evidence.job_logs.get("payload_activity", {})
    return activity if isinstance(activity, dict) else {}


def _prmon_diagnosis(process_identity: dict[str, Any], symbol_quality: dict[str, Any],
                     signals: dict[str, Any]) -> dict[str, Any]:
    """Build the diagnosis for a core belonging to the memory monitor, not the payload.

    Args:
        process_identity: Resolved process identity.
        symbol_quality: Symbol/build-identity assessment.
        signals: Deterministic signal set.

    Returns:
        A complete diagnosis dict flagging payload diagnosis as inapplicable.
    """
    return {
        "available": True,
        "classification": "monitor-process-core",
        "phase": "monitoring-process",
        "component": "prmon",
        "confidence": process_identity.get("confidence", "medium"),
        "root_cause_established": False,
        "payload_diagnosis_applicable": False,
        "summary": "The core metadata identifies the captured process as prmon rather than the payload; payload-loop diagnosis is not applicable to this core.",
        "process_identity": process_identity,
        "symbol_evidence_quality": symbol_quality,
        "signals": signals,
        "supporting_evidence": [process_identity.get("reason", "The captured process is prmon.")],
        "limitations": [
            "Payload logs in the job directory describe the payload and must not be attributed to the prmon core."
        ],
    }


def _unclassified_diagnosis(reason: str, process_identity: dict[str, Any],
                            symbol_quality: dict[str, Any],
                            signals: dict[str, Any]) -> dict[str, Any]:
    """Build the refusal shape used whenever no rule may safely classify the core.

    Returning a populated dict with ``available: False`` rather than raising
    keeps the signal set and evidence-quality assessment visible to the caller,
    so a refusal can be explained rather than merely reported.

    Args:
        reason: Why classification was refused.
        process_identity: Resolved process identity.
        symbol_quality: Symbol/build-identity assessment.
        signals: Deterministic signal set.

    Returns:
        An unavailable-diagnosis dict.
    """
    return {
        "available": False,
        "classification": "unclassified",
        "confidence": "low",
        "root_cause_established": False,
        "process_identity": process_identity,
        "symbol_evidence_quality": symbol_quality,
        "signals": signals,
        "reason": reason,
    }


def _remote_close_diagnosis(signals: dict[str, Any],
                            completed_payload: bool) -> dict[str, Any]:
    """Describe a hang captured while ROOT/XRootD was closing a remote file.

    Args:
        signals: Deterministic signal set.
        completed_payload: Whether the payload's own logs show event processing
            reaching its successful end state.  This only raises confidence and
            sharpens the wording; it is never evidence that the job succeeded.

    Returns:
        Mapping with the classification fields and supporting evidence for this
        signature.
    """
    supporting: list[str] = []
    if completed_payload:
        supporting.extend([
            "Payload reports that the EventLoop worker finished successfully.",
            "Payload EventLoop status reports one success and zero failures; this is not evidence that the PanDA job completed.",
            "Payload reached output-file post-processing before becoming silent.",
        ])
    supporting.append(
        "Primary thread is in Py_Exit(sts=0) -> TROOT::CloseFiles -> TNetXNGFile::Close -> XrdCl::File::Close -> StreamMutex::Lock."
    )
    if signals["xrootd_shutdown_events"]:
        supporting.append("A concurrent XRootD poller thread is in ShutdownEvents during socket close/error handling.")
    if signals["xrootd_stream_mutex_wait"]:
        supporting.append("A concurrent XRootD task thread waits in StreamMutex::Lock while running Stream::Tick.")
    return {
        "classification": (
            "post-event-processing-remote-file-close-hang"
            if completed_payload else "remote-file-close-hang"
        ),
        "family": (
            "post-event-processing-xrootd-shutdown-hang"
            if completed_payload else "xrootd-shutdown-hang"
        ),
        "subtype": "remote-file-close",
        "phase": "process-shutdown",
        "component": "ROOT/XRootD",
        "confidence": "high" if completed_payload else "medium",
        "summary": (
            "The EventLoop reported successful event processing, but the payload did not complete normally; it later hung while ROOT/XRootD was closing a remote file during process shutdown."
            if completed_payload else
            "The process was captured in a clean Python exit path while ROOT/XRootD was closing a remote file."
        ),
        "supporting_evidence": supporting,
    }


def _poller_shutdown_diagnosis(signals: dict[str, Any],
                               completed_payload: bool) -> dict[str, Any]:
    """Describe a hang captured during XRootD/XrdCl shutdown finalization.

    Args:
        signals: Deterministic signal set.
        completed_payload: Whether the payload's own logs show event processing
            reaching its successful end state.

    Returns:
        Mapping with the classification fields and supporting evidence for this
        signature.
    """
    supporting: list[str] = []
    if signals["eventloop_worker_success"]:
        supporting.append("Payload reports that the EventLoop worker finished successfully.")
    if signals["eventloop_status_success"]:
        supporting.append("Payload EventLoop status reports one success and zero failures; this is not evidence that the PanDA job completed.")
    if signals["output_postprocessing"]:
        supporting.append("Payload reached output-file post-processing before becoming silent.")
    supporting.append("Primary thread is in Py_Exit(sts=0) -> XrdCl::DefaultEnv::Finalize -> Poller::Stop/SendCmd.")
    if signals["xrootd_stream_mutex_wait"]:
        supporting.append("A concurrent XRootD thread waits in StreamMutex::Lock while running Stream::Tick.")
    if signals["xrootd_read_timeout_force_disconnect"]:
        supporting.append("A concurrent XRootD thread handles OnReadTimeout with forced disconnect activity.")
    return {
        "classification": (
            "post-event-processing-shutdown-hang"
            if completed_payload else "shutdown-finalization-hang"
        ),
        "family": (
            "post-event-processing-xrootd-shutdown-hang"
            if completed_payload else "xrootd-shutdown-hang"
        ),
        "subtype": "poller-finalization",
        "phase": "process-shutdown",
        "component": "XRootD/XrdCl",
        "confidence": "high" if completed_payload else "medium",
        "summary": (
            "The EventLoop reported successful event processing, but the payload did not complete normally; it later hung during XRootD/XrdCl shutdown finalization."
            if completed_payload else
            "The process was captured in a clean Python exit path while blocked during XRootD/XrdCl shutdown finalization."
        ),
        "supporting_evidence": supporting,
    }


def _diagnosis_limitations(evidence: CoreEvidence, signals: dict[str, Any],
                           symbol_quality: dict[str, Any],
                           confidence: str) -> tuple[list[str], str]:
    """Collect evidence-quality caveats and cap confidence where they apply.

    A single core snapshot cannot prove causality, so correlated XRootD faults
    are recorded as observations rather than causes.  A gdb executable-match
    warning caps a "high" confidence to "medium": the stack signature may be
    strong while the identity of the binary it was read against is not.

    Args:
        evidence: The assembled evidence.
        signals: Deterministic signal set.
        symbol_quality: Symbol/build-identity assessment.
        confidence: Confidence proposed by the matched signature.

    Returns:
        Tuple of ``(limitations, adjusted_confidence)``.
    """
    limitations = ["A single core snapshot does not prove the exact lock cycle."]
    if signals["xrootd_read_timeout_force_disconnect"]:
        limitations.append(
            "The concurrent XRootD read timeout/forced-disconnect path is observed, but causality is not established."
        )
    if signals["xrootd_socket_fault"]:
        limitations.append(
            "Concurrent XRootD socket fault/error handling is observed, but causality is not established."
        )
    if symbol_quality["gdb_executable_match_warning"]:
        limitations.append(
            "GDB warns that the core may not match the supplied executable; key Build IDs could not verify the executable/system-library identity."
        )
        if confidence == "high":
            confidence = "medium"
    if evidence.targeted_threads and all(
        not item.get("frame_details_available", True) for item in evidence.targeted_threads
    ):
        limitations.append("Selected XRootD frames lack usable argument/local DWARF in this optimized build.")
    return limitations, confidence


def derive_structured_diagnosis(evidence: CoreEvidence) -> dict[str, Any]:
    """Build a conservative machine-readable diagnosis from deterministic evidence.

    Classification describes the captured phase/component, not an initiating
    root cause. Evidence-quality limitations can lower confidence without
    discarding a strongly supported process-phase classification.

    Args:
        evidence: The assembled evidence.

    Returns:
        A diagnosis dict.  ``available`` is ``False`` when no rule matched or
        when the evidence was too degraded to classify safely; downstream
        consumers must check it before trusting ``classification``.
    """
    process_identity = evidence.process_identity or derive_process_identity(evidence)
    symbol_quality = derive_symbol_evidence_quality(evidence)
    signals = _diagnosis_signals(evidence, process_identity)

    if process_identity.get("kind") == "prmon":
        return _prmon_diagnosis(process_identity, symbol_quality, signals)

    if (process_identity.get("kind") == "unknown"
            and symbol_quality.get("level") in {"degraded", "low"}):
        return _unclassified_diagnosis(
            "Process identity is unknown and symbol/build identity is degraded; "
            "refusing to classify the stack signature.",
            process_identity, symbol_quality, signals,
        )

    completed_payload = (
        signals["eventloop_worker_success"]
        and signals["eventloop_status_success"]
        and signals["output_postprocessing"]
    )
    poller_shutdown_signature = (
        evidence.mode == "hang"
        and signals["clean_python_exit"]
        and signals["xrootd_finalization"]
        and signals["xrootd_poller_stop_wait"]
    )
    remote_close_signature = (
        evidence.mode == "hang"
        and signals["clean_python_exit"]
        and signals["root_close_files"]
        and signals["xrootd_remote_file_close"]
        and signals["xrootd_close_stream_mutex_wait"]
    )

    # Remote-file close is checked first: its signature is the more specific of
    # the two and both can be present in the same shutdown stack.
    if remote_close_signature:
        matched = _remote_close_diagnosis(signals, completed_payload)
    elif poller_shutdown_signature:
        matched = _poller_shutdown_diagnosis(signals, completed_payload)
    else:
        return _unclassified_diagnosis(
            "No supported deterministic diagnosis rule matched the captured state.",
            process_identity, symbol_quality, signals,
        )

    limitations, confidence = _diagnosis_limitations(
        evidence, signals, symbol_quality, str(matched["confidence"]),
    )

    return {
        "available": True,
        "classification": matched["classification"],
        "family": matched["family"],
        "subtype": matched["subtype"],
        "phase": matched["phase"],
        "component": matched["component"],
        "confidence": confidence,
        "root_cause_established": False,
        "job_completion": {
            "event_processing_completed": bool(completed_payload),
            "payload_process_exited_normally": False,
            "job_completed_normally": False,
            "reason": (
                "The EventLoop reached its successful end state, but the core captures the payload still alive and hung during shutdown."
                if completed_payload else
                "The core captures the payload still alive and hung; normal process completion is not established."
            ),
        },
        "summary": matched["summary"],
        "process_identity": process_identity,
        "symbol_evidence_quality": symbol_quality,
        "signals": signals,
        "supporting_evidence": matched["supporting_evidence"],
        "limitations": limitations,
    }


def split_thread_stacks(text: str) -> list[tuple[str, str, str]]:
    """Split ``thread apply all bt`` output into per-thread backtraces.

    Args:
        text: Output of ``thread apply all bt``.

    Returns:
        A list of ``(thread_id, thread_name, backtrace)`` tuples.
    """
    header = re.compile(r"^Thread\s+(\d+)\s+\(.*?(?:\"([^\"]*)\")?\s*\):\s*$", re.M)
    matches = list(header.finditer(text))
    stacks: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        stacks.append((match.group(1), match.group(2) or "", body))
    return stacks


def group_thread_stacks(text: str, max_groups: int, redact_enabled: bool) -> list[ThreadGroup]:
    """Collapse identical thread backtraces into groups.

    An ATLAS/Gaudi job routinely has 100+ threads parked on the same condition
    variable. Grouping them keeps the evidence within a sane token budget and
    makes the one genuinely interesting thread stand out.

    Args:
        text: Output of ``thread apply all bt``.
        max_groups: Maximum number of groups to retain.
        redact_enabled: Whether to scrub secrets from the representative stacks.

    Returns:
        Thread groups sorted so that busy stacks appear before idle ones.
    """
    buckets: dict[str, ThreadGroup] = {}
    for thread_id, thread_name, backtrace in split_thread_stacks(text):
        signature = "\n".join(normalise_frame_line(ln) for ln in backtrace.splitlines() if ln.strip().startswith("#"))
        if not signature:
            continue
        group = buckets.get(signature)
        if group is None:
            trimmed, was_cut = truncate(redact(backtrace, redact_enabled), SECTION_LIMITS["thread_group"])
            state = _classify_thread_stack(backtrace)
            buckets[signature] = ThreadGroup(
                count=1,
                thread_ids=[thread_id],
                names=[thread_name] if thread_name else [],
                backtrace=trimmed + ("" if not was_cut else ""),
                idle=(state == "idle"),
                state=state,
            )
            continue
        group.count += 1
        if len(group.thread_ids) < 10:
            group.thread_ids.append(thread_id)
        if thread_name and thread_name not in group.names:
            group.names.append(thread_name)

    state_rank = {"blocked": 0, "active": 1, "idle": 2}
    groups = sorted(buckets.values(), key=lambda grp: (state_rank.get(grp.state, 1), -grp.count))
    return groups[:max_groups]


def derive_deterministic_observations(primary_backtrace: str,
                                      thread_groups: list[ThreadGroup]) -> list[str]:
    """Derive conservative, pattern-based observations without an LLM.

    These are intentionally factual stack-state statements, not root-cause
    claims. They make ``--no-llm`` useful while preserving the distinction
    between evidence and synthesis.
    """
    observations: list[str] = []
    all_stacks = "\n".join(group.backtrace for group in thread_groups)

    clean_python_exit = "Py_Exit (sts=0)" in primary_backtrace
    xrootd_finalize = (
        "XrdCl::DefaultEnv::Finalize" in primary_backtrace
        and "XrdCl::PostMaster::Stop" in primary_backtrace
        and "XrdSys::IOEvents::Poller::Stop" in primary_backtrace
    )
    if clean_python_exit and xrootd_finalize:
        observations.append(
            "Process is already in Py_Exit(sts=0) and is blocked while XRootD/XrdCl finalization stops the poller."
        )
    elif xrootd_finalize:
        observations.append(
            "Primary thread is blocked while XRootD/XrdCl finalization stops the poller."
        )

    remote_file_close = (
        "TROOT::CloseFiles" in primary_backtrace
        and "TNetXNGFile::Close" in primary_backtrace
        and "XrdCl::File::Close" in primary_backtrace
        and "XrdCl::StreamMutex::Lock" in primary_backtrace
    )
    if clean_python_exit and remote_file_close:
        observations.append(
            "Process is already in Py_Exit(sts=0) and is blocked while ROOT/XRootD closes a remote file."
        )

    if "XrdCl::Stream::OnReadTimeout" in all_stacks and "ForceDisconnect" in all_stacks:
        observations.append(
            "A concurrent XRootD thread is handling a read timeout and forced disconnect during the captured state."
        )
    if "XrdCl::StreamMutex::Lock" in all_stacks and "XrdCl::Stream::Tick" in all_stacks:
        observations.append(
            "Another XRootD thread is waiting in StreamMutex::Lock while processing Stream::Tick."
        )
    return observations


def _backtrace_has_unknown_frames(text: str) -> bool:
    """Return whether actual backtrace frames, rather than args/locals, lack symbols."""
    return bool(re.search(r"^#\d+\s+.*(?:\bin \?\?|\s\?\?\s*$)", text, re.M))


def summarise_shared_libraries(text: str) -> dict[str, Any]:
    """Summarise ``info sharedlibrary`` without conflating symbol states.

    GDB reports three materially different states: ``Yes`` means symbols were
    read, ``Yes (*)`` means symbols were read but full/separate debugging
    information is absent, and ``No`` means symbols were not read. Only ``No``
    belongs in ``without_symbols``. A plain ``Yes`` is not a guarantee that
    every optimized function in that DSO has recoverable arguments or locals.
    """
    total = 0
    without_symbols: list[str] = []
    without_full_debug: list[str] = []
    with_symbols_count = 0
    pattern = re.compile(
        r"^\s*0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+\s+(Yes(?:\s+\(\*\))?|No)\s+(.+?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        status, path = match.groups()
        if not path.startswith("/"):
            continue
        total += 1
        if status == "No":
            if len(without_symbols) < 40:
                without_symbols.append(path)
        elif "(*)" in status:
            with_symbols_count += 1
            if len(without_full_debug) < 40:
                without_full_debug.append(path)
        else:
            with_symbols_count += 1
    return {
        "total_loaded": total,
        "with_symbols_count": with_symbols_count,
        "without_symbols": without_symbols,
        "without_symbols_count": len(without_symbols),
        "without_full_debug_info": without_full_debug,
        "without_full_debug_info_count": len(without_full_debug),
    }


def detect_mode(requested: str, signal: str | None, generated_by: str | None) -> tuple[str, str]:
    """Classify the dump as a crash or a hang snapshot.

    Args:
        requested: The user's ``--mode`` value.
        signal: The terminating signal, if known.
        generated_by: The recorded command line, if known.

    Returns:
        A tuple of the resolved mode and a short explanation of how it was set.
    """
    if requested in ("crash", "hang"):
        return requested, "explicitly supplied via --mode"
    if signal in CRASH_SIGNALS:
        return "crash", f"inferred from fault signal {signal}"
    if signal in HANG_SIGNALS:
        return "hang", f"inferred from signal {signal}, which is normally externally delivered"
    if generated_by and "gcore" in generated_by:
        return "hang", "inferred from a gcore-generated snapshot"
    return "hang", "no fault signal found; defaulting to hang analysis"


# --------------------------------------------------------------------------- #
# Evidence assembly
# --------------------------------------------------------------------------- #


def _build_phase_plan(args: argparse.Namespace) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build the ordered list of gdb phases to execute.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A list of ``(phase_name, [(section_name, gdb_command), ...])`` tuples.
    """
    primary: list[tuple[str, str]] = [
        ("backtrace", f"bt {args.max_frames}"),
        ("frame", "info frame"),
        ("args", "info args"),
    ]
    if args.locals:
        primary.append(("locals", "info locals"))
    primary.append(("registers", "info registers"))
    return [
        ("metadata", [
            ("program", "info program"),
            ("threads", "info threads"),
            ("files", "info files"),
            ("debug_file_directory", "show debug-file-directory"),
            ("auto_load_python_scripts", "info auto-load python-scripts"),
        ]),
        ("primary_thread", primary),
        ("all_threads", [("all_threads", f"thread apply all bt {args.max_frames}")]),
        ("python", [
            ("py_helper", _python_helper_bootstrap_command(
                getattr(args, "python_gdb_helper", "") or "",
                _python_version_hint(
                    Path(args.job_dir) if getattr(args, "job_dir", None) else None
                ),
            )),
            ("py_bt", "py-bt"),
            ("py_list", "py-list"),
        ]),
        ("libraries", [("libraries", "info sharedlibrary")]),
    ]


def _trim_primary_sections(sections: dict[str, str], redact_enabled: bool) -> tuple[dict[str, str], list[str]]:
    """Redact and size-limit the primary-thread sections.

    Args:
        sections: Raw per-command output keyed by section name.
        redact_enabled: Whether to scrub secrets.

    Returns:
        A tuple of the trimmed sections and the names of any that were truncated.
    """
    trimmed: dict[str, str] = {}
    truncated: list[str] = []
    for name in ("backtrace", "frame", "args", "locals", "registers"):
        body = sections.get(name, "").strip()
        if not body:
            continue
        body, was_cut = truncate(redact(body, redact_enabled), SECTION_LIMITS.get(name, 6_000))
        trimmed[name] = body
        if was_cut:
            truncated.append(f"primary_thread.{name}")
    return trimmed, truncated


def _build_id_warnings(build_ids: dict[str, Any]) -> list[str]:
    """Turn Build-ID comparison results into evidence-quality warnings.

    A Build-ID mismatch means gdb read symbols from a different build than the
    one that produced the core, so frame names and line numbers can be
    confidently wrong.  That has to reach the user as a warning rather than
    quietly degrade the analysis.

    Args:
        build_ids: Result of :func:`collect_build_id_evidence`.

    Returns:
        Warning strings, empty when identity was fully verified.
    """
    warnings: list[str] = []
    if not build_ids.get("available"):
        warnings.append(
            "Build-ID comparison was not available; executable/system-library identity was not verified."
        )
    elif not build_ids.get("checked"):
        warnings.append(
            f"Build-ID coverage is insufficient: eu-unstrip enumerated {build_ids.get('module_count', 0)} module(s), "
            "but no executable or critical system-library Build IDs could be verified."
        )
    for item in build_ids.get("checked", []):
        if item.get("match") is False:
            warnings.append(
                f"Build-ID mismatch for {item.get('name')}: core {item.get('core_build_id')} vs "
                f"analysis file {item.get('file_build_id')}. Stack unwinding/symbols may be misleading."
            )
    return warnings


def _phase_raw_chunk(name: str, commands: Sequence[tuple[str, str]],
                     result: GdbPhaseResult) -> str:
    """Render one gdb phase's unprocessed output for the raw transcript.

    Args:
        name: Phase name.
        commands: The ``(section, command)`` pairs the phase ran.
        result: Phase result.

    Returns:
        A banner-delimited transcript chunk.
    """
    banner = "=" * 70
    rendered = "; ".join(cmd for _, cmd in commands)
    return f"\n{banner}\n# phase: {name} ({rendered})\n{banner}\n{result.stdout}\n{result.stderr}"


def _record_phase(evidence: CoreEvidence, result: GdbPhaseResult,
                  args: argparse.Namespace, name: str,
                  timeout_consequence: str) -> None:
    """Append a phase record to the evidence and warn if the phase timed out.

    Args:
        evidence: Evidence bundle to update in place.
        result: Phase result.
        args: Parsed arguments, read for the timeout and redaction settings.
        name: Phase name used in the timeout warning.
        timeout_consequence: What is lost when this phase times out, appended
            to the warning so the gap is explicit rather than inferred from a
            missing key.
    """
    evidence.phases.append({
        "name": result.name,
        "commands": result.commands,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_s": result.duration_s,
        "stderr_excerpt": redact(result.stderr[:500], not args.no_redact),
    })
    if result.timed_out:
        evidence.warnings.append(
            f"gdb phase '{name}' timed out after {args.gdb_timeout}s; {timeout_consequence}."
        )


def _attach_library_evidence(evidence: CoreEvidence, sections: dict[str, str],
                             redact_enabled: bool) -> None:
    """Summarise loaded shared libraries and warn about missing symbols.

    Args:
        evidence: Evidence bundle to update in place.
        sections: Parsed gdb section output.
        redact_enabled: Unused here; accepted so the call site reads uniformly
            with the other assembly helpers.
    """
    del redact_enabled  # summarise_shared_libraries does its own handling
    evidence.shared_libraries = summarise_shared_libraries(sections.get("libraries", ""))
    missing = evidence.shared_libraries.get("without_symbols_count", 0)
    if missing:
        evidence.warnings.append(
            f"GDB could not read symbols for {missing} loaded shared librar"
            f"{'y' if missing == 1 else 'ies'}."
        )


def _gdb_metadata(sections: dict[str, str], combined: str,
                  redact_enabled: bool) -> dict[str, Any]:
    """Collect gdb's own configuration and startup warnings.

    Kept in the evidence because a surprising diagnosis is often explained by
    gdb's setup — a missing debug-file directory, a libpython auto-load that
    did not happen — rather than by the core.

    Args:
        sections: Parsed gdb section output.
        combined: Full raw transcript, scanned for warning lines.
        redact_enabled: Whether to scrub credentials from retained text.

    Returns:
        Metadata dict for :attr:`CoreEvidence.gdb_metadata`.
    """
    files_excerpt, files_cut = truncate(
        redact(sections.get("files", ""), redact_enabled), 4_000
    )
    return {
        "debug_file_directory": redact(sections.get("debug_file_directory", ""), redact_enabled),
        "auto_load_python_scripts": redact(sections.get("auto_load_python_scripts", ""), redact_enabled),
        "info_files_excerpt": files_excerpt,
        "info_files_truncated": files_cut,
        "startup_warnings": list(dict.fromkeys(
            redact(line.strip(), redact_enabled)
            for line in combined.splitlines()
            if "warning:" in line.lower()
        ))[:20],
    }


def _collect_evidence_local(
    args: argparse.Namespace, progress: bool = True, detail: bool = False,
) -> tuple[CoreEvidence, str]:
    """Drive gdb and assemble the structured evidence bundle.

    Args:
        args: Parsed command-line arguments.
        progress: Whether to print basic phase-level progress to stderr. gdb
            buffers all output of a phase until it exits, so without this a
            large core can appear to hang with no indication anything is
            happening.
        detail: Whether to also print periodic heartbeat messages while a
            phase is running, and note when a phase's evidence gets trimmed
            to fit the budget. Has no effect if ``progress`` is ``False``.

    Returns:
        A tuple of the :class:`CoreEvidence` and the concatenated raw gdb output.

    Raises:
        FileNotFoundError: If the core file or gdb cannot be found.
    """
    core_path = Path(args.core_file).expanduser().resolve()
    if not core_path.is_file():
        raise FileNotFoundError(f"Core file not found: {core_path}")

    gdb_path = find_gdb(args.gdb)
    evidence = CoreEvidence()
    evidence.environment = collect_runtime_environment()
    stat = core_path.stat()
    size_mib = stat.st_size / (1024 ** 2)
    evidence.core_file = {
        "path": str(core_path),
        "size_bytes": stat.st_size,
        "size_human": f"{size_mib:.1f} MiB",
        "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    }
    evidence.gdb = {"path": gdb_path, "version": gdb_version(gdb_path)}

    warning_threshold = getattr(args, "large_core_warning_mib", DEFAULT_LARGE_CORE_WARNING_MIB)
    if progress:
        print(f"[*] Core file: {core_path.name} ({size_mib:.1f} MiB), gdb {evidence.gdb['version']}",
              file=sys.stderr)
        if warning_threshold and size_mib >= warning_threshold:
            print(
                f"[*] This core is above {warning_threshold} MiB. gdb reloads the whole core once per "
                "analysis phase, so each of the phases below can take from several seconds to a few "
                "minutes on a core this size -- that is expected, not a hang. Pass -v for a periodic "
                "'still running' heartbeat during long phases.",
                file=sys.stderr,
            )
        print("[*] Resolving executable...", file=sys.stderr)

    heartbeat_interval = getattr(args, "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
    release_info = _parse_release_info(Path(args.job_dir) if getattr(args, "job_dir", None) else None)
    debug_init = _debug_file_init(args, release_info)
    probe = run_gdb_phase(
        gdb_path, core_path, None, "probe", [("program", "info program")], args.gdb_timeout,
        progress=progress, detail=detail, heartbeat_interval=heartbeat_interval,
    )
    evidence.executable = resolve_executable(
        gdb_path, core_path, args.exe, probe.stdout + probe.stderr, args.gdb_timeout,
        progress=progress, detail=detail, heartbeat_interval=heartbeat_interval,
    )
    exe_path = evidence.executable.get("path")
    if progress:
        print(f"[*] Executable: {exe_path or 'UNRESOLVED'} (via {evidence.executable['source']})", file=sys.stderr)

    evidence.build_ids, unstrip_raw = collect_build_id_evidence(core_path, exe_path)
    evidence.warnings.extend(_build_id_warnings(evidence.build_ids))

    raw_chunks: list[str] = [f"$ gdb -c {core_path.name} -ex 'info program'\n{probe.stdout}\n{probe.stderr}"]
    if unstrip_raw:
        raw_chunks.append(f"\n{'=' * 70}\n# eu-unstrip -n --core {core_path.name}\n{'=' * 70}\n{unstrip_raw}")
    sections: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, commands in _build_phase_plan(args):
        result = run_gdb_phase(
            gdb_path, core_path, exe_path, name, commands, args.gdb_timeout,
            progress=progress, detail=detail, heartbeat_interval=heartbeat_interval,
            extra_init=debug_init,
        )
        sections.update(result.sections)
        errors[name] = result.stderr
        raw_chunks.append(_phase_raw_chunk(name, commands, result))
        _record_phase(evidence, result, args, name, "its evidence is missing")

    # Select interesting thread/frame pairs only after the all-thread phase has
    # established the shape of the hang.  All focused inspections run in one
    # additional gdb process so large cores are reloaded only once more.
    evidence.thread_groups = group_thread_stacks(
        sections.get("all_threads", ""), args.max_thread_groups, not args.no_redact
    )
    targets = select_targeted_threads(
        evidence.thread_groups, getattr(args, "max_targeted_threads", DEFAULT_MAX_TARGETED_THREADS)
    )
    if targets:
        commands = _build_targeted_phase(targets, args.locals)
        result = run_gdb_phase(
            gdb_path, core_path, exe_path, "targeted_threads", commands, args.gdb_timeout,
            progress=progress, detail=detail, heartbeat_interval=heartbeat_interval,
            extra_init=debug_init,
        )
        sections.update(result.sections)
        errors["targeted_threads"] = result.stderr
        raw_chunks.append(_phase_raw_chunk("targeted_threads", commands, result))
        _record_phase(
            evidence, result, args, "targeted_threads",
            "focused frame evidence is missing",
        )
        if not result.timed_out:
            evidence.targeted_threads = summarise_targeted_threads(
                targets, result.sections, not args.no_redact
            )

    combined = "\n".join(raw_chunks)
    evidence.signal = parse_signal(combined)
    evidence.generated_by = redact(parse_generated_by(combined) or "", not args.no_redact) or None
    evidence.thread_count = parse_thread_count(sections.get("threads", ""))
    evidence.warnings.extend(w for w in collect_warnings(combined) if w not in evidence.warnings)
    evidence.warnings.extend(evidence.executable.get("notes", []))
    evidence.mode, evidence.mode_source = detect_mode(args.mode, evidence.signal, evidence.generated_by)

    primary, truncated = _trim_primary_sections(sections, not args.no_redact)
    evidence.primary_thread = primary
    evidence.truncated_sections.extend(truncated)
    evidence.observations = derive_deterministic_observations(
        primary.get("backtrace", ""), evidence.thread_groups
    )
    if _backtrace_has_unknown_frames(
        sections.get("backtrace", "") + "\n" + sections.get("all_threads", "")
    ):
        evidence.warnings.append("Some backtrace frames have no symbol information.")
    py_helper = _parse_python_helper(sections.get("py_helper", ""))
    evidence.python = _summarise_python(
        sections.get("py_bt", ""), sections.get("py_list", ""), errors.get("python", ""),
        not args.no_redact, py_helper,
    )
    _attach_library_evidence(evidence, sections, not args.no_redact)
    evidence.gdb_metadata = _gdb_metadata(sections, combined, not args.no_redact)
    evidence.gdb_metadata["python_helper"] = py_helper
    # Recorded because every path derived per-job — the debug-file directory
    # above all — is derived from this, and a reader checking why one resolved
    # the way it did needs to see what was parsed.
    evidence.gdb_metadata["release"] = _parse_release_info(
        Path(args.job_dir) if getattr(args, "job_dir", None) else None
    )
    return evidence, combined


def core_evidence_from_dict(payload: dict[str, Any]) -> CoreEvidence:
    """Reconstruct :class:`CoreEvidence` from a JSON-serialised dictionary.

    Older evidence bundles only carried the boolean ``idle`` field. Preserve
    their meaning when loading them after the additive ``state`` field was
    introduced in 0.2.1.
    """
    data = dict(payload)
    groups: list[ThreadGroup] = []
    for raw_group in data.get("thread_groups", []):
        group = dict(raw_group)
        if "state" not in group:
            group["state"] = "idle" if group.get("idle") else "active"
        groups.append(ThreadGroup(**group))
    data["thread_groups"] = groups
    allowed = set(CoreEvidence.__dataclass_fields__)
    return CoreEvidence(**{key: value for key, value in data.items() if key in allowed})


def _container_path(host_path: Path, job_dir: Path) -> str:
    """Translate a path under the job directory to its standard ``/srv`` mount."""
    try:
        rel = host_path.resolve().relative_to(job_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Path {host_path} is outside --job-dir {job_dir}. The ATLAS container backend currently "
            "requires the core and release setup to live under the job directory mounted at /srv."
        ) from exc
    return "/srv" if str(rel) == "." else f"/srv/{rel.as_posix()}"


def _container_worker_args(args: argparse.Namespace, core_in_container: str,
                           worker_in_container: str, json_in_container: str,
                           raw_in_container: str, job_dir: Path) -> list[str]:
    """Build the evidence-only analyzer command executed inside the container."""
    argv = [
        "python3", worker_in_container, core_in_container,
        "--execution", "local",
        "--mode", args.mode,
        "--max-frames", str(args.max_frames),
        "--max-thread-groups", str(args.max_thread_groups),
        "--max-targeted-threads", str(getattr(args, "max_targeted_threads", DEFAULT_MAX_TARGETED_THREADS)),
        "--max-evidence-chars", str(args.max_evidence_chars),
        "--gdb-timeout", str(args.gdb_timeout),
        "--heartbeat-interval", str(getattr(args, "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)),
        "--large-core-warning-mib", "0",
        "--no-llm",
        "--json", json_in_container,
        "--raw-gdb", raw_in_container,
        "--quiet",
    ]
    if not args.locals:
        argv.append("--no-locals")
    if args.no_redact:
        argv.append("--no-redact")
    if args.exe:
        exe = Path(args.exe).expanduser()
        if exe.is_absolute() and exe.exists():
            try:
                exe_arg = _container_path(exe, job_dir)
            except RuntimeError:
                exe_arg = str(exe)
        else:
            exe_arg = args.exe
        argv += ["--exe", exe_arg]
    if args.gdb:
        argv += ["--gdb", args.gdb]
    debug_dir = getattr(args, "debug_file_directory", "") or ""
    if debug_dir:
        # Not staged: unlike the CPython helper this is expected to be a CVMFS
        # path, already visible inside the container.
        argv += ["--debug-file-directory", debug_dir]
    helper = getattr(args, "python_gdb_helper", "") or ""
    if helper:
        # Passed as an argument rather than inherited: ALRB launches apptainer
        # with --cleanenv, so nothing from the host environment survives into
        # the container.
        argv += ["--python-gdb-helper", helper]
    return argv


#: Characters of container output retained in a failure message.  ALRB prints a
#: message-of-the-day and, on error, its full command menu, so an untrimmed
#: tail is mostly noise wrapped across a terminal panel.
CONTAINER_DIAGNOSTIC_CHARS = 2000

#: Lines that carry the actual reason a container run failed.  ALRB's own
#: errors are ``Error: …``; the runner and gdb use the other spellings.
_ERROR_LINE_RE = re.compile(
    r"^\s*(?:Error|ERROR|error|Fatal|FATAL|fatal)\b.*$|^\s*\S*(?:Error|Exception):\s.*$"
)


#: Markers the bootstrap prints so its outcome can be parsed out of the phase.
PY_HELPER_LOADED_MARKER = "@@BAMBOO_PYHELPER_LOADED@@"
PY_HELPER_TRIED_MARKER = "@@BAMBOO_PYHELPER_TRIED@@"
PY_HELPER_VERSION_MARKER = "@@BAMBOO_PYHELPER_VERSION@@"
PY_HELPER_OBJECTS_MARKER = "@@BAMBOO_PYHELPER_OBJECTS@@"
PY_HELPER_TYPES_MARKER = "@@BAMBOO_PYHELPER_TYPES@@"

#: Filenames the CPython helper is shipped under.  ``python-gdb.py`` is the
#: installed name; ``libpython.py`` is what it is called in the CPython source
#: tree, which is where most people fetch it from.
PY_HELPER_NAMES: tuple[str, ...] = ("python-gdb.py", "libpython.py")

#: Preferred name, used when staging a single file.
PY_HELPER_NAME = PY_HELPER_NAMES[0]

#: Subdirectory of --job-dir into which an override is staged for the container.
PY_HELPER_STAGE_DIR = ".core_dump_analyzer_pygdb"

#: Bounds on staging a directory of helpers.  The real file is around 100 KiB
#: and a deployment carries a handful of versions; anything far outside that is
#: a misconfigured path rather than a helper tree.
PY_HELPER_MAX_FILES = 12
PY_HELPER_MAX_BYTES = 4 * 1024 * 1024

#: Run inside gdb before ``py-bt``, to load the CPython helper that defines it.
#:
#: gdb auto-loads ``<objfile>-gdb.py`` next to each loaded object, which is how
#: the libstdc++ pretty-printers arrive.  CPython's helper is not packaged that
#: way in the ATLAS/LCG releases: for job 7272161793 gdb's own
#: ``info auto-load python-scripts`` listed exactly one script, libstdc++'s, so
#: auto-load was working and simply had no libpython candidate to find.  Hence
#: searching explicitly rather than relying on auto-load, and recording what was
#: searched so "not available" can be told apart from "not looked for".
_PY_HELPER_BOOTSTRAP_SRC = """
import os
import re
import gdb

override = OVERRIDE
hint = HINT
names = NAMES

version = ""
python_objects = []
for objfile in gdb.objfiles():
    name = getattr(objfile, "filename", "") or ""
    if not name:
        continue
    base = os.path.basename(name)
    # Match the interpreter however it is packaged: libpython3.13.so.1.0, a
    # python3.13 binary, or a plain `python` whose path carries the version.
    if not re.search(r"(?:lib)?python\\d", base) and not re.search(r"/python\\d+\\.\\d+/", name):
        continue
    python_objects.append(name)
    if not version:
        found = re.search(r"python(\\d+)\\.(\\d+)", base) or re.search(r"python(\\d+)\\.(\\d+)", name)
        if found:
            version = found.group(1) + "." + found.group(2)

if not version:
    version = hint

candidates = []
if override:
    if os.path.isdir(override):
        # A directory of per-version helpers.  The helper reads CPython's own
        # struct layouts, which change between minor versions, so the version
        # decides which one applies.
        if version:
            entries = []
            try:
                entries = sorted(os.listdir(override))
            except OSError:
                entries = []
            exact = [e for e in entries if e == version]
            patch = [e for e in entries if e.startswith(version + ".")]
            for entry in exact + sorted(patch, reverse=True):
                for leaf in names:
                    candidates.append(os.path.join(override, entry, leaf))
        for leaf in names:
            candidates.append(os.path.join(override, leaf))
    else:
        candidates.append(override)

for name in python_objects:
    base = os.path.basename(name)
    candidates.append(name + "-gdb.py")
    root = os.path.dirname(os.path.dirname(name))
    candidates.append(os.path.join(root, "share", "gdb", "auto-load", base + "-gdb.py"))

loaded = ""
tried = []
for candidate in candidates:
    if candidate in tried:
        continue
    tried.append(candidate)
    if not os.path.isfile(candidate):
        continue
    try:
        gdb.execute("source " + candidate, to_string=True)
    except gdb.error:
        continue
    loaded = candidate
    break

# Whether the helper can actually read the interpreter.  This is the one
# discriminator between the two reasons py-bt yields nothing: a helper built
# for the wrong minor version, and a libpython stripped of DWARF.  Without
# type information gdb cannot resolve PyObject, and every frame walk the
# helper attempts fails at the first step.
types_ok = "no"
if loaded:
    try:
        gdb.lookup_type("PyObject")
        types_ok = "yes"
    except Exception:
        types_ok = "no"

print("LOADED_MARKER " + loaded)
print("TYPES_MARKER " + types_ok)
print("TRIED_MARKER " + "|".join(tried))
print("VERSION_MARKER " + version)
print("OBJECTS_MARKER " + "|".join(python_objects[:8]))
"""


def _python_helper_bootstrap_command(override: str = "", hint: str = "") -> str:
    """Build the single gdb command that loads the CPython helper.

    Rendered as ``python exec('…')`` so it stays one ``-ex`` argument: the
    phase runner pairs every command with a section marker and cannot carry a
    multi-line ``python … end`` block.

    Args:
        override: Explicit helper path or directory, or ``""`` to search only.
        hint: CPython minor version to fall back on when the core's object
            names do not carry one — see :func:`_python_version_hint`.

    Returns:
        A gdb command string.
    """
    source = (
        _PY_HELPER_BOOTSTRAP_SRC
        .replace("OVERRIDE", repr(override))
        .replace("HINT", repr(hint))
        .replace("NAMES", repr(list(PY_HELPER_NAMES)))
        .replace("LOADED_MARKER", PY_HELPER_LOADED_MARKER)
        .replace("TRIED_MARKER", PY_HELPER_TRIED_MARKER)
        .replace("VERSION_MARKER", PY_HELPER_VERSION_MARKER)
        .replace("OBJECTS_MARKER", PY_HELPER_OBJECTS_MARKER)
        .replace("TYPES_MARKER", PY_HELPER_TYPES_MARKER)
    )
    return "python exec(" + repr(source) + ")"


#: Files in the job directory likely to name the interpreter version.  The
#: payload's own stdout carries ``PYTHONPATH``, which on an ATLAS release
#: contains ``.../lib/python3.13/site-packages``.
PY_VERSION_HINT_FILES: tuple[str, ...] = ("payload.stdout", "payload.stderr", "my_release_setup.sh")

#: Bytes read from each hint file.  The version appears in setup output near
#: the start; reading a 244 KB stdout in full to find it would be wasteful.
PY_VERSION_HINT_BYTES = 512 * 1024

_PY_VERSION_RE = re.compile(r"python(\d+)\.(\d+)")


def _python_version_hint(job_dir: Path | None) -> str:
    """Guess the payload's CPython minor version from the job directory.

    A fallback for cores whose object names do not carry a version — a plain
    ``python`` executable, or an interpreter statically linked into something
    else. The job directory usually gives it away regardless: an ATLAS
    release's ``PYTHONPATH`` names ``lib/python3.13/site-packages`` and that
    lands in the payload's own stdout.

    A hint, not an authority: the bootstrap prefers what it reads from the core
    and only falls back to this.

    Args:
        job_dir: Reconstructed job directory, or ``None``.

    Returns:
        A ``"X.Y"`` string, or ``""`` when nothing suggested one.
    """
    if job_dir is None or not job_dir.is_dir():
        return ""
    for name in PY_VERSION_HINT_FILES:
        candidate = job_dir / name
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(PY_VERSION_HINT_BYTES)
        except OSError:
            continue
        found = _PY_VERSION_RE.search(text)
        if found:
            return f"{found.group(1)}.{found.group(2)}"
    return ""


def _parse_python_helper(text: str) -> dict[str, Any]:
    """Interpret the bootstrap's markers.

    Args:
        text: Output of the bootstrap section.

    Returns:
        ``{"loaded", "searched", "python_version"}``.  ``loaded`` is ``""``
        when no helper was found, which is a different fact from not having
        looked; ``python_version`` is the CPython minor version detected from
        the core, and selects which helper in a directory applies.
    """
    loaded = ""
    version = ""
    types_ok = False
    objects: list[str] = []
    searched: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(PY_HELPER_LOADED_MARKER):
            loaded = stripped[len(PY_HELPER_LOADED_MARKER):].strip()
        elif stripped.startswith(PY_HELPER_TRIED_MARKER):
            payload = stripped[len(PY_HELPER_TRIED_MARKER):].strip()
            searched = [item for item in payload.split("|") if item]
        elif stripped.startswith(PY_HELPER_VERSION_MARKER):
            version = stripped[len(PY_HELPER_VERSION_MARKER):].strip()
        elif stripped.startswith(PY_HELPER_TYPES_MARKER):
            types_ok = stripped[len(PY_HELPER_TYPES_MARKER):].strip() == "yes"
        elif stripped.startswith(PY_HELPER_OBJECTS_MARKER):
            payload = stripped[len(PY_HELPER_OBJECTS_MARKER):].strip()
            objects = [item for item in payload.split("|") if item]
    return {
        "loaded": loaded,
        "searched": searched,
        "python_version": version,
        "python_objects": objects,
        "interpreter_types": types_ok,
    }


def _last_error_line(text: str) -> str:
    """Return the most specific error line in container output.

    ALRB prints a message-of-the-day and, when it bails, its full command
    menu — so the last lines of a failed run are almost never the reason it
    failed.  The reason for job 7272161793 was a single line
    (``Error: unable to source setupfile /srv/my_release_setup.sh``) sitting
    ahead of roughly six kilobytes of ROOT security notices.

    The *last* match wins rather than the first: a nested failure reports the
    outermost cause last, and that is the one worth leading with.

    Args:
        text: Combined stdout and stderr from the container run.

    Returns:
        The matching line, stripped, or ``""`` when nothing matches.  An empty
        result means the caller falls back to the raw tail, which is the
        pre-existing behaviour.
    """
    matches = [line.strip() for line in text.splitlines() if _ERROR_LINE_RE.match(line)]
    return matches[-1] if matches else ""


def _stage_python_helper(override: str, job_dir: Path) -> tuple[str, list[Path]]:
    """Copy a CPython gdb helper into the job directory for the container.

    The container sees almost nothing of the host. ALRB binds ``/cvmfs``, the
    user's home, the working directory at ``/srv`` and little else, all with
    ``--contain``, so a helper anywhere outside those — ``/data/bamboo/tools``,
    for instance — simply does not exist as far as the bootstrap is concerned.
    Passing the host path through unchanged silently found nothing.

    Staging it under ``--job-dir`` reuses the one directory that is already
    mounted, and puts the copy in the same list the worker and runner scripts
    use, so it is cleaned up on success and kept on failure alongside them.

    A directory is copied as a tree of ``python-gdb.py`` files only, bounded in
    count and size: the point is to carry a handful of small helpers, not to
    mirror an arbitrary path into the job directory.

    Args:
        override: Host path to a helper file or a directory of per-version
            helpers, or ``""``.
        job_dir: The job directory, mounted at ``/srv``.

    Returns:
        ``(container_path, created)``. *container_path* is ``""`` when there
        was nothing usable to stage, in which case the bootstrap falls back to
        searching next to the core's own Python objects.
    """
    if not override:
        return "", []

    source = Path(override).expanduser()
    staged_root = job_dir / PY_HELPER_STAGE_DIR
    created: list[Path] = []

    if source.is_file():
        staged_root.mkdir(parents=True, exist_ok=True)
        target = staged_root / PY_HELPER_NAME
        shutil.copy2(source, target)
        created.extend([target, staged_root])
        return _container_path(target, job_dir), created

    if not source.is_dir():
        print(
            f"[!] BAMBOO_CORE_DUMP_PYTHON_GDB points at {source}, which is neither a "
            "file nor a directory; ignoring it.",
            file=sys.stderr,
        )
        return "", []

    matches: list[Path] = []
    for leaf in PY_HELPER_NAMES:
        matches.extend(source.rglob(leaf))

    copied = 0
    total = 0
    for candidate in sorted(set(matches)):
        if copied >= PY_HELPER_MAX_FILES:
            break
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size > PY_HELPER_MAX_BYTES:
            continue
        relative = candidate.relative_to(source)
        target = staged_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        created.append(target)
        copied += 1
        total += size

    if not copied:
        wanted = " or ".join(PY_HELPER_NAMES)
        print(
            f"[!] No {wanted} found under {source}; ignoring "
            "BAMBOO_CORE_DUMP_PYTHON_GDB.",
            file=sys.stderr,
        )
        return "", created

    # Deepest first, so directories are empty by the time they are unlinked.
    for directory in sorted(
        {path.parent for path in created}, key=lambda item: len(item.parts), reverse=True
    ):
        if directory not in created:
            created.append(directory)
    return _container_path(staged_root, job_dir), created


def _collect_evidence_atlas_container(
    args: argparse.Namespace, progress: bool = True, detail: bool = False,
) -> tuple[CoreEvidence, str]:
    """Run the deterministic collector once inside an ATLAS AlmaLinux container.

    The original PanDA ``container_script.sh`` is intentionally not executed.
    Instead, ``atlasLocalSetup.sh`` sets up the requested container and release,
    then runs an analyzer-owned worker that invokes this script in evidence-only
    local mode. LLM synthesis remains in the host process.
    """
    core_path = Path(args.core_file).expanduser().resolve()
    if not core_path.is_file():
        raise FileNotFoundError(f"Core file not found: {core_path}")
    job_dir = Path(getattr(args, "job_dir", None) or core_path.parent).expanduser().resolve()
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")
    core_in_container = _container_path(core_path, job_dir)

    release_value = getattr(args, "release_setup", None)
    release_setup = Path(release_value).expanduser().resolve() if release_value else job_dir / "my_release_setup.sh"
    if not release_setup.is_file():
        raise FileNotFoundError(
            f"Release setup not found: {release_setup}. Pass --release-setup or place my_release_setup.sh in --job-dir."
        )
    release_in_container = _container_path(release_setup, job_dir)

    alrb = Path(getattr(args, "atlas_local_root_base", DEFAULT_ATLAS_LOCAL_ROOT_BASE)).expanduser().resolve()
    atlas_setup = alrb / "user" / "atlasLocalSetup.sh"
    if not atlas_setup.is_file():
        raise FileNotFoundError(f"ATLAS Local Root Base setup not found: {atlas_setup}")

    created: list[Path] = []
    succeeded = False
    try:
        worker_fd, worker_name = tempfile.mkstemp(prefix=".core_dump_analyzer_worker_", suffix=".py", dir=job_dir)
        os.close(worker_fd)
        worker_path = Path(worker_name)
        shutil.copy2(Path(__file__).resolve(), worker_path)
        created.append(worker_path)

        json_fd, json_name = tempfile.mkstemp(prefix=".core_dump_analyzer_evidence_", suffix=".json", dir=job_dir)
        os.close(json_fd)
        json_path = Path(json_name)
        created.append(json_path)

        raw_fd, raw_name = tempfile.mkstemp(prefix=".core_dump_analyzer_gdb_", suffix=".txt", dir=job_dir)
        os.close(raw_fd)
        raw_path = Path(raw_name)
        created.append(raw_path)

        runner_fd, runner_name = tempfile.mkstemp(prefix=".core_dump_analyzer_runner_", suffix=".sh", dir=job_dir)
        os.close(runner_fd)
        runner_path = Path(runner_name)
        created.append(runner_path)

        worker_in_container = _container_path(worker_path, job_dir)
        json_in_container = _container_path(json_path, job_dir)
        raw_in_container = _container_path(raw_path, job_dir)
        runner_in_container = _container_path(runner_path, job_dir)

        # The override is a *host* path; the inner worker needs one the
        # container can see.  Stage it under the job directory and hand the
        # /srv form onward.
        helper_in_container, helper_created = _stage_python_helper(
            getattr(args, "python_gdb_helper", "") or "", job_dir
        )
        created.extend(helper_created)
        container_args = copy.copy(args)
        container_args.python_gdb_helper = helper_in_container

        worker_argv = _container_worker_args(
            container_args, core_in_container, worker_in_container,
            json_in_container, raw_in_container, job_dir,
        )
        runner_path.write_text(
            "#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(worker_argv) + "\n",
            encoding="utf-8",
        )
        runner_path.chmod(0o700)

        platform = getattr(args, "atlas_platform", DEFAULT_ATLAS_PLATFORM)
        extra_args = getattr(args, "container_extra_args", "-c -i")
        release_rel = release_setup.resolve().relative_to(job_dir.resolve()).as_posix()
        # BAMBOO PATCH (see module note on vendoring): `cd` into the job
        # directory *inside* the command string, not only via subprocess cwd.
        #
        # atlasLocalSetup.sh binds $PWD at /srv, and _container_path maps every
        # host path under --job-dir to /srv/<rel> on that assumption.  But this
        # runs under `bash -lc`, which sources /etc/profile and the user's
        # profile before it reaches this string — and on a CERN AFS account
        # that chain ends in the home directory, discarding the cwd the parent
        # set.  ALRB then bound $HOME at /srv and reported
        # "unable to source setupfile /srv/my_release_setup.sh", which reads as
        # a missing file and is in fact a wrong mount.
        #
        # The guard below is not redundant with `cd || exit 1`: it turns any
        # future drift between job_dir and the /srv assumption into one legible
        # line instead of ALRB's help menu.
        guard = (
            "test -f " + shlex.quote(release_rel) + " || { "
            "echo \"Error: the release setup is not in the working directory "
            "($PWD); the /srv mount would be wrong.\" >&2; "
            "exit 1; }"
        )
        source_cmd = (
            f"cd {shlex.quote(str(job_dir))} || exit 1; "
            f"{guard}; "
            f"export ATLAS_LOCAL_ROOT_BASE={shlex.quote(str(alrb))}; "
            f"source {shlex.quote(str(atlas_setup))} "
            f"-c {shlex.quote(platform)} "
            f"-s {shlex.quote(release_in_container)} "
            f"-r {shlex.quote(runner_in_container)} "
            f"-e {shlex.quote(extra_args)}"
        )
        if progress:
            print(f"[*] ATLAS container analysis starting ({platform}, job dir {job_dir})...", file=sys.stderr)
        started = time.monotonic()
        stop_event = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        if progress and detail:
            heartbeat_thread = threading.Thread(
                target=_report_heartbeat,
                args=("atlas-container", started, stop_event,
                      getattr(args, "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)),
                daemon=True,
            )
            heartbeat_thread.start()
        try:
            proc = subprocess.run(
                ["bash", "-lc", source_cmd], cwd=job_dir, capture_output=True, text=True, check=False,
                timeout=getattr(args, "container_timeout", DEFAULT_CONTAINER_TIMEOUT),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ATLAS container analysis timed out after "
                f"{getattr(args, 'container_timeout', DEFAULT_CONTAINER_TIMEOUT)}s"
            ) from exc
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
        if progress:
            print(f"[*] ATLAS container analysis completed in {time.monotonic() - started:.1f}s", file=sys.stderr)

        if proc.returncode != 0 or not json_path.stat().st_size:
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            headline = _last_error_line(combined)
            tail = combined[-CONTAINER_DIAGNOSTIC_CHARS:]
            message = (
                f"ATLAS container evidence collector failed with exit code {proc.returncode}."
            )
            if headline:
                message += f"\n{headline}"
            raise RuntimeError(f"{message}\n\n--- container output (tail) ---\n{tail}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        evidence = core_evidence_from_dict(payload["evidence"])
        evidence.environment["execution_backend"] = "atlas-container"
        evidence.environment["atlas_platform"] = platform
        evidence.environment["release_setup"] = str(release_setup)
        evidence.environment["job_dir"] = str(job_dir)
        evidence.core_file["container_path"] = core_in_container
        evidence.core_file["path"] = str(core_path)
        raw = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
        succeeded = True
        return evidence, raw
    finally:
        # Keep the staged worker, runner and evidence files after a failure.
        # They are the only record of the exact command ALRB was handed, and
        # deleting them on the way out left nothing to inspect for precisely
        # the runs that needed inspecting.
        if getattr(args, "keep_container_artifacts", False) or not succeeded:
            if not succeeded:
                print(
                    f"[!] Container artifacts kept in {job_dir} for debugging.",
                    file=sys.stderr,
                )
        else:
            for path in created:
                try:
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink()
                except OSError:
                    pass


def collect_evidence(
    args: argparse.Namespace, progress: bool = True, detail: bool = False,
) -> tuple[CoreEvidence, str]:
    """Collect core evidence, then optionally correlate bounded payload/job logs on the host."""
    execution = getattr(args, "execution", "local")
    if execution == "atlas-container":
        evidence, raw = _collect_evidence_atlas_container(args, progress=progress, detail=detail)
    else:
        evidence, raw = _collect_evidence_local(args, progress=progress, detail=detail)

    job_dir_value = getattr(args, "job_dir", None)
    collect_logs = getattr(args, "collect_job_logs", True)
    # The in-container worker is a local backend with no --job-dir, so it never
    # recursively scans logs. Correlation happens once, on the host, after the
    # matching-environment core evidence has been returned.
    if collect_logs and job_dir_value:
        job_dir = Path(job_dir_value).expanduser().resolve()
        if job_dir.is_dir():
            evidence.job_logs = collect_job_log_evidence(
                job_dir,
                explicit=getattr(args, "job_log", None),
                max_files=getattr(args, "max_job_log_files", DEFAULT_MAX_JOB_LOG_FILES),
                max_matches=getattr(args, "max_job_log_matches", DEFAULT_MAX_JOB_LOG_MATCHES),
                tail_lines=getattr(args, "job_log_tail_lines", DEFAULT_JOB_LOG_TAIL_LINES),
                redact_enabled=not args.no_redact,
                core_mtime=Path(args.core_file).expanduser().resolve().stat().st_mtime
                if Path(args.core_file).expanduser().resolve().is_file() else None,
                failure_mode=evidence.mode,
            )
            activity = evidence.job_logs.get("payload_activity", {})
            silence = activity.get("last_write_before_core_s")
            if isinstance(silence, (int, float)) and silence >= 300:
                observation = (
                    f"{activity.get('latest_payload_file', 'Payload log')} was last modified "
                    f"{activity.get('last_write_before_core_human', _format_duration(float(silence)))} before the core capture"
                )
                last_line = activity.get("last_nonempty_line")
                if isinstance(last_line, dict) and last_line.get("text"):
                    observation += f"; its last non-empty line was: {last_line['text']}"
                latest_progress = activity.get("latest_progress")
                if (isinstance(latest_progress, dict) and latest_progress.get("text")
                        and (not isinstance(last_line, dict) or latest_progress.get("line") != last_line.get("line"))):
                    observation += f"; latest retained progress: {latest_progress['text']}"
                evidence.observations.append(observation + ".")
            for item in derive_payload_log_observations(
                evidence.job_logs, evidence.primary_thread.get("backtrace", "")
            ):
                if item not in evidence.observations:
                    evidence.observations.append(item)
            if progress and evidence.job_logs.get("available"):
                print(
                    f"[*] Payload/job-log correlation: {len(evidence.job_logs.get('files', []))} file(s), "
                    f"{len(evidence.job_logs.get('matches', []))} relevant line(s)",
                    file=sys.stderr,
                )
    evidence.process_identity = derive_process_identity(evidence)
    evidence.diagnosis = derive_structured_diagnosis(evidence)
    return evidence, raw


#: The banner an ATLAS release prints on setup, e.g.::
#:
#:     Using AnalysisBase/25.2.103 [cmake] with platform x86_64-el9-gcc15-opt
#:             at /cvmfs/atlas.cern.ch/repo/sw/software/25.2
#:
#: Parsed rather than assumed because the release differs per job, and any
#: path derived from it — a debug-file directory in particular — is wrong for
#: every job but one if it is hardcoded.
_RELEASE_RE = re.compile(
    r"Using\s+(?P<project>[A-Za-z][\w.+-]*)/(?P<release>[\w.+-]+)\s*"
    r"(?:\[[^\]]*\]\s*)?with\s+platform\s+(?P<platform>\S+)"
)
_RELEASE_BASE_RE = re.compile(r"^\s*at\s+(?P<base>/\S+)\s*$", re.MULTILINE)


def _parse_release_info(job_dir: Path | None) -> dict[str, str]:
    """Read the ATLAS release, platform and CVMFS base from the payload log.

    Args:
        job_dir: Reconstructed job directory, or ``None``.

    Returns:
        ``{"project", "release", "platform", "base"}``, with missing pieces
        omitted.  Empty when the banner is not present, which is a normal
        outcome for a payload that is not an ATLAS release.
    """
    if job_dir is None or not job_dir.is_dir():
        return {}
    for name in PY_VERSION_HINT_FILES:
        candidate = job_dir / name
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(PY_VERSION_HINT_BYTES)
        except OSError:
            continue
        found = _RELEASE_RE.search(text)
        if not found:
            continue
        info = {k: v for k, v in found.groupdict().items() if v}
        base = _RELEASE_BASE_RE.search(text[found.end():found.end() + 500])
        if base:
            info["base"] = base.group("base")
        return info
    return {}


def _debug_file_init(
    args: argparse.Namespace, release: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Build the gdb settings that add a separate debug-file directory.

    ATLAS releases ship libpython and the XrdCl libraries stripped of DWARF,
    which is what stops ``py-bt`` and what leaves optimised frames without
    argument or local-variable data.  Where a site does publish matching
    ``.debug`` trees, pointing gdb at them recovers both.

    The setting is a **template**, not a path, because a fixed path is wrong
    for every job but one: ``/cvmfs/atlas.cern.ch/repo/sw/software`` holds a
    hundred-odd releases and each job names its own.  ``{project}``,
    ``{release}``, ``{platform}`` and ``{base}`` are substituted from the
    payload log's setup banner — for job 7272161793, ``AnalysisBase``,
    ``25.2.103``, ``x86_64-el9-gcc15-opt`` and
    ``/cvmfs/atlas.cern.ch/repo/sw/software/25.2``.

    The expanded directory must exist, and is silently dropped when it does
    not.  A ``set debug-file-directory`` naming a missing path is not an error
    in gdb, it simply has no effect, so validating here is the difference
    between a setting that quietly does nothing and one that says so.  The
    check is on the host, which is correct for the CVMFS paths this is for:
    the container sees the same ``/cvmfs``.

    Args:
        args: Parsed arguments.
        release: Result of :func:`_parse_release_info`, or ``None``.

    Returns:
        Settings for ``run_gdb_phase(extra_init=...)``, empty when unset,
        unexpandable, or pointing at a directory that is not there.
    """
    template = (getattr(args, "debug_file_directory", "") or "").strip()
    if not template:
        return ()

    fields = {"project": "", "release": "", "platform": "", "base": "", **(release or {})}
    try:
        directory = template.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        print(
            f"[!] --debug-file-directory template {template!r} could not be expanded ({exc}); "
            "ignoring it.",
            file=sys.stderr,
        )
        return ()

    if "{" in template and not all(fields[key] for key in ("project", "release", "platform")):
        print(
            "[!] --debug-file-directory uses placeholders but the payload log carried no "
            "release banner; ignoring it.",
            file=sys.stderr,
        )
        return ()

    if not Path(directory).is_dir():
        print(
            f"[!] --debug-file-directory resolved to {directory}, which does not exist; "
            "ignoring it. gdb would accept it silently and load no debug files.",
            file=sys.stderr,
        )
        return ()

    return (f"set debug-file-directory {directory}",)


def _summarise_python(
    text: str, source: str, stderr: str, redact_enabled: bool,
    helper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Interpret the output of the ``py-bt`` / ``py-list`` phase.

    Detection is positive rather than negative: real ``py-bt`` output always
    contains a Python traceback header or a ``File "..."`` frame. gdb reports an
    unavailable command on stderr, so that stream must be inspected too.

    Args:
        text: Output of ``py-bt``.
        source: Output of ``py-list``.
        stderr: Standard error of the Python phase.
        redact_enabled: Whether to scrub secrets.
        helper: Result of :func:`_parse_python_helper`, used to say *where* the
            CPython helper was looked for.  ``None`` when the bootstrap did not
            run, which is reported as such rather than as absence.

    Returns:
        A dictionary describing whether Python frames were available and, if so,
        the Python-level backtrace and surrounding source.
    """
    if "Undefined command" in stderr or "Undefined command" in text:
        reason = (
            "py-bt is not available because the libpython/CPython GDB helper "
            "(python-gdb.py) is not loaded. This is separate from native Python "
            "symbol availability or full DWARF debug information."
        )
        searched = list((helper or {}).get("searched") or [])
        if searched:
            shown = ", ".join(searched[:4])
            more = f" (+{len(searched) - 4} more)" if len(searched) > 4 else ""
            version = (helper or {}).get("python_version") or ""
            reason += (
                f" It was searched for and not found at: {shown}{more}. The release "
                "does not ship it next to its Python objects; pass "
                "--python-gdb-helper to point at one"
                + (
                    f", built for CPython {version} — the helper reads interpreter "
                    "structures that change between minor versions, so it must match."
                    if version else "."
                )
            )
        elif helper is not None:
            version = (helper or {}).get("python_version") or ""
            objects = (helper or {}).get("python_objects") or []
            if not objects and not version:
                reason += (
                    " No interpreter object was recognised among the core's loaded "
                    "objects and the job directory suggested no version, so there was "
                    "nowhere to look. Pass --python-gdb-helper with a path to a "
                    "specific helper file to bypass the search."
                )
            else:
                reason += (
                    f" Nothing was found to try for CPython {version or 'an unknown version'}"
                    + (f" (interpreter objects: {', '.join(objects[:3])})" if objects else "")
                    + "."
                )
        return {"available": False, "reason": reason}
    has_frames = bool(
        re.search(r"Traceback \(most recent call first\)", text)
        or re.search(r'^\s*File "[^"]+", line \d+, in ', text, re.M)
    )
    if not has_frames:
        info = helper or {}
        if info.get("loaded"):
            version = info.get("python_version") or ""
            if info.get("interpreter_types"):
                return {
                    "available": False,
                    "reason": (
                        f"The CPython helper at {info['loaded']} loaded and gdb can resolve "
                        "the interpreter's types, but py-bt still produced no Python frames. "
                        "The interpreter was most likely not executing Python code at the "
                        "point the core was taken — for a process captured inside a C-level "
                        "shutdown or wait, that is expected and not a gap in the evidence."
                    ),
                }
            return {
                "available": False,
                "reason": (
                    f"The CPython helper at {info['loaded']} loaded, but gdb cannot resolve "
                    "PyObject, so the helper has no interpreter structures to walk and py-bt "
                    "produces nothing. This build's libpython carries no DWARF debug "
                    "information"
                    + (f" (CPython {version})" if version else "")
                    + " — a property of how the release was packaged, not of the helper, "
                    "which matched and loaded correctly. Point --debug-file-directory at a "
                    "directory holding the matching .debug file if the release ships one; "
                    "otherwise no Python-level backtrace is obtainable from this core and "
                    "the C-level stacks are the whole of the evidence."
                ),
            }
        return {"available": False, "reason": "py-bt produced no Python frames; this is likely not a Python process."}

    backtrace, _ = truncate(redact(text.strip(), redact_enabled), SECTION_LIMITS["python_backtrace"])
    context, _ = truncate(redact(source.strip(), redact_enabled), SECTION_LIMITS["python_source"])
    return {"available": True, "backtrace": backtrace, "source_context": context}


def _serialized_size(evidence: CoreEvidence) -> int:
    """Return the character length of the evidence exactly as sent to the LLM.

    Uses the same ``indent=2`` formatting as :func:`build_user_prompt` so this
    is a faithful stand-in for the size of the real prompt, not just a proxy.

    Args:
        evidence: The assembled evidence.

    Returns:
        Length in characters of the JSON-serialised evidence.
    """
    return len(json.dumps(evidence.to_dict(), indent=2, default=str))


def _shrink_shared_libraries(evidence: CoreEvidence) -> bool:
    """Halve the list of symbol-less shared libraries, if any remain.

    Args:
        evidence: The assembled evidence, mutated in place.

    Returns:
        ``True`` if the list was shrunk, ``False`` if it was already empty.
    """
    libraries = evidence.shared_libraries.get("without_symbols") or []
    if not libraries:
        return False
    evidence.shared_libraries["without_symbols"] = libraries[: len(libraries) // 2]
    if "shared_libraries.without_symbols" not in evidence.truncated_sections:
        evidence.truncated_sections.append("shared_libraries.without_symbols")
    return True


def _pop_thread_group(evidence: CoreEvidence) -> bool:
    """Drop the least interesting remaining thread group.

    Groups are already sorted busy-before-idle (see :func:`group_thread_stacks`),
    so this always removes an idle group before a busy one, and always leaves
    at least one group so the model has *some* thread evidence.

    Args:
        evidence: The assembled evidence, mutated in place.

    Returns:
        ``True`` if a group was dropped, ``False`` if only one group remains.
    """
    if len(evidence.thread_groups) <= 1:
        return False
    evidence.thread_groups.pop()
    if "thread_groups" not in evidence.truncated_sections:
        evidence.truncated_sections.append("thread_groups")
    return True


def _drop_python_source(evidence: CoreEvidence) -> bool:
    """Remove the ``py-list`` source context, keeping the ``py-bt`` traceback.

    Args:
        evidence: The assembled evidence, mutated in place.

    Returns:
        ``True`` if source context was present and dropped, ``False`` otherwise.
    """
    if not evidence.python.get("source"):
        return False
    evidence.python["source"] = ""
    if "python.source" not in evidence.truncated_sections:
        evidence.truncated_sections.append("python.source")
    return True


def _shrink_text_field(
    container: dict[str, str], key: str, label: str, evidence: CoreEvidence, floor: int = 500,
) -> bool:
    """Halve a text field toward a floor, keeping head and tail.

    ``truncate()`` can never produce text shorter than ``limit + len(marker)``,
    since the marker itself is always inserted. The stopping condition compares
    against that true minimum rather than the bare floor, so this reliably
    terminates instead of "succeeding" forever at a length it can never reduce
    further -- which would hang :func:`enforce_global_budget` in an infinite
    loop on any evidence still over budget once a field reaches its floor.

    Args:
        container: The dict holding the field, e.g. ``evidence.primary_thread``.
        key: The key within ``container`` to shrink.
        label: Name recorded in ``evidence.truncated_sections`` when shrunk.
        evidence: The evidence bundle, for recording that truncation happened.
        floor: Minimum length to shrink toward; returns ``False`` once reached.

    Returns:
        ``True`` if the field was shrunk, ``False`` if absent or already at
        the floor.
    """
    body = container.get(key, "")
    min_len = floor + len(TRUNCATION_MARKER)
    if len(body) <= min_len:
        return False
    trimmed, _ = truncate(body, max(floor, len(body) // 2))
    container[key] = trimmed
    if label not in evidence.truncated_sections:
        evidence.truncated_sections.append(label)
    return True


def enforce_global_budget(evidence: CoreEvidence, limit: int, detail: bool = False) -> CoreEvidence:
    """Shrink the evidence bundle to fit a character budget for the LLM prompt.

    Applies a cascade of reduction stages, cheapest evidence first, moving to
    the next stage only once the current one stops helping (e.g. thread groups
    are down to one, or a text field has hit its floor). This exists because
    per-section limits (:data:`SECTION_LIMITS`) cap each field individually,
    but do not cap the bundle as a whole -- a job with both a huge ``locals``
    dump and many distinct thread stacks could previously exceed ``limit`` even
    after every thread group but one had been dropped.

    Stage order (least to most valuable evidence):
        1. ``shared_libraries.without_symbols``
        2. ``thread_groups`` (idle groups first, per their existing sort)
        3. ``python.source``
        4. ``primary_thread.locals``
        5. ``primary_thread.registers``
        6. ``primary_thread.args``
        7. ``python.backtrace``
        8. ``primary_thread.backtrace`` (last resort)

    Args:
        evidence: The assembled evidence.
        limit: Maximum total serialised size in characters.
        detail: Whether to log which sections were trimmed to stderr.

    Returns:
        The supplied evidence object, mutated to fit within ``limit`` wherever the
        cascade was able to. Callers should pass a disposable/deep-copied LLM input,
        never the canonical evidence artifact. See ``evidence.warnings`` for whether it fully
        succeeded.
    """
    stages: list[tuple[str, Callable[[], bool]]] = [
        ("shared_libraries.without_symbols", lambda: _shrink_shared_libraries(evidence)),
        ("thread_groups", lambda: _pop_thread_group(evidence)),
        ("python.source", lambda: _drop_python_source(evidence)),
        ("primary_thread.locals",
         lambda: _shrink_text_field(evidence.primary_thread, "locals", "primary_thread.locals", evidence)),
        ("primary_thread.registers",
         lambda: _shrink_text_field(evidence.primary_thread, "registers", "primary_thread.registers", evidence)),
        ("primary_thread.args",
         lambda: _shrink_text_field(evidence.primary_thread, "args", "primary_thread.args", evidence)),
        ("python.backtrace",
         lambda: _shrink_text_field(evidence.python, "backtrace", "python.backtrace", evidence)),
        ("primary_thread.backtrace",
         lambda: _shrink_text_field(
             evidence.primary_thread, "backtrace", "primary_thread.backtrace", evidence, floor=1000)),
    ]

    stage_index = 0
    while _serialized_size(evidence) > limit and stage_index < len(stages):
        _, shrink = stages[stage_index]
        if not shrink():
            stage_index += 1

    if _serialized_size(evidence) > limit:
        evidence.warnings.append(
            f"Evidence remains above the {limit}-character budget even after all reduction stages; "
            "sending it as-is. Consider raising --max-evidence-chars or lowering --max-thread-groups."
        )
    if detail and evidence.truncated_sections:
        print(
            f"[*] Evidence trimmed to fit the {limit}-character budget: {', '.join(evidence.truncated_sections)}",
            file=sys.stderr,
        )
    return evidence


# --------------------------------------------------------------------------- #
# LLM synthesis
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_BASE = """You are an expert in post-mortem debugging of large C++ and Python \
scientific applications, specifically ATLAS/Athena jobs running on distributed grid computing \
infrastructure.

You will be given structured evidence extracted from a core dump using gdb. Your audience is a \
computing operations shifter or a physicist who submitted the job. They do not read gdb output and \
do not care about frame numbers, register values or mangled symbol names. Translate, do not transcribe.

Hard rules:
- Never invent stack frames, function names, file names or line numbers. Use only what appears in the evidence.
- If symbols are missing, the core is truncated, or the executable did not resolve, say so FIRST and \
lower your confidence accordingly. A confident wrong answer is worse than an honest "insufficient evidence".
- Distinguish clearly between application code (Athena algorithms, physics code, user Python) and \
framework or system noise (TBB, GaudiHive, libc, pthread, the Python interpreter loop). The interesting \
frame is almost always the deepest one belonging to application code.
- A top frame in pthread_cond_wait, futex or epoll does not by itself make a thread irrelevant. Inspect deeper \
frames: a thread blocked while stopping a subsystem, acquiring a lock, handling a timeout, or finalizing can be \
central to a hang. Treat only genuinely parked worker-loop stacks as idle.
- "next_steps" must stay inside what this evidence supports: further diagnosis, a software or \
configuration change in the culprit component, or a check the reader can actually perform. Do NOT propose \
changes to how the PanDA pilot, the workload management system or site operations behave -- you are looking \
at one core file and have no evidence about those systems. In particular: a pilot that killed a looping job \
does not stage out the payload's outputs afterwards, and there is no user-facing error quota to exempt a job \
from. Inventing a plausible-sounding operational mechanism is a failure mode, not a helpful suggestion.
- A step that verifies an existing artifact must name that artifact and how to inspect it. Re-running the \
job produces a different artifact and verifies nothing about the one you are analysing; do not offer it as \
a check.
- Do not cite a configuration setting unless you can say what it does and how that bears on this failure. A \
setting that sounds related but governs something else sends the reader down a dead end with your \
authority behind it.
"""

SYSTEM_PROMPT_CRASH = """
This dump is being analysed as a CRASH. Focus on: the faulting thread, the signal, the faulting frame, \
the likely memory error (null dereference, use-after-free, buffer overrun, bad cast, stack exhaustion, \
uncaught C++ exception leading to abort), and which component owns the bug.
"""

SYSTEM_PROMPT_HANG = """
This dump is being analysed as a HANG or LOOPING JOB. The core was most likely produced deliberately by \
the pilot or a watchdog after the job exceeded its wall-clock or looping-job time limit, so there is no \
"crash" to explain.

Focus instead on: which thread is actually doing work, what that work is, and why it is not finishing. \
Look specifically for infinite or very slow loops, unbounded I/O waits, deadlocks (two threads each \
blocked on a lock the other holds), lock convoys, pathological allocation or garbage-collection behaviour, \
and a single-threaded bottleneck while the rest of the pool is idle. If a Python backtrace is present it is \
usually the most informative evidence available; lead with it.

Critical completion semantics for PanDA looping jobs:
- "worker finished successfully" and "current job status: ... success" are EventLoop/application-level markers. \
They do NOT prove that the payload process exited or that the PanDA job completed successfully.
- If the deterministic diagnosis says the process hung during shutdown, the overall outcome is still a HANG: \
event processing may have finished, but the payload/job did not complete normally. Never describe such a job \
as having run completely or successfully.
- Treat evidence.diagnosis.job_completion and evidence.diagnosis.summary as authoritative constraints on this distinction.

Rules for claims about locks, and they are strict, because a lock cycle is the easiest thing in a core \
dump to describe confidently and wrongly:
- A thread blocked acquiring a lock does NOT hold that lock. Never write a cycle in which the same thread \
both holds a mutex and waits for it -- that is not a deadlock, it is a contradiction, and it tells the \
reader nothing.
- When you say a thread waits on a lock, name the thread you believe holds it AND the frame that shows it \
holding it. A frame entering a function documented to take the lock counts; a frame merely blocked inside \
the acquire path does not.
- If no frame establishes ownership, say the holder is unidentified. That is an honest and useful answer. \
Assigning ownership you cannot support contradicts your own limitations section and is worse than saying \
nothing.
- Prefer the smallest cycle the frames support. One thread that blocks waiting for an acknowledgement only \
it could send -- a poller thread waiting inside its own shutdown path, say -- is a self-deadlock, and is \
both simpler and more common than a three-way cycle. Other threads blocked behind it are consequences, not \
participants. Reach for a multi-thread cycle only when the frames force it.
"""

RESPONSE_SCHEMA = """
Return ONLY a JSON object, with no preamble, commentary or Markdown fences. Use exactly these keys:

{
  "verdict": "one sentence, plain language, what happened",
  "classification": "crash" | "hang" | "deadlock" | "resource_exhaustion" | "undetermined",
  "confidence": "high" | "medium" | "low",
  "confidence_reason": "one sentence on what limits or supports your confidence",
  "likely_cause": "2-4 sentences explaining the most probable cause in plain language",
  "supporting_evidence": ["specific frames or observations from the evidence that support the verdict"],
  "culprit_component": "the software component or package most likely responsible, or 'unknown'",
  "busy_threads": "short description of what the non-idle threads were doing",
  "limitations": ["anything that weakened this analysis, e.g. missing symbols, truncated core"],
  "next_steps": ["concrete, ordered actions the operator or job owner should take"],
  "explanation": "a longer plain-language narrative, 1-3 short paragraphs, safe to show a non-expert"
}
"""


def build_system_prompt(mode: str) -> str:
    """Assemble the system prompt for the requested analysis mode.

    Args:
        mode: Either ``"crash"`` or ``"hang"``.

    Returns:
        The full system prompt text.
    """
    specific = SYSTEM_PROMPT_CRASH if mode == "crash" else SYSTEM_PROMPT_HANG
    return SYSTEM_PROMPT_BASE + specific + RESPONSE_SCHEMA


def build_user_prompt(evidence: CoreEvidence) -> str:
    """Render the evidence bundle into the user message.

    Args:
        evidence: The assembled evidence.

    Returns:
        The user message text.
    """
    diagnosis = evidence.diagnosis if isinstance(evidence.diagnosis, dict) else {}
    contract = ""
    classification = str(diagnosis.get("classification", ""))
    if evidence.mode == "hang" and diagnosis.get("available") and "hang" in classification:
        completion = diagnosis.get("job_completion", {})
        contract = (
            "SYNTHESIS CONSTRAINTS (deterministic evidence; do not override):\n"
            f"- Deterministic classification: {classification}.\n"
            f"- Deterministic summary: {diagnosis.get('summary', '')}\n"
            f"- Job completion facts: {json.dumps(completion, sort_keys=True)}\n"
            f"- Root cause established: {bool(diagnosis.get('root_cause_established', False))}."
            " Do not promote a correlated timeout, socket fault, or lock wait into a proven"
            " initiating cause unless this is true.\n"
            "- EventLoop success markers describe event-processing/application state only. They are NOT proof of normal payload exit or PanDA job success.\n"
            "- The core exists because the still-running payload was captured in a hang. Your verdict, likely_cause, and explanation must all preserve that fact.\n\n"
        )
    return (
        contract
        + "Here is the gdb evidence extracted from the core dump. Analyse it and respond with the "
        "JSON object described in your instructions.\n\n"
        f"```json\n{json.dumps(evidence.to_dict(), indent=2, default=str)}\n```"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of a model response.

    Args:
        text: The raw model response, possibly wrapped in Markdown fences.

    Returns:
        The parsed object, or ``None`` if no valid JSON object was found.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _deterministic_hang_verdict(diagnosis: dict[str, Any]) -> str:
    """Return a user-facing verdict that cannot confuse EventLoop success with job success."""
    subtype = diagnosis.get("subtype")
    if subtype == "remote-file-close":
        return (
            "Event processing reached its successful end state, but the payload did not complete: "
            "it hung during ROOT/XRootD remote-file closure in process shutdown."
        )
    if subtype == "poller-finalization":
        return (
            "Event processing reached its successful end state, but the payload did not complete: "
            "it hung during XRootD/XrdCl shutdown finalization."
        )
    return "The payload did not complete normally; it was captured still running in a shutdown hang."


def _claims_normal_job_success(value: Any) -> bool:
    """Detect an LLM claim that the overall payload/job completed successfully."""
    if not isinstance(value, str):
        return False
    text = " ".join(value.lower().split())
    patterns = (
        r"\b(?:job|payload|process)\b.{0,80}\b(?:completed|finished|ran|exited|terminated)\b.{0,40}\bsuccess",
        r"\b(?:completed|finished|ran|exited|terminated)\b.{0,40}\bsuccess.{0,80}\b(?:job|payload|process)\b",
        r"\b(?:job|payload|process)\b.{0,80}\bcompleted normally\b",
        r"\b(?:job|payload|process)\b.{0,80}\b(?:succeeded|successful)\b",
        r"\b(?:job|payload|process)\b.{0,80}\bran to completion\b",
        r"\bno (?:hang|failure|problem)\b",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _safe_hang_explanation(diagnosis: dict[str, Any]) -> str:
    """Build a conservative fallback narrative from deterministic diagnosis fields."""
    summary = str(diagnosis.get("summary") or _deterministic_hang_verdict(diagnosis))
    limitations = diagnosis.get("limitations") or []
    caveat = ""
    if limitations:
        caveat = " " + str(limitations[0])
    return summary + caveat


def reconcile_llm_analysis(evidence: CoreEvidence, analysis: dict[str, Any]) -> dict[str, Any]:
    """Keep LLM prose subordinate to deterministic job-outcome evidence.

    The model is an explanation layer, not a second classifier.  In particular,
    EventLoop completion markers must never be promoted into a claim that a
    looping PanDA payload exited normally or that the overall job succeeded.
    """
    diagnosis = evidence.diagnosis if isinstance(evidence.diagnosis, dict) else {}
    classification = str(diagnosis.get("classification", ""))
    if not (evidence.mode == "hang" and diagnosis.get("available") and "hang" in classification):
        return analysis

    analysis["classification"] = "hang"
    analysis["verdict"] = _deterministic_hang_verdict(diagnosis)

    confidence_order = {"low": 0, "medium": 1, "high": 2}
    deterministic_confidence = str(diagnosis.get("confidence", "low"))
    llm_confidence = str(analysis.get("confidence", "low"))
    if confidence_order.get(llm_confidence, 0) > confidence_order.get(deterministic_confidence, 0):
        analysis["confidence"] = deterministic_confidence
        analysis["confidence_reason"] = (
            "Confidence is capped by the deterministic diagnosis and its evidence-quality limitations."
        )

    contradictory_fields = [
        key for key in ("likely_cause", "busy_threads", "explanation")
        if _claims_normal_job_success(analysis.get(key))
    ]
    for key in contradictory_fields:
        analysis[key] = _safe_hang_explanation(diagnosis)

    for key in ("supporting_evidence", "limitations", "next_steps"):
        value = analysis.get(key)
        if isinstance(value, list):
            analysis[key] = [item for item in value if not _claims_normal_job_success(item)]

    if contradictory_fields:
        analysis.setdefault("limitations", [])
        if isinstance(analysis["limitations"], list):
            analysis["limitations"].append(
                "LLM text that contradicted the deterministic hang outcome was discarded."
            )
    return analysis


def _cap_user_prompt(prompt: str, max_evidence_chars: int, evidence: CoreEvidence) -> str:
    """Apply a hard, last-resort ceiling to the rendered user prompt.

    :func:`enforce_global_budget` should already have brought the evidence
    under ``max_evidence_chars`` before this is ever called. This is a second,
    independent check applied to the actual prompt text right before it goes
    over the wire: defense-in-depth so that a future evidence field, or a call
    site that forgets to run the budget pass, can never send an unbounded (and
    unboundedly expensive) prompt to the API.

    Args:
        prompt: The fully rendered user message.
        max_evidence_chars: The evidence budget the caller intended to enforce.
        evidence: The evidence bundle, so a warning can be recorded if this
            cap actually had to do something.

    Returns:
        ``prompt``, unchanged if already within the hard cap, otherwise
        truncated to it.
    """
    hard_cap = max_evidence_chars * HARD_CAP_MULTIPLIER
    if len(prompt) <= hard_cap:
        return prompt
    evidence.warnings.append(
        f"The rendered LLM prompt ({len(prompt):,} chars) exceeded the hard {hard_cap:,}-char cost cap "
        f"({HARD_CAP_MULTIPLIER}x --max-evidence-chars) even after evidence reduction, and was truncated "
        "before being sent. This should not normally happen; if it does routinely, lower "
        "--max-thread-groups or investigate what is making the evidence so large."
    )
    capped, _ = truncate(prompt, hard_cap, marker="\n... [TRUNCATED FOR COST PROTECTION] ...")
    return capped


#: Signature of an LLM completion backend. Takes the system prompt, the user
#: prompt, an optional model override and a token ceiling; returns the response
#: text plus call metadata.
LLMCompletion = Callable[[str, str, str | None, int], tuple[str, dict[str, Any]]]

#: Backend names accepted by ``--llm-backend``.
LLM_BACKENDS: tuple[str, ...] = ("auto", "bamboo", "anthropic")


def _complete_via_anthropic(system: str, user: str, model: str | None,
                            max_tokens: int) -> tuple[str, dict[str, Any]]:
    """Complete a prompt with the Anthropic SDK directly.

    This is the fallback for standalone use on a host where Bamboo is not
    installed. It is the only path in this module that names a specific
    provider.

    Args:
        system: System prompt.
        user: User prompt.
        model: Model identifier, or ``None`` to use :data:`DEFAULT_MODEL`.
        max_tokens: Maximum response tokens.

    Returns:
        Tuple of ``(response_text, meta)``.

    Raises:
        RuntimeError: If the SDK is missing, the key is unset, or the call fails.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is not installed. Run: pip install -r requirements.txt, "
            "or use --llm-backend bamboo to go through Bamboo's configured provider."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it, or run with --no-llm.")

    resolved = model or DEFAULT_MODEL
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=resolved,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/transport failure uniformly
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

    # Narrow on ``block.type`` rather than ``getattr(block, "type", "")``:
    # the SDK's content blocks are a discriminated union keyed on a Literal
    # ``type``, so the direct comparison tells a type checker that only
    # ``TextBlock`` survives the filter.  Through ``getattr`` it cannot, and
    # every non-text member of the union is reported as lacking ``.text``.
    text = "".join(block.text for block in response.content if block.type == "text")
    meta = {
        "backend": "anthropic",
        "provider": "anthropic",
        "model": resolved,
        "input_tokens": getattr(response.usage, "input_tokens", None),
        "output_tokens": getattr(response.usage, "output_tokens", None),
        "stop_reason": getattr(response, "stop_reason", None),
    }
    return text, meta


def _complete_via_bamboo(system: str, user: str, model: str | None,
                         max_tokens: int) -> tuple[str, dict[str, Any]]:
    """Complete a prompt through Bamboo's configured LLM provider.

    Whichever provider Bamboo is set up for — Anthropic, OpenAI, Gemini,
    Mistral or an OpenAI-compatible endpoint — is used, so a standalone run
    from a Bamboo checkout does not need a second, analyzer-specific API key.
    The reasoning profile is selected because core-dump synthesis is the same
    class of work as log analysis.

    Args:
        system: System prompt.
        user: User prompt.
        model: Model identifier overriding the profile's model, or ``None`` to
            use whatever the profile specifies.
        max_tokens: Maximum response tokens.

    Returns:
        Tuple of ``(response_text, meta)``.

    Raises:
        RuntimeError: If Bamboo is not importable, an event loop is already
            running, or the provider call fails.
    """
    try:
        import asyncio
        import dataclasses as _dataclasses

        from bamboo.llm.config_loader import build_model_registry_from_config
        from bamboo.llm.factory import build_client
        from bamboo.llm.selector import LLMSelector
        from bamboo.llm.types import GenerateParams
    except ImportError as exc:
        raise RuntimeError(
            "Bamboo is not importable, so --llm-backend bamboo is unavailable. "
            "Use --llm-backend anthropic, or run with --no-llm."
        ) from exc

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "The bamboo LLM backend cannot be used from inside a running event loop. "
            "An async caller should collect evidence with --no-llm and synthesise "
            "through its own provider stack instead."
        )

    registry = build_model_registry_from_config(object())
    spec = LLMSelector(registry=registry).select("log_analysis")
    if model:
        spec = _dataclasses.replace(spec, model=model)

    client = build_client(spec)

    async def _run() -> Any:
        try:
            return await client.generate(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                GenerateParams(max_tokens=max_tokens),
            )
        finally:
            await client.close()

    try:
        response = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface any provider failure uniformly
        raise RuntimeError(f"{spec.provider} API call failed: {exc}") from exc

    usage = getattr(response, "usage", None)
    meta = {
        "backend": "bamboo",
        "provider": spec.provider,
        "model": spec.model,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "stop_reason": None,
    }
    return getattr(response, "text", "") or "", meta


def resolve_llm_backend(name: str = "auto") -> tuple[str, LLMCompletion]:
    """Select the completion backend for synthesis.

    ``auto`` prefers Bamboo whenever it is importable, so that a run from a
    Bamboo checkout uses the provider that installation is already configured
    and credentialed for, and falls back to the Anthropic SDK otherwise.

    Args:
        name: One of :data:`LLM_BACKENDS`.

    Returns:
        Tuple of ``(resolved_backend_name, completion_callable)``.

    Raises:
        RuntimeError: If *name* is not a known backend.
    """
    if name == "anthropic":
        return "anthropic", _complete_via_anthropic
    if name == "bamboo":
        return "bamboo", _complete_via_bamboo
    if name != "auto":
        raise RuntimeError(f"Unknown LLM backend: {name}. Choose one of {', '.join(LLM_BACKENDS)}.")
    try:
        import bamboo.llm.factory  # noqa: F401
    except ImportError:
        return "anthropic", _complete_via_anthropic
    return "bamboo", _complete_via_bamboo


def analyze_with_llm(
    evidence: CoreEvidence,
    model: str | None,
    max_tokens: int,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    progress: bool = True,
    detail: bool = False,
    backend: str = "auto",
) -> dict[str, Any]:
    """Send the evidence to an LLM and parse the structured verdict.

    The provider is not fixed: synthesis goes through whichever backend
    :func:`resolve_llm_backend` selects. The reconciliation step afterwards is
    not optional — it is what stops the model reading EventLoop completion
    markers as evidence that a looping PanDA job succeeded — so any alternative
    synthesis path must apply :func:`reconcile_llm_analysis` too.

    Args:
        evidence: The assembled evidence.
        model: Model identifier, or ``None`` to let the backend use its
            configured default.
        max_tokens: Maximum tokens for the response.
        max_evidence_chars: The evidence budget already applied by
            :func:`enforce_global_budget`, reused here to derive a hard cost
            ceiling on the actual outgoing prompt (see :func:`_cap_user_prompt`).
            Defaults to :data:`DEFAULT_MAX_EVIDENCE_CHARS` so this function
            remains usable on its own, without requiring every caller to
            thread the CLI's budget value through explicitly.
        progress: Whether to print a line before and after the API call.
        detail: Whether to also log the outgoing prompt size and a rough
            estimated token count before the call. The estimate is a simple
            ``chars / 4`` heuristic, not a real tokenizer -- good enough to
            catch an unexpectedly huge payload, not for billing.
        backend: Backend selector; see :data:`LLM_BACKENDS`.

    Returns:
        A dictionary with the parsed analysis plus ``_meta`` describing the call.

    Raises:
        RuntimeError: If no backend is usable or the call fails.
    """
    backend_name, complete = resolve_llm_backend(backend)
    user_prompt = _cap_user_prompt(build_user_prompt(evidence), max_evidence_chars, evidence)

    if detail:
        estimated_tokens = len(user_prompt) // CHARS_PER_TOKEN_ESTIMATE
        print(
            f"[*] Evidence prompt: {len(user_prompt):,} chars (~{estimated_tokens:,} est. input tokens; "
            "rough chars/4 heuristic, not exact)",
            file=sys.stderr,
        )
    if progress:
        print(
            f"[*] Querying {model or 'the configured model'} via {backend_name} "
            f"({evidence.mode} mode)...",
            file=sys.stderr,
        )

    text, meta = complete(build_system_prompt(evidence.mode), user_prompt, model, max_tokens)
    meta["mode"] = evidence.mode
    parsed = extract_json_object(text)
    if progress:
        print(
            f"[*] Response received from {meta.get('provider')}/{meta.get('model')} "
            f"(in: {meta['input_tokens']} tok, out: {meta['output_tokens']} tok)",
            file=sys.stderr,
        )
    if parsed is None:
        return {"verdict": "The model did not return parsable JSON.", "explanation": text, "_meta": meta}
    parsed = reconcile_llm_analysis(evidence, parsed)
    parsed["_meta"] = meta
    return parsed


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _bullets(values: Any, indent: str = "  ") -> str:
    """Render a value as an indented bullet list.

    Args:
        values: A list of strings, a single string, or ``None``.
        indent: Leading whitespace for each bullet.

    Returns:
        The formatted bullet list, or an empty string.
    """
    if not values:
        return ""
    items = values if isinstance(values, list) else [values]
    return "\n".join(f"{indent}- {item}" for item in items)


def _append_deterministic_diagnosis(lines: list[str], evidence: CoreEvidence) -> None:
    """Append deterministic diagnosis before any optional LLM explanation."""
    if not evidence.diagnosis.get("available"):
        return
    diagnosis = evidence.diagnosis
    lines += ["DETERMINISTIC DIAGNOSIS", "-" * 78]
    lines.append(f"  Classification: {diagnosis.get('classification', 'unclassified')}")
    if diagnosis.get("family"):
        lines.append(f"  Family        : {diagnosis.get('family')}")
    if diagnosis.get("subtype"):
        lines.append(f"  Subtype       : {diagnosis.get('subtype')}")
    lines.append(f"  Phase         : {diagnosis.get('phase', 'unknown')}")
    lines.append(f"  Component     : {diagnosis.get('component', 'unknown')}")
    lines.append(f"  Confidence    : {diagnosis.get('confidence', 'unknown')}")
    lines.append(f"  Root cause    : {'established' if diagnosis.get('root_cause_established') else 'not established'}")
    completion = diagnosis.get("job_completion")
    if isinstance(completion, dict):
        event_done = "yes" if completion.get("event_processing_completed") else "no"
        exited = "yes" if completion.get("payload_process_exited_normally") else "no"
        completed = "yes" if completion.get("job_completed_normally") else "no"
        lines.append(f"  EventLoop end : {event_done}")
        lines.append(f"  Payload exit  : {exited}")
        lines.append(f"  Normal job end: {completed}")
    if diagnosis.get("summary"):
        lines.append(f"  Summary       : {diagnosis.get('summary')}")
    if diagnosis.get("limitations"):
        lines.append("  Limitations:")
        lines.extend(f"    - {item}" for item in diagnosis.get("limitations", []))
    lines.append("")


def _render_header(evidence: CoreEvidence) -> list[str]:
    """Render the fixed-width header block of the report.

    Args:
        evidence: The assembled evidence.

    Returns:
        Header lines, ending with a blank separator.
    """
    lines = [
        f"Core file    : {evidence.core_file.get('path')} ({evidence.core_file.get('size_human')})",
        f"Executable   : {evidence.executable.get('path') or 'UNRESOLVED'} "
        f"(via {evidence.executable.get('source')})",
        f"Signal       : {evidence.signal or 'none recorded'}",
        f"Threads      : {evidence.thread_count if evidence.thread_count is not None else 'unknown'}",
        f"Analysis mode: {evidence.mode} ({evidence.mode_source})",
    ]
    if evidence.environment:
        backend = evidence.environment.get("execution_backend", "local")
        os_name = evidence.environment.get("os", "unknown")
        lines.append(f"Environment  : {os_name} ({backend})")
    if evidence.build_ids.get("available"):
        checked = len(evidence.build_ids.get("checked", []))
        mismatches = evidence.build_ids.get("mismatch_count", 0)
        if checked:
            lines.append(f"Build IDs    : {checked} key module(s) checked, {mismatches} mismatch(es)")
        else:
            lines.append(
                f"Build IDs    : UNVERIFIED (eu-unstrip enumerated "
                f"{evidence.build_ids.get('module_count', 0)} module(s); 0 key modules checked)"
            )
    if evidence.process_identity:
        lines.append(
            f"Core process : {evidence.process_identity.get('kind', 'unknown')} "
            f"({evidence.process_identity.get('confidence', 'low')} confidence)"
        )
    if evidence.generated_by:
        lines.append(f"Generated by : {evidence.generated_by}")
    if evidence.python.get("available"):
        lines.append("Python frames: available (py-bt)")
    lines.append("")
    return lines


def _render_thread_summary(evidence: CoreEvidence) -> list[str]:
    """Render the per-thread-group summary table.

    Args:
        evidence: The assembled evidence.

    Returns:
        Report lines for the thread summary section.
    """
    lines = ["THREAD SUMMARY", "-" * 78]
    for group in evidence.thread_groups[:10]:
        state = "BUSY" if group.state == "active" else ("BLOCKED" if group.state == "blocked" else "idle")
        tids = ",".join(group.thread_ids[:3])
        context = _thread_context_frame(group.backtrace)
        lines.append(f"  [{state}] {group.count:>4} thread(s) T{tids}: {context[:115]}")
    lines.append("")
    return lines


def _render_targeted_threads(evidence: CoreEvidence) -> list[str]:
    """Render focused frame/args/locals evidence for the selected threads.

    Args:
        evidence: The assembled evidence.

    Returns:
        Report lines, empty when no threads were targeted.
    """
    if not evidence.targeted_threads:
        return []
    lines = ["TARGETED FRAME EVIDENCE", "-" * 78]
    unavailable = 0
    for target in evidence.targeted_threads:
        lines.append(
            f"  T{target.get('thread_id')} frame {target.get('frame')} "
            f"[{str(target.get('state', '')).upper()}]: {str(target.get('context', '?'))[:105]}"
        )
        if target.get("frame_details_available", True):
            for label in ("args", "locals"):
                value = str(target.get(label, "")).strip()
                if value:
                    compact = " | ".join(line.strip() for line in value.splitlines() if line.strip())
                    lines.append(f"    {label}: {compact[:220]}")
        else:
            unavailable += 1
    if unavailable:
        lines.append(
            f"  Note: arguments/locals were unavailable for {unavailable} selected frame(s); "
            "this can occur in optimized functions even when GDB reports symbols read for the library."
        )
    lines.append("")
    return lines


def _render_payload_activity(activity: dict[str, Any]) -> list[str]:
    """Render the payload-activity summary and its retained tail.

    The "last modified N before core capture" figure is the headline evidence
    for a looping job, so it is rendered even when no keyword matched.

    Args:
        activity: The ``payload_activity`` sub-dict of the job-log evidence.

    Returns:
        Report lines, empty when no payload activity was recorded.
    """
    if not activity:
        return []
    line = (
        f"  Payload activity: {activity.get('latest_payload_file', '?')} last modified "
        f"{activity.get('last_write_before_core_human', '?')} before core capture"
    )
    last_line = activity.get("last_nonempty_line")
    if isinstance(last_line, dict) and last_line.get("text"):
        line += f"; last non-empty line {last_line.get('line', '?')}: {str(last_line.get('text'))[:120]}"
    latest_progress = activity.get("latest_progress")
    if (isinstance(latest_progress, dict) and latest_progress.get("text")
            and (not isinstance(last_line, dict) or latest_progress.get("line") != last_line.get("line"))):
        line += f"; latest retained progress: {str(latest_progress.get('text'))[:120]}"
    lines = [line]
    tail = activity.get("tail")
    if isinstance(tail, list) and tail:
        lines.append("  Payload tail (last non-empty lines):")
        for item in tail[-8:]:
            if isinstance(item, dict):
                lines.append(
                    f"    {activity.get('latest_payload_file', '?')}:{item.get('line', '?')}: "
                    f"{str(item.get('text', ''))[:170]}"
                )
    return lines


def _render_job_logs(evidence: CoreEvidence) -> list[str]:
    """Render the payload/job log correlation section.

    Args:
        evidence: The assembled evidence.

    Returns:
        Report lines, empty when no job logs were scanned.
    """
    if not evidence.job_logs.get("available"):
        return []
    profile = evidence.job_logs.get("profile", "general")
    title = "PAYLOAD LOG CORRELATION" if profile == "payload-centric" else "JOB LOG CORRELATION"
    lines = [title, "-" * 78]
    counts = evidence.job_logs.get("category_counts", {})
    matches = evidence.job_logs.get("matches", [])
    lines.append(
        "  Scanned: " + str(len(evidence.job_logs.get("files", []))) +
        " file(s); retained relevant lines: " + str(len(matches)) +
        (f" ({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})" if counts else "")
    )
    if evidence.job_logs.get("pilotlog_default_excluded"):
        lines.append(
            "  Scope: payload stdout/stderr plus log-like files under workDir; "
            "pilotlog.txt excluded for hang mode."
        )
    lines += _render_payload_activity(evidence.job_logs.get("payload_activity", {}))
    for match in matches[:12]:
        display_file = match.get("relative_file") or Path(str(match.get("file", "?"))).name
        lines.append(
            f"  [{str(match.get('category', '?')).upper()}] {display_file}:"
            f"{match.get('line', '?')}: {str(match.get('text', ''))[:180]}"
        )
    if len(matches) > 12:
        lines.append("  ... additional matched lines are retained in JSON.")
    lines.append("")
    return lines


def _render_analysis(analysis: dict[str, Any]) -> list[str]:
    """Render the LLM analysis sections.

    Args:
        analysis: The reconciled analysis dict.

    Returns:
        Report lines for the analysis, ending with the model/token footer.
    """
    lines: list[str] = []
    sections: list[tuple[str, Any]] = [
        ("VERDICT", analysis.get("verdict")),
        ("CLASSIFICATION", f"{analysis.get('classification', '?')} "
                           f"(confidence: {analysis.get('confidence', '?')}"
                           f" - {analysis.get('confidence_reason', '')})"),
        ("LIKELY CAUSE", analysis.get("likely_cause")),
        ("CULPRIT COMPONENT", analysis.get("culprit_component")),
        ("BUSY THREADS", analysis.get("busy_threads")),
    ]
    for title, body in sections:
        if body:
            lines += [title, "-" * 78, str(body), ""]

    for title, key in (("SUPPORTING EVIDENCE", "supporting_evidence"),
                       ("LIMITATIONS", "limitations"),
                       ("NEXT STEPS", "next_steps")):
        rendered = _bullets(analysis.get(key))
        if rendered:
            lines += [title, "-" * 78, rendered, ""]

    if analysis.get("explanation"):
        lines += ["EXPLANATION", "-" * 78, str(analysis["explanation"]), ""]

    meta = analysis.get("_meta", {})
    lines.append(f"[model: {meta.get('model')} | in: {meta.get('input_tokens')} tok | "
                 f"out: {meta.get('output_tokens')} tok]")
    return lines


def render_report(evidence: CoreEvidence, analysis: dict[str, Any] | None) -> str:
    """Build the human-readable report printed to stdout.

    This is the CLI's own fixed-width presentation.  Callers embedding the
    analyzer in another interface should render from the evidence and analysis
    dicts directly rather than reusing this text.

    Args:
        evidence: The assembled evidence.
        analysis: The parsed LLM analysis, or ``None`` when ``--no-llm`` is used.

    Returns:
        The formatted report text.
    """
    rule = "=" * 78
    lines = [rule, "CORE DUMP ANALYSIS", rule, ""]
    lines += _render_header(evidence)

    if evidence.warnings:
        lines += ["EVIDENCE QUALITY WARNINGS", "-" * 78, _bullets(evidence.warnings), ""]

    if analysis is None:
        lines += ["(--no-llm: showing extracted evidence only)", ""]

    _append_deterministic_diagnosis(lines, evidence)

    if analysis is None:
        if evidence.observations:
            lines += ["DETERMINISTIC OBSERVATIONS", "-" * 78, _bullets(evidence.observations), ""]
        lines += _render_thread_summary(evidence)
        lines += _render_targeted_threads(evidence)
        lines += _render_job_logs(evidence)
        return "\n".join(lines)

    lines += _render_analysis(analysis)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector to parse, or ``None`` to use ``sys.argv``.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="analyze_core_dump.py",
        description="Analyze a core dump with gdb and explain it in plain language using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment:\n"
            "  ANTHROPIC_API_KEY     required for --llm-backend anthropic, unless --no-llm\n"
            "  LLM_DEFAULT_PROVIDER  provider for --llm-backend bamboo (anthropic, openai,\n"
            "                        gemini, mistral, openai_compat); the matching provider key\n"
            "                        variable must also be set\n"
            "  CORE_ANALYSIS_MODEL   model override (falls back to LLM_DEFAULT_MODEL, then to\n"
            f"                        the backend's own default; {DEFAULT_MODEL} for anthropic)\n"
        ),
    )
    parser.add_argument("core_file", help="Path to the core dump file, e.g. core.123456")
    parser.add_argument(
        "--execution", choices=["local", "atlas-container"], default="local",
        help="Where to collect deterministic evidence. 'local' uses the current OS; "
             "'atlas-container' recreates the ATLAS release/container environment.",
    )
    parser.add_argument("--job-dir", default=None,
                        help="PanDA job directory mounted as /srv in atlas-container mode "
                             "(default: directory containing the core).")
    parser.add_argument("--release-setup", default=None,
                        help="Release setup script for atlas-container mode "
                             "(default: <job-dir>/my_release_setup.sh).")
    parser.add_argument("--debug-file-directory", default="",
                        help="Directory holding separate .debug files, applied before the core is "
                             "loaded. Accepts {project}, {release}, {platform} and {base} "
                             "placeholders, filled from the payload log's release banner, since a "
                             "fixed path is wrong for every job but one. Dropped when the expanded "
                             "directory does not exist. Must be visible inside the release "
                             "container, so in practice under /cvmfs.")
    parser.add_argument("--python-gdb-helper", default="",
                        help="Path to a CPython gdb helper (python-gdb.py) to source before py-bt. "
                             "Only needed when the release does not ship one next to its Python "
                             "objects; the helper is searched for automatically otherwise.")
    parser.add_argument("--atlas-platform", default=DEFAULT_ATLAS_PLATFORM,
                        help=f"ATLAS container platform (default: {DEFAULT_ATLAS_PLATFORM}).")
    parser.add_argument("--atlas-local-root-base", default=DEFAULT_ATLAS_LOCAL_ROOT_BASE,
                        help="ATLASLocalRootBase path on the host.")
    parser.add_argument("--container-extra-args", default="-c -i",
                        help="Raw Apptainer arguments passed through atlasLocalSetup.sh -e "
                             "(default: '-c -i').")
    parser.add_argument("--container-timeout", type=int, default=DEFAULT_CONTAINER_TIMEOUT,
                        help=f"Whole container evidence-run timeout in seconds "
                             f"(default: {DEFAULT_CONTAINER_TIMEOUT}).")
    parser.add_argument("--keep-container-artifacts", action="store_true",
                        help="Keep generated worker/runner/evidence files in --job-dir for debugging.")
    parser.add_argument("--job-log", action="append", default=None,
                        help="Specific payload/job log to correlate (repeatable; relative paths use --job-dir). "
                             "If omitted, hang mode discovers payload stdout/stderr and log-like files under workDir.")
    parser.add_argument("--no-job-logs", dest="collect_job_logs", action="store_false", default=True,
                        help="Disable bounded host-side payload/job log correlation.")
    parser.add_argument("--max-job-log-files", type=int, default=DEFAULT_MAX_JOB_LOG_FILES,
                        help=f"Maximum discovered payload/job log files to scan (default: {DEFAULT_MAX_JOB_LOG_FILES}).")
    parser.add_argument("--max-job-log-matches", type=int, default=DEFAULT_MAX_JOB_LOG_MATCHES,
                        help=f"Maximum matched payload/job-log lines retained (default: {DEFAULT_MAX_JOB_LOG_MATCHES}).")
    parser.add_argument("--job-log-tail-lines", type=int, default=DEFAULT_JOB_LOG_TAIL_LINES,
                        help=f"Non-empty tail lines retained per payload/runtime log (default: {DEFAULT_JOB_LOG_TAIL_LINES}; 0 disables).")
    parser.add_argument("--exe", default=None,
                        help="Path to the ELF executable. For athena.py jobs this is the Python "
                             "interpreter binary, NOT the .py script. Usually auto-detected.")
    parser.add_argument("--mode", choices=["auto", "hang", "crash"], default="auto",
                        help="Analysis framing. 'auto' infers it from the terminating signal.")
    parser.add_argument("--model", default=None,
                        help="Model to use, overriding the environment and the backend's own default.")
    parser.add_argument("--llm-backend", choices=list(LLM_BACKENDS), default="auto",
                        help="Where synthesis runs. 'bamboo' uses Bamboo's configured provider "
                             "(Anthropic, OpenAI, Gemini, Mistral or an OpenAI-compatible endpoint); "
                             "'anthropic' calls the Anthropic SDK directly; 'auto' prefers bamboo "
                             "when it is importable (default: auto).")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Stack frames per thread (default: {DEFAULT_MAX_FRAMES}).")
    parser.add_argument("--max-thread-groups", type=int, default=DEFAULT_MAX_THREAD_GROUPS,
                        help=f"Distinct thread backtraces to keep (default: {DEFAULT_MAX_THREAD_GROUPS}).")
    parser.add_argument("--max-targeted-threads", type=int, default=DEFAULT_MAX_TARGETED_THREADS,
                        help=f"Non-idle thread groups to inspect with focused frame/args/locals commands; "
                             f"0 disables this extra phase (default: {DEFAULT_MAX_TARGETED_THREADS}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Maximum response tokens (default: {DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--max-evidence-chars", type=int, default=DEFAULT_MAX_EVIDENCE_CHARS,
                        help=f"Evidence size budget (default: {DEFAULT_MAX_EVIDENCE_CHARS}).")
    parser.add_argument("--no-locals", dest="locals", action="store_false", default=True,
                        help="Skip 'info locals'. Locals are collected by default because loop "
                             "counters are often the payoff for a looping job.")
    parser.add_argument("--no-redact", action="store_true",
                        help="Disable scrubbing of tokens, proxies and keys from gdb output.")
    parser.add_argument("--gdb", default=None, help="Path to the gdb executable (default: search PATH).")
    parser.add_argument("--gdb-timeout", type=int, default=DEFAULT_GDB_TIMEOUT,
                        help=f"Per-phase gdb timeout in seconds (default: {DEFAULT_GDB_TIMEOUT}).")
    parser.add_argument("--no-llm", action="store_true", help="Extract evidence only; skip the LLM call.")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Write the full evidence and analysis to this JSON file.")
    parser.add_argument("--raw-gdb", default=None, help="Write the unprocessed gdb output to this file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Also log a heartbeat during long gdb phases and the outgoing evidence "
                             "size/token estimate. Basic phase progress is logged by default; see -q.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress all progress logging to stderr, including -v output.")
    parser.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL,
                        help="Seconds between -v heartbeat messages during a gdb phase "
                             f"(default: {DEFAULT_HEARTBEAT_INTERVAL}).")
    parser.add_argument("--large-core-warning-mib", type=int, default=DEFAULT_LARGE_CORE_WARNING_MIB,
                        help="Core size in MiB above which a one-time slow-analysis note is printed. "
                             f"0 disables it (default: {DEFAULT_LARGE_CORE_WARNING_MIB}).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def resolve_logging_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    """Resolve effective progress/heartbeat verbosity from the CLI flags.

    ``--quiet`` always wins: it suppresses both the default progress lines and
    anything ``-v``/``--verbose`` would otherwise add.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple of ``(progress, detail)``: whether to print basic phase-level
        progress at all, and whether to additionally print heartbeats and
        size/token estimates.
    """
    quiet = getattr(args, "quiet", False)
    return not quiet, bool(getattr(args, "verbose", False)) and not quiet


def resolve_model(explicit: str | None) -> str:
    """Determine which model to use.

    Args:
        explicit: A ``--model`` value, or ``None``.

    Returns:
        The model identifier, preferring ``--model``, then ``CORE_ANALYSIS_MODEL``,
        then ``LLM_DEFAULT_MODEL``, then the built-in default.
    """
    return explicit or os.environ.get("CORE_ANALYSIS_MODEL") or os.environ.get("LLM_DEFAULT_MODEL") or DEFAULT_MODEL


def model_override(explicit: str | None) -> str | None:
    """Determine the model override to hand to the synthesis backend.

    Distinct from :func:`resolve_model`: this returns ``None`` when no model was
    requested anywhere, so that the Bamboo backend can use the model its own
    configuration selected rather than having a hardcoded default forced over
    it.  Only an explicitly expressed preference overrides the backend.

    Args:
        explicit: A ``--model`` value, or ``None``.

    Returns:
        The model identifier from ``--model``, ``CORE_ANALYSIS_MODEL`` or
        ``LLM_DEFAULT_MODEL``, or ``None`` when none of those is set.
    """
    return (
        explicit
        or os.environ.get("CORE_ANALYSIS_MODEL")
        or os.environ.get("LLM_DEFAULT_MODEL")
        or None
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` to use ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on a handled error, ``130`` on interrupt.
    """
    args = parse_args(argv)
    progress, detail = resolve_logging_flags(args)
    started = time.monotonic()
    try:
        evidence, raw = collect_evidence(args, progress=progress, detail=detail)
    except (FileNotFoundError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    if args.raw_gdb:
        Path(args.raw_gdb).write_text(redact(raw, not args.no_redact), encoding="utf-8")
        if detail:
            print(f"[*] Raw gdb output written to {args.raw_gdb}", file=sys.stderr)

    analysis: dict[str, Any] | None = None
    if not args.no_llm:
        # Cost control belongs to the LLM input, not to the deterministic
        # evidence artifact.  Keep the report/JSON complete and reduce only a
        # deep copy that is about to be sent to the model.
        llm_evidence = enforce_global_budget(
            copy.deepcopy(evidence), args.max_evidence_chars, detail=detail
        )
        try:
            analysis = analyze_with_llm(
                llm_evidence, model_override(args.model), args.max_tokens, args.max_evidence_chars,
                progress=progress, detail=detail,
                backend=getattr(args, "llm_backend", "auto"),
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(render_report(evidence, analysis))

    if args.json_out:
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "tool_version": __version__,
            "evidence": evidence.to_dict(),
            "analysis": analysis,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if detail:
            print(f"[*] JSON written to {args.json_out}", file=sys.stderr)

    if progress:
        print(f"[*] Done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
