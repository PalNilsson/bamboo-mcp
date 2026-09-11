"""Pilot log record and Python traceback parsing — ATLAS plugin copy.

Pure parsing helpers with no bamboo, plugin or network dependencies.  Used by
:mod:`askpanda_atlas.log_analysis_impl` to locate the *actual* exception in a
pilot log, and by :mod:`askpanda_atlas.pilot_source_analysis_impl` to turn that
exception into a list of pilot3 source frames to fetch from GitHub.

Why this module exists
----------------------
``log_analysis_impl`` used to anchor its context-window extraction on either a
per-error-code search string (``_PILOT_CODE_PATTERNS``) or, failing that, the
first 40 characters of ``piloterrordiag`` used as a literal regex.  Both are
unreliable: ``piloterrordiag`` is a *summary* written by a different pilot code
path than the log record itself, so the wordings routinely differ.  For pilot
error code 1310 the metadata says ``"Exception caught during payload
execution"`` while the log record says ``"execute payloads caught an exception
(cannot recover): timed out, Traceback (most recent call last):"`` — no match,
so extraction silently fell back to the tail of the log, which for a failed job
is stage-out and log-archiving boilerplate.

The functions here anchor instead on two *format-level invariants* that hold
regardless of error code, pilot version or experiment:

1. Pilot log records begin with ``YYYY-MM-DD HH:MM:SS,mmm | LEVEL |``; a
   record's continuation lines carry no such prefix.
2. A Python traceback always begins with ``Traceback (most recent call
   last):``, continues with indented ``File "...", line N, in func`` frames,
   and terminates with an unindented ``ExceptionType: message`` line.

Neither invariant depends on a lookup table, so new pilot error codes are
handled without code changes.

This file is intentionally duplicated in ``askpanda_epic`` (kept byte-identical
apart from the module docstring).  Plugin packages must stay independently
installable and must not import each other or bamboo core at module scope, and
``log_analysis_impl`` is already duplicated the same way.  Any change here must
be mirrored in ``askpanda_epic/_traceback_parse.py``.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

# ---------------------------------------------------------------------------
# Log record structure
# ---------------------------------------------------------------------------

# Matches the standard pilot log record prefix, e.g.
#   2026-08-17 08:38:24,986 | CRITICAL | pilot.control.payload | execute_payloads | ...
# Both ',' and '.' are accepted as the sub-second separator, and the level is
# captured so records can be ranked by severity.
_RECORD_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*\|\s*(?P<level>[A-Z]+)\s*\|"
)

# Severity ranking used to choose between several candidate tracebacks in one
# log.  Unprefixed text (e.g. payload.stdout, which has no log levels at all)
# ranks as _UNLEVELLED_RANK so that a traceback found there is still usable.
_SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
}
_UNLEVELLED_RANK: int = 35


def record_level(line: str) -> str:
    """Return the log level of a pilot log record's first line.

    Args:
        line: A single physical line from a log file.

    Returns:
        The upper-case level name (e.g. ``"CRITICAL"``) when *line* starts a
        pilot log record, or an empty string when it does not (a continuation
        line, or a log with no record prefixes at all).
    """
    match = _RECORD_PREFIX_RE.match(line)
    return match.group("level") if match else ""


# ---------------------------------------------------------------------------
# Traceback block detection
# ---------------------------------------------------------------------------

_TRACEBACK_MARKER: str = "Traceback (most recent call last):"

# Lines that link chained tracebacks together.  A block continues across these
# so that ``raise ... from exc`` chains are captured whole rather than being
# cut at the first exception line.
_CHAIN_MARKERS: tuple[str, ...] = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)

# Matches a traceback frame header line, capturing the file, line number and
# function.  Applied per-line (the caller supplies traceback text only), so
# leading whitespace is tolerated but not required.
_FRAME_LINE_RE: re.Pattern[str] = re.compile(
    r'^\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<lineno>\d+),\s+in\s+(?P<func>\S+)'
)

# Matches the terminal ``ExceptionType: message`` line of a traceback.  The
# type must be a bare dotted identifier immediately followed by a colon, which
# excludes prose lines such as the chain markers above ("During handling of
# the above exception, ...:") and timestamped log prefixes (they start with a
# digit).
_EXCEPTION_LINE_RE: re.Pattern[str] = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(?P<message>.*)$"
)

# Matches a bare exception type on its own line (no message), e.g. a plain
# ``KeyboardInterrupt`` or ``MemoryError``.
_BARE_EXCEPTION_LINE_RE: re.Pattern[str] = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|Timeout|Failure))\s*$"
)

# Recognises the ``pilot/...`` portion of a traceback file path.  Pilot code is
# unpacked into a job-specific scratch directory (``/tmp/atlas_XXXX/pilot3/``),
# so only the repo-relative tail is stable enough to resolve against GitHub.
# ``pilot3/`` does not match because "pilot" must be followed by "/".
_PILOT_PATH_RE: re.Pattern[str] = re.compile(r"(?P<pilot_path>pilot/[^\"]+\.py)$")


# ---------------------------------------------------------------------------
# State codec helpers
# ---------------------------------------------------------------------------
#
# ``to_state``/``from_state`` on the dataclasses below are a *round-trippable*
# serialisation, deliberately distinct from the existing ``as_dict`` methods.
# ``as_dict`` is an evidence projection for the LLM: it renames ``exc_type`` to
# ``type``, drops ``raw`` entirely and injects a derived ``deepest_pilot_frame``
# that is not a field.  Nothing can be reconstructed from it.  ``to_state`` uses
# the attribute names verbatim and carries every field, so that
# ``from_state(x.to_state()) == x`` and a JSON Schema can be written against it.
# Both exist; neither replaces the other.  Reach for ``as_dict`` when building
# tool evidence and ``to_state`` when state has to survive a round trip through
# a caller.
#
# The coercions below are tolerant rather than strict.  State may arrive from a
# trusted round trip (a recorded fixture) or from a caller that composed it
# itself, and in the latter case a JSON number that arrived as ``"42"`` should
# not take down the analysis.  Values that cannot be coerced fall back to the
# field's zero value, matching the fail-open posture the rest of the tree takes
# toward malformed input.


def coerce_str(value: Any) -> str:
    """Coerce a state value to a string.

    Args:
        value: Value read from a state mapping.

    Returns:
        The value as a string, or ``""`` when it is ``None``.  A non-string is
        rendered with :func:`str` rather than rejected, so a caller that sent a
        number where text was expected still produces usable state.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def coerce_optional_str(value: Any) -> str | None:
    """Coerce a state value to a string, preserving ``None``.

    Distinct from :func:`coerce_str` because several fields use ``None`` and
    ``""`` to mean different things — a log URL of ``None`` means the file was
    never fetched, whereas ``""`` would claim it was fetched from nowhere.

    Args:
        value: Value read from a state mapping.

    Returns:
        ``None`` when *value* is ``None``, otherwise the coerced string.
    """
    if value is None:
        return None
    return coerce_str(value)


def coerce_int(value: Any) -> int:
    """Coerce a state value to an integer.

    Args:
        value: Value read from a state mapping.

    Returns:
        The value as an ``int``, or ``0`` when it cannot be converted.  A
        ``bool`` is rejected rather than silently becoming 0/1: ``True`` as a
        line number is a caller error worth zeroing, not worth honouring.

        A ``float`` is truncated rather than rejected.  JSON draws no
        int/float distinction, so a producer is entitled to encode line 412 as
        ``412.0``; routing that through ``int(str(...))`` would raise on the
        ``"412.0"`` text and silently zero a perfectly good line number.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (OverflowError, ValueError):  # inf, nan
            return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def coerce_bool(value: Any) -> bool:
    """Coerce a state value to a boolean.

    Args:
        value: Value read from a state mapping.

    Returns:
        The value as a ``bool``.  Strings are matched case-insensitively
        against a small true-set, so the ``"true"`` that a caller hand-writing
        JSON is likely to produce does not become ``True`` merely by being a
        non-empty string — and ``"false"`` emphatically does not.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def coerce_mapping(value: Any) -> dict[str, Any]:
    """Coerce a state value to a string-keyed mapping.

    Args:
        value: Value read from a state mapping.

    Returns:
        A dict with string keys, or an empty dict when *value* is not a
        mapping.  Keys are stringified so a caller that used integer keys does
        not produce lookups that silently miss.
    """
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in cast("dict[Any, Any]", value).items()}


@dataclass(frozen=True)
class Frame:
    """One frame of a Python traceback.

    Attributes:
        file: Full file path exactly as it appeared in the traceback.
        lineno: Line number within *file*.
        func: Name of the function or method executing in this frame.
        pilot_path: Repo-relative pilot3 path (e.g. ``"pilot/util/https.py"``)
            when *file* is a pilot3 module, otherwise an empty string.
    """

    file: str
    lineno: int
    func: str
    pilot_path: str = ""

    @property
    def is_pilot(self) -> bool:
        """Return ``True`` when this frame belongs to the pilot3 codebase.

        Returns:
            ``True`` if a repo-relative pilot path was resolved for this
            frame, ``False`` for standard library and CVMFS frames.
        """
        return bool(self.pilot_path)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the frame.

        Returns:
            Dict with ``file``, ``lineno``, ``func``, ``pilot_path`` and
            ``is_pilot`` keys, suitable for inclusion in tool evidence.
        """
        return {
            "file": self.file,
            "lineno": self.lineno,
            "func": self.func,
            "pilot_path": self.pilot_path,
            "is_pilot": self.is_pilot,
        }

    def to_state(self) -> dict[str, Any]:
        """Return a round-trippable representation of the frame.

        Distinct from :meth:`as_dict`, which additionally carries the derived
        ``is_pilot`` flag.  ``is_pilot`` is omitted here precisely because it is
        derived: round-tripping it would let a caller hand back a frame whose
        flag contradicted its ``pilot_path``, and the property would then
        disagree with the field it was restored from.

        Returns:
            Dict keyed by the dataclass field names, accepted by
            :meth:`from_state`.
        """
        return {
            "file": self.file,
            "lineno": self.lineno,
            "func": self.func,
            "pilot_path": self.pilot_path,
        }

    @classmethod
    def from_state(cls, state: Any) -> Frame:
        """Rebuild a frame from :meth:`to_state` output.

        Args:
            state: Mapping as produced by :meth:`to_state`.  Typed ``Any``
                rather than ``Mapping`` because this value routinely arrives
                as an unvalidated MCP tool argument, where the static type is
                whatever JSON decoded to; the guard below is the validation,
                and declaring a narrower type would only move the lie.
                Unknown keys are
                ignored and missing keys fall back to the field defaults, so
                state written by an older or newer version of this module still
                loads.

        Returns:
            The reconstructed :class:`Frame`.  A non-mapping yields a frame of
            defaults rather than raising, since this is reachable from
            caller-supplied input.
        """
        if not isinstance(state, Mapping):
            return cls(file="", lineno=0, func="")
        return cls(
            file=coerce_str(state.get("file")),
            lineno=coerce_int(state.get("lineno")),
            func=coerce_str(state.get("func")),
            pilot_path=coerce_str(state.get("pilot_path")),
        )


@dataclass(frozen=True)
class TracebackBlock:
    """A contiguous Python traceback located within a log file.

    Attributes:
        text: The traceback text verbatim, including the line that carries the
            ``Traceback (most recent call last):`` marker (which in pilot logs
            also carries the timestamp/level prefix) and the terminal
            exception line.
        start_line: Zero-based index of the first line of the block.
        end_line: Zero-based index one past the last line of the block.
        level: Log level of the record containing the marker, or an empty
            string when the log has no record prefixes (e.g. payload.stdout).
    """

    text: str
    start_line: int
    end_line: int
    level: str = ""

    @property
    def severity(self) -> int:
        """Return the numeric severity used to rank candidate tracebacks.

        Returns:
            The rank of :attr:`level`, or :data:`_UNLEVELLED_RANK` for
            unlevelled logs so that they outrank WARNING but not ERROR.
        """
        if not self.level:
            return _UNLEVELLED_RANK
        return _SEVERITY_RANK.get(self.level, _UNLEVELLED_RANK)


def _is_traceback_body_line(line: str) -> bool:
    """Return ``True`` when *line* is part of a traceback's indented body.

    Traceback bodies consist of indented ``File "..."`` frame headers, the
    indented source line for each frame, and (Python 3.11+) indented
    ``~~~^^^`` column-marker lines.  Blank lines inside a traceback are
    treated as body lines because Python emits them between chained
    exceptions.

    Args:
        line: A single physical line, without the trailing newline.

    Returns:
        ``True`` if the line continues the traceback body.
    """
    if not line.strip():
        return True
    return line[:1].isspace()


def _next_nonblank(lines: list[str], start: int) -> int:
    """Return the index of the next non-blank line at or after *start*.

    Args:
        lines: All lines of the log.
        start: Index to begin scanning from.

    Returns:
        Index of the next non-blank line, or ``len(lines)`` if none remain.
    """
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def find_traceback_blocks(log_text: str) -> list[TracebackBlock]:
    """Locate every Python traceback in a log file.

    Works on both structured pilot logs (where the ``Traceback`` marker shares
    a physical line with the ``timestamp | LEVEL |`` prefix and the body
    follows as unprefixed continuation lines) and unstructured payload logs
    (``payload.stdout``, ``payload.stderr``, ``setup.stdout``) which have no
    record prefixes at all.

    A block starts at the line containing the traceback marker and extends over
    all indented body lines.  It normally ends at the first unindented
    ``ExceptionType: message`` line — except when a ``During handling of the
    above exception...`` chain marker follows, in which case the chained
    traceback is absorbed into the same block.  Chained exceptions must stay
    together: splitting them would report the *first* exception as the failure
    when it is the last one that actually propagated.

    If no exception line is found the block ends where the indented body ends,
    so a truncated traceback still yields frames.

    Args:
        log_text: Full text of the log file.

    Returns:
        List of :class:`TracebackBlock` in the order they appear.  Empty if the
        log contains no traceback.
    """
    lines = log_text.splitlines()
    blocks: list[TracebackBlock] = []
    i = 0
    total = len(lines)

    while i < total:
        if _TRACEBACK_MARKER not in lines[i]:
            i += 1
            continue

        start = i
        level = record_level(lines[i])
        j = i + 1
        end = j
        # Set when a chain marker has been seen, which licenses absorbing the
        # following "Traceback (most recent call last):" line into this block
        # instead of treating it as the start of an unrelated traceback.
        expect_chained = False

        while j < total:
            line = lines[j]

            if any(marker in line for marker in _CHAIN_MARKERS):
                expect_chained = True
                j += 1
                end = j
                continue

            if expect_chained and _TRACEBACK_MARKER in line:
                expect_chained = False
                j += 1
                end = j
                continue

            if _is_traceback_body_line(line):
                j += 1
                end = j
                continue

            # Unindented, non-blank line: the terminal exception line.
            if _EXCEPTION_LINE_RE.match(line) or _BARE_EXCEPTION_LINE_RE.match(line):
                j += 1
                end = j
                # A chain marker after the exception line means the traceback
                # continues; keep going rather than closing the block here.
                lookahead = _next_nonblank(lines, j)
                if lookahead < total and any(
                    marker in lines[lookahead] for marker in _CHAIN_MARKERS
                ):
                    j = lookahead
                    continue
            break

        # Trim trailing blank lines that were swallowed as body lines.
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1

        blocks.append(TracebackBlock(
            text="\n".join(lines[start:end]),
            start_line=start,
            end_line=end,
            level=level,
        ))
        i = max(end, start + 1)

    return blocks


def select_primary_traceback(blocks: list[TracebackBlock]) -> TracebackBlock | None:
    """Choose the traceback most likely to explain the job failure.

    Ranks by log severity first (a CRITICAL traceback beats an ERROR one, which
    beats a retried WARNING), then by position, preferring the *last* block of
    equal severity.  Rationale: the pilot logs its fatal exception immediately
    before entering clean-up, whereas earlier same-severity records are usually
    retried operations that eventually succeeded.  The count of all blocks is
    reported separately in evidence so a caller can tell when this choice
    discarded alternatives.

    Args:
        blocks: Blocks returned by :func:`find_traceback_blocks`.

    Returns:
        The selected block, or ``None`` when *blocks* is empty.
    """
    if not blocks:
        return None
    return max(blocks, key=lambda b: (b.severity, b.start_line))


# ---------------------------------------------------------------------------
# Exception parsing
# ---------------------------------------------------------------------------

@dataclass
class ExceptionInfo:
    """Structured form of the exception raised in a traceback.

    Attributes:
        exc_type: Short exception class name (e.g. ``"TimeoutError"``).  For
            dotted names only the final component is kept here.
        exc_type_full: Exception name exactly as printed, which may be dotted
            (e.g. ``"pilot.common.exception.StageInFailure"``).
        message: Exception message text, stripped.
        frames: All traceback frames in call order (outermost first).
        level: Log level of the record the traceback was found in, or an empty
            string for unlevelled logs.
        raw: The traceback text verbatim.
    """

    exc_type: str = ""
    exc_type_full: str = ""
    message: str = ""
    frames: list[Frame] = field(default_factory=list)
    level: str = ""
    raw: str = ""

    @property
    def pilot_frames(self) -> list[Frame]:
        """Return only the frames belonging to the pilot3 codebase.

        Returns:
            Frames whose path resolved to a repo-relative ``pilot/...`` path,
            in call order.
        """
        return [f for f in self.frames if f.is_pilot]

    @property
    def deepest_pilot_frame(self) -> Frame | None:
        """Return the innermost pilot3 frame in the traceback.

        This is the pilot code that was executing when the exception surfaced,
        and therefore the correct starting point for source-level analysis —
        the frames below it are typically standard library or CVMFS code that
        cannot be fixed in pilot3.

        Returns:
            The last pilot frame, or ``None`` when the traceback contains no
            pilot frames (e.g. a pure payload traceback).
        """
        pilot = self.pilot_frames
        return pilot[-1] if pilot else None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for tool evidence.

        Returns:
            Dict with ``type``, ``type_full``, ``message``, ``level``,
            ``frames`` and ``deepest_pilot_frame`` keys.
        """
        deepest = self.deepest_pilot_frame
        return {
            "type": self.exc_type,
            "type_full": self.exc_type_full,
            "message": self.message,
            "level": self.level,
            "frames": [f.as_dict() for f in self.frames],
            "deepest_pilot_frame": deepest.as_dict() if deepest else None,
        }

    def to_state(self) -> dict[str, Any]:
        """Return a round-trippable representation of the exception.

        Three differences from :meth:`as_dict` make this the one to serialise
        with.  The keys are the field names (``exc_type``, not ``type``), the
        derived ``deepest_pilot_frame`` is omitted, and ``raw`` — the verbatim
        traceback text, which :meth:`as_dict` drops — is carried.

        ``raw`` can be large.  It is included because excluding it would make
        the round trip lossy for the one field most likely to be wanted
        downstream; capping belongs at the tool boundary that emits the state,
        where the budget is known, not in the codec.

        Returns:
            Dict keyed by the dataclass field names, accepted by
            :meth:`from_state`.
        """
        return {
            "exc_type": self.exc_type,
            "exc_type_full": self.exc_type_full,
            "message": self.message,
            "frames": [f.to_state() for f in self.frames],
            "level": self.level,
            "raw": self.raw,
        }

    @classmethod
    def from_state(cls, state: Any) -> ExceptionInfo:
        """Rebuild an exception from :meth:`to_state` output.

        Args:
            state: Mapping as produced by :meth:`to_state`; see
                :meth:`Frame.from_state` for why this is typed ``Any``.
                Unknown keys are
                ignored and missing keys fall back to the field defaults.

        Returns:
            The reconstructed :class:`ExceptionInfo`.  A ``frames`` value that
            is not a list yields no frames rather than raising, and individual
            frame entries that are not mappings are skipped, so one malformed
            frame does not discard the rest of the traceback.
        """
        if not isinstance(state, Mapping):
            return cls()

        raw_frames: Any = state.get("frames")
        frames: list[Frame] = []
        if isinstance(raw_frames, (list, tuple)):
            for entry in cast("list[Any]", list(raw_frames)):
                if isinstance(entry, Mapping):
                    frames.append(Frame.from_state(cast("Mapping[str, Any]", entry)))

        return cls(
            exc_type=coerce_str(state.get("exc_type")),
            exc_type_full=coerce_str(state.get("exc_type_full")),
            message=coerce_str(state.get("message")),
            frames=frames,
            level=coerce_str(state.get("level")),
            raw=coerce_str(state.get("raw")),
        )


def parse_frames(traceback_text: str) -> list[Frame]:
    """Extract every frame from traceback text, in call order.

    Both pilot3 and non-pilot (standard library, CVMFS) frames are returned;
    callers distinguish them via :attr:`Frame.is_pilot`.  Keeping the non-pilot
    frames matters for diagnosis: a ``TimeoutError`` raised inside
    ``socket.recv_into`` under ``urllib`` says "the HTTP peer stopped
    responding", which a pilot-only frame list would not convey.

    Args:
        traceback_text: Text of a single traceback block.

    Returns:
        List of :class:`Frame`, outermost call first.  Duplicate frames are
        preserved because recursion depth is diagnostically meaningful.
    """
    frames: list[Frame] = []
    for line in traceback_text.splitlines():
        match = _FRAME_LINE_RE.match(line)
        if not match:
            continue
        file_path = match.group("file")
        pilot_match = _PILOT_PATH_RE.search(file_path)
        try:
            lineno = int(match.group("lineno"))
        except ValueError:  # pragma: no cover - regex guarantees digits
            lineno = 0
        frames.append(Frame(
            file=file_path,
            lineno=lineno,
            func=match.group("func"),
            pilot_path=pilot_match.group("pilot_path") if pilot_match else "",
        ))
    return frames


def _find_terminal_exception(traceback_text: str) -> tuple[str, str]:
    """Locate the final ``ExceptionType: message`` line of a traceback.

    Scans unindented lines only, so indented source lines that happen to
    contain a colon (e.g. ``d = {"a": 1}``) are never mistaken for the
    exception line.  The *last* match wins, which is correct for chained
    tracebacks where the final exception is the one that propagated.

    Args:
        traceback_text: Text of a single traceback block.

    Returns:
        Tuple of ``(exception_type, message)``, both empty when no exception
        line is present (a truncated traceback).
    """
    exc_type = ""
    message = ""
    for line in traceback_text.splitlines():
        if not line or line[:1].isspace():
            continue
        if _TRACEBACK_MARKER in line:
            continue
        bare = _BARE_EXCEPTION_LINE_RE.match(line)
        if bare:
            exc_type, message = bare.group("type"), ""
            continue
        match = _EXCEPTION_LINE_RE.match(line)
        if match:
            exc_type, message = match.group("type"), match.group("message").strip()
    return exc_type, message


def parse_exception(traceback_text: str, level: str = "") -> ExceptionInfo:
    """Parse traceback text into a structured :class:`ExceptionInfo`.

    Args:
        traceback_text: Text of a single traceback block, as produced by
            :func:`find_traceback_blocks`.
        level: Log level of the record the traceback came from, propagated
            into the result for evidence purposes.

    Returns:
        Populated :class:`ExceptionInfo`.  Fields are empty rather than absent
        when the traceback is truncated or malformed, so callers can rely on
        the shape.
    """
    exc_type_full, message = _find_terminal_exception(traceback_text)
    return ExceptionInfo(
        exc_type=exc_type_full.rsplit(".", 1)[-1],
        exc_type_full=exc_type_full,
        message=message,
        frames=parse_frames(traceback_text),
        level=level,
        raw=traceback_text,
    )


def find_primary_exception(log_text: str) -> tuple[ExceptionInfo | None, TracebackBlock | None, int]:
    """Find and parse the most relevant exception in a log file.

    Convenience wrapper combining :func:`find_traceback_blocks`,
    :func:`select_primary_traceback` and :func:`parse_exception`.

    Args:
        log_text: Full text of the log file.

    Returns:
        Tuple of ``(exception_info, block, total_blocks)``.  The first two
        elements are ``None`` when the log contains no traceback;
        ``total_blocks`` is the number of tracebacks found, so callers can
        record in evidence that alternatives were discarded.
    """
    blocks = find_traceback_blocks(log_text)
    block = select_primary_traceback(blocks)
    if block is None:
        return None, None, 0
    return parse_exception(block.text, block.level), block, len(blocks)


# ---------------------------------------------------------------------------
# Pilot version detection
# ---------------------------------------------------------------------------

# Version strings look like 3.14.0.22 (major.minor.patch.build); some older
# pilots report three components only.
_VERSION_CORE: str = r"\d+\.\d+(?:\.\d+){0,2}"

# Forms seen in pilot logs, in the order they are tried:
#   ... | INFO | pilot | main | pilot version: 3.14.0.22
#   *** PanDA Pilot version 3.14.0.22 ***
#   PILOTVERSION=3.14.0.22
_PILOT_VERSION_RES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"pilot\s+version\s*[:=]?\s*v?(?P<ver>{_VERSION_CORE})", re.IGNORECASE),
    re.compile(rf"PanDA\s+Pilot\s+version\s*[:=]?\s*v?(?P<ver>{_VERSION_CORE})", re.IGNORECASE),
    re.compile(rf"PILOTVERSION\s*[:=]\s*v?(?P<ver>{_VERSION_CORE})", re.IGNORECASE),
)

# The pilotid metadata field packs the version into a pipe-delimited string,
# e.g. "https://.../log.tgz|PR|3.14.0.22".  Prefer four-component matches.
_PILOTID_VERSION_RE: re.Pattern[str] = re.compile(r"(?P<ver>\d+\.\d+\.\d+\.\d+)")
_PILOTID_VERSION_SHORT_RE: re.Pattern[str] = re.compile(r"(?P<ver>\d+\.\d+\.\d+)")


def parse_pilot_version(log_text: str) -> str:
    """Extract the pilot release version from pilot log text.

    Args:
        log_text: Full text of ``pilotlog.txt`` (the version is logged during
            start-up, so an excerpt taken from the failure point will usually
            *not* contain it).

    Returns:
        Version string such as ``"3.14.0.22"``, or an empty string when no
        version line is present.
    """
    for pattern in _PILOT_VERSION_RES:
        match = pattern.search(log_text)
        if match:
            return match.group("ver")
    return ""


def parse_pilot_version_from_pilotid(pilotid: str) -> str:
    """Extract the pilot release version from the ``pilotid`` metadata field.

    Fallback for jobs where ``pilotlog.txt`` was not downloaded (e.g. pilot
    error code 1305, where ``payload.stdout`` is the primary log).

    Args:
        pilotid: The ``pilotid`` string from BigPanDA job metadata.

    Returns:
        Version string such as ``"3.14.0.22"``, or an empty string when the
        field is absent or contains no version-like token.
    """
    if not pilotid:
        return ""
    match = _PILOTID_VERSION_RE.search(pilotid)
    if match:
        return match.group("ver")
    match = _PILOTID_VERSION_SHORT_RE.search(pilotid)
    return match.group("ver") if match else ""


# ---------------------------------------------------------------------------
# Budget-aware traceback truncation
# ---------------------------------------------------------------------------

_ELISION_MARKER: str = "\n  ... [traceback truncated by Bamboo — middle frames omitted] ...\n"

# Fraction of the budget spent on the head of an over-long traceback.  The tail
# gets the remainder because it holds the deepest frames and the exception line,
# which are the parts that identify the failure.
_HEAD_FRACTION: float = 0.35


def truncate_traceback(traceback_text: str, max_chars: int) -> str:
    """Shorten a traceback to fit a character budget without losing the ends.

    A plain ``text[:max_chars]`` would discard the terminal
    ``ExceptionType: message`` line — the single most diagnostic line in the
    whole traceback.  This function keeps the head (which shows where the
    pilot entered the failing call chain) and the tail (the deepest frames and
    the exception line), eliding the middle.

    Args:
        traceback_text: Traceback text to shorten.
        max_chars: Maximum number of characters to return.  Values that cannot
            accommodate the elision marker fall back to keeping the tail,
            since the exception line matters more than the entry point.

    Returns:
        The traceback unchanged when it already fits, otherwise a head + marker
        + tail composition no longer than *max_chars*.
    """
    if max_chars <= 0:
        return ""
    if len(traceback_text) <= max_chars:
        return traceback_text

    marker_len = len(_ELISION_MARKER)
    if max_chars <= marker_len * 2:
        return traceback_text[-max_chars:]

    budget = max_chars - marker_len
    head_chars = int(budget * _HEAD_FRACTION)
    tail_chars = budget - head_chars
    return traceback_text[:head_chars] + _ELISION_MARKER + traceback_text[-tail_chars:]


def iter_record_starts(log_text: str) -> Iterator[tuple[int, str]]:
    """Yield the index and level of every line that starts a pilot log record.

    Exposed for callers that need to align a character offset or line index
    back onto record boundaries (for example to include whole records of
    preceding context rather than cutting one in half).

    Args:
        log_text: Full text of the log file.

    Yields:
        Tuples of ``(line_index, level)`` for each record-starting line.
    """
    for index, line in enumerate(log_text.splitlines()):
        level = record_level(line)
        if level:
            yield index, level


__all__ = [
    "ExceptionInfo",
    "Frame",
    "TracebackBlock",
    "coerce_bool",
    "coerce_int",
    "coerce_mapping",
    "coerce_optional_str",
    "coerce_str",
    "find_primary_exception",
    "find_traceback_blocks",
    "iter_record_starts",
    "parse_exception",
    "parse_frames",
    "parse_pilot_version",
    "parse_pilot_version_from_pilotid",
    "record_level",
    "select_primary_traceback",
    "truncate_traceback",
]
