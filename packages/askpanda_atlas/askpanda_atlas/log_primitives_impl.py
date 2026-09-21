"""ATLAS log-fetch primitives — canonical implementation.

Bamboo's ``panda_log_analysis`` is a compound tool: one call runs metadata
fetch, file listing, log download, excerpt, classify and evidence bundling.
Agentic frameworks built around *code mode* — where the model writes code
composing small tool primitives rather than selecting one compound tool per
turn — need a finer-grained surface.  This module provides it, over the same
implementation functions the monolith uses, without changing the monolith.

The primitives are advertised only under the ``primitive`` tool profile (see
:mod:`bamboo.tools._tool_profiles`), so they never enter Bamboo's own planner
catalog and cannot dilute its tool-selection accuracy.

The primitive set
-----------------
Five tools decompose ``fetch_and_analyse`` up to evidence bundling::

    meta = atlas.log.fetch_metadata(job_id)
    plan = atlas.log.plan_fetch(job_id)
    while True:
        for entry in plan["next"]:
            got = atlas.log.fetch_text(job_id, entry["filename"], entry["role"])
            fetched.append(got)
            observed["fetched"].append(entry["filename"])
            observed.update(got["signals"])
        if plan["done"]:
            break
        plan = atlas.log.plan_fetch(job_id, observed)
    verdict = atlas.log.classify(meta, fetched)

:func:`list_files` is not on that path; it answers "what is in this job's
tarball" for callers that want the listing itself rather than the plan derived
from it.

Every decision on that path is taken server-side.  ``plan_fetch`` chooses the
files, ``fetch_text`` chooses the character budget from the ``role``
``plan_fetch`` assigned and computes the ``setup_has_error`` signal, and
``classify`` joins the excerpts and picks which exception to trust.  The agent
carries opaque dicts between calls; it never evaluates a domain predicate.

``fetch_text`` returns an excerpt computed over the **full** downloaded text
rather than the raw text capped.  Traceback anchoring searches the whole file
in the monolith, so excerpting at the tool boundary is what keeps the two
paths comparable — and it means no uncapped log is ever serialised across the
wire.  The budget comes from :data:`ENV_MAX_CHARS`, read independently of the
monolith's ``_MAX_EXCERPT_CHARS`` so that raising it for a large-context
code-mode agent cannot change the orchestrated path.

Equivalence with the monolith
-----------------------------
The composed loop and one ``fetch_and_analyse`` call must reach the same
excerpt, the same exception and the same verdict for the same job.  That is
asserted scenario by scenario in
``packages/askpanda_atlas/tests/test_log_equivalence.py``, and it is the
reason several rules here are transcriptions rather than reinventions: the
character budget per role, the ``payload.stderr`` separator, the
stderr-first exception precedence, and the file the pilot version may be
parsed from.

Two differences remain and are intended.  ``context.exception.raw`` is capped
at the tool boundary (D-27) where the monolith carries it whole, and the
primitives stop at :func:`classify` — the evidence bundling, link block and
core-dump probe that ``fetch_and_analyse`` performs afterwards have no
primitive counterpart by design.

Why planning is a tool rather than a static rule
------------------------------------------------
``_fetch_logs_payload`` is not a static plan.  For pilot error code 1305 it
fetches ``setup.stdout`` first and *only* if
:func:`~askpanda_atlas.log_analysis_impl._setup_log_has_error` fires does it
return early and skip ``payload.stdout``/``payload.stderr``.  That is a
decision taken mid-flight, on file **content**.

A one-shot ``plan_fetch(job_id) -> [ordered files]`` cannot express that
without exporting the predicate to the agent, which would duplicate a domain
rule into a tool description where it would drift.  So :func:`plan_fetch` is
**re-entrant** instead: the agent fetches what ``next`` names, and hands the
opaque ``signals`` mapping it got back in as ``observed`` on the following
call.  The agent never evaluates a domain predicate; it carries a dict between
two calls.

Loop shape
----------
``done`` means *no further* :func:`plan_fetch` call is required.  It may be
``True`` alongside a non-empty ``next``, which is the terminal plan: fetch
those files and stop.  A caller that re-plans anyway gets
``{"next": [], "done": True}``, so the naive ``while not done`` loop also
terminates — at most three calls either way.

Return shape
------------
Success::

    {"job_id": int,
     "strategy": "payload_1305" | "pilotlog" | "metadata_only",
     "next": [{"filename": str, "role": str, "url": str, "reason": str}],
     "done": bool,
     "notes": [str]}

Failure::

    {"job_id": int, "error": str}

``role`` maps one-to-one onto the fields of
:class:`~askpanda_atlas.log_analysis_impl._LogFetchResult`: ``setup`` onto
``setup_log_url``, ``primary`` onto ``log_url``, ``secondary`` onto
``stderr_url``.  That is deliberate, so an equivalence check between a
primitive loop and one ``fetch_and_analyse`` call is a field comparison rather
than an interpretation.

The declared ``outputSchema`` carries **no top-level ``required``** and always
declares ``error``.  The MCP SDK rejects a tool that advertises an
``outputSchema`` and then returns unstructured content, so a failure has to be
expressible as structured content too; a schema that required the success keys
would make every error path a protocol error.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any, cast

from askpanda_atlas._fallback_http import get_base_url
from askpanda_atlas._traceback_parse import (
    coerce_bool,
    coerce_int,
    coerce_str,
    parse_pilot_version,
    parse_pilot_version_from_pilotid,
    truncate_traceback,
)
from askpanda_atlas.log_analysis_impl import (
    _STDERR_RESERVED_CHARS,
    _TRACEBACK_RESERVED_CHARS,
    FailureContext,
    _fetch_file_listing,
    _fetch_log_text,
    _fetch_metadata,
    _file_is_nonempty,
    _log_file_url,
    _select_log_filename,
    _setup_log_has_error,
    _top_level_file_index,
    classify_failure,
    extract_failure_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Pilot error code meaning "the user payload failed".  The payload logs are
#: the primary evidence for it and ``pilotlog.txt`` for everything else.
_PAYLOAD_FAILURE_CODE: int = 1305

STRATEGY_PAYLOAD_1305: str = "payload_1305"
STRATEGY_PILOTLOG: str = "pilotlog"
STRATEGY_METADATA_ONLY: str = "metadata_only"

ROLE_SETUP: str = "setup"
ROLE_PRIMARY: str = "primary"
ROLE_SECONDARY: str = "secondary"

SETUP_LOG: str = "setup.stdout"
PAYLOAD_STDOUT: str = "payload.stdout"
PAYLOAD_STDERR: str = "payload.stderr"

#: The only file a pilot version may be parsed from; see :func:`fetch_text`.
#: Equal to what
#: :func:`~askpanda_atlas.log_analysis_impl._select_log_filename` returns for
#: every non-1305 job, which is the only path on which the monolith parses a
#: version at all.
PILOT_LOG: str = "pilotlog.txt"

#: Key of the signal :func:`plan_fetch` consumes from a previous fetch.
SETUP_SIGNAL: str = "setup_has_error"

#: Job statuses for which ``fetch_and_analyse`` downloads logs at all.  For any
#: other status it returns metadata-only evidence, and so does this planner.
_LOG_BEARING_STATUSES: frozenset[str] = frozenset({"failed", "holding", "cancelled"})

_REASON_SETUP_FIRST: str = (
    "Pilot error 1305 is a payload failure, but a failed release or container "
    "setup produces the same code with empty payload logs, so setup.stdout is "
    "read first.  Pass the returned signals back as 'observed'."
)
_REASON_PAYLOAD_STDOUT: str = (
    "Pilot error 1305 indicates the user payload failed, so payload.stdout is "
    "the primary log."
)
_REASON_PAYLOAD_STDERR: str = (
    "Python tracebacks and segfault reports are written to stderr, so "
    "payload.stderr often holds the exception that actually terminated the "
    "payload even when payload.stdout has one too."
)
_REASON_PILOTLOG: str = (
    "Failures other than pilot error 1305 are diagnosed from the pilot's own "
    "log rather than from the payload's."
)

_NOTE_SETUP_ERROR: str = (
    "A setup error was reported in setup.stdout, so payload.stdout and "
    "payload.stderr are skipped: the payload never ran and those files are "
    "empty."
)
_NOTE_SETUP_CLEAN: str = (
    "setup.stdout reported no setup error, so the payload logs are the "
    "primary evidence."
)

#: Environment variable capping the characters any primitive returns.
#: Deliberately separate from ``log_analysis_impl._MAX_EXCERPT_CHARS``: a
#: code-mode agent may have a much larger context than Bamboo's own synthesis
#: step, and raising the budget for one must not move the other.
ENV_MAX_CHARS: str = "BAMBOO_PRIMITIVE_MAX_CHARS"

#: Default for :data:`ENV_MAX_CHARS`.  Numerically equal to
#: ``_MAX_EXCERPT_CHARS`` so that, unconfigured, the primitive path and the
#: monolith excerpt identically.
DEFAULT_MAX_CHARS: int = 8000

#: Most listing entries :func:`list_files` will return.  A job log tarball is
#: routinely thousands of files; returning all of them would swamp the agent's
#: context for no diagnostic gain.
MAX_LISTING_ENTRIES: int = 500

#: Separator ``_fetch_logs_payload`` puts between the two payload excerpts.
#: Transcribed rather than imported because it is written inline there; the
#: agreement is pinned by a test.
STDERR_SEPARATOR: str = "\n\n--- payload.stderr ---\n"

#: Job metadata fields the primitives project out of the BigPanDA ``job`` dict.
#: This is the set ``fetch_and_analyse`` promotes into evidence, plus two that
#: it consumes without promoting: ``commandtopilot``, which
#: :func:`~askpanda_atlas.log_analysis_impl.classify_failure` searches for the
#: JEDI-reassignment signal, and ``pilotid``, which carries the pilot version
#: when no pilot log was downloaded.  Omitting either would make a
#: classification taken over this subset disagree with one taken over the full
#: job dict.
_METADATA_FIELDS: tuple[str, ...] = (
    "jobstatus",
    "jobsubstatus",
    "computingsite",
    "cloud",
    "atlasrelease",
    "jeditaskid",
    "attemptnr",
    "maxattempt",
    "transformation",
    "exeerrorcode",
    "exeerrordiag",
    "taskbuffererrorcode",
    "taskbuffererrordiag",
    "ddmerrorcode",
    "ddmerrordiag",
    "starttime",
    "endtime",
    "duration",
    "commandtopilot",
)

#: Keys of one normalised listing entry, as
#: :func:`~askpanda_atlas.log_analysis_impl._normalise_listing_entry` builds it.
_LISTING_FIELDS: tuple[str, ...] = (
    "relative_path",
    "name",
    "dirname",
    "size_bytes",
    "modification",
)

_ERROR_METADATA: str = "Failed to fetch job metadata from BigPanDA"


# ---------------------------------------------------------------------------
# Shared tool-boundary helpers
# ---------------------------------------------------------------------------

def _tool_result(payload: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Return the ``(content, structured)`` tuple every primitive must yield.

    The MCP SDK answers a result carrying no structured content from a tool
    that advertises an ``outputSchema`` with *"Output validation error:
    outputSchema defined but no structured output returned"* — an opaque
    protocol error in place of whatever precise message the tool meant to
    send.  Routing every return path of every primitive through this one
    function is what makes "did we forget the structured half?" a question
    with a single answer rather than one per ``return`` statement.

    ``bamboo.tools.base`` is imported here rather than at module scope so the
    rest of this module stays importable when bamboo core is absent.

    Args:
        payload: The structured payload, conforming to the calling tool's
            declared ``outputSchema``.

    Returns:
        Two-element tuple of the JSON-serialised payload as MCP content and
        the payload itself.
    """
    from bamboo.tools.base import text_content  # deferred — see docstring

    return text_content(json.dumps(payload)), payload


def _coerce_job_id(arguments: dict[str, Any]) -> tuple[int | None, str]:
    """Read and validate the ``job_id`` argument.

    Args:
        arguments: The tool's argument dict.

    Returns:
        Tuple of the parsed job ID and an error message.  Exactly one is
        meaningful: on success the message is empty, on failure the ID is
        ``None``.
    """
    raw: Any = arguments.get("job_id")
    if raw is None:
        return None, "missing job_id"
    try:
        return int(raw), ""
    except (ValueError, TypeError):
        return None, "job_id must be an integer"


def _coerce_timeout(arguments: dict[str, Any]) -> int:
    """Read the optional ``timeout`` argument.

    Args:
        arguments: The tool's argument dict.

    Returns:
        The requested timeout in seconds, or 60 when absent, zero or
        unparseable.  A bad timeout degrades rather than failing the call:
        it is an optimisation, not part of the question being asked.
    """
    try:
        return int(arguments.get("timeout") or 60)
    except (ValueError, TypeError):
        return 60


def _max_chars() -> int:
    """Return the character budget primitives cap their output at.

    Read at call time rather than at import so the variable is testable
    without reimporting the module, matching how
    :func:`bamboo.tools._tool_profiles.active_profile` reads its own.

    Returns:
        The value of :data:`ENV_MAX_CHARS`, or :data:`DEFAULT_MAX_CHARS` when
        it is unset, unparseable or not positive.  A misconfigured budget
        falls back rather than raising, for the same reason an unrecognised
        tool profile does: a configuration typo must not make every call fail.
    """
    raw: str = os.getenv(ENV_MAX_CHARS, "").strip()
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "%s=%r is not an integer; using %d", ENV_MAX_CHARS, raw, DEFAULT_MAX_CHARS
        )
        return DEFAULT_MAX_CHARS
    if value <= 0:
        logger.warning(
            "%s=%d is not positive; using %d", ENV_MAX_CHARS, value, DEFAULT_MAX_CHARS
        )
        return DEFAULT_MAX_CHARS
    return value


def _role_budget(role: str, pilot_error_code: int, budget: int) -> int:
    """Return the character budget for one file, given the role it plays.

    Transcribes the allocation in
    :func:`~askpanda_atlas.log_analysis_impl._fetch_logs_payload`: on the
    payload path ``payload.stdout`` is excerpted against
    ``_MAX_EXCERPT_CHARS - _STDERR_RESERVED_CHARS`` so that appending
    ``payload.stderr`` afterwards cannot push the combined text over budget,
    and ``payload.stderr`` itself gets the reservation.  The reduction is
    unconditional there — taken before it is known whether ``payload.stderr``
    has any content — so it is unconditional here too.

    ``setup.stdout`` and the pilot log are excerpted against the whole budget,
    since neither is ever joined to a second file.

    Args:
        role: One of the ``ROLE_*`` constants.
        pilot_error_code: The job's pilot error code, which decides whether
            the primary file is a payload log sharing budget with stderr.
        budget: Total budget from :func:`_max_chars`.

    Returns:
        The budget for this file, always at least one character.  A budget at
        or below the stderr reservation is not reduced further: halving an
        already-tiny budget would leave nothing of the traceback, and an
        operator who set the budget that low did not mean to disable
        excerpting.
    """
    if role == ROLE_SECONDARY:
        return min(_STDERR_RESERVED_CHARS, budget)
    if role == ROLE_PRIMARY and pilot_error_code == _PAYLOAD_FAILURE_CODE:
        if budget > _STDERR_RESERVED_CHARS:
            return budget - _STDERR_RESERVED_CHARS
    return budget


def _capped_context_state(context: FailureContext, budget: int) -> dict[str, Any]:
    """Serialise a failure context, capping the verbatim traceback.

    ``ExceptionInfo.to_state`` carries ``raw`` — the traceback exactly as
    printed — uncapped, deliberately: excluding it would make the round trip
    lossy for the field most likely to be wanted downstream, and the codec
    does not know the budget.  The tool boundary does, so the cap is applied
    here.  :func:`~askpanda_atlas._traceback_parse.truncate_traceback` is used
    rather than a slice because it elides the middle frames and keeps the
    terminal exception line, which is the part a slice would discard.

    Args:
        context: The context to serialise.
        budget: Character budget for this file.

    Returns:
        State dict accepted by :meth:`FailureContext.from_state`.
    """
    state: dict[str, Any] = context.to_state()
    exception: Any = state.get("exception")
    if isinstance(exception, dict):
        raw: str = cast("dict[str, Any]", exception).get("raw") or ""
        if raw:
            cast("dict[str, Any]", exception)["raw"] = truncate_traceback(
                raw, min(_TRACEBACK_RESERVED_CHARS, budget)
            )
    return state


# ---------------------------------------------------------------------------
# Reading the caller's observations
# ---------------------------------------------------------------------------

def _observed_fetched(observed: Any) -> frozenset[str]:
    """Return the filenames the caller reports having already fetched.

    Read tolerantly, in the style of the ``from_state`` codecs: ``observed``
    arrives as an unvalidated MCP tool argument, so a non-mapping, a
    wrong-typed ``fetched`` or an unknown key must degrade to "nothing
    observed" rather than raise.

    The accepted container types are listed rather than tested for
    iterability, which is what rejects a bare ``"fetched": "setup.stdout"``.
    Iterating a string would decompose it into characters; worse, a
    substring test against one would let ``"payload.stdout"`` answer for
    ``"payload.stdout.gz"``.

    Args:
        observed: The ``observed`` argument as supplied by the caller.

    Returns:
        Frozen set of filenames; empty when nothing usable was supplied.
    """
    if not isinstance(observed, Mapping):
        return frozenset()

    raw: Any = cast("Mapping[str, Any]", observed).get("fetched")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()

    return frozenset(
        item for item in cast("list[Any]", list(raw)) if isinstance(item, str)
    )


def _observed_setup_seen(observed: Any, fetched: frozenset[str]) -> bool:
    """Report whether ``setup.stdout`` has already been fetched.

    Two routes count, because a caller may report either the signal or the
    filename: the presence of the :data:`SETUP_SIGNAL` key, or ``setup.stdout``
    appearing in ``fetched``.

    Args:
        observed: The ``observed`` argument as supplied by the caller.
        fetched: Result of :func:`_observed_fetched` for the same argument.

    Returns:
        ``True`` when the file has been fetched, ``False`` otherwise.
    """
    if SETUP_LOG in fetched:
        return True
    return isinstance(observed, Mapping) and SETUP_SIGNAL in observed


def _observed_setup_has_error(observed: Any) -> bool:
    """Report whether the fetched ``setup.stdout`` carried a setup error.

    Uses :func:`~askpanda_atlas._traceback_parse.coerce_bool` rather than
    ``bool()``: the signal may arrive JSON-encoded as the string ``"false"``,
    which is truthy under ``bool()`` and would invert the decision, skipping
    the payload logs for a job whose setup was fine.

    A missing signal reads as ``False``.  That is the correct default and not
    merely a safe one: ``_fetch_logs_payload`` also falls through to the
    payload logs when ``setup.stdout`` was fetched but came back empty, and an
    empty file yields no signal.

    Args:
        observed: The ``observed`` argument as supplied by the caller.

    Returns:
        ``True`` only when the caller positively reported a setup error.
    """
    if not isinstance(observed, Mapping):
        return False
    return coerce_bool(cast("Mapping[str, Any]", observed).get(SETUP_SIGNAL))


def _pilot_error_code(job: Mapping[str, Any]) -> int:
    """Return the job's pilot error code as an integer.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.

    Returns:
        The numeric code, or ``0`` when the field is absent or unparseable.
        Mirrors the coercion ``fetch_and_analyse`` performs, so the two agree
        on which strategy a job takes.
    """
    try:
        return int(job.get("piloterrorcode") or 0)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _next_entry(
    job_id: int,
    filename: str,
    role: str,
    reason: str,
    base_url: str,
) -> dict[str, str]:
    """Build one entry of a plan's ``next`` list.

    Args:
        job_id: PanDA job ID.
        filename: Log filename relative to the job directory.
        role: One of :data:`ROLE_SETUP`, :data:`ROLE_PRIMARY`,
            :data:`ROLE_SECONDARY`.
        reason: Why this file is worth fetching, for the agent's benefit.
        base_url: BigPanDA base URL.

    Returns:
        Dict with ``filename``, ``role``, ``url`` and ``reason`` keys.
    """
    return {
        "filename": filename,
        "role": role,
        "url": _log_file_url(job_id, filename, base_url),
        "reason": reason,
    }


def _plan(
    job_id: int,
    strategy: str,
    nxt: list[dict[str, str]],
    done: bool,
    notes: list[str],
) -> dict[str, Any]:
    """Assemble a successful plan payload.

    Args:
        job_id: PanDA job ID.
        strategy: One of the ``STRATEGY_*`` constants.
        nxt: Files to fetch now, possibly empty.
        done: Whether a further :func:`plan_fetch` call is required.
        notes: Human-readable remarks, typically recording a skipped file.

    Returns:
        The plan dict, conforming to the tool's declared ``outputSchema``.
    """
    return {
        "job_id": job_id,
        "strategy": strategy,
        "next": nxt,
        "done": done,
        "notes": notes,
    }


def _plan_payload_1305(
    job_id: int,
    base_url: str,
    index: dict[str, int] | None,
    observed: Any,
) -> dict[str, Any]:
    """Plan the next fetch for a pilot-1305 (payload failure) job.

    Transcribes the ordering decisions in
    :func:`~askpanda_atlas.log_analysis_impl._fetch_logs_payload`:
    ``setup.stdout`` first, an early return when it reports a setup error, and
    otherwise ``payload.stdout`` then ``payload.stderr``, skipping any file the
    size index confirms to be zero-length.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        index: Top-level file-size index, or ``None`` when the listing was
            unavailable (fail-open: every file is then offered).
        observed: The caller's observations from previous fetches.

    Returns:
        The plan dict.
    """
    notes: list[str] = []
    nxt: list[dict[str, str]] = []
    fetched: frozenset[str] = _observed_fetched(observed)

    if not _observed_setup_seen(observed, fetched):
        if _file_is_nonempty(index, SETUP_LOG):
            nxt.append(
                _next_entry(
                    job_id, SETUP_LOG, ROLE_SETUP, _REASON_SETUP_FIRST, base_url
                )
            )
            # Not done: whether the payload logs are worth fetching depends on
            # what this file contains, which only the caller can report back.
            return _plan(job_id, STRATEGY_PAYLOAD_1305, nxt, False, notes)
        notes.append(f"{SETUP_LOG} is zero-length; skipping.")
    elif _observed_setup_has_error(observed):
        notes.append(_NOTE_SETUP_ERROR)
        return _plan(job_id, STRATEGY_PAYLOAD_1305, nxt, True, notes)
    else:
        notes.append(_NOTE_SETUP_CLEAN)

    for filename, role, reason in (
        (PAYLOAD_STDOUT, ROLE_PRIMARY, _REASON_PAYLOAD_STDOUT),
        (PAYLOAD_STDERR, ROLE_SECONDARY, _REASON_PAYLOAD_STDERR),
    ):
        if filename in fetched:
            continue
        if not _file_is_nonempty(index, filename):
            notes.append(f"{filename} is zero-length; skipping.")
            continue
        nxt.append(_next_entry(job_id, filename, role, reason, base_url))

    return _plan(job_id, STRATEGY_PAYLOAD_1305, nxt, True, notes)


def _plan_pilotlog(
    job: dict[str, Any],
    job_id: int,
    base_url: str,
    index: dict[str, int] | None,
    observed: Any,
) -> dict[str, Any]:
    """Plan the next fetch for any pilot error code other than 1305.

    Transcribes
    :func:`~askpanda_atlas.log_analysis_impl._fetch_logs_pilotlog`: one file,
    chosen by
    :func:`~askpanda_atlas.log_analysis_impl._select_log_filename`, skipped
    when the size index confirms it is zero-length.

    Args:
        job: The ``job`` dict from the BigPanDA metadata response.
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        index: Top-level file-size index, or ``None`` when unavailable.
        observed: The caller's observations from previous fetches.

    Returns:
        The plan dict.  Always ``done``: there is no content-dependent second
        step on this path.
    """
    notes: list[str] = []
    nxt: list[dict[str, str]] = []
    fetched: frozenset[str] = _observed_fetched(observed)
    filename: str = _select_log_filename(job)

    if filename in fetched:
        return _plan(job_id, STRATEGY_PILOTLOG, nxt, True, notes)

    if not _file_is_nonempty(index, filename):
        notes.append(f"{filename} is zero-length; skipping.")
        return _plan(job_id, STRATEGY_PILOTLOG, nxt, True, notes)

    nxt.append(_next_entry(job_id, filename, ROLE_PRIMARY, _REASON_PILOTLOG, base_url))
    return _plan(job_id, STRATEGY_PILOTLOG, nxt, True, notes)


def plan_fetch(
    job_id: int,
    base_url: str,
    timeout: int,
    observed: Any = None,
) -> dict[str, Any]:
    """Decide which of a job's log files to fetch next.

    Intentionally synchronous so it can be offloaded to a thread pool via
    ``asyncio.to_thread``, matching ``fetch_and_analyse``.  Both the metadata
    fetch and the file listing go through the in-process caches, so the
    re-entrant calls this contract requires cost one HTTP round trip in total
    rather than one each.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds for each request.
        observed: What the caller learned from the fetches the previous plan
            asked for — the ``signals`` mapping returned by the fetch
            primitive, optionally with a ``fetched`` list of filenames.  Read
            tolerantly; ``None`` means "nothing fetched yet".

    Returns:
        A plan dict, or ``{"job_id": ..., "error": ...}`` when the job's
        metadata could not be read.
    """
    payload: dict[str, Any] | None = _fetch_metadata(job_id, base_url, timeout)
    if payload is None:
        return {
            "job_id": job_id,
            "error": "Failed to fetch job metadata from BigPanDA",
        }

    job: dict[str, Any] = payload.get("job") or {}
    if not job:
        return {
            "job_id": job_id,
            "error": f"Job {job_id} was not found in BigPanDA.",
        }

    jobstatus: str = str(job.get("jobstatus") or "")
    if jobstatus not in _LOG_BEARING_STATUSES:
        return _plan(
            job_id,
            STRATEGY_METADATA_ONLY,
            [],
            True,
            [
                f"Job status is {jobstatus!r}; logs are only downloaded for "
                f"{', '.join(sorted(_LOG_BEARING_STATUSES))}.",
            ],
        )

    # One listing feeds every zero-length decision below.  ``None`` means the
    # listing could not be fetched and is propagated as fail-open, so an
    # unavailable listing offers files rather than suppressing them.
    listing: list[dict[str, Any]] | None = _fetch_file_listing(
        job_id, base_url, timeout
    )
    file_index: dict[str, int] | None = (
        None if listing is None
        else {
            record["relative_path"]: int(record["size_bytes"])
            for record in listing
        }
    )
    # Basename-safe view: a nested namesake such as ``workDir/setup.stdout``
    # must not answer for the root-level file.
    index: dict[str, int] | None = _top_level_file_index(file_index)

    if _pilot_error_code(job) == _PAYLOAD_FAILURE_CODE:
        return _plan_payload_1305(job_id, base_url, index, observed)
    return _plan_pilotlog(job, job_id, base_url, index, observed)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

#: Declared output shape.  Deliberately carries no top-level ``required``; see
#: the module docstring for why a failure must be expressible under it.
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {
            "type": "integer",
            "description": "The job the plan applies to.",
        },
        "strategy": {
            "type": "string",
            "enum": [
                STRATEGY_PAYLOAD_1305,
                STRATEGY_PILOTLOG,
                STRATEGY_METADATA_ONLY,
            ],
            "description": "Which log-selection rule this job falls under.",
        },
        "next": {
            "type": "array",
            "description": "Files to fetch now.  May be empty.",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path relative to the job directory.",
                    },
                    "role": {
                        "type": "string",
                        "enum": [ROLE_SETUP, ROLE_PRIMARY, ROLE_SECONDARY],
                        "description": "What this file contributes to the diagnosis.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Filebrowser URL of the file.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this file is worth fetching.",
                    },
                },
                "required": ["filename", "role", "url", "reason"],
                "additionalProperties": False,
            },
        },
        "done": {
            "type": "boolean",
            "description": (
                "True when no further plan_fetch call is required.  May be "
                "true alongside a non-empty 'next': fetch those and stop."
            ),
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Remarks, typically recording a skipped file.",
        },
        "error": {
            "type": "string",
            "description": "Present instead of a plan when planning failed.",
        },
    },
    "additionalProperties": False,
}


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``atlas.log.plan_fetch``.

    Returns:
        Definition dict carrying ``outputSchema`` and restricting itself to
        the ``primitive`` profile, so it is advertised only to a code-mode
        agent and never enters Bamboo's own planner catalog.
    """
    return {
        "name": "atlas.log.plan_fetch",
        "description": (
            "Decide which of a failed PanDA job's log files to download next, "
            "and in what order. Call this before fetching anything: which "
            "files are worth reading depends on the job's pilot error code, "
            "on which files are non-empty, and — for payload failures — on "
            "what an earlier file turned out to contain. Fetch the files "
            "listed in 'next', then call this again passing the signals you "
            "got back as 'observed', until 'done' is true. At most three "
            "calls. This is a planning primitive for code-mode composition; "
            "for a single-call diagnosis of a failed job use "
            "panda_log_analysis instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid) to plan for.",
                },
                "observed": {
                    "type": "object",
                    "description": (
                        "What previous fetches reported. Pass the 'signals' "
                        "mapping returned by the fetch primitive, optionally "
                        "with a 'fetched' list of filenames already read. "
                        "Omit on the first call."
                    ),
                    "properties": {
                        "fetched": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filenames already fetched.",
                        },
                        SETUP_SIGNAL: {
                            "type": "boolean",
                            "description": (
                                "Whether the fetched setup.stdout reported a "
                                "setup error.  Computed server-side by the "
                                "fetch primitive; do not derive it yourself."
                            ),
                        },
                    },
                    "additionalProperties": True,
                },
                "timeout": {
                    "type": "integer",
                    "description": "HTTP timeout in seconds.  Default: 60.",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "outputSchema": _OUTPUT_SCHEMA,
        "profiles": ["primitive"],
        "tags": ["atlas", "panda", "log", "primitive", "code-mode"],
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class AtlasLogPlanFetchTool:
    """MCP tool wrapping :func:`plan_fetch`.

    Every ``call()`` return path yields the two-element
    ``(unstructured, structured)`` tuple the MCP SDK calls
    ``CombinationContent``.  That is not stylistic: the SDK rejects a result
    carrying no structured content from a tool that advertises an
    ``outputSchema``, turning what should be a clean argument error into an
    opaque *"Output validation error"*.  A plain ``text_content(...)`` return
    on any error path would reintroduce exactly that.
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

    async def call(self, arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Plan the next log fetch for a job.

        The blocking HTTP calls are offloaded via ``asyncio.to_thread``.

        Args:
            arguments: Dict with required ``job_id`` (int) and optional
                ``observed`` (mapping) and ``timeout`` (int).

        Returns:
            A ``(content, structured)`` tuple.  ``content`` is a one-element
            MCP content list holding the JSON-serialised payload; ``structured``
            is the same payload as a dict, validated by the SDK against the
            declared ``outputSchema``.
        """
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be a dict"})

        job_id, error = _coerce_job_id(arguments)
        if job_id is None:
            return _tool_result({"error": error})

        timeout: int = _coerce_timeout(arguments)
        base_url: str = get_base_url()

        try:
            plan: dict[str, Any] = await asyncio.to_thread(
                plan_fetch, job_id, base_url, timeout, arguments.get("observed")
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error planning log fetch for job %d", job_id)
            return _tool_result({"job_id": job_id, "error": repr(exc)})

        return _tool_result(plan)


plan_fetch_tool = AtlasLogPlanFetchTool()


# ---------------------------------------------------------------------------
# atlas.log.fetch_metadata
# ---------------------------------------------------------------------------

def fetch_metadata(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
    """Fetch the job-metadata subset the other primitives consume.

    Facts only.  Which log file to read and whether this job has logs at all
    are decisions, and decisions belong to :func:`plan_fetch`; a second place
    to learn "which file" is a second place for that rule to drift.

    A metadata fetch that fails and a job that does not exist both report via
    ``error``, with distinguishable messages.  Neither is expressible as a
    partial success — every field would be ``null`` — and the alternative of
    a ``found`` flag would make a caller that forgot to check it proceed with
    a job dict full of nulls.

    Intentionally synchronous, for ``asyncio.to_thread``.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        The metadata subset, or ``{"job_id": ..., "error": ...}``.
    """
    payload: dict[str, Any] | None = _fetch_metadata(job_id, base_url, timeout)
    if payload is None:
        return {"job_id": job_id, "error": _ERROR_METADATA}

    job: dict[str, Any] = payload.get("job") or {}
    if not job:
        return {"job_id": job_id, "error": f"Job {job_id} was not found in BigPanDA."}

    pilotid: str = str(job.get("pilotid") or "")
    result: dict[str, Any] = {
        "job_id": job_id,
        "monitor_url": f"{base_url}/job?pandaid={job_id}",
        # Coerced exactly as ``fetch_and_analyse`` coerces them, so a strategy
        # or a classification taken over this subset matches one taken over
        # the full job dict.
        "piloterrorcode": _pilot_error_code(job),
        "piloterrordiag": str(job.get("piloterrordiag") or ""),
        "pilotid": pilotid,
        "pilot_version_from_pilotid": parse_pilot_version_from_pilotid(pilotid),
    }
    # Pass-through, uncoerced: these are BigPanDA's values and inventing a
    # type for them here would be a second opinion that could disagree with
    # the monolith's evidence.  Absent fields are present as ``null`` rather
    # than omitted, so a consumer can rely on the shape instead of probing
    # with ``in`` — the same contract ``_build_exception_evidence`` keeps.
    for field_name in _METADATA_FIELDS:
        result[field_name] = job.get(field_name)
    return result


_METADATA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": "integer", "description": "The job described."},
        "monitor_url": {
            "type": "string",
            "description": "BigPanDA monitor page for the job.",
        },
        "piloterrorcode": {
            "type": "integer",
            "description": "Pilot error code; 0 when absent or unparseable.",
        },
        "piloterrordiag": {
            "type": "string",
            "description": "Pilot error diagnosis text; empty when absent.",
        },
        "pilotid": {
            "type": "string",
            "description": "Raw pilotid field; empty when absent.",
        },
        "pilot_version_from_pilotid": {
            "type": "string",
            "description": (
                "Pilot version parsed from pilotid.  Use the version "
                "fetch_text reports from the pilot log in preference to this "
                "one; this is the fallback when no pilot log was read."
            ),
        },
        # Deliberately untyped: BigPanDA decides these types, and declaring
        # one here would reject a job whose field came back as a string where
        # another job's came back as an integer.
        **{
            field_name: {
                "description": f"BigPanDA job field {field_name!r}, verbatim.",
            }
            for field_name in _METADATA_FIELDS
        },
        "error": {
            "type": "string",
            "description": "Present instead of metadata when the fetch failed.",
        },
    },
    "additionalProperties": False,
}


def get_metadata_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``atlas.log.fetch_metadata``.

    Returns:
        Definition dict carrying ``outputSchema`` and restricting itself to
        the ``primitive`` profile.
    """
    return {
        "name": "atlas.log.fetch_metadata",
        "description": (
            "Fetch a PanDA job's metadata: status, site, error codes and "
            "diagnoses, timing, and the pilot version implied by its pilot "
            "ID. Returns facts, not decisions — call atlas.log.plan_fetch to "
            "learn which log files to read. Pass the result to "
            "atlas.log.classify as 'job'. This is a primitive for code-mode "
            "composition; for a single-call diagnosis of a failed job use "
            "panda_log_analysis instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "HTTP timeout in seconds.  Default: 60.",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "outputSchema": _METADATA_OUTPUT_SCHEMA,
        "profiles": ["primitive"],
        "tags": ["atlas", "panda", "log", "primitive", "code-mode"],
    }


class AtlasLogFetchMetadataTool:
    """MCP tool wrapping :func:`fetch_metadata`.

    Like every primitive here, each ``call()`` return path goes through
    :func:`_tool_result`; see that function for why.
    """

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_metadata_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Fetch one job's metadata subset.

        Args:
            arguments: Dict with required ``job_id`` (int) and optional
                ``timeout`` (int).

        Returns:
            A ``(content, structured)`` tuple.
        """
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be a dict"})

        job_id, error = _coerce_job_id(arguments)
        if job_id is None:
            return _tool_result({"error": error})

        try:
            payload: dict[str, Any] = await asyncio.to_thread(
                fetch_metadata, job_id, get_base_url(), _coerce_timeout(arguments)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error fetching metadata for job %d", job_id)
            return _tool_result({"job_id": job_id, "error": repr(exc)})

        return _tool_result(payload)


fetch_metadata_tool = AtlasLogFetchMetadataTool()


# ---------------------------------------------------------------------------
# atlas.log.list_files
# ---------------------------------------------------------------------------

def _ordered_listing(listing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order a listing so job-root entries come first.

    The diagnostic files are all at the job-directory root, while the bulk of
    a tarball is nested under ``workDir``.  A job with several thousand nested
    files would otherwise push ``setup.stdout`` past
    :data:`MAX_LISTING_ENTRIES` and out of the result.  The partition is
    applied unconditionally rather than only when truncating, so the order
    does not change shape at the cap.

    Args:
        listing: Normalised entries from ``_fetch_file_listing``.

    Returns:
        The same entries, root-level ones first, each group in listing order.
    """
    root: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for record in listing:
        path = str(record.get("relative_path") or "")
        (nested if "/" in path else root).append(record)
    return root + nested


def _listing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one listing entry onto the shape the ``outputSchema`` declares.

    Coerced rather than passed through.  ``_normalise_listing_entry`` always
    supplies all five keys, but the schema declares ``dirname`` and
    ``modification`` as strings with ``additionalProperties: false``, so a
    sparse entry reaching here would emit ``null`` against a string type and
    the SDK would reject the whole result — replacing a usable listing with
    *"Output validation error"*.  A tool owns the shape it advertises; the
    boundary is where that is guaranteed, not where it is assumed.

    Args:
        record: One normalised entry from ``_fetch_file_listing``.

    Returns:
        Dict with the five declared keys, each of the declared type.
    """
    return {
        "relative_path": coerce_str(record.get("relative_path")),
        "name": coerce_str(record.get("name")),
        "dirname": coerce_str(record.get("dirname")),
        "size_bytes": coerce_int(record.get("size_bytes")),
        "modification": coerce_str(record.get("modification")),
    }


def list_files(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
    """List the files in a job's log tarball, with sizes.

    An unavailable listing is reported as ``listing_available: false`` with an
    empty ``files``, not as an ``error``.  ``_fetch_file_listing`` returning
    ``None`` means "unknown", and every consumer in the monolith treats that
    as fail-open — it attempts the download anyway.  A primitive that called
    the same condition fatal would disagree with :func:`plan_fetch` about the
    same job in the same session.

    Intentionally synchronous, for ``asyncio.to_thread``.

    Args:
        job_id: PanDA job ID.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with ``files``, ``total``, ``truncated``, ``listing_available``
        and ``notes``.
    """
    listing: list[dict[str, Any]] | None = _fetch_file_listing(
        job_id, base_url, timeout
    )
    if listing is None:
        return {
            "job_id": job_id,
            "listing_available": False,
            "files": [],
            "total": 0,
            "truncated": False,
            "notes": [
                "The file listing could not be fetched, so it is unknown which "
                "files exist.  Treat this as unknown rather than as an empty "
                "job: fetching a log may still succeed.",
            ],
        }

    notes: list[str] = []
    ordered: list[dict[str, Any]] = _ordered_listing(listing)
    total: int = len(ordered)
    if total > MAX_LISTING_ENTRIES:
        notes.append(
            f"{total} files listed; showing the first {MAX_LISTING_ENTRIES} "
            f"with job-root files first."
        )
        ordered = ordered[:MAX_LISTING_ENTRIES]

    files: list[dict[str, Any]] = [_listing_record(record) for record in ordered]
    return {
        "job_id": job_id,
        "listing_available": True,
        "files": files,
        "total": total,
        "truncated": total > MAX_LISTING_ENTRIES,
        "notes": notes,
    }


_LIST_FILES_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": "integer", "description": "The job listed."},
        "listing_available": {
            "type": "boolean",
            "description": (
                "False when the listing could not be fetched.  That means "
                "'unknown', not 'no files': a log fetch may still succeed."
            ),
        },
        "files": {
            "type": "array",
            "description": "Listing entries, job-root files first.",
            "items": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "Path relative to the job directory.",
                    },
                    "name": {"type": "string", "description": "Basename."},
                    "dirname": {
                        "type": "string",
                        "description": "Directory, empty for job-root files.",
                    },
                    "size_bytes": {
                        "type": "integer",
                        "description": "Size in bytes; 0 means the file is empty.",
                    },
                    "modification": {
                        "type": "string",
                        "description": "Modification timestamp as reported.",
                    },
                },
                "additionalProperties": False,
            },
        },
        "total": {
            "type": "integer",
            "description": "Entries in the full listing, before truncation.",
        },
        "truncated": {
            "type": "boolean",
            "description": "True when 'files' holds fewer entries than 'total'.",
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Remarks, typically recording truncation.",
        },
        "error": {
            "type": "string",
            "description": "Present instead of a listing when the call failed.",
        },
    },
    "additionalProperties": False,
}


def get_list_files_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``atlas.log.list_files``.

    Returns:
        Definition dict carrying ``outputSchema`` and restricting itself to
        the ``primitive`` profile.
    """
    return {
        "name": "atlas.log.list_files",
        "description": (
            "List the files in a PanDA job's log tarball with their sizes, "
            "job-root files first. Use this to see what a job produced — a "
            "zero size means the file is empty and not worth fetching. You "
            "do not need this to diagnose a failure: atlas.log.plan_fetch "
            "already consults the listing and names the files worth reading. "
            "'listing_available': false means the listing could not be "
            "fetched, which is not the same as the job having no files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "HTTP timeout in seconds.  Default: 60.",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "outputSchema": _LIST_FILES_OUTPUT_SCHEMA,
        "profiles": ["primitive"],
        "tags": ["atlas", "panda", "log", "primitive", "code-mode"],
    }


class AtlasLogListFilesTool:
    """MCP tool wrapping :func:`list_files`.

    Like every primitive here, each ``call()`` return path goes through
    :func:`_tool_result`; see that function for why.
    """

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_list_files_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """List one job's files.

        Args:
            arguments: Dict with required ``job_id`` (int) and optional
                ``timeout`` (int).

        Returns:
            A ``(content, structured)`` tuple.
        """
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be a dict"})

        job_id, error = _coerce_job_id(arguments)
        if job_id is None:
            return _tool_result({"error": error})

        try:
            payload: dict[str, Any] = await asyncio.to_thread(
                list_files, job_id, get_base_url(), _coerce_timeout(arguments)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error listing files for job %d", job_id)
            return _tool_result({"job_id": job_id, "error": repr(exc)})

        return _tool_result(payload)


list_files_tool = AtlasLogListFilesTool()


# ---------------------------------------------------------------------------
# atlas.log.fetch_text
# ---------------------------------------------------------------------------

#: Roles :func:`fetch_text` recognises, as :func:`plan_fetch` assigns them.
_ROLES: frozenset[str] = frozenset({ROLE_SETUP, ROLE_PRIMARY, ROLE_SECONDARY})

#: Shared shape of a ``FailureContext`` state dict, as both :func:`fetch_text`
#: and :func:`classify` emit it.  ``exception`` is declared loosely on purpose:
#: pinning the frame shape here would duplicate ``Frame.to_state`` into a
#: schema, where it could disagree with the codec after a change to either.
_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Excerpt and parsed exception, as produced by the same extractor "
        "panda_log_analysis uses.  Pass these through to atlas.log.classify "
        "unchanged."
    ),
    "properties": {
        "excerpt": {
            "type": "string",
            "description": "The diagnostic section of the log, within budget.",
        },
        "exception": {
            "type": ["object", "null"],
            "description": (
                "Parsed Python exception, or null when the log had no "
                "traceback.  Carries 'exc_type', 'message', 'frames', "
                "'level' and a truncated 'raw'."
            ),
            "additionalProperties": True,
        },
        "traceback_count": {
            "type": "integer",
            "description": "Distinct tracebacks found; above 1 means others were discarded.",
        },
    },
    "additionalProperties": False,
}


def _file_context(
    text: str,
    filename: str,
    pilot_error_code: int,
    pilot_error_diag: str,
    budget: int,
    setup_has_error: bool,
) -> FailureContext:
    """Excerpt one downloaded file, applying the monolith's setup-log rule.

    Extraction runs over the **full** text, not over a pre-capped slice: the
    traceback scan in
    :func:`~askpanda_atlas.log_analysis_impl.extract_failure_context` searches
    the whole file, and anchoring it on a slice would find a different
    traceback — or none — than ``fetch_and_analyse`` finds for the same job.

    The one deviation from plain extraction is the rule
    ``_fetch_logs_payload`` applies to an erroring ``setup.stdout``: setup
    failures are shell output rather than tracebacks, so when no traceback is
    present the whole capped file is kept instead of an anchored window, which
    would crop the asetup/release diagnostics.  That rule keys on the
    *filename*, as it does in the monolith, not on the caller-supplied role.

    Args:
        text: Full downloaded file content.
        filename: Name of the file, relative to the job directory.
        pilot_error_code: The job's pilot error code.
        pilot_error_diag: The job's pilot error diagnosis text.
        budget: Character budget for this file, from :func:`_role_budget`.
        setup_has_error: Whether this file is an erroring ``setup.stdout``.

    Returns:
        The populated :class:`FailureContext`.
    """
    context: FailureContext = extract_failure_context(
        text, filename, pilot_error_code, pilot_error_diag, budget
    )
    if filename == SETUP_LOG and setup_has_error and context.exception is None:
        return FailureContext(
            excerpt=text[:budget],
            exception=None,
            traceback_count=context.traceback_count,
        )
    return context


def fetch_text(
    job_id: int,
    filename: str,
    role: str,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    """Download one of a job's log files and return its diagnostic excerpt.

    Returns an excerpt rather than the raw file.  A pilot log is routinely
    tens of megabytes, and the useful part of it is chosen by a rule
    (traceback-first, then a pilot-code anchor, then the tail) that the agent
    must not have to reimplement.

    The character budget is derived server-side from *role* — the value
    :func:`plan_fetch` assigned to this file — so the agent never picks one.
    Metadata is refetched here to learn the pilot error code the excerpt rule
    and the budget depend on; it comes from the same 60-second TTL cache
    :func:`plan_fetch` populated, so in a composed loop it costs no request.

    ``signals`` carries :data:`SETUP_SIGNAL` only when the file *is*
    ``setup.stdout``.  Emitting it for any other file would be actively
    harmful: :func:`_observed_setup_seen` counts the key's presence as "setup
    has been read", so a ``setup_has_error: false`` picked up from
    ``payload.stdout`` and merged into ``observed`` would make the next
    :func:`plan_fetch` skip the setup log entirely.

    ``pilot_version`` is keyed on the filename for the same reason.  The
    monolith parses a version only in ``_fetch_logs_pilotlog``, which runs
    only for non-1305 jobs, where the file is always :data:`PILOT_LOG`; on the
    payload path it falls back to the ``pilotid`` metadata field without
    looking at ``payload.stdout`` at all.  ``parse_pilot_version`` matches its
    pattern anywhere in the text, so a payload that echoed a version line
    would otherwise make a composed loop report a version the monolith does
    not — and the caller has no way to know which of its results to trust.

    Intentionally synchronous, for ``asyncio.to_thread``.

    Args:
        job_id: PanDA job ID.
        filename: Log filename relative to the job directory.
        role: One of the ``ROLE_*`` constants.  An unrecognised role degrades
            to :data:`ROLE_PRIMARY` with a note rather than failing the call,
            matching the fail-open rule the tool-profile switch uses: a typo
            should cost accuracy, not the answer.
        base_url: BigPanDA base URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with ``context``, ``signals``, ``bytes``, ``truncated``,
        ``available`` and ``pilot_version``, or
        ``{"job_id": ..., "filename": ..., "error": ...}``.
    """
    notes: list[str] = []
    if role not in _ROLES:
        notes.append(f"Unrecognised role {role!r}; treated as {ROLE_PRIMARY!r}.")
        role = ROLE_PRIMARY

    payload: dict[str, Any] | None = _fetch_metadata(job_id, base_url, timeout)
    if payload is None:
        return {"job_id": job_id, "filename": filename, "error": _ERROR_METADATA}

    job: dict[str, Any] = payload.get("job") or {}
    if not job:
        return {
            "job_id": job_id,
            "filename": filename,
            "error": f"Job {job_id} was not found in BigPanDA.",
        }

    pilot_error_code: int = _pilot_error_code(job)
    budget: int = _role_budget(role, pilot_error_code, _max_chars())
    url: str = _log_file_url(job_id, filename, base_url)

    text: str | None = _fetch_log_text(job_id, filename, base_url, timeout)
    setup_has_error: bool = _setup_log_has_error(text or "")
    signals: dict[str, Any] = (
        {SETUP_SIGNAL: setup_has_error} if filename == SETUP_LOG else {}
    )

    result: dict[str, Any] = {
        "job_id": job_id,
        "filename": filename,
        "role": role,
        "url": url,
        "signals": signals,
    }

    if not text:
        notes.append(
            f"{filename} is empty." if text == ""
            else f"{filename} could not be downloaded; it may not exist."
        )
        result.update({
            "available": False,
            "bytes": 0,
            "truncated": False,
            "context": FailureContext().to_state(),
            "pilot_version": "",
            "notes": notes,
        })
        return result

    context: FailureContext = _file_context(
        text,
        filename,
        pilot_error_code,
        str(job.get("piloterrordiag") or ""),
        budget,
        setup_has_error,
    )
    result.update({
        "available": True,
        # The file's true size, comparable with list_files' ``size_bytes``;
        # ``len(text)`` would count characters and disagree for any log
        # carrying non-ASCII output.
        "bytes": len(text.encode("utf-8")),
        "truncated": len(context.excerpt) < len(text),
        "context": _capped_context_state(context, budget),
        # Parsed from the full text: the pilot reports its version at
        # start-up, so an excerpt taken at the failure point will not have it.
        # Keyed on the filename for the same reason ``signals`` is: see the
        # docstring.
        "pilot_version": parse_pilot_version(text) if filename == PILOT_LOG else "",
        "notes": notes,
    })
    return result


_FETCH_TEXT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": "integer", "description": "The job the file belongs to."},
        "filename": {"type": "string", "description": "The file that was read."},
        "role": {
            "type": "string",
            "description": "The role the file was read under, after validation.",
        },
        "url": {"type": "string", "description": "Filebrowser URL of the file."},
        "available": {
            "type": "boolean",
            "description": "False when the file was empty or could not be downloaded.",
        },
        "bytes": {
            "type": "integer",
            "description": "Size of the whole file in bytes, not of the excerpt.",
        },
        "truncated": {
            "type": "boolean",
            "description": "True when the excerpt is shorter than the whole file.",
        },
        "context": _CONTEXT_SCHEMA,
        "signals": {
            "type": "object",
            "description": (
                "Domain predicates computed server-side.  Merge into the "
                "'observed' argument of atlas.log.plan_fetch; do not derive "
                "them yourself."
            ),
            "properties": {
                SETUP_SIGNAL: {
                    "type": "boolean",
                    "description": (
                        "Whether setup.stdout reported a fatal setup error.  "
                        "Present only when the file read was setup.stdout."
                    ),
                },
            },
            "additionalProperties": True,
        },
        "pilot_version": {
            "type": "string",
            "description": (
                "Pilot version, parsed from pilotlog.txt only; empty for any "
                "other file and when absent.  For a job whose pilot log was "
                "not read, use fetch_metadata's pilot_version_from_pilotid."
            ),
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Remarks, typically recording an unreadable file.",
        },
        "error": {
            "type": "string",
            "description": "Present instead of content when the call failed.",
        },
    },
    "additionalProperties": False,
}


def get_fetch_text_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``atlas.log.fetch_text``.

    Returns:
        Definition dict carrying ``outputSchema`` and restricting itself to
        the ``primitive`` profile.
    """
    return {
        "name": "atlas.log.fetch_text",
        "description": (
            "Download one of a PanDA job's log files and return its "
            "diagnostic excerpt, not the raw file — a pilot log is routinely "
            "tens of megabytes and the useful part is selected server-side. "
            "Pass the 'filename' and 'role' exactly as atlas.log.plan_fetch "
            "gave them: the role sets the character budget. Merge the "
            "returned 'signals' into 'observed' on your next plan_fetch call, "
            "and collect the whole result for atlas.log.classify."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "PanDA job ID (pandaid).",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Log filename relative to the job directory, as named "
                        "in a plan_fetch 'next' entry (e.g. setup.stdout)."
                    ),
                },
                "role": {
                    "type": "string",
                    "enum": [ROLE_SETUP, ROLE_PRIMARY, ROLE_SECONDARY],
                    "description": (
                        "The role plan_fetch assigned this file.  Default: "
                        "primary."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "HTTP timeout in seconds.  Default: 60.",
                },
            },
            "required": ["job_id", "filename"],
            "additionalProperties": False,
        },
        "outputSchema": _FETCH_TEXT_OUTPUT_SCHEMA,
        "profiles": ["primitive"],
        "tags": ["atlas", "panda", "log", "primitive", "code-mode"],
    }


class AtlasLogFetchTextTool:
    """MCP tool wrapping :func:`fetch_text`.

    Like every primitive here, each ``call()`` return path goes through
    :func:`_tool_result`; see that function for why.
    """

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_fetch_text_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Download and excerpt one log file.

        Args:
            arguments: Dict with required ``job_id`` (int) and ``filename``
                (str), and optional ``role`` (str) and ``timeout`` (int).

        Returns:
            A ``(content, structured)`` tuple.
        """
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be a dict"})

        job_id, error = _coerce_job_id(arguments)
        if job_id is None:
            return _tool_result({"error": error})

        raw_filename: Any = arguments.get("filename")
        if not isinstance(raw_filename, str) or not raw_filename.strip():
            return _tool_result({
                "job_id": job_id,
                "error": "filename must be a non-empty string",
            })
        filename: str = raw_filename.strip()

        raw_role: Any = arguments.get("role")
        role: str = raw_role if isinstance(raw_role, str) else ROLE_PRIMARY

        try:
            payload: dict[str, Any] = await asyncio.to_thread(
                fetch_text,
                job_id,
                filename,
                role,
                get_base_url(),
                _coerce_timeout(arguments),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Unexpected error fetching %s for job %d", filename, job_id
            )
            return _tool_result({
                "job_id": job_id, "filename": filename, "error": repr(exc),
            })

        return _tool_result(payload)


fetch_text_tool = AtlasLogFetchTextTool()


# ---------------------------------------------------------------------------
# atlas.log.classify
# ---------------------------------------------------------------------------

def _has_content(context: FailureContext) -> bool:
    """Report whether a context carries anything worth classifying.

    Args:
        context: A rebuilt failure context.

    Returns:
        ``True`` when it has an excerpt or a parsed exception.  A context with
        neither came from a file that was empty or absent, which the monolith
        treats as "no log" rather than as an empty log.
    """
    return bool(context.excerpt) or context.exception is not None


def _fetched_entries(fetched: Any) -> list[tuple[str, FailureContext]]:
    """Read the caller's ``fetch_text`` results into role/context pairs.

    Read tolerantly, in the ``from_state`` style: ``fetched`` arrives as an
    unvalidated tool argument, so a non-list, a non-mapping member or a
    missing ``context`` degrades rather than raising.  One malformed entry
    costs its own content, not the whole classification.

    A missing or unrecognised ``role`` reads as :data:`ROLE_PRIMARY` rather
    than being dropped, matching :func:`fetch_text`: content the caller
    actually fetched should be classified even when it is mislabelled.

    Args:
        fetched: The ``fetched`` argument as supplied by the caller — a list
            of ``fetch_text`` results, or of ``{"role", "context"}`` pairs.

    Returns:
        List of ``(role, context)`` pairs in the order supplied.
    """
    if not isinstance(fetched, (list, tuple)):
        return []

    entries: list[tuple[str, FailureContext]] = []
    for item in cast("list[Any]", list(fetched)):
        if not isinstance(item, Mapping):
            continue
        entry: Mapping[str, Any] = cast("Mapping[str, Any]", item)
        raw_role: Any = entry.get("role")
        role: str = raw_role if raw_role in _ROLES else ROLE_PRIMARY
        entries.append((role, FailureContext.from_state(entry.get("context"))))
    return entries


def _combine_contexts(
    entries: list[tuple[str, FailureContext]],
) -> tuple[FailureContext, list[str]]:
    """Join the fetched contexts the way ``_fetch_logs_payload`` joins them.

    Two rules are transcribed.  The excerpts are concatenated with the
    separator the monolith writes between ``payload.stdout`` and
    ``payload.stderr``; and when both files carry a traceback the *stderr* one
    is preferred, because Python tracebacks and segfault reports are written
    to stderr, so that is the exception which actually terminated the payload.

    Both are domain rules.  Leaving them to the agent would put a separator
    string and a precedence rule into a tool description, where they would
    drift from the implementation that has to agree with them.

    A setup context is used only when no payload content was supplied.  That
    covers two cases, and the monolith covers both the same way: the erroring
    ``setup.stdout``, where :func:`plan_fetch` ends the loop so no payload
    entry exists, and the clean ``setup.stdout`` whose payload logs then
    turned out to be empty, which ``_fetch_logs_payload`` falls back on in its
    own ``elif setup_text`` branch.

    Args:
        entries: Role/context pairs from :func:`_fetched_entries`.

    Returns:
        Tuple of the combined context and any remarks.
    """
    notes: list[str] = []
    grouped: dict[str, list[FailureContext]] = {role: [] for role in sorted(_ROLES)}
    for role, context in entries:
        grouped[role].append(context)
    for role in sorted(grouped):
        if len(grouped[role]) > 1:
            notes.append(
                f"{len(grouped[role])} {role} files supplied; using the first."
            )

    primary: FailureContext | None = next(iter(grouped[ROLE_PRIMARY]), None)
    secondary: FailureContext | None = next(iter(grouped[ROLE_SECONDARY]), None)
    setup: FailureContext | None = next(iter(grouped[ROLE_SETUP]), None)

    stderr_usable: bool = secondary is not None and _has_content(secondary)
    if stderr_usable or (primary is not None and _has_content(primary)):
        base: FailureContext = primary or FailureContext()
        if secondary is not None and stderr_usable:
            chosen: FailureContext = secondary if secondary.exception else base
            return FailureContext(
                excerpt=base.excerpt + STDERR_SEPARATOR + secondary.excerpt,
                exception=chosen.exception,
                traceback_count=chosen.traceback_count,
            ), notes
        return base, notes

    if setup is not None and _has_content(setup):
        notes.append(
            "No payload log content was supplied; classified from setup.stdout."
        )
        return setup, notes

    return FailureContext(), notes


def classify(job: Any, fetched: Any) -> dict[str, Any]:
    """Classify a job failure from its metadata and the logs already fetched.

    Pure: no network, no job ID, nothing to cache.  Everything it needs has
    already been fetched by :func:`fetch_metadata` and :func:`fetch_text`, so
    making it a function of its arguments keeps it cheap to call, trivial to
    test, and safe to call twice.

    The excerpt it classifies is the one it builds from *fetched*, and it is
    returned alongside the verdict — the classification and the text it was
    taken from are the pair an agent needs to explain the answer.

    Args:
        job: The metadata subset from :func:`fetch_metadata`.  Read
            tolerantly; a non-mapping yields a metadata-free classification
            rather than an error.
        fetched: List of :func:`fetch_text` results, in the order they were
            fetched.  May be empty, which is the correct input for a job whose
            status carries no logs: the monolith classifies such a job from
            metadata alone, and so does this.

    Returns:
        Dict with ``failure_type``, the combined ``context`` and ``notes``.
    """
    job_dict: dict[str, Any] = (
        dict(cast("Mapping[str, Any]", job)) if isinstance(job, Mapping) else {}
    )
    entries: list[tuple[str, FailureContext]] = _fetched_entries(fetched)
    context, notes = _combine_contexts(entries)

    if not entries:
        notes.append(
            "No log content was supplied; classified from job metadata alone."
        )

    return {
        "failure_type": classify_failure(job_dict, context.excerpt, context.exception),
        "context": _capped_context_state(context, _max_chars()),
        "notes": notes,
    }


_CLASSIFY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "failure_type": {
            "type": "string",
            "description": (
                "Short failure category, e.g. 'stagein_timeout', "
                "'payload_error', 'reassigned_by_jedi'.  'unknown' when "
                "nothing matched."
            ),
        },
        "context": _CONTEXT_SCHEMA,
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Remarks about how the contexts were combined.",
        },
        "error": {
            "type": "string",
            "description": "Present instead of a verdict when the call failed.",
        },
    },
    "additionalProperties": False,
}


def get_classify_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``atlas.log.classify``.

    Returns:
        Definition dict carrying ``outputSchema`` and restricting itself to
        the ``primitive`` profile.
    """
    return {
        "name": "atlas.log.classify",
        "description": (
            "Classify a PanDA job failure from the metadata and the logs you "
            "have already fetched. Pass the atlas.log.fetch_metadata result "
            "as 'job' and the atlas.log.fetch_text results as 'fetched', in "
            "the order you fetched them — this joins their excerpts and picks "
            "which traceback to trust, so do not merge them yourself. Call it "
            "with an empty 'fetched' for a job that has no logs; it will "
            "classify from metadata alone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": {
                    "type": "object",
                    "description": (
                        "The atlas.log.fetch_metadata result, unmodified."
                    ),
                    "additionalProperties": True,
                },
                "fetched": {
                    "type": "array",
                    "description": (
                        "The atlas.log.fetch_text results, in fetch order.  "
                        "Each needs at least its 'role' and 'context'."
                    ),
                    "items": {"type": "object", "additionalProperties": True},
                },
            },
            "required": ["job"],
            "additionalProperties": False,
        },
        "outputSchema": _CLASSIFY_OUTPUT_SCHEMA,
        "profiles": ["primitive"],
        "tags": ["atlas", "panda", "log", "primitive", "code-mode"],
    }


class AtlasLogClassifyTool:
    """MCP tool wrapping :func:`classify`.

    The only primitive that does no I/O, so its ``call()`` runs inline rather
    than through ``asyncio.to_thread``.  Like every primitive here, each
    return path goes through :func:`_tool_result`; see that function for why.
    """

    def __init__(self) -> None:
        """Initialise with the tool definition."""
        self._def: dict[str, Any] = get_classify_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Classify a failure from supplied metadata and log contexts.

        Args:
            arguments: Dict with required ``job`` (mapping) and optional
                ``fetched`` (list).

        Returns:
            A ``(content, structured)`` tuple.
        """
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be a dict"})

        if "job" not in arguments:
            return _tool_result({"error": "missing job"})

        try:
            payload: dict[str, Any] = classify(
                arguments.get("job"), arguments.get("fetched")
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error classifying a failure")
            return _tool_result({"error": repr(exc)})

        return _tool_result(payload)


classify_tool = AtlasLogClassifyTool()


__all__ = [
    "DEFAULT_MAX_CHARS",
    "ENV_MAX_CHARS",
    "MAX_LISTING_ENTRIES",
    "PAYLOAD_STDERR",
    "PAYLOAD_STDOUT",
    "PILOT_LOG",
    "ROLE_PRIMARY",
    "ROLE_SECONDARY",
    "ROLE_SETUP",
    "SETUP_LOG",
    "SETUP_SIGNAL",
    "STDERR_SEPARATOR",
    "STRATEGY_METADATA_ONLY",
    "STRATEGY_PAYLOAD_1305",
    "STRATEGY_PILOTLOG",
    "AtlasLogClassifyTool",
    "AtlasLogFetchMetadataTool",
    "AtlasLogFetchTextTool",
    "AtlasLogListFilesTool",
    "AtlasLogPlanFetchTool",
    "classify",
    "classify_tool",
    "fetch_metadata",
    "fetch_metadata_tool",
    "fetch_text",
    "fetch_text_tool",
    "get_classify_definition",
    "get_definition",
    "get_fetch_text_definition",
    "get_list_files_definition",
    "get_metadata_definition",
    "list_files",
    "list_files_tool",
    "plan_fetch",
    "plan_fetch_tool",
]
