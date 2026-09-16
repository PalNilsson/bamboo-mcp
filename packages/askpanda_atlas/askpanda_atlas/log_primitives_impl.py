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
from collections.abc import Mapping
from typing import Any, cast

from askpanda_atlas._fallback_http import get_base_url
from askpanda_atlas._traceback_parse import coerce_bool
from askpanda_atlas.log_analysis_impl import (
    _fetch_file_listing,
    _fetch_metadata,
    _file_is_nonempty,
    _log_file_url,
    _select_log_filename,
    _top_level_file_index,
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

        ``bamboo.tools.base`` is imported here (deferred) so the rest of this
        module stays importable when bamboo core is not installed.  The
        blocking HTTP calls are offloaded via ``asyncio.to_thread``.

        Args:
            arguments: Dict with required ``job_id`` (int) and optional
                ``observed`` (mapping) and ``timeout`` (int).

        Returns:
            A ``(content, structured)`` tuple.  ``content`` is a one-element
            MCP content list holding the JSON-serialised payload; ``structured``
            is the same payload as a dict, validated by the SDK against the
            declared ``outputSchema``.
        """
        from bamboo.tools.base import text_content  # deferred — see class docstring

        def _result(payload: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
            return text_content(json.dumps(payload)), payload

        if not isinstance(arguments, dict):
            return _result({"error": "arguments must be a dict"})

        raw_job_id: Any = arguments.get("job_id")
        if raw_job_id is None:
            return _result({"error": "missing job_id"})
        try:
            job_id: int = int(raw_job_id)
        except (ValueError, TypeError):
            return _result({"error": "job_id must be an integer"})

        timeout: int = 60
        try:
            timeout = int(arguments.get("timeout") or 60)
        except (ValueError, TypeError):
            pass

        base_url: str = get_base_url()

        try:
            plan: dict[str, Any] = await asyncio.to_thread(
                plan_fetch, job_id, base_url, timeout, arguments.get("observed")
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error planning log fetch for job %d", job_id)
            return _result({"job_id": job_id, "error": repr(exc)})

        return _result(plan)


plan_fetch_tool = AtlasLogPlanFetchTool()

__all__ = [
    "AtlasLogPlanFetchTool",
    "PAYLOAD_STDERR",
    "PAYLOAD_STDOUT",
    "ROLE_PRIMARY",
    "ROLE_SECONDARY",
    "ROLE_SETUP",
    "SETUP_LOG",
    "SETUP_SIGNAL",
    "STRATEGY_METADATA_ONLY",
    "STRATEGY_PAYLOAD_1305",
    "STRATEGY_PILOTLOG",
    "get_definition",
    "plan_fetch",
    "plan_fetch_tool",
]
