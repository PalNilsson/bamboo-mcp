"""The composed loop, the result it produces, and the synthesis call.

This module transcribes the composition documented in ``docs/code-mode.md``
and nothing else.  Read that document before changing anything here: each of
the four rules below exists because the primitives place a domain decision
server-side, and a caller that paraphrases one of them stops being equivalent
to ``panda_log_analysis`` without any test here going red.

1. ``observed`` accumulates the ``signals`` mapping a fetch returned, by
   ``update``.  It is never reconstructed from the excerpt, and no predicate
   is recomputed client-side.  ``plan_fetch`` counts the *presence* of
   ``setup_has_error`` as "the setup log has been read", so synthesising the
   key from a file that is not ``setup.stdout`` makes the next plan skip the
   setup log entirely.
2. ``filename`` and ``role`` are passed to ``fetch_text`` exactly as
   ``plan_fetch`` gave them.  The role sets the character budget.
3. ``done`` means *no further plan call is required*, and may be ``true``
   alongside a non-empty ``next`` — that is the terminal plan.  The loop
   therefore fetches first and tests ``done`` afterwards.
4. ``classify`` receives the metadata subset unmodified and the fetch results
   **in fetch order**.  It joins the excerpts itself because both join rules —
   the separator, and the precedence that prefers the stderr traceback — are
   domain rules.

Failures arrive as data.  Every primitive declares an ``outputSchema`` with no
top-level ``required`` and every error path returns a normal structured result
carrying an ``error`` string, so ``error`` is checked before anything else is
read.  A failed ``fetch_text`` is **not** fatal: the entry is still appended,
because ``classify`` reads the list tolerantly and a job whose payload log was
unreadable still has a verdict worth reporting.

Transport belongs to the caller.  Nothing here imports an MCP client; the
:class:`MCPCallable` protocol is the whole dependency, which is what lets the
CLI pass a real session and a test pass the primitive tool objects directly.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool names and loop constants
# ---------------------------------------------------------------------------

TOOL_FETCH_METADATA: str = "atlas.log.fetch_metadata"
TOOL_PLAN_FETCH: str = "atlas.log.plan_fetch"
TOOL_FETCH_TEXT: str = "atlas.log.fetch_text"
TOOL_CLASSIFY: str = "atlas.log.classify"
TOOL_LIST_FILES: str = "atlas.log.list_files"
TOOL_LLM_ANSWER: str = "bamboo_llm_answer"

#: The five primitives, in the order ``docs/code-mode.md`` introduces them.
#: Used by :func:`missing_primitives` for the advisory preflight only — the
#: profile switch gates advertising, not dispatch, so a name absent from
#: ``tools/list`` is still callable and absence is a warning, not a failure.
PRIMITIVE_TOOLS: tuple[str, ...] = (
    TOOL_PLAN_FETCH,
    TOOL_FETCH_METADATA,
    TOOL_LIST_FILES,
    TOOL_FETCH_TEXT,
    TOOL_CLASSIFY,
)

#: Hard client-side bound on ``plan_fetch`` calls for one job, matching the
#: bound the primitives assert server-side.  Defence in depth rather than
#: control flow: a caller that re-plans past the terminal plan is answered
#: ``{"next": [], "done": true}``, so the loop below already terminates.
MAX_PLAN_CALLS: int = 3

ROLE_SETUP: str = "setup"
ROLE_PRIMARY: str = "primary"
ROLE_SECONDARY: str = "secondary"

#: Evidence key each role's URL lands in on the ``panda_log_analysis`` side.
#: Mirrors ``_LogFetchResult``'s URL fields so :meth:`JobAnalysisResult.to_evidence`
#: is a field mapping rather than an interpretation.
_URL_KEY_FOR_ROLE: dict[str, str] = {
    ROLE_SETUP: "setup_log_url",
    ROLE_PRIMARY: "log_url",
    ROLE_SECONDARY: "stderr_url",
}

OUTCOME_ANALYSED: str = "analysed"
OUTCOME_NO_LOGS: str = "no_logs"
OUTCOME_ERROR: str = "error"

DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_TEMPERATURE: float = 0.1

#: Substrings that mark a tool-call failure as "this server does not have that
#: tool" rather than "that call went wrong".  The distinction matters because
#: the first is fatal for every subsequent job too — an ePIC server has no
#: ``atlas.log.*`` at all — while the second is a per-job problem.
_UNKNOWN_TOOL_MARKERS: tuple[str, ...] = (
    "unknown tool",
    "tool not found",
    "no such tool",
    "not registered",
)

_UNKNOWN_TOOL_ADVICE: str = (
    "The ATLAS log primitives are not registered on this server. They are "
    "ATLAS-only: check ASKPANDA_PLUGIN is 'atlas' and that askpanda_atlas is "
    "installed. (BAMBOO_TOOL_PROFILE does not matter here — it gates "
    "advertising, not dispatch.)"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class JobAgentError(RuntimeError):
    """Base class for failures raised while analysing a job."""


class ToolCallError(JobAgentError):
    """An MCP tool call did not complete."""


class ToolUnavailableError(ToolCallError):
    """A required primitive is not registered on the server.

    Fatal for the whole run rather than for one job: every subsequent job
    would fail identically, so :func:`analyse_jobs` re-raises instead of
    recording it per job.
    """


class PrimitiveError(JobAgentError):
    """A primitive returned a structured payload carrying an ``error`` key."""


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------

class MCPCallable(Protocol):
    """The slice of an MCP client this module needs.

    Satisfied by :class:`interfaces.shared.mcp_client.MCPAsyncClient` and by
    any test double that dispatches the two coroutines.  Declaring a protocol
    rather than importing the client keeps this module free of the MCP SDK,
    so an evaluation harness can drive the loop in-process.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one MCP tool.

        Args:
            name: Tool name as the server registers it.
            arguments: JSON-compatible argument mapping.

        Returns:
            The raw tool result.
        """
        ...  # pragma: no cover - protocol declaration

    async def list_tools(self) -> Any:
        """List the tools the server advertises.

        Returns:
            The raw listing result.
        """
        ...  # pragma: no cover - protocol declaration


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def text_from_result(result: Any) -> str:
    """Return the concatenated text content blocks of a tool result.

    Args:
        result: Raw result from ``call_tool``.  Content blocks may be SDK
            objects carrying ``type``/``text`` attributes or plain dicts, and
            an in-process tool object returns the ``(content, structured)``
            tuple instead.

    Returns:
        The joined text of every text block, or an empty string when the
        result carries none.
    """
    content: Any = result
    if isinstance(result, tuple) and len(result) == 2:
        content = result[0]
    else:
        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = result.get("content")

    if not isinstance(content, (list, tuple)):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            continue
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(p for p in parts if p)


def structured_payload(result: Any, *, tool: str) -> dict[str, Any]:
    """Return the structured payload of a primitive's result.

    Read from ``structuredContent`` rather than from the text half.  Both
    carry the same payload — ``_tool_result`` JSON-serialises one into the
    other — but the text half is what a generic agent truncates, and a
    truncated excerpt is invalid JSON rather than a short one.

    Args:
        result: Raw result from ``call_tool``, an SDK ``CallToolResult``, a
            mapping with a ``structuredContent`` key, or the
            ``(content, structured)`` tuple an in-process tool object returns.
        tool: Tool name, for the error message only.

    Returns:
        The payload dict.

    Raises:
        ToolCallError: When no structured payload can be recovered.  Falling
            back to parsing the text half is attempted first, because an
            ``mcp`` SDK below 1.10.0 drops structured content silently and
            that failure is otherwise reported as an empty answer.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return dict(result[1])

    structured: Any = getattr(result, "structuredContent", None)
    if structured is None and isinstance(result, dict):
        structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return dict(structured)

    text: str = text_from_result(result)
    if text:
        try:
            parsed: Any = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            logger.debug("%s: recovered payload from text content", tool)
            return dict(parsed)

    raise ToolCallError(
        f"{tool} returned no structured content. The server's mcp SDK must be "
        f">= 1.10.0; below that floor outputSchema results are dropped silently."
    )


def _advertised_tool_names(listing: Any) -> list[str]:
    """Return the tool names in a ``tools/list`` result.

    Args:
        listing: Raw result from ``list_tools``.

    Returns:
        Names in listing order; empty when the shape is unrecognised.
    """
    tools: Any = getattr(listing, "tools", None)
    if tools is None and isinstance(listing, dict):
        tools = listing.get("tools")
    if tools is None:
        tools = listing
    if not isinstance(tools, (list, tuple)):
        return []

    names: list[str] = []
    for item in tools:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = getattr(item, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return names


async def missing_primitives(client: MCPCallable) -> list[str]:
    """Return the primitives this server does not advertise.

    Advisory only.  ``BAMBOO_TOOL_PROFILE`` gates ``tools/list`` and not
    ``call_tool``, so a server under the default ``orchestrated`` profile
    advertises none of these and serves all of them.  A non-empty result is
    worth one warning, never a refusal to proceed — refusing would be
    refusing on a property that does not hold.

    Args:
        client: A connected MCP client.

    Returns:
        The missing names in :data:`PRIMITIVE_TOOLS` order.  An empty list
        when the listing could not be read at all, since "we could not ask"
        is not "they are absent".
    """
    try:
        listing = await client.list_tools()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("preflight: could not list tools: %s", exc)
        return []

    advertised = set(_advertised_tool_names(listing))
    if not advertised:
        return []
    return [name for name in PRIMITIVE_TOOLS if name not in advertised]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class JobAnalysisResult:
    """Everything one analysed job produced.

    Attributes:
        job_id: The PanDA job ID analysed.
        outcome: One of :data:`OUTCOME_ANALYSED`, :data:`OUTCOME_NO_LOGS` or
            :data:`OUTCOME_ERROR`.
        metadata: The ``atlas.log.fetch_metadata`` payload, unmodified.
        plans: Each ``atlas.log.plan_fetch`` payload, in call order.
        fetched: Each ``atlas.log.fetch_text`` payload, in fetch order.
        verdict: The ``atlas.log.classify`` payload.
        listing: The ``atlas.log.list_files`` payload, when it was requested.
        fetch_order: Filenames downloaded, in order.  Pinned by the scenario
            tests because an agreeing verdict says the loop arrived somewhere
            and an agreeing download order says it arrived the documented way.
        answer_markdown: The synthesised prose, empty when synthesis was off.
        notes: Remarks accumulated from the primitives and from the loop.
        error: Why the analysis stopped, when ``outcome`` is
            :data:`OUTCOME_ERROR`.
        tool_calls: MCP tool calls made for this job, synthesis excluded.
        llm_calls: Synthesis calls made for this job: 0 or 1.
        elapsed_s: Wall-clock seconds spent on this job.
    """

    job_id: int
    outcome: str = OUTCOME_ERROR
    metadata: dict[str, Any] = field(default_factory=dict)
    plans: list[dict[str, Any]] = field(default_factory=list)
    fetched: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    listing: dict[str, Any] | None = None
    fetch_order: list[str] = field(default_factory=list)
    answer_markdown: str = ""
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    tool_calls: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0

    # -- derived ---------------------------------------------------------

    @property
    def failure_type(self) -> str:
        """Return the classification verdict.

        Returns:
            The ``failure_type`` the classifier reached, or ``"unknown"``.
        """
        return str(self.verdict.get("failure_type") or "unknown")

    @property
    def monitor_url(self) -> str:
        """Return the BigPanDA page for the job.

        Returns:
            The monitor URL, or an empty string when metadata was not read.
        """
        return str(self.metadata.get("monitor_url") or "")

    @property
    def log_available(self) -> bool:
        """Report whether any log content was obtained.

        Returns:
            ``True`` when at least one fetch came back available.  This is the
            same predicate the REST facade derives its ``no_log`` flag from,
            which is why :data:`OUTCOME_NO_LOGS` is keyed on it rather than on
            the ``metadata_only`` strategy: a job whose only log file was
            empty has no log to show either.
        """
        return any(entry.get("available") is True for entry in self.fetched)

    @property
    def pilot_version(self) -> str:
        """Return the pilot release the job ran.

        Returns:
            The first non-empty version any fetched file reported — only
            ``pilotlog.txt`` ever reports one — falling back to the version
            parsed out of ``pilotid``.  Empty when neither is available.
        """
        for entry in self.fetched:
            version = str(entry.get("pilot_version") or "")
            if version:
                return version
        return str(self.metadata.get("pilot_version_from_pilotid") or "")

    @property
    def strategy(self) -> str:
        """Return the log-selection strategy the first plan chose.

        Returns:
            ``payload_1305``, ``pilotlog``, ``metadata_only``, or an empty
            string when no plan was obtained.
        """
        return str(self.plans[0].get("strategy") or "") if self.plans else ""

    # -- projections -----------------------------------------------------

    def to_evidence(self) -> dict[str, Any]:
        """Project the result onto ``panda_log_analysis``'s evidence keys.

        The comparable subset, in the monolith's own vocabulary, so that a
        consumer written against the compound tool — the REST facade's
        ``evidence`` field, the monitor's rendering, the Track A
        granularity study — reads this without a translation layer.

        Three differences from the monolith's bundle are intended and
        documented in ``docs/code-mode.md``: there is no link block,
        follow-up offer or core-dump probe here; ``context.exception.raw`` is
        capped at the tool boundary, so only ``exception_type`` and
        ``traceback_count`` are comparable; and a URL appears only for a file
        that was actually downloaded, where ``_fetch_logs_payload`` assigns
        ``log_url`` before it knows whether it will read the file.

        Returns:
            Evidence dict with the monolith's key names.
        """
        context: dict[str, Any] = self.verdict.get("context") or {}
        exception: Any = context.get("exception")
        urls: dict[str, str] = {}
        for entry in self.fetched:
            key = _URL_KEY_FOR_ROLE.get(str(entry.get("role") or ""))
            url = str(entry.get("url") or "")
            if key and url and not urls.get(key):
                urls[key] = url

        return {
            "failure_type": self.failure_type,
            "log_excerpt": str(context.get("excerpt") or ""),
            "exception_type": (
                exception.get("exc_type") if isinstance(exception, dict) else None
            ),
            "traceback_count": int(context.get("traceback_count") or 0),
            "pilot_version": self.pilot_version,
            "log_available": self.log_available,
            "log_url": urls.get("log_url", ""),
            "setup_log_url": urls.get("setup_log_url", ""),
            "stderr_url": urls.get("stderr_url", ""),
            "monitor_url": self.monitor_url,
            "piloterrorcode": self.metadata.get("piloterrorcode"),
            "piloterrordiag": self.metadata.get("piloterrordiag"),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole result for JSON output.

        Returns:
            JSON-compatible dict carrying every field plus the derived
            ``evidence`` projection.
        """
        return {
            "job_id": self.job_id,
            "outcome": self.outcome,
            "failure_type": self.failure_type,
            "strategy": self.strategy,
            "monitor_url": self.monitor_url,
            "pilot_version": self.pilot_version,
            "fetch_order": list(self.fetch_order),
            "answer_markdown": self.answer_markdown,
            "evidence": self.to_evidence(),
            "metadata": self.metadata,
            "plans": self.plans,
            "fetched": self.fetched,
            "verdict": self.verdict,
            "listing": self.listing,
            "notes": list(self.notes),
            "error": self.error,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

_SYSTEM_SYNTHESIS: str = (
    "You are Bamboo, an assistant for the ATLAS experiment's PanDA "
    "distributed computing system. You are given the evidence that Bamboo's "
    "log-analysis primitives gathered for one failed grid job.\n\n"
    "Explain why the job failed and what the operator or user should do next.\n\n"
    "Rules:\n"
    "- Use only the evidence given. Do not invent pilot error codes, site "
    "names, release numbers, file names or exception types.\n"
    "- Lead with the cause in one or two sentences, then the supporting "
    "detail.\n"
    "- Distinguish a payload failure (the user's job) from an infrastructure "
    "failure (pilot, stage-in/stage-out, setup, site) and say which this is.\n"
    "- When the evidence does not determine the cause, say so plainly and "
    "name what is missing. A confident wrong answer is worse than an "
    "uncertain one.\n"
    "- Keep it under roughly 300 words. Markdown is fine; no diagrams."
)

#: Metadata fields worth putting in front of the model, with the label it sees.
#: A subset rather than the whole metadata payload: the twenty pass-through
#: fields include several that are empty on most jobs, and an evidence brief
#: padded with blanks teaches the model that blanks are normal.
_BRIEF_FIELDS: tuple[tuple[str, str], ...] = (
    ("jobstatus", "Job status"),
    ("jobsubstatus", "Job substatus"),
    ("computingsite", "Site"),
    ("cloud", "Cloud"),
    ("atlasrelease", "Release"),
    ("transformation", "Transformation"),
    ("jeditaskid", "Task ID"),
    ("attemptnr", "Attempt"),
    ("maxattempt", "Max attempts"),
    ("exeerrorcode", "Exe error code"),
    ("exeerrordiag", "Exe error diag"),
    ("taskbuffererrorcode", "TaskBuffer error code"),
    ("taskbuffererrordiag", "TaskBuffer error diag"),
    ("ddmerrorcode", "DDM error code"),
    ("ddmerrordiag", "DDM error diag"),
    ("duration", "Duration"),
    ("commandtopilot", "Command to pilot"),
)


def build_synthesis_brief(result: JobAnalysisResult) -> str:
    """Render the evidence a job produced as the synthesis prompt's user turn.

    Separated from the call so it can be inspected and asserted without an
    LLM, and so the Track A faithfulness layer has a stable, reproducible
    input to score an answer against.

    Args:
        result: A completed analysis, before synthesis.

    Returns:
        The brief as plain text.
    """
    lines: list[str] = [f"PanDA job {result.job_id}"]
    if result.monitor_url:
        lines.append(f"Monitor: {result.monitor_url}")

    code = result.metadata.get("piloterrorcode")
    diag = str(result.metadata.get("piloterrordiag") or "")
    if code or diag:
        lines.append(f"Pilot error: {code or 0} — {diag or '(no diagnosis text)'}")

    for key, label in _BRIEF_FIELDS:
        value = result.metadata.get(key)
        if value in (None, "", 0):
            continue
        lines.append(f"{label}: {value}")

    if result.pilot_version:
        lines.append(f"Pilot version: {result.pilot_version}")

    lines.append(f"Bamboo classification: {result.failure_type}")
    lines.append(f"Log selection strategy: {result.strategy or 'unknown'}")

    if result.fetch_order:
        lines.append(f"Files read: {', '.join(result.fetch_order)}")
    else:
        lines.append("Files read: none (no log files were downloaded)")

    context: dict[str, Any] = result.verdict.get("context") or {}
    exception: Any = context.get("exception")
    if isinstance(exception, dict) and exception.get("exc_type"):
        lines.append(
            f"Exception: {exception.get('exc_type')}: "
            f"{exception.get('exc_value') or ''}".rstrip(": ")
        )
    count = int(context.get("traceback_count") or 0)
    if count:
        lines.append(f"Tracebacks found: {count}")

    for note in result.notes:
        lines.append(f"Note: {note}")

    excerpt = str(context.get("excerpt") or "")
    if excerpt:
        lines.append("")
        lines.append("Log excerpt:")
        lines.append("```")
        lines.append(excerpt)
        lines.append("```")
    else:
        lines.append("")
        lines.append("No log excerpt is available for this job.")

    return "\n".join(lines)


async def _synthesise(
    client: MCPCallable,
    result: JobAnalysisResult,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Turn gathered evidence into prose with one server-side LLM call.

    Routed through ``bamboo_llm_answer`` rather than an LLM client of this
    agent's own, so the answer uses whatever provider and model the server is
    configured with and the agent needs no API key.  The server prepends its
    own system prompt and extends with these messages.

    Args:
        client: A connected MCP client.
        result: The completed analysis to describe.
        max_tokens: Token budget for the completion.
        temperature: Sampling temperature.

    Returns:
        The model's text, or an empty string when the call produced none.

    Raises:
        ToolCallError: When the ``bamboo_llm_answer`` call itself fails.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_SYNTHESIS},
        {"role": "user", "content": build_synthesis_brief(result)},
    ]
    try:
        raw = await client.call_tool(
            TOOL_LLM_ANSWER,
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ToolCallError(f"{TOOL_LLM_ANSWER} failed: {exc}") from exc
    return text_from_result(raw).strip()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

async def _call(
    client: MCPCallable,
    tool: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call one primitive and return its structured payload.

    Args:
        client: A connected MCP client.
        tool: The primitive's registered name.
        arguments: Arguments for the call.

    Returns:
        The structured payload, which may carry an ``error`` key.

    Raises:
        ToolUnavailableError: When the server does not know the tool.
        ToolCallError: When the call failed for any other reason.
    """
    try:
        raw = await client.call_tool(tool, arguments)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        message = str(exc).lower()
        if any(marker in message for marker in _UNKNOWN_TOOL_MARKERS):
            raise ToolUnavailableError(f"{tool}: {exc}. {_UNKNOWN_TOOL_ADVICE}") from exc
        raise ToolCallError(f"{tool} failed: {exc}") from exc
    return structured_payload(raw, tool=tool)


async def _call_checked(
    client: MCPCallable,
    tool: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call one primitive and reject a payload carrying an ``error``.

    Used for the three steps whose failure leaves nothing to classify.
    ``fetch_text`` is deliberately not one of them.

    Args:
        client: A connected MCP client.
        tool: The primitive's registered name.
        arguments: Arguments for the call.

    Returns:
        The structured payload.

    Raises:
        PrimitiveError: When the payload carries an ``error`` key.
        ToolCallError: When the call itself failed.
    """
    payload = await _call(client, tool, arguments)
    error = payload.get("error")
    if error:
        raise PrimitiveError(f"{tool}: {error}")
    return payload


def _plan_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the fetchable entries of a plan.

    Args:
        plan: A ``plan_fetch`` payload.

    Returns:
        Entries carrying a non-empty ``filename``.  Malformed entries are
        dropped rather than raising, matching the tolerance the primitives
        apply to their own arguments: one bad entry costs its own file, not
        the whole analysis.
    """
    raw: Any = plan.get("next")
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("filename") or ""):
            entries.append(item)
    return entries


async def _run_fetch_loop(
    client: MCPCallable,
    job_id: int,
    base_args: dict[str, Any],
    result: JobAnalysisResult,
    progress: Callable[[str], None] | None,
) -> None:
    """Drive the plan/fetch loop, filling ``result`` in place.

    Args:
        client: A connected MCP client.
        job_id: The job being analysed, for progress messages.
        base_args: ``job_id`` and, when set, ``timeout``.
        result: The result being built; ``plans``, ``fetched``,
            ``fetch_order``, ``notes`` and ``tool_calls`` are appended to.
        progress: Optional status callback.

    Raises:
        PrimitiveError: When a ``plan_fetch`` call returns an error payload.
        ToolCallError: When a call fails.
    """
    plan = await _call_checked(client, TOOL_PLAN_FETCH, dict(base_args))
    result.tool_calls += 1
    result.plans.append(plan)
    result.notes.extend(str(n) for n in (plan.get("notes") or []))

    observed: dict[str, Any] = {"fetched": []}
    plan_calls = 1

    while True:
        for entry in _plan_entries(plan):
            filename = str(entry["filename"])
            role = str(entry.get("role") or ROLE_PRIMARY)
            if progress:
                progress(f"job {job_id}: fetching {filename} ({role})")

            got = await _call(
                client,
                TOOL_FETCH_TEXT,
                {**base_args, "filename": filename, "role": role},
            )
            result.tool_calls += 1
            result.fetched.append(got)
            result.fetch_order.append(filename)
            result.notes.extend(str(n) for n in (got.get("notes") or []))

            fetch_error = got.get("error")
            if fetch_error:
                # Not fatal: classify reads the list tolerantly, and a job
                # whose payload log was unreadable still has a verdict.
                result.notes.append(f"{filename}: {fetch_error}")

            observed["fetched"].append(filename)
            signals: Any = got.get("signals")
            if isinstance(signals, dict):
                observed.update(signals)

        if plan.get("done"):
            break

        if plan_calls >= MAX_PLAN_CALLS:
            result.notes.append(
                f"stopped after {MAX_PLAN_CALLS} plan_fetch calls without a "
                f"terminal plan; classifying on what was fetched"
            )
            break

        if progress:
            progress(f"job {job_id}: re-planning ({plan_calls + 1}/{MAX_PLAN_CALLS})")
        # Snapshot rather than pass the live dict: the loop keeps mutating
        # ``observed``, and a client that serialises the arguments after
        # returning control would send a later round's state.
        snapshot: dict[str, Any] = {
            **observed, "fetched": list(observed["fetched"]),
        }
        plan = await _call_checked(
            client, TOOL_PLAN_FETCH, {**base_args, "observed": snapshot}
        )
        plan_calls += 1
        result.tool_calls += 1
        result.plans.append(plan)
        result.notes.extend(str(n) for n in (plan.get("notes") or []))


async def analyse_job(
    client: MCPCallable,
    job_id: int,
    *,
    timeout: int | None = None,
    synthesise: bool = True,
    with_listing: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    progress: Callable[[str], None] | None = None,
) -> JobAnalysisResult:
    """Analyse one failed PanDA job by composing the log primitives.

    Runs the composition documented in ``docs/code-mode.md``: metadata, then
    plan/fetch until the plan says ``done``, then classify — and, unless
    ``synthesise`` is off, one ``bamboo_llm_answer`` call to turn the evidence
    into prose.  No domain predicate is evaluated here; every decision stays
    server-side.

    Args:
        client: A connected MCP client.
        job_id: PanDA job ID (``pandaid``) to analyse.
        timeout: HTTP timeout in seconds passed to each primitive.  ``None``
            leaves the server's default of 60.
        synthesise: Whether to produce ``answer_markdown``.  Off means zero
            LLM calls and a fully deterministic result.
        with_listing: Also call ``atlas.log.list_files``.  Off the diagnosis
            path — it answers "what did this job produce" — so it is opt-in
            and its failure is recorded as a note, never as an error.
        max_tokens: Token budget for the synthesis call.
        temperature: Sampling temperature for the synthesis call.
        progress: Optional callback receiving one short status line per step.

    Returns:
        The completed :class:`JobAnalysisResult`.  A per-job failure is
        reported as ``outcome == "error"`` with ``error`` set, not raised, so
        a batch run can continue.

    Raises:
        ToolUnavailableError: When a primitive is not registered on the
            server.  Fatal for the whole run: every other job would fail the
            same way, so this one error is allowed to propagate.
    """
    started = time.monotonic()
    result = JobAnalysisResult(job_id=job_id)
    base_args: dict[str, Any] = {"job_id": job_id}
    if timeout is not None:
        base_args["timeout"] = timeout

    try:
        if progress:
            progress(f"job {job_id}: fetching metadata")
        result.metadata = await _call_checked(
            client, TOOL_FETCH_METADATA, dict(base_args)
        )
        result.tool_calls += 1

        if with_listing:
            listing = await _call(client, TOOL_LIST_FILES, dict(base_args))
            result.tool_calls += 1
            result.listing = listing
            if listing.get("error"):
                result.notes.append(f"{TOOL_LIST_FILES}: {listing['error']}")

        await _run_fetch_loop(client, job_id, base_args, result, progress)

        if progress:
            progress(f"job {job_id}: classifying")
        result.verdict = await _call_checked(
            client,
            TOOL_CLASSIFY,
            {"job": result.metadata, "fetched": result.fetched},
        )
        result.tool_calls += 1
        result.notes.extend(str(n) for n in (result.verdict.get("notes") or []))

    except ToolUnavailableError:
        raise
    except JobAgentError as exc:
        result.outcome = OUTCOME_ERROR
        result.error = str(exc)
        result.elapsed_s = time.monotonic() - started
        return result

    result.outcome = OUTCOME_ANALYSED if result.log_available else OUTCOME_NO_LOGS

    if synthesise:
        if progress:
            progress(f"job {job_id}: synthesising")
        try:
            result.answer_markdown = await _synthesise(
                client, result, max_tokens=max_tokens, temperature=temperature
            )
            result.llm_calls += 1
        except ToolCallError as exc:
            # The evidence is already gathered and is the valuable part, so a
            # synthesis failure degrades the result rather than discarding it.
            result.notes.append(f"synthesis failed: {exc}")

    result.elapsed_s = time.monotonic() - started
    return result


async def analyse_jobs(
    client: MCPCallable,
    job_ids: list[int],
    *,
    timeout: int | None = None,
    synthesise: bool = True,
    with_listing: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    progress: Callable[[str], None] | None = None,
) -> list[JobAnalysisResult]:
    """Analyse several jobs over one MCP session, in order.

    Sequential by design.  Concurrency would multiply the load this agent puts
    on BigPanDA for no gain — the metadata and listing caches are per job, so
    there is nothing for parallel jobs to share — and it would interleave the
    progress output into noise.

    Args:
        client: A connected MCP client, reused across every job.
        job_ids: The jobs to analyse, in order.
        timeout: HTTP timeout in seconds passed to each primitive.
        synthesise: Whether to produce ``answer_markdown`` per job.
        with_listing: Also call ``atlas.log.list_files`` per job.
        max_tokens: Token budget for each synthesis call.
        temperature: Sampling temperature for each synthesis call.
        progress: Optional callback receiving one short status line per step.

    Returns:
        One result per job, in the order given.

    Raises:
        ToolUnavailableError: Propagated from the first job that hits it,
            since no later job could succeed either.
    """
    results: list[JobAnalysisResult] = []
    for job_id in job_ids:
        results.append(
            await analyse_job(
                client,
                job_id,
                timeout=timeout,
                synthesise=synthesise,
                with_listing=with_listing,
                max_tokens=max_tokens,
                temperature=temperature,
                progress=progress,
            )
        )
    return results


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "MAX_PLAN_CALLS",
    "OUTCOME_ANALYSED",
    "OUTCOME_ERROR",
    "OUTCOME_NO_LOGS",
    "PRIMITIVE_TOOLS",
    "TOOL_CLASSIFY",
    "TOOL_FETCH_METADATA",
    "TOOL_FETCH_TEXT",
    "TOOL_LIST_FILES",
    "TOOL_LLM_ANSWER",
    "TOOL_PLAN_FETCH",
    "JobAgentError",
    "JobAnalysisResult",
    "MCPCallable",
    "PrimitiveError",
    "ToolCallError",
    "ToolUnavailableError",
    "analyse_job",
    "analyse_jobs",
    "build_synthesis_brief",
    "missing_primitives",
    "structured_payload",
    "text_from_result",
]
