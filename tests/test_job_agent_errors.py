"""Error paths, bounds and payload extraction for the job agent.

``tests/test_job_agent.py`` drives the loop over real primitives and pins what
it does with a well-behaved server.  This module covers what it does with a
badly-behaved one, which the scenario table structurally cannot reach: a
primitive returning an ``error`` payload, a server that does not know the
tool, a ``plan_fetch`` that never terminates, a malformed plan entry, and an
``mcp`` SDK old enough to drop structured content.

The divide that matters here is which failures are fatal and to what.  A
``fetch_text`` failure costs one file; ``classify`` reads the list tolerantly
and the job still has a verdict.  A ``fetch_metadata``, ``plan_fetch`` or
``classify`` failure costs the job, and is returned as
``outcome == "error"`` rather than raised so that a batch continues.  A tool
the server does not have costs the whole run, because every later job would
fail identically, and that one is allowed to propagate.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from interfaces.agent.job_agent import composer
from interfaces.agent.job_agent.composer import (
    MAX_PLAN_CALLS,
    OUTCOME_ANALYSED,
    OUTCOME_ERROR,
    OUTCOME_NO_LOGS,
    JobAnalysisResult,
    ToolCallError,
    ToolUnavailableError,
    analyse_job,
    analyse_jobs,
    structured_payload,
    text_from_result,
)

_JOB_ID = 6799893074


# ---------------------------------------------------------------------------
# A scripted client
# ---------------------------------------------------------------------------

class Result:
    """A tool result carrying both halves.

    Attributes:
        content: Unstructured content list.
        structuredContent: Structured payload.
    """

    def __init__(self, payload: dict[str, Any] | None) -> None:
        """Build a result from a payload.

        Args:
            payload: The structured payload, or ``None`` for a text-only
                result such as ``bamboo_llm_answer`` returns.
        """
        text = json.dumps(payload) if payload is not None else "an answer"
        self.content = [{"type": "text", "text": text}]
        self.structuredContent = payload  # noqa: N815 - the SDK's own spelling


class ScriptedClient:
    """A client returning canned payloads per tool.

    Attributes:
        calls: Every ``(name, arguments)`` pair received, in order.
    """

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        plans: list[dict[str, Any]] | None = None,
        fetches: dict[str, dict[str, Any]] | None = None,
        verdict: dict[str, Any] | None = None,
        listing: dict[str, Any] | None = None,
        unknown: set[str] | None = None,
        llm_raises: bool = False,
    ) -> None:
        """Script one server's responses.

        Args:
            metadata: ``fetch_metadata`` payload.
            plans: ``plan_fetch`` payloads, consumed in order; the last one
                repeats once exhausted.
            fetches: ``fetch_text`` payloads keyed by filename.  A filename
                with no entry gets a generic available result.
            verdict: ``classify`` payload.
            listing: ``list_files`` payload.
            unknown: Tool names the server does not know.
            llm_raises: Whether ``bamboo_llm_answer`` fails.
        """
        self._metadata = metadata if metadata is not None else _metadata_payload()
        self._plans = list(plans or [_terminal_plan([])])
        self._fetches = dict(fetches or {})
        self._verdict = verdict if verdict is not None else _verdict_payload()
        self._listing = listing if listing is not None else {"job_id": _JOB_ID, "files": []}
        self._unknown = set(unknown or ())
        self._llm_raises = llm_raises
        self._plan_index = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        """Advertise every primitive.

        Returns:
            A listing dict.
        """
        return {"tools": [{"name": n} for n in composer.PRIMITIVE_TOOLS]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Return the scripted response for one call.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            A :class:`Result`.

        Raises:
            RuntimeError: When the tool is scripted as unknown, or when the
                synthesis call is scripted to fail.
        """
        self.calls.append((name, dict(arguments)))
        if name in self._unknown:
            raise RuntimeError(f"Unknown tool: {name}")

        if name == composer.TOOL_FETCH_METADATA:
            return Result(self._metadata)
        if name == composer.TOOL_LIST_FILES:
            return Result(self._listing)
        if name == composer.TOOL_PLAN_FETCH:
            plan = self._plans[min(self._plan_index, len(self._plans) - 1)]
            self._plan_index += 1
            return Result(plan)
        if name == composer.TOOL_FETCH_TEXT:
            filename = str(arguments.get("filename") or "")
            return Result(self._fetches.get(filename) or _fetch_payload(filename))
        if name == composer.TOOL_CLASSIFY:
            return Result(self._verdict)
        if name == composer.TOOL_LLM_ANSWER:
            if self._llm_raises:
                raise RuntimeError("provider unavailable")
            return Result(None)
        raise RuntimeError(f"Unknown tool: {name}")

    def count(self, tool: str) -> int:
        """Count calls to one tool.

        Args:
            tool: Tool name.

        Returns:
            The number of calls.
        """
        return sum(1 for name, _ in self.calls if name == tool)


def _metadata_payload(**overrides: Any) -> dict[str, Any]:
    """Build a ``fetch_metadata`` payload.

    Args:
        **overrides: Keys to replace.

    Returns:
        The payload.
    """
    payload: dict[str, Any] = {
        "job_id": _JOB_ID,
        "monitor_url": f"https://bigpanda.cern.ch/job?pandaid={_JOB_ID}",
        "piloterrorcode": 1305,
        "piloterrordiag": "Payload execution failed",
        "pilotid": "host|3.14.0.22",
        "pilot_version_from_pilotid": "3.14.0.22",
        "jobstatus": "failed",
        "computingsite": "CERN-PROD",
    }
    payload.update(overrides)
    return payload


def _terminal_plan(entries: list[Any]) -> dict[str, Any]:
    """Build a terminal ``plan_fetch`` payload.

    Args:
        entries: The ``next`` entries.  Typed loosely so a test can place
            a deliberately malformed entry among well-formed ones.

    Returns:
        The payload.
    """
    return {
        "job_id": _JOB_ID,
        "strategy": "pilotlog",
        "next": entries,
        "done": True,
        "notes": [],
    }


def _entry(filename: str, role: str = composer.ROLE_PRIMARY) -> dict[str, Any]:
    """Build one plan entry.

    Args:
        filename: File to fetch.
        role: Fetch role.

    Returns:
        The entry.
    """
    return {
        "filename": filename,
        "role": role,
        "url": f"https://bigpanda.cern.ch/filebrowser/?filename={filename}",
        "reason": "test",
    }


def _fetch_payload(filename: str, **overrides: Any) -> dict[str, Any]:
    """Build a successful ``fetch_text`` payload.

    Args:
        filename: The file read.
        **overrides: Keys to replace.

    Returns:
        The payload.
    """
    payload: dict[str, Any] = {
        "job_id": _JOB_ID,
        "filename": filename,
        "url": f"https://bigpanda.cern.ch/filebrowser/?filename={filename}",
        "role": composer.ROLE_PRIMARY,
        "available": True,
        "bytes": 1234,
        "truncated": False,
        "context": {"excerpt": "boom", "exception": None, "traceback_count": 0},
        "signals": {},
        "pilot_version": "",
        "notes": [],
    }
    payload.update(overrides)
    return payload


def _verdict_payload(**overrides: Any) -> dict[str, Any]:
    """Build a ``classify`` payload.

    Args:
        **overrides: Keys to replace.

    Returns:
        The payload.
    """
    payload: dict[str, Any] = {
        "failure_type": "payload_error",
        "context": {"excerpt": "boom", "exception": None, "traceback_count": 0},
        "notes": [],
    }
    payload.update(overrides)
    return payload


def run(client: ScriptedClient, **kwargs: Any) -> JobAnalysisResult:
    """Analyse one job synchronously.

    Args:
        client: The scripted client.
        **kwargs: Forwarded to :func:`analyse_job`.

    Returns:
        The result.
    """
    kwargs.setdefault("synthesise", False)
    return asyncio.run(analyse_job(client, _JOB_ID, **kwargs))


# ---------------------------------------------------------------------------
# Fatal per-job failures
# ---------------------------------------------------------------------------

def test_a_metadata_error_ends_the_job_without_raising() -> None:
    """A failed metadata fetch is reported, not raised."""
    client = ScriptedClient(
        metadata={"job_id": _JOB_ID, "error": "Failed to fetch job metadata from BigPanDA"}
    )
    result = run(client)

    assert result.outcome == OUTCOME_ERROR
    assert result.error is not None
    assert "Failed to fetch job metadata" in result.error
    assert client.count(composer.TOOL_PLAN_FETCH) == 0


def test_a_plan_error_ends_the_job() -> None:
    """A failed plan leaves nothing to fetch, so the job stops there."""
    client = ScriptedClient(plans=[{"job_id": _JOB_ID, "error": "listing exploded"}])
    result = run(client)

    assert result.outcome == OUTCOME_ERROR
    assert result.error is not None
    assert "listing exploded" in result.error
    assert client.count(composer.TOOL_CLASSIFY) == 0


def test_a_classify_error_ends_the_job() -> None:
    """A failed classification leaves no verdict to report."""
    client = ScriptedClient(
        plans=[_terminal_plan([_entry("pilotlog.txt")])],
        verdict={"error": "classifier blew up"},
    )
    result = run(client)

    assert result.outcome == OUTCOME_ERROR
    assert result.error is not None
    assert "classifier blew up" in result.error


# ---------------------------------------------------------------------------
# Non-fatal failures
# ---------------------------------------------------------------------------

def test_an_unreadable_file_is_noted_and_the_job_still_gets_a_verdict() -> None:
    """A ``fetch_text`` error costs its file, not the analysis.

    ``classify`` reads the fetched list tolerantly, so the entry is still
    passed through: a job whose payload log could not be read is still worth
    classifying from its metadata and whatever else came back.
    """
    client = ScriptedClient(
        plans=[_terminal_plan([_entry("payload.stdout")])],
        fetches={
            "payload.stdout": {
                "job_id": _JOB_ID,
                "filename": "payload.stdout",
                "error": "download timed out",
            }
        },
    )
    result = run(client)

    assert result.outcome == OUTCOME_NO_LOGS
    assert result.error is None
    assert result.fetch_order == ["payload.stdout"]
    assert any("download timed out" in note for note in result.notes)
    assert client.count(composer.TOOL_CLASSIFY) == 1

    passed = [a for n, a in client.calls if n == composer.TOOL_CLASSIFY][0]
    assert passed["fetched"] == result.fetched


def test_a_synthesis_failure_degrades_rather_than_discards() -> None:
    """The evidence is the valuable part and survives a dead LLM provider."""
    client = ScriptedClient(
        plans=[_terminal_plan([_entry("pilotlog.txt")])], llm_raises=True
    )
    result = run(client, synthesise=True)

    assert result.outcome == OUTCOME_ANALYSED
    assert result.error is None
    assert result.answer_markdown == ""
    assert result.llm_calls == 0
    assert any("synthesis failed" in note for note in result.notes)
    assert result.failure_type == "payload_error"


def test_a_listing_error_is_a_note_not_a_failure() -> None:
    """``list_files`` is off the diagnosis path, so its failure cannot be fatal."""
    client = ScriptedClient(
        plans=[_terminal_plan([_entry("pilotlog.txt")])],
        listing={"job_id": _JOB_ID, "error": "filebrowser unreachable"},
    )
    result = run(client, with_listing=True)

    assert result.outcome == OUTCOME_ANALYSED
    assert result.error is None
    assert any("filebrowser unreachable" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Fatal for the run
# ---------------------------------------------------------------------------

def test_an_unregistered_primitive_aborts_the_run() -> None:
    """An ePIC server has no ``atlas.log.*``, and no later job would fare better."""
    client = ScriptedClient(unknown={composer.TOOL_FETCH_METADATA})

    with pytest.raises(ToolUnavailableError) as excinfo:
        run(client)

    message = str(excinfo.value)
    assert "ASKPANDA_PLUGIN" in message
    assert "BAMBOO_TOOL_PROFILE does not matter" in message


def test_an_unregistered_primitive_propagates_out_of_a_batch() -> None:
    """The batch stops rather than recording the same error once per job."""
    client = ScriptedClient(unknown={composer.TOOL_FETCH_METADATA})

    with pytest.raises(ToolUnavailableError):
        asyncio.run(analyse_jobs(client, [1, 2, 3], synthesise=False))

    assert client.count(composer.TOOL_FETCH_METADATA) == 1


def test_a_per_job_error_does_not_stop_a_batch() -> None:
    """One job failing to classify leaves the rest of the batch to run."""

    class _FlakyClient(ScriptedClient):
        """Fails ``classify`` for the second job only."""

        def __init__(self) -> None:
            """Script a terminal plan and a counter."""
            super().__init__(plans=[_terminal_plan([_entry("pilotlog.txt")])])
            self._classify_calls = 0

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            """Fail the second classification.

            Args:
                name: Tool name.
                arguments: Tool arguments.

            Returns:
                A :class:`Result`.
            """
            if name == composer.TOOL_CLASSIFY:
                self._classify_calls += 1
                if self._classify_calls == 2:
                    self.calls.append((name, dict(arguments)))
                    return Result({"error": "second job only"})
            return await super().call_tool(name, arguments)

    client = _FlakyClient()
    results = asyncio.run(analyse_jobs(client, [11, 22, 33], synthesise=False))

    assert [r.job_id for r in results] == [11, 22, 33]
    assert [r.outcome for r in results] == [
        OUTCOME_ANALYSED, OUTCOME_ERROR, OUTCOME_ANALYSED
    ]


# ---------------------------------------------------------------------------
# Loop bounds and tolerance
# ---------------------------------------------------------------------------

def test_a_plan_that_never_terminates_is_cut_off_at_the_bound() -> None:
    """A server stuck on ``done: false`` costs three plans, not an infinite loop."""
    never_done = {
        "job_id": _JOB_ID,
        "strategy": "payload_1305",
        "next": [_entry("a.txt")],
        "done": False,
        "notes": [],
    }
    client = ScriptedClient(plans=[never_done])
    result = run(client)

    assert client.count(composer.TOOL_PLAN_FETCH) == MAX_PLAN_CALLS
    assert any("plan_fetch calls" in note for note in result.notes)
    assert client.count(composer.TOOL_CLASSIFY) == 1


def test_a_terminal_plan_with_files_is_fetched_before_the_loop_exits() -> None:
    """``done: true`` alongside a non-empty ``next`` is the terminal plan.

    Testing ``done`` before fetching would silently skip the payload logs of
    every 1305 job, which is the most common failure Bamboo is asked about.
    """
    client = ScriptedClient(
        plans=[_terminal_plan([_entry("payload.stdout"), _entry("payload.stderr")])]
    )
    result = run(client)

    assert result.fetch_order == ["payload.stdout", "payload.stderr"]
    assert client.count(composer.TOOL_PLAN_FETCH) == 1


def test_a_malformed_plan_entry_costs_its_own_file_only() -> None:
    """One unusable entry is skipped; the rest of the plan is still fetched."""
    plan = _terminal_plan([
        {"role": "primary"},               # no filename
        "not an entry",                    # not a mapping
        _entry("pilotlog.txt"),
    ])
    client = ScriptedClient(plans=[plan])
    result = run(client)

    assert result.fetch_order == ["pilotlog.txt"]
    assert result.outcome == OUTCOME_ANALYSED


def test_the_timeout_is_passed_to_every_primitive_that_takes_one() -> None:
    """``--timeout`` reaches the network-touching primitives, and not ``classify``."""
    client = ScriptedClient(plans=[_terminal_plan([_entry("pilotlog.txt")])])
    run(client, timeout=15)

    for name, args in client.calls:
        if name == composer.TOOL_CLASSIFY:
            assert "timeout" not in args
        else:
            assert args.get("timeout") == 15


# ---------------------------------------------------------------------------
# Payload extraction
# ---------------------------------------------------------------------------

def test_structured_content_is_preferred_over_the_text_half() -> None:
    """Both halves carry the payload; the structured one is authoritative."""
    result = Result({"job_id": 1, "failure_type": "x"})
    result.content = [{"type": "text", "text": '{"job_id": 999}'}]

    assert structured_payload(result, tool="t")["job_id"] == 1


def test_a_tuple_result_is_read_as_the_in_process_tool_shape() -> None:
    """An in-process tool object returns ``(content, structured)``."""
    payload = {"job_id": 7}
    assert structured_payload(([{"type": "text", "text": "{}"}], payload), tool="t") == payload


def test_a_text_only_result_is_recovered_by_parsing() -> None:
    """An SDK that dropped structured content is recovered from, not failed on."""

    class _TextOnly:
        """A result carrying no structured half."""

        content = [{"type": "text", "text": '{"job_id": 5, "error": "nope"}'}]

    payload = structured_payload(_TextOnly(), tool="atlas.log.classify")
    assert payload == {"job_id": 5, "error": "nope"}


def test_an_unreadable_result_names_the_sdk_floor() -> None:
    """The failure an mcp SDK below 1.10.0 causes is diagnosed, not guessed at."""

    class _Empty:
        """A result carrying nothing usable."""

        content: list[Any] = []

    with pytest.raises(ToolCallError) as excinfo:
        structured_payload(_Empty(), tool="atlas.log.plan_fetch")

    assert "1.10.0" in str(excinfo.value)


def test_text_is_joined_across_content_blocks() -> None:
    """Both dict blocks and attribute-style blocks are read."""

    class _Block:
        """An SDK-style text block."""

        type = "text"
        text = "second"

    class _Mixed:
        """A result with one dict block and one object block."""

        content = [{"type": "text", "text": "first"}, _Block()]

    assert text_from_result(_Mixed()) == "first\nsecond"
