"""The job agent's loop, driven over the shared log-analysis scenarios.

``packages/askpanda_atlas/tests/test_log_equivalence.py`` already holds the
*implementation* functions of the two log-analysis paths to one answer.  This
module holds the **client loop** to the same standard: it drives
:func:`interfaces.agent.job_agent.analyse_job` against a test double that
dispatches to the real primitive tool objects, and compares what comes out
against ``fetch_and_analyse`` over the same scenario.

Why the fetch order is asserted
-------------------------------
An agreeing verdict says the agent arrived where the monolith arrived.  An
agreeing **download order** says it arrived the documented way — setup log
first on a 1305 job, the early stop when that setup log reports an error, the
zero-length skips, the re-plan with ``observed``.  An agent that fetched
everything unconditionally and classified from the union would pass a verdict
comparison and fail this one, which is the point: the equivalence that makes
the composition worth having is a property of *that* loop, not of any loop.

``expect_fetched`` and ``expect_failure_type`` come from ``log_scenarios.py``
rather than from the monolith, so a rule changed in both paths at once — the
lockstep-drift failure — still has to be written down in the scenario table
where it is reviewable.

Loading the scenario table
--------------------------
By explicit path under a private module name rather than by putting
``packages/askpanda_atlas/tests`` on ``sys.path``.  The two suites are run as
separate pytest invocations on purpose; a shared top-level module name across
both rootdirs is exactly what breaks that.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

from askpanda_atlas import log_analysis_impl as mono
from askpanda_atlas import log_primitives_impl as impl
from interfaces.agent.job_agent import composer
from interfaces.agent.job_agent.composer import (
    MAX_PLAN_CALLS,
    OUTCOME_ANALYSED,
    OUTCOME_NO_LOGS,
    analyse_job,
    analyse_jobs,
    build_synthesis_brief,
    missing_primitives,
)

# ---------------------------------------------------------------------------
# Scenario table
# ---------------------------------------------------------------------------

_SCENARIOS_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "packages" / "askpanda_atlas" / "tests" / "log_scenarios.py"
)

if not _SCENARIOS_PATH.is_file():  # pragma: no cover - fixture tree absent
    pytest.skip(
        "askpanda_atlas test fixtures are not present in this checkout",
        allow_module_level=True,
    )

_SPEC = importlib.util.spec_from_file_location(
    "_bamboo_job_agent_log_scenarios", _SCENARIOS_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_SCENARIOS_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCENARIOS_MODULE
_SPEC.loader.exec_module(_SCENARIOS_MODULE)

SCENARIOS: list[Any] = list(_SCENARIOS_MODULE.SCENARIOS)

_JOB_ID = 6799893074
#: The primitives resolve the base URL themselves through ``get_base_url``,
#: so the monolith has to be driven with the same one or every URL comparison
#: fails on the host rather than on the path.
_BASE_URL = impl.get_base_url()
_TIMEOUT = 60

#: Evidence key each role's URL lands in on the monolith side.
_URL_KEY_FOR_ROLE: dict[str, str] = {
    composer.ROLE_SETUP: "setup_log_url",
    composer.ROLE_PRIMARY: "log_url",
    composer.ROLE_SECONDARY: "stderr_url",
}


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeToolResult:
    """An MCP ``CallToolResult`` as far as the composer reads one.

    Attributes:
        content: The unstructured content list.
        structuredContent: The structured payload, or ``None``.
    """

    def __init__(self, content: Any, structured: Any) -> None:
        """Store both halves of a tool result.

        Args:
            content: Unstructured content list.
            structured: Structured payload.
        """
        self.content = content
        self.structuredContent = structured  # noqa: N815 - the SDK's own spelling


class FakeListing:
    """A ``tools/list`` result carrying only names.

    Attributes:
        tools: The advertised tool descriptors.
    """

    def __init__(self, names: list[str]) -> None:
        """Build a listing from tool names.

        Args:
            names: Names to advertise.
        """
        self.tools = [{"name": name} for name in names]


class PrimitiveClient:
    """MCP client double dispatching to the real primitive tool objects.

    Calls go through the tool classes rather than the bare implementation
    functions, so the argument coercion, the ``(content, structured)`` tuple
    and the error payloads are all exercised — everything between the agent
    and the implementation except the wire itself.

    Attributes:
        calls: Every ``(name, arguments)`` pair received, in order.
        llm_text: What ``bamboo_llm_answer`` returns.
    """

    def __init__(
        self,
        *,
        advertised: list[str] | None = None,
        llm_text: str = "The payload raised ValueError during stage-in.",
    ) -> None:
        """Initialise the double.

        Args:
            advertised: Names ``list_tools`` reports.  Defaults to the five
                primitives plus ``bamboo_llm_answer``.
            llm_text: Text the synthesis call returns.
        """
        self._tools: dict[str, Any] = {
            composer.TOOL_PLAN_FETCH: impl.plan_fetch_tool,
            composer.TOOL_FETCH_METADATA: impl.fetch_metadata_tool,
            composer.TOOL_LIST_FILES: impl.list_files_tool,
            composer.TOOL_FETCH_TEXT: impl.fetch_text_tool,
            composer.TOOL_CLASSIFY: impl.classify_tool,
        }
        self._advertised = (
            advertised
            if advertised is not None
            else [*composer.PRIMITIVE_TOOLS, composer.TOOL_LLM_ANSWER]
        )
        self.llm_text = llm_text
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        """Return the advertised tool listing.

        Returns:
            A :class:`FakeListing`.
        """
        return FakeListing(self._advertised)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch one tool call to the real primitive.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            A :class:`FakeToolResult`.

        Raises:
            RuntimeError: When the name is not one this double serves, with
                the wording a real server uses so the composer's
                unknown-tool detection is exercised.
        """
        self.calls.append((name, dict(arguments)))
        if name == composer.TOOL_LLM_ANSWER:
            return FakeToolResult([{"type": "text", "text": self.llm_text}], None)
        tool = self._tools.get(name)
        if tool is None:
            raise RuntimeError(f"Unknown tool: {name}")
        content, structured = await tool.call(arguments)
        return FakeToolResult(content, structured)

    def names_called(self, tool: str) -> int:
        """Count the calls made to one tool.

        Args:
            tool: Tool name.

        Returns:
            The number of calls.
        """
        return sum(1 for name, _ in self.calls if name == tool)


def wire(monkeypatch: pytest.MonkeyPatch, module: Any, scenario: Any) -> list[str]:
    """Patch one module's metadata, listing and download entry points.

    ``log_primitives_impl`` imports the three fetch helpers into its own
    namespace, so the monolith and the primitives have to be patched
    separately and each gets its own spy.  One shared spy would record both
    paths into one list and make the fetch-order comparison meaningless.

    Args:
        monkeypatch: Pytest fixture.
        module: ``log_analysis_impl`` or ``log_primitives_impl``.
        scenario: The scenario supplying the responses.

    Returns:
        List accumulating the filenames this module downloads, in order.
    """
    fetched: list[str] = []

    def _metadata(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
        return {"job": scenario.job}

    def _listing(job_id: int, base_url: str, timeout: int) -> list[dict[str, Any]] | None:
        return scenario.listing()

    def _text(job_id: int, filename: str, base_url: str, timeout: int) -> str | None:
        fetched.append(filename)
        return scenario.text_for(filename)

    monkeypatch.setattr(module, "_fetch_metadata", _metadata)
    monkeypatch.setattr(module, "_fetch_file_listing", _listing)
    monkeypatch.setattr(module, "_fetch_log_text", _text)
    return fetched


def run_agent(client: PrimitiveClient, **kwargs: Any) -> Any:
    """Run one analysis synchronously.

    Args:
        client: The MCP client double.
        **kwargs: Forwarded to :func:`analyse_job`.

    Returns:
        The :class:`~interfaces.agent.job_agent.JobAnalysisResult`.
    """
    kwargs.setdefault("synthesise", False)
    return asyncio.run(analyse_job(client, _JOB_ID, **kwargs))


# ---------------------------------------------------------------------------
# The loop, scenario by scenario
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agent_reproduces_the_expected_fetch_order(
    monkeypatch: pytest.MonkeyPatch, scenario: Any
) -> None:
    """The agent downloads exactly the files the scenario table pins, in order.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the scenario table.
    """
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    assert result.error is None, result.error
    assert result.fetch_order == list(scenario.expect_fetched)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agent_reaches_the_expected_verdict(
    monkeypatch: pytest.MonkeyPatch, scenario: Any
) -> None:
    """The agent's classification matches the scenario table.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the scenario table.
    """
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    assert result.failure_type == scenario.expect_failure_type


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agent_evidence_agrees_with_the_monolith(
    monkeypatch: pytest.MonkeyPatch, scenario: Any
) -> None:
    """``to_evidence`` agrees with ``fetch_and_analyse`` key by key.

    The comparison is restricted to the roles the agent actually downloaded:
    ``_fetch_logs_payload`` assigns ``log_url`` before it knows whether it
    will read ``payload.stdout``, so the monolith can link a file neither path
    read.  ``context.exception.raw`` is excluded for the same documented
    reason — it is capped at the tool boundary and whole in the monolith.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the scenario table.
    """
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())
    evidence = result.to_evidence()

    wire(monkeypatch, mono, scenario)
    reference: dict[str, Any] = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)[
        "evidence"
    ]

    roles = {str(entry.get("role") or "") for entry in result.fetched}

    assert evidence["failure_type"] == reference["failure_type"]
    assert evidence["log_excerpt"] == (reference["log_excerpt"] or "")
    assert evidence["exception_type"] == reference["exception_type"]
    assert evidence["traceback_count"] == reference["traceback_count"]
    assert evidence["pilot_version"] == (reference["pilot_version"] or "")
    assert evidence["log_available"] == reference["log_available"]

    for role, key in _URL_KEY_FOR_ROLE.items():
        if role in roles and reference[key]:
            assert evidence[key] == reference[key]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agent_stays_within_the_plan_call_bound(
    monkeypatch: pytest.MonkeyPatch, scenario: Any
) -> None:
    """No scenario costs more than the documented three ``plan_fetch`` calls.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the scenario table.
    """
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()
    run_agent(client)

    assert client.names_called(composer.TOOL_PLAN_FETCH) <= MAX_PLAN_CALLS


# ---------------------------------------------------------------------------
# Loop shape
# ---------------------------------------------------------------------------

def _scenario(name: str) -> Any:
    """Return one scenario by name.

    Args:
        name: The scenario's ``name`` field.

    Returns:
        The scenario.
    """
    return _SCENARIOS_MODULE.by_name(name)


def test_the_agent_passes_filename_and_role_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``fetch_text`` argument pair came from a plan entry verbatim.

    The role sets the character budget server-side, so an agent that
    normalised, defaulted or re-derived it would silently change how much of
    each log is read.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()
    result = run_agent(client)

    offered: set[tuple[str, str]] = set()
    for plan in result.plans:
        for entry in plan.get("next") or []:
            offered.add((str(entry["filename"]), str(entry["role"])))

    for name, args in client.calls:
        if name == composer.TOOL_FETCH_TEXT:
            assert (str(args["filename"]), str(args["role"])) in offered


def test_the_agent_feeds_back_signals_rather_than_recomputing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-plan's ``observed`` carries the signals the fetch returned.

    ``plan_fetch`` counts the presence of ``setup_has_error`` as "the setup
    log has been read", so a caller that dropped or synthesised the key would
    change which files the next plan offers.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = _scenario("payload_1305_stdout_and_stderr")
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()
    run_agent(client)

    replans = [
        args for name, args in client.calls
        if name == composer.TOOL_PLAN_FETCH and "observed" in args
    ]
    assert replans, "a clean 1305 setup log must trigger exactly one re-plan"

    observed = replans[0]["observed"]
    assert observed["fetched"] == ["setup.stdout"]
    assert observed.get(impl.SETUP_SIGNAL) is False


def test_classify_receives_the_metadata_subset_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``classify`` is handed the metadata payload exactly as it arrived.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()
    result = run_agent(client)

    classify_args = [
        args for name, args in client.calls if name == composer.TOOL_CLASSIFY
    ]
    assert len(classify_args) == 1
    assert classify_args[0]["job"] == result.metadata
    assert classify_args[0]["fetched"] == result.fetched


def test_a_metadata_only_job_downloads_nothing_and_reports_no_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job that is not failed, holding or cancelled is classified from metadata.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = _scenario("metadata_only_finished_job")
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    assert result.fetch_order == []
    assert result.strategy == "metadata_only"
    assert result.outcome == OUTCOME_NO_LOGS
    assert result.log_available is False


def test_a_job_with_readable_logs_reports_analysed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal failed job with a readable log comes back ``analysed``.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = _scenario("pilotlog_stagein_timeout")
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    assert result.outcome == OUTCOME_ANALYSED
    assert result.log_available is True
    assert result.fetch_order == ["pilotlog.txt"]


def test_the_listing_is_only_fetched_when_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``atlas.log.list_files`` is off the diagnosis path and opt-in.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)

    default_client = PrimitiveClient()
    default_result = run_agent(default_client)
    assert default_client.names_called(composer.TOOL_LIST_FILES) == 0
    assert default_result.listing is None

    listing_client = PrimitiveClient()
    listing_result = run_agent(listing_client, with_listing=True)
    assert listing_client.names_called(composer.TOOL_LIST_FILES) == 1
    assert listing_result.listing is not None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def test_synthesis_is_one_call_and_off_by_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis costs exactly one LLM call, and ``--no-synthesis`` costs none.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)

    quiet = PrimitiveClient()
    quiet_result = run_agent(quiet, synthesise=False)
    assert quiet.names_called(composer.TOOL_LLM_ANSWER) == 0
    assert quiet_result.llm_calls == 0
    assert quiet_result.answer_markdown == ""

    loud = PrimitiveClient(llm_text="Because the payload crashed.")
    loud_result = run_agent(loud, synthesise=True)
    assert loud.names_called(composer.TOOL_LLM_ANSWER) == 1
    assert loud_result.llm_calls == 1
    assert loud_result.answer_markdown == "Because the payload crashed."


def test_the_brief_carries_the_excerpt_and_names_no_absent_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthesis brief holds the evidence and omits empty metadata fields.

    An evidence brief padded with blank fields teaches the model that blanks
    are normal, which is how an answer starts inventing values for them.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = _scenario("pilotlog_stagein_timeout")
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    brief = build_synthesis_brief(result)
    assert f"PanDA job {_JOB_ID}" in brief
    assert result.failure_type in brief
    assert "pilotlog.txt" in brief
    assert (result.verdict["context"]["excerpt"] or "") in brief
    assert ": None" not in brief
    assert ": \n" not in brief


def test_the_synthesis_payload_is_a_system_and_user_message_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bamboo_llm_answer`` is called with the house message shape.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()
    run_agent(client, synthesise=True, max_tokens=512, temperature=0.3)

    calls = [args for name, args in client.calls if name == composer.TOOL_LLM_ANSWER]
    assert len(calls) == 1
    messages = calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["temperature"] == 0.3


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_a_batch_reuses_one_session_and_keeps_job_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several jobs analysed in order over a single client.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)
    client = PrimitiveClient()

    job_ids = [_JOB_ID, _JOB_ID + 1, _JOB_ID + 2]
    results = asyncio.run(analyse_jobs(client, job_ids, synthesise=False))

    assert [r.job_id for r in results] == job_ids
    assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# Preflight and result shapes
# ---------------------------------------------------------------------------

def test_preflight_reports_primitives_the_server_does_not_advertise() -> None:
    """A server advertising only the monolith reports all five as missing."""
    client = PrimitiveClient(advertised=["panda_log_analysis", "bamboo_health"])
    assert asyncio.run(missing_primitives(client)) == list(composer.PRIMITIVE_TOOLS)


def test_preflight_is_silent_when_the_listing_cannot_be_read() -> None:
    """An unreadable listing is "we could not ask", not "they are absent"."""

    class _Broken:
        """A client whose listing call fails."""

        async def list_tools(self) -> Any:
            """Fail the way a disconnected session does.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("no session")

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            """Unused.

            Args:
                name: Ignored.
                arguments: Ignored.

            Raises:
                AssertionError: Always; the preflight must not call a tool.
            """
            raise AssertionError("preflight must not call tools")

    assert asyncio.run(missing_primitives(_Broken())) == []


def test_to_dict_round_trips_through_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serialised result is JSON-compatible as written.

    Args:
        monkeypatch: Pytest fixture.
    """
    scenario = SCENARIOS[0]
    wire(monkeypatch, impl, scenario)
    result = run_agent(PrimitiveClient())

    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["job_id"] == _JOB_ID
    assert payload["failure_type"] == result.failure_type
    assert payload["fetch_order"] == result.fetch_order
    assert payload["evidence"]["failure_type"] == result.failure_type
