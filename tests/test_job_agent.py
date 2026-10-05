"""The job agent's command line: endpoints, job IDs, rendering, exit codes.

Everything here is testable without a server because the CLI module holds only
presentation and process concerns — the analysis is
:mod:`interfaces.agent.job_agent.composer`, and the one place the two meet
(:func:`~interfaces.agent.job_agent.cli._run`) is replaced in the two tests
that exercise argument plumbing end to end.

The exit-code contract is the part worth pinning: an operator script reacts to
the number, not to the prose, and the difference between "a job broke" and "a
job had nothing to show" is the difference between paging someone and not.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from interfaces.agent.job_agent import cli
from interfaces.agent.job_agent.composer import (
    OUTCOME_ANALYSED,
    OUTCOME_ERROR,
    OUTCOME_NO_LOGS,
    JobAnalysisResult,
)


def _result(
    job_id: int = 6799893074,
    outcome: str = OUTCOME_ANALYSED,
    **overrides: Any,
) -> JobAnalysisResult:
    """Build a completed result for the renderers.

    Args:
        job_id: The job ID.
        outcome: The outcome.
        **overrides: Attributes to set afterwards.

    Returns:
        The result.
    """
    result = JobAnalysisResult(
        job_id=job_id,
        outcome=outcome,
        metadata={
            "monitor_url": f"https://bigpanda.cern.ch/job?pandaid={job_id}",
            "piloterrorcode": 1305,
            "piloterrordiag": "Payload execution failed",
            "jobstatus": "failed",
            "computingsite": "CERN-PROD",
            "atlasrelease": "21.0.15",
            "jeditaskid": 12345678,
            "attemptnr": 2,
            "maxattempt": 5,
            "pilot_version_from_pilotid": "3.14.0.22",
        },
        plans=[{"strategy": "payload_1305", "next": [], "done": True}],
        fetched=[{
            "filename": "payload.stdout",
            "role": "primary",
            "available": True,
            "url": "https://bigpanda.cern.ch/filebrowser/?filename=payload.stdout",
            "pilot_version": "",
        }],
        verdict={
            "failure_type": "payload_error",
            "context": {
                "excerpt": "Segmentation fault",
                "exception": {"exc_type": "ValueError"},
                "traceback_count": 1,
            },
        },
        fetch_order=["payload.stdout"],
        answer_markdown="The payload segfaulted.",
        tool_calls=4,
        llm_calls=1,
        elapsed_s=1.5,
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

def test_url_beats_host_and_port() -> None:
    """``--url`` is the only form that can carry TLS or a custom path."""
    assert cli.resolve_url("https://host/x/mcp", "other", 9000) == "https://host/x/mcp"


def test_host_and_port_compose_an_endpoint() -> None:
    """The ergonomic form builds the usual ``/mcp`` path."""
    assert cli.resolve_url(None, "aipanda033.cern.ch", 8000) == (
        "http://aipanda033.cern.ch:8000/mcp"
    )


def test_either_half_of_host_and_port_falls_back_to_the_other_default() -> None:
    """Naming one half must not silently fall through to the environment."""
    assert cli.resolve_url(None, "remote", None) == "http://remote:8000/mcp"
    assert cli.resolve_url(None, None, 9001) == "http://localhost:9001/mcp"


def test_the_environment_is_used_only_when_nothing_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BAMBOO_MCP_HTTP_URL`` is a default, not an override.

    Args:
        monkeypatch: Pytest fixture.
    """
    monkeypatch.setenv("BAMBOO_MCP_HTTP_URL", "http://env:1234/mcp")
    assert cli.resolve_url(None, None, None) == "http://env:1234/mcp"
    assert cli.resolve_url(None, "cli", None) == "http://cli:8000/mcp"


def test_the_endpoint_falls_back_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no flags and no environment, the local server is assumed.

    Args:
        monkeypatch: Pytest fixture.
    """
    monkeypatch.delenv("BAMBOO_MCP_HTTP_URL", raising=False)
    assert cli.resolve_url(None, None, None) == cli.DEFAULT_HTTP_URL


# ---------------------------------------------------------------------------
# Job IDs
# ---------------------------------------------------------------------------

def test_repeated_flags_keep_their_order() -> None:
    """``--panda-id`` repeats, and the batch runs in the order given."""
    assert cli.read_job_ids(["3", "1", "2"], None) == [3, 1, 2]


def test_duplicate_ids_are_analysed_once() -> None:
    """A repeat would cost a second full analysis for an identical answer."""
    assert cli.read_job_ids(["7", "7", "8", "7"], None) == [7, 8]


def test_ids_are_read_from_stdin_when_no_flag_was_given() -> None:
    """Piped input is the batch form that composes with other tools."""
    assert cli.read_job_ids(None, "11\n12\n\n13\n") == [11, 12, 13]
    assert cli.read_job_ids(None, "21 22\t23") == [21, 22, 23]


def test_a_flag_wins_over_stdin() -> None:
    """Explicit IDs are not merged with whatever happened to be on stdin."""
    assert cli.read_job_ids(["5"], "99\n") == [5]


def test_a_leading_hash_is_tolerated() -> None:
    """Job IDs pasted from a ticket often arrive as ``#6799893074``."""
    assert cli.read_job_ids(["#6799893074"], None) == [6799893074]


def test_a_non_numeric_id_names_the_offender() -> None:
    """The error says which value was wrong, not that one of them was."""
    with pytest.raises(ValueError) as excinfo:
        cli.read_job_ids(["123", "task-42"], None)
    assert "task-42" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

def test_everything_analysed_is_success() -> None:
    """All jobs analysed exits 0."""
    assert cli.exit_code([_result(), _result(2)]) == cli.EXIT_OK


def test_a_job_without_logs_exits_three() -> None:
    """Nothing broke; one job simply had nothing to show."""
    assert cli.exit_code([_result(outcome=OUTCOME_NO_LOGS)]) == cli.EXIT_NO_LOGS


def test_an_error_outranks_a_missing_log() -> None:
    """A run with both reports the breakage, which is the one needing attention."""
    results = [_result(outcome=OUTCOME_NO_LOGS), _result(2, outcome=OUTCOME_ERROR)]
    assert cli.exit_code(results) == cli.EXIT_ERROR


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_text_output_leads_with_the_job_and_the_verdict() -> None:
    """The first line answers "which job, and what was wrong"."""
    out = cli.format_text(_result(), verbose=False)
    assert "Job 6799893074 — payload_error" in out
    assert "CERN-PROD" in out
    assert "1305" in out
    assert "The payload segfaulted." in out
    assert "(tools=4, llm=1, 1.50s)" in out


def test_text_output_hides_the_excerpt_unless_asked() -> None:
    """The excerpt can be thousands of characters; ``--verbose`` opts in."""
    quiet = cli.format_text(_result(), verbose=False)
    loud = cli.format_text(_result(), verbose=True)

    assert "Segmentation fault" not in quiet
    assert "Segmentation fault" in loud
    assert "log excerpt" in loud


def test_text_output_marks_a_job_with_no_log_content() -> None:
    """An operator must not read "unknown" as "we found nothing wrong"."""
    out = cli.format_text(_result(outcome=OUTCOME_NO_LOGS), verbose=False)
    assert "no log content" in out


def test_text_output_for_a_failed_analysis_shows_why() -> None:
    """A failed analysis renders its reason rather than an empty fact block."""
    out = cli.format_text(
        _result(outcome=OUTCOME_ERROR, error="metadata fetch failed"), verbose=False
    )
    assert "ANALYSIS FAILED" in out
    assert "metadata fetch failed" in out


def test_markdown_output_is_the_answer_with_a_heading() -> None:
    """The form to paste into a ticket."""
    out = cli.format_markdown(_result())
    assert out.startswith("## Job 6799893074 — payload_error")
    assert "[BigPanDA](" in out
    assert "The payload segfaulted." in out


def test_markdown_falls_back_to_a_summary_without_synthesis() -> None:
    """``--no-synthesis --format markdown`` must not produce an empty document."""
    out = cli.format_markdown(_result(answer_markdown=""))
    assert "payload_error" in out
    assert "payload.stdout" in out


def test_json_is_an_object_for_one_job_and_an_array_for_several() -> None:
    """One job in, one object out; a batch is always an array."""
    single = json.loads(cli.render([_result()], "json", verbose=False))
    assert isinstance(single, dict)
    assert single["job_id"] == 6799893074

    several = json.loads(cli.render([_result(1), _result(2)], "json", verbose=False))
    assert isinstance(several, list)
    assert [r["job_id"] for r in several] == [1, 2]


def test_jsonl_is_one_object_per_line_whatever_the_batch_size() -> None:
    """The predictable shape for a pipeline."""
    out = cli.render([_result(1), _result(2)], "jsonl", verbose=False)
    lines = out.splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["job_id"] for line in lines] == [1, 2]

    assert len(cli.render([_result(1)], "jsonl", verbose=False).splitlines()) == 1


def test_every_format_is_renderable() -> None:
    """No format left unimplemented behind the choices list."""
    for fmt in cli.FORMATS:
        assert cli.render([_result()], fmt, verbose=False)


# ---------------------------------------------------------------------------
# Connect hints
# ---------------------------------------------------------------------------

_401 = "Client error '401 Unauthorized' for url 'http://localhost:8000/mcp'"
_403 = "Client error '403 Forbidden' for url 'http://localhost:8000/mcp'"


def test_a_401_without_a_token_says_how_to_supply_one() -> None:
    """The common case: auth is on at the server and the client knows nothing of it."""
    hint = cli.connect_hint(_401, "")
    assert "--token" in hint
    assert "BAMBOO_MCP_TOKEN" in hint
    assert "BAMBOO_MCP_TOKENS_FILE" in hint


def test_a_401_with_a_token_does_not_repeat_the_advice() -> None:
    """Telling someone to pass a token when they did sends them the wrong way."""
    hint = cli.connect_hint(_401, "some-token")
    assert "--token" not in hint
    assert "raw token value" in hint


def test_a_403_points_at_the_allowlist_not_at_the_missing_header() -> None:
    """403 means the token arrived and was refused — a different problem."""
    hint = cli.connect_hint(_403, "some-token")
    assert "allowlist" in hint
    assert "--token" not in hint


def test_an_ordinary_connection_failure_gets_no_hint() -> None:
    """A refused connection is self-explanatory; advice would be noise."""
    assert cli.connect_hint("All connection attempts failed", "") == ""


# ---------------------------------------------------------------------------
# Connect logging
# ---------------------------------------------------------------------------

def test_the_rollback_traceback_is_hidden_at_the_default_level() -> None:
    """Mistyping a port must not produce forty lines of anyio traceback."""
    import logging

    client_logger = logging.getLogger("interfaces.shared.mcp_client")
    before = client_logger.level

    with cli.quiet_connect_logging("WARNING"):
        assert client_logger.level == logging.ERROR

    assert client_logger.level == before


def test_an_explicit_log_level_is_honoured() -> None:
    """``--log-level INFO`` means the operator asked for the detail."""
    import logging

    client_logger = logging.getLogger("interfaces.shared.mcp_client")
    before = client_logger.level

    with cli.quiet_connect_logging("INFO"):
        assert client_logger.level == before

    assert client_logger.level == before


def test_the_level_is_restored_after_a_failed_connect() -> None:
    """The suppression is scoped to the call, including on the error path."""
    import logging

    client_logger = logging.getLogger("interfaces.shared.mcp_client")
    before = client_logger.level

    with pytest.raises(RuntimeError):
        with cli.quiet_connect_logging("WARNING"):
            raise RuntimeError("connection refused")

    assert client_logger.level == before


# ---------------------------------------------------------------------------
# Parser and main
# ---------------------------------------------------------------------------

def test_parser_defaults_are_the_documented_ones() -> None:
    """Defaults match the help text and the module docstring."""
    args = cli.build_parser().parse_args(["--panda-id", "1"])

    assert args.panda_id == ["1"]
    assert args.format == "text"
    assert args.no_synthesis is False
    assert args.with_listing is False
    assert args.host is None and args.port is None and args.url is None
    assert args.timeout is None


def test_repeating_the_flag_collects_every_value() -> None:
    """``action="append"`` rather than ``nargs``, so IDs cannot swallow a flag."""
    args = cli.build_parser().parse_args(
        ["--panda-id", "1", "--panda-id", "2", "--no-synthesis"]
    )
    assert args.panda_id == ["1", "2"]
    assert args.no_synthesis is True


class _FakeStdin:
    """A stdin that claims to be a terminal.

    Attributes:
        text: What :meth:`read` returns.
    """

    def __init__(self, text: str = "", tty: bool = True) -> None:
        """Configure the fake.

        Args:
            text: Text to return from ``read``.
            tty: What ``isatty`` reports.
        """
        self.text = text
        self._tty = tty

    def isatty(self) -> bool:
        """Report whether this is a terminal.

        Returns:
            The configured value.
        """
        return self._tty

    def read(self) -> str:
        """Return the configured text.

        Returns:
            The text.
        """
        return self.text


def test_no_job_ids_prints_help_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A usage error is a configuration problem, not an analysis failure.

    Args:
        monkeypatch: Pytest fixture.
        capsys: Pytest fixture.
    """
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())
    assert cli.main([]) == cli.EXIT_CONNECT
    assert "--panda-id" in capsys.readouterr().err + capsys.readouterr().out


def test_a_bad_job_id_exits_one_without_connecting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ID is rejected before a session is opened.

    Args:
        monkeypatch: Pytest fixture.
        capsys: Pytest fixture.
    """
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())

    async def _must_not_run(args: Any, job_ids: list[int]) -> int:
        """Fail if the runner is reached.

        Args:
            args: Ignored.
            job_ids: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("must not connect")

    monkeypatch.setattr(cli, "_run", _must_not_run)
    assert cli.main(["--panda-id", "oops"]) == cli.EXIT_CONNECT
    assert "oops" in capsys.readouterr().err


def test_main_forwards_the_parsed_ids_and_returns_the_runner_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argument plumbing from the command line through to the composer.

    Args:
        monkeypatch: Pytest fixture.
    """
    seen: dict[str, Any] = {}

    async def _capture(args: Any, job_ids: list[int]) -> int:
        """Record the arguments the runner received.

        Args:
            args: Parsed arguments.
            job_ids: Resolved job IDs.

        Returns:
            A distinctive exit code.
        """
        seen["job_ids"] = job_ids
        seen["format"] = args.format
        seen["synthesise"] = not args.no_synthesis
        seen["timeout"] = args.timeout
        return cli.EXIT_NO_LOGS

    monkeypatch.setattr(cli, "_run", _capture)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin())

    code = cli.main([
        "--panda-id", "11", "--panda-id", "12",
        "--format", "jsonl", "--no-synthesis", "--timeout", "30",
    ])

    assert code == cli.EXIT_NO_LOGS
    assert seen == {
        "job_ids": [11, 12],
        "format": "jsonl",
        "synthesise": False,
        "timeout": 30,
    }


def test_ids_piped_in_reach_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cut -f1 jobs.tsv | python -m interfaces.agent.job_agent`` works.

    Args:
        monkeypatch: Pytest fixture.
    """
    seen: dict[str, Any] = {}

    async def _capture(args: Any, job_ids: list[int]) -> int:
        """Record the job IDs.

        Args:
            args: Ignored.
            job_ids: Resolved job IDs.

        Returns:
            Success.
        """
        seen["job_ids"] = job_ids
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "_run", _capture)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("101\n102\n", tty=False))

    assert cli.main([]) == cli.EXIT_OK
    assert seen["job_ids"] == [101, 102]
