"""Tests for panda_log_analysis tool (askpanda_epic plugin implementation).

All external HTTP calls are patched; no network access is required.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest

from askpanda_epic.log_analysis import (
    panda_log_analysis_tool,
    classify_failure,
    extract_log_excerpt,
)
from askpanda_epic.log_analysis_impl import (
    _extract_context_window,
    _extract_tail,
    _select_log_filename,
    _strip_payload_noise,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_JOB_STAGEIN_TIMEOUT: dict = {
    "pandaid": 6799893074,
    "jobstatus": "failed",
    "jobsubstatus": "",
    "computingsite": "UKI-SCOTGRID-GLASGOW_CEPH",
    "cloud": "UK",
    "atlasrelease": "",
    "jeditaskid": 46249501,
    "attemptnr": 1,
    "maxattempt": 3,
    "transformation": "epic_reco",
    "piloterrorcode": 1151,
    "piloterrordiag": (
        "File transfer timed out during stage-in: "
        "epic_data:epic_data.00483532... timeout=6842 seconds"
    ),
    "exeerrorcode": 0,
    "exeerrordiag": "",
    "taskbuffererrorcode": 0,
    "taskbuffererrordiag": "",
    "ddmerrorcode": 0,
    "ddmerrordiag": "",
    "starttime": "2025-09-08 05:50:33",
    "endtime": "2025-09-08 10:32:20",
    "duration": "4:41:47",
    "commandtopilot": "",
}

_SAMPLE_JOB_REASSIGNED: dict = {
    **_SAMPLE_JOB_STAGEIN_TIMEOUT,
    "pandaid": 6837798305,
    "jobstatus": "closed",
    "jobsubstatus": "toreassign",
    "piloterrorcode": 0,
    "piloterrordiag": "",
    "taskbuffererrorcode": 100,
    "taskbuffererrordiag": "reassigned by JEDI",
    "commandtopilot": "tobekilled",
}

_SAMPLE_JOB_PAYLOAD: dict = {
    **_SAMPLE_JOB_STAGEIN_TIMEOUT,
    "pandaid": 1111,
    "piloterrorcode": 1305,
    "piloterrordiag": "Payload error: AthenaMP exited with code 1",
}

_SAMPLE_PAYLOAD: dict = {
    "job": _SAMPLE_JOB_STAGEIN_TIMEOUT,
    "files": [],
    "dsfiles": [],
}

_SAMPLE_PILOT_LOG = "\n".join([
    "2025-09-08 05:50:33 | INFO | startup",
    "2025-09-08 05:50:34 | INFO | stage-in starting",
    "2025-09-08 10:32:18 | INFO | handle_rucio_error | TimeoutException: Timeout reached, timeout=6842 seconds",
    "2025-09-08 10:32:18 | WARNING | failed to transfer_files: File transfer timed out during stage-in",
    "2025-09-08 10:32:19 | ERROR | pilot error set to 1151",
    "2025-09-08 10:32:20 | INFO | job ended",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack(result: list) -> dict:
    """Deserialise the JSON-wrapped MCPContent returned by the tool.

    Args:
        result: Return value of tool.call().

    Returns:
        Deserialised dict with ``evidence`` and ``text`` keys.
    """
    return json.loads(result[0]["text"])


def _make_metadata_response(job: dict) -> dict:
    """Build a metadata response dict as BigPanDA would return.

    Args:
        job: Job metadata dict.

    Returns:
        Full metadata response payload.
    """
    return {"job": job, "files": [], "dsfiles": []}


# ---------------------------------------------------------------------------
# Unit tests: pure functions
# ---------------------------------------------------------------------------


def test_classify_failure_stagein_timeout() -> None:
    """Stage-in timeout is correctly classified."""
    result = classify_failure(_SAMPLE_JOB_STAGEIN_TIMEOUT, _SAMPLE_PILOT_LOG)
    assert result == "stagein_timeout"


def test_classify_failure_stageout_timeout_not_misclassified_as_stagein() -> None:
    """Stage-out timeout (code 1152) is classified as stageout_timeout, not stagein_timeout.

    Regression test for a misclassification where the piloterrordiag for code
    1152 begins with "File transfer timed out during stage-out", which also
    matches the stagein_timeout keyword "file transfer timed out".
    stageout_timeout must appear before stagein_timeout in _FAILURE_PATTERNS,
    and its keyword list must include the exact diag prefix so the more
    specific match wins.
    """
    job = {
        **_SAMPLE_JOB_STAGEIN_TIMEOUT,
        "piloterrorcode": 1152,
        "piloterrordiag": (
            "File transfer timed out during stage-out: "
            "hc_test:output.1.10f2da00_79600.pool.root to CERN-PROD_SCRATCHDISK, "
            "copy command timed out: TimeoutException: Timeout reached, timeout=410 seconds"
        ),
    }
    result = classify_failure(job, "")
    assert result == "stageout_timeout", (
        f"Expected 'stageout_timeout', got '{result}'. "
        "Code 1152 diag starts with 'File transfer timed out during stage-out' — "
        "stageout_timeout must be checked before stagein_timeout."
    )


def test_classify_failure_reassigned() -> None:
    """JEDI reassignment is correctly classified from metadata."""
    result = classify_failure(_SAMPLE_JOB_REASSIGNED, "")
    assert result == "reassigned_by_jedi"


def test_classify_failure_unknown() -> None:
    """Unrecognised errors fall back to 'unknown'."""
    job = {k: "" for k in _SAMPLE_JOB_STAGEIN_TIMEOUT}
    result = classify_failure(job, "something completely unrecognised")
    assert result == "unknown"


def test_classify_failure_segfault_from_log() -> None:
    """Segfault classification is driven by log excerpt."""
    job = {**_SAMPLE_JOB_STAGEIN_TIMEOUT, "piloterrordiag": ""}
    result = classify_failure(job, "Segmentation fault in AthenaMP\n")
    assert result == "segfault"


def test_extract_context_window_finds_match() -> None:
    """Context window extraction returns lines up to the pattern match."""
    lines = ["line1\n", "line2\n", "ERROR: timeout\n", "line4\n"]
    log_text = "".join(lines)
    result = _extract_context_window(log_text, "timeout", n_lines=10)
    assert "ERROR: timeout" in result
    assert "line1" in result
    assert "line4" not in result


def test_extract_context_window_no_match_returns_empty() -> None:
    """Context window extraction returns empty string when pattern not found."""
    result = _extract_context_window("line1\nline2\n", "NOTPRESENT", n_lines=5)
    assert result == ""


def test_extract_tail_returns_last_n_lines() -> None:
    """Tail extraction returns only the last N lines."""
    log = "\n".join(f"line{i}" for i in range(20))
    result = _extract_tail(log, n_lines=5)
    assert "line19" in result
    assert "line15" in result
    assert "line14" not in result


def test_select_log_filename_payload_error() -> None:
    """Pilot error 1305 selects payload.stdout."""
    job = {**_SAMPLE_JOB_STAGEIN_TIMEOUT, "piloterrorcode": 1305}
    assert _select_log_filename(job) == "payload.stdout"


def test_select_log_filename_pilot_error() -> None:
    """All other pilot errors select pilotlog.txt."""
    assert _select_log_filename(_SAMPLE_JOB_STAGEIN_TIMEOUT) == "pilotlog.txt"


def test_extract_log_excerpt_uses_pattern_for_pilotlog() -> None:
    """extract_log_excerpt uses code-specific pattern for pilotlog.txt."""
    excerpt = extract_log_excerpt(
        _SAMPLE_PILOT_LOG, "pilotlog.txt",
        pilot_error_code=1151,
        pilot_error_diag="File transfer timed out",
    )
    assert "timed out" in excerpt.lower() or "timeout" in excerpt.lower()


def test_extract_log_excerpt_uses_char_tail_for_payload() -> None:
    """extract_log_excerpt uses a char-based tail for payload.stdout (code 1305).

    A char-based tail guarantees that ERROR lines near the end of a long
    verbose stdout are captured even when the per-line budget would cut them
    off.  The excerpt should contain the final lines regardless of how many
    verbose INFO lines precede them.
    """
    # Simulate verbose init lines followed by ERROR lines at the very end
    info_lines = [f"INFO    Initializing tool_{i} from /cvmfs/data_{i}.root" for i in range(500)]
    error_lines = [
        "NTupleMaker   ERROR   Failed to call setupTree",
        "EventLoop     ERROR   Failed to call processInputs",
        "abort EL_JOBID=0",
    ]
    long_log = "\n".join(info_lines + error_lines)
    excerpt = extract_log_excerpt(
        long_log, "payload.stdout",
        pilot_error_code=1305,
        pilot_error_diag="",
    )
    # ERROR lines at the end must be captured
    assert "NTupleMaker" in excerpt
    assert "abort EL_JOBID=0" in excerpt
    # Beginning of a very long log should not be present
    assert "tool_0" not in excerpt


# ---------------------------------------------------------------------------
# Integration tests: full tool.call() with HTTP mocked
# ---------------------------------------------------------------------------


def test_log_analysis_success_with_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful analysis: metadata fetched, log downloaded, failure classified."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_STAGEIN_TIMEOUT),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: _SAMPLE_PILOT_LOG,
    )
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6799893074}))
    res = _unpack(result)
    ev = res["evidence"]

    assert ev["job_id"] == 6799893074
    assert ev["failure_type"] == "stagein_timeout"
    assert ev["log_available"] is True
    assert ev["log_excerpt"] is not None
    assert ev["piloterrorcode"] == 1151
    assert "stagein_timeout" in res["text"]


def test_log_analysis_metadata_only_for_closed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """For non-failed jobs (closed/reassigned) no log is downloaded."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_REASSIGNED),
    )
    fetch_log_called = []

    def _no_log(*args, **kwargs):  # type: ignore[no-untyped-def]
        fetch_log_called.append(True)
        return None

    monkeypatch.setattr("askpanda_epic.log_analysis_impl._fetch_log_text", _no_log)

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6837798305}))
    res = _unpack(result)

    assert not fetch_log_called, "Log should not be fetched for non-failed jobs"
    assert res["evidence"]["failure_type"] == "reassigned_by_jedi"


def test_log_analysis_log_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When log download fails, analysis still succeeds using metadata only."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_STAGEIN_TIMEOUT),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: None,
    )
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6799893074}))
    res = _unpack(result)
    ev = res["evidence"]

    assert ev["log_available"] is False
    assert ev["log_excerpt"] is None
    # Failure type should still come from metadata
    assert ev["failure_type"] == "stagein_timeout"


def test_log_analysis_job_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When metadata fetch returns no job, not_found is set in evidence."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: {"job": None, "files": [], "dsfiles": []},
    )
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 9999}))
    res = _unpack(result)
    assert res["evidence"].get("not_found") is True


def test_log_analysis_metadata_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When metadata HTTP request fails, an error is returned."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: None,
    )
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 9999}))
    res = _unpack(result)
    assert "error" in res["evidence"]


def test_log_analysis_missing_job_id() -> None:
    """Missing job_id produces a validation error in evidence."""
    result = asyncio.run(panda_log_analysis_tool.call({}))
    assert _unpack(result)["evidence"]["error"] == "missing job_id"


def test_log_analysis_invalid_job_id() -> None:
    """Non-integer job_id produces a validation error in evidence."""
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": "not-a-number"}))
    assert "integer" in _unpack(result)["evidence"]["error"]


def test_log_analysis_invalid_arguments() -> None:
    """Non-dict arguments produce a validation error in evidence."""
    result = asyncio.run(panda_log_analysis_tool.call("bad"))  # type: ignore[arg-type]
    assert "dict" in _unpack(result)["evidence"]["error"]


def test_log_analysis_payload_error_uses_payload_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pilot error 1305 fetches payload.stdout then payload.stderr.

    Both files are attempted: stdout is the primary log, stderr is fetched
    additionally since some failures (tracebacks, segfaults) only appear there.
    The stderr content should be present in the log excerpt.
    """
    fetched_filenames: list[str] = []

    def _capture_log(job_id: int, filename: str, base_url: str, timeout: int) -> str:
        fetched_filenames.append(filename)
        if filename == "payload.stderr":
            return "Segmentation fault (core dumped)\n" * 5
        return "Traceback (most recent call last):\n  AthenaMP crash\n" * 50

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr("askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log)
    # setup.stdout confirmed zero-length in index → skipped by _file_is_nonempty;
    # payload files are non-empty so they are downloaded as expected.
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({
            "setup.stdout": 0,
            "payload.stdout": 50000,
            "payload.stderr": 200,
        }),
    )

    import json
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    assert fetched_filenames == ["payload.stdout", "payload.stderr"]
    evidence = json.loads(result[0]["text"])["evidence"]
    assert evidence.get("stderr_url") is not None
    assert "payload.stderr" in (evidence.get("log_excerpt") or "")


def test_strip_payload_noise_removes_ls_sections() -> None:
    """_strip_payload_noise removes PanDA pilot ls directory listings.

    The pilot appends ``=== ls in <dir> ===`` sections between the application
    error output and the result footer.  These must be stripped before the
    char-tail is taken so the character budget is not wasted on file listings.
    """
    noisy = (
        "NTupleMaker   ERROR   Failed to call setupTree\n"
        "abort EL_JOBID=0\n"
        "\n=== ls in run dir : ./ ===\n"
        "total 408\n"
        "drwx------ 4 user user 4096 Apr 17 13:41 somedir\n"
        "lrwxrwxrwx 1 user user   44 Apr 17 13:41 DAOD.root -> /srv/DAOD.root\n"
        "-rw------- 1 user user  760 Apr 17 13:42 tmp.stderr.uuid\n"
        "\n=== ls in work dir : /srv ===\n"
        "total 572\n"
        "-rw------- 1 user user 375451 Apr 17 13:42 payload.stdout\n"
        "\n==== Result ====\n"
        "ERROR: payload execution failed with 1\n"
    )
    cleaned = _strip_payload_noise(noisy)
    # Application errors and result footer must be preserved
    assert "NTupleMaker   ERROR" in cleaned
    assert "abort EL_JOBID=0" in cleaned
    assert "==== Result ====" in cleaned
    assert "ERROR: payload execution failed" in cleaned
    # ls boilerplate must be removed
    assert "=== ls in run dir" not in cleaned
    assert "=== ls in work dir" not in cleaned
    assert "lrwxrwxrwx" not in cleaned
    assert "drwx------" not in cleaned
    # Result should be significantly shorter
    assert len(cleaned) < len(noisy) * 0.6


def test_classify_failure_pilot_monitoring_error_getpwuid() -> None:
    """pilot_monitoring_error is returned for getpwuid UID-not-found errors.

    Regression test for job 7099503721: the pilot's CPU monitoring code
    raised a KeyError when getpwuid() could not resolve UID 6435 on the
    worker node.  This is a pilot infrastructure issue, not a user payload
    failure — it must NOT be classified as payload_error.
    """
    log_excerpt = (
        "2026-04-17 02:05:23,070 | WARNING  | Exception caught: "
        "'getpwuid(): uid not found: 6435'\n"
        "2026-04-17 02:05:23,077 | WARNING  | Traceback (most recent call last):\n"
        "  File \".../pilot/util/monitoring.py\", line 193, in set_cpu_consumption_time\n"
        "    cpuconsumptiontime = get_current_cpu_consumption_time(job.pid)\n"
        "  File \".../pilot/util/processes.py\", line 619, "
        "in get_current_cpu_consumption_time\n"
        "    ps_cache = get_ps_cache()\n"
        "  File \".../pilot/util/processes.py\", line 191, in get_ps_cache\n"
        "    _ps_cache = list_processes_and_threads()\n"
        "  File \".../pilot/util/psutils.py\", line 428, "
        "in list_processes_and_threads\n"
        "    current_user = getpass.getuser()\n"
        "KeyError: 'getpwuid(): uid not found: 6435'\n"
    )
    job = {
        **_SAMPLE_JOB_STAGEIN_TIMEOUT,
        "pandaid": 7099503721,
        "piloterrorcode": 1354,
        "piloterrordiag": "Exception caught: 'getpwuid(): uid not found: 6435'",
        "taskbuffererrorcode": 0,
        "taskbuffererrordiag": "",
        "exeerrorcode": 0,
        "exeerrordiag": "",
        "jobsubstatus": "",
        "commandtopilot": "",
    }
    result = classify_failure(job, log_excerpt)
    assert result == "pilot_monitoring_error", (
        f"Expected 'pilot_monitoring_error', got '{result}'. "
        "getpwuid UID errors are pilot infrastructure failures, not payload errors."
    )


def test_extract_log_excerpt_pilot_error_1354() -> None:
    """extract_log_excerpt captures the full traceback for pilot error code 1354.

    The context window must include both the WARNING anchor line and the
    traceback lines that follow it.  With standard preceding-context extraction
    the function would return at the WARNING line and miss the File/KeyError
    lines entirely.

    Since the traceback-first extractor was introduced this no longer depends on
    1354 appearing in _TRAILING_CONTEXT_CODES — the traceback is located by its
    own format.  A bounded trailing window of _TRACEBACK_TRAILING_LINES lines is
    deliberately included after the traceback (the pilot logs the resulting
    error code and state transition there), so the assertion below checks that
    the tail is *bounded*, not that it is absent.  The failure mode being
    guarded against is the excerpt degenerating into a tail extraction that
    contains no traceback at all.
    """
    from askpanda_epic.log_analysis_impl import (
        _CONTEXT_LINES,
        _TRACEBACK_TRAILING_LINES,
    )

    preamble = "INFO | some startup line\n" * 50
    traceback_block = (
        "2026-04-17 02:05:23,070 | WARNING  | Exception caught: "
        "'getpwuid(): uid not found: 6435'\n"
        "2026-04-17 02:05:23,077 | WARNING  | Traceback (most recent call last):\n"
        "  File \".../psutils.py\", line 428, in list_processes_and_threads\n"
        "    current_user = getpass.getuser()\n"
        "KeyError: 'getpwuid(): uid not found: 6435'\n"
        "\n"  # blank line terminates the traceback block — mirrors real pilot logs
    )
    # Append unrelated tail lines that would dominate if the anchor missed
    tail_line = "INFO | some unrelated pilot cleanup line"
    tail = (tail_line + "\n") * _CONTEXT_LINES
    log_text = preamble + traceback_block + tail

    excerpt = extract_log_excerpt(
        log_text,
        "pilotlog.txt",
        pilot_error_code=1354,
        pilot_error_diag="Exception caught: 'getpwuid(): uid not found: 6435'",
    )
    assert "getpwuid" in excerpt, "Excerpt must contain the getpwuid error line."
    assert "list_processes_and_threads" in excerpt, (
        "Excerpt must contain the traceback File line that follows the WARNING."
    )
    assert "KeyError" in excerpt, "Excerpt must contain the final KeyError line."

    # The traceback must come before the cleanup tail, i.e. the excerpt is
    # anchored on the traceback rather than being a tail extraction.
    assert excerpt.index("KeyError") < excerpt.index(tail_line), (
        "The traceback must precede the trailing context; if it does not, the "
        "excerpt degenerated into tail extraction."
    )
    # Only the bounded trailing window may bleed through, never all 40 lines.
    assert excerpt.count(tail_line) <= _TRACEBACK_TRAILING_LINES, (
        f"At most {_TRACEBACK_TRAILING_LINES} trailing lines may be included; "
        f"got {excerpt.count(tail_line)}."
    )


def test_extract_context_window_with_trailing() -> None:
    """_extract_context_window_with_trailing includes lines after the match."""
    from askpanda_epic.log_analysis_impl import _extract_context_window_with_trailing

    log = (
        "line 1\n"
        "line 2\n"
        "ANCHOR line\n"
        "line after 1\n"
        "line after 2\n"
        "\n"                       # blank line — should stop collection here
        "line after blank\n"
    )
    result = _extract_context_window_with_trailing(log, "ANCHOR", n_before=10, n_trailing=20)
    assert "ANCHOR line" in result
    assert "line after 1" in result
    assert "line after 2" in result
    # Blank line itself is included as the stop sentinel, content after it is not
    assert "line after blank" not in result


def test_extract_context_window_with_trailing_no_match() -> None:
    """Returns empty string when the pattern is not found."""
    from askpanda_epic.log_analysis_impl import _extract_context_window_with_trailing

    result = _extract_context_window_with_trailing(
        "INFO | nothing here\n", "MISSING", n_before=10, n_trailing=10
    )
    assert result == ""


def test_extract_context_window_with_trailing_respects_n_trailing() -> None:
    """Stops after n_trailing lines even without a blank line."""
    from askpanda_epic.log_analysis_impl import _extract_context_window_with_trailing

    log = "ANCHOR\n" + "".join(f"line {i}\n" for i in range(50))
    result = _extract_context_window_with_trailing(log, "ANCHOR", n_before=5, n_trailing=3)
    assert "line 0" in result
    assert "line 1" in result
    assert "line 2" in result
    assert "line 3" not in result


def test_get_definition() -> None:
    """get_definition returns required MCP fields."""
    d = panda_log_analysis_tool.get_definition()
    assert d["name"] == "panda_log_analysis"
    assert "job_id" in d["inputSchema"]["properties"]
    assert d["inputSchema"]["required"] == ["job_id"]
    assert d["inputSchema"]["additionalProperties"] is False


def test_the_definition_stays_profile_agnostic() -> None:
    """The ePIC copy must be advertised under every tool profile.

    The ATLAS monolith withdraws from the primitive surface because
    ``atlas.log.*`` replaces it there.  Nothing replaces this one — the
    primitives are ATLAS-only — so restricting it the same way would leave an
    ePIC server running under ``BAMBOO_TOOL_PROFILE=primitive`` with no log
    analysis at all.  The divergence lives in ``tests/plugin_mirror_spec.py``
    and is regenerated mechanically; this asserts the outcome.
    """
    assert "profiles" not in panda_log_analysis_tool.get_definition()


@pytest.mark.parametrize("profile", ["orchestrated", "primitive", "both"])
def test_the_description_never_names_atlas_primitives(
    profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ePIC alternative exists, so none may be advertised under any profile.

    Args:
        profile: Server profile to set before reading the definition.
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("BAMBOO_TOOL_PROFILE", profile)
    description = panda_log_analysis_tool.get_definition()["description"]
    assert "atlas.log." not in description
    assert "BigPanDA" not in description


# ---------------------------------------------------------------------------
# Unit tests: new helpers (_setup_log_has_error, _file_is_nonempty,
#             _fetch_file_index, classify_failure setup pattern)
# ---------------------------------------------------------------------------


def test_setup_log_has_error_matches_no_release() -> None:
    """_setup_log_has_error returns True for 'No matched release is found'."""
    from askpanda_epic.log_analysis_impl import _setup_log_has_error

    log = (
        "sourcing /srv/my_release_setup.sh\n"
        "AtlasSetup(WARNING): Deprecated tag ignored\n"
        "!!!ERROR!!! No matched release is found\n"
        "22:50:44 2026/04/19\n"
    )
    assert _setup_log_has_error(log) is True


def test_setup_log_has_error_matches_error_bang() -> None:
    """_setup_log_has_error returns True for bare !!!ERROR!!! line."""
    from askpanda_epic.log_analysis_impl import _setup_log_has_error

    assert _setup_log_has_error("some preamble\n!!!ERROR!!! something bad\n") is True


def test_setup_log_has_error_case_insensitive() -> None:
    """_setup_log_has_error is case-insensitive."""
    from askpanda_epic.log_analysis_impl import _setup_log_has_error

    assert _setup_log_has_error("No Matched Release Is Found\n") is True


def test_setup_log_has_error_clean_log_returns_false() -> None:
    """_setup_log_has_error returns False when setup completed without error."""
    from askpanda_epic.log_analysis_impl import _setup_log_has_error

    clean = (
        "Info: /cvmfs mounted\n"
        "sourcing /srv/my_release_setup.sh\n"
        "Athena release 23.0.32 set up successfully\n"
    )
    assert _setup_log_has_error(clean) is False


def test_file_is_nonempty_positive() -> None:
    """_file_is_nonempty returns True for a file with size > 0."""
    from askpanda_epic.log_analysis_impl import _file_is_nonempty

    index = {"setup.stdout": 4096, "payload.stdout": 0}
    assert _file_is_nonempty(index, "setup.stdout") is True


def test_file_is_nonempty_zero_size() -> None:
    """_file_is_nonempty returns False for a confirmed zero-length file."""
    from askpanda_epic.log_analysis_impl import _file_is_nonempty

    index = {"payload.stdout": 0, "payload.stderr": 0}
    assert _file_is_nonempty(index, "payload.stdout") is False
    assert _file_is_nonempty(index, "payload.stderr") is False


def test_file_is_nonempty_missing_from_index() -> None:
    """_file_is_nonempty returns True when the filename is absent from the index.

    A file absent from the listing may simply not have been flushed yet;
    we attempt the download and rely on the HTTP layer for a definitive answer.
    """
    from askpanda_epic.log_analysis_impl import _file_is_nonempty

    assert _file_is_nonempty({"pilotlog.txt": 1024}, "setup.stdout") is True


def test_file_is_nonempty_none_index_fail_open() -> None:
    """_file_is_nonempty returns True when the index is None (fail-open)."""
    from askpanda_epic.log_analysis_impl import _file_is_nonempty

    assert _file_is_nonempty(None, "payload.stdout") is True
    assert _file_is_nonempty(None, "setup.stdout") is True


def test_fetch_file_index_parses_list_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_fetch_file_index parses a JSON list of {name, size} dicts.

    ``cached_fetch_jsonish`` is patched inside the impl module so that the
    real parsing logic inside ``_fetch_file_index`` is exercised end-to-end.
    """
    from askpanda_epic.log_analysis_impl import _fetch_file_index

    fake_listing = [
        {"name": "setup.stdout", "size": 2048},
        {"name": "payload.stdout", "size": 0},
        {"name": "payload.stderr", "size": 0},
        {"name": "pilotlog.txt", "size": 98304},
    ]

    # _fetch_file_index calls cached_fetch_jsonish via a deferred import inside
    # the function body.  We patch it on the cache module that gets imported.
    import askpanda_epic._cache as cache_mod
    monkeypatch.setattr(
        cache_mod,
        "cached_fetch_jsonish",
        lambda url, timeout: (200, "application/json", "", fake_listing),
    )

    result = _fetch_file_index(1, "https://bigpanda.cern.ch", 30)
    assert result == {
        "setup.stdout": 2048,
        "payload.stdout": 0,
        "payload.stderr": 0,
        "pilotlog.txt": 98304,
    }


def test_classify_failure_setup_release_not_found() -> None:
    """setup_release_not_found is classified when setup log excerpt is used."""
    setup_excerpt = (
        "AtlasSetup(Warning): aarch64 not supported on x86_64\n"
        "!!!ERROR!!! No matched release is found\n"
        "22:50:44 2026/04/19\n"
    )
    job = {
        **_SAMPLE_JOB_PAYLOAD,
        "piloterrordiag": "General payload setup verification error",
    }
    result = classify_failure(job, setup_excerpt)
    assert result == "setup_release_not_found", (
        f"Expected 'setup_release_not_found', got '{result}'. "
        "The setup log excerpt should drive classification when the "
        "release is not found."
    )


# ---------------------------------------------------------------------------
# Integration tests: setup.stdout-first flow and zero-length guards
# ---------------------------------------------------------------------------

# Shared setup log fixtures
_SETUP_LOG_WITH_ERROR = (
    "Info: /cvmfs mounted\n"
    "sourcing /srv/my_release_setup.sh\n"
    "AtlasSetup(WARNING): Deprecated tag ignored\n"
    "AtlasSetup(Warning): aarch64 not supported on x86_64-like machine\n"
    "\t/cvmfs/atlas.cern.ch/repo/sw/software/23.0/.../aarch64-centos7-gcc11-opt\n"
    "!!!ERROR!!! No matched release is found\n"
    "22:50:44 2026/04/19\n"
)

_SETUP_LOG_CLEAN = (
    "Info: /cvmfs mounted\n"
    "sourcing /srv/my_release_setup.sh\n"
    "Athena release 23.0.32 set up successfully\n"
)


def _index_with_zero_payload() -> dict[str, int]:
    """Return a file index where setup.stdout has content but payload logs are empty."""
    return {
        "setup.stdout": len(_SETUP_LOG_WITH_ERROR),
        "setup.stderr": 0,
        "payload.stdout": 0,
        "payload.stderr": 0,
    }


def _listing_stub(
    sizes: dict[str, int] | None,
    modification: str = "2026-08-19 01:42:51",
) -> Callable[[int, str, int], list[dict[str, Any]] | None]:
    """Build a ``_fetch_file_listing`` replacement from a ``{name: size}`` map.

    The production seam returns normalised listing records rather than a size
    index, so tests that want to control which files appear non-empty supply
    the map they care about and let this helper expand it into root-level
    records.

    Args:
        sizes: Mapping of root-level filename to size in bytes, or ``None``
            to simulate an unavailable listing (the fail-open case).
        modification: UTC timestamp string applied to every record.

    Returns:
        A callable with the ``(job_id, base_url, timeout)`` signature of
        :func:`~askpanda_epic.log_analysis_impl._fetch_file_listing`.
    """
    def _stub(
        job_id: int, base_url: str, timeout: int,
    ) -> list[dict[str, Any]] | None:
        if sizes is None:
            return None
        return [
            {
                "relative_path": name,
                "name": name,
                "dirname": "",
                "size_bytes": size,
                "modification": modification,
            }
            for name, size in sizes.items()
        ]

    return _stub


def test_setup_log_checked_before_payload_for_code_1305(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup.stdout is the first file fetched for pilot error 1305.

    The fetch order must be setup.stdout → (conditionally) payload.stdout →
    payload.stderr.  setup.stdout must appear first in the fetched list.
    """
    fetched: list[str] = []

    def _capture_log(
        job_id: int, filename: str, base_url: str, timeout: int
    ) -> str | None:
        fetched.append(filename)
        if filename == "setup.stdout":
            return _SETUP_LOG_CLEAN  # no error → fall through to payload
        if filename == "payload.stdout":
            return "Some payload output\n"
        return None

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({
            "setup.stdout": 100, "payload.stdout": 100, "payload.stderr": 0
        }),
    )

    asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    assert fetched[0] == "setup.stdout", (
        f"First fetched file should be setup.stdout, got {fetched}"
    )


def test_setup_error_skips_payload_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When setup.stdout contains a fatal error, payload logs are never fetched.

    The pilot error 1305 job has a fatal 'No matched release is found' in
    setup.stdout.  payload.stdout and payload.stderr would be empty anyway;
    fetching them wastes a round-trip and must be suppressed.
    """
    fetched: list[str] = []

    def _capture_log(
        job_id: int, filename: str, base_url: str, timeout: int
    ) -> str | None:
        fetched.append(filename)
        if filename == "setup.stdout":
            return _SETUP_LOG_WITH_ERROR
        # payload logs should never be reached
        return None

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub(_index_with_zero_payload()),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    res = _unpack(result)
    ev = res["evidence"]

    assert "payload.stdout" not in fetched, (
        "payload.stdout must not be fetched when setup error found"
    )
    assert "payload.stderr" not in fetched, (
        "payload.stderr must not be fetched when setup error found"
    )
    assert ev["failure_type"] == "setup_release_not_found"
    assert ev["log_available"] is True
    assert ev["setup_log_url"] is not None
    assert "No matched release" in (ev["log_excerpt"] or "")


def test_setup_error_excerpt_used_as_primary_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When setup error found, the setup log excerpt appears in evidence.log_excerpt."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: (
            _SETUP_LOG_WITH_ERROR if filename == "setup.stdout" else None
        ),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub(_index_with_zero_payload()),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    ev = _unpack(result)["evidence"]

    assert ev["setup_log_excerpt"] is not None
    assert "No matched release" in ev["setup_log_excerpt"]
    # The primary log_excerpt must also contain the setup content
    assert "No matched release" in (ev["log_excerpt"] or "")


def test_zero_length_payload_files_not_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload.stdout and payload.stderr with size=0 in the index are never fetched.

    When the file index confirms both payload files are zero-length and
    setup.stdout has no error, no download attempt should be made for the
    zero-length files.
    """
    fetched: list[str] = []

    def _capture_log(
        job_id: int, filename: str, base_url: str, timeout: int
    ) -> str | None:
        fetched.append(filename)
        return _SETUP_LOG_CLEAN if filename == "setup.stdout" else None

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({
            "setup.stdout": len(_SETUP_LOG_CLEAN),
            "payload.stdout": 0,
            "payload.stderr": 0,
        }),
    )

    asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))

    assert "payload.stdout" not in fetched, (
        "payload.stdout confirmed zero-length; must not be downloaded"
    )
    assert "payload.stderr" not in fetched, (
        "payload.stderr confirmed zero-length; must not be downloaded"
    )


def test_setup_log_nonempty_no_error_falls_through_to_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If setup.stdout has no error, payload.stdout is still fetched (if non-empty).

    A clean setup.stdout should not suppress payload log downloads —
    the payload may still have failed for other reasons.
    """
    fetched: list[str] = []

    def _capture_log(
        job_id: int, filename: str, base_url: str, timeout: int
    ) -> str | None:
        fetched.append(filename)
        if filename == "setup.stdout":
            return _SETUP_LOG_CLEAN
        if filename == "payload.stdout":
            return "AthenaMP ERROR: abort\n"
        return None

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({
            "setup.stdout": len(_SETUP_LOG_CLEAN),
            "payload.stdout": 100,
            "payload.stderr": 0,
        }),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    ev = _unpack(result)["evidence"]

    assert "payload.stdout" in fetched, (
        "payload.stdout must be fetched when setup.stdout is clean"
    )
    assert ev["log_available"] is True


def test_file_index_unavailable_falls_through_to_old_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the file index cannot be fetched, downloads proceed as before.

    _fetch_file_index returning None must not suppress any log downloads;
    the system falls back to attempting all files (fail-open).
    """
    fetched: list[str] = []

    def _capture_log(
        job_id: int, filename: str, base_url: str, timeout: int
    ) -> str | None:
        fetched.append(filename)
        if filename == "setup.stdout":
            return _SETUP_LOG_CLEAN
        if filename == "payload.stdout":
            return "Some payload output\n"
        return None

    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text", _capture_log
    )
    # Index unavailable — None → fail-open
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub(None),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    ev = _unpack(result)["evidence"]

    assert "setup.stdout" in fetched
    assert "payload.stdout" in fetched
    assert ev["log_available"] is True


def test_setup_log_url_in_links_md(monkeypatch: pytest.MonkeyPatch) -> None:
    """links_md includes the Setup Log URL when setup.stdout was fetched."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_PAYLOAD),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: (
            _SETUP_LOG_WITH_ERROR if filename == "setup.stdout" else None
        ),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub(_index_with_zero_payload()),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 1111}))
    ev = _unpack(result)["evidence"]

    links_md = ev.get("links_md", "")
    assert "Setup Log" in links_md, (
        f"links_md must contain 'Setup Log' when setup.stdout was fetched. Got:\n{links_md}"
    )
    assert "setup.stdout" in links_md


def test_setup_log_url_absent_when_not_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup_log_url is None in evidence when no setup log was fetched.

    For non-1305 errors (pilotlog.txt path), setup.stdout is never attempted
    and setup_log_url must be None.
    """
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(
            _SAMPLE_JOB_STAGEIN_TIMEOUT
        ),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: _SAMPLE_PILOT_LOG,
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({"pilotlog.txt": 500}),
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6799893074}))
    ev = _unpack(result)["evidence"]

    assert ev.get("setup_log_url") is None


def test_log_analysis_links_md_uses_panda_monitor_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """links_md uses 'PanDA Monitor' label (not 'BigPanDA Monitor') for ePIC."""
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: _make_metadata_response(_SAMPLE_JOB_STAGEIN_TIMEOUT),
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: _SAMPLE_PILOT_LOG,
    )
    monkeypatch.setattr(
        "askpanda_epic.log_analysis_impl._fetch_file_listing",
        _listing_stub({"pilotlog.txt": 500}),
    )
    result = asyncio.run(panda_log_analysis_tool.call({"job_id": 6799893074}))
    ev = _unpack(result)["evidence"]
    assert "PanDA Monitor" in ev.get("links_md", "")
    assert "BigPanDA Monitor" not in ev.get("links_md", "")
