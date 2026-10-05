"""An unreadable log must be reported as unreadable, not as absent.

Both log-analysis surfaces used to collapse every download failure into the
same answer: ``log_available: false`` and the note *"it may not exist"*. When
BigPanDA's filebrowser began requiring a token, every job in the system
started reporting that — and because ``classify_failure`` falls back to
``piloterrordiag``, the surrounding answer stayed fluent and plausible. A
stage-in failure diagnosed from metadata alone reads almost exactly like one
diagnosed from the log, so nothing surfaced the change.

These tests pin the distinction at both surfaces. The reason reaches the
caller, and a 404 and a 401 do not say the same thing: one means the job has
no log, the other means Bamboo was refused and the log exists but was never
opened.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests  # type: ignore[import]

from askpanda_atlas import log_analysis_impl as mono
from askpanda_atlas import log_primitives_impl as impl
from askpanda_atlas._cache import clear as cache_clear

_JOB_ID = 6799893074
_BASE_URL = "https://bigpanda.example.org"
_TIMEOUT = 60

_JOB: dict[str, Any] = {
    "pandaid": _JOB_ID,
    "jobstatus": "failed",
    "piloterrorcode": 1151,
    "piloterrordiag": "File transfer timed out during stage-in",
    "computingsite": "GRIF-LAL",
}


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    """Clear the HTTP cache around every test.

    Yields:
        None.
    """
    cache_clear()
    yield
    cache_clear()


def _denied_response() -> MagicMock:
    """Build a mock 401 response.

    Returns:
        A response whose ``raise_for_status`` raises, as requests does.
    """
    resp = MagicMock()
    resp.status_code = 401
    resp.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized", response=resp
    )
    return resp


def _missing_response() -> MagicMock:
    """Build a mock 404 response.

    Returns:
        A response reporting the file is absent.
    """
    resp = MagicMock()
    resp.status_code = 404
    return resp


def _wire_metadata(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Stub metadata and listing so only the download is exercised.

    Args:
        monkeypatch: Pytest fixture.
        module: The module whose helpers to patch.
    """
    monkeypatch.setattr(
        module, "_fetch_metadata",
        lambda job_id, base_url, timeout: {"job": dict(_JOB)},
    )
    monkeypatch.setattr(
        module, "_fetch_file_listing",
        lambda job_id, base_url, timeout: None,
    )


# ---------------------------------------------------------------------------
# The primitive surface
# ---------------------------------------------------------------------------

def test_a_refused_download_says_so_in_the_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fetch_text`` names the credential rather than guessing at absence.

    Args:
        monkeypatch: Pytest fixture.
    """
    _wire_metadata(monkeypatch, impl)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    with patch("requests.get", return_value=_denied_response()):
        _content, payload = asyncio.run(
            impl.fetch_text_tool.call({"job_id": _JOB_ID, "filename": "pilotlog.txt"})
        )

    assert payload["available"] is False
    note = " ".join(payload["notes"])
    assert "401" in note
    assert "PANDA_MONITOR_TOKEN" in note
    assert "may not exist" not in note


def test_an_absent_file_still_says_it_may_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 keeps the old wording, because the old wording is true for it.

    Args:
        monkeypatch: Pytest fixture.
    """
    _wire_metadata(monkeypatch, impl)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    with patch("requests.get", return_value=_missing_response()):
        _content, payload = asyncio.run(
            impl.fetch_text_tool.call({"job_id": _JOB_ID, "filename": "pilotlog.txt"})
        )

    assert payload["available"] is False
    note = " ".join(payload["notes"])
    assert "404" in note
    assert "PANDA_MONITOR_TOKEN" not in note


# ---------------------------------------------------------------------------
# The compound surface
# ---------------------------------------------------------------------------

def test_the_monolith_evidence_carries_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``panda_log_analysis`` reports why its evidence has no log.

    Without this the monitor renders a metadata-only diagnosis that is
    indistinguishable from a full one.

    Args:
        monkeypatch: Pytest fixture.
    """
    _wire_metadata(monkeypatch, mono)

    with patch("requests.get", return_value=_denied_response()):
        evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]

    assert evidence["log_available"] is False
    reason = str(evidence["log_unavailable_reason"])
    assert "401" in reason
    assert "PANDA_MONITOR_TOKEN" in reason


def test_a_readable_log_reports_no_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The field is absent-as-``None`` on the happy path, not an empty string.

    Args:
        monkeypatch: Pytest fixture.
    """
    _wire_metadata(monkeypatch, mono)
    ok = MagicMock()
    ok.status_code = 200
    ok.text = "pilot log content\n"
    ok.raise_for_status = MagicMock()

    with patch("requests.get", return_value=ok):
        evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]

    assert evidence["log_available"] is True
    assert evidence["log_unavailable_reason"] is None


def test_the_first_failure_wins_over_a_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the payload path the most diagnostic file's reason is the one kept.

    ``_fetch_into`` records the first failure rather than the last, so a
    ``payload.stderr`` that is merely absent cannot overwrite a ``401`` on the
    file that mattered.

    Args:
        monkeypatch: Pytest fixture.
    """
    result = mono._LogFetchResult()
    monkeypatch.setattr(
        mono, "_fetch_log_text", lambda job_id, filename, base_url, timeout: None
    )
    monkeypatch.setattr(
        mono, "_log_failure_reason",
        lambda job_id, filename, base_url: f"reason for {filename}",
    )

    mono._fetch_into(result, _JOB_ID, "payload.stdout", _BASE_URL, _TIMEOUT)
    mono._fetch_into(result, _JOB_ID, "payload.stderr", _BASE_URL, _TIMEOUT)

    assert result.log_unavailable_reason == "payload.stdout: reason for payload.stdout"


def test_the_reason_round_trips_through_the_state_codec() -> None:
    """``_LogFetchResult`` carries the reason across a serialisation boundary."""
    original = mono._LogFetchResult(log_unavailable_reason="pilotlog.txt: HTTP 401")
    restored = mono._LogFetchResult.from_state(original.to_state())

    assert restored.log_unavailable_reason == "pilotlog.txt: HTTP 401"
