"""Tests for the in-process TTL cache (askpanda_atlas._cache).

All tests are synchronous and require no network access.  The cache is
cleared before each test via the ``autouse`` fixture so tests are isolated.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests  # type: ignore[import]

from askpanda_atlas._cache import (  # type: ignore[import]
    METADATA_TTL,
    LOG_MISSING_TTL,
    LOG_TTL,
    PARTIAL_SUFFIX,
    _MISS,
    _get,
    _set,
    cached_fetch_jsonish,
    cached_fetch_log,
    clear,
    head_remote_file,
    invalidate,
    last_log_failure,
    stats,
    stream_to_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache() -> None:
    """Clear the cache before every test to ensure isolation."""
    clear()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_metadata_ttl_is_sixty_seconds() -> None:
    """METADATA_TTL is 60 seconds."""
    assert METADATA_TTL == 60.0


def test_log_ttl_is_infinite() -> None:
    """LOG_TTL is math.inf — logs never expire."""
    assert LOG_TTL == math.inf


# ---------------------------------------------------------------------------
# Internal primitives: _get / _set
# ---------------------------------------------------------------------------


def test_get_returns_miss_for_unknown_key() -> None:
    """_get returns the _MISS sentinel for a key that has never been set."""
    assert _get("https://example.com/never-set") is _MISS


def test_set_and_get_roundtrip() -> None:
    """A value stored with _set is retrievable via _get before expiry."""
    _set("key1", {"data": 42}, ttl=60.0)
    result = _get("key1")
    assert result == {"data": 42}


def test_expired_entry_returns_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry whose TTL has elapsed is evicted and returns _MISS."""
    _set("key2", "value", ttl=1.0)
    # Advance monotonic time past the TTL.
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 2.0)
    assert _get("key2") is _MISS


def test_infinite_ttl_never_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry stored with ttl=math.inf is still returned far in the future."""
    _set("log_key", "log text", ttl=math.inf)
    original = time.monotonic
    # 10 years in the future.
    monkeypatch.setattr(time, "monotonic", lambda: original() + 3.15e8)
    assert _get("log_key") == "log text"


def test_none_value_is_cached() -> None:
    """Storing None is valid — distinguishable from _MISS."""
    _set("key_none", None, ttl=60.0)
    result = _get("key_none")
    assert result is None
    assert result is not _MISS


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


def test_invalidate_removes_entry() -> None:
    """invalidate() evicts a specific key."""
    _set("to_remove", "data", ttl=60.0)
    invalidate("to_remove")
    assert _get("to_remove") is _MISS


def test_invalidate_nonexistent_key_is_safe() -> None:
    """invalidate() on a key that doesn't exist raises no error."""
    invalidate("https://example.com/ghost")  # must not raise


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_all_entries() -> None:
    """clear() evicts every cached entry."""
    _set("a", 1, ttl=60.0)
    _set("b", 2, ttl=60.0)
    _set("c", 3, ttl=math.inf)
    clear()
    assert _get("a") is _MISS
    assert _get("b") is _MISS
    assert _get("c") is _MISS


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_reflects_current_entries() -> None:
    """stats() returns the correct entry count and URL list."""
    _set("https://bigpanda.cern.ch/jobs/?jeditaskid=1&json", "x", ttl=60.0)
    _set("https://bigpanda.cern.ch/job?pandaid=2&json", "y", ttl=60.0)
    s = stats()
    assert s["entries"] == 2
    assert "https://bigpanda.cern.ch/jobs/?jeditaskid=1&json" in s["urls"]


def test_stats_counts_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """stats() correctly counts entries that have exceeded their TTL."""
    _set("exp1", "v", ttl=1.0)
    _set("exp2", "v", ttl=1.0)
    _set("live", "v", ttl=math.inf)
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 2.0)
    s = stats()
    assert s["expired"] == 2
    assert s["entries"] == 3  # not yet evicted, just counted as expired


def test_stats_empty_cache() -> None:
    """stats() on an empty cache returns zeros."""
    s = stats()
    assert s["entries"] == 0
    assert s["expired"] == 0
    assert s["urls"] == []


# ---------------------------------------------------------------------------
# cached_fetch_jsonish
# ---------------------------------------------------------------------------

_SAMPLE_RESPONSE: tuple[int, str, str, dict[str, Any]] = (
    200, "application/json", '{"jobs": []}', {"jobs": []}
)


def test_cached_fetch_jsonish_calls_real_on_miss() -> None:
    """On a cache miss, fetch_jsonish is called once and the result is stored."""
    url = "https://bigpanda.cern.ch/jobs/?jeditaskid=99&json"
    mock_fetch = MagicMock(return_value=_SAMPLE_RESPONSE)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        result = cached_fetch_jsonish(url)

    assert result == _SAMPLE_RESPONSE
    mock_fetch.assert_called_once_with(url, 30)


def test_cached_fetch_jsonish_returns_cached_on_hit() -> None:
    """On a cache hit, fetch_jsonish is NOT called again."""
    url = "https://bigpanda.cern.ch/jobs/?jeditaskid=100&json"
    mock_fetch = MagicMock(return_value=_SAMPLE_RESPONSE)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        first = cached_fetch_jsonish(url)
        second = cached_fetch_jsonish(url)

    assert first == second == _SAMPLE_RESPONSE
    mock_fetch.assert_called_once()   # only one real HTTP call


def test_cached_fetch_jsonish_separate_urls_are_independent() -> None:
    """Different URLs are cached independently."""
    url_a = "https://bigpanda.cern.ch/jobs/?jeditaskid=1&json"
    url_b = "https://bigpanda.cern.ch/jobs/?jeditaskid=2&json"
    resp_a = (200, "application/json", '{"jobs": [{"pandaid": 1}]}', {"jobs": [{"pandaid": 1}]})
    resp_b = (200, "application/json", '{"jobs": [{"pandaid": 2}]}', {"jobs": [{"pandaid": 2}]})

    def _side_effect(url: str, timeout: int) -> Any:
        return resp_a if "jeditaskid=1" in url else resp_b

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", side_effect=_side_effect):
        a = cached_fetch_jsonish(url_a)
        b = cached_fetch_jsonish(url_b)

    assert a[3] == {"jobs": [{"pandaid": 1}]}
    assert b[3] == {"jobs": [{"pandaid": 2}]}


def test_cached_fetch_jsonish_uses_metadata_ttl_by_default() -> None:
    """Default TTL is METADATA_TTL (60 s); the entry expires accordingly."""
    url = "https://bigpanda.cern.ch/jobs/?jeditaskid=77&json"
    mock_fetch = MagicMock(return_value=_SAMPLE_RESPONSE)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        cached_fetch_jsonish(url)   # populates cache

    # The entry should be present.
    assert _get(url) is not _MISS


def test_cached_fetch_jsonish_expired_entry_refetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired metadata entry triggers a fresh HTTP request."""
    url = "https://bigpanda.cern.ch/jobs/?jeditaskid=55&json"
    mock_fetch = MagicMock(return_value=_SAMPLE_RESPONSE)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        cached_fetch_jsonish(url)  # first call — populates cache

    # Advance time past METADATA_TTL.
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + METADATA_TTL + 1)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        cached_fetch_jsonish(url)  # second call — cache expired → re-fetch

    assert mock_fetch.call_count == 2


def test_cached_fetch_jsonish_infinite_ttl_never_refetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry stored with ttl=math.inf is never re-fetched."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=1&json&filename=pilotlog.txt"
    mock_fetch = MagicMock(return_value=_SAMPLE_RESPONSE)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        cached_fetch_jsonish(url, ttl=math.inf)

    # Advance time by 10 years.
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 3.15e8)

    with patch("askpanda_atlas._fallback_http.fetch_jsonish", mock_fetch):
        cached_fetch_jsonish(url, ttl=math.inf)

    mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# cached_fetch_log
# ---------------------------------------------------------------------------


def test_cached_fetch_log_calls_requests_on_miss() -> None:
    """On a cache miss, requests.get is called and the text is stored."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=42&json&filename=pilotlog.txt"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "=== PILOT LOG ==="
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = cached_fetch_log(url)

    assert result == "=== PILOT LOG ==="
    mock_get.assert_called_once()


def test_cached_fetch_log_returns_cached_on_hit() -> None:
    """On a cache hit, requests.get is NOT called again for a log."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=43&json&filename=pilotlog.txt"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "log line 1\nlog line 2"
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        first = cached_fetch_log(url)
        second = cached_fetch_log(url)

    assert first == second == "log line 1\nlog line 2"
    mock_get.assert_called_once()   # only one real HTTP download


def test_cached_fetch_log_returns_none_for_404() -> None:
    """A 404 response yields None, which is itself cached to avoid re-requests."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=44&json&filename=missing.txt"
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("requests.get", return_value=mock_resp) as mock_get:
        first = cached_fetch_log(url)
        second = cached_fetch_log(url)

    assert first is None
    assert second is None
    mock_get.assert_called_once()  # 404 result is cached — no second request


def test_cached_fetch_log_404_expires_and_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 is cached briefly, not forever.

    A job that has just failed may not have had its log tarball uploaded yet.
    Under the old infinite TTL the first premature look decided the answer for
    the lifetime of the process, so a log that appeared a minute later was
    never seen.

    Args:
        monkeypatch: Pytest fixture.
    """
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=45&json&filename=gone.txt"
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("requests.get", return_value=mock_resp) as first:
        cached_fetch_log(url)
        cached_fetch_log(url)
    first.assert_called_once()      # within the window, still cached

    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + LOG_MISSING_TTL + 1)

    with patch("requests.get", return_value=mock_resp) as second:
        assert cached_fetch_log(url) is None
    second.assert_called_once()     # window passed — asked again


def test_an_unauthorized_log_is_never_cached() -> None:
    """A 401 must not decide this URL's answer for the process lifetime.

    BigPanDA's filebrowser began requiring a token, and under the old policy
    every log URL attempted before the token was configured kept returning
    "no log" until the server was restarted — indistinguishable from a job
    that genuinely has none.
    """
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=47&json&filename=pilotlog.txt"
    denied = MagicMock()
    denied.status_code = 401
    denied.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized", response=denied
    )

    with patch("requests.get", return_value=denied) as calls:
        assert cached_fetch_log(url) is None
        assert cached_fetch_log(url) is None
    assert calls.call_count == 2     # retried, not pinned


def test_the_reason_for_a_failure_is_recoverable() -> None:
    """``last_log_failure`` explains an empty result without re-fetching."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=48&json&filename=pilotlog.txt"
    denied = MagicMock()
    denied.status_code = 401
    denied.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized", response=denied
    )

    with patch("requests.get", return_value=denied):
        assert cached_fetch_log(url) is None

    failure = last_log_failure(url)
    assert failure.status == 401
    assert "PANDA_MONITOR_TOKEN" in failure.reason


def test_a_success_clears_an_earlier_failure_reason() -> None:
    """Once the file is readable, the stale reason must not linger."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=49&json&filename=pilotlog.txt"
    denied = MagicMock()
    denied.status_code = 401
    denied.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized", response=denied
    )
    with patch("requests.get", return_value=denied):
        cached_fetch_log(url)
    assert last_log_failure(url).reason

    ok = MagicMock()
    ok.status_code = 200
    ok.text = "log content"
    ok.raise_for_status = MagicMock()
    with patch("requests.get", return_value=ok):
        assert cached_fetch_log(url) == "log content"

    assert last_log_failure(url).reason == ""


def test_no_token_configured_sends_no_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured deployment behaves exactly as it did before.

    Args:
        monkeypatch: Pytest fixture.
    """
    monkeypatch.delenv("PANDA_MONITOR_TOKEN", raising=False)
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=50&json&filename=pilotlog.txt"
    ok = MagicMock()
    ok.status_code = 200
    ok.text = "content"
    ok.raise_for_status = MagicMock()

    with patch("requests.get", return_value=ok) as call:
        cached_fetch_log(url)

    assert "Authorization" not in call.call_args.kwargs["headers"]


def test_a_configured_token_is_sent_as_a_token_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token reaches the filebrowser request.

    Args:
        monkeypatch: Pytest fixture.
    """
    monkeypatch.setenv("PANDA_MONITOR_TOKEN", "s3cr3t")
    monkeypatch.delenv("PANDA_MONITOR_TOKEN_SCHEME", raising=False)
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=51&json&filename=pilotlog.txt"
    ok = MagicMock()
    ok.status_code = 200
    ok.text = "content"
    ok.raise_for_status = MagicMock()

    with patch("requests.get", return_value=ok) as call:
        cached_fetch_log(url)

    assert call.call_args.kwargs["headers"]["Authorization"] == "Token s3cr3t"


def test_cached_fetch_log_after_clear_refetches() -> None:
    """After clear(), a log is re-downloaded on the next request."""
    url = "https://bigpanda.cern.ch/filebrowser/?pandaid=46&json&filename=pilotlog.txt"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "log content"
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        cached_fetch_log(url)   # populate
        clear()                  # flush everything
        cached_fetch_log(url)   # must re-fetch

    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Uncached binary media: head_remote_file
# ---------------------------------------------------------------------------


def _head_response(
    status: int = 200,
    content_length: str | None = "1065033128",
    content_type: str = "application/octet-stream",
    accept_ranges: str = "bytes",
    last_modified: str = "Wed, 19 Aug 2026 08:18:20 GMT",
) -> MagicMock:
    """Build a mock ``HEAD`` response.

    Args:
        status: HTTP status code.
        content_length: Raw ``Content-Length`` header, or ``None`` to omit.
        content_type: Raw ``Content-Type`` header.
        accept_ranges: Raw ``Accept-Ranges`` header.
        last_modified: Raw ``Last-Modified`` header.

    Returns:
        A mock response object.
    """
    headers = {
        "Content-Type": content_type,
        "Accept-Ranges": accept_ranges,
        "Last-Modified": last_modified,
    }
    if content_length is not None:
        headers["Content-Length"] = content_length
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = headers
    return mock_resp


def test_head_remote_file_parses_the_headers_it_needs() -> None:
    """The preflight surfaces size, range support and the modification time."""
    with patch("requests.head", return_value=_head_response()):
        info = head_remote_file("https://bigpanda.cern.ch/media/x/core.18277")

    assert info is not None
    assert info.ok
    assert info.content_length == 1065033128
    assert info.accept_ranges is True
    assert info.last_modified == "Wed, 19 Aug 2026 08:18:20 GMT"


def test_head_remote_file_flags_an_html_body_as_not_ok() -> None:
    """An HTML preflight is an authentication failure, not a success.

    The SSO-gated endpoint answers HTTP 200, so the status code alone cannot
    distinguish it from a real file.
    """
    response = _head_response(content_type="text/html; charset=utf-8")
    with patch("requests.head", return_value=response):
        info = head_remote_file("https://bigpanda.cern.ch/file/?lfn=x")

    assert info is not None
    assert info.is_html is True
    assert info.ok is False


def test_head_remote_file_reports_a_missing_content_length_as_none() -> None:
    """An absent Content-Length is None, not zero.

    Zero would mean "empty file" and would pass a disk preflight that should
    have been treated as unknown.
    """
    with patch("requests.head", return_value=_head_response(content_length=None)):
        info = head_remote_file("https://bigpanda.cern.ch/media/x/core.1")

    assert info is not None
    assert info.content_length is None


def test_head_remote_file_distinguishes_refusal_from_unreachable() -> None:
    """A 404 is reported through the object; a transport error returns None."""
    with patch("requests.head", return_value=_head_response(status=404)):
        refused = head_remote_file("https://bigpanda.cern.ch/media/x/gone")
    with patch("requests.head", side_effect=requests.ConnectionError("no route")):
        unreachable = head_remote_file("https://bigpanda.cern.ch/media/x/gone")

    assert refused is not None and refused.status_code == 404 and not refused.ok
    assert unreachable is None


def test_head_remote_file_is_not_cached() -> None:
    """The preflight leaves no entry in the store."""
    with patch("requests.head", return_value=_head_response()):
        head_remote_file("https://bigpanda.cern.ch/media/x/core.1")

    assert stats()["entries"] == 0


# ---------------------------------------------------------------------------
# Uncached binary media: stream_to_file
# ---------------------------------------------------------------------------


def _get_response(
    status: int = 200,
    content_type: str = "application/octet-stream",
    chunks: tuple[bytes, ...] = (b"abcd", b"efgh"),
) -> MagicMock:
    """Build a mock streaming ``GET`` response.

    Args:
        status: HTTP status code.
        content_type: Raw ``Content-Type`` header.
        chunks: Body chunks yielded by ``iter_content``.

    Returns:
        A mock response usable as a context manager.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = {"Content-Type": content_type}
    mock_resp.iter_content = MagicMock(return_value=iter(chunks))
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_stream_to_file_writes_and_publishes_the_file(tmp_path: Path) -> None:
    """A verified transfer lands at the destination path.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", return_value=_get_response()):
        result = stream_to_file("https://x/core.18277", dest, expected_bytes=8)

    assert result.ok
    assert result.bytes_written == 8
    assert dest.read_bytes() == b"abcdefgh"
    assert not dest.with_name(dest.name + PARTIAL_SUFFIX).exists()


def test_stream_to_file_rejects_an_html_body_before_writing(tmp_path: Path) -> None:
    """An HTML body is an SSO login page and never reaches the disk.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    response = _get_response(content_type="text/html; charset=utf-8")
    with patch("requests.get", return_value=response):
        result = stream_to_file("https://x/core.18277", dest)

    assert not result.ok
    assert "SSO" in result.error
    assert not dest.exists()
    assert not dest.with_name(dest.name + PARTIAL_SUFFIX).exists()


def test_stream_to_file_rejects_a_short_transfer(tmp_path: Path) -> None:
    """A transfer that ends cleanly but short is a failure.

    Without the size check it is indistinguishable from success, and the
    truncated file would be analysed as though it were complete.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", return_value=_get_response()):
        result = stream_to_file("https://x/core.18277", dest, expected_bytes=999)

    assert not result.ok
    assert "size mismatch" in result.error
    assert not dest.exists()


def test_stream_to_file_keeps_the_partial_file_for_a_retry(tmp_path: Path) -> None:
    """A failed transfer leaves its bytes behind rather than deleting them.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", return_value=_get_response()):
        stream_to_file("https://x/core.18277", dest, expected_bytes=999)

    assert dest.with_name(dest.name + PARTIAL_SUFFIX).read_bytes() == b"abcdefgh"


def test_stream_to_file_resumes_from_a_partial_download(tmp_path: Path) -> None:
    """A resumed transfer appends to the existing bytes and sends a Range header.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    dest.with_name(dest.name + PARTIAL_SUFFIX).write_bytes(b"abcd")
    response = _get_response(status=206, chunks=(b"efgh",))

    with patch("requests.get", return_value=response) as mock_get:
        result = stream_to_file(
            "https://x/core.18277", dest, expected_bytes=8, allow_resume=True,
        )

    assert result.ok and result.resumed
    assert result.bytes_written == 8
    assert dest.read_bytes() == b"abcdefgh"
    assert mock_get.call_args.kwargs["headers"]["Range"] == "bytes=4-"


def test_stream_to_file_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    """A server answering 200 to a Range request sent the whole body.

    Appending it to the carried-over prefix would silently produce a corrupt
    file of plausible length.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    dest.with_name(dest.name + PARTIAL_SUFFIX).write_bytes(b"abcd")
    response = _get_response(status=200, chunks=(b"abcd", b"efgh"))

    with patch("requests.get", return_value=response):
        result = stream_to_file(
            "https://x/core.18277", dest, expected_bytes=8, allow_resume=True,
        )

    assert result.ok and not result.resumed
    assert dest.read_bytes() == b"abcdefgh"


def test_stream_to_file_does_not_resume_a_complete_partial_file(tmp_path: Path) -> None:
    """A ``.part`` already at the expected size is not appended to.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    dest.with_name(dest.name + PARTIAL_SUFFIX).write_bytes(b"abcdefgh")
    response = _get_response(chunks=(b"abcd", b"efgh"))

    with patch("requests.get", return_value=response) as mock_get:
        result = stream_to_file(
            "https://x/core.18277", dest, expected_bytes=8, allow_resume=True,
        )

    assert result.ok and not result.resumed
    assert "Range" not in mock_get.call_args.kwargs["headers"]
    assert dest.read_bytes() == b"abcdefgh"


def test_stream_to_file_does_not_resume_without_a_known_size(tmp_path: Path) -> None:
    """Without an expected size a partial file cannot be told from a complete one.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    dest.with_name(dest.name + PARTIAL_SUFFIX).write_bytes(b"abcd")

    with patch("requests.get", return_value=_get_response()) as mock_get:
        result = stream_to_file("https://x/core.18277", dest, allow_resume=True)

    assert result.ok and not result.resumed
    assert "Range" not in mock_get.call_args.kwargs["headers"]


def test_stream_to_file_reports_a_404(tmp_path: Path) -> None:
    """A missing file is reported rather than written as an error page.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", return_value=_get_response(status=404)):
        result = stream_to_file("https://x/core.18277", dest)

    assert not result.ok
    assert "404" in result.error
    assert not dest.exists()


def test_stream_to_file_survives_a_transport_error(tmp_path: Path) -> None:
    """A dropped connection is reported, not raised.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", side_effect=requests.ConnectionError("reset")):
        result = stream_to_file("https://x/core.18277", dest)

    assert not result.ok
    assert result.status_code == 0
    assert not dest.exists()


def test_stream_to_file_never_caches(tmp_path: Path) -> None:
    """A core dump must never enter the infinite-TTL store.

    Routing one through the cached text path would decode a gigabyte of binary
    into a str and pin it for the lifetime of the process.

    Args:
        tmp_path: Temporary directory.
    """
    dest = tmp_path / "core.18277"
    with patch("requests.get", return_value=_get_response()):
        stream_to_file("https://x/core.18277", dest, expected_bytes=8)

    assert stats()["entries"] == 0
