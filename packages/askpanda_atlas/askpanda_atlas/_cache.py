"""In-process TTL cache for BigPanDA HTTP responses.

Prevents redundant downloads within a session by caching the raw responses
returned by :func:`~askpanda_atlas._fallback_http.fetch_jsonish` and log
text returned by ``requests.get``.

TTL policy
----------
- Task and job **metadata** (``/jobs/``, ``/job?pandaid=``): 60 seconds.
  These may change while the process is running (jobs start, finish, fail),
  but polling more frequently than once per minute is wasteful.
- **Log files** (``/filebrowser/``): infinite TTL (``math.inf``).
  Once a pilot or payload log exists it is immutable.  There is never a
  reason to re-download it during the same process lifetime.

Thread safety
-------------
All cache operations are protected by a :class:`threading.Lock` so the
cache is safe for concurrent use from ``asyncio.to_thread`` workers.

Binary media files are deliberately **not** cached
--------------------------------------------------
:func:`head_remote_file` and :func:`stream_to_file` live here beside the
cached wrappers but share none of their machinery: they never read or write
the store.  A core dump is a gigabyte-scale binary, and routing one through
:func:`cached_fetch_log` would decode it into a ``str`` via ``resp.text``
and then pin it under :data:`LOG_TTL` (``math.inf``) for the lifetime of the
process.  Use :func:`stream_to_file` for anything that is not known-small
text.

Usage
-----
Replace direct calls to :func:`~askpanda_atlas._fallback_http.fetch_jsonish`
and ``requests.get`` with the cached wrappers::

    from askpanda_atlas._cache import cached_fetch_jsonish, cached_fetch_log

    # Metadata (60-second TTL)
    status, ctype, body, payload = cached_fetch_jsonish(url, timeout)

    # Log text (infinite TTL)
    text = cached_fetch_log(url, timeout)

    # Binary media, streamed straight to disk and never cached
    info = head_remote_file(url, timeout)
    result = stream_to_file(url, dest, timeout, expected_bytes=info.content_length)
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_TTL: float = 60.0   # seconds — task and job metadata
LOG_TTL: float = math.inf     # logs are immutable once written

#: TTL for the recorded *reason* a download failed.  Only the reason is kept
#: this way, never the failed outcome itself: a caller that wants the file
#: still re-requests it, while a caller that wants to explain an empty result
#: can find out why without issuing a second request of its own.
LOG_REASON_TTL: float = 300.0

#: TTL for a log file the server answered ``404`` for.  Bounded rather than
#: infinite: a job that has just failed may not have had its log tarball
#: uploaded yet, and under :data:`LOG_TTL` the first premature look would
#: decide the answer for the lifetime of the process.
LOG_MISSING_TTL: float = 300.0

#: User-Agent sent with every request originating from this module.
USER_AGENT: str = "AskPanDA/1.0"

#: Read size for streamed binary downloads.  One MiB keeps the number of
#: ``write`` calls low for a gigabyte-scale core without holding a large
#: buffer resident.
DEFAULT_BINARY_CHUNK_BYTES: int = 1024 * 1024

#: Default per-request deadline for binary media.  Matches the file-download
#: budget in the core-analysis worker rather than the 30 s used for metadata:
#: BigPanDA untars a job log server-side on first access, so the first media
#: request for a job can be far slower than the steady state.
DEFAULT_BINARY_TIMEOUT_S: float = 300.0

#: Suffix of the in-progress download.  The destination path is only created
#: by an atomic rename once every verification has passed, so a caller never
#: observes a truncated file at the real path — and a partial transfer stays
#: on disk under this suffix where :func:`stream_to_file` can resume it.
PARTIAL_SUFFIX: str = ".part"

#: A media endpoint answering with HTML is never returning file data.  The
#: SSO-gated BigPanDA download endpoint replies to an unauthenticated request
#: with an **HTTP 200** login page rather than a 401/403, so the status code
#: alone cannot distinguish it from success.
HTML_CONTENT_TYPES: frozenset[str] = frozenset(
    {"text/html", "application/xhtml+xml"}
)

# ---------------------------------------------------------------------------
# Internal store
# ---------------------------------------------------------------------------

_lock: threading.Lock = threading.Lock()

# key → (expiry_timestamp, value)
# expiry_timestamp == math.inf means the entry never expires.
_store: dict[str, tuple[float, Any]] = {}


# ---------------------------------------------------------------------------
# Core cache primitives
# ---------------------------------------------------------------------------


def _get(key: str) -> Any:
    """Return cached value for *key*, or ``_MISS`` if absent or expired.

    Args:
        key: Cache key (typically a URL string).

    Returns:
        Cached value, or the sentinel :data:`_MISS`.
    """
    with _lock:
        entry = _store.get(key)
    if entry is None:
        return _MISS
    expiry, value = entry
    if expiry != math.inf and time.monotonic() > expiry:
        with _lock:
            _store.pop(key, None)
        return _MISS
    return value


def _set(key: str, value: Any, ttl: float) -> None:
    """Store *value* under *key* with the given *ttl* in seconds.

    Args:
        key: Cache key.
        value: Value to store.
        ttl: Time-to-live in seconds.  ``math.inf`` means never expire.
    """
    expiry = math.inf if ttl == math.inf else time.monotonic() + ttl
    with _lock:
        _store[key] = (expiry, value)


class _MissType:
    """Sentinel singleton used to distinguish a missing cache entry from ``None``."""

    _instance: "_MissType | None" = None

    def __new__(cls) -> "_MissType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISS>"


_MISS = _MissType()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cached_fetch_jsonish(
    url: str,
    timeout: int = 30,
    ttl: float = METADATA_TTL,
) -> tuple[int, str, str, dict[str, Any] | None]:
    """Fetch a URL via fetch_jsonish, returning the cached result on repeat calls.

    On a cache hit, returns the previously fetched 4-tuple without making
    an HTTP request.  On a miss, delegates to the real ``fetch_jsonish``,
    stores the result, and returns it.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds (only used on a cache miss).
        ttl: Time-to-live in seconds for this entry.  Defaults to
            :data:`METADATA_TTL` (60 s).  Pass ``math.inf`` for
            responses that should never expire (e.g. log files served
            through this wrapper).

    Returns:
        4-tuple ``(status_code, content_type, body_text, parsed_json_or_none)``
        as returned by ``fetch_jsonish``.
    """
    cached = _get(url)
    if cached is not _MISS:
        return cached  # type: ignore[return-value]

    from askpanda_atlas._fallback_http import fetch_jsonish  # type: ignore[import]

    result = fetch_jsonish(url, timeout)
    status = result[0]
    if 200 <= status < 300:
        _set(url, result, ttl)
    else:
        # Never cache a failure here.  Callers may pass ``math.inf`` — the
        # file listing does — and a 401 or a 5xx pinned under an infinite TTL
        # keeps answering for the lifetime of the process long after the
        # cause is fixed.  Re-asking a failing endpoint is the cheaper
        # mistake.
        logger.debug("Not caching non-2xx response (HTTP %d) for %s", status, url)
    return result


@dataclass(frozen=True)
class LogFetch:
    """Outcome of one log-file download.

    ``cached_fetch_log`` used to collapse every failure into ``None``, so a
    missing file, an expired credential and a read timeout were
    indistinguishable to the caller — which is how a BigPanDA ``401`` spent an
    afternoon presenting as "the log may not exist".  The reason travels with
    the result now.

    Attributes:
        text: Log content, or ``None`` when it could not be read.
        status: HTTP status code, or ``None`` when the request never got one
            (a timeout, a DNS failure, a TLS error).
        reason: Short human-readable cause, empty on success.  Written for
            whoever reads it in an evidence bundle or a tool note, so it names
            the thing to go and check.
    """

    text: str | None
    status: int | None = None
    reason: str = ""


def _log_failure_reason(status: int | None, detail: str) -> str:
    """Describe a failed log download in one short phrase.

    Args:
        status: HTTP status code, or ``None`` when the request never
            completed.
        detail: Exception text, used only when there is no status.

    Returns:
        The reason string for :attr:`LogFetch.reason`.
    """
    if status == 404:
        return "not found (HTTP 404) — the log may not have been uploaded yet"
    if status in (401, 403):
        return (
            f"access denied (HTTP {status}) — BigPanDA requires a token; "
            f"check PANDA_MONITOR_TOKEN on the server"
        )
    if status is not None:
        return f"download failed (HTTP {status})"
    return f"download failed: {detail}"


def _reason_key(url: str) -> str:
    """Return the cache key under which a failure reason is recorded.

    Args:
        url: The log file's URL.

    Returns:
        The side-channel key, namespaced so it cannot collide with the URL.
    """
    return f"logreason:{url}"


def last_log_failure(url: str) -> LogFetch:
    """Return why the most recent download of *url* failed, without re-fetching.

    Reads the cache only.  This exists so a caller that obtained its text
    through the ordinary :func:`cached_fetch_log` path — and got ``None`` —
    can still explain itself, without that path having to carry a second
    return value through every intermediate function.

    Args:
        url: The log file's URL.

    Returns:
        The recorded :class:`LogFetch`, or an empty one when nothing is
        recorded.  An empty reason means "not known here", never "no failure".
    """
    recorded = _get(_reason_key(url))
    if isinstance(recorded, LogFetch):
        return recorded
    main = _get(url)
    if isinstance(main, LogFetch) and main.text is None:
        return main
    return LogFetch(None, None, "")


def cached_fetch_log_detailed(
    url: str,
    timeout: int = 60,
) -> LogFetch:
    """Fetch a log file, reporting why when it cannot be read.

    Caching is deliberately asymmetric, because the three outcomes have
    different lifetimes:

    - **Success** is cached under :data:`LOG_TTL` (``math.inf``).  A log file
      is immutable once written.
    - **404** is cached under :data:`LOG_MISSING_TTL`.  "Not uploaded yet" is
      a state that ends, and an infinite TTL would freeze a premature look
      into the answer for the whole process.
    - **Everything else is not cached at all.**  A 401, a 5xx and a timeout
      are all conditions that get fixed, and the old code pinned them under
      ``math.inf``: one expired token, or one network blip, and that URL
      returned "no log" until someone restarted the server — with no way to
      tell that from a job that genuinely has no log.

    Args:
        url: Full URL of the log file (filebrowser endpoint).
        timeout: HTTP timeout in seconds (only used on a cache miss).

    Returns:
        A :class:`LogFetch` carrying the text, or the status and reason.
    """
    cached = _get(url)
    if cached is not _MISS:
        return cached  # type: ignore[return-value]

    import requests  # type: ignore[import]

    from askpanda_atlas._fallback_http import (  # type: ignore[import]
        panda_monitor_headers,
    )

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, **panda_monitor_headers()},
            stream=True,
        )
        status: int | None = resp.status_code
        if status == 404:
            logger.info("Log file not found (404): %s", url)
            result = LogFetch(None, status, _log_failure_reason(status, ""))
            _set(url, result, LOG_MISSING_TTL)
            return result
        resp.raise_for_status()
        result = LogFetch(resp.text, status, "")
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = _log_failure_reason(status, f"{type(exc).__name__}: {exc}")
        logger.warning("Log download failed for %s: %s", url, exc)
        # The outcome is not cached — see the docstring; a transient failure
        # must not decide this URL's answer for the lifetime of the process.
        # The reason is, briefly, so a caller can say why it has no text.
        failure = LogFetch(None, status, reason)
        _set(_reason_key(url), failure, LOG_REASON_TTL)
        return failure

    _set(url, result, LOG_TTL)
    with _lock:
        _store.pop(_reason_key(url), None)
    return result


def cached_fetch_log(
    url: str,
    timeout: int = 60,
) -> str | None:
    """Fetch a log file, returning its text only.

    Retained for callers that have nothing to do with the reason.  New code
    should prefer :func:`cached_fetch_log_detailed`, which can say why an
    empty result is empty.

    Args:
        url: Full URL of the log file (filebrowser endpoint).
        timeout: HTTP timeout in seconds (only used on a cache miss).

    Returns:
        Log text as a string, or ``None`` if it could not be read.
    """
    return cached_fetch_log_detailed(url, timeout).text


# ---------------------------------------------------------------------------
# Uncached binary media access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteFileInfo:
    """Result of a ``HEAD`` preflight against a media URL.

    Attributes:
        url: The URL that was probed.
        status_code: HTTP status returned, or ``0`` when the request itself
            failed before a response was received.
        content_length: Parsed ``Content-Length``, or ``None`` when the header
            was absent or unparsable.
        content_type: Raw ``Content-Type`` header, ``""`` when absent.
        accept_ranges: ``True`` when the server advertises byte-range support,
            making a resumed download possible.
        last_modified: Raw ``Last-Modified`` header, ``""`` when absent.
    """

    url: str
    status_code: int
    content_length: int | None
    content_type: str
    accept_ranges: bool
    last_modified: str

    @property
    def ok(self) -> bool:
        """Return ``True`` for a 2xx response that is not an HTML page."""
        return 200 <= self.status_code < 300 and not self.is_html

    @property
    def is_html(self) -> bool:
        """Return ``True`` when the response body is an HTML document.

        Treated as an authentication failure by every caller: a media URL has
        no legitimate reason to serve HTML.
        """
        return _is_html_content_type(self.content_type)


@dataclass(frozen=True)
class BinaryFetchResult:
    """Outcome of a streamed binary download.

    Attributes:
        url: Source URL.
        path: Destination path.  Only exists on disk when *ok* is ``True``.
        ok: ``True`` when the file was fully transferred, verified and moved
            into place.
        bytes_written: Total bytes on disk for this transfer, including any
            bytes carried over from a resumed partial file.
        status_code: HTTP status of the response, or ``0`` on a transport
            failure.
        content_type: Raw ``Content-Type`` of the response.
        resumed: ``True`` when the transfer continued a pre-existing partial
            file rather than starting from byte zero.
        error: Human-readable failure reason, ``""`` on success.
    """

    url: str
    path: Path
    ok: bool
    bytes_written: int
    status_code: int
    content_type: str
    resumed: bool
    error: str


def _is_html_content_type(content_type: str) -> bool:
    """Return ``True`` when a ``Content-Type`` header denotes an HTML document.

    Args:
        content_type: Raw header value, possibly carrying a ``charset``
            parameter.

    Returns:
        ``True`` for an HTML media type.
    """
    return content_type.split(";")[0].strip().lower() in HTML_CONTENT_TYPES


def _parse_content_length(raw: str | None) -> int | None:
    """Parse a ``Content-Length`` header into a non-negative integer.

    Args:
        raw: Raw header value, or ``None`` when absent.

    Returns:
        The parsed length, or ``None`` when absent, unparsable or negative.
    """
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def head_remote_file(
    url: str,
    timeout: float = DEFAULT_BINARY_TIMEOUT_S,
) -> RemoteFileInfo | None:
    """Probe a media URL with ``HEAD`` without transferring the body.

    Used before a large download to obtain the authoritative size for a disk
    preflight, to detect byte-range support, and to catch an SSO redirect
    before a gigabyte of HTML login page is streamed to disk.

    The result is **not** cached.

    Args:
        url: Full media URL to probe.
        timeout: Request deadline in seconds.

    Returns:
        A :class:`RemoteFileInfo`, or ``None`` when the request could not be
        made at all.  A non-2xx response is reported through the returned
        object rather than as ``None``, so callers can distinguish "the server
        said no" from "the server could not be reached".
    """
    import requests  # type: ignore[import]

    from askpanda_atlas._fallback_http import (  # type: ignore[import]
        panda_monitor_headers,
    )

    try:
        resp = requests.head(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, **panda_monitor_headers()},
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        logger.warning("HEAD preflight failed for %s: %s", url, exc)
        return None

    accept_ranges = "bytes" in str(resp.headers.get("Accept-Ranges", "")).lower()
    return RemoteFileInfo(
        url=url,
        status_code=int(resp.status_code),
        content_length=_parse_content_length(resp.headers.get("Content-Length")),
        content_type=str(resp.headers.get("Content-Type", "")),
        accept_ranges=accept_ranges,
        last_modified=str(resp.headers.get("Last-Modified", "")),
    )


def _resume_offset(part: Path, expected_bytes: int | None, allow_resume: bool) -> int:
    """Return the byte offset a partial download can legitimately resume from.

    Resuming is only safe when the final size is known: without it there is no
    way to tell a partial file apart from a complete one, and appending to a
    complete file would silently corrupt it.

    Args:
        part: Path of the in-progress ``.part`` file.
        expected_bytes: Authoritative final size, or ``None`` when unknown.
        allow_resume: Whether the server advertised byte-range support.

    Returns:
        The offset to resume from, or ``0`` to start over.
    """
    if not allow_resume or expected_bytes is None or expected_bytes <= 0:
        return 0
    try:
        existing = part.stat().st_size
    except OSError:
        return 0
    return existing if 0 < existing < expected_bytes else 0


def _stream_rejection(status_code: int, content_type: str) -> str:
    """Return a failure reason for a response that must not be written to disk.

    Args:
        status_code: HTTP status of the response.
        content_type: Raw ``Content-Type`` header.

    Returns:
        A human-readable reason, or ``""`` when the response is usable.
    """
    if _is_html_content_type(content_type):
        return (
            f"server returned an HTML page (HTTP {status_code}) instead of file "
            "data — this endpoint requires CERN SSO authentication"
        )
    if status_code == 404:
        return "file not found (HTTP 404)"
    if not 200 <= status_code < 300:
        return f"HTTP {status_code}"
    return ""


def _write_stream(response: Any, part: Path, mode: str, chunk_bytes: int) -> int:
    """Write a streamed response body to *part*, returning the bytes written.

    Args:
        response: An open streaming ``requests`` response.
        part: Destination ``.part`` file.
        mode: ``"wb"`` to start over, ``"ab"`` to append to a resumed file.
        chunk_bytes: Read size.

    Returns:
        Number of bytes written by this call, excluding anything already
        present from a previous attempt.
    """
    written = 0
    with open(part, mode) as handle:
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
    return written


def _size_mismatch(total: int, expected_bytes: int | None) -> str:
    """Return a failure reason when the transferred size is not the expected one.

    Args:
        total: Bytes present in the ``.part`` file after the transfer.
        expected_bytes: Authoritative size, or ``None`` to skip the check.

    Returns:
        A human-readable reason, or ``""`` when the size is correct or
        unverifiable.
    """
    if expected_bytes is None or total == expected_bytes:
        return ""
    return (
        f"size mismatch: transferred {total} bytes but expected "
        f"{expected_bytes} — the partial file has been kept for a retry"
    )


def stream_to_file(
    url: str,
    dest: Path,
    timeout: float = DEFAULT_BINARY_TIMEOUT_S,
    expected_bytes: int | None = None,
    chunk_bytes: int = DEFAULT_BINARY_CHUNK_BYTES,
    allow_resume: bool = False,
) -> BinaryFetchResult:
    """Stream a binary media file to disk, verifying it before publishing it.

    Never caches.  Never holds the body in memory.  Three guards apply, in
    order, and each one is the response to an observed failure mode:

    1. **HTML rejection.**  A ``text/html`` body is an SSO login page served
       with HTTP 200, not file data, and is rejected before a single byte is
       written.
    2. **Size verification.**  The transferred length is checked against
       *expected_bytes* (normally the job listing's own ``size``).  A short
       transfer that ends cleanly is otherwise indistinguishable from success.
    3. **Atomic publication.**  Bytes land in ``<dest>.part`` and are renamed
       onto *dest* only once both guards pass, so *dest* never exists in a
       truncated state.

    On failure the ``.part`` file is deliberately left behind: it is the input
    to a resumed retry, and no tool in this package deletes files.

    Args:
        url: Full media URL.
        dest: Final destination path.  Its parent must already exist.
        timeout: Request deadline in seconds.
        expected_bytes: Authoritative size for verification, or ``None`` to
            accept whatever length arrives.
        chunk_bytes: Read size for the streaming loop.
        allow_resume: Permit continuing a previous partial transfer with a
            ``Range`` request.  Only meaningful when the server advertised
            ``Accept-Ranges: bytes`` and *expected_bytes* is known.

    Returns:
        A :class:`BinaryFetchResult`.  ``dest`` exists if and only if
        ``ok`` is ``True``.
    """
    import requests  # type: ignore[import]

    from askpanda_atlas._fallback_http import (  # type: ignore[import]
        panda_monitor_headers,
    )

    part = dest.with_name(dest.name + PARTIAL_SUFFIX)
    resume_from = _resume_offset(part, expected_bytes, allow_resume)
    # The media endpoint is SSO-gated and answers an unauthenticated request
    # with a 200 HTML login page, so it is a candidate for the same token.
    headers = {"User-Agent": USER_AGENT, **panda_monitor_headers()}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        resp = requests.get(url, timeout=timeout, headers=headers, stream=True)
    except requests.RequestException as exc:
        logger.warning("Binary fetch failed for %s: %s", url, exc)
        return BinaryFetchResult(
            url=url, path=dest, ok=False, bytes_written=0, status_code=0,
            content_type="", resumed=False, error=str(exc),
        )

    with resp:
        status_code = int(resp.status_code)
        content_type = str(resp.headers.get("Content-Type", ""))
        rejection = _stream_rejection(status_code, content_type)
        if rejection:
            logger.warning("Refusing to write %s: %s", url, rejection)
            return BinaryFetchResult(
                url=url, path=dest, ok=False, bytes_written=0,
                status_code=status_code, content_type=content_type,
                resumed=False, error=rejection,
            )

        # A server that ignores Range answers 200 with the whole body, so the
        # carried-over prefix must be discarded rather than appended to.
        resumed = resume_from > 0 and status_code == 206
        if resume_from and not resumed:
            logger.info("Range request ignored for %s; restarting transfer", url)
            resume_from = 0

        try:
            written = _write_stream(resp, part, "ab" if resumed else "wb", chunk_bytes)
        except (OSError, requests.RequestException) as exc:
            logger.warning("Binary fetch aborted for %s: %s", url, exc)
            return BinaryFetchResult(
                url=url, path=dest, ok=False, bytes_written=resume_from,
                status_code=status_code, content_type=content_type,
                resumed=resumed, error=str(exc),
            )

    total = resume_from + written
    mismatch = _size_mismatch(total, expected_bytes)
    if mismatch:
        logger.warning("Binary fetch rejected for %s: %s", url, mismatch)
        return BinaryFetchResult(
            url=url, path=dest, ok=False, bytes_written=total,
            status_code=status_code, content_type=content_type,
            resumed=resumed, error=mismatch,
        )

    os.replace(part, dest)
    return BinaryFetchResult(
        url=url, path=dest, ok=True, bytes_written=total,
        status_code=status_code, content_type=content_type,
        resumed=resumed, error="",
    )


def invalidate(url: str) -> None:
    """Remove a single URL from the cache.

    Useful in tests or when a caller knows a resource has changed.

    Args:
        url: URL key to evict.
    """
    with _lock:
        _store.pop(url, None)


def clear() -> None:
    """Evict all entries from the cache.

    Primarily intended for tests and the ``/clear`` TUI command.
    """
    with _lock:
        _store.clear()


def stats() -> dict[str, Any]:
    """Return a snapshot of cache statistics for diagnostics.

    Returns:
        Dict with ``"entries"`` (count), ``"urls"`` (sorted list of keys),
        and ``"expired"`` (count of entries past their TTL but not yet
        evicted).
    """
    now = time.monotonic()
    with _lock:
        items = list(_store.items())
    expired = sum(1 for _, (exp, _) in items if exp != math.inf and now > exp)
    return {
        "entries": len(items),
        "expired": expired,
        "urls": sorted(k for k, _ in items),
    }
