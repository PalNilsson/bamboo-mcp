"""In-process TTL cache for BigPanDA HTTP responses.

Prevents redundant downloads within a session by caching the raw responses
returned by :func:`~askpanda_epic._fallback_http.fetch_jsonish` and log
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

Usage
-----
Replace direct calls to :func:`~askpanda_epic._fallback_http.fetch_jsonish`
and ``requests.get`` with the cached wrappers::

    from askpanda_epic._cache import cached_fetch_jsonish, cached_fetch_log

    # Metadata (60-second TTL)
    status, ctype, body, payload = cached_fetch_jsonish(url, timeout)

    # Log text (infinite TTL)
    text = cached_fetch_log(url, timeout)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_TTL: float = 60.0   # seconds — task and job metadata
#: User-Agent sent with every request originating from this module.
USER_AGENT: str = "AskPanDA/1.0"

LOG_TTL: float = math.inf
#: TTL for the recorded *reason* a download failed.  Only the reason is kept
#: this way, never the failed outcome itself: a caller that wants the file
#: still re-requests it, while a caller that wants to explain an empty result
#: can find out why without issuing a second request of its own.
LOG_REASON_TTL: float = 300.0

#: TTL for a log file the server answered ``404`` for.  Bounded rather than
#: infinite: a job that has just failed may not have had its log tarball
#: uploaded yet, and under :data:`LOG_TTL` the first premature look would
#: decide the answer for the lifetime of the process.
LOG_MISSING_TTL: float = 300.0     # logs are immutable once written

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

    from askpanda_epic._fallback_http import fetch_jsonish  # type: ignore[import]

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

    from askpanda_epic._fallback_http import (  # type: ignore[import]
        panda_monitor_headers,
    )

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, **panda_monitor_headers(url)},
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
