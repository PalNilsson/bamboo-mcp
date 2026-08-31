r"""Fire-and-forget prompt/response logger for OpenSearch.

Every call to :func:`~bamboo.tools.bamboo_executor.call_llm` can be logged to
an OpenSearch index for observability and analysis.  Logging is **opt-in**:
the module is a no-op unless the environment variable
``BAMBOO_OPENSEARCH_PROMPTLOG`` is set to a non-empty value (used as the HTTP
Basic-auth password for the OpenSearch cluster).

Privacy / GDPR
--------------
All text content is passed through :func:`redact_names` **before** the
document is built.  The redactor replaces potential personal identifiers
(CERN/ATLAS usernames, real names, values of known PanDA name fields) with the
token ``user_<XXXXXXXX>`` where ``XXXXXXXX`` is an 8-character lowercase hex
CRC32 digest of the original identifier.  The same identifier always maps to
the same token, so log entries remain joinable without storing the raw name.

.. warning::
    CRC32 is a non-cryptographic checksum.  An attacker who possesses the
    full CERN username list (~10 k entries) could reverse the mapping by
    exhaustive lookup in under a second.  If the OpenSearch index is ever
    accessible outside the CERN network, upgrade the hash to HMAC-SHA256
    keyed by a secret stored in a new env var (e.g.
    ``BAMBOO_PROMPTLOG_HASH_KEY``).  The code is structured to make that a
    one-line change in :func:`_crc32_token`.

Environment variables
---------------------
``BAMBOO_OPENSEARCH_PROMPTLOG``
    HTTP Basic-auth **password** for the OpenSearch cluster.  **Must be set**
    for logging to activate.  When absent the module is entirely passive.

``BAMBOO_OPENSEARCH_PROMPTLOG_INDEX``
    Base index name.  Defaults to ``bamboomcp-promptlog``.  A date suffix
    ``-YYYY.MM.DD`` is appended automatically, giving daily rollover that
    matches the ``atlas_harvesterworkers-*`` convention.

``ASKPANDA_OPENSEARCH_HOST``
    Base URL of the OpenSearch cluster.
    Default: ``https://os-atlas.cern.ch/os``

``ASKPANDA_OPENSEARCH_USER``
    HTTP Basic-auth username.  Default: ``pilot-monitor-agent``

``ASKPANDA_OPENSEARCH_CA``
    Path to the CA certificate bundle.
    Default: ``/etc/pki/tls/certs/CERN-bundle.pem``

``ASKPANDA_OPENSEARCH_VERIFY_CERTS``
    Set to ``"false"`` to disable TLS certificate verification (local
    development without the CERN CA bundle).

Document schema
---------------
Each indexed document contains::

    {
        "@timestamp":    "2026-04-17T14:33:01.123456Z",
        "session_id":    "uuid4 — stable for the process lifetime",
        "turn_number":   1,
        "provider":      "gemini",
        "model":         "gemini-2.0-flash",
        "max_tokens":    2048,
        "system_prompt": "You are AskPanDA...",
        "user_prompt":   "User question:\njobs at BNL...\n\nEvidence:...",
        "response":      "There are 42 running jobs...",
        "tools_used":    ["cric_query"],
        "input_tokens":  null,    # int when provider returns usage, else null
        "output_tokens": null,
    }

Only the current turn is stored.  Chat history is intentionally excluded —
``session_id`` + ``turn_number`` let you reconstruct the full conversation
in order, without any redundancy.  ``turn_number`` is a 1-based integer
incremented once per ``log_prompt()`` call within the process lifetime.
"""
from __future__ import annotations

import asyncio
import binascii
import logging
import os
import re
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from bamboo import session_scope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UI notification callback — optional bridge to TUI / Streamlit.
# ---------------------------------------------------------------------------

#: Signature: ``fn(severity: str, message: str) -> None``.
#: *severity* is one of ``"debug"``, ``"info"``, ``"warning"``, ``"error"``.
NotifyFn = Callable[[str, str], None]

#: Registered UI notification callback, or ``None`` when no UI is attached.
_notify_callback: NotifyFn | None = None


def register_notify_callback(fn: NotifyFn) -> None:
    """Register a callback that receives prompt-log status notifications.

    The callback is invoked synchronously inside the background thread that
    writes to OpenSearch, so it **must** be thread-safe.  For Textual UIs use
    ``app.call_from_thread``; for Streamlit append to ``st.session_state``.

    Only one callback is active at a time.  Calling this function a second
    time replaces the previous registration.

    Args:
        fn: Callable with signature ``(severity: str, message: str) -> None``
            where *severity* is ``"debug"``, ``"info"``, ``"warning"``, or
            ``"error"``.
    """
    global _notify_callback  # pylint: disable=global-statement
    _notify_callback = fn


def clear_notify_callback() -> None:
    """Remove the currently registered UI notification callback.

    Safe to call even when no callback is registered.
    """
    global _notify_callback  # pylint: disable=global-statement
    _notify_callback = None


def _notify(severity: str, message: str) -> None:
    """Invoke the registered UI notification callback if one is set.

    Exceptions raised by the callback are swallowed and logged at DEBUG level
    so a broken callback can never kill the prompt-logging background thread.

    Args:
        severity: One of ``"debug"``, ``"info"``, ``"warning"``, ``"error"``.
        message: Human-readable status message.
    """
    if _notify_callback is None:
        return
    try:
        _notify_callback(severity, message)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("prompt_log: notify callback raised %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Process-wide session ID — set once at import time.
#
# This is the right identifier under stdio, where one process serves one user.
# On the shared HTTP server it is not: every connected client would be indexed
# under the same session_id, which makes a conversation impossible to
# reconstruct from the index and distorts any per-session aggregation.
# _effective_session_id() therefore prefers the transport-bound session id
# when one is active, and falls back to this constant otherwise.
# ---------------------------------------------------------------------------

_SESSION_ID: str = str(uuid.uuid4())


def _effective_session_id() -> str:
    """Return the session id to record on prompt-log documents.

    Returns:
        The active Bamboo session id when a scope is bound, otherwise the
        process-wide :data:`_SESSION_ID`.
    """
    return session_scope.current_session_id() or _SESSION_ID


#: 1-based turn counter, incremented once per :func:`log_prompt` call.
#: Combined with ``session_id`` this gives a stable, human-readable reference
#: for any individual exchange (e.g. "session abc123, turn 3").
_turn_counter: int = 0

# ---------------------------------------------------------------------------
# Circuit breaker — disables logging after repeated write failures.
# ---------------------------------------------------------------------------

#: Number of consecutive write failures that trip the circuit breaker.
_CIRCUIT_BREAKER_THRESHOLD: int = 3

#: Consecutive failure counter.  Reset to zero on any successful write.
_consecutive_failures: int = 0

#: Set to True once the threshold is reached; cleared only on process restart.
_circuit_open: bool = False
#: Set to True after the index template has been applied once per process.
_template_applied: bool = False

# ---------------------------------------------------------------------------
# Per-turn event log — polled by the bamboo_promptlog_status MCP tool.
# ---------------------------------------------------------------------------

#: Ring buffer of the most recent prompt-log events.  Each entry is a dict
#: with keys ``"turn"``, ``"severity"``, and ``"message"``.  The buffer holds
#: at most 20 entries; oldest are discarded automatically.
_event_log: deque[dict[str, Any]] = deque(maxlen=20)

#: Stores the (index, doc_id) of the most recently indexed document so
#: rating tools can locate the document without a search query.
#:
#: Process-wide, and therefore only consulted when no session scope is active
#: (the stdio case).  Scoped callers read the per-session bucket instead — see
#: :func:`get_last_doc_id` — because a single-slot process-wide deque under
#: concurrency hands one user's rating to another user's document.
_last_doc_store: deque[tuple[str, str]] = deque(maxlen=1)

#: Key under which the last ``(index, doc_id)`` pair is stored inside a
#: session's prompt-log bucket.
_LAST_DOC_KEY: str = "last_doc"


def drain_events() -> list[dict[str, Any]]:
    """Return all buffered prompt-log events and clear the buffer.

    Called by ``BambooPromptLogStatusTool`` after each LLM response so the
    TUI and Streamlit interfaces can surface write confirmations and errors
    without requiring in-process callbacks (which do not work across the
    stdio subprocess boundary).

    Returns:
        List of event dicts, each with ``"turn"`` (int), ``"severity"``
        (``"info"`` / ``"warning"`` / ``"error"``), and ``"message"`` (str)
        keys.  Empty list when nothing has been logged since the last drain.
    """
    events = list(_event_log)
    _event_log.clear()
    return events


def record_last_doc(index: str, doc_id: str, session_id: str | None = None) -> None:
    """Record the coordinates of a freshly indexed document.

    Args:
        index: Index the document was written to.
        doc_id: OpenSearch document ``_id``.
        session_id: Session the document belongs to.  When ``None`` the pair
            goes to the process-wide store, matching stdio behaviour.  Callers
            running in a thread must pass this explicitly, since context
            variables do not cross a thread boundary.
    """
    if session_id is None:
        _last_doc_store.append((index, doc_id))
        return
    bucket = session_scope.bucket(
        session_scope.PROMPTLOG_BUCKET, session_id=session_id
    )
    bucket[_LAST_DOC_KEY] = (index, doc_id)


def get_last_doc_id() -> tuple[str, str] | None:
    """Return ``(index, doc_id)`` of the most recently indexed document.

    Used by rating tools to locate the correct document without a search.
    When a session scope is active the lookup is confined to that session, so
    a rating can never be applied to a document belonging to another client.

    Returns:
        Tuple of ``(index_name, doc_id)``, or ``None`` when nothing has been
        indexed for the caller's session.
    """
    session_id = session_scope.current_session_id()
    if session_id is not None:
        stored = session_scope.bucket(
            session_scope.PROMPTLOG_BUCKET, session_id=session_id
        ).get(_LAST_DOC_KEY)
        if isinstance(stored, tuple) and len(stored) == 2:
            return (str(stored[0]), str(stored[1]))
        return None
    return _last_doc_store[-1] if _last_doc_store else None


def update_rating(index: str, doc_id: str, rating: int) -> dict[str, Any]:
    """Update the ``rating`` field of an existing prompt-log document.

    Calls the OpenSearch ``update`` API with a partial document.  Uses
    the write credential (``BAMBOO_OPENSEARCH_PROMPTLOG``) since it
    modifies an existing document.

    Args:
        index: Index name, e.g. ``bamboomcp-promptlog-2026.05.26``.
        doc_id: OpenSearch document ``_id``.
        rating: Integer rating 1–5.

    Returns:
        The raw OpenSearch update response dict.

    Raises:
        ValueError: If *rating* is not in the range 1–5.
        RuntimeError: If ``BAMBOO_OPENSEARCH_PROMPTLOG`` is not set.
        ImportError: If ``opensearch-py`` is not installed.
    """
    if not 1 <= rating <= 5:
        raise ValueError(f"rating must be 1–5, got {rating!r}")
    client = _create_os_client()
    return client.update(
        index=index,
        id=doc_id,
        body={"doc": {"rating": rating}},
    )


# ---------------------------------------------------------------------------
# OpenSearch connection constants
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_BASE: str = "bamboomcp-promptlog"

# ---------------------------------------------------------------------------
# Redaction — privacy-preserving name pseudonymisation
# ---------------------------------------------------------------------------

# PanDA / BigPanDA JSON field names that are known to carry personal
# identifiers.  Values of these fields are *always* redacted regardless of
# their format.
_PANDA_NAME_FIELDS: frozenset[str] = frozenset({
    "prodUserName",
    "produsername",
    "username",
    "userName",
    "user_name",
    "owner",
    "submittedBy",
    "submitted_by",
    "createdBy",
    "created_by",
    "modifiedBy",
    "modified_by",
    "assignedTo",
    "assigned_to",
    "lockedBy",
    "locked_by",
    "requestedBy",
    "requested_by",
    "account",
    "dn",           # Distinguished Name — always personal
    "fullName",
    "full_name",
    "firstName",
    "first_name",
    "lastName",
    "last_name",
    "email",
    "mail",
})

# Regex: key-value pair where the key is a known PanDA name field.
# Uses \b word boundary (Python re does not support variable-width lookbehinds).
# Matches:  "prodUserName": "jsmith"
#           prodUserName: jsmith
_RE_PANDA_FIELD: re.Pattern[str] = re.compile(
    r'\b(' + "|".join(re.escape(f) for f in sorted(_PANDA_NAME_FIELDS)) + r')'
    r'("?\s*:\s*"?)([A-Za-z0-9._@/=-]{2,64})("?)',
    re.IGNORECASE,
)

# Capitalised word pairs: two consecutive title-case words not in the safe
# whitelist.  Run BEFORE contextual triggers so "John Smith" is matched as
# a unit rather than "John" being consumed alone by a trigger like "by John".
_RE_NAME_PAIR: re.Pattern[str] = re.compile(
    r'\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b',
)

# Contextual triggers: token that immediately follows a trigger phrase.
# The optional (?:\s+user)? handles "for user jsmith" as a single match.
_RE_CONTEXTUAL: re.Pattern[str] = re.compile(
    r'\b(?:user|for|by|owner|account|submitted\s+by|created\s+by|modified\s+by'
    r'|owned\s+by|assigned\s+to|locked\s+by)'
    r'(?:\s+user)?\s+([A-Za-z][A-Za-z0-9._-]{1,63})\b',
    re.IGNORECASE,
)

# Technical term pairs that look like title-case word pairs but are NOT names.
_SAFE_PAIRS: frozenset[str] = frozenset({
    # PanDA / ATLAS
    "Monte Carlo", "Big Panda", "BigPanda", "Grid Job", "Computing Site",
    "Task Status", "Job Status", "Error Code", "Pilot Error", "Queue Status",
    "Site Name", "Cloud Name", "Task Name", "Job Name", "Work Queue",
    "Input File", "Output File", "Log File", "Data Set", "Dataset Name",
    "Job Type", "Task Type", "Job Queue", "Job Retry",
    # Physics
    "Standard Model", "Higgs Boson", "Dark Matter", "Large Hadron",
    "Atlas Detector", "Inner Detector", "Liquid Argon",
    # General technical
    "True False", "None None",
})

# Tokens that must never be pseudonymised regardless of context.
# Note: "user" is intentionally absent — it is a contextual *trigger* word
# in _RE_CONTEXTUAL and must not be treated as a safe identifier value.
_SAFE_TOKENS: frozenset[str] = frozenset({
    # PanDA statuses
    "running", "finished", "failed", "pending", "activated", "submitted",
    "starting", "holding", "merging", "transferring", "cancelled", "broken",
    "aborted", "done", "online", "offline", "test", "brokeroff",
    # Technical tokens
    "true", "false", "null", "none", "error", "warning", "info",
    "atlas", "panda", "cern", "grid", "wlcg", "adcops",
    "mcore", "score", "managed",
    # Calendar words (avoid "submitted by April" style false positives)
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})


def _crc32_token(value: str) -> str:
    """Return a pseudonym token for *value* using CRC32.

    The same *value* always produces the same token, so pseudonymised log
    entries remain joinable.  The token format is ``user_XXXXXXXX`` where
    ``XXXXXXXX`` is the 8-character zero-padded lowercase hex CRC32.

    Args:
        value: The raw identifier to pseudonymise.

    Returns:
        Pseudonym string in the form ``user_XXXXXXXX``.
    """
    checksum = binascii.crc32(value.encode("utf-8")) & 0xFFFFFFFF
    return f"user_{checksum:08x}"


def redact_names(text: str) -> str:
    """Replace potential personal identifiers in *text* with CRC32 pseudonyms.

    Applies three redaction passes in order:

    1. **PanDA field values** — values of known name-carrying JSON fields
       (``prodUserName``, ``owner``, ``email``, etc.).
    2. **Capitalised word pairs** — two consecutive title-case words not in
       the technical-term whitelist (catches "John Smith" as a unit before
       any contextual pass can split it).
    3. **Contextual triggers** — tokens that immediately follow words such as
       ``"user"``, ``"for"``, ``"submitted by"``, ``"owned by"``, etc.
       Handles the pattern ``"for user jsmith"`` via an optional intermediate
       ``user`` token in the regex.

    Tokens present in ``_SAFE_TOKENS`` are never replaced.  Tokens already
    in ``user_[0-9a-f]{8}`` form are left unchanged.

    Args:
        text: Raw text string to redact (may be serialised JSON, a plain-text
              prompt, or a response string).

    Returns:
        Copy of *text* with personal identifiers replaced by ``user_XXXXXXXX``
        tokens.
    """
    if not text:
        return text

    # Pass 1: structured PanDA field values.
    def _replace_panda_field(m: re.Match[str]) -> str:
        field, sep, value, close = m.group(1), m.group(2), m.group(3), m.group(4)
        if value.lower() in _SAFE_TOKENS:
            return m.group(0)
        return f"{field}{sep}{_crc32_token(value)}{close}"

    text = _RE_PANDA_FIELD.sub(_replace_panda_field, text)

    # Pass 2: capitalised word pairs — run before contextual so "John Smith"
    # is matched as a whole before "by John" can consume "John" alone.
    def _replace_name_pair(m: re.Match[str]) -> str:
        first, last = m.group(1), m.group(2)
        pair = f"{first} {last}"
        if pair in _SAFE_PAIRS:
            return pair
        if first.lower() in _SAFE_TOKENS or last.lower() in _SAFE_TOKENS:
            return pair
        return f"{_crc32_token(first)} {_crc32_token(last)}"

    text = _RE_NAME_PAIR.sub(_replace_name_pair, text)

    # Pass 3: contextual triggers ("user jsmith", "for jsmith", …).
    def _replace_contextual(m: re.Match[str]) -> str:
        prefix = m.group(0)[: m.start(1) - m.start(0)]
        value = m.group(1)
        if value.lower() in _SAFE_TOKENS:
            return m.group(0)
        if re.fullmatch(r"user_[0-9a-f]{8}", value):
            return m.group(0)
        return prefix + _crc32_token(value)

    text = _RE_CONTEXTUAL.sub(_replace_contextual, text)

    return text

# ---------------------------------------------------------------------------
# OpenSearch helpers
# ---------------------------------------------------------------------------


def _is_logging_enabled() -> bool:
    """Return True when the prompt-log password env var is set.

    Returns:
        True if ``BAMBOO_OPENSEARCH_PROMPTLOG`` is set to a non-empty value.
    """
    return bool(os.environ.get("BAMBOO_OPENSEARCH_PROMPTLOG", ""))


def _build_index_name() -> str:
    """Return today's prompt-log index name with a UTC date suffix.

    Format: ``<base>-YYYY.MM.DD``, e.g. ``bamboomcp-promptlog-2026.04.17``.

    Returns:
        Index name string for today's UTC date.
    """
    base = os.environ.get("BAMBOO_OPENSEARCH_PROMPTLOG_INDEX", _DEFAULT_INDEX_BASE)
    today = datetime.now(tz=timezone.utc).strftime("%Y.%m.%d")
    return f"{base}-{today}"


def _create_os_client() -> Any:
    """Create an authenticated OpenSearch client for prompt logging.

    Delegates to :func:`bamboo.llm.opensearch_client.create_os_client` using
    the ``BAMBOO_OPENSEARCH_PROMPTLOG`` write password.  Kept as a module-level
    function so existing call sites and tests that patch it by name continue to
    work without modification.

    Returns:
        An :class:`opensearchpy.OpenSearch` client instance.

    Raises:
        ImportError: If ``opensearch-py`` is not installed.
        RuntimeError: If ``BAMBOO_OPENSEARCH_PROMPTLOG`` is not set.
    """
    from bamboo.llm.opensearch_client import create_os_client as _shared_factory

    password = os.environ.get("BAMBOO_OPENSEARCH_PROMPTLOG", "")
    if not password:
        raise RuntimeError(
            "BAMBOO_OPENSEARCH_PROMPTLOG is not set — prompt logging is disabled."
        )
    return _shared_factory(password)


_PROMPTLOG_TEMPLATE: dict = {
    "index_patterns": ["bamboomcp-promptlog-*"],
    "template": {
        "mappings": {
            "properties": {
                "@timestamp": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis",
                },
                "session_id": {"type": "keyword"},
                "turn_number": {"type": "integer"},
                "provider": {"type": "keyword"},
                "model": {"type": "keyword"},
                "max_tokens": {"type": "integer"},
                "tools_used": {"type": "keyword"},
                "input_tokens": {"type": "integer"},
                "output_tokens": {"type": "integer"},
                "system_prompt": {"type": "text"},
                "raw_question": {"type": "text"},
                "user_prompt": {"type": "text"},
                "response": {"type": "text"},
                "rating": {"type": "integer"},
            }
        }
    },
}
_PROMPTLOG_TEMPLATE_NAME: str = "bamboomcp-promptlog"


def _ensure_index_template(client: Any) -> None:
    """Apply the bamboomcp-promptlog index template if not yet applied.

    Idempotent — skipped after the first successful call per process (or
    after a permanent 403 permission error).

    A 403 AuthorizationException means the OpenSearch user lacks
    ``indices:admin/index_template/put`` permission.  Retrying would be
    pointless, so the flag is set to suppress further attempts and the
    failure is logged at INFO level rather than WARNING — it does not
    affect document writes, only date-range mapping quality.

    All other failures are logged at WARNING level and do not abort writes.

    Args:
        client: An authenticated :class:`opensearchpy.OpenSearch` client.
    """
    global _template_applied  # pylint: disable=global-statement
    if _template_applied:
        return
    try:
        client.indices.put_index_template(
            name=_PROMPTLOG_TEMPLATE_NAME,
            body=_PROMPTLOG_TEMPLATE,
        )
        _template_applied = True
        logger.debug(
            "prompt_log: index template '%s' applied", _PROMPTLOG_TEMPLATE_NAME
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        exc_str = str(exc)
        if "403" in exc_str or "AuthorizationException" in exc_str:
            # Permanent permission error — suppress future attempts and log
            # quietly; document writes are unaffected.
            _template_applied = True
            logger.info(
                "prompt_log: index template '%s' skipped (insufficient permissions) — "
                "date range queries may not work if mapping was auto-detected as text",
                _PROMPTLOG_TEMPLATE_NAME,
            )
        else:
            logger.warning(
                "prompt_log: failed to apply index template '%s': %s — "
                "date range queries may not work if mapping was auto-detected as text",
                _PROMPTLOG_TEMPLATE_NAME,
                exc,
            )


def _write_document(doc: dict[str, Any]) -> tuple[str, str] | None:
    """Write *doc* to OpenSearch synchronously.

    Intended to be called from a background thread via
    :func:`asyncio.to_thread` so it never blocks the event loop.

    The ``(index, doc_id)`` return value exists so the scheduling coroutine can
    attribute the document to the session that produced it.  This function
    cannot do that attribution itself: it runs in a worker thread, where the
    session context variable is not visible.

    Implements a simple circuit breaker: after
    :data:`_CIRCUIT_BREAKER_THRESHOLD` consecutive failures the circuit is
    opened and all subsequent calls return immediately with a single ``ERROR``
    log line.  The counter resets to zero on any successful write.

    Args:
        doc: Fully-built document dict to index.

    Returns:
        ``(index, doc_id)`` on a successful write, or ``None`` when the write
        was skipped or failed.
    """
    global _consecutive_failures, _circuit_open  # pylint: disable=global-statement

    if _circuit_open:
        return None

    index = _build_index_name()
    turn = doc.get("turn_number", "?")
    _notify("debug", f"prompt_log: sending turn {turn} to index '{index}'…")
    logger.debug("prompt_log: sending turn %s to index '%s'", turn, index)

    try:
        client = _create_os_client()
        _ensure_index_template(client)
        resp = client.index(index=index, body=doc)
        doc_id = resp.get("_id", "?") if isinstance(resp, dict) else "?"
        result = resp.get("result", "?") if isinstance(resp, dict) else "?"
        session_id = doc.get("session_id", "?")
        msg = (
            f"prompt_log: turn {turn} indexed — "
            f"index={index!r} id={doc_id!r} result={result!r} "
            f"session={session_id!r}"
        )
        logger.debug(msg)
        _notify("info", msg)
        _event_log.append({"turn": turn, "severity": "info", "message": msg})
        _last_doc_store.append((index, doc_id))
        # Success — reset failure counter.
        _consecutive_failures = 0
        return (str(index), str(doc_id))
    except ImportError:
        imp_msg = (
            f"prompt_log: turn {turn} — opensearch-py not installed; "
            f"install with: pip install opensearch-py"
        )
        logger.debug(imp_msg)
        _notify("warning", imp_msg)
        _event_log.append({"turn": turn, "severity": "warning", "message": imp_msg})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _consecutive_failures += 1
        if _consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            _circuit_open = True
            err_msg = (
                f"prompt_log: circuit breaker tripped after {_consecutive_failures} "
                f"consecutive write failures — prompt logging disabled for this session. "
                f"Check BAMBOO_OPENSEARCH_PROMPTLOG credentials and write "
                f"access to index '{index}'. Last error: {exc}"
            )
            logger.error(err_msg)
            _notify("error", err_msg)
            _event_log.append({"turn": turn, "severity": "error", "message": err_msg})
        else:
            warn_msg = (
                f"prompt_log: write failure {_consecutive_failures}/"
                f"{_CIRCUIT_BREAKER_THRESHOLD} — {exc}"
            )
            logger.warning(warn_msg)
            _notify("warning", warn_msg)
            _event_log.append({"turn": turn, "severity": "warning", "message": warn_msg})

    return None


async def _index_and_record(doc: dict[str, Any], session_id: str | None) -> None:
    """Index *doc* in a worker thread, then attribute it to *session_id*.

    The attribution has to happen here rather than inside
    :func:`_write_document`, because that function runs in a thread where the
    session context variable is invisible.  *session_id* is captured by the
    caller while still on the event loop and passed through explicitly.

    Args:
        doc: Fully-built document dict to index.
        session_id: Session that produced the document, or ``None`` when no
            scope was active.
    """
    coords = await asyncio.to_thread(_write_document, doc)
    if session_id is None:
        return
    if isinstance(coords, tuple) and len(coords) == 2:
        record_last_doc(str(coords[0]), str(coords[1]), session_id=session_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def log_prompt(
    system_prompt: str,
    user_prompt: str,
    response: str,
    tools_used: list[str],
    provider: str,
    model: str,
    max_tokens: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    raw_question: str | None = None,
) -> None:
    """Fire-and-forget: build a redacted document and ship it to OpenSearch.

    Returns immediately after scheduling the write as an
    :func:`asyncio.create_task`.  The main request pipeline is never blocked
    by the OpenSearch write.

    Only the current turn is stored — chat history is deliberately excluded.
    ``session_id`` + ``turn_number`` are sufficient to reconstruct a full
    conversation in order.

    If prompt logging is disabled (``BAMBOO_OPENSEARCH_PROMPTLOG`` not set)
    this function is a no-op and returns in microseconds.

    Args:
        system_prompt: The system prompt string for this call, before redaction.
        user_prompt: The synthesised user prompt for this call (contains the
            question and injected evidence), before redaction.
        raw_question: The user's original question as typed, before synthesis
            prompt construction.  Stored as a ``keyword`` field to enable
            accurate frequency aggregations (``raw_question.keyword``).
            When ``None`` the field is omitted from the document.
        response: Raw LLM response text, before redaction.
        tools_used: Names of the MCP tools called during this turn (e.g.
            ``["cric_query"]``).
        provider: LLM provider string (e.g. ``"gemini"``).
        model: LLM model string (e.g. ``"gemini-2.0-flash"``).
        max_tokens: ``max_tokens`` value passed to the LLM for this call.
        input_tokens: Input token count from the LLM usage object, or
            ``None`` when unavailable.
        output_tokens: Output token count from the LLM usage object, or
            ``None`` when unavailable.
    """
    if not _is_logging_enabled():
        return

    global _turn_counter  # pylint: disable=global-statement
    _turn_counter += 1
    turn_number = _turn_counter
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    session_id = session_scope.current_session_id()
    effective_session_id = _effective_session_id()

    doc: dict[str, Any] = {
        "@timestamp": timestamp,
        "session_id": effective_session_id,
        "turn_number": turn_number,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "system_prompt": redact_names(system_prompt),
        "user_prompt": redact_names(user_prompt),
        **({"raw_question": raw_question} if raw_question is not None else {}),
        "response": redact_names(response),
        "tools_used": tools_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    # asyncio.to_thread (inside _index_and_record): the synchronous OpenSearch
    # client never blocks the event loop.
    # create_task: caller gets control back immediately (fire-and-forget).
    asyncio.create_task(  # noqa: RUF006 — intentional fire-and-forget
        _index_and_record(doc, session_id),
        name=f"prompt_log_{effective_session_id[:8]}_{turn_number}",
    )


__all__ = [
    "log_prompt",
    "redact_names",
    "drain_events",
    "get_last_doc_id",
    "record_last_doc",
    "update_rating",
    "register_notify_callback",
    "clear_notify_callback",
    "NotifyFn",
    "_SESSION_ID",
    "_DEFAULT_INDEX_BASE",
    "_CIRCUIT_BREAKER_THRESHOLD",
    "_turn_counter",
]
