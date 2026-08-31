# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors
# - Paul Nilsson, paul.nilsson@cern.ch, 2026

"""REST facade for failure analysis, used by the PanDA monitor.

Why a REST surface exists alongside MCP
---------------------------------------
The "Analyze failure" button on a failed job page cannot speak MCP.  Streamable
HTTP means JSON-RPC plus a session handshake plus SSE plus a bearer token, and
putting that token in page JavaScript hands it to every monitor visitor.  So
the browser talks to the monitor's own backend, which talks to this facade with
a service token held server-side.

Four endpoints::

    POST /api/v1/analysis            {job_id, mode?}  -> 200 done | 202 running
    GET  /api/v1/analysis/{id}                        -> 200
    POST /api/v1/analysis/{id}/rating {rating: 1..5}  -> 204
    GET  /api/v1/capabilities                         -> 200

Everything is off unless ``BAMBOO_REST_ENABLED`` is set, so deploying this
changes nothing until somebody decides otherwise.

One question, one code path
---------------------------
The work is ``bamboo_answer`` asked "Analyze job N and explain the failure",
which is the same sentence a chat user would type and the same code path it
takes.  ``bypass_fast_path`` is pinned to ``False`` rather than left to the
environment, because ``BAMBOO_FAST_PATH`` is read only by the Streamlit and
Textual interfaces and a REST caller should not inherit an interface's
debugging switch.  With the fast path on, that phrasing routes deterministically
to ``panda_log_analysis``; ``tests/test_rest_routing.py`` pins that, so a change
to the routing heuristics breaks a test rather than the button.

Asynchronous by default
-----------------------
An analysis takes tens of seconds and nginx in front of the monitor typically
gives up at sixty, so the request returns an identifier to poll.  A short
inline wait (``BAMBOO_REST_INLINE_WAIT_S``) means the quick cases and every
cache hit still finish in one round trip.

Admission happens before work starts
------------------------------------
In order: cache, then single-flight claim, then budget, then a concurrency
slot.  Refusing early is the point — a 429 before anything runs is a clean
answer, whereas discovering the budget is gone halfway through an analysis
wastes the tokens it took to find out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Any

from bamboo import analysis_store as store
from bamboo import cost_guard, session_scope
from bamboo.auth import TokenAuthError
from bamboo.llm import prompt_log

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Path prefix this module owns.
API_PREFIX: str = "/api/v1"

#: Seconds to wait for an analysis before answering 202.
DEFAULT_INLINE_WAIT_S: float = 8.0

#: Seconds to wait for the fire-and-forget prompt-log write to report its
#: document id, so a rating has somewhere to land.
DEFAULT_PROMPTLOG_WAIT_S: float = 2.0

#: Largest request body accepted, in bytes.  Bodies here are a handful of
#: fields; anything larger is a mistake or an attack.
MAX_BODY_BYTES: int = 64 * 1024

#: Suggested client poll interval, echoed in every non-terminal response.
POLL_AFTER_S: int = 2

#: Analysis flavours this facade will start.  Core-dump analysis is
#: deliberately absent: it holds a single global slot and serialises, so it
#: cannot back a button that appears on every failed job page.
SUPPORTED_MODES: frozenset[str] = frozenset({"failure"})

_ANALYSIS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Strong references to running analysis tasks, so they are not garbage
#: collected mid-flight, plus the handle the inline wait needs.
_tasks: dict[str, asyncio.Task[None]] = {}

#: Bounds concurrent analyses and the queue waiting for one.
_limiter = cost_guard.ConcurrencyLimiter()


def rest_enabled() -> bool:
    """Report whether the REST surface is switched on.

    Returns:
        ``True`` when ``BAMBOO_REST_ENABLED`` is truthy.
    """
    return os.getenv("BAMBOO_REST_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def inline_wait_s() -> float:
    """Return how long a request waits before answering 202.

    Returns:
        ``$BAMBOO_REST_INLINE_WAIT_S``, or :data:`DEFAULT_INLINE_WAIT_S`.
    """
    raw = os.getenv("BAMBOO_REST_INLINE_WAIT_S", "").strip()
    if not raw:
        return DEFAULT_INLINE_WAIT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_INLINE_WAIT_S
    return value if value >= 0 else DEFAULT_INLINE_WAIT_S


def header_value(scope: Scope, name: bytes) -> str | None:
    """Return one header from an ASGI scope.

    Args:
        scope: ASGI scope.
        name: Lowercase header name as bytes.

    Returns:
        The decoded value, or ``None`` when absent or undecodable.
    """
    headers: Sequence[tuple[bytes, bytes]] = scope.get("headers") or []
    for key, value in headers:
        if key.lower() == name:
            try:
                return value.decode("utf-8").strip()
            except UnicodeDecodeError:  # pragma: no cover - malformed header
                return None
    return None


def authenticate(auth: Any, scope: Scope) -> str:
    """Verify the bearer token on a request.

    Shared with the ``/mcp`` route so both surfaces enforce one policy; a second
    implementation is a second thing to forget to update.

    Args:
        auth: The server's ``TokenAuth``, or ``None`` when unconfigured.
        scope: ASGI scope.

    Returns:
        The authenticated client id, or ``"auth-disabled"``.

    Raises:
        TokenAuthError: If the header is missing, malformed, or unknown.
    """
    if auth is None or not getattr(auth, "enabled", False):
        return "auth-disabled"
    return str(auth.verify_bearer_token(header_value(scope, b"authorization")))


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a JSON response.

    Args:
        send: ASGI send callable.
        status: HTTP status code.
        payload: JSON-serialisable body.
        extra_headers: Additional headers to include.
    """
    body = json.dumps(payload, default=str).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    headers.extend(extra_headers or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_error(
    send: Send,
    status: int,
    code: str,
    message: str,
    retry_after_s: float | None = None,
) -> None:
    """Send a structured error.

    A machine-readable ``code`` matters because the monitor renders a different
    panel for "budget spent" than for "job unknown", and matching on prose
    breaks the first time the wording is improved.

    Args:
        send: ASGI send callable.
        status: HTTP status code.
        code: Machine-readable error code.
        message: Human-readable explanation.
        retry_after_s: Suggested retry delay, echoed as a ``Retry-After``
            header when present.
    """
    extra: list[tuple[bytes, bytes]] = []
    if retry_after_s is not None:
        extra.append((b"retry-after", str(int(max(1, retry_after_s))).encode("ascii")))
    await _send_json(
        send,
        status,
        {"error": {"code": code, "message": message, "retry_after_s": retry_after_s}},
        extra_headers=extra,
    )


async def _read_body(receive: Receive) -> bytes:
    """Read a request body, refusing anything oversized.

    Args:
        receive: ASGI receive callable.

    Returns:
        The body bytes.

    Raises:
        ValueError: If the body exceeds :data:`MAX_BODY_BYTES`.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            break
        chunk = message.get("body") or b""
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        chunks.append(chunk)
        if not message.get("more_body"):
            break
    return b"".join(chunks)


async def _parse_json_body(receive: Receive) -> dict[str, Any]:
    """Read and parse a JSON request body.

    Args:
        receive: ASGI receive callable.

    Returns:
        The parsed object, or an empty dict for an empty body.

    Raises:
        ValueError: If the body is oversized or not a JSON object.
    """
    raw = await _read_body(receive)
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"body is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("body must be a JSON object")
    return parsed


def _response_for(record: store.AnalysisRecord) -> dict[str, Any]:
    """Build the response envelope for a record.

    The envelope is identical for POST and GET so the monitor has one shape to
    render, whether an answer arrived inline, from cache, or after polling.

    Args:
        record: The record to describe.

    Returns:
        JSON-serialisable response body.
    """
    return {
        "analysis_id": record.analysis_id,
        "job_id": record.job_id,
        "mode": record.mode,
        "state": record.state,
        "cached": record.cached,
        "elapsed_s": round(record.elapsed_s, 3),
        "poll_after_s": None if record.is_terminal() else POLL_AFTER_S,
        "answer_markdown": record.answer_markdown,
        "evidence": record.evidence,
        "promptlog": record.promptlog,
        "error": record.error,
    }


def _active_model() -> str:
    """Return the model string the analysis will use, for the cache key.

    Returns:
        The default profile's model, or ``"unknown"`` when the selector cannot
        be consulted.  ``"unknown"`` is deliberately a usable key rather than
        an error: a cache that degrades to per-process is better than a request
        that fails because a registry lookup moved.
    """
    try:
        from bamboo.llm.runtime import get_llm_selector

        selector = get_llm_selector()
        registry = getattr(selector, "registry", None)
        if registry is None:
            return "unknown"
        spec = registry.get(getattr(selector, "default_profile", "default"))
        return str(getattr(spec, "model", "unknown"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("rest: cannot determine active model: %s", exc)
        return "unknown"


def _extract_evidence() -> tuple[dict[str, Any] | None, bool]:
    """Read the evidence produced by the caller's own analysis.

    Read from the session bucket rather than the process-wide store, which is
    the whole reason session scoping landed first: under concurrency, "the last
    tool that ran" belongs to whoever ran it, not to this request.

    Returns:
        Tuple of the evidence dict (or ``None``) and whether the analysis found
        no log to read.
    """
    bucket = session_scope.bucket(session_scope.EVIDENCE_BUCKET)
    tool_name = bucket.get("last_tool")
    stored = bucket.get(str(tool_name), {}) if tool_name else {}
    if not isinstance(stored, dict):
        return None, False

    # The store holds {"evidence": {...}, "text": ...}; unwrap exactly one
    # layer, and tolerate a tool that stored the evidence dict directly.
    evidence = stored.get("evidence", stored)
    if not isinstance(evidence, dict):
        return None, False

    no_log = evidence.get("log_available") is False
    return evidence, bool(no_log)


async def _await_promptlog_coords() -> dict[str, str] | None:
    """Wait briefly for the prompt-log write to report its document id.

    The write is fire-and-forget by design, so the id is not available the
    instant the answer is.  Without it a rating has nowhere to land, and the
    process-wide "last document" fallback would attribute one client's rating
    to another client's turn.  A bounded wait is the compromise; returning
    ``None`` is an acceptable outcome, and the rating endpoint says so plainly.

    Returns:
        ``{"index": ..., "doc_id": ...}``, or ``None``.
    """
    if not prompt_log._is_logging_enabled():  # pylint: disable=protected-access
        return None

    deadline = asyncio.get_running_loop().time() + DEFAULT_PROMPTLOG_WAIT_S
    while asyncio.get_running_loop().time() < deadline:
        coords = prompt_log.get_last_doc_id()
        if coords is not None:
            return {"index": coords[0], "doc_id": coords[1]}
        await asyncio.sleep(0.1)
    return None


async def _run_analysis(record: store.AnalysisRecord) -> None:
    """Run one analysis to completion and persist the outcome.

    Args:
        record: The queued record to work on.
    """
    scope_id = f"rest:{record.analysis_id}"
    session_scope.set_session_id(scope_id)

    try:
        async with _limiter.slot():
            store.mark_running(record)
            await _execute(record)
    except cost_guard.AdmissionRefused as exc:
        store.mark_failed(record, str(exc))
    except asyncio.CancelledError:
        store.mark_failed(record, "The analysis was cancelled during shutdown.")
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("rest: analysis %s failed", record.analysis_id)
        store.mark_failed(record, f"{type(exc).__name__}: {exc}")
    finally:
        store.release_claim(record.cache_key, record.analysis_id)
        session_scope.clear_session(scope_id)


async def _execute(record: store.AnalysisRecord) -> None:
    """Ask the question and record the answer.

    Args:
        record: The running record.
    """
    from bamboo.tools.bamboo_answer import bamboo_answer_tool

    question = f"Analyze job {record.job_id} and explain the failure"
    result = await bamboo_answer_tool.call(
        {"question": question, "bypass_fast_path": False}
    )

    answer = ""
    if result and isinstance(result[0], dict):
        answer = str(result[0].get("text", ""))

    evidence, no_log = _extract_evidence()
    promptlog = await _await_promptlog_coords()

    store.mark_complete(
        record,
        answer_markdown=answer,
        evidence=evidence,
        promptlog=promptlog,
        no_log=no_log,
    )


def _start_analysis(record: store.AnalysisRecord) -> asyncio.Task[None]:
    """Schedule an analysis and keep a strong reference to its task.

    Args:
        record: The queued record.

    Returns:
        The scheduled task.
    """
    task = asyncio.create_task(_run_analysis(record), name=f"analysis-{record.analysis_id}")
    _tasks[record.analysis_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(record.analysis_id, None))
    return task


async def _wait_briefly(task: asyncio.Task[None]) -> None:
    """Give an analysis a chance to finish before answering 202.

    Args:
        task: The running analysis task.
    """
    wait = inline_wait_s()
    if wait <= 0:
        return
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=wait)


def _parse_job_id(body: dict[str, Any]) -> int:
    """Extract and validate the job id from a request body.

    Args:
        body: Parsed request body.

    Returns:
        The job id.

    Raises:
        ValueError: If the job id is missing or not a positive integer.
    """
    raw = body.get("job_id")
    if raw is None:
        raise ValueError("job_id is required")
    try:
        job_id = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"job_id must be an integer, got {raw!r}") from exc
    if job_id <= 0:
        raise ValueError("job_id must be positive")
    return job_id


async def _post_analysis(receive: Receive, send: Send, client_id: str) -> None:
    """Handle ``POST /api/v1/analysis``.

    Args:
        receive: ASGI receive callable.
        send: ASGI send callable.
        client_id: Authenticated client, recorded for attribution.
    """
    try:
        body = await _parse_json_body(receive)
        job_id = _parse_job_id(body)
    except ValueError as exc:
        await _send_error(send, 400, "invalid_request", str(exc))
        return

    mode = str(body.get("mode") or "failure")
    if mode not in SUPPORTED_MODES:
        await _send_error(
            send,
            400,
            "invalid_request",
            f"mode must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}",
        )
        return

    requested_by = str(body.get("user") or "") or client_id
    cache_key = store.cache_key_for(job_id, mode, _active_model())

    cached = store.lookup_cache(cache_key)
    if cached is not None:
        await _send_json(send, 200, _response_for(cached))
        return

    record = store.create(job_id, mode, cache_key, requested_by=requested_by)

    holder = store.try_claim(cache_key, record.analysis_id)
    if holder is not None:
        # Somebody else is already answering this exact question; hand back
        # their id so the client polls one analysis instead of starting a
        # second identical one.
        existing = store.load(holder)
        if existing is not None:
            await _send_json(send, 202, _response_for(existing))
            return
        store.try_claim(cache_key, record.analysis_id)

    budget = cost_guard.check_budget()
    if not budget.allowed:
        store.release_claim(cache_key, record.analysis_id)
        store.mark_failed(record, "Daily analysis budget is spent.")
        await _send_error(
            send,
            429,
            "budget_exhausted",
            f"Daily analysis budget of ${budget.budget_usd:.2f} is spent "
            f"(${budget.spent_usd:.2f} recorded).",
            retry_after_s=budget.retry_after_s,
        )
        return

    task = _start_analysis(record)
    await _wait_briefly(task)

    final = store.load(record.analysis_id) or record
    status = 200 if final.is_terminal() else 202
    await _send_json(send, status, _response_for(final))


async def _get_analysis(send: Send, analysis_id: str) -> None:
    """Handle ``GET /api/v1/analysis/{id}``.

    Args:
        send: ASGI send callable.
        analysis_id: Identifier from the path.
    """
    try:
        record = store.load(analysis_id)
    except ValueError:
        await _send_error(send, 400, "invalid_request", "malformed analysis id")
        return

    if record is None:
        await _send_error(send, 404, "not_found", f"no analysis {analysis_id}")
        return

    await _send_json(send, 200, _response_for(record))


async def _post_rating(receive: Receive, send: Send, analysis_id: str) -> None:
    """Handle ``POST /api/v1/analysis/{id}/rating``.

    Args:
        receive: ASGI receive callable.
        send: ASGI send callable.
        analysis_id: Identifier from the path.
    """
    try:
        body = await _parse_json_body(receive)
    except ValueError as exc:
        await _send_error(send, 400, "invalid_request", str(exc))
        return

    try:
        rating = int(body.get("rating"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        await _send_error(send, 400, "invalid_request", "rating must be an integer 1-5")
        return

    if not 1 <= rating <= 5:
        await _send_error(send, 400, "invalid_request", f"rating must be 1-5, got {rating}")
        return

    try:
        record = store.load(analysis_id)
    except ValueError:
        await _send_error(send, 400, "invalid_request", "malformed analysis id")
        return

    if record is None:
        await _send_error(send, 404, "not_found", f"no analysis {analysis_id}")
        return

    coords = record.promptlog or {}
    index, doc_id = coords.get("index"), coords.get("doc_id")
    if not index or not doc_id:
        await _send_error(
            send,
            409,
            "no_promptlog",
            "This analysis has no prompt-log document to rate; prompt logging "
            "was disabled or the write had not completed.",
        )
        return

    try:
        await asyncio.to_thread(prompt_log.update_rating, index, doc_id, rating)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("rest: rating %s failed: %s", analysis_id, exc)
        await _send_error(send, 502, "rating_failed", f"could not store the rating: {exc}")
        return

    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


async def _get_capabilities(send: Send) -> None:
    """Handle ``GET /api/v1/capabilities``.

    Args:
        send: ASGI send callable.
    """
    spend = cost_guard.current_spend()
    await _send_json(
        send,
        200,
        {
            "modes": sorted(SUPPORTED_MODES),
            "model": _active_model(),
            "plugin": os.getenv("ASKPANDA_PLUGIN", "atlas").strip().lower(),
            "inline_wait_s": inline_wait_s(),
            "poll_after_s": POLL_AFTER_S,
            "limits": {
                "max_concurrency": _limiter.max_concurrency,
                "max_queue": _limiter.max_queue,
                "in_flight": _limiter.in_flight,
            },
            "budget": {
                "daily_usd": cost_guard.daily_budget_usd(),
                "spent_usd": round(spend.usd, 4),
                "calls_today": spend.calls,
                "unpriced_calls_today": spend.unpriced_calls,
            },
        },
    )


def _match_analysis_path(path: str) -> tuple[str, str] | None:
    """Match a path against the analysis routes.

    Args:
        path: Request path with :data:`API_PREFIX` still attached.

    Returns:
        Tuple of route name and analysis id, or ``None`` when unmatched.
    """
    rest = path[len(API_PREFIX):].strip("/")
    parts = rest.split("/")

    if len(parts) == 2 and parts[0] == "analysis" and _ANALYSIS_ID_RE.match(parts[1]):
        return "detail", parts[1]
    if (
        len(parts) == 3
        and parts[0] == "analysis"
        and parts[2] == "rating"
        and _ANALYSIS_ID_RE.match(parts[1])
    ):
        return "rating", parts[1]
    return None


async def handle(scope: Scope, receive: Receive, send: Send, auth: Any = None) -> None:
    """Dispatch one request under :data:`API_PREFIX`.

    Args:
        scope: ASGI scope.
        receive: ASGI receive callable.
        send: ASGI send callable.
        auth: The server's ``TokenAuth``, or ``None``.
    """
    if not rest_enabled():
        await _send_error(
            send,
            404,
            "not_found",
            "The REST analysis API is disabled; set BAMBOO_REST_ENABLED=1 to enable it.",
        )
        return

    try:
        client_id = authenticate(auth, scope)
    except TokenAuthError as exc:
        message = str(exc)
        status = 403 if "Invalid token" in message else 401
        await _send_error(send, status, "unauthorized", message)
        return

    path = str(scope.get("path", ""))
    method = str(scope.get("method", "GET")).upper()

    if path.rstrip("/") == f"{API_PREFIX}/capabilities":
        if method != "GET":
            await _send_error(send, 405, "method_not_allowed", f"{method} not allowed here")
            return
        await _get_capabilities(send)
        return

    if path.rstrip("/") == f"{API_PREFIX}/analysis":
        if method != "POST":
            await _send_error(send, 405, "method_not_allowed", f"{method} not allowed here")
            return
        await _post_analysis(receive, send, client_id)
        return

    matched = _match_analysis_path(path)
    if matched is None:
        await _send_error(send, 404, "not_found", f"no route for {path}")
        return

    route, analysis_id = matched
    if route == "detail":
        if method != "GET":
            await _send_error(send, 405, "method_not_allowed", f"{method} not allowed here")
            return
        await _get_analysis(send, analysis_id)
        return

    if method != "POST":
        await _send_error(send, 405, "method_not_allowed", f"{method} not allowed here")
        return
    await _post_rating(receive, send, analysis_id)


__all__ = [
    "API_PREFIX",
    "DEFAULT_INLINE_WAIT_S",
    "MAX_BODY_BYTES",
    "SUPPORTED_MODES",
    "authenticate",
    "handle",
    "header_value",
    "inline_wait_s",
    "rest_enabled",
]
