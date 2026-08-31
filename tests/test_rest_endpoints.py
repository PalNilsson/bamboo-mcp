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

"""Tests for :mod:`bamboo.entrypoints.rest`.

Driven at the ASGI level — the handler is called with a scope, a receive
callable and a collecting send callable — so no server has to be started and
no port has to be bound.  ``bamboo_answer`` is stubbed throughout: what is
under test is the facade's contract with the monitor, not the analysis itself.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bamboo import analysis_store as store
from bamboo import cost_guard, session_scope
from bamboo.entrypoints import rest

_JOB: int = 7272161793


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path) -> Any:
    """Point every store at a temporary root and enable the API.

    Args:
        tmp_path: pytest-provided temporary directory.

    Yields:
        The temporary root.
    """
    env = {
        "BAMBOO_REST_ENABLED": "1",
        "BAMBOO_REST_STORE_ROOT": str(tmp_path / "rest"),
        "BAMBOO_COST_STATE_ROOT": str(tmp_path / "cost"),
        "BAMBOO_REST_INLINE_WAIT_S": "5",
    }
    with patch.dict(os.environ, env):
        session_scope.reset_all()
        yield tmp_path
        session_scope.reset_all()


class _Sent:
    """Collects an ASGI response.

    Attributes:
        status: HTTP status code.
        headers: Response headers.
        body: Raw response body.
    """

    def __init__(self) -> None:
        """Start with an empty response."""
        self.status: int = 0
        self.headers: list[tuple[bytes, bytes]] = []
        self.body: bytes = b""

    async def __call__(self, message: MutableMapping[str, Any]) -> None:
        """Accept one ASGI message.

        Args:
            message: The ASGI message.
        """
        if message["type"] == "http.response.start":
            self.status = int(message["status"])
            self.headers = list(message.get("headers") or [])
        elif message["type"] == "http.response.body":
            self.body += message.get("body") or b""

    def json(self) -> dict[str, Any]:
        """Parse the body as JSON.

        Returns:
            The parsed object.
        """
        return json.loads(self.body.decode("utf-8"))

    def header(self, name: bytes) -> str | None:
        """Return one response header.

        Args:
            name: Lowercase header name.

        Returns:
            The decoded value, or ``None``.
        """
        for key, value in self.headers:
            if key.lower() == name:
                return value.decode("utf-8")
        return None


def _receive_for(body: dict[str, Any] | None) -> Any:
    """Build an ASGI receive callable delivering *body* as JSON.

    Args:
        body: Object to encode, or ``None`` for an empty body.

    Returns:
        The receive callable.
    """
    payload = b"" if body is None else json.dumps(body).encode("utf-8")

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return _receive


async def _call(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    auth: Any = None,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> _Sent:
    """Invoke the REST handler once.

    Args:
        method: HTTP method.
        path: Request path.
        body: JSON body, or ``None``.
        auth: TokenAuth object, or ``None``.
        headers: Request headers.

    Returns:
        The collected response.
    """
    sent = _Sent()
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }
    await rest.handle(scope, _receive_for(body), sent, auth=auth)
    return sent


def _stub_answer(text: str = "The payload ran out of memory.", evidence: Any = None) -> Any:
    """Patch ``bamboo_answer`` with a stub that also writes evidence.

    Args:
        text: Answer text the stub returns.
        evidence: Evidence dict the stub records, or ``None`` for a default.

    Returns:
        A context manager patching the tool.
    """
    payload = evidence if evidence is not None else {"job_id": _JOB, "log_available": True}

    async def _call_stub(_arguments: dict[str, Any]) -> list[dict[str, Any]]:
        bucket = session_scope.bucket(session_scope.EVIDENCE_BUCKET)
        bucket["panda_log_analysis"] = {"evidence": payload, "text": text}
        bucket["last_tool"] = "panda_log_analysis"
        return [{"type": "text", "text": text}]

    from bamboo.tools.bamboo_answer import bamboo_answer_tool

    return patch.object(bamboo_answer_tool, "call", side_effect=_call_stub)


class _Auth:
    """Minimal stand-in for TokenAuth.

    Attributes:
        enabled: Whether tokens are enforced.
    """

    enabled = True

    def verify_bearer_token(self, header: str | None) -> str:
        """Accept exactly one token.

        Args:
            header: Authorization header value.

        Returns:
            The client id.

        Raises:
            TokenAuthError: On a missing or wrong token.
        """
        from bamboo.auth import TokenAuthError

        if not header:
            raise TokenAuthError("Missing Authorization header.")
        if header != "Bearer good-token":
            raise TokenAuthError("Invalid token.")
        return "panda-monitor"


class TestFeatureFlag:
    """Nothing is reachable until the operator opts in."""

    def test_disabled_by_default(self) -> None:
        """A deployment that does not set the flag is unchanged."""
        with patch.dict(os.environ, {"BAMBOO_REST_ENABLED": "0"}):
            sent = asyncio.run(_call("GET", "/api/v1/capabilities"))

        assert sent.status == 404
        assert sent.json()["error"]["code"] == "not_found"

    def test_enabled_flag_opens_the_api(self) -> None:
        """With the flag set the routes answer."""
        sent = asyncio.run(_call("GET", "/api/v1/capabilities"))
        assert sent.status == 200


class TestAuth:
    """The same policy as /mcp, enforced by the same helper."""

    def test_missing_token_is_401(self) -> None:
        """No credentials means unauthenticated, not forbidden."""
        sent = asyncio.run(_call("GET", "/api/v1/capabilities", auth=_Auth()))

        assert sent.status == 401
        assert sent.json()["error"]["code"] == "unauthorized"

    def test_wrong_token_is_403(self) -> None:
        """A presented-but-unknown token is forbidden."""
        headers = [(b"authorization", b"Bearer nope")]
        sent = asyncio.run(_call("GET", "/api/v1/capabilities", auth=_Auth(), headers=headers))

        assert sent.status == 403

    def test_valid_token_passes(self) -> None:
        """A known token reaches the route."""
        headers = [(b"authorization", b"Bearer good-token")]
        sent = asyncio.run(_call("GET", "/api/v1/capabilities", auth=_Auth(), headers=headers))

        assert sent.status == 200

    def test_no_auth_configured_allows(self) -> None:
        """A deployment without tokens is not accidentally locked out."""
        sent = asyncio.run(_call("GET", "/api/v1/capabilities", auth=None))
        assert sent.status == 200


class TestRouting:
    """Paths, methods, and malformed input."""

    def test_unknown_path_is_404(self) -> None:
        """An unmatched route says so rather than guessing."""
        sent = asyncio.run(_call("GET", "/api/v1/nope"))
        assert sent.status == 404

    def test_wrong_method_is_405(self) -> None:
        """Method mismatches are distinguishable from missing routes."""
        sent = asyncio.run(_call("GET", "/api/v1/analysis"))
        assert sent.status == 405

    def test_malformed_analysis_id_is_rejected(self) -> None:
        """Ids go into a filesystem path, so the route will not match junk."""
        sent = asyncio.run(_call("GET", "/api/v1/analysis/..%2F..%2Fetc"))
        assert sent.status == 404

    def test_missing_job_id_is_400(self) -> None:
        """A body without a job id is a client error with a reason."""
        sent = asyncio.run(_call("POST", "/api/v1/analysis", body={}))

        assert sent.status == 400
        assert "job_id" in sent.json()["error"]["message"]

    def test_non_numeric_job_id_is_400(self) -> None:
        """A job id must be an integer."""
        sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": "abc"}))
        assert sent.status == 400

    def test_unsupported_mode_is_400(self) -> None:
        """Core-dump analysis is deliberately not offered here."""
        body = {"job_id": _JOB, "mode": "core_dump"}
        sent = asyncio.run(_call("POST", "/api/v1/analysis", body=body))

        assert sent.status == 400
        assert "mode" in sent.json()["error"]["message"]

    def test_invalid_json_is_400(self) -> None:
        """A malformed body does not reach the analysis path."""

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"{{{", "more_body": False}

        async def _run() -> _Sent:
            sent = _Sent()
            scope = {"type": "http", "method": "POST", "path": "/api/v1/analysis", "headers": []}
            await rest.handle(scope, _receive, sent, auth=None)
            return sent

        assert asyncio.run(_run()).status == 400

    def test_oversized_body_is_400(self) -> None:
        """A large body is refused before it is parsed."""

        async def _receive() -> dict[str, Any]:
            return {
                "type": "http.request",
                "body": b"x" * (rest.MAX_BODY_BYTES + 1),
                "more_body": False,
            }

        async def _run() -> _Sent:
            sent = _Sent()
            scope = {"type": "http", "method": "POST", "path": "/api/v1/analysis", "headers": []}
            await rest.handle(scope, _receive, sent, auth=None)
            return sent

        assert asyncio.run(_run()).status == 400


class TestAnalysisLifecycle:
    """Start, poll, and finish."""

    def test_fast_analysis_completes_inline(self) -> None:
        """A quick answer arrives in one round trip."""
        with _stub_answer():
            sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        payload = sent.json()
        assert sent.status == 200
        assert payload["state"] == "complete"
        assert "out of memory" in payload["answer_markdown"]
        assert payload["evidence"]["job_id"] == _JOB

    def test_slow_analysis_returns_202_then_polls(self) -> None:
        """A long analysis hands back an id and finishes in the background."""

        async def _slow(_arguments: dict[str, Any]) -> list[dict[str, Any]]:
            await asyncio.sleep(0.3)
            return [{"type": "text", "text": "done eventually"}]

        from bamboo.tools.bamboo_answer import bamboo_answer_tool

        async def _run() -> tuple[_Sent, dict[str, Any]]:
            with patch.dict(os.environ, {"BAMBOO_REST_INLINE_WAIT_S": "0.05"}):
                with patch.object(bamboo_answer_tool, "call", side_effect=_slow):
                    started = _Sent()
                    scope = {
                        "type": "http",
                        "method": "POST",
                        "path": "/api/v1/analysis",
                        "headers": [],
                    }
                    await rest.handle(scope, _receive_for({"job_id": _JOB}), started, auth=None)
                    analysis_id = started.json()["analysis_id"]

                    await asyncio.sleep(0.5)

                    polled = _Sent()
                    poll_scope = {
                        "type": "http",
                        "method": "GET",
                        "path": f"/api/v1/analysis/{analysis_id}",
                        "headers": [],
                    }
                    await rest.handle(poll_scope, _receive_for(None), polled, auth=None)
                    return started, polled.json()

        started, final = asyncio.run(_run())

        assert started.status == 202
        assert started.json()["state"] in {"queued", "running"}
        assert started.json()["poll_after_s"] == rest.POLL_AFTER_S
        assert final["state"] == "complete"
        assert final["answer_markdown"] == "done eventually"

    def test_polling_an_unknown_id_is_404(self) -> None:
        """An id nobody issued is not found."""
        sent = asyncio.run(_call("GET", "/api/v1/analysis/" + "0" * 32))
        assert sent.status == 404

    def test_analysis_failure_is_reported_not_raised(self) -> None:
        """A crash inside the tool becomes a failed record, not a 500."""
        from bamboo.tools.bamboo_answer import bamboo_answer_tool

        async def _boom(_arguments: dict[str, Any]) -> list[dict[str, Any]]:
            raise RuntimeError("BigPanDA is down")

        with patch.object(bamboo_answer_tool, "call", side_effect=_boom):
            sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        payload = sent.json()
        assert sent.status == 200
        assert payload["state"] == "failed"
        assert "BigPanDA is down" in payload["error"]

    def test_evidence_comes_from_the_caller_session(self) -> None:
        """Evidence is read per session, not from whatever ran last."""
        session_scope.bucket(session_scope.EVIDENCE_BUCKET)["last_tool"] = "someone_elses_tool"
        session_scope.bucket(session_scope.EVIDENCE_BUCKET)["someone_elses_tool"] = {
            "evidence": {"job_id": 111}
        }

        with _stub_answer():
            sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        assert sent.json()["evidence"]["job_id"] == _JOB


class TestCaching:
    """A second click on the same job should be free."""

    def test_second_request_is_served_from_cache(self) -> None:
        """The tool is called once for two identical requests."""
        with _stub_answer() as stub:
            first = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))
            second = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert second.json()["answer_markdown"] == first.json()["answer_markdown"]
        assert stub.call_count == 1

    def test_a_different_job_is_not_cached(self) -> None:
        """The cache key includes the job."""
        with _stub_answer() as stub:
            asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))
            asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB + 1}))

        assert stub.call_count == 2

    def test_failures_are_retried_not_cached(self) -> None:
        """A transient failure does not poison the next click."""
        from bamboo.tools.bamboo_answer import bamboo_answer_tool

        async def _boom(_arguments: dict[str, Any]) -> list[dict[str, Any]]:
            raise RuntimeError("timeout")

        with patch.object(bamboo_answer_tool, "call", side_effect=_boom):
            asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        with _stub_answer() as stub:
            second = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        assert stub.call_count == 1
        assert second.json()["state"] == "complete"

    def test_no_log_answers_get_the_short_ttl(self, _isolated: Path) -> None:
        """A job still uploading is re-analysed soon rather than in a week."""
        evidence = {"job_id": _JOB, "log_available": False}
        with _stub_answer(text="No log yet.", evidence=evidence):
            asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        key = store.cache_key_for(_JOB, "failure", rest._active_model())
        pointer = json.loads(
            (Path(os.environ["BAMBOO_REST_STORE_ROOT"]) / "cache" / f"{key}.json").read_text(
                encoding="utf-8"
            )
        )
        import time

        assert pointer["expires_at"] - time.time() <= store.NO_LOG_CACHE_TTL_S


class TestSingleFlight:
    """Concurrent clicks on one job must not multiply the work."""

    def test_two_concurrent_requests_share_one_analysis(self) -> None:
        """The second caller is handed the first caller's id."""
        from bamboo.tools.bamboo_answer import bamboo_answer_tool

        async def _slow(_arguments: dict[str, Any]) -> list[dict[str, Any]]:
            await asyncio.sleep(0.3)
            return [{"type": "text", "text": "one answer"}]

        async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
            with patch.dict(os.environ, {"BAMBOO_REST_INLINE_WAIT_S": "0.05"}):
                with patch.object(bamboo_answer_tool, "call", side_effect=_slow) as stub:
                    first, second = await asyncio.gather(
                        _call("POST", "/api/v1/analysis", body={"job_id": _JOB}),
                        _call("POST", "/api/v1/analysis", body={"job_id": _JOB}),
                    )
                    await asyncio.sleep(0.5)
                    assert stub.call_count == 1
                    return first.json(), second.json()

        first_payload, second_payload = asyncio.run(_run())

        assert first_payload["analysis_id"] == second_payload["analysis_id"]


class TestBudget:
    """Refusal happens before any work starts."""

    def test_exhausted_budget_is_429_with_retry_after(self) -> None:
        """The monitor is told why and when to come back."""
        from bamboo.llm.types import TokenUsage

        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "0.000001"}):
            cost_guard.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=10_000, output_tokens=0)
            )
            with _stub_answer() as stub:
                sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        assert sent.status == 429
        assert sent.json()["error"]["code"] == "budget_exhausted"
        assert sent.header(b"retry-after") is not None
        assert stub.call_count == 0

    def test_budget_refusal_releases_the_claim(self) -> None:
        """A refused request must not lock the job out of later analysis."""
        from bamboo.llm.types import TokenUsage

        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "0.000001"}):
            cost_guard.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=10_000, output_tokens=0)
            )
            asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        with _stub_answer():
            sent = asyncio.run(_call("POST", "/api/v1/analysis", body={"job_id": _JOB}))

        assert sent.json()["state"] == "complete"


class TestRating:
    """Ratings must land on the right prompt-log document."""

    def _complete_with_promptlog(self) -> str:
        """Create a completed analysis carrying prompt-log coordinates.

        Returns:
            The analysis id.
        """
        key = store.cache_key_for(_JOB, "failure", "m")
        record = store.create(_JOB, "failure", key)
        store.mark_complete(
            record,
            answer_markdown="answer",
            promptlog={"index": "bamboomcp-promptlog-2026.08.31", "doc_id": "doc-1"},
        )
        return record.analysis_id

    def test_rating_is_forwarded(self) -> None:
        """A valid rating reaches update_rating with the record's own ids."""
        analysis_id = self._complete_with_promptlog()

        with patch("bamboo.llm.prompt_log.update_rating", return_value={}) as update:
            sent = asyncio.run(
                _call("POST", f"/api/v1/analysis/{analysis_id}/rating", body={"rating": 4})
            )

        assert sent.status == 204
        update.assert_called_once_with("bamboomcp-promptlog-2026.08.31", "doc-1", 4)

    def test_out_of_range_rating_is_400(self) -> None:
        """The 1-5 scale is enforced at the edge."""
        analysis_id = self._complete_with_promptlog()
        sent = asyncio.run(
            _call("POST", f"/api/v1/analysis/{analysis_id}/rating", body={"rating": 9})
        )

        assert sent.status == 400

    def test_non_numeric_rating_is_400(self) -> None:
        """A non-integer rating is rejected before any lookup."""
        analysis_id = self._complete_with_promptlog()
        sent = asyncio.run(
            _call("POST", f"/api/v1/analysis/{analysis_id}/rating", body={"rating": "good"})
        )

        assert sent.status == 400

    def test_missing_promptlog_is_409(self) -> None:
        """With prompt logging off there is nothing to rate, and it says so."""
        key = store.cache_key_for(_JOB, "failure", "m")
        record = store.mark_complete(store.create(_JOB, "failure", key), answer_markdown="a")

        sent = asyncio.run(
            _call("POST", f"/api/v1/analysis/{record.analysis_id}/rating", body={"rating": 5})
        )

        assert sent.status == 409
        assert sent.json()["error"]["code"] == "no_promptlog"

    def test_rating_an_unknown_analysis_is_404(self) -> None:
        """An id nobody issued cannot be rated."""
        sent = asyncio.run(
            _call("POST", "/api/v1/analysis/" + "0" * 32 + "/rating", body={"rating": 3})
        )
        assert sent.status == 404

    def test_backend_failure_is_502(self) -> None:
        """An OpenSearch outage is reported as upstream, not client, error."""
        analysis_id = self._complete_with_promptlog()

        with patch("bamboo.llm.prompt_log.update_rating", side_effect=RuntimeError("no route")):
            sent = asyncio.run(
                _call("POST", f"/api/v1/analysis/{analysis_id}/rating", body={"rating": 2})
            )

        assert sent.status == 502


class TestCapabilities:
    """What the monitor needs to render its panel sensibly."""

    def test_reports_limits_and_budget(self) -> None:
        """Concurrency and spend are visible without reading the logs."""
        sent = asyncio.run(_call("GET", "/api/v1/capabilities"))
        payload = sent.json()

        assert payload["modes"] == ["failure"]
        assert "max_concurrency" in payload["limits"]
        assert "daily_usd" in payload["budget"]
        assert payload["poll_after_s"] == rest.POLL_AFTER_S
