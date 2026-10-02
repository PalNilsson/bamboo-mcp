"""Tests for bamboo.llm.prompt_log.

Covers:
- _crc32_token: format, determinism, collision avoidance.
- redact_names: all three passes, safe tokens, safe pairs, no-double-redaction.
- redact_messages: list redaction, role preservation, mutation safety.
- _is_logging_enabled / _build_index_name: env-var behaviour.
- log_prompt: no-op when disabled, document shape + redaction when enabled,
              graceful handling of missing opensearch-py.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import bamboo.llm.prompt_log as _pl_module
from bamboo.llm.prompt_log import (
    _SESSION_ID,
    _build_index_name,
    _crc32_token,
    _is_logging_enabled,
    _DEFAULT_INDEX_BASE,
    _CIRCUIT_BREAKER_THRESHOLD,
    log_prompt,
    redact_names,
)


# ---------------------------------------------------------------------------
# Draining the fire-and-forget writes
# ---------------------------------------------------------------------------

async def _drain_writes() -> None:
    """Wait for every background prompt-log write to finish.

    ``log_prompt`` hands the write to ``asyncio.create_task`` and returns
    immediately, so awaiting it tells you nothing about whether the document
    reached ``_write_document``.  Sleeping a fixed interval instead makes the
    test a race against the thread pool: fine on an idle laptop, not on a
    loaded CI box, and invisible until it is not.

    Args:
        None.
    """
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)


# ---------------------------------------------------------------------------
# _crc32_token
# ---------------------------------------------------------------------------


class TestCrc32Token:
    """Tests for the CRC32 pseudonymisation helper."""

    def test_returns_user_prefix(self) -> None:
        """Token always starts with 'user_'."""
        assert _crc32_token("jsmith").startswith("user_")

    def test_returns_eight_hex_chars(self) -> None:
        """Token suffix is exactly 8 lowercase hex characters."""
        suffix = _crc32_token("jsmith")[len("user_"):]
        assert len(suffix) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", suffix)

    def test_deterministic(self) -> None:
        """Same input always produces the same token."""
        assert _crc32_token("atlas_user") == _crc32_token("atlas_user")

    def test_different_inputs_differ(self) -> None:
        """Different inputs produce different tokens."""
        assert _crc32_token("jsmith") != _crc32_token("jdoe")


# ---------------------------------------------------------------------------
# redact_names — Pass 1: PanDA structured fields
# ---------------------------------------------------------------------------


class TestRedactNamesPandaFields:
    """Pass 1: known PanDA name field values are always replaced."""

    def test_json_produsername(self) -> None:
        """prodUserName value is replaced; other fields are untouched."""
        text = '{"prodUserName": "jsmith", "status": "running"}'
        result = redact_names(text)
        assert "jsmith" not in result
        assert "running" in result
        assert "user_" in result

    def test_json_owner(self) -> None:
        """owner field value is replaced."""
        result = redact_names('{"owner": "atlas_jdoe"}')
        assert "atlas_jdoe" not in result
        assert "user_" in result

    def test_json_email(self) -> None:
        """email field value is replaced."""
        result = redact_names('"email": "jsmith@cern.ch"')
        assert "jsmith@cern.ch" not in result

    def test_created_by_field(self) -> None:
        """createdBy field value is replaced."""
        result = redact_names('"createdBy": "john_doe"')
        assert "john_doe" not in result

    def test_safe_token_as_field_value_not_replaced(self) -> None:
        """A safe token value (e.g. 'running') in a non-name field is kept."""
        result = redact_names('{"status": "running"}')
        assert "running" in result

    def test_same_name_produces_same_token(self) -> None:
        """The same name always maps to the same pseudonym."""
        t1 = redact_names('"prodUserName": "jsmith"')
        t2 = redact_names('"prodUserName": "jsmith"')
        assert t1 == t2


# ---------------------------------------------------------------------------
# redact_names — Pass 2: capitalised word pairs
# ---------------------------------------------------------------------------


class TestRedactNamesNamePairs:
    """Pass 2: two consecutive title-case words are treated as a name."""

    def test_first_last_name(self) -> None:
        """'John Smith' is fully pseudonymised."""
        result = redact_names("Submitted by John Smith")
        assert "John" not in result
        assert "Smith" not in result

    def test_safe_pair_monte_carlo(self) -> None:
        """'Monte Carlo' is whitelisted and not replaced."""
        assert "Monte Carlo" in redact_names("Monte Carlo simulation")

    def test_safe_pair_atlas_detector(self) -> None:
        """'Atlas Detector' is whitelisted and not replaced."""
        assert "Atlas Detector" in redact_names("Atlas Detector upgrade")

    def test_single_title_case_word_not_replaced(self) -> None:
        """A lone capitalised word is not pseudonymised."""
        result = redact_names("Task completed Monday")
        assert "Task" in result
        assert "Monday" in result


# ---------------------------------------------------------------------------
# redact_names — Pass 3: contextual triggers
# ---------------------------------------------------------------------------


class TestRedactNamesContextual:
    """Pass 3: identifiers following known trigger phrases are replaced."""

    def test_for_username(self) -> None:
        """'for jsmith' is redacted."""
        result = redact_names("show all tasks for jsmith please")
        assert "jsmith" not in result

    def test_for_user_username(self) -> None:
        """'for user jsmith' (two-word trigger) is fully redacted."""
        result = redact_names("show jobs for user jsmith")
        assert "jsmith" not in result

    def test_submitted_by(self) -> None:
        """'submitted by jdoe' is redacted."""
        result = redact_names("task submitted by jdoe yesterday")
        assert "jdoe" not in result

    def test_already_pseudonymised_not_doubled(self) -> None:
        """A token already in user_XXXXXXXX form is not re-pseudonymised."""
        pseudo = _crc32_token("jsmith")
        result = redact_names(f"user {pseudo}")
        assert pseudo in result

    def test_safe_token_after_trigger_not_replaced(self) -> None:
        """'user running' — 'running' is a safe token and survives."""
        assert "running" in redact_names("user running")


# ---------------------------------------------------------------------------
# redact_names — safe tokens
# ---------------------------------------------------------------------------


class TestRedactNamesSafeTokens:
    """PanDA status and technical tokens must survive all passes."""

    @pytest.mark.parametrize("token", [
        "running", "finished", "failed", "online", "offline",
        "mcore", "score", "managed", "atlas", "panda",
    ])
    def test_safe_token_survives(self, token: str) -> None:
        """Safe token is returned unchanged."""
        assert token in redact_names(token)


# ---------------------------------------------------------------------------
# _is_logging_enabled
# ---------------------------------------------------------------------------


class TestIsLoggingEnabled:
    """Logging gate controlled by BAMBOO_OPENSEARCH_PROMPTLOG."""

    def test_disabled_when_env_absent(self) -> None:
        """Returns False when env var is unset."""
        env = {k: v for k, v in os.environ.items()
               if k != "BAMBOO_OPENSEARCH_PROMPTLOG"}
        with patch.dict(os.environ, env, clear=True):
            assert _is_logging_enabled() is False

    def test_enabled_when_env_set(self) -> None:
        """Returns True when env var is non-empty."""
        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "secret"}):
            assert _is_logging_enabled() is True

    def test_disabled_when_env_empty_string(self) -> None:
        """Returns False when env var is set to empty string."""
        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": ""}):
            assert _is_logging_enabled() is False


# ---------------------------------------------------------------------------
# _build_index_name
# ---------------------------------------------------------------------------


class TestBuildIndexName:
    """Daily-rollover index name."""

    def test_contains_default_base(self) -> None:
        """Default base 'bamboomcp-promptlog' is present."""
        assert _DEFAULT_INDEX_BASE in _build_index_name()
        assert "bamboomcp-promptlog" in _build_index_name()

    def test_contains_date_suffix(self) -> None:
        """Name ends with YYYY.MM.DD."""
        assert re.search(r"\d{4}\.\d{2}\.\d{2}$", _build_index_name())

    def test_custom_base_from_env(self) -> None:
        """Custom base name is respected."""
        with patch.dict(os.environ,
                        {"BAMBOO_OPENSEARCH_PROMPTLOG_INDEX": "my-log"}):
            assert _build_index_name().startswith("my-log-")


# ---------------------------------------------------------------------------
# log_prompt — no-op when disabled
# ---------------------------------------------------------------------------


class TestLogPromptDisabled:
    """No OpenSearch activity when env var is absent."""

    def test_no_op_when_disabled(self) -> None:
        """_write_document is never called when logging is disabled."""
        env = {k: v for k, v in os.environ.items()
               if k != "BAMBOO_OPENSEARCH_PROMPTLOG"}
        with patch.dict(os.environ, env, clear=True):
            with patch("bamboo.llm.prompt_log._write_document") as mock_write:
                asyncio.run(
                    log_prompt(
                        system_prompt="You are AskPanDA.",
                        user_prompt="hello",
                        response="Hi",
                        tools_used=[],
                        provider="gemini",
                        model="gemini-2.0-flash",
                        max_tokens=512,
                    )
                )
                mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# log_prompt — document shape and redaction when enabled
# ---------------------------------------------------------------------------


class TestLogPromptEnabled:
    """When enabled, log_prompt fires a background task with a correct doc."""

    def test_document_shape_and_redaction(self) -> None:
        """Document contains expected keys; names are pseudonymised."""
        captured: list[dict[str, Any]] = []

        def _fake_write(doc: dict[str, Any]) -> None:
            captured.append(doc)

        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "testpw"}):
            with patch("bamboo.llm.prompt_log._write_document",
                       side_effect=_fake_write):

                async def _run() -> None:
                    await log_prompt(
                        system_prompt="You are AskPanDA.",
                        user_prompt='{"prodUserName": "jsmith"} show jobs',
                        response="The task owner is John Smith",
                        tools_used=["cric_query"],
                        provider="gemini",
                        model="gemini-2.0-flash",
                        max_tokens=2048,
                        input_tokens=100,
                        output_tokens=50,
                    )
                    await _drain_writes()

                asyncio.run(_run())

        assert len(captured) == 1
        doc = captured[0]

        # All required keys present
        for key in ("@timestamp", "session_id", "turn_number", "provider", "model",
                    "max_tokens", "system_prompt", "user_prompt", "response",
                    "tools_used", "input_tokens", "output_tokens"):
            assert key in doc, f"Missing key: {key}"

        # Correct values passed through
        assert doc["session_id"] == _SESSION_ID

        # turn_number must be a positive integer
        assert isinstance(doc["turn_number"], int)
        assert doc["turn_number"] >= 1
        assert doc["provider"] == "gemini"
        assert doc["model"] == "gemini-2.0-flash"
        assert doc["input_tokens"] == 100
        assert doc["output_tokens"] == 50
        assert doc["tools_used"] == ["cric_query"]

        # Names must be redacted in user_prompt and response
        assert "jsmith" not in doc["user_prompt"]
        assert "John" not in doc["response"]
        assert "Smith" not in doc["response"]

        # Pseudonyms present
        full_text = doc["user_prompt"] + doc["response"]
        assert "user_" in full_text

        # History must NOT be present — only the current turn is stored
        assert "messages" not in doc

    def test_null_tokens_stored_as_none(self) -> None:
        """When token counts are unavailable, null is stored (not omitted)."""
        captured: list[dict[str, Any]] = []

        def _fake_write(doc: dict[str, Any]) -> None:
            captured.append(doc)

        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "testpw"}):
            with patch("bamboo.llm.prompt_log._write_document",
                       side_effect=_fake_write):

                async def _run() -> None:
                    await log_prompt(
                        system_prompt="You are AskPanDA.",
                        user_prompt="hello",
                        response="ok",
                        tools_used=[],
                        provider="gemini",
                        model="gemini-2.0-flash",
                        max_tokens=512,
                    )
                    await _drain_writes()

                asyncio.run(_run())

        doc = captured[0]
        assert doc["input_tokens"] is None
        assert doc["output_tokens"] is None

    def test_opensearch_unavailable_does_not_raise(self) -> None:
        """Missing opensearch-py is silently swallowed — never crashes the caller."""
        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "pw"}):
            with patch.dict(sys.modules, {"opensearchpy": None}):  # type: ignore[dict-item]

                async def _run() -> None:
                    await log_prompt(
                        system_prompt="You are AskPanDA.",
                        user_prompt="hello",
                        response="ok",
                        tools_used=[],
                        provider="gemini",
                        model="gemini-2.0-flash",
                        max_tokens=512,
                    )
                    await _drain_writes()

                asyncio.run(_run())

    def test_turn_number_increments(self) -> None:
        """turn_number increments by 1 on each log_prompt call."""
        captured: list[dict[str, Any]] = []

        def _fake_write(doc: dict[str, Any]) -> None:
            captured.append(doc)

        with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "testpw"}):
            with patch("bamboo.llm.prompt_log._write_document",
                       side_effect=_fake_write):

                async def _run() -> None:
                    # Reset counter so test is order-independent
                    _pl_module._turn_counter = 0
                    for _ in range(3):
                        await log_prompt(
                            system_prompt="You are AskPanDA.",
                            user_prompt="hello",
                            response="ok",
                            tools_used=[],
                            provider="gemini",
                            model="gemini-2.0-flash",
                            max_tokens=512,
                        )
                    await _drain_writes()

                asyncio.run(_run())

        assert len(captured) == 3
        # Sorted, not as captured: the three writes run concurrently in the
        # default thread pool, so arrival order is not assignment order.
        # The property under test is that each call takes the next number,
        # which is decided on the event loop before the task is spawned.
        assert sorted(d["turn_number"] for d in captured) == [1, 2, 3]

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """After threshold consecutive failures logging is disabled for the session."""

    def _reset_circuit(self) -> None:
        """Reset module-level circuit breaker state between tests."""
        _pl_module._consecutive_failures = 0
        _pl_module._circuit_open = False

    def test_warning_on_first_failures(self) -> None:
        """A WARNING is emitted for each failure below the threshold."""
        self._reset_circuit()
        boom = Exception("403 Forbidden")
        with patch("bamboo.llm.prompt_log._create_os_client", side_effect=boom):
            with patch("bamboo.llm.prompt_log.logger") as mock_log:
                _pl_module._write_document({"@timestamp": "t"})
                mock_log.warning.assert_called_once()
                # Circuit should still be closed after one failure
                assert not _pl_module._circuit_open

    def test_circuit_opens_at_threshold(self) -> None:
        """Circuit opens and ERROR is logged when threshold is reached."""
        self._reset_circuit()
        boom = Exception("403 Forbidden")
        with patch("bamboo.llm.prompt_log._create_os_client", side_effect=boom):
            with patch("bamboo.llm.prompt_log.logger") as mock_log:
                for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
                    _pl_module._write_document({"@timestamp": "t"})
                assert _pl_module._circuit_open
                mock_log.error.assert_called_once()

    def test_no_writes_after_circuit_opens(self) -> None:
        """Once the circuit is open, _create_os_client is never called again."""
        self._reset_circuit()
        _pl_module._circuit_open = True
        with patch("bamboo.llm.prompt_log._create_os_client") as mock_client:
            _pl_module._write_document({"@timestamp": "t"})
            mock_client.assert_not_called()

    def test_success_resets_failure_counter(self) -> None:
        """A successful write resets the consecutive failure counter to zero."""
        self._reset_circuit()
        boom = Exception("503 Service Unavailable")
        # One failure to increment the counter...
        with patch("bamboo.llm.prompt_log._create_os_client", side_effect=boom):
            _pl_module._write_document({"@timestamp": "t"})
        assert _pl_module._consecutive_failures == 1

        # ...then a success resets it.
        mock_client = MagicMock()
        with patch("bamboo.llm.prompt_log._create_os_client",
                   return_value=mock_client):
            _pl_module._write_document({"@timestamp": "t"})
        assert _pl_module._consecutive_failures == 0
        assert not _pl_module._circuit_open
