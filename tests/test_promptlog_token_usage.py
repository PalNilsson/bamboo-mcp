"""Tests that synthesis token usage reaches the prompt log.

Every prompt-log document in OpenSearch carried ``input_tokens: null`` and
``output_tokens: null``, because ``LLMPassthroughTool`` read ``resp.usage``
into the tracing span and then returned only ``text_content(resp.text)``, so
``call_llm`` had nothing to pass and hardcoded ``None`` for both fields.
``docs/opensearch.md`` documents the fields as populated and offers a
"average output token count per model" aggregation, which therefore returned
nothing.

Coverage:
- Usage survives the hop from the provider client to ``log_prompt``.
- A reported zero stays zero rather than becoming ``None``.
- Absent usage yields ``None`` for both fields without raising.
- ``call()`` still returns a one-element ``list[MCPContent]`` for MCP clients.
- The usage belongs to the call that produced it under concurrency.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bamboo.llm.types import LLMResponse, ModelSpec, TokenUsage
from bamboo.tools.bamboo_executor import call_llm
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime_mocks(
    responses: list[LLMResponse] | LLMResponse,
) -> tuple[MagicMock, MagicMock]:
    """Build fake selector and manager returning *responses* from generate().

    Mirrors the chain ``get_llm_selector()`` → ``selector.registry.get()`` →
    ``await manager.get_client()`` → ``await client.generate()`` that
    :meth:`LLMPassthroughTool.generate_text` walks.

    Args:
        responses: A single response returned for every call, or a list used
            as a side-effect sequence.

    Returns:
        Tuple of ``(fake_selector, fake_manager)`` for use as patch targets.
    """
    fake_client = MagicMock()
    if isinstance(responses, list):
        fake_client.generate = AsyncMock(side_effect=responses)
    else:
        fake_client.generate = AsyncMock(return_value=responses)

    fake_manager = MagicMock()
    fake_manager.get_client = AsyncMock(return_value=fake_client)

    fake_registry = MagicMock()
    fake_registry.get = MagicMock(
        return_value=ModelSpec(provider="anthropic", model="stub-model")
    )

    fake_selector = MagicMock()
    fake_selector.default_profile = "default"
    fake_selector.registry = fake_registry

    return fake_selector, fake_manager


async def _call_llm_capturing_log(
    response: LLMResponse,
) -> dict[str, Any]:
    """Run ``call_llm`` against *response* and return the ``log_prompt`` kwargs.

    ``log_prompt`` is scheduled with :func:`asyncio.create_task` inside a
    ``try`` that swallows exceptions, so the patch replaces it with a
    coroutine function: a ``MagicMock`` would not be awaitable and the failure
    would be silently logged instead of surfacing as a kwargs mismatch.

    Args:
        response: The response the fake provider client returns.

    Returns:
        The keyword arguments ``log_prompt`` was called with.

    Raises:
        AssertionError: If ``log_prompt`` was never called.
    """
    captured: dict[str, Any] = {}

    async def _fake_log_prompt(**kwargs: Any) -> None:
        """Record the call instead of writing to OpenSearch.

        Args:
            **kwargs: Whatever ``call_llm`` passed.
        """
        captured.update(kwargs)

    fake_selector, fake_manager = _runtime_mocks(response)

    with (
        patch("bamboo.tools.llm_passthrough.get_llm_selector",
              MagicMock(return_value=fake_selector)),
        patch("bamboo.tools.llm_passthrough.get_llm_manager",
              MagicMock(return_value=fake_manager)),
        patch("bamboo.llm.prompt_log.log_prompt", _fake_log_prompt),
        patch("bamboo.tools.llm_passthrough.get_bamboo_system_prompt",
              AsyncMock(return_value=MagicMock(text="stub system prompt"))),
    ):
        await call_llm(system="sys", user="usr", raw_question="why did job 1 fail?")
        # log_prompt is fire-and-forget; let the scheduled task run.
        await asyncio.sleep(0)

    assert captured, "log_prompt was never called"
    return captured


# ---------------------------------------------------------------------------
# Usage reaches the prompt log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_reaches_log_prompt() -> None:
    """Provider-reported token counts arrive in the log_prompt call."""
    captured = await _call_llm_capturing_log(
        LLMResponse(text="answer", usage=TokenUsage(123, 45, 168))
    )

    assert captured["input_tokens"] == 123
    assert captured["output_tokens"] == 45


@pytest.mark.asyncio
async def test_zero_usage_is_not_coerced_to_none() -> None:
    """A reported zero stays zero.

    The guard in ``call_llm`` is ``is not None`` rather than truthiness. With
    a truthiness check a genuine zero would be logged as ``null``, which is
    indistinguishable from "the provider told us nothing" — the exact
    confusion this commit exists to remove. The existing narrow-waist harness
    builds its fake response with ``TokenUsage(0, 0, 0)``, so the wrong
    implementation passes every other test in the suite.
    """
    captured = await _call_llm_capturing_log(
        LLMResponse(text="answer", usage=TokenUsage(0, 0, 0))
    )

    assert captured["input_tokens"] == 0
    assert captured["output_tokens"] == 0
    assert captured["input_tokens"] is not None
    assert captured["output_tokens"] is not None


@pytest.mark.asyncio
async def test_absent_usage_logs_none() -> None:
    """A provider reporting no usage yields None for both fields."""
    captured = await _call_llm_capturing_log(
        LLMResponse(text="answer", usage=None)
    )

    assert captured["input_tokens"] is None
    assert captured["output_tokens"] is None


@pytest.mark.asyncio
async def test_partial_usage_is_passed_through_unchanged() -> None:
    """A provider filling only one field does not lose the one it filled."""
    captured = await _call_llm_capturing_log(
        LLMResponse(text="answer", usage=TokenUsage(input_tokens=77))
    )

    assert captured["input_tokens"] == 77
    assert captured["output_tokens"] is None


@pytest.mark.asyncio
async def test_response_text_is_unchanged_by_the_split() -> None:
    """call_llm still returns the model's text, not a content wrapper."""
    fake_selector, fake_manager = _runtime_mocks(
        LLMResponse(text="the answer text", usage=TokenUsage(1, 2, 3))
    )

    with (
        patch("bamboo.tools.llm_passthrough.get_llm_selector",
              MagicMock(return_value=fake_selector)),
        patch("bamboo.tools.llm_passthrough.get_llm_manager",
              MagicMock(return_value=fake_manager)),
        patch("bamboo.tools.llm_passthrough.get_bamboo_system_prompt",
              AsyncMock(return_value=MagicMock(text="stub"))),
    ):
        text = await call_llm(system="sys", user="usr")

    assert text == "the answer text"


# ---------------------------------------------------------------------------
# The MCP surface is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_still_returns_mcp_content_list() -> None:
    """call() keeps its narrow-waist return type after the split.

    ``call()`` is registered in the MCP tool registry, so its return value has
    to stay a one-element ``list[MCPContent]``. Returning the tuple that
    ``generate_text`` produces would be a protocol-level break for any client
    invoking ``bamboo_llm_answer`` directly.
    """
    fake_selector, fake_manager = _runtime_mocks(
        LLMResponse(text="mcp text", usage=TokenUsage(5, 6, 11))
    )

    with (
        patch("bamboo.tools.llm_passthrough.get_llm_selector",
              MagicMock(return_value=fake_selector)),
        patch("bamboo.tools.llm_passthrough.get_llm_manager",
              MagicMock(return_value=fake_manager)),
        patch("bamboo.tools.llm_passthrough.get_bamboo_system_prompt",
              AsyncMock(return_value=MagicMock(text="stub"))),
    ):
        result = await bamboo_llm_answer_tool.call({"question": "hello"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0].get("text") == "mcp text"


@pytest.mark.asyncio
async def test_generate_text_returns_text_and_usage() -> None:
    """generate_text hands back both halves of what the provider returned."""
    fake_selector, fake_manager = _runtime_mocks(
        LLMResponse(text="inner text", usage=TokenUsage(11, 22, 33))
    )

    with (
        patch("bamboo.tools.llm_passthrough.get_llm_selector",
              MagicMock(return_value=fake_selector)),
        patch("bamboo.tools.llm_passthrough.get_llm_manager",
              MagicMock(return_value=fake_manager)),
        patch("bamboo.tools.llm_passthrough.get_bamboo_system_prompt",
              AsyncMock(return_value=MagicMock(text="stub"))),
    ):
        text, usage = await bamboo_llm_answer_tool.generate_text(
            {"question": "hello"}
        )

    assert text == "inner text"
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 22


# ---------------------------------------------------------------------------
# Attribution under concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_is_attributed_to_its_own_call() -> None:
    """Two concurrent synthesis calls do not swap token counts.

    An explicit return value is used rather than a "last usage" side channel
    precisely so this holds. A shared slot would make the pairing depend on
    interleaving, and the wrong pairing is invisible in the index — the totals
    still look plausible.
    """
    responses = [
        LLMResponse(text="first", usage=TokenUsage(100, 10, 110)),
        LLMResponse(text="second", usage=TokenUsage(200, 20, 220)),
    ]
    fake_selector, fake_manager = _runtime_mocks(responses)

    seen: list[tuple[str, int | None, int | None]] = []

    async def _fake_log_prompt(**kwargs: Any) -> None:
        """Record (response, input_tokens, output_tokens) per call.

        Args:
            **kwargs: Whatever ``call_llm`` passed.
        """
        seen.append((
            kwargs["response"],
            kwargs["input_tokens"],
            kwargs["output_tokens"],
        ))

    with (
        patch("bamboo.tools.llm_passthrough.get_llm_selector",
              MagicMock(return_value=fake_selector)),
        patch("bamboo.tools.llm_passthrough.get_llm_manager",
              MagicMock(return_value=fake_manager)),
        patch("bamboo.llm.prompt_log.log_prompt", _fake_log_prompt),
        patch("bamboo.tools.llm_passthrough.get_bamboo_system_prompt",
              AsyncMock(return_value=MagicMock(text="stub"))),
    ):
        await asyncio.gather(
            call_llm(system="sys", user="one"),
            call_llm(system="sys", user="two"),
        )
        await asyncio.sleep(0)

    assert len(seen) == 2, f"expected two logged turns, got {seen}"
    by_text = {text: (inp, out) for text, inp, out in seen}
    assert by_text["first"] == (100, 10)
    assert by_text["second"] == (200, 20)
