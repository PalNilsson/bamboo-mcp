"""Direct LLM passthrough tool.

This tool is intentionally simple: it forwards the full chat prompt (history)
to the configured **default** LLM profile and returns the model's raw text.

Use-cases:
  1) Sanity-check that LLM configuration (keys, provider adapters, networking)
     works end-to-end through MCP.
  2) Provide an explicit "bypass reasoning engine" path later, when the
     orchestration layer starts selecting tools.
"""
from __future__ import annotations

import os
from typing import Any, cast

from bamboo.prompts.templates import get_bamboo_system_prompt
from bamboo.tools.base import text_content, coerce_messages

from bamboo.llm.exceptions import LLMError
from bamboo.llm.runtime import get_llm_manager, get_llm_selector
from bamboo.llm.types import GenerateParams, Message, TokenUsage
from bamboo.tracing import EVENT_LLM_CALL, span


class LLMPassthroughTool:
    """Calls the default LLM with the full provided prompt.

    This tool forwards either a provided `messages` chat history or a single
    `question` string (wrapped as a user message) to the project's configured
    default LLM profile. The raw text response from the model is returned as
    a single text content block.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool discovery definition.

        The returned mapping describes the tool's name, description and the
        expected input schema so clients can discover and validate calls.

        Returns:
            Dict[str, Any]: Tool definition compatible with MCP discovery.
        """
        return {
            "name": "bamboo_llm_answer",
            "description": (
                "Send a question or conversation directly to the default LLM and "
                "return its response. Use for open-ended questions, follow-ups, or "
                "tasks that do not require live PanDA data. "
                "Does not call any other tools."
            ),
            "inputSchema": {
                "type": "object",
                "anyOf": [
                    {"required": ["question"]},
                    {"required": ["messages"]}
                ],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User question (used if messages is not provided).",
                    },
                    "messages": {
                        "type": "array",
                        "description": (
                            "Optional full chat history as a list of {role, content}. "
                            "If provided, it is sent to the LLM as-is (plus a system prompt)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature.",
                        "default": 0.2,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional max tokens for the completion.",
                    },
                },
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute the passthrough call against the default LLM.

        Thin MCP wrapper around :meth:`generate_text`.  The token usage the
        inner method returns is discarded here because an MCP tool's return
        value must be a content list; in-process callers that want the usage
        should call :meth:`generate_text` directly.

        Args:
            arguments: Tool arguments; either a `messages` list or a `question`
                string is required. Optional `temperature` and `max_tokens`
                controls the generation parameters.

        Returns:
            List[Dict[str, Any]]: A one-element list containing the model's
            raw text response annotated with a debug header.

        Raises:
            ValueError: If neither `question` nor non-empty `messages` is provided.
            RuntimeError: If the configured LLM selector does not expose a
                registry or the manager cannot obtain a client.
        """
        text, _usage = await self.generate_text(arguments)
        return text_content(text)

    async def generate_text(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, TokenUsage | None]:
        """Call the default LLM and return its text together with token usage.

        This is the in-process entry point.  :meth:`call` wraps it for MCP,
        where the return value has to be a content list and the usage
        therefore cannot travel with it.  Returning the usage explicitly —
        rather than stashing it somewhere the caller reads back — keeps it
        unambiguous which call the numbers belong to when several are in
        flight at once.

        Args:
            arguments: Tool arguments; either a `messages` list or a `question`
                string is required. Optional `temperature` and `max_tokens`
                control the generation parameters.

        Returns:
            Tuple of the model's raw text response and its
            :class:`~bamboo.llm.types.TokenUsage`, which is ``None`` when the
            provider adapter reported none.

        Raises:
            ValueError: If neither `question` nor non-empty `messages` is provided.
            RuntimeError: If the configured LLM selector does not expose a
                registry or the manager cannot obtain a client.
        """
        selector = get_llm_selector()
        manager = get_llm_manager()

        # Determine which profile is considered "default".
        default_profile = getattr(selector, "default_profile", "default")
        registry = getattr(selector, "registry", None)
        if registry is None:
            raise RuntimeError("LLM selector does not expose a registry.")

        model_spec = registry.get(default_profile)
        client = await manager.get_client(model_spec)

        temperature = float(arguments.get("temperature", 0.2))
        max_tokens = arguments.get("max_tokens")
        max_tokens_int = int(max_tokens) if max_tokens is not None else None

        # Build message list.
        plugin_id: str = os.getenv("ASKPANDA_PLUGIN", "atlas").strip().lower()
        sys_prompt = await get_bamboo_system_prompt(plugin_id=plugin_id)
        sys_text = getattr(sys_prompt, "text", None) or getattr(sys_prompt, "content", None)
        if isinstance(sys_text, list):
            # Some MCP prompt objects use content items.
            sys_text = "\n".join([str(x.get("text", "")) for x in sys_text if isinstance(x, dict)])
        system_message: Message = {"role": "system", "content": str(sys_text or "")}

        messages_arg = arguments.get("messages")
        messages: list[Message] = [system_message]
        if isinstance(messages_arg, list) and messages_arg:
            messages.extend(cast(list[Message], coerce_messages(messages_arg)))
        else:
            question = str(arguments.get("question", "")).strip()
            if not question:
                raise ValueError("Either 'question' or non-empty 'messages' must be provided.")
            messages.append({"role": "user", "content": question})

        try:
            async with span(
                EVENT_LLM_CALL,
                tool="bamboo_llm_answer",
                provider=model_spec.provider,
                model=model_spec.model,
            ) as _llm_span:
                resp = await client.generate(
                    messages=messages,
                    params=GenerateParams(temperature=temperature, max_tokens=max_tokens_int),
                )
                _usage = resp.usage
                _llm_span.set(
                    input_tokens=_usage.input_tokens if _usage else None,
                    output_tokens=_usage.output_tokens if _usage else None,
                )
            return resp.text, _usage
        except LLMError:
            # Re-raise so the orchestrating tool (bamboo_answer) can apply its
            # own friendly-error handler.  Do not swallow here — the span will
            # still have been emitted via the finally branch of the context manager.
            raise
        manager = get_llm_manager()

        # Determine which profile is considered "default".
        default_profile = getattr(selector, "default_profile", "default")
        registry = getattr(selector, "registry", None)
        if registry is None:
            raise RuntimeError("LLM selector does not expose a registry.")

        model_spec = registry.get(default_profile)
        client = await manager.get_client(model_spec)

        temperature = float(arguments.get("temperature", 0.2))
        max_tokens = arguments.get("max_tokens")
        max_tokens_int = int(max_tokens) if max_tokens is not None else None

        # Build message list.
        plugin_id: str = os.getenv("ASKPANDA_PLUGIN", "atlas").strip().lower()
        sys_prompt = await get_bamboo_system_prompt(plugin_id=plugin_id)
        sys_text = getattr(sys_prompt, "text", None) or getattr(sys_prompt, "content", None)
        if isinstance(sys_text, list):
            # Some MCP prompt objects use content items.
            sys_text = "\n".join([str(x.get("text", "")) for x in sys_text if isinstance(x, dict)])
        system_message: Message = {"role": "system", "content": str(sys_text or "")}

        messages_arg = arguments.get("messages")
        messages: list[Message] = [system_message]
        if isinstance(messages_arg, list) and messages_arg:
            messages.extend(cast(list[Message], coerce_messages(messages_arg)))
        else:
            question = str(arguments.get("question", "")).strip()
            if not question:
                raise ValueError("Either 'question' or non-empty 'messages' must be provided.")
            messages.append({"role": "user", "content": question})

        try:
            async with span(
                EVENT_LLM_CALL,
                tool="bamboo_llm_answer",
                provider=model_spec.provider,
                model=model_spec.model,
            ) as _llm_span:
                resp = await client.generate(
                    messages=messages,
                    params=GenerateParams(temperature=temperature, max_tokens=max_tokens_int),
                )
                _usage = resp.usage
                _llm_span.set(
                    input_tokens=_usage.input_tokens if _usage else None,
                    output_tokens=_usage.output_tokens if _usage else None,
                )
            return text_content(resp.text)
        except LLMError:
            # Re-raise so the orchestrating tool (bamboo_answer) can apply its
            # own friendly-error handler.  Do not swallow here — the span will
            # still have been emitted via the finally branch of the context manager.
            raise


bamboo_llm_answer_tool = LLMPassthroughTool()


def get_llm_info() -> str:
    """Return a human-readable string describing the active LLM provider and model.

    Used by the TUI to display the LLM selection once at startup rather than
    repeating it in every response panel.

    Returns:
        String of the form ``"provider=<p> model=<m>"``, or an empty string
        if the selector is not yet configured.
    """
    try:
        selector = get_llm_selector()
        registry = getattr(selector, "registry", None)
        if registry is None:
            return ""
        default_profile = getattr(selector, "default_profile", "default")
        model_spec = registry.get(default_profile)
        if model_spec is None:
            return ""
        return f"provider={model_spec.provider} model={model_spec.model}"
    except Exception:  # pylint: disable=broad-exception-caught
        return ""
