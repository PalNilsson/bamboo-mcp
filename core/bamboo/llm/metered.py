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

"""Token accounting wrapper around provider clients.

Why the wrapper sits here rather than at a call site
----------------------------------------------------
``client.generate()`` is called from at least ten places: synthesis
(``llm_passthrough``), the LLM planner, the topic guard, the promptlog
NL-to-DSL translator, the connectivity probe, and several ATLAS plugin
implementations.  Metering any one of them would miss the rest, and the
planner in particular is not a rounding error — its prompt carries the whole
tool catalogue.

Every one of those obtains its client from ``bamboo.llm.factory.build_client``,
directly or through ``LLMClientManager``, so wrapping the factory output
catches all of them and any added later.  It also catches the core-dump
analyzer, which calls ``build_client`` from a detached worker process; because
the counter is a locked file rather than a process global, that spend lands in
the same day's total.

The wrapper delegates everything except :meth:`generate`, where it reads the
already-normalised ``LLMResponse.usage`` and hands it to
:mod:`bamboo.cost_guard`.  Recording never interferes with the response: an
accounting failure is logged and swallowed.

The pre-call budget check is opt-in (``BAMBOO_COST_ENFORCE``).  Off by default
because refusing a call mid-conversation is a worse outcome than a small
overspend; the REST facade gets its enforcement from ``check_budget()`` at
admission instead, where a refusal is a clean 429 before any work starts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from bamboo import cost_guard
from bamboo.llm.base import LLMClient
from bamboo.llm.types import GenerateParams, LLMResponse, Message, ModelSpec

logger = logging.getLogger(__name__)


class MeteredLLMClient(LLMClient):
    """Delegating client that records token usage for every completion.

    Attributes:
        inner: The provider client being wrapped.
    """

    def __init__(self, inner: LLMClient) -> None:
        """Wrap a provider client.

        Args:
            inner: The client to delegate to.
        """
        super().__init__(inner.model_spec)
        self.inner = inner

    @property
    def model_spec(self) -> ModelSpec:
        """Return the wrapped client's model spec.

        Returns:
            The model specification.
        """
        return self.inner.model_spec

    async def close(self) -> None:
        """Close the wrapped client."""
        await self.inner.close()

    async def generate(
        self,
        messages: Sequence[Message],
        params: GenerateParams,
    ) -> LLMResponse:
        """Generate a completion and record its token usage.

        Args:
            messages: Conversation messages in normalized format.
            params: Generation parameters.

        Returns:
            The wrapped client's response, unmodified.

        Raises:
            BudgetExhausted: If ``BAMBOO_COST_ENFORCE`` is set and the daily
                budget is already spent.
        """
        spec = self.inner.model_spec

        if cost_guard.enforcement_enabled():
            status = cost_guard.check_budget()
            if not status.allowed:
                raise cost_guard.BudgetExhausted(
                    f"Daily LLM budget of ${status.budget_usd:.2f} is spent "
                    f"(${status.spent_usd:.2f} recorded). Resets in "
                    f"{status.retry_after_s / 3600:.1f} h."
                )

        response = await self.inner.generate(messages=messages, params=params)

        try:
            cost_guard.record_usage(spec.provider, spec.model, response.usage)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Accounting must never break the call that produced it.
            logger.warning(
                "cost_guard: failed to record usage for %s/%s: %s",
                spec.provider,
                spec.model,
                exc,
            )

        return response


__all__ = ["MeteredLLMClient"]
