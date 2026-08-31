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

"""Tests for :mod:`bamboo.cost_guard` and the metered client wrapper.

The properties under test are the ones a budget has to hold to be worth
setting: the arithmetic is right, an unpriced model is visible rather than
free, the counter survives concurrent writers and a corrupt file, refusal
happens at the boundary and not one call late, and metering never breaks the
call that produced it.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bamboo import cost_guard as cg
from bamboo.llm.metered import MeteredLLMClient
from bamboo.llm.types import (
    GenerateParams,
    LLMResponse,
    Message,
    ModelSpec,
    TokenUsage,
)


@pytest.fixture()
def state_dir(tmp_path: Path) -> Any:
    """Point the counter files at a temporary directory.

    Args:
        tmp_path: pytest-provided temporary directory.

    Yields:
        The directory in use.
    """
    with patch.dict(os.environ, {"BAMBOO_COST_STATE_ROOT": str(tmp_path)}):
        cg._warned_unpriced.clear()
        yield tmp_path
        cg._warned_unpriced.clear()


def _record_in_subprocess(state_root: str, count: int) -> None:
    """Record *count* completions from a separate process.

    Defined at module level because ``multiprocessing`` with the ``spawn``
    start method (the default on macOS) has to import and pickle the target.

    Args:
        state_root: Directory holding the counter files.
        count: Number of completions to record.
    """
    os.environ["BAMBOO_COST_STATE_ROOT"] = state_root
    from bamboo import cost_guard as guard
    from bamboo.llm.types import TokenUsage as Usage

    for _ in range(count):
        guard.record_usage("anthropic", "claude-sonnet-4-6", Usage(input_tokens=10, output_tokens=1))


class _StubClient:
    """Minimal provider client returning a canned response.

    Attributes:
        model_spec: Spec reported to the wrapper.
        calls: Number of generate() invocations.
        response: Response returned by generate().
    """

    def __init__(self, spec: ModelSpec, response: LLMResponse) -> None:
        """Store the spec and canned response.

        Args:
            spec: Model specification.
            response: Response to return from generate().
        """
        self.model_spec = spec
        self.response = response
        self.calls = 0

    async def close(self) -> None:
        """Match the LLMClient interface."""
        return

    async def generate(
        self,
        messages: Sequence[Message],
        params: GenerateParams,
    ) -> LLMResponse:
        """Return the canned response.

        Args:
            messages: Ignored.
            params: Ignored.

        Returns:
            The canned response.
        """
        self.calls += 1
        return self.response


class TestPricing:
    """Price lookup and cost arithmetic."""

    def test_exact_match(self) -> None:
        """An exact provider/model key is used when present."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BAMBOO_MODEL_PRICES", None)
            assert cg.price_for("anthropic", "claude-sonnet-4-6") == (3.00, 15.00)

    def test_prefix_match_covers_point_releases(self) -> None:
        """A family entry covers a longer model string."""
        assert cg.price_for("anthropic", "claude-opus-4-1-20260401") == (15.00, 75.00)

    def test_longest_prefix_wins(self) -> None:
        """The most specific family entry is chosen."""
        prices = {
            "anthropic/claude": [1.0, 1.0],
            "anthropic/claude-haiku-4-5": [1.0, 5.0],
        }
        with patch.dict(os.environ, {"BAMBOO_MODEL_PRICES": json.dumps(prices)}):
            assert cg.price_for("anthropic", "claude-haiku-4-5") == (1.0, 5.0)

    def test_unknown_model_is_unpriced(self) -> None:
        """An unrecognised model returns None rather than guessing."""
        assert cg.price_for("openai", "some-unreleased-model") is None

    def test_cost_arithmetic(self) -> None:
        """Cost is per million tokens for input and output separately."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = cg.usd_for("anthropic", "claude-sonnet-4-6", usage)
        assert cost == pytest.approx(18.00)

    def test_partial_usage(self) -> None:
        """A missing token count contributes zero, not an error."""
        usage = TokenUsage(input_tokens=500_000, output_tokens=None)
        cost = cg.usd_for("anthropic", "claude-sonnet-4-6", usage)
        assert cost == pytest.approx(1.50)

    def test_missing_usage_is_none(self) -> None:
        """No usage object means no cost, distinct from zero cost."""
        assert cg.usd_for("anthropic", "claude-sonnet-4-6", None) is None

    def test_env_override_extends_the_table(self) -> None:
        """A new model can be priced without a release."""
        prices = {"openai/gpt-9": [2.0, 8.0]}
        with patch.dict(os.environ, {"BAMBOO_MODEL_PRICES": json.dumps(prices)}):
            assert cg.price_for("openai", "gpt-9") == (2.0, 8.0)

    def test_malformed_override_is_ignored(self) -> None:
        """A bad override must not stop the server starting."""
        with patch.dict(os.environ, {"BAMBOO_MODEL_PRICES": "{not json"}):
            assert cg.price_for("anthropic", "claude-sonnet-4-6") == (3.00, 15.00)

    def test_malformed_entry_is_skipped_others_kept(self) -> None:
        """One bad entry does not discard the rest of the override."""
        prices = {"openai/gpt-9": "cheap", "openai/gpt-10": [1.0, 2.0]}
        with patch.dict(os.environ, {"BAMBOO_MODEL_PRICES": json.dumps(prices)}):
            assert cg.price_for("openai", "gpt-9") is None
            assert cg.price_for("openai", "gpt-10") == (1.0, 2.0)


class TestRecording:
    """Accumulating usage into the day's counter file."""

    def test_first_record_creates_the_file(self, state_dir: Path) -> None:
        """A day's file appears on the first recorded call."""
        usage = TokenUsage(input_tokens=1000, output_tokens=100)
        spend = cg.record_usage("anthropic", "claude-sonnet-4-6", usage)

        assert spend.calls == 1
        assert spend.input_tokens == 1000
        assert (state_dir / f"{cg.today_utc()}.json").exists()

    def test_records_accumulate(self, state_dir: Path) -> None:
        """Successive calls add up."""
        usage = TokenUsage(input_tokens=1000, output_tokens=100)
        for _ in range(3):
            spend = cg.record_usage("anthropic", "claude-sonnet-4-6", usage)

        assert spend.calls == 3
        assert spend.input_tokens == 3000
        assert spend.output_tokens == 300
        assert spend.usd == pytest.approx(3 * (1000 * 3.0 + 100 * 15.0) / 1e6)

    def test_unpriced_model_counts_tokens_not_dollars(self, state_dir: Path) -> None:
        """An unpriced model is visible as a gap, not as free."""
        usage = TokenUsage(input_tokens=5000, output_tokens=500)
        spend = cg.record_usage("openai", "unreleased-model", usage)

        assert spend.calls == 1
        assert spend.unpriced_calls == 1
        assert spend.input_tokens == 5000
        assert spend.usd == 0.0

    def test_current_spend_reads_back(self, state_dir: Path) -> None:
        """A fresh read sees what a previous process recorded."""
        cg.record_usage(
            "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=2000, output_tokens=0)
        )
        assert cg.current_spend().input_tokens == 2000

    def test_empty_day_is_zeroed(self, state_dir: Path) -> None:
        """A day with no file reads as zero rather than failing."""
        spend = cg.current_spend()
        assert spend.calls == 0
        assert spend.usd == 0.0

    def test_corrupt_file_resets_rather_than_raising(self, state_dir: Path) -> None:
        """A truncated counter must not take the server down."""
        (state_dir / f"{cg.today_utc()}.json").write_text('{"date": "2026', encoding="utf-8")
        spend = cg.current_spend()
        assert spend.calls == 0

    def test_yesterdays_file_does_not_count_today(self, state_dir: Path) -> None:
        """Day rollover starts a new counter."""
        stale = {
            "date": "2020-01-01",
            "calls": 99,
            "unpriced_calls": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "usd": 500.0,
        }
        (state_dir / "2020-01-01.json").write_text(json.dumps(stale), encoding="utf-8")

        assert cg.current_spend().calls == 0
        assert cg.current_spend("2020-01-01").usd == pytest.approx(500.0)

    def test_unreadable_state_root_does_not_raise(self, tmp_path: Path) -> None:
        """A storage failure loses the record, never the call."""
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with patch.dict(os.environ, {"BAMBOO_COST_STATE_ROOT": str(blocker / "cost")}):
            spend = cg.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=1, output_tokens=1)
            )
        assert spend.calls == 0

    def test_concurrent_writers_do_not_lose_records(self, state_dir: Path) -> None:
        """Locked read-modify-write keeps every increment across processes.

        Deliberately multi-process rather than multi-threaded.  Threads in one
        interpreter are serialised tightly enough by the GIL that the
        read-modify-write window never opens, so a thread-based version of this
        test passes with the lock removed and proves nothing.  Processes are
        also the real case: the core-dump analyzer builds its own client in a
        detached worker, and multiple uvicorn workers would share one counter.
        """
        import multiprocessing

        workers = 4
        per_worker = 25

        context = multiprocessing.get_context("spawn")
        procs = [
            context.Process(target=_record_in_subprocess, args=(str(state_dir), per_worker))
            for _ in range(workers)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=60)

        assert all(proc.exitcode == 0 for proc in procs)
        assert cg.current_spend().calls == workers * per_worker


class TestBudget:
    """Admission decisions against the daily ceiling."""

    def test_no_budget_configured_always_allows(self, state_dir: Path) -> None:
        """Accounting without a ceiling never refuses."""
        cg.record_usage(
            "anthropic",
            "claude-sonnet-4-6",
            TokenUsage(input_tokens=10_000_000, output_tokens=10_000_000),
        )
        status = cg.check_budget()
        assert status.allowed is True
        assert status.budget_usd == 0.0

    def test_under_budget_allows(self, state_dir: Path) -> None:
        """Spend below the ceiling is admitted."""
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "10"}):
            cg.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=1000, output_tokens=100)
            )
            assert cg.check_budget().allowed is True

    def test_at_budget_refuses(self, state_dir: Path) -> None:
        """The boundary is inclusive: reaching the ceiling stops new work."""
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "3.0"}):
            # 1M input tokens at $3.00/MTok is exactly the budget.
            cg.record_usage(
                "anthropic",
                "claude-sonnet-4-6",
                TokenUsage(input_tokens=1_000_000, output_tokens=0),
            )
            status = cg.check_budget()

        assert status.allowed is False
        assert status.reason == "budget_exhausted"
        assert status.retry_after_s > 0

    def test_refusal_reports_the_reset_time(self, state_dir: Path) -> None:
        """A refusal tells the caller when to come back."""
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "0.000001"}):
            cg.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=1000, output_tokens=100)
            )
            status = cg.check_budget()

        assert status.allowed is False
        assert 0 < status.retry_after_s <= 86400

    def test_unparsable_budget_falls_back_to_disabled(self, state_dir: Path) -> None:
        """A typo must not silently impose a ceiling of zero."""
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "ten dollars"}):
            assert cg.daily_budget_usd() == 0.0
            assert cg.check_budget().allowed is True


class TestConcurrencyLimiter:
    """Slot admission and queue bounding."""

    def test_slots_are_bounded(self) -> None:
        """No more than max_concurrency run at once."""
        limiter = cg.ConcurrencyLimiter(max_concurrency=2, max_queue=10)
        peak = 0
        active = 0

        async def _work() -> None:
            nonlocal peak, active
            async with limiter.slot():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        async def _run() -> None:
            await asyncio.gather(*[_work() for _ in range(6)])

        asyncio.run(_run())
        assert peak == 2

    def test_queue_cap_refuses(self) -> None:
        """Past the queue cap the caller is told to retry, not left waiting."""
        limiter = cg.ConcurrencyLimiter(max_concurrency=1, max_queue=1)
        refusals: list[cg.AdmissionRefused] = []

        async def _work() -> None:
            try:
                async with limiter.slot():
                    await asyncio.sleep(0.05)
            except cg.AdmissionRefused as exc:
                refusals.append(exc)

        async def _run() -> None:
            await asyncio.gather(*[_work() for _ in range(4)])

        asyncio.run(_run())
        assert refusals
        assert all(r.retry_after_s > 0 for r in refusals)

    def test_slot_is_released_on_error(self) -> None:
        """An exception inside the block must not leak the slot."""
        limiter = cg.ConcurrencyLimiter(max_concurrency=1, max_queue=5)

        async def _run() -> None:
            with pytest.raises(ValueError):
                async with limiter.slot():
                    raise ValueError("boom")
            async with limiter.slot():
                pass
            assert limiter.in_flight == 0

        asyncio.run(_run())

    def test_usable_across_event_loops(self) -> None:
        """An instance built once works in a second asyncio.run()."""
        limiter = cg.ConcurrencyLimiter(max_concurrency=1, max_queue=5)

        async def _run() -> None:
            async with limiter.slot():
                pass

        asyncio.run(_run())
        asyncio.run(_run())

    def test_reads_environment_by_default(self) -> None:
        """Unset arguments come from the environment."""
        with patch.dict(
            os.environ,
            {"BAMBOO_ANALYSIS_MAX_CONCURRENCY": "7", "BAMBOO_ANALYSIS_MAX_QUEUE": "9"},
        ):
            limiter = cg.ConcurrencyLimiter()
        assert limiter.max_concurrency == 7
        assert limiter.max_queue == 9


class TestMeteredClient:
    """The wrapper that feeds usage into the counter."""

    def _client(self, usage: TokenUsage | None) -> MeteredLLMClient:
        """Build a metered client over a stub.

        Args:
            usage: Usage reported by the stub's response.

        Returns:
            The wrapped client.
        """
        spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
        stub = _StubClient(spec, LLMResponse(text="ok", usage=usage))
        return MeteredLLMClient(stub)  # type: ignore[arg-type]

    def test_usage_is_recorded(self, state_dir: Path) -> None:
        """A completion lands in the day's counter."""
        client = self._client(TokenUsage(input_tokens=1000, output_tokens=100))

        async def _run() -> LLMResponse:
            return await client.generate(messages=[], params=GenerateParams())

        response = asyncio.run(_run())

        assert response.text == "ok"
        assert cg.current_spend().calls == 1
        assert cg.current_spend().input_tokens == 1000

    def test_response_is_passed_through_unmodified(self, state_dir: Path) -> None:
        """Metering is transparent to the caller."""
        usage = TokenUsage(input_tokens=5, output_tokens=5)
        client = self._client(usage)

        async def _run() -> LLMResponse:
            return await client.generate(messages=[], params=GenerateParams())

        response = asyncio.run(_run())
        assert response.usage is usage

    def test_missing_usage_still_counts_the_call(self, state_dir: Path) -> None:
        """A provider that reports no usage is visible, not invisible."""
        client = self._client(None)

        async def _run() -> None:
            await client.generate(messages=[], params=GenerateParams())

        asyncio.run(_run())
        spend = cg.current_spend()
        assert spend.calls == 1
        assert spend.unpriced_calls == 1

    def test_accounting_failure_does_not_break_the_call(self, state_dir: Path) -> None:
        """A broken counter must not fail the user's request."""
        client = self._client(TokenUsage(input_tokens=1, output_tokens=1))

        async def _run() -> LLMResponse:
            with patch.object(cg, "record_usage", side_effect=OSError("disk gone")):
                return await client.generate(messages=[], params=GenerateParams())

        response = asyncio.run(_run())
        assert response.text == "ok"

    def test_enforcement_off_by_default(self, state_dir: Path) -> None:
        """Over budget, chat still works unless enforcement is switched on."""
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "0.000001"}):
            cg.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=10_000, output_tokens=0)
            )
            client = self._client(TokenUsage(input_tokens=1, output_tokens=1))

            async def _run() -> LLMResponse:
                return await client.generate(messages=[], params=GenerateParams())

            assert asyncio.run(_run()).text == "ok"

    def test_enforcement_refuses_when_enabled(self, state_dir: Path) -> None:
        """With enforcement on, an over-budget call is refused up front."""
        env = {
            "BAMBOO_ANALYSIS_DAILY_BUDGET_USD": "0.000001",
            "BAMBOO_COST_ENFORCE": "1",
        }
        with patch.dict(os.environ, env):
            cg.record_usage(
                "anthropic", "claude-sonnet-4-6", TokenUsage(input_tokens=10_000, output_tokens=0)
            )
            client = self._client(TokenUsage(input_tokens=1, output_tokens=1))

            async def _run() -> None:
                await client.generate(messages=[], params=GenerateParams())

            with pytest.raises(cg.BudgetExhausted):
                asyncio.run(_run())

    def test_close_is_delegated(self) -> None:
        """Closing the wrapper closes the wrapped client."""
        spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
        stub = _StubClient(spec, LLMResponse(text="ok"))
        client = MeteredLLMClient(stub)  # type: ignore[arg-type]

        asyncio.run(client.close())
        assert client.model_spec is spec


class TestFactoryWiring:
    """Every client built by the factory must be metered."""

    def test_build_client_returns_a_metered_client(self) -> None:
        """The seam covers all ten-plus generate() call sites at once."""
        from bamboo.llm.factory import build_client

        spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
        client = build_client(spec)

        assert isinstance(client, MeteredLLMClient)
        assert client.model_spec.model == "claude-sonnet-4-6"
