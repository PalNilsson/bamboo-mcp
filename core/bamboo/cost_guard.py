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

"""Spend accounting and admission control for LLM-backed work.

Why this module exists
----------------------
An "Analyze failure" button on every failed job page in the PanDA monitor
turns LLM spend from something a handful of chat users generate into something
a page view generates.  Two limits are needed before that button ships: a
ceiling on how much can be spent in a day, and a ceiling on how many analyses
run at once.  Neither existed.

Accounting, not enforcement
---------------------------
This module records what was spent and answers whether a *new* piece of work
should be admitted.  It does not abort work in flight.  Enforcement is a
decision for the caller, and belongs at admission time — the REST facade calls
:func:`check_budget` before starting an analysis and returns a 429 rather than
letting a request run and fail halfway.  Interactive chat is deliberately not
gated by default: cutting a user off mid-conversation to save a fraction of a
cent is worse than the overspend.  Set ``BAMBOO_COST_ENFORCE=1`` to have the
metered client refuse calls once the budget is gone.

Where the numbers come from
---------------------------
``bamboo.llm.types.TokenUsage`` is already normalised across providers and
hangs off every ``LLMResponse``.  Until now it was read only for tracing spans
and then discarded, so the prompt-log's ``input_tokens`` and ``output_tokens``
fields were always null.  :class:`bamboo.llm.metered.MeteredLLMClient` wraps
every client built by ``bamboo.llm.factory.build_client`` and feeds the usage
here, which is why the count covers planner and topic-guard calls and not only
synthesis.

Prices are per million tokens and **must be verified against the provider's
pricing page before a budget is enforced**.  The built-in table is a starting
point, not an authority: model prices change without any signal reaching this
repository.  ``BAMBOO_MODEL_PRICES`` overrides it without a release.  A model
with no price is still counted in tokens, and its calls are counted in
``unpriced_calls``, so an unpriced model shows up as a visible gap rather than
as free.

Persistence
-----------
The daily counter is a JSON file per UTC date, updated under ``flock``.  A file
rather than a process global for three reasons: a restart must not reset the
day's spend; the core-dump analyzer builds its own client in a *detached
worker process* and its spend has to land in the same counter; and multiple
uvicorn workers, if ever enabled, must share one budget.

Environment variables
---------------------
``BAMBOO_COST_STATE_ROOT``
    Directory holding the per-day counter files (default:
    ``/tmp/bamboo/cost``).  Note that ``/tmp`` does not survive a reboot; set
    this to a persistent path on a deployment where the budget matters.

``BAMBOO_ANALYSIS_DAILY_BUDGET_USD``
    Daily ceiling in USD.  ``0`` (the default) disables the budget entirely,
    leaving accounting active.

``BAMBOO_MODEL_PRICES``
    JSON object overriding or extending the price table, e.g.
    ``{"anthropic/claude-sonnet-4-6": [3.0, 15.0]}`` — input and output USD per
    million tokens.

``BAMBOO_COST_ENFORCE``
    When ``1``, the metered client raises :class:`BudgetExhausted` once the
    budget is gone, instead of only recording.  Off by default.

``BAMBOO_ANALYSIS_MAX_CONCURRENCY``
    Concurrent analyses admitted by :class:`ConcurrencyLimiter` (default: 4).

``BAMBOO_ANALYSIS_MAX_QUEUE``
    Callers allowed to wait for a slot before admission is refused
    (default: 20).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bamboo.llm.types import TokenUsage

try:  # pragma: no cover - POSIX only, and Bamboo targets Linux and macOS.
    import fcntl

    _HAVE_FCNTL: bool = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False

logger = logging.getLogger(__name__)

#: Directory holding per-day counter files when the environment is unset.
#: Mirrors the ``/tmp/bamboo/...`` convention used by the core-dump analyzer.
DEFAULT_STATE_ROOT: str = "/tmp/bamboo/cost"

#: Concurrent analyses admitted when the environment is unset.
DEFAULT_MAX_CONCURRENCY: int = 4

#: Waiters allowed to queue for a slot when the environment is unset.
DEFAULT_MAX_QUEUE: int = 20

#: USD per *million* tokens as ``"provider/model": (input, output)``.
#:
#: Verify these against the provider's pricing page before enabling a budget.
#: Lookup is exact first, then longest matching prefix, so a family entry such
#: as ``"anthropic/claude-opus-4"`` covers its point releases.
DEFAULT_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-haiku-4": (1.00, 5.00),
    "anthropic/claude-opus-4": (15.00, 75.00),
}


class BudgetExhausted(RuntimeError):
    """Raised when a call is refused because the daily budget is spent."""


class AdmissionRefused(RuntimeError):
    """Raised when a caller is refused a concurrency slot.

    Attributes:
        retry_after_s: Suggested wait before retrying, in seconds.
    """

    def __init__(self, message: str, retry_after_s: float = 5.0) -> None:
        """Record the refusal and the suggested retry delay.

        Args:
            message: Human-readable reason.
            retry_after_s: Suggested wait before retrying, in seconds.
        """
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class DailySpend:
    """Accumulated spend for one UTC day.

    Attributes:
        date: UTC date as ``YYYY-MM-DD``.
        calls: Completed LLM calls recorded.
        unpriced_calls: Calls whose model had no price entry.  Their tokens are
            counted, their dollars are not, so a non-zero value means ``usd``
            understates the true spend.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
        usd: Total priced spend in USD.
    """

    date: str
    calls: int = 0
    unpriced_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Returns:
            Dict with one key per field.
        """
        return {
            "date": self.date,
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
        }


@dataclass(frozen=True)
class BudgetStatus:
    """Verdict on whether new work should be admitted.

    Attributes:
        allowed: Whether the caller may proceed.
        reason: Machine-readable reason when refused, otherwise empty.
        spent_usd: Spend so far today.
        budget_usd: Configured ceiling, ``0.0`` when no budget is set.
        retry_after_s: Seconds until the budget resets, when refused.
    """

    allowed: bool
    reason: str = ""
    spent_usd: float = 0.0
    budget_usd: float = 0.0
    retry_after_s: float = 0.0


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or not positive.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    """Read a non-negative float from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or negative.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def state_root() -> Path:
    """Return the directory holding per-day counter files.

    Returns:
        ``$BAMBOO_COST_STATE_ROOT``, or :data:`DEFAULT_STATE_ROOT`.
    """
    return Path(os.getenv("BAMBOO_COST_STATE_ROOT") or DEFAULT_STATE_ROOT)


def daily_budget_usd() -> float:
    """Return the configured daily ceiling in USD.

    Returns:
        ``$BAMBOO_ANALYSIS_DAILY_BUDGET_USD``, or ``0.0`` when no budget is
        configured.  Zero means accounting continues but nothing is refused.
    """
    return _env_float("BAMBOO_ANALYSIS_DAILY_BUDGET_USD", 0.0)


def enforcement_enabled() -> bool:
    """Return whether the metered client should refuse calls over budget.

    Returns:
        ``True`` when ``BAMBOO_COST_ENFORCE`` is a truthy value.
    """
    return os.getenv("BAMBOO_COST_ENFORCE", "").strip().lower() in {"1", "true", "yes", "on"}


def today_utc() -> str:
    """Return the current UTC date as ``YYYY-MM-DD``.

    Returns:
        The date string used to name the day's counter file.
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def seconds_until_utc_midnight() -> float:
    """Return the seconds remaining until the budget resets.

    Returns:
        Seconds until the next UTC midnight, always positive.
    """
    now = datetime.now(tz=timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1.0, (tomorrow - now).total_seconds())


def price_table() -> dict[str, tuple[float, float]]:
    """Return the effective price table.

    The built-in table is overlaid with ``BAMBOO_MODEL_PRICES`` when that
    variable holds a JSON object of ``"provider/model": [input, output]``
    entries.  A malformed value is ignored with a warning rather than raising,
    so a bad override cannot stop the server from starting.

    Returns:
        Mapping of ``"provider/model"`` to ``(input_usd, output_usd)`` per
        million tokens.
    """
    table = dict(DEFAULT_MODEL_PRICES)
    raw = os.getenv("BAMBOO_MODEL_PRICES", "").strip()
    if not raw:
        return table

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("cost_guard: BAMBOO_MODEL_PRICES is not valid JSON: %s", exc)
        return table

    if not isinstance(parsed, dict):
        logger.warning("cost_guard: BAMBOO_MODEL_PRICES must be a JSON object.")
        return table

    for key, value in parsed.items():
        try:
            in_usd, out_usd = value
            table[str(key)] = (float(in_usd), float(out_usd))
        except (TypeError, ValueError):
            logger.warning(
                "cost_guard: ignoring malformed price entry for %r: %r", key, value
            )
    return table


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    """Return the price pair for a model, or ``None`` when unpriced.

    Args:
        provider: Provider string, e.g. ``"anthropic"``.
        model: Model string, e.g. ``"claude-sonnet-4-6"``.

    Returns:
        ``(input_usd, output_usd)`` per million tokens, or ``None`` when
        neither an exact nor a prefix entry matches.
    """
    table = price_table()
    key = f"{provider}/{model}"
    exact = table.get(key)
    if exact is not None:
        return exact

    candidates = [k for k in table if key.startswith(k)]
    if not candidates:
        return None
    return table[max(candidates, key=len)]


def usd_for(provider: str, model: str, usage: TokenUsage | None) -> float | None:
    """Return the cost of one call in USD.

    Args:
        provider: Provider string.
        model: Model string.
        usage: Normalised token usage, or ``None`` when the provider did not
            report any.

    Returns:
        Cost in USD, or ``None`` when the model is unpriced or usage is
        missing.
    """
    if usage is None:
        return None
    prices = price_for(provider, model)
    if prices is None:
        return None
    in_usd, out_usd = prices
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    return (input_tokens * in_usd + output_tokens * out_usd) / 1_000_000.0


def _counter_path(date: str | None = None) -> Path:
    """Return the counter file path for a UTC date.

    Args:
        date: Date as ``YYYY-MM-DD``, or ``None`` for today.

    Returns:
        Path to the JSON counter file.
    """
    return state_root() / f"{date or today_utc()}.json"


def _parse_state(text: str, date: str) -> DailySpend:
    """Parse a counter file body, tolerating corruption.

    A truncated or hand-edited file resets the day's count rather than raising:
    losing today's accumulated total is a smaller failure than refusing to
    serve because a JSON file is malformed.

    Args:
        text: File contents.
        date: UTC date the file belongs to.

    Returns:
        The parsed spend, or a zeroed record.
    """
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        logger.warning("cost_guard: counter file for %s is corrupt; resetting.", date)
        return DailySpend(date=date)

    if not isinstance(data, dict) or data.get("date") != date:
        return DailySpend(date=date)

    try:
        return DailySpend(
            date=date,
            calls=int(data.get("calls", 0)),
            unpriced_calls=int(data.get("unpriced_calls", 0)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            usd=float(data.get("usd", 0.0)),
        )
    except (TypeError, ValueError):
        logger.warning("cost_guard: counter file for %s has bad fields; resetting.", date)
        return DailySpend(date=date)


@contextlib.contextmanager
def _locked_counter(date: str) -> Iterator[Any]:
    """Open the day's counter file with an exclusive lock held.

    Args:
        date: UTC date as ``YYYY-MM-DD``.

    Yields:
        The open file handle, positioned at the start.
    """
    path = _counter_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - closed in finally
    try:
        if _HAVE_FCNTL:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        yield handle
    finally:
        try:
            if _HAVE_FCNTL:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def current_spend(date: str | None = None) -> DailySpend:
    """Return the recorded spend for a UTC day.

    Args:
        date: Date as ``YYYY-MM-DD``, or ``None`` for today.

    Returns:
        The day's accumulated spend; a zeroed record when nothing is recorded.
    """
    day = date or today_utc()
    path = _counter_path(day)
    if not path.exists():
        return DailySpend(date=day)
    try:
        return _parse_state(path.read_text(encoding="utf-8"), day)
    except OSError as exc:
        logger.warning("cost_guard: cannot read counter for %s: %s", day, exc)
        return DailySpend(date=day)


def record_usage(
    provider: str,
    model: str,
    usage: TokenUsage | None,
) -> DailySpend:
    """Add one call's usage to the day's counter.

    Never raises on a storage failure: dropping an accounting record must not
    fail the user-visible call that produced it.  A failure is logged and the
    unmodified in-memory view is returned.

    Args:
        provider: Provider string.
        model: Model string.
        usage: Normalised token usage, or ``None`` when unavailable.

    Returns:
        The day's spend after this call.
    """
    day = today_utc()
    cost = usd_for(provider, model, usage)
    input_tokens = (usage.input_tokens or 0) if usage else 0
    output_tokens = (usage.output_tokens or 0) if usage else 0

    try:
        with _locked_counter(day) as handle:
            state = _parse_state(handle.read(), day)
            updated = replace(
                state,
                calls=state.calls + 1,
                unpriced_calls=state.unpriced_calls + (1 if cost is None else 0),
                input_tokens=state.input_tokens + input_tokens,
                output_tokens=state.output_tokens + output_tokens,
                usd=state.usd + (cost or 0.0),
            )
            handle.seek(0)
            handle.truncate()
            json.dump(updated.as_dict(), handle)
            handle.flush()
        if cost is None:
            _warn_unpriced(provider, model)
        return updated
    except OSError as exc:
        logger.warning("cost_guard: cannot record usage for %s/%s: %s", provider, model, exc)
        return current_spend(day)


#: Models already warned about, so an unpriced model logs once per process
#: rather than once per call.
_warned_unpriced: set[str] = set()


def _warn_unpriced(provider: str, model: str) -> None:
    """Log a one-off warning for a model with no price entry.

    Args:
        provider: Provider string.
        model: Model string.
    """
    key = f"{provider}/{model}"
    if key in _warned_unpriced:
        return
    _warned_unpriced.add(key)
    logger.warning(
        "cost_guard: no price for %s; its tokens are counted but its cost is "
        "not. Add an entry to BAMBOO_MODEL_PRICES to include it in the budget.",
        key,
    )


def check_budget() -> BudgetStatus:
    """Return whether new work should be admitted against the daily budget.

    Call this at admission time, before starting work, so a refusal is a clean
    early answer rather than a failure halfway through an analysis.

    Returns:
        A :class:`BudgetStatus`.  Always allowed when no budget is configured.
    """
    budget = daily_budget_usd()
    spend = current_spend()
    if budget <= 0:
        return BudgetStatus(allowed=True, spent_usd=spend.usd, budget_usd=0.0)

    if spend.usd >= budget:
        return BudgetStatus(
            allowed=False,
            reason="budget_exhausted",
            spent_usd=spend.usd,
            budget_usd=budget,
            retry_after_s=seconds_until_utc_midnight(),
        )
    return BudgetStatus(allowed=True, spent_usd=spend.usd, budget_usd=budget)


class ConcurrencyLimiter:
    """Bounds concurrent work and the queue waiting for it.

    A semaphore alone gives an unbounded queue, which converts a spike into a
    slow-motion outage: every caller waits, none is told to go away, and the
    ones at the back have long since given up by the time they are served.  A
    queue cap turns the tail into an immediate, actionable refusal instead.

    The semaphore is created lazily and rebuilt if the running event loop
    changes, so an instance constructed at import time is safe to use from any
    loop.

    Attributes:
        max_concurrency: Slots available at once.
        max_queue: Callers allowed to wait for a slot.
    """

    def __init__(self, max_concurrency: int | None = None, max_queue: int | None = None) -> None:
        """Configure the limiter.

        Args:
            max_concurrency: Slots available at once.  ``None`` reads
                ``BAMBOO_ANALYSIS_MAX_CONCURRENCY``.
            max_queue: Callers allowed to wait.  ``None`` reads
                ``BAMBOO_ANALYSIS_MAX_QUEUE``.
        """
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else _env_int("BAMBOO_ANALYSIS_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        self.max_queue = (
            max_queue
            if max_queue is not None
            else _env_int("BAMBOO_ANALYSIS_MAX_QUEUE", DEFAULT_MAX_QUEUE)
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._waiting: int = 0
        self._in_flight: int = 0

    @property
    def waiting(self) -> int:
        """Return the number of callers currently queued for a slot.

        Returns:
            Queue depth.
        """
        return self._waiting

    @property
    def in_flight(self) -> int:
        """Return the number of slots currently held.

        Returns:
            Active count.
        """
        return self._in_flight

    def _sem(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the running loop.

        Returns:
            The semaphore, rebuilt if the loop changed since it was created.
        """
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._loop = loop
            self._waiting = 0
            self._in_flight = 0
        return self._semaphore

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold a slot for the duration of the block, or refuse admission.

        Yields:
            ``None`` once a slot is held.

        Raises:
            AdmissionRefused: If the queue is already at ``max_queue``.
        """
        semaphore = self._sem()
        if self._waiting >= self.max_queue:
            raise AdmissionRefused(
                f"{self._waiting} request(s) already waiting for one of "
                f"{self.max_concurrency} slots; try again shortly.",
                retry_after_s=5.0,
            )

        self._waiting += 1
        try:
            await semaphore.acquire()
        finally:
            self._waiting -= 1

        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1
            semaphore.release()


__all__ = [
    "AdmissionRefused",
    "BudgetExhausted",
    "BudgetStatus",
    "ConcurrencyLimiter",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_QUEUE",
    "DEFAULT_MODEL_PRICES",
    "DEFAULT_STATE_ROOT",
    "DailySpend",
    "check_budget",
    "current_spend",
    "daily_budget_usd",
    "enforcement_enabled",
    "price_for",
    "price_table",
    "record_usage",
    "seconds_until_utc_midnight",
    "state_root",
    "today_utc",
    "usd_for",
]
