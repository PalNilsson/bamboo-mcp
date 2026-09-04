"""Lifecycle tests for :class:`~interfaces.shared.mcp_client.MCPClientSync`.

These tests cover the background event-loop thread rather than MCP protocol
behaviour: whether a failed or timed-out connection leaves a thread, a running
loop or an in-flight coroutine behind.

The failure being guarded against was observed in production: a ``connect()``
that exceeded the client deadline left its coroutine running on an orphaned
daemon thread, which then consumed a full CPU core for 17 hours because nothing
held a reference through which it could be closed.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any

import pytest

import httpx

from interfaces.shared.mcp_client import (
    DEFAULT_CLIENT_TIMEOUT_S,
    DEFAULT_HTTP_CONNECT_TIMEOUT_S,
    MCPAsyncClient,
    MCPClientSync,
    MCPServerConfig,
    _MCPSetupError,
    _default_client_timeout,
    _default_http_connect_timeout,
    _aclose_quietly,
    _connect_error,
    _env_timeout,
    _http_status_of,
    _is_protocol_error,
    _make_http_client_factory,
    _originating_error,
    _with_connect_timeout,
    _wrap_error,
)

LOOP_THREAD_NAME = "mcp-client-loop"


def _live_loop_threads() -> list[threading.Thread]:
    """Return every currently alive MCP client loop thread.

    Returns:
        Alive threads named :data:`LOOP_THREAD_NAME`.
    """
    return [t for t in threading.enumerate() if t.name == LOOP_THREAD_NAME and t.is_alive()]


@pytest.fixture(autouse=True)
def _no_leaked_loop_threads() -> Iterator[None]:
    """Fail any test in this module that leaves a loop thread behind.

    Yields:
        None.
    """
    before = len(_live_loop_threads())
    yield
    # Joining is the client's job; give a moment for an orderly exit only.
    for thread in _live_loop_threads():
        thread.join(timeout=1.0)
    after = len(_live_loop_threads())
    assert after <= before, f"leaked {after - before} '{LOOP_THREAD_NAME}' thread(s)"


@pytest.fixture
def cfg() -> MCPServerConfig:
    """Return a stdio config that is never actually connected.

    Returns:
        A default stdio :class:`MCPServerConfig`.
    """
    return MCPServerConfig(transport="stdio")


class TestEnvTimeout:
    """Tests for environment-driven timeout parsing."""

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not-a-number", "0", "-5", "1e", "None"],
        ids=["unset", "blank", "words", "zero", "negative", "partial", "none-literal"],
    )
    def test_unusable_values_fall_back(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """An unusable value falls back instead of raising.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            raw: Environment value under test.
        """
        monkeypatch.setenv("BAMBOO_TEST_TIMEOUT", raw)
        assert _env_timeout("BAMBOO_TEST_TIMEOUT", 300.0) == 300.0

    def test_valid_value_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A positive float is honoured.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("BAMBOO_TEST_TIMEOUT", "12.5")
        assert _env_timeout("BAMBOO_TEST_TIMEOUT", 300.0) == 12.5

    def test_client_timeout_typo_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed client timeout must not break every MCP call.

        The previous ``int(os.environ.get(...))`` raised :class:`ValueError` on
        each call rather than falling back.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("BAMBOO_MCP_CLIENT_TIMEOUT", "300s")
        assert _default_client_timeout() == DEFAULT_CLIENT_TIMEOUT_S

    def test_client_timeout_accepts_sub_second(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sub-second deadlines survive parsing.

        The previous integer cast floored these to zero.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("BAMBOO_MCP_CLIENT_TIMEOUT", "0.5")
        assert _default_client_timeout() == 0.5


class TestFailedInitTeardown:
    """Tests that a failed construction leaves nothing running."""

    def test_raising_connect_leaves_no_loop_thread(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``connect()`` that raises must not orphan the loop thread.

        This is the mutation target for the teardown guard: removing the
        ``except BaseException`` block in ``__init__`` must fail this test.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _boom(_self: Any) -> None:
            raise OSError("no such file or directory")

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _boom)

        before = len(_live_loop_threads())
        with pytest.raises(RuntimeError):
            MCPClientSync(cfg)

        assert len(_live_loop_threads()) == before

    def test_timed_out_connect_leaves_no_loop_thread(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``connect()`` that never returns must not orphan the loop thread.

        This is the exact production failure: the deadline expires, ``__init__``
        propagates, and the coroutine is left running on a daemon thread.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        started = threading.Event()

        async def _hang(_self: Any) -> None:
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _hang)
        monkeypatch.setenv("BAMBOO_MCP_CLIENT_TIMEOUT", "0.25")

        before = len(_live_loop_threads())
        with pytest.raises(RuntimeError, match="timed out"):
            MCPClientSync(cfg)

        assert started.is_set(), "connect() never reached the loop thread"
        assert len(_live_loop_threads()) == before

    def test_timed_out_call_cancels_the_coroutine(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Abandoning the future must also cancel the coroutine behind it.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        cancelled = threading.Event()

        async def _noop_connect(_self: Any) -> None:
            return None

        async def _hang(_self: Any) -> Any:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(
            "interfaces.shared.mcp_client.MCPAsyncClient.connect", _noop_connect
        )
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.list_tools", _hang)
        monkeypatch.setenv("BAMBOO_MCP_CLIENT_TIMEOUT", "0.25")

        client = MCPClientSync(cfg)
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                client.list_tools()
            assert cancelled.wait(timeout=2.0), "coroutine was never cancelled"
        finally:
            client.close()


class TestShutdown:
    """Tests for the shared loop-thread shutdown helper."""

    def test_close_stops_thread_and_closes_loop(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``close()`` joins the thread and closes the loop.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _noop(_self: Any) -> None:
            return None

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _noop)

        client = MCPClientSync(cfg)
        client.close()

        assert not client._thread.is_alive()
        assert client._loop.is_closed()

    def test_close_is_idempotent(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second ``close()`` must not raise on an already closed loop.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _noop(_self: Any) -> None:
            return None

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _noop)

        client = MCPClientSync(cfg)
        client.close()
        client.close()

    def test_lazy_client_closes_without_connecting(self, cfg: MCPServerConfig) -> None:
        """A never-connected client still shuts its loop thread down.

        Args:
            cfg: Server configuration fixture.
        """
        client = MCPClientSync(cfg, connect_on_init=False)
        client.close()

        assert not client._thread.is_alive()
        assert client._loop.is_closed()


class TestSingleTaskSessionLifecycle:
    """Tests that one task owns the session from open to close.

    anyio cancel scopes are task-affine.  Entering a transport in one task and
    exiting it in another raises ``Attempted to exit cancel scope in a different
    task`` and can leave the scope re-delivering cancellation forever, which is
    what burned a CPU core in production.
    """

    def test_connect_and_aclose_run_in_the_same_task(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``connect()`` and ``aclose()`` must share one asyncio task.

        This is the mutation target for the runner: routing ``aclose()`` back
        through ``_run`` as a fresh coroutine must fail this test.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        tasks: dict[str, object] = {}

        async def _connect(_self: Any) -> None:
            tasks["connect"] = asyncio.current_task()

        async def _aclose(_self: Any) -> None:
            tasks["aclose"] = asyncio.current_task()

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _connect)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _aclose)

        client = MCPClientSync(cfg)
        client.close()

        assert tasks["connect"] is not None
        assert tasks["aclose"] is tasks["connect"], "session was closed by a different task"

    def test_tool_calls_run_outside_the_owning_task(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tool calls stay independently cancellable, i.e. separate tasks.

        The runner owns the session lifetime only.  If tool calls were serialised
        onto the runner task they could not be cancelled without tearing the
        session down.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        tasks: dict[str, object] = {}

        async def _connect(_self: Any) -> None:
            tasks["connect"] = asyncio.current_task()

        async def _noop(_self: Any) -> None:
            return None

        async def _list_tools(_self: Any) -> str:
            tasks["call"] = asyncio.current_task()
            return "ok"

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _connect)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.list_tools", _list_tools)

        client = MCPClientSync(cfg)
        try:
            assert client.list_tools() == "ok"
            assert tasks["call"] is not tasks["connect"]
        finally:
            client.close()

    def test_session_is_closed_exactly_once(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated ``close()`` calls close the session once.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        calls = {"aclose": 0}

        async def _noop(_self: Any) -> None:
            return None

        async def _aclose(_self: Any) -> None:
            calls["aclose"] += 1

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _aclose)

        client = MCPClientSync(cfg)
        client.close()
        client.close()

        assert calls["aclose"] == 1

    def test_lazy_connect_starts_the_runner_on_first_call(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``connect_on_init=False`` the session opens on first use.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        calls = {"connect": 0}

        async def _connect(_self: Any) -> None:
            calls["connect"] += 1

        async def _noop(_self: Any) -> None:
            return None

        async def _list_tools(_self: Any) -> str:
            return "ok"

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _connect)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.list_tools", _list_tools)

        client = MCPClientSync(cfg, connect_on_init=False)
        try:
            assert calls["connect"] == 0
            client.list_tools()
            client.list_tools()
            assert calls["connect"] == 1, "session opened more than once"
        finally:
            client.close()

    def test_reuse_after_close_is_refused(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A closed client must not silently start a second loop.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _noop(_self: Any) -> None:
            return None

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _noop)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _noop)

        client = MCPClientSync(cfg)
        client.close()

        with pytest.raises(RuntimeError, match="closed"):
            client.ensure_connected()

    def test_startup_failure_reports_the_original_error(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connect failure surfaces its own message, not a readiness timeout.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _refused(_self: Any) -> None:
            raise ConnectionRefusedError("nobody home")

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _refused)

        with pytest.raises(RuntimeError, match="connection refused"):
            MCPClientSync(cfg)


class TestConnectTimeoutIsSeparate:
    """Tests that the TCP connect budget is independent of the read budget.

    A single ``httpx`` timeout value applies to connect as well as read, so
    raising the read budget to 300 s for long-running tools also made an
    unreachable endpoint take 300 s to report.
    """

    def test_scalar_timeout_gains_a_short_connect(self) -> None:
        """A scalar timeout keeps its read budget but gains a short connect."""
        resolved = _with_connect_timeout(300.0, 5.0)

        assert resolved.connect == 5.0
        assert resolved.read == 300.0

    def test_existing_phases_are_preserved(self) -> None:
        """Only the connect phase is replaced.

        The SDK derives read from ``sse_read_timeout``; shortening it would
        break long-lived streams.
        """
        original = httpx.Timeout(30.0, read=300.0, write=17.0, pool=11.0)

        resolved = _with_connect_timeout(original, 5.0)

        assert resolved.connect == 5.0
        assert resolved.read == 300.0
        assert resolved.write == 17.0
        assert resolved.pool == 11.0

    def test_none_timeout_still_bounds_connect(self) -> None:
        """An unlimited timeout still gets a bounded connect phase."""
        resolved = _with_connect_timeout(None, 5.0)

        assert resolved.connect == 5.0
        assert resolved.read is None

    def test_config_default_is_short_and_independent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The connect default does not track the per-call budget.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("BAMBOO_MCP_HTTP_TIMEOUT", "300")
        monkeypatch.delenv("BAMBOO_MCP_HTTP_CONNECT_TIMEOUT", raising=False)

        cfg = MCPServerConfig(transport="http")

        assert cfg.http_timeout_s == 300.0
        assert cfg.http_connect_timeout_s == DEFAULT_HTTP_CONNECT_TIMEOUT_S
        assert cfg.http_connect_timeout_s < cfg.http_timeout_s

    def test_connect_timeout_is_overridable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The connect budget can be raised for a high-latency network.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("BAMBOO_MCP_HTTP_CONNECT_TIMEOUT", "12")
        assert _default_http_connect_timeout() == 12.0

    def test_factory_matches_the_sdk_call_contract(self) -> None:
        """The injected factory works when called the way the SDK calls it.

        The mcp streamable-HTTP helper invokes
        ``httpx_client_factory(headers=..., timeout=..., auth=...)`` with
        ``httpx.Timeout(timeout, read=sse_read_timeout)``.  Reproduced here so a
        signature change in the SDK surfaces as a test failure rather than a
        connect that silently reverts to the read budget.
        """
        factory = _make_http_client_factory(5.0)

        client = factory(
            headers={"Authorization": "Bearer x"},
            timeout=httpx.Timeout(300.0, read=300.0),
            auth=None,
        )
        try:
            assert client.timeout.connect == 5.0
            assert client.timeout.read == 300.0
        finally:
            asyncio.run(client.aclose())


class TestPartialConnectIsUnwound:
    """Tests that a failed connect leaves no transport, session or client.

    Observed against mcp 1.29.1: an unreachable HTTP endpoint let the transport
    and session establish, then surfaced as ``CancelledError`` from inside the
    SDK's task group with both still attached. The caller saw an exception and
    had no reference through which to close them, and the async generator was
    later finalised by loop shutdown from a foreign task — raising
    ``Attempted to exit cancel scope in a different task``.
    """

    def test_failed_connect_leaves_no_residue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Everything established before the failure is detached and closed.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        closed: list[str] = []

        async def _establish_then_fail(self: Any, stack: AsyncExitStack) -> None:
            async def _record() -> None:
                closed.append("transport")

            stack.push_async_callback(_record)
            self._transport_cm = object()
            self._session = object()
            self._http_client = object()
            self.http_session_id = "abc123"
            raise ConnectionRefusedError("nobody home")

        monkeypatch.setattr(
            "interfaces.shared.mcp_client.MCPAsyncClient._connect_into", _establish_then_fail
        )

        client = MCPAsyncClient(MCPServerConfig(transport="http"))
        with pytest.raises(ConnectionRefusedError):
            asyncio.run(client.connect())

        assert closed == ["transport"], "the partial transport was not unwound"
        assert client._session is None
        assert client._transport_cm is None
        assert client._http_client is None
        assert client._stack is None
        assert client.http_session_id is None

    def test_aclose_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Closing twice unwinds once and does not raise.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        closed: list[str] = []

        async def _establish(self: Any, stack: AsyncExitStack) -> None:
            async def _record() -> None:
                closed.append("transport")

            stack.push_async_callback(_record)

        monkeypatch.setattr(
            "interfaces.shared.mcp_client.MCPAsyncClient._connect_into", _establish
        )

        async def _scenario() -> None:
            client = MCPAsyncClient(MCPServerConfig(transport="http"))
            await client.connect()
            await client.aclose()
            await client.aclose()

        asyncio.run(_scenario())

        assert closed == ["transport"]


class TestTeardownIsBestEffort:
    """Tests that one failing teardown stage does not skip the others.

    ``aclose`` previously ran its stages in sequence with no ``except``, so a
    session close that raised propagated immediately and the transport and HTTP
    client were never closed at all.
    """

    def test_every_stage_runs_when_an_earlier_one_raises(self) -> None:
        """A raising exit does not prevent the remaining exits."""
        ran: list[str] = []

        async def _ok() -> None:
            ran.append("transport")

        async def _boom() -> None:
            ran.append("session")
            raise RuntimeError("session close failed")

        async def _scenario() -> None:
            stack = AsyncExitStack()
            # LIFO: session unwinds first and raises, transport must still run.
            stack.push_async_callback(_ok)
            stack.push_async_callback(_boom)
            await _aclose_quietly(stack, "test teardown")

        asyncio.run(_scenario())

        assert ran == ["session", "transport"], "teardown stopped at the failure"

    def test_cancellation_during_teardown_is_not_propagated(self) -> None:
        """A wedged cancel scope surfaces as CancelledError and is swallowed.

        Teardown runs on failure paths and in ``finally`` blocks; re-raising
        would replace the error the caller actually needs.
        """
        async def _cancelled() -> None:
            raise asyncio.CancelledError("wedged scope")

        async def _scenario() -> None:
            stack = AsyncExitStack()
            stack.push_async_callback(_cancelled)
            await _aclose_quietly(stack, "test teardown")

        asyncio.run(_scenario())

    def test_failed_startup_closes_the_session(
        self, cfg: MCPServerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner closes any partial session before reporting failure.

        Args:
            cfg: Server configuration fixture.
            monkeypatch: Pytest monkeypatch fixture.
        """
        calls = {"aclose": 0}

        async def _boom(_self: Any) -> None:
            raise ConnectionRefusedError("nobody home")

        async def _aclose(_self: Any) -> None:
            calls["aclose"] += 1

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _boom)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _aclose)

        with pytest.raises(RuntimeError, match="connection refused"):
            MCPClientSync(cfg)

        assert calls["aclose"] == 1, "startup failure did not close the session"


class _DuckGroup(Exception):
    """A group-shaped exception for Python 3.10, which has no ``ExceptionGroup``.

    ``_originating_error`` duck-types on ``exceptions`` precisely so the 3.10
    floor in ``pyproject.toml`` is supported; this exercises that contract even
    on interpreters that do have the builtin.
    """

    exceptions: tuple[BaseException, ...] = ()


def _duck_group(*members: BaseException) -> _DuckGroup:
    """Return a duck-typed stand-in for an exception group.

    Args:
        *members: Exceptions the group should contain.

    Returns:
        A ``_DuckGroup`` carrying ``members``.
    """
    group = _DuckGroup("unhandled errors in a TaskGroup")
    group.exceptions = tuple(members)
    return group


def _group(*members: BaseException) -> BaseException:
    """Return a real exception group, matching what anyio raises.

    Args:
        *members: Exceptions the group should contain.

    Returns:
        A ``BaseExceptionGroup`` on Python 3.11+, otherwise a duck-typed
        stand-in with the same ``exceptions`` attribute.
    """
    try:
        return BaseExceptionGroup("unhandled errors in a TaskGroup", list(members))
    except NameError:  # pragma: no cover - Python 3.10 only
        return _duck_group(*members)


class TestOriginatingError:
    """Tests for unwrapping the error that actually explains a failure.

    Every HTTP transport failure leaves the SDK's task group as a contentless
    ``CancelledError``; the ``httpx`` error that caused it is only reachable
    inside the ``ExceptionGroup`` raised when the task group unwinds.
    """

    def test_a_single_member_group_yields_its_member(self) -> None:
        """The one real error inside a task-group failure is returned."""
        real = httpx.ConnectError("All connection attempts failed")

        assert _originating_error(_group(real)) is real

    def test_nesting_is_unwrapped(self) -> None:
        """Groups inside groups are traversed, as nested task groups produce."""
        real = httpx.ConnectError("All connection attempts failed")

        assert _originating_error(_group(_group(real))) is real

    def test_cancellations_are_skipped_in_favour_of_the_real_error(self) -> None:
        """A cancellation alongside a real error does not shadow it."""
        real = httpx.HTTPStatusError("401", request=None, response=None)  # type: ignore[arg-type]
        group = _group(asyncio.CancelledError("Cancelled via cancel scope"), real)

        assert _originating_error(group) is real

    def test_a_cancellation_only_group_yields_nothing(self) -> None:
        """Cancellation is a consequence, not an explanation."""
        group = _group(
            asyncio.CancelledError("Cancelled via cancel scope"),
            asyncio.CancelledError("Cancelled via cancel scope"),
        )

        assert _originating_error(group) is None

    def test_a_bare_cancellation_yields_nothing(self) -> None:
        """This is what a refused HTTP endpoint raises, and it explains nothing."""
        assert _originating_error(asyncio.CancelledError("Cancelled via cancel scope")) is None

    def test_an_ordinary_exception_is_its_own_explanation(self) -> None:
        """A non-group, non-cancellation exception is returned unchanged."""
        exc = ConnectionRefusedError("nobody home")

        assert _originating_error(exc) is exc

    def test_the_cause_chain_is_not_followed(self) -> None:
        """``httpx`` wraps ``httpcore``, and the ``httpx`` layer is the useful one."""
        inner = OSError("All connection attempts failed")
        outer = httpx.ConnectError("All connection attempts failed")
        outer.__cause__ = inner

        assert _originating_error(_group(outer)) is outer

    def test_duck_typed_groups_are_supported(self) -> None:
        """Groups are matched on ``exceptions``, for the Python 3.10 floor."""
        real = httpx.ConnectError("All connection attempts failed")

        assert _originating_error(_duck_group(real)) is real


class TestConnectRaisesTheRealError:
    """Tests that a failed connect reports the failure, not the cancellation.

    Confirmed against mcp 1.29.1 for connection refused, DNS failure and HTTP
    401, on both transport shapes: ``connect()`` used to raise
    ``CancelledError("Cancelled via cancel scope …")`` with no cause. That is
    unclassifiable, and being a ``BaseException`` it also slipped past the
    ``except Exception`` guards in ``scripts/bamboo_agent.py``.
    """

    @staticmethod
    def _failing_connect(
        raised: BaseException, rollback: BaseException | None
    ) -> Any:
        """Build a ``_connect_into`` replacement that fails during unwind too.

        Args:
            raised: Exception the connect attempt raises.
            rollback: Exception the stack unwind raises, if any.

        Returns:
            A coroutine function suitable for monkeypatching ``_connect_into``.
        """
        async def _connect_into(_self: Any, stack: AsyncExitStack) -> None:
            async def _unwind() -> None:
                if rollback is not None:
                    raise rollback

            stack.push_async_callback(_unwind)
            raise raised

        return _connect_into

    def _connect(
        self,
        monkeypatch: pytest.MonkeyPatch,
        raised: BaseException,
        rollback: BaseException | None,
    ) -> BaseException:
        """Run a failing connect and return whatever it raised.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            raised: Exception the connect attempt raises.
            rollback: Exception the stack unwind raises, if any.

        Returns:
            The exception that propagated out of ``connect()``.
        """
        monkeypatch.setattr(
            "interfaces.shared.mcp_client.MCPAsyncClient._connect_into",
            self._failing_connect(raised, rollback),
        )
        client = MCPAsyncClient(MCPServerConfig(transport="http"))
        try:
            asyncio.run(client.connect())
        except BaseException as exc:  # noqa: B036 - the exception is the assertion
            return exc
        raise AssertionError("connect() did not raise")

    def test_a_refused_endpoint_reports_the_connect_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``httpx.ConnectError`` recovered from the unwind is raised.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        real = httpx.ConnectError("All connection attempts failed")
        cancelled = asyncio.CancelledError("Cancelled via cancel scope 0x1")

        raised = self._connect(monkeypatch, cancelled, _group(real))

        assert raised is real, "connect() still reports the cancellation"
        assert isinstance(raised, Exception), "an `except Exception` guard would miss this"

    def test_the_cancellation_is_kept_as_the_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the SDK raised stays in the chain for the traceback.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        real = httpx.ConnectError("All connection attempts failed")
        cancelled = asyncio.CancelledError("Cancelled via cancel scope 0x1")

        raised = self._connect(monkeypatch, cancelled, _group(real))

        assert raised.__cause__ is cancelled

    def test_a_cancelled_unwind_leaves_the_cancellation_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Caller-driven cancellation must still propagate as cancellation.

        This is the case where ``_start_session`` or ``_run`` cancels the task:
        the unwind is cancelled too and recovers nothing, so there is no
        licence to convert the cancellation into an ordinary exception.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        cancelled = asyncio.CancelledError("Cancelled via cancel scope 0x1")

        raised = self._connect(
            monkeypatch, cancelled, _group(asyncio.CancelledError("cancelled unwind"))
        )

        assert raised is cancelled

    def test_a_clean_unwind_leaves_the_cancellation_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing recovered there is nothing better to report.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        cancelled = asyncio.CancelledError("Cancelled via cancel scope 0x1")

        raised = self._connect(monkeypatch, cancelled, None)

        assert raised is cancelled

    def test_an_informative_failure_is_not_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connect error that already explains itself wins over the unwind.

        The stdio path builds an actionable ``RuntimeError`` naming the command
        and args; a teardown error must not displace it.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        original = RuntimeError("Failed to start MCP server subprocess.")

        raised = self._connect(
            monkeypatch, original, _group(httpx.ConnectError("noise from teardown"))
        )

        assert raised is original

    def test_the_partial_connect_is_still_unwound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting the real error does not cost the C3b rollback guarantee.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        real = httpx.ConnectError("All connection attempts failed")
        client = MCPAsyncClient(MCPServerConfig(transport="http"))
        closed: list[str] = []

        async def _connect_into(self: Any, stack: AsyncExitStack) -> None:
            async def _record() -> None:
                closed.append("transport")
                raise _group(real)

            stack.push_async_callback(_record)
            self._transport_cm = object()
            self._session = object()
            raise asyncio.CancelledError("Cancelled via cancel scope 0x1")

        monkeypatch.setattr(
            "interfaces.shared.mcp_client.MCPAsyncClient._connect_into", _connect_into
        )

        with pytest.raises(httpx.ConnectError):
            asyncio.run(client.connect())

        assert closed == ["transport"], "the partial transport was not unwound"
        assert client._session is None
        assert client._transport_cm is None
        assert client._stack is None

    def test_the_unwind_error_is_still_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Promoting the error to the caller does not remove the log record.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            caplog: Pytest log capture fixture.
        """
        real = httpx.ConnectError("All connection attempts failed")

        with caplog.at_level("WARNING", logger="interfaces.shared.mcp_client"):
            self._connect(
                monkeypatch,
                asyncio.CancelledError("Cancelled via cancel scope 0x1"),
                _group(real),
            )

        assert any("rolling back a failed connect" in r.message for r in caplog.records)


class TestConnectErrorDecision:
    """Unit tests for the decision table in ``_connect_error``."""

    def test_an_informative_exception_is_returned_unchanged(self) -> None:
        """Nothing recovered from the unwind can improve on it."""
        exc = ConnectionRefusedError("nobody home")

        assert _connect_error(exc, _group(httpx.ConnectError("teardown noise"))) is exc

    def test_a_cancellation_is_replaced_by_the_recovered_error(self) -> None:
        """This is the refused-endpoint case."""
        real = httpx.ConnectError("All connection attempts failed")
        cancelled = asyncio.CancelledError("Cancelled via cancel scope")

        assert _connect_error(cancelled, _group(real)) is real

    def test_a_cancellation_survives_a_clean_unwind(self) -> None:
        """No rollback exception means nothing to promote."""
        cancelled = asyncio.CancelledError("Cancelled via cancel scope")

        assert _connect_error(cancelled, None) is cancelled

    def test_a_cancellation_survives_a_cancelled_unwind(self) -> None:
        """Cooperative cancellation is preserved."""
        cancelled = asyncio.CancelledError("Cancelled via cancel scope")
        rollback = _group(asyncio.CancelledError("cancelled unwind"))

        assert _connect_error(cancelled, rollback) is cancelled


class McpError(Exception):
    """Stand-in for the SDK's protocol error, matched by class name.

    ``interfaces.shared.mcp_client`` cannot import the real
    ``mcp.shared.exceptions.McpError``, because the stub ``mcp`` modules
    installed by ``tests/conftest.py`` leave ``mcp`` a non-package. Matching on
    the name is the documented contract, and this class exercises it.
    """


def _status_error(status: int) -> httpx.HTTPStatusError:
    """Build a real ``httpx.HTTPStatusError`` carrying ``status``.

    Args:
        status: HTTP status code the response should report.

    Returns:
        An ``HTTPStatusError`` with a genuine request and response attached.
    """
    request = httpx.Request("POST", "http://server.example/mcp")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


HTTP_CFG = MCPServerConfig(
    transport="http", http_url="http://server.example/mcp", http_connect_timeout_s=5.0
)
STDIO_CFG = MCPServerConfig(
    transport="stdio", stdio_command="/usr/bin/python3", stdio_args=["-m", "bamboo.server"]
)
SUBPROCESS_ADVICE = "python -m bamboo.server"


class TestHttpErrorsDoNotGiveSubprocessAdvice:
    """Tests that HTTP failures are reported as HTTP failures.

    Every HTTP transport failure used to arrive as a bare ``CancelledError``
    and be reported as *"This can happen during startup if the server
    subprocess exits immediately. Check that the server starts correctly:
    python -m bamboo.server"*. On this transport the server is a separate
    long-running process, usually on another host behind an SSH tunnel, so
    that advice is wrong and sends the reader to the wrong machine.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("All connection attempts failed"),
            ConnectionRefusedError("nobody home"),
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("read timed out"),
            _status_error(401),
            _status_error(503),
            McpError("Connection closed"),
            asyncio.CancelledError("Cancelled via cancel scope"),
            ValueError("something unforeseen"),
        ],
        ids=[
            "connect-error",
            "refused",
            "connect-timeout",
            "read-timeout",
            "401",
            "503",
            "protocol-error",
            "cancelled",
            "unclassified",
        ],
    )
    def test_no_http_failure_suggests_starting_a_subprocess(self, exc: BaseException) -> None:
        """No HTTP diagnosis tells the user to run the server locally.

        Args:
            exc: The originating failure under test.
        """
        message = str(_wrap_error(exc, HTTP_CFG))

        assert SUBPROCESS_ADVICE not in message, message
        assert HTTP_CFG.http_url in message, "the diagnosis does not name the endpoint"

    def test_a_refused_endpoint_is_named_as_unreachable(self) -> None:
        """The remedy is checking the server and the tunnel, not a restart."""
        message = str(_wrap_error(httpx.ConnectError("All connection attempts failed"), HTTP_CFG))

        assert "Cannot reach the MCP server" in message
        assert "SSH tunnel" in message

    def test_a_connect_timeout_names_its_own_budget(self) -> None:
        """The connect budget is separate from the per-call one, so name it."""
        message = str(_wrap_error(httpx.ConnectTimeout("timed out"), HTTP_CFG))

        assert "5 seconds" in message, message
        assert "BAMBOO_MCP_HTTP_CONNECT_TIMEOUT" in message

    def test_a_read_timeout_names_the_per_call_budget(self) -> None:
        """A stalled request is the other timeout, and the other variable."""
        message = str(_wrap_error(httpx.ReadTimeout("read timed out"), HTTP_CFG))

        assert "BAMBOO_MCP_HTTP_TIMEOUT" in message
        assert "BAMBOO_MCP_HTTP_CONNECT_TIMEOUT" not in message

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_rejected_request_points_at_the_token(self, status: int) -> None:
        """Bearer-token format is the usual cause and the usual mistake.

        Args:
            status: The rejection status code.
        """
        message = str(_wrap_error(_status_error(status), HTTP_CFG))

        assert f"HTTP {status}" in message
        assert "raw token value" in message

    def test_other_statuses_are_reported_without_guessing(self) -> None:
        """A 503 gets the status and the endpoint, and no invented remedy."""
        message = str(_wrap_error(_status_error(503), HTTP_CFG))

        assert "returned HTTP 503" in message
        assert "raw token value" not in message

    def test_a_protocol_error_says_the_handshake_did_not_finish(self) -> None:
        """Something answered, so reachability is not the problem."""
        message = str(_wrap_error(McpError("Connection closed"), HTTP_CFG))

        assert "did not complete the MCP handshake" in message

    def test_an_unexplained_cancellation_says_so(self) -> None:
        """With nothing recovered, the honest answer is that nothing is known.

        A cancellation still reaches here when the unwind recovered no cause —
        it must not be dressed up as a diagnosis.
        """
        message = str(_wrap_error(asyncio.CancelledError("Cancelled via cancel scope"), HTTP_CFG))

        assert "was cancelled, with no underlying error reported" in message

    def test_a_group_wrapped_failure_is_classified_through_the_group(self) -> None:
        """This is the shape the SDK's task group actually produces."""
        group = _group(httpx.ConnectError("All connection attempts failed"))

        message = str(_wrap_error(group, HTTP_CFG))

        assert "Cannot reach the MCP server" in message


class TestStdioErrorsKeepSubprocessAdvice:
    """Tests that stdio failures still say how to start the server.

    On stdio the server *is* a subprocess this client spawns, so the advice
    that is wrong for HTTP is right here.
    """

    def test_a_cancellation_keeps_the_startup_advice(self) -> None:
        """An immediately exiting subprocess is the usual cause here."""
        message = str(_wrap_error(asyncio.CancelledError("cancelled"), STDIO_CFG))

        assert SUBPROCESS_ADVICE in message

    def test_a_protocol_error_reports_the_command(self) -> None:
        """Knowing which command was spawned is what makes this actionable."""
        message = str(_wrap_error(McpError("Connection closed"), STDIO_CFG))

        assert "exited before the session was established" in message
        assert "/usr/bin/python3 -m bamboo.server" in message
        assert SUBPROCESS_ADVICE in message

    def test_a_refusal_keeps_its_wording(self) -> None:
        """Pre-existing behaviour on this path is preserved."""
        message = str(_wrap_error(ConnectionRefusedError("nobody home"), STDIO_CFG))

        assert "connection refused" in message

    def test_a_missing_executable_keeps_its_wording(self) -> None:
        """Pre-existing behaviour on this path is preserved."""
        message = str(_wrap_error(OSError("No such file or directory"), STDIO_CFG))

        assert "Make sure it's running" in message

    def test_an_unclassified_error_keeps_its_wording(self) -> None:
        """Pre-existing behaviour on this path is preserved."""
        message = str(_wrap_error(ValueError("something unforeseen"), STDIO_CFG))

        assert "Failed to create MCP client: ValueError" in message

    def test_no_config_falls_back_to_stdio_advice(self) -> None:
        """``MCPServerConfig`` defaults to stdio, so the advice should too."""
        message = str(_wrap_error(asyncio.CancelledError("cancelled"), None))

        assert SUBPROCESS_ADVICE in message


class TestActionableErrorsAreNotRewrapped:
    """Tests that this module does not wrap its own advice a second time.

    The stdio path builds a ``RuntimeError`` naming the command, the args and
    how to start the server; ``_wrap_error`` could not tell it from any other
    ``RuntimeError`` and produced ``"Failed to create MCP client: RuntimeError:
    Failed to connect to MCP server via stdio. …"`` with two sets of advice.
    """

    def test_a_setup_error_is_returned_unchanged(self) -> None:
        """Identity, not a copy: the message is already the right one."""
        original = _MCPSetupError("Failed to start MCP server subprocess.")

        assert _wrap_error(original, STDIO_CFG) is original

    def test_a_setup_error_survives_the_http_path_too(self) -> None:
        """"streamable_http_client is not available" needs no re-diagnosis."""
        original = _MCPSetupError("streamable_http_client is not available")

        assert _wrap_error(original, HTTP_CFG) is original

    def test_a_group_wrapped_setup_error_is_returned_unchanged(self) -> None:
        """A task group around it does not make it less actionable."""
        original = _MCPSetupError("MCP session not connected.")

        assert _wrap_error(_group(original), HTTP_CFG) is original

    def test_a_setup_error_is_still_a_runtime_error(self) -> None:
        """Existing ``except RuntimeError`` handlers must keep matching."""
        assert issubclass(_MCPSetupError, RuntimeError)


class TestClassificationHelpers:
    """Unit tests for the two predicates the classifier relies on."""

    def test_a_status_is_read_from_the_response(self) -> None:
        """The ordinary case, with a real response attached."""
        assert _http_status_of(_status_error(418)) == 418

    def test_a_missing_response_yields_no_status(self) -> None:
        """A hand-built error without a response must not raise."""
        assert _http_status_of(ValueError("no response here")) is None

    def test_a_non_integer_status_yields_no_status(self) -> None:
        """Anything unexpected on the response is ignored rather than shown."""
        exc = ValueError("odd")
        exc.response = type("R", (), {"status_code": "418"})()  # type: ignore[attr-defined]

        assert _http_status_of(exc) is None

    def test_a_protocol_error_is_matched_by_name(self) -> None:
        """The SDK class cannot be imported here, so the name is the contract."""
        assert _is_protocol_error(McpError("Connection closed"))

    def test_a_subclass_of_a_protocol_error_is_matched(self) -> None:
        """The MRO is walked, so SDK subclasses match too."""
        class Derived(McpError):
            """A subclass of the protocol error."""

        assert _is_protocol_error(Derived("Connection closed"))

    def test_an_unrelated_exception_is_not_matched(self) -> None:
        """The matcher is narrow: only that one name."""
        assert not _is_protocol_error(ValueError("unrelated"))


class TestTransportAwareErrorsReachTheCaller:
    """Tests that the transport-appropriate message survives the sync wrapper."""

    def test_a_failed_http_startup_reports_an_http_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``MCPClientSync`` construction over HTTP gives HTTP advice.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _refused(_self: Any) -> None:
            raise httpx.ConnectError("All connection attempts failed")

        async def _aclose(_self: Any) -> None:
            return None

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _refused)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _aclose)

        with pytest.raises(RuntimeError) as excinfo:
            MCPClientSync(HTTP_CFG)

        message = str(excinfo.value)
        assert "Cannot reach the MCP server" in message
        assert SUBPROCESS_ADVICE not in message, message

    def test_a_failed_http_call_reports_an_http_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A per-call failure is classified the same way as a startup failure.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        async def _ok(_self: Any) -> None:
            return None

        async def _boom(_self: Any) -> Any:
            raise httpx.ReadTimeout("read timed out")

        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.connect", _ok)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.aclose", _ok)
        monkeypatch.setattr("interfaces.shared.mcp_client.MCPAsyncClient.list_tools", _boom)

        client = MCPClientSync(HTTP_CFG)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                client.list_tools()
        finally:
            client.close()

        message = str(excinfo.value)
        assert "BAMBOO_MCP_HTTP_TIMEOUT" in message
        assert SUBPROCESS_ADVICE not in message, message
