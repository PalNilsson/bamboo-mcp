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
from typing import Any

import pytest

import httpx

from interfaces.shared.mcp_client import (
    DEFAULT_CLIENT_TIMEOUT_S,
    DEFAULT_HTTP_CONNECT_TIMEOUT_S,
    MCPClientSync,
    MCPServerConfig,
    _default_client_timeout,
    _default_http_connect_timeout,
    _env_timeout,
    _make_http_client_factory,
    _with_connect_timeout,
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
