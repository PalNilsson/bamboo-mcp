"""
Shared MCP client helpers for AskPanDA interfaces.

This module provides a Streamlit-friendly synchronous wrapper around the MCP async
client, supporting both:

  - STDIO transport (dev): spawns a local MCP server subprocess
  - Streamable HTTP transport (prod): connects to an MCP server endpoint URL

Why a background event loop thread?
Streamlit runs scripts in a managed thread and may interrupt execution during reruns.
Running the MCP async session on a dedicated event-loop thread and calling it via
`run_coroutine_threadsafe` avoids cancellation issues and makes the UI stable.

Compatible with MCP streamable HTTP client signature:

  streamable_http_client(url, *, http_client=None, terminate_on_close=True)
    -> async generator yielding (read_stream, write_stream, get_session_id_callback)
"""

from __future__ import annotations

import asyncio
import os
import concurrent.futures
import logging
import subprocess
import sys
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import httpx

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

# At runtime we import dynamically (some environments may not provide the helper).
# For static type checkers, import the symbol under TYPE_CHECKING so pyright can see it.
if TYPE_CHECKING:
    try:
        from mcp.client.streamable_http import streamable_http_client  # type: ignore
    except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover - only for static analysis
        streamable_http_client = None  # type: ignore

streamable_http_client: Any = None


def _merge_stdio_env(stdio_env: dict[str, str] | None) -> dict[str, str] | None:
    """Merge stdio env overrides over the parent process environment.

    Inline Textual UIs are sensitive to stray output from child processes. While we
    cannot fully control the stdio transport internals here, inheriting the parent
    environment ensures expected config (LLM defaults, plugin selection, etc.) is
    visible to the server process.

    Args:
        stdio_env: Optional environment overrides.

    Returns:
        A merged environment dict, or None if no env should be passed.
    """
    env = os.environ.copy()
    if stdio_env:
        env.update({str(k): str(v) for k, v in stdio_env.items()})
    # Make Python child process unbuffered to reduce partial protocol writes.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # If the server supports file logging, encourage it to avoid writing to stdout.
    env.setdefault("BAMBOO_LOG_TO_FILE", "1")
    return env


TransportType = Literal["stdio", "http"]

#: Default per-call HTTP transport timeout, in seconds, overridable with
#: ``BAMBOO_MCP_HTTP_TIMEOUT``.
#:
#: This is a per-tool-call deadline, not a connection timeout, so it has to
#: accommodate the slowest tool rather than the typical one.  It was 30 s, which
#: predates any long-running tool; ``atlas.core_dump_analysis`` waits inline for
#: a gdb run over a multi-gigabyte core and would have been killed mid-call on
#: the HTTP deployment while the detached worker carried on, leaving the user a
#: transport error instead of an answer.
#:
#: Keep this comfortably above ``BAMBOO_CORE_ANALYSIS_INLINE_WAIT`` plus the
#: synthesis LLM call, and keep ``BAMBOO_MCP_CLIENT_TIMEOUT`` (the client-side
#: future deadline, default 120 s) above *this* — the two are independent
#: ceilings and the lower one wins.
DEFAULT_HTTP_TIMEOUT_S: float = 300.0

#: Default client-side deadline, in seconds, for waiting on a single MCP call,
#: overridable with ``BAMBOO_MCP_CLIENT_TIMEOUT``.  Kept equal to
#: :data:`DEFAULT_HTTP_TIMEOUT_S` because the two are independent ceilings on
#: the same call and the lower one silently wins.
DEFAULT_CLIENT_TIMEOUT_S: float = 300.0


def _env_timeout(var: str, default: float) -> float:
    """Read a positive timeout, in seconds, from an environment variable.

    An unset, blank, unparsable or non-positive value falls back to ``default``
    rather than raising.  A typo in a deployment environment must not be able to
    stop the client from starting, nor to turn every subsequent MCP call into a
    :class:`ValueError`.

    Args:
        var: Environment variable name to read.
        default: Value to fall back to.

    Returns:
        The parsed timeout in seconds, or ``default``.
    """
    raw = os.environ.get(var, "")
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _default_http_timeout() -> float:
    """Resolve the HTTP transport timeout from the environment.

    Returns:
        ``BAMBOO_MCP_HTTP_TIMEOUT`` as a float, or
        :data:`DEFAULT_HTTP_TIMEOUT_S` when it is unset or unparsable.
    """
    return _env_timeout("BAMBOO_MCP_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT_S)


def _default_client_timeout() -> float:
    """Resolve the client-side per-call deadline from the environment.

    Returns:
        ``BAMBOO_MCP_CLIENT_TIMEOUT`` as a float, or
        :data:`DEFAULT_CLIENT_TIMEOUT_S` when it is unset or unparsable.
    """
    return _env_timeout("BAMBOO_MCP_CLIENT_TIMEOUT", DEFAULT_CLIENT_TIMEOUT_S)


#: Seconds to wait for the session runner task to finish closing the session.
#:
#: Deliberately short and *not* derived from the per-call deadline: a close is
#: usually driven by a user action (a Streamlit reconnect, an interpreter exit)
#: and must not appear to hang for the several minutes a slow tool call is
#: allowed.  Overrunning it is logged and the loop thread is torn down anyway.
SHUTDOWN_TIMEOUT_S: float = 10.0


#: Default TCP connect timeout, in seconds, overridable with
#: ``BAMBOO_MCP_HTTP_CONNECT_TIMEOUT``.
#:
#: Kept small and independent of :data:`DEFAULT_HTTP_TIMEOUT_S`.  A single
#: ``httpx`` timeout value applies to connect as well as read, so raising the
#: read budget for slow tools also made "nothing is listening on this port"
#: take the full read budget to report.  Establishing a TCP connection to a
#: live server is a sub-second operation on any network this client runs on, so
#: a refusal or a black hole should surface in seconds.
DEFAULT_HTTP_CONNECT_TIMEOUT_S: float = 5.0


def _default_http_connect_timeout() -> float:
    """Resolve the HTTP connect timeout from the environment.

    Returns:
        ``BAMBOO_MCP_HTTP_CONNECT_TIMEOUT`` as a float, or
        :data:`DEFAULT_HTTP_CONNECT_TIMEOUT_S` when unset or unparsable.
    """
    return _env_timeout("BAMBOO_MCP_HTTP_CONNECT_TIMEOUT", DEFAULT_HTTP_CONNECT_TIMEOUT_S)


def _with_connect_timeout(
    timeout: httpx.Timeout | float | None, connect_timeout_s: float
) -> httpx.Timeout:
    """Return ``timeout`` with only its connect budget replaced.

    The read, write and pool budgets are preserved deliberately: the MCP SDK
    derives read from ``sse_read_timeout`` and shortening it would break
    long-lived streams.

    Args:
        timeout: An existing ``httpx`` timeout, a scalar applied to every
            phase, or None for no limit.
        connect_timeout_s: Connect budget to apply, in seconds.

    Returns:
        A timeout whose connect phase is ``connect_timeout_s``.
    """
    if isinstance(timeout, httpx.Timeout):
        return httpx.Timeout(
            connect=connect_timeout_s,
            read=timeout.read,
            write=timeout.write,
            pool=timeout.pool,
        )
    return httpx.Timeout(timeout, connect=connect_timeout_s)


def _make_http_client_factory(connect_timeout_s: float) -> Any:
    """Build an ``httpx_client_factory`` that caps the connect timeout.

    The MCP streamable-HTTP helper constructs its own client from a factory it
    calls with ``headers``, ``timeout`` and ``auth`` keywords, passing
    ``httpx.Timeout(timeout, read=sse_read_timeout)`` — which leaves connect
    equal to the per-call budget.  This wrapper overrides connect and delegates
    to the SDK's own factory where available, so any other client options it
    sets are preserved.

    Args:
        connect_timeout_s: Connect budget to apply, in seconds.

    Returns:
        A callable suitable for the SDK's ``httpx_client_factory`` parameter.
    """
    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
        auth: Any = None,
    ) -> httpx.AsyncClient:
        """Create an ``httpx.AsyncClient`` with a bounded connect timeout.

        Args:
            headers: Headers supplied by the SDK.
            timeout: Timeout computed by the SDK.
            auth: Auth object supplied by the SDK.

        Returns:
            A configured async HTTP client.
        """
        resolved = _with_connect_timeout(timeout, connect_timeout_s)
        try:
            import importlib  # pylint: disable=import-outside-toplevel

            base = getattr(
                importlib.import_module("mcp.shared._httpx_utils"),
                "create_mcp_http_client",
                None,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            base = None
        if base is not None:
            return base(headers=headers, timeout=resolved, auth=auth)
        return httpx.AsyncClient(
            headers=headers, timeout=resolved, auth=auth, follow_redirects=True
        )

    return factory


def _wrap_error(exc: BaseException) -> RuntimeError:
    """Map a transport or session exception onto an actionable RuntimeError.

    Shared by the per-call path (:meth:`MCPClientSync._run`) and the session
    startup path (:meth:`MCPClientSync._start_session`) so both report a
    failure the same way.

    Args:
        exc: The exception raised on the loop thread.

    Returns:
        A ``RuntimeError`` carrying ``exc`` as its cause.
    """
    if isinstance(exc, (asyncio.CancelledError, concurrent.futures.CancelledError)):
        return RuntimeError(
            "MCP server connection was cancelled.\n"
            "This can happen during startup if the server subprocess exits immediately.\n"
            "Check that the server starts correctly:\n"
            "  python -m bamboo.server\n"
            f"Original error: {type(exc).__name__}"
        )
    if isinstance(exc, ConnectionRefusedError):
        return RuntimeError(
            "Failed to connect to MCP server (connection refused).\n"
            "Ensure the server is running:\n"
            "  python -m bamboo.server\n"
            f"Original error: {exc}"
        )
    if isinstance(exc, OSError):
        if "Connection refused" in str(exc) or "No such file or directory" in str(exc):
            return RuntimeError(
                "Cannot connect to MCP server. Make sure it's running:\n"
                "  python -m bamboo.server\n"
                f"Original error: {exc}"
            )
        return RuntimeError(str(exc))
    return RuntimeError(
        f"Failed to create MCP client: {type(exc).__name__}: {exc}\n"
        "Is the MCP server running?\n"
        "  python -m bamboo.server"
    )


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    Attributes:
        transport: "stdio" (spawn local server) or "http" (connect to HTTP endpoint).
        stdio_command: Executable for stdio server (typically sys.executable).
        stdio_args: Args for stdio server (e.g., ["-m", "bamboo.server"]).
        stdio_env: Optional environment overrides for stdio server (merged over parent env).
        http_url: Streamable HTTP endpoint URL (e.g., "http://localhost:8000/mcp").
        http_headers: Optional headers (auth, etc.) for HTTP transport.
        terminate_on_close: If True, sends DELETE to terminate session on close.
        http_timeout_s: Per-call HTTP client timeout (seconds).  Defaults from
            ``BAMBOO_MCP_HTTP_TIMEOUT``; see :data:`DEFAULT_HTTP_TIMEOUT_S`.
        http_connect_timeout_s: TCP connect timeout (seconds), kept separate so
            that a refused or unreachable endpoint fails fast even when the
            per-call read budget is minutes.  Defaults from
            ``BAMBOO_MCP_HTTP_CONNECT_TIMEOUT``.
    """

    transport: TransportType = "stdio"

    # stdio options
    stdio_command: str = field(default_factory=lambda: sys.executable)
    stdio_args: list[str] = field(default_factory=lambda: ["-m", "bamboo.server"])
    stdio_env: dict[str, str] | None = None

    # http options
    http_url: str = "http://localhost:8000/mcp"
    http_headers: dict[str, str] | None = None
    terminate_on_close: bool = True
    http_timeout_s: float = field(default_factory=_default_http_timeout)
    http_connect_timeout_s: float = field(default_factory=_default_http_connect_timeout)


class MCPAsyncClient:
    """Async MCP client with stdio and HTTP transports."""

    def __init__(self, cfg: MCPServerConfig, *, connect_on_init: bool = True):
        """Initialize the async client.

        Args:
            cfg: Server connection configuration.
            connect_on_init: If True, connect immediately. If False, connect lazily on first use.
        """
        self.cfg = cfg
        self._session: ClientSession | None = None

        # Underlying transport context manager (stdio_client or streamable_http_client)
        self._transport_cm: Any = None

        # For HTTP: keep a configured AsyncClient if headers are needed
        self._http_client: httpx.AsyncClient | None = None

        # For debugging/observability
        self.http_session_id: str | None = None

    async def connect(self) -> "MCPAsyncClient":
        """Connect and initialize the MCP session.

        Returns:
            Self.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self.cfg.transport == "stdio":
            params = StdioServerParameters(
                command=self.cfg.stdio_command,
                args=self.cfg.stdio_args,
                env=_merge_stdio_env(self.cfg.stdio_env),
            )
            try:
                self._transport_cm = stdio_client(params)
                read_stream, write_stream = await self._transport_cm.__aenter__()  # pylint: disable=unnecessary-dunder-call
            except (BrokenPipeError, EOFError, subprocess.SubprocessError) as e:
                raise RuntimeError(
                    f"Failed to start MCP server subprocess. Is the MCP server running?\n"
                    f"Command: {self.cfg.stdio_command}\n"
                    f"Args: {self.cfg.stdio_args}\n"
                    f"Try starting it manually:\n"
                    f"  python -m bamboo.server\n"
                    f"Original error: {e}"
                ) from e
            except Exception as e:  # pylint: disable=broad-exception-caught
                raise RuntimeError(
                    f"Failed to connect to MCP server via stdio. Is the MCP server running?\n"
                    f"Command: {self.cfg.stdio_command}\n"
                    f"Args: {self.cfg.stdio_args}\n"
                    f"Try starting it manually:\n"
                    f"  python -m bamboo.server\n"
                    f"Original error: {e}"
                ) from e

        else:
            # Dynamically import the helper — which function exists, and its
            # signature, varies across mcp SDK releases.  Verified against
            # installed wheels:
            #   mcp 1.x: streamablehttp_client(url, headers, timeout,
            #            sse_read_timeout, terminate_on_close,
            #            httpx_client_factory, auth)
            #            and also streamable_http_client(url, http_client,
            #            terminate_on_close)
            #   mcp 2.x: streamable_http_client(url, http_client,
            #            terminate_on_close) only
            # Preferring the hyphen-free name therefore selects the
            # header/timeout form on 1.x and the http_client form on 2.x.
            import importlib  # pylint: disable=import-outside-toplevel
            import inspect  # pylint: disable=import-outside-toplevel
            try:
                _mod = importlib.import_module("mcp.client.streamable_http")
                func = (
                    getattr(_mod, "streamablehttp_client", None)
                    or getattr(_mod, "streamable_http_client", None)
                )
            except Exception:  # pylint: disable=broad-exception-caught
                func = None

            if func is None:
                raise RuntimeError("streamable_http_client is not available in this environment")

            # Detect signature to handle both mcp SDK shapes.
            _params = inspect.signature(func).parameters
            timeout = _with_connect_timeout(
                httpx.Timeout(self.cfg.http_timeout_s), self.cfg.http_connect_timeout_s
            )
            if "http_client" in _params:
                # We build the client, so the split timeout applies directly.
                self._http_client = httpx.AsyncClient(
                    headers=self.cfg.http_headers, timeout=timeout
                )
                self._transport_cm = func(
                    self.cfg.http_url,
                    http_client=self._http_client,
                    terminate_on_close=self.cfg.terminate_on_close,
                )
            else:
                # The SDK builds its own client here, from
                # httpx.Timeout(timeout, read=sse_read_timeout) — which leaves
                # connect equal to the per-call budget.  Where it accepts a
                # factory we supply one that overrides connect only.
                kwargs: dict[str, Any] = {
                    "headers": self.cfg.http_headers,
                    "timeout": self.cfg.http_timeout_s,
                    "terminate_on_close": self.cfg.terminate_on_close,
                }
                if "httpx_client_factory" in _params:
                    kwargs["httpx_client_factory"] = _make_http_client_factory(
                        self.cfg.http_connect_timeout_s
                    )
                else:
                    logger.debug(
                        "MCP transport does not accept httpx_client_factory; "
                        "connect timeout will follow the per-call budget."
                    )
                self._transport_cm = func(self.cfg.http_url, **kwargs)

            # New SDK yields (read_stream, write_stream, get_session_id_callback);
            # older SDK yields only (read_stream, write_stream).
            _entered = await self._transport_cm.__aenter__()  # pylint: disable=unnecessary-dunder-call
            if len(_entered) == 3:
                read_stream, write_stream, get_session_id = _entered
                try:
                    self.http_session_id = get_session_id()
                except Exception:  # pylint: disable=broad-exception-caught
                    self.http_session_id = None
            else:
                read_stream, write_stream = _entered
                self.http_session_id = None

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()  # pylint: disable=unnecessary-dunder-call

        # Initialize MCP session
        await self._session.initialize()
        return self

    async def aclose(self) -> None:
        """Close session and transport cleanly."""
        # Close session first
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            finally:
                self._session = None

        # Then close transport
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            finally:
                self._transport_cm = None

        # Then close HTTP client if used
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            finally:
                self._http_client = None

    async def list_tools(self) -> Any:
        """List tools from the MCP server."""
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.list_tools()

    async def list_prompts(self) -> Any:
        """List prompts from the MCP server."""
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.list_prompts()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool.

        Args:
            name: Tool name.
            arguments: Tool arguments JSON object.

        Returns:
            Tool call result.
        """
        if self._session is None:
            raise RuntimeError("MCP session not connected.")
        return await self._session.call_tool(name, arguments)


class MCPClientSync:
    """Synchronous wrapper around MCPAsyncClient for Streamlit and other sync UIs.

    Runs a dedicated asyncio event loop in a background thread and calls into it
    using `asyncio.run_coroutine_threadsafe`.

    The MCP session is owned end to end by a single long-lived task on that loop
    (:meth:`_session_runner`).  That task opens the session, waits, and closes
    it, so ``__aenter__`` and ``__aexit__`` on the transport always run in the
    same task.  anyio cancel scopes are task-affine: entering in one task and
    exiting in another raises ``Attempted to exit cancel scope in a different
    task`` and can leave the scope stuck re-delivering cancellation, which
    burns a CPU core indefinitely.  Individual tool calls still run as their own
    tasks via :meth:`_run` — they only *use* the session, they do not own its
    lifetime, so they remain independently cancellable.
    """

    def __init__(self, cfg: MCPServerConfig, *, connect_on_init: bool = True):
        """Create and connect the MCP client synchronously.

        Args:
            cfg: Server connection configuration.
            connect_on_init: If True, connect immediately. If False, connect lazily on first use.

        Raises:
            RuntimeError: If connection/initialization fails.
        """
        self.cfg = cfg
        self._client = MCPAsyncClient(cfg)
        self._connected = False
        self._connect_on_init = connect_on_init

        # Session runner state.  ``_shutdown_event`` is an asyncio primitive and
        # is therefore created on the loop thread, inside the runner task.
        self._state_lock = threading.Lock()
        self._loop_ready = threading.Event()
        self._session_ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._runner_future: concurrent.futures.Future[None] | None = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-client-loop", daemon=True)
        self._thread.start()

        # Wait for the loop to actually be running rather than sleeping and
        # hoping.  ``_shutdown_loop_thread`` decides whether to stop the loop
        # from ``is_running()``, so a not-yet-started loop could otherwise be
        # left running forever by a teardown that raced startup.
        if not self._loop_ready.wait(timeout=SHUTDOWN_TIMEOUT_S):
            self._shutdown_loop_thread()
            raise RuntimeError("MCP client event loop thread failed to start.")

        # Optionally connect on the loop thread (e.g., Streamlit wants immediate readiness).
        if self._connect_on_init:
            self.ensure_connected()

    def _run_loop(self) -> None:
        """Event loop thread target."""
        asyncio.set_event_loop(self._loop)
        # Scheduled rather than set directly: this fires once the loop is
        # actually processing callbacks, which is what callers need to know.
        self._loop.call_soon(self._loop_ready.set)
        self._loop.run_forever()

    async def _session_runner(self) -> None:
        """Own the MCP session for its entire lifetime on one task.

        Opens the session, signals readiness, waits for a shutdown request, then
        closes the session.  Both the open and the close happen here, in this
        task, which is the invariant the anyio transports require.

        A failure during startup is recorded on ``_startup_error`` and re-raised
        by :meth:`_start_session` on the calling thread, so the caller sees the
        original exception rather than a bare readiness timeout.
        """
        self._shutdown_event = asyncio.Event()
        try:
            await self._client.connect()
        except BaseException as exc:  # noqa: B036 - recorded and re-raised on the caller's thread
            self._startup_error = exc
            return
        finally:
            # Set even on failure: the caller is blocked on this and must not
            # wait out the full deadline for an error that already happened.
            self._session_ready.set()

        try:
            await self._shutdown_event.wait()
        finally:
            await self._client.aclose()

    def _shutdown_loop_thread(self, *, join_timeout: float = 2.0) -> None:
        """Stop the background event loop and join its thread.

        Safe to call repeatedly, and safe on a partially initialised client.
        The loop is closed only once the thread has actually exited, because
        closing a running loop raises :class:`RuntimeError` — which, on the
        failure path in :meth:`__init__`, would mask the original connection
        error.

        Args:
            join_timeout: Seconds to wait for the loop thread to exit.
        """
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=join_timeout)
        if not self._thread.is_alive() and not self._loop.is_closed():
            self._loop.close()

    def _run(self, coro: "asyncio.Future[Any] | Coroutine[Any, Any, Any]") -> Any:
        """Run a coroutine on the background loop and wait for the result.

        Args:
            coro: Coroutine to execute.

        Returns:
            Result of the coroutine.

        Raises:
            RuntimeError: If the coroutine fails or times out.
        """
        # run_coroutine_threadsafe expects a coroutine; if a Future is passed
        # (unlikely in our usage), forward it as-is; otherwise use it directly.
        # Timeout for waiting on the result. Override with
        # BAMBOO_MCP_CLIENT_TIMEOUT (seconds).
        #
        # This is the outer of two independent ceilings on a single tool call:
        # this future deadline and the transport's own http_timeout_s. The lower
        # one wins, so raising only one of them has no effect. It was 120 s,
        # sized for large BigPanDA task fetches (60–90 s for tasks with many
        # thousands of jobs); a long-running tool that waits inline, plus the
        # planner and synthesis LLM calls around it, does not fit in that.
        _timeout = _default_client_timeout()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        try:
            return fut.result(timeout=_timeout)
        except concurrent.futures.TimeoutError as e:
            # Giving up on the *future* does not stop the coroutine: it keeps
            # running on the loop thread, still holding its transport, with no
            # remaining reference through which it could ever be closed.  A
            # timed-out ``connect()`` orphaned exactly this way was left
            # spinning a full core for 17 hours on aipanda033.
            fut.cancel()
            raise RuntimeError(
                f"MCP server call timed out after {_timeout:g} seconds.\n"
                "Is the MCP server running and responding?\n"
                "Try increasing BAMBOO_MCP_CLIENT_TIMEOUT if fetching large tasks."
            ) from e
        except (asyncio.CancelledError, concurrent.futures.CancelledError) as e:
            # CancelledError is a BaseException, so it must be caught ahead of
            # the bare re-raise below to keep its actionable message.
            raise _wrap_error(e) from e
        except Exception as e:
            raise _wrap_error(e) from e
        except BaseException:
            # KeyboardInterrupt / SystemExit in the calling thread while the
            # coroutine is still in flight.  Same orphaning hazard as the
            # timeout path above, so cancel before unwinding.
            fut.cancel()
            raise

    def ensure_connected(self) -> None:
        """Connect to the MCP server if not already connected.

        Raises:
            RuntimeError: If the session fails to open, or if the client has
                already been closed.
        """
        if self._connected:
            return
        with self._state_lock:
            if self._connected:
                return
            self._start_session()

    def _start_session(self) -> None:
        """Launch the session runner task and wait for it to be ready.

        Must be called with ``_state_lock`` held.

        Raises:
            RuntimeError: If startup fails or exceeds the client deadline.  In
                either case the loop thread is torn down first, so a failed
                construction cannot leave an orphan behind.
        """
        if self._loop.is_closed():
            raise RuntimeError("MCP client has been closed and cannot be reused.")

        self._session_ready.clear()
        self._startup_error = None
        self._runner_future = asyncio.run_coroutine_threadsafe(self._session_runner(), self._loop)

        timeout = _default_client_timeout()
        if not self._session_ready.wait(timeout=timeout):
            # The runner is still inside connect().  Cancel it and tear the
            # thread down: leaving it in place is what orphaned a spinning
            # loop thread in production.
            self._runner_future.cancel()
            self._shutdown_loop_thread()
            raise RuntimeError(
                f"MCP server connection timed out after {timeout:g} seconds.\n"
                "Is the MCP server running and responding?\n"
                "Try increasing BAMBOO_MCP_CLIENT_TIMEOUT if fetching large tasks."
            )

        if self._startup_error is not None:
            error = self._startup_error
            self._shutdown_loop_thread()
            raise _wrap_error(error)

        self._connected = True

    def _request_session_shutdown(self) -> None:
        """Ask the runner task to close the session, and wait for it.

        Signalling rather than submitting ``aclose()`` as a fresh coroutine is
        the point of the runner: the close has to happen in the task that did
        the open.
        """
        event = self._shutdown_event
        future = self._runner_future
        if event is not None:
            self._loop.call_soon_threadsafe(event.set)
        if future is None:
            return
        try:
            future.result(timeout=SHUTDOWN_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "MCP session did not close within %.0fs; abandoning it and stopping the loop.",
                SHUTDOWN_TIMEOUT_S,
            )
            future.cancel()
        except concurrent.futures.CancelledError:
            pass
        except Exception:
            # A teardown failure must not mask whatever the caller was doing.
            logger.warning("MCP session close raised.", exc_info=True)

    def close(self) -> None:
        """Close the MCP session and stop the background loop.

        Safe to call more than once.
        """
        try:
            if self._connected:
                self._request_session_shutdown()
            self._connected = False
        finally:
            self._shutdown_loop_thread()

    def list_tools(self) -> Any:
        """List tools (sync)."""
        self.ensure_connected()
        return self._run(self._client.list_tools())

    def list_prompts(self) -> Any:
        """List prompts (sync)."""
        self.ensure_connected()
        return self._run(self._client.list_prompts())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call tool (sync)."""
        self.ensure_connected()
        return self._run(self._client.call_tool(name, arguments))

    @property
    def http_session_id(self) -> str | None:
        """Return HTTP MCP session id if using HTTP transport."""
        return self._client.http_session_id
