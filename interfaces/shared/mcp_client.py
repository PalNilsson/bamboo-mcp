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
import subprocess
import sys
import threading
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import httpx

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

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
            # Dynamically import the helper — the function was renamed and its
            # signature changed between mcp SDK versions:
            #   < 1.x:  streamable_http_client(url, *, http_client, terminate_on_close)
            #   >= 1.x: streamablehttp_client(url, *, headers, timeout, terminate_on_close)
            # We detect which version is present and call accordingly.
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

            # Detect signature to handle both old and new mcp SDK versions.
            _params = inspect.signature(func).parameters
            timeout = httpx.Timeout(self.cfg.http_timeout_s)
            if "http_client" in _params:
                # Old signature: pass a pre-built httpx.AsyncClient.
                self._http_client = httpx.AsyncClient(
                    headers=self.cfg.http_headers, timeout=timeout
                )
                self._transport_cm = func(
                    self.cfg.http_url,
                    http_client=self._http_client,
                    terminate_on_close=self.cfg.terminate_on_close,
                )
            else:
                # New signature (mcp >= 1.x): pass headers and timeout directly.
                self._transport_cm = func(
                    self.cfg.http_url,
                    headers=self.cfg.http_headers,
                    timeout=self.cfg.http_timeout_s,
                    terminate_on_close=self.cfg.terminate_on_close,
                )

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

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-client-loop", daemon=True)
        self._thread.start()

        # Give the event loop thread time to start
        time.sleep(0.1)

        # Optionally connect on the loop thread (e.g., Streamlit wants immediate readiness).
        if self._connect_on_init:
            try:
                self._run(self._client.connect())
            except BaseException:
                # A failed connect must not leave the loop thread running.  The
                # thread is a daemon and ``__init__`` is propagating, so the
                # caller never receives a reference to call ``close()`` on:
                # without this teardown the thread survives for the life of the
                # process, still owning whatever half-open transport
                # ``connect()`` had reached.  Streamlit's
                # ``@st.cache_resource`` does not memoise exceptions, so the
                # next rerun constructs another client and the orphans
                # accumulate one per failed attempt.
                self._shutdown_loop_thread()
                raise
            self._connected = True

    def _run_loop(self) -> None:
        """Event loop thread target."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

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
            raise RuntimeError(
                f"MCP server connection was cancelled.\n"
                f"This can happen during startup if the server subprocess exits immediately.\n"
                f"Check that the server starts correctly:\n"
                f"  python -m bamboo.server\n"
                f"Original error: {type(e).__name__}"
            ) from e
        except ConnectionRefusedError as e:
            raise RuntimeError(
                f"Failed to connect to MCP server (connection refused).\n"
                f"Ensure the server is running:\n"
                f"  python -m bamboo.server\n"
                f"Original error: {e}"
            ) from e
        except OSError as e:
            if "Connection refused" in str(e) or "No such file or directory" in str(e):
                raise RuntimeError(
                    f"Cannot connect to MCP server. Make sure it's running:\n"
                    f"  python -m bamboo.server\n"
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(str(e)) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create MCP client: {type(e).__name__}: {e}\n"
                f"Is the MCP server running?\n"
                f"  python -m bamboo.server"
            ) from e
        except BaseException:
            # KeyboardInterrupt / SystemExit in the calling thread while the
            # coroutine is still in flight.  Same orphaning hazard as the
            # timeout path above, so cancel before unwinding.
            fut.cancel()
            raise

    def ensure_connected(self) -> None:
        """Connect to the MCP server if not already connected."""
        if self._connected:
            return
        self._run(self._client.connect())
        self._connected = True

    def close(self) -> None:
        """Close the MCP session and stop the background loop."""
        try:
            if self._connected:
                self._run(self._client.aclose())
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
