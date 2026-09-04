#!/usr/bin/env python3
"""Textual TUI for Bamboo (formerly AskPanDA).

This interface provides a terminal UI for interacting with the Bamboo MCP server.

Key goals:
* Fast, non-blocking initial render (no awaits in on_mount).
* Single-task ownership of MCP connect/use/close to avoid AnyIO cancel-scope errors.
* Clean shutdown via /exit, /quit, Ctrl+D/Ctrl+Q.
* Inline mode by default for easy copy/paste in most terminals.
* Forward LLM defaults from environment variables.

Environment variables:
  - LLM_DEFAULT_PROVIDER
  - LLM_DEFAULT_MODEL
  - ASKPANDA_LLM_DEFAULT_PROVIDER (also forwarded)
  - ASKPANDA_LLM_DEFAULT_MODEL (also forwarded)
  - ASKPANDA_PLUGIN (default plugin id, e.g. "atlas", "epic", "cgsim")

Examples:
  python3 interfaces/textual/chat.py --transport stdio
  python3 interfaces/textual/chat.py --transport http --http-url http://localhost:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import re
import os
import sys
import tempfile
import time
import webbrowser
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.events import Key
from textual.widgets import Footer, Header, Input, RichLog, Static

from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig

DEFAULT_PLUGIN = os.getenv("ASKPANDA_PLUGIN", "atlas")

#: Plain-text superuser password from env var.  Empty string = feature disabled.
_SUPERUSER_PASSWORD: str = os.getenv("BAMBOO_SUPERUSER_PASSWORD", "")

# Maximum number of user+assistant turn *pairs* to keep in context.
# Each pair = 2 messages (1 user + 1 assistant), so 10 pairs = 20 messages.
_DEFAULT_HISTORY_TURNS = 10
try:
    _MAX_HISTORY_TURNS: int = int(os.getenv("BAMBOO_HISTORY_TURNS", str(_DEFAULT_HISTORY_TURNS)))
except ValueError:
    _MAX_HISTORY_TURNS = _DEFAULT_HISTORY_TURNS

ENV_LLM_DEFAULT_PROVIDER = "LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL = "LLM_DEFAULT_MODEL"
ENV_LLM_DEFAULT_PROVIDER_ALT = "ASKPANDA_LLM_DEFAULT_PROVIDER"
ENV_LLM_DEFAULT_MODEL_ALT = "ASKPANDA_LLM_DEFAULT_MODEL"

ANSWER_TOOL_CANDIDATES: List[str] = ["bamboo_answer", "askpanda_answer", "bamboo_plan"]


_TASK_CMD_RE = re.compile(r"^/task\s+(\d{1,12})\s*$", re.IGNORECASE)
_JOB_CMD_RE = re.compile(r"^/job\s+(\d{1,12})\s*$", re.IGNORECASE)
_RATE_CMD_RE = re.compile(r"^/rate\s+([1-5])\s*$", re.IGNORECASE)

# Matches Markdown inline links: [label](url)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
# Matches bare URLs not already captured inside markdown link parentheses
_BARE_URL_RE = re.compile(r"(?<!\()(https?://[^\s\)\]\>\"']+)")

FALLBACK_BANNER = r"""
 ____                  _                    __  __  ____ ____
| __ )  __ _ _ __ ___ | |__   ___   ___    |  \/  |/ ___|  _ \
|  _ \ / _` | '_ ` _ \| '_ \ / _ \ / _ \   | |\/| | |   | |_) |
| |_) | (_| | | | | | | |_) | (_) | (_) |  | |  | | |___|  __/
|____/ \__,_|_| |_| |_|_.__/ \___/ \___/   |_|  |_|\____|_|
""".strip("\n").splitlines()


# ---------------------------------------------------------------------------
# Question-parsing helpers for auto-chart (client-side, no bamboo-core import)
# ---------------------------------------------------------------------------

#: Harvester worker statuses recognised in user questions.
_HARVESTER_STATUSES: tuple[str, ...] = (
    "running", "submitted", "finished", "failed", "cancelled", "missed", "idle",
)


def _extract_status_from_question(question: str) -> str:
    """Extract a Harvester worker status from a user question.

    Scans for the first occurrence of a known status word.  Returns
    ``"running"`` as the default when no status word is found, since
    that is the most operationally relevant status for charts.

    Args:
        question: Raw user question text.

    Returns:
        Lowercase status string, e.g. ``"running"``, ``"failed"``.
    """
    q = question.lower()
    for status in _HARVESTER_STATUSES:
        if status in q:
            return status
    return "running"


def _extract_window_from_question(
    question: str,
) -> tuple[str, str] | None:
    """Extract a time window from a user question (client-side mirror).

    Duplicates the core logic of
    ``bamboo_answer._extract_time_window_from_question`` so the TUI does
    not need to import server-side modules.

    Recognised patterns (case-insensitive):
    - ``"last/past N hours/minutes/days"``
    - ``"yesterday"`` / ``"since yesterday"``
    - ``"today"``

    Args:
        question: Raw user question text.

    Returns:
        ``(from_dt, to_dt)`` ISO-8601 strings (UTC, no timezone suffix),
        or ``None`` when no temporal expression is found.
    """
    import re as _re
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    q = question.lower()
    now = _dt.now(tz=_tz.utc).replace(microsecond=0)

    def _fmt(d: _dt) -> str:
        return d.strftime("%Y-%m-%dT%H:%M:%S")

    m = _re.search(
        r"(?:last|past|in\s+the\s+last|in\s+the\s+past)\s+(\d+)\s+"
        r"(hour|hours|hr|hrs|minute|minutes|min|mins|day|days)",
        q,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            delta = _td(minutes=n)
        elif unit.startswith("day"):
            delta = _td(days=n)
        else:
            delta = _td(hours=n)
        return _fmt(now - delta), _fmt(now)

    if _re.search(r"\b(?:since\s+)?yesterday\b", q):
        midnight = now.replace(hour=0, minute=0, second=0)
        return _fmt(midnight - _td(days=1)), _fmt(now)

    if _re.search(r"\b(?:since\s+)?today\b", q):
        return _fmt(now.replace(hour=0, minute=0, second=0)), _fmt(now)

    return None


def _now() -> str:
    """Return a short local timestamp for transcript headers.

    Returns:
        str: Local time formatted as ``HH:MM:SS``.
    """
    return time.strftime("%H:%M:%S")


def _span_detail(span_rec: Dict[str, Any]) -> str:
    """Format the most useful extra fields from a trace span as a short string.

    Args:
        span_rec: A parsed trace span dict from the NDJSON trace file.

    Returns:
        A compact human-readable summary of the span's event-specific fields,
        e.g. ``"route=rag"`` or ``"allowed=True reason=keyword_allow"``.
    """
    event = str(span_rec.get("event", ""))

    if event == "tool_call":
        keys = span_rec.get("args_keys", [])
        return f"args={keys}" if keys else ""

    if event == "guard":
        allowed = span_rec.get("allowed")
        reason = span_rec.get("reason", "")
        llm = span_rec.get("llm_used", False)
        llm_tag = " llm=yes" if llm else ""
        return f"allowed={allowed} reason={reason}{llm_tag}"

    if event == "retrieval":
        backend = span_rec.get("backend", "")
        hits = span_rec.get("hits")
        hits_str = str(hits) if hits is not None else "?"
        return f"backend={backend} hits={hits_str}"

    if event == "llm_call":
        provider = span_rec.get("provider", "")
        model = span_rec.get("model", "")
        inp = span_rec.get("input_tokens")
        out = span_rec.get("output_tokens")
        tokens = ""
        if inp is not None or out is not None:
            tokens = f" tokens={inp}→{out}"
        return f"{provider}/{model}{tokens}"

    if event == "synthesis":
        route = span_rec.get("route", "")
        return f"route={route}"

    # Unknown event — show all extra keys.
    skip = {"bamboo_trace", "event", "tool", "ts", "duration_ms"}
    extras = {k: v for k, v in span_rec.items() if k not in skip}
    return str(extras) if extras else ""


def _pretty(obj: Any) -> str:
    """Serialize an object as pretty JSON when possible.

    Args:
        obj (Any): Value to serialize.

    Returns:
        str: JSON-formatted output, or ``str(obj)`` if serialization fails.
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _strip_mermaid_blocks(text: str) -> tuple[str, int]:
    r"""Remove fenced Mermaid code blocks from an LLM response.

    The TUI cannot render Mermaid diagrams.  Storing raw Mermaid syntax in
    conversation history pollutes the LLM context window on follow-up
    questions.  This helper strips the blocks before display and storage,
    returning the clean text and the count of diagrams removed so the caller
    can append a short note directing the user to Streamlit.

    Args:
        text (str): Raw assistant response text, potentially containing one
            or more ` ```mermaid ``` ` fenced blocks.

    Returns:
        tuple[str, int]: ``(clean_text, diagram_count)`` where ``clean_text``
        has all Mermaid fenced blocks removed and double blank lines
        collapsed, and ``diagram_count`` is the number of blocks stripped.
    """
    pattern = re.compile(r"```mermaid\s*\n.*?```", re.DOTALL)
    count = len(pattern.findall(text))
    clean = pattern.sub("", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, count


def _extract_text(result: Any) -> str:
    """Extract displayable text from an MCP tool result.

    Args:
        result: The raw tool result (SDK object, dict, string, etc.).

    Returns:
        A best-effort string suitable for rendering in the transcript.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join([p for p in parts if p]).strip()
        for key in ("text", "output", "message", "result"):
            val = result.get(key)
            if isinstance(val, str):
                return val.strip()
        return _pretty(result)

    content2 = getattr(result, "content", None)
    if isinstance(content2, list):
        parts2: List[str] = []
        for item in content2:
            txt = getattr(item, "text", None)
            if isinstance(txt, str):
                parts2.append(txt)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts2.append(str(item.get("text", "")))
        return "\n".join([p for p in parts2 if p]).strip()

    return _pretty(result)


#: Imported from the shared guard module; extended by BAMBOO_SUPERUSER_TOOLS.
from interfaces.shared.superuser_guard import is_superuser_question as _is_superuser_question  # noqa: E402


def _tool_names_from_list_tools(list_tools_result: Any) -> List[str]:
    """Extract tool names from an MCP ``list_tools`` response.

    Args:
        list_tools_result (Any): Raw MCP SDK object or dict response.

    Returns:
        List[str]: Tool names discovered in the response.
    """
    tools = getattr(list_tools_result, "tools", None)
    if isinstance(tools, list):
        names: List[str] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.append(name)
            elif isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.append(tool["name"])
        return names

    if isinstance(list_tools_result, dict):
        tools2 = list_tools_result.get("tools")
        if isinstance(tools2, list):
            out: List[str] = []
            for tool in tools2:
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    out.append(tool["name"])
            return out

    return []


# ---------------------------------------------------------------------------
# Token cost estimation
# ---------------------------------------------------------------------------

# Per-model pricing in USD per 1 million tokens: (input_rate, output_rate).
# Add new models here as needed; unknown models fall back to _DEFAULT_COST_PER_MTOK.
# Prices are approximate — verify against current provider documentation.
_MODEL_COST_PER_MTOK: Dict[str, tuple] = {
    # Mistral
    "mistral-large-latest": (2.00, 6.00),
    "mistral-large-2411": (2.00, 6.00),
    "mistral-small-latest": (0.20, 0.60),
    "mistral-small-2501": (0.20, 0.60),
    "open-mistral-nemo": (0.15, 0.15),
    # Anthropic
    "claude-opus-4-5": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Fallback rate used when the model is not in _MODEL_COST_PER_MTOK.
_DEFAULT_COST_PER_MTOK: tuple = (1.00, 3.00)


def _estimate_cost(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate the LLM cost of a request from its trace spans.

    Sums ``input_tokens`` and ``output_tokens`` across all ``llm_call``
    spans and prices them using :data:`_MODEL_COST_PER_MTOK`.  When a
    model is not found in the table the default rate is used and the model
    name is recorded in ``unknown_models`` so the caller can warn the user.

    Args:
        spans: Parsed trace span dicts from ``_last_spans``.

    Returns:
        Dict with keys ``calls`` (per-call breakdown list),
        ``total_input``, ``total_output``, ``total_tokens``,
        ``total_cost_usd``, and ``unknown_models``.
    """
    calls: List[Dict[str, Any]] = []
    unknown_models: List[str] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for span in spans:
        if span.get("event") != "llm_call":
            continue
        model = str(span.get("model", "unknown"))
        provider = str(span.get("provider", ""))
        inp = int(span.get("input_tokens") or 0)
        out = int(span.get("output_tokens") or 0)
        duration_ms = float(span.get("duration_ms", 0.0))

        rate = (
            _MODEL_COST_PER_MTOK.get(model) or
            _MODEL_COST_PER_MTOK.get(model.lower())
        )
        if rate is None:
            unknown_models.append(model)
            rate = _DEFAULT_COST_PER_MTOK

        call_cost = (inp / 1_000_000) * rate[0] + (out / 1_000_000) * rate[1]

        calls.append({
            "provider": provider,
            "model": model,
            "input_tokens": inp,
            "output_tokens": out,
            "duration_ms": duration_ms,
            "cost_usd": call_cost,
            "rate_in": rate[0],
            "rate_out": rate[1],
        })

        total_input += inp
        total_output += out
        total_cost += call_cost

    return {
        "calls": calls,
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": total_cost,
        "unknown_models": list(set(unknown_models)),
    }


class BambooTui(App):
    """Textual-based terminal UI for Bamboo."""

    ENABLE_MOUSE = False

    CSS = """
    Screen {
        height: 100%;
    }

    /* Inline apps must opt-in to a fixed height, otherwise they may render as 0 lines. */
    Screen:inline {
        height: 50;
        border: none;
    }

    #root {
        layout: vertical;
        height: 100%;
    }

    #banner {
        height: 9;
        min-height: 9;
        padding: 1 2;
    }

    #transcript {
        height: 1fr;
        overflow-y: auto;
        border: round $surface;
    }

    #input_row {
        height: 3;
        padding: 0 2;
    }

    #input {
        height: 3;
    }

    #thinking {
        height: auto;
        padding: 0 2 0 3;
        display: none;
    }

    #thinking.active {
        display: block;
    }

    """

    BINDINGS = [
        Binding("escape", "focus_input", "Focus input", show=False),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("pageup", "scroll_up", "Scroll up", show=True),
        Binding("pagedown", "scroll_down", "Scroll down", show=True),
        Binding("shift+pageup", "scroll_up", "Scroll up", show=False),
        Binding("shift+pagedown", "scroll_down", "Scroll down", show=False),
        Binding("ctrl+home", "scroll_home", "Scroll to top", show=False),
        Binding("ctrl+end", "scroll_end_action", "Scroll to bottom", show=False),
    ]

    def __init__(self, cfg: MCPServerConfig, plugin_id: str) -> None:
        """Initialize the app.

        Args:
            cfg (MCPServerConfig): MCP server config (stdio or HTTP).
            plugin_id (str): Active plugin namespace (e.g. "atlas").
        """
        super().__init__()
        self.cfg = cfg
        self.plugin_id = plugin_id

        # Sync wrapper; only used via asyncio.to_thread.
        self.mcp: MCPClientSync = MCPClientSync(cfg, connect_on_init=False)

        # One task owns the full MCP lifecycle.
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._mcp_task: Optional[asyncio.Task[None]] = None

        self.tool_names: List[str] = []
        self.answer_tool: Optional[str] = None

        self.banner_lines: Sequence[str] = FALLBACK_BANNER[:]
        self.display_name: str = f"Bamboo MCP – {plugin_id.upper()}"
        self.help_text: str = "Enter to send • /help"

        self.debug_mode: bool = False
        # Initialise fast-path from env var; default True when unset.
        # Set BAMBOO_FAST_PATH=0 (or off/false) to start with LLM planner.
        _fp_env = os.getenv("BAMBOO_FAST_PATH", "1").strip().lower()
        self.fast_path_mode: bool = _fp_env not in ("0", "off", "false")
        self._history: List[Dict[str, str]] = []  # In-memory chat history for multi-turn context
        self._mcp_ready: bool = False
        self._last_raw_result: Any = None  # Most recent raw MCP tool result for /json
        self._last_task_id: Optional[int] = None  # Most recent task ID queried
        self._last_job_id: Optional[int] = None   # Most recent job ID queried
        self._command_history: List[str] = []     # Input history for arrow-up/down recall
        self._cmd_history_pos: int = -1           # Current position in _command_history (-1 = not browsing)
        self._thinking_task: Optional[Any] = None  # Textual Timer for animated thinking indicator
        self._last_spans: List[Dict[str, Any]] = []  # Trace spans for last request
        self._last_response_links: List[Tuple[str, str]] = []  # (label, url) pairs from last assistant response
        self._last_response_text: str = ""  # raw text of last assistant response, for /script
        self._last_rated_doc: tuple[str, str] | None = None  # (index, doc_id) for /rate
        # Session-level cost accumulators — updated after every completed request.
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._superuser: bool = False  # Unlocked by /superuser <password>
        self._session_cost_usd: float = 0.0
        self._session_request_count: int = 0
        self._trace_file: str = os.path.join(
            tempfile.gettempdir(), f"bamboo_trace_{os.getpid()}.jsonl"
        )
        self._cric_table_file: str = os.path.join(
            tempfile.gettempdir(), f"bamboo_cric_table_{os.getpid()}.txt"
        )
        # Question queue — allows the user to keep typing while a response is
        # in progress.  Submitted questions are enqueued and drained one at a
        # time by _question_worker so responses are always returned in order.
        self._question_queue: asyncio.Queue[str] = asyncio.Queue()
        self._is_thinking: bool = False

        self.banner_widget: Optional[Static] = None
        self.transcript: Optional[RichLog] = None
        self.thinking_widget: Optional[Static] = None
        self.input_widget: Optional[Input] = None

    def compose(self) -> ComposeResult:
        """Build the Textual widget tree.

        Returns:
            ComposeResult: Stream of widgets to render.
        """
        yield Header(show_clock=True)

        with Container(id="root"):
            with Vertical(id="banner"):
                self.banner_widget = Static()
                yield self.banner_widget

            self.transcript = RichLog(id="transcript", wrap=True, markup=False)
            yield self.transcript

            self.thinking_widget = Static("", id="thinking")
            yield self.thinking_widget

            with Container(id="input_row"):
                self.input_widget = Input(
                    id="input",
                    placeholder="Bamboo MCP > (Enter to send)",
                )
                yield self.input_widget

        yield Footer()

    async def on_mount(self) -> None:
        """Mount UI quickly and start MCP initialization in the background."""
        if self.banner_widget:
            self.banner_widget.can_focus = False
        if self.transcript:
            self.transcript.can_focus = False

        self._render_banner_placeholder()
        self._write_system("Starting… initializing MCP…")

        # Disable mouse capture so the terminal can handle text selection normally.
        # Trackpad scrolling is handled via PageUp/PageDown bindings instead.
        try:
            driver = getattr(self, "_driver", None)
            if driver and hasattr(driver, "disable_mouse_support"):
                driver.disable_mouse_support()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # IMPORTANT: no awaits here; let Textual finish initial render.
        self._mcp_task = asyncio.create_task(self._mcp_main(), name="bamboo.mcp_main")
        asyncio.create_task(self._question_worker(), name="bamboo.question_worker")
        self.action_focus_input()

    async def on_exit(self) -> None:
        """Request clean shutdown and wait for MCP task to finish."""
        self._shutdown_event.set()

        if self._mcp_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._mcp_task), timeout=5)
            except asyncio.TimeoutError:
                # If the task is stuck, do a best-effort close without cancelling.
                try:
                    await self._mcp_close()
                except Exception:
                    pass
            except Exception:
                pass

    async def _mcp_main(self) -> None:
        """Own the MCP lifecycle in a single task."""
        _startup_timeout: int = int(os.getenv("BAMBOO_STARTUP_TIMEOUT", "20"))
        self._startup_phase: str = "connecting"
        try:
            # Don't wrap _mcp_connect in wait_for — the MCPClientSync._run()
            # has its own timeout (BAMBOO_MCP_CLIENT_TIMEOUT, default 120 s).
            # Using asyncio.wait_for here cancels the underlying thread call,
            # which raises CancelledError through _run() even when the server
            # is starting correctly but slowly.
            await self._mcp_connect()
            self._startup_phase = "loading tools"
            await asyncio.wait_for(self._refresh_tools(), timeout=_startup_timeout)
            self._detect_answer_tool()

            self._startup_phase = "loading banner"
            await asyncio.wait_for(self._load_banner(), timeout=_startup_timeout)
            self._render_banner()
            self._startup_phase = "ready"

            self._mcp_ready = True
            # Fetch LLM info from the server-side health tool — calling
            # get_llm_info() directly here fails because the LLM selector
            # is not yet initialised at this point in the startup sequence.
            llm_info = ""
            try:
                health_res = await self._to_thread(
                    self.mcp.call_tool, "bamboo_health", {}
                )
                health_text = _extract_text(health_res)
                for line in health_text.splitlines():
                    if line.strip().startswith("- llm_info:"):
                        llm_info = line.split(":", 1)[1].strip()
                        break
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            # Inform the user that a live LLM round-trip is about to happen.
            # Written here (after layout is complete) so the panel renders at
            # full terminal width rather than the narrow pre-layout size.
            self._write_system("Verifying LLM connection… please wait…")
            # Run the LLM probe before writing the status line so the verified
            # badge and any warning appear in the correct order.
            probe_status, probe_warning = await self._run_llm_probe()

            base_line = (
                f"Connected via {self.cfg.transport}. "
                f"Answer tool: {self.answer_tool or 'UNKNOWN'}."
            )
            if llm_info and llm_info != "not configured":
                # Build a Rich Text object so the tick can be coloured green
                # without affecting the rest of the message.
                msg = Text(base_line)
                msg.append(f"\n[LLM selected] {llm_info}")
                if probe_status == "ok":
                    msg.append(" ✓ verified", style="green")
            else:
                msg = Text(base_line)
            self._write_panel(msg, title=f"{_now()}  system", border_style="dim")
            if probe_warning:
                self._write_warning(probe_warning)

            await self._shutdown_event.wait()

        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            phase = getattr(self, "_startup_phase", "unknown")
            self._write_error(
                f"Startup timed out while {phase} "
                f"(limit: {_startup_timeout}s). "
                "The server node may be under load. "
                "Retry, or set BAMBOO_STARTUP_TIMEOUT to a higher value "
                "(e.g. export BAMBOO_STARTUP_TIMEOUT=60)."
            )
            await self._shutdown_event.wait()
        except Exception as exc:
            self._write_error(
                f"Startup failed: {type(exc).__name__}: {exc or '(no detail)'}"
            )
            await self._shutdown_event.wait()
        finally:
            try:
                await self._mcp_close()
            except Exception:
                pass

    async def _to_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking callable in a worker thread.

        Args:
            func (Any): Callable to execute.
            *args (Any): Positional arguments for ``func``.
            **kwargs (Any): Keyword arguments for ``func``.

        Returns:
            Any: Value returned by ``func``.
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _mcp_connect(self) -> None:
        """Connect to the MCP server (best-effort across client implementations).

        The shared MCP client wrapper may evolve. This method avoids hard-coding a
        single entrypoint name (e.g., `connect`) by trying a small set of common
        lifecycle method names.

        Raises:
            AttributeError: If no known connect method exists.
            Exception: Propagates any connect failure from the underlying client.
        """
        # Prefer a dedicated connect() / open() / start() API if available.
        for name in ("connect", "open", "start", "acquire"):
            fn = getattr(self.mcp, name, None)
            if callable(fn):
                await self._to_thread(fn)
                return

        # As a fallback, support context-manager style wrappers.
        enter = getattr(self.mcp, "__enter__", None)
        if callable(enter):
            await self._to_thread(enter)
            return

        # This client wrapper connects lazily (e.g., on first list_tools/call_tool).
        # Treat "connect" as a no-op.
        return

    async def _mcp_close(self) -> None:
        """Close the MCP client connection using a best-effort strategy."""
        for name in ("close", "aclose", "stop", "shutdown", "release"):
            fn = getattr(self.mcp, name, None)
            if callable(fn):
                await self._to_thread(fn)
                return

        exit_fn = getattr(self.mcp, "__exit__", None)
        if callable(exit_fn):
            await self._to_thread(exit_fn, None, None, None)
            return

    async def _run_llm_probe(self) -> tuple[str, str]:
        """Call bamboo_llm_probe and return the probe status and any warning text.

        Sends a minimal single-token request to the configured LLM provider via
        the server-side probe tool.  Returns a ``(status, warning)`` tuple so the
        caller can annotate the "Connected" line before writing it and display any
        warning immediately after.

        The method is best-effort: if the tool is absent (older server version) or
        the response cannot be parsed, it returns ``("", "")`` silently.

        Returns:
            Tuple of ``(status, warning)`` where ``status`` is the probe status
            string (e.g. ``"ok"``, ``"auth_error"``) and ``warning`` is a
            human-readable warning message, or an empty string when no warning
            should be displayed.
        """
        import json as _json  # noqa: PLC0415

        _AUTH_WARNING = (
            "⚠️  LLM authentication failed at startup. "
            "Check that your API key is correct and has not expired, "
            "then restart the server."
        )
        _CONFIG_WARNING = (
            "⚠️  LLM is not configured. "
            "Set the required API key environment variable and restart the server."
        )
        _PROVIDER_WARNING = (
            "⚠️  LLM provider returned an error at startup. "
            "Responses may not work — check server logs for details."
        )

        try:
            raw = await self._to_thread(self.mcp.call_tool, "bamboo_llm_probe", {})
        except Exception:  # pylint: disable=broad-exception-caught
            # Tool absent or call failed — degrade gracefully.
            return "", ""

        try:
            text = _extract_text(raw)
            probe = _json.loads(text)
            status = probe.get("status", "")
        except Exception:  # pylint: disable=broad-exception-caught
            return "", ""

        if status == "ok" or not status:
            return status, ""

        if status == "auth_error":
            return status, _AUTH_WARNING
        if status == "config_error":
            detail = probe.get("detail", "")
            return status, f"⚠️  LLM configuration error: {detail}"
        if status in ("not_configured",):
            return status, _CONFIG_WARNING
        # rate_limit / timeout / provider_error — less critical, but still noteworthy.
        detail = probe.get("detail", status)
        return status, _PROVIDER_WARNING + f" ({detail})"

    def _render_banner_placeholder(self) -> None:
        """Render a fallback banner before plugin manifest data is loaded."""
        if not self.banner_widget:
            return
        banner_text = "\n".join(FALLBACK_BANNER)
        self.banner_widget.update(
            Panel(
                Text(banner_text, style="#0b80c3", no_wrap=True),
                title=self.display_name,
                subtitle=f"plugin={self.plugin_id}",
                border_style="blue",
                padding=(0, 1),
                expand=True,
            )
        )
        n_lines = len(FALLBACK_BANNER)
        target_height = n_lines + 4
        banner_container = self.query_one("#banner")
        banner_container.styles.height = target_height
        banner_container.styles.min_height = target_height

    def action_scroll_up(self) -> None:
        """Scroll the transcript up by one page."""
        if self.transcript:
            self.transcript.scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll the transcript down by one page."""
        if self.transcript:
            self.transcript.scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        """Scroll the transcript to the top."""
        if self.transcript:
            self.transcript.scroll_home(animate=False)

    def action_scroll_end_action(self) -> None:
        """Scroll the transcript to the bottom."""
        if self.transcript:
            self.transcript.scroll_end(animate=False)

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        if self.input_widget:
            self.input_widget.focus()

    async def action_clear(self) -> None:
        """Clear the transcript, conversation history, and HTTP response cache."""
        if self.transcript:
            self.transcript.clear()
        self._history = []
        try:
            from askpanda_atlas._cache import clear as _cache_clear  # type: ignore[import]
            _cache_clear()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self._write_system("Transcript cleared, context memory reset, and HTTP cache flushed.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter on the input field.

        Questions are placed on ``_question_queue`` immediately so the input
        widget stays responsive while a previous response is in progress.
        The worker drains the queue and calls ``_handle_question`` one at a
        time, preserving submission order.

        Args:
            event (Input.Submitted): Submitted input event.
        """
        text = (event.value or "").strip()
        if self.input_widget:
            self.input_widget.value = ""

        if not text:
            self.action_focus_input()
            return

        # Append to command history, avoiding consecutive duplicates.
        if not self._command_history or self._command_history[-1] != text:
            self._command_history.append(text)
        self._cmd_history_pos = -1

        if text.startswith("/"):
            # Slash commands are lightweight — run them directly.
            await self._handle_command(text)
        else:
            # Enqueue the question; _question_worker will process it when ready.
            await self._question_queue.put(text)
            if self._is_thinking:
                # Let the user know their question is queued.
                n = self._question_queue.qsize()
                self._write_system(
                    f"Queued (position {n}) — will answer once the current response completes."
                )

        self.action_focus_input()

    async def _question_worker(self) -> None:
        """Drain the question queue one item at a time.

        Runs for the lifetime of the app.  Waits for a question, processes it
        fully via ``_handle_question``, then moves to the next.  This ensures
        responses are always returned in submission order and the input widget
        is never blocked.

        ``asyncio.CancelledError`` is re-raised so Textual can cancel this
        task cleanly on shutdown without logging a spurious error.
        """
        try:
            while True:
                question = await self._question_queue.get()
                self._is_thinking = True
                try:
                    await self._handle_question(question)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    self._write_error(f"Unexpected error: {exc}")
                finally:
                    self._is_thinking = False
                    self._question_queue.task_done()
        except asyncio.CancelledError:
            pass  # Clean shutdown — exit silently.

    async def _refresh_tools(self) -> None:
        """Refresh the tool list from the MCP server."""
        res = await self._to_thread(self.mcp.list_tools)
        self.tool_names = sorted(_tool_names_from_list_tools(res))

    def _detect_answer_tool(self) -> None:
        """Select a preferred answer tool based on available tools."""
        for name in ANSWER_TOOL_CANDIDATES:
            if name in self.tool_names:
                self.answer_tool = name
                return
        for name in self.tool_names:
            if "answer" in name:
                self.answer_tool = name
                return
        self.answer_tool = None

    async def _load_banner(self) -> None:
        """Load banner/branding from <plugin>.ui_manifest if available."""
        tool = f"{self.plugin_id}.ui_manifest"
        if tool not in self.tool_names:
            self.banner_lines = FALLBACK_BANNER[:]
            self.display_name = f"Bamboo MCP – {self.plugin_id.upper()}"
            self.help_text = "Enter to send • /help"
            return

        try:
            res = await self._to_thread(self.mcp.call_tool, tool, {})
            txt = _extract_text(res)
            if self.debug_mode:
                self._write_tool(tool, {}, res)

            manifest = json.loads(txt) if txt else {}
            if isinstance(manifest, dict):
                self.display_name = str(manifest.get("display_name") or self.display_name)

                banner = manifest.get("banner")
                if isinstance(banner, list) and all(isinstance(x, str) for x in banner):
                    self.banner_lines = banner
                else:
                    self.banner_lines = FALLBACK_BANNER[:]

                help_text = manifest.get("help")
                if isinstance(help_text, str) and help_text.strip():
                    self.help_text = help_text.strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            import sys as _sys
            print(f"[bamboo] _load_banner failed for tool '{tool}': {exc}", file=_sys.stderr)
            self.banner_lines = FALLBACK_BANNER[:]
            self.display_name = f"Bamboo MCP – {self.plugin_id.upper()}"
            self.help_text = "Enter to send • /help"

    def _render_banner(self) -> None:
        """Render the banner panel and resize the container to fit.

        Height formula: 2 (Panel borders) + 2 (CSS padding top+bottom) +
        number of banner lines.  This ensures the bottom border is always
        visible regardless of how tall the plugin's banner art is.
        """
        if not self.banner_widget:
            return
        banner_text = "\n".join(self.banner_lines)
        self.banner_widget.update(
            Panel(
                Text(banner_text, style="#0b80c3", no_wrap=True),
                title=self.display_name,
                subtitle=f"plugin={self.plugin_id}",
                border_style="#1b90f3",
                padding=(0, 1),
                expand=True,
            )
        )
        # Resize the outer Vertical container so the bottom border is never clipped.
        n_lines = len(self.banner_lines)
        target_height = n_lines + 4  # 2 Panel borders + 2 CSS padding rows
        banner_container = self.query_one("#banner")
        banner_container.styles.height = target_height
        banner_container.styles.min_height = target_height

    async def on_key(self, event: Key) -> None:
        """Handle arrow-up/down for command history recall in the input field.

        Arrow-up navigates backwards through :attr:`_command_history`;
        arrow-down navigates forwards.  Any other key resets the browsing
        position so the next up-press always starts from the most recent entry.

        Args:
            event: The key event from Textual.
        """
        if not self.input_widget or not self.input_widget.has_focus:
            return

        if not self._command_history:
            return

        if event.key == "up":
            event.stop()
            event.prevent_default()
            if self._cmd_history_pos == -1:
                # Start browsing from the most recent entry.
                self._cmd_history_pos = len(self._command_history) - 1
            elif self._cmd_history_pos > 0:
                self._cmd_history_pos -= 1
            self.input_widget.value = self._command_history[self._cmd_history_pos]
            # Move cursor to end of the restored text.
            self.input_widget.cursor_position = len(self.input_widget.value)

        elif event.key == "down":
            event.stop()
            event.prevent_default()
            if self._cmd_history_pos == -1:
                return  # Already at the empty prompt — nothing to do.
            if self._cmd_history_pos < len(self._command_history) - 1:
                self._cmd_history_pos += 1
                self.input_widget.value = self._command_history[self._cmd_history_pos]
            else:
                # Past the end — clear the input and stop browsing.
                self._cmd_history_pos = -1
                self.input_widget.value = ""
            self.input_widget.cursor_position = len(self.input_widget.value)

        else:
            # Any other key resets history browsing.
            self._cmd_history_pos = -1

    async def _fetch_cric_table(self, sentinel: str) -> str:
        """Fetch the pre-formatted CRIC full-list table from the temp file.

        ``execute_plan`` writes the full table to ``BAMBOO_CRIC_TABLE_FILE``
        and returns only a short sentinel ``"__CRIC_TABLE_READY__:<N>"``
        through the MCP pipe to avoid macOS pipe-buffer truncation (8 KB
        limit for subprocess stdout pipes).  This method reads the table
        directly from the temp file, which has no size restriction.

        Args:
            sentinel: The sentinel string from execute_plan, e.g.
                ``"__CRIC_TABLE_READY__:230"``.

        Returns:
            The formatted table string, or a fallback message if the read fails.
        """
        row_count_str = sentinel.split(":", 1)[1].strip()
        try:
            row_count = int(row_count_str)
        except ValueError:
            row_count = 0
        table_text = ""
        read_error = ""
        try:
            with open(self._cric_table_file, "r", encoding="utf-8") as _fh:
                table_text = _fh.read()
        except FileNotFoundError:
            read_error = f"File not found: {self._cric_table_file}"
        except Exception as _exc:  # pylint: disable=broad-exception-caught
            read_error = f"Read error: {_exc}"
        if table_text:
            return table_text
        # Surface diagnostic info so the user can see what went wrong
        diag = f" ({read_error})" if read_error else ""
        return (
            f"{row_count} PanDA queues — table file unavailable{diag}.\n"
            f"Tip: the table was recovered via fetch-cap fallback. "
            f"Try asking again after restarting the TUI to reload the server."
        )

    async def _handle_question(self, question: str) -> None:
        """Send a question to the answer tool and render the response.

        Appends the user turn to the in-memory history, sends the full history
        to the server on each call, then records the assistant reply.  History
        is capped at ``_MAX_HISTORY_TURNS`` user+assistant pairs so the context
        window stays manageable.

        Displays a "thinking" indicator while the server processes the request,
        then replaces it with the assistant response on completion.

        Args:
            question (str): User prompt text.
        """
        self._write_user(question)

        if not self._mcp_ready:
            self._write_system("Not connected yet. Please try again in a moment…")
            return

        if not self.answer_tool:
            self._write_error("No answer tool found. Try /tools to inspect server tools.")
            return

        # Pre-dispatch superuser guard: block questions routing to superuser
        # tools when the session is not authenticated.
        if (
            _SUPERUSER_PASSWORD
            and not self._superuser
            and _is_superuser_question(question, self.tool_names)
        ):
            self._write_system(
                "🔒 This question requires superuser mode. "
                "Run /superuser <password> to unlock developer tools."
            )
            return

        # Append the current user turn to history before sending.
        self._history.append({"role": "user", "content": question})

        args: Dict[str, Any] = {
            "question": question,
            "messages": list(self._history),
            "include_raw": self.debug_mode,
            "bypass_fast_path": not self.fast_path_mode,
        }

        # Store task/job IDs if present so /json and /inspect can access them.
        try:
            import re as _re
            m = _re.search(r"\btask[:#/\-\s]+(\d{4,12})\b", question, _re.IGNORECASE)
            if m:
                self._last_task_id = int(m.group(1))
            m_job = _re.search(
                r"\b(?:job|pandaid|panda[\s_-]?id)[:#/\-\s]+(\d{4,12})\b",
                question, _re.IGNORECASE,
            )
            if m_job:
                self._last_job_id = int(m_job.group(1))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        thinking: bool = self._write_thinking()
        try:
            _pre_pos = self._snapshot_trace_position()
            res = await self._to_thread(self.mcp.call_tool, self.answer_tool, args)
            self._last_spans = self._collect_spans(_pre_pos)
            self._last_raw_result = res  # Available via /json
            # Accumulate session totals from this request's spans.
            _req_est = _estimate_cost(self._last_spans)
            self._session_input_tokens += _req_est["total_input"]
            self._session_output_tokens += _req_est["total_output"]
            self._session_cost_usd += _req_est["total_cost_usd"]
            self._session_request_count += 1
            if self.debug_mode:
                self._write_tool(self.answer_tool, args, res)

            out = _extract_text(res) or "*(No text output; enable /debug on to see raw result.)*"

            # CRIC full-list bypass: execute_plan returns a short sentinel
            # "__CRIC_TABLE_READY__:<N>" instead of the full table to avoid
            # macOS pipe-buffer truncation (8 KB limit).  The formatted table
            # is fetched via a second bamboo_last_evidence(mode="table") call.
            if out.startswith("__CRIC_TABLE_READY__:"):
                out = await self._fetch_cric_table(out)

            # Strip Mermaid diagram blocks before display and history storage.
            # The TUI cannot render Mermaid; raw syntax in history pollutes the
            # LLM context window on follow-up questions.
            clean_out, diagram_count = _strip_mermaid_blocks(out)
            if diagram_count:
                clean_out += (
                    f"\n\n_[{diagram_count} Mermaid diagram{'s' if diagram_count > 1 else ''} "
                    f"omitted — open in Streamlit to view.]_"
                )

            self._replace_thinking(thinking, clean_out)

            # Record the assistant reply (clean, no Mermaid syntax) and enforce the history cap.
            self._history.append({"role": "assistant", "content": clean_out})
            self._cap_history()

            # Auto-chart: silently append a pilot chart when the response
            # came from panda_harvester_workers and has multiple statuses.
            await self._try_auto_chart()

            # Poll the server for any buffered OpenSearch prompt-log events
            # (write confirmations or errors) and surface them in the transcript.
            await self._fetch_promptlog_events()

        except Exception as exc:
            self._replace_thinking(thinking, None)
            self._write_error(str(exc))
            # Remove the user turn we optimistically appended on failure.
            if self._history and self._history[-1] == {"role": "user", "content": question}:
                self._history.pop()

    def _cap_history(self) -> None:
        """Trim ``_history`` to at most ``_MAX_HISTORY_TURNS`` user+assistant pairs.

        Each pair consists of one user message followed by one assistant
        message (2 entries).  The most recent turns are kept; older turns
        are discarded from the front of the list.
        """
        max_messages = _MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    async def _handle_command(self, cmdline: str) -> None:
        """Handle slash commands.

        Dispatches the ``/task <id>`` and ``/job <id>`` shorthands directly,
        then delegates all other commands to :meth:`_dispatch_slash_command`.

        Args:
            cmdline: Raw command line beginning with ``/``.
        """
        m_task = _TASK_CMD_RE.match(cmdline.strip())
        if m_task:
            task_id = m_task.group(1)
            await self._handle_question(
                f"Summarize the status of task {task_id} including dataset info."
            )
            return

        m_job = _JOB_CMD_RE.match(cmdline.strip())
        if m_job:
            job_id = m_job.group(1)
            await self._handle_question(
                f"Analyse the failure of job {job_id} and explain why it failed."
            )
            return

        m_rate = _RATE_CMD_RE.match(cmdline.strip())
        if m_rate:
            await self._handle_rate_command(int(m_rate.group(1)))
            return

        await self._dispatch_slash_command(cmdline)

    async def _handle_rate_command(self, rating: int) -> None:
        """Submit a star rating for the most recent response.

        Args:
            rating: Integer 1–5.
        """
        _STARS = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
        if self._last_rated_doc is None:
            self._write_system(
                "No response to rate yet — ask a question first."
            )
            return
        if "bamboo_promptlog_rate" not in self.tool_names:
            self._write_system(
                "bamboo_promptlog_rate tool not available — "
                "is BAMBOO_OPENSEARCH_PROMPTLOG set on the server?"
            )
            return
        index, doc_id = self._last_rated_doc
        try:
            res = await self._to_thread(
                self.mcp.call_tool,
                "bamboo_promptlog_rate",
                {"index": index, "doc_id": doc_id, "rating": rating},
            )
            text = _extract_text(res) or "{}"
            parsed = json.loads(text)
            if parsed.get("error"):
                self._write_system(f"Rating failed: {parsed['error']}")
            else:
                stars = _STARS.get(rating, str(rating))
                self._write_system(
                    f"Rated {stars} ({rating}/5) — "
                    f"index={index!r} id={doc_id!r}"
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_system(f"Rating error: {exc}")

    async def _dispatch_slash_command(self, cmdline: str) -> None:  # noqa: C901
        """Dispatch a slash command to the appropriate handler.

        Split out from :meth:`_handle_command` to keep cyclomatic complexity
        under the project limit (max-complexity = 15).

        Args:
            cmdline (str): Raw command line beginning with ``/``.
        """
        parts = cmdline.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/help", "/?"):
            self._cmd_help()
            return
        if cmd in ("/exit", "/quit"):
            self.exit()
            return
        if cmd == "/tools":
            await self._cmd_tools()
            return
        if cmd == "/tracing":
            self._handle_tracing_command()
            return
        if cmd == "/json":
            await self._handle_json_command()
            return
        if cmd == "/inspect":
            await self._handle_inspect_command()
            return
        if cmd == "/chart":
            await self._handle_chart_command()
            return
            self._handle_history_command()
            return
        if cmd == "/plugin":
            await self._cmd_plugin(args)
            return
        if cmd == "/debug":
            self._cmd_debug(args)
            return
        if cmd == "/fastpath":
            self._cmd_fastpath(args)
            return
        if cmd == "/costs":
            self._cmd_costs()
            return
        if cmd == "/clear":
            await self.action_clear()
            return
        if cmd == "/links":
            self._cmd_links(args)
            return
        if cmd == "/superuser":
            self._cmd_superuser(args)
            return
        if cmd == "/rates":
            scope = args[0].lower() if args else ""
            if scope == "today":
                date_clause = (
                    "today (filter @timestamp gte:now/d)"
                )
            elif scope == "week":
                date_clause = "this week (filter @timestamp gte:now-7d/d)"
            elif scope == "month":
                date_clause = "this month (filter @timestamp gte:now-30d/d)"
            else:
                date_clause = "all time (no date filter)"
            rates_q = (
                f"Show all rated responses for {date_clause} as a markdown "
                "table sorted by rating descending. "
                "Call the opensearch_promptlog_query tool. "
                "Pass max_hits=50 as a separate argument. "
                "Pass source_fields=[@timestamp,turn_number,session_id,"
                "model,tools_used,raw_question,rating] as a separate argument. "
                "Pass this exact JSON as the query argument (nothing else): "
                '{"query":{"exists":{"field":"rating"}},'
                '"sort":[{"rating":{"order":"desc"}}]} '
                "Each hit includes _id automatically. "
                "Format as markdown table with these exact columns: "
                "Doc ID (full _id value) | Time | Turn | "
                "Session (first 8 chars) | Model | "
                "Tools | Question (raw_question field, first 60 chars) | Rating. "
                "The Doc ID is the only way to retrieve the full response. "
            )
            await self._handle_question(rates_q)
            return
        if cmd == "/faq":
            scope = args[0].lower() if args else ""
            if scope == "today":
                faq_q = (
                    "What are the most frequently asked questions in Bamboo today? "
                    "Use a terms aggregation on raw_question.keyword filtered to gte:now/d."
                )
            elif scope == "week":
                faq_q = (
                    "What are the most frequently asked questions in Bamboo this week? "
                    "Use a terms aggregation on raw_question.keyword filtered to gte:now-7d/d."
                )
            elif scope == "month":
                faq_q = (
                    "What are the most frequently asked questions in Bamboo this month? "
                    "Use a terms aggregation on raw_question.keyword filtered to gte:now-30d/d."
                )
            else:
                faq_q = (
                    "What are the most frequently asked questions in Bamboo across all time? "
                    "Use a terms aggregation on raw_question.keyword with no date filter."
                )
            await self._handle_question(faq_q)
            return
        if cmd == "/script":
            self._cmd_script(args[0] if args else "")
            return
        self._write_system(f"Unknown command: {cmdline} (try /help)")

    def _cmd_help(self) -> None:
        """Display the slash-command reference."""
        self._write_system(
            "Commands:\n"
            "  /help                 Show this help\n"
            "  /tools                List tools exposed by the MCP server\n"
            "  /task <id>             Shorthand for: status of task <id>\n"
            "  /job <id>              Shorthand for: analyse failure of job <id>\n"
            "  /json                 Show verbatim BigPanDA API response (raw server JSON)\n"
            "  /inspect              Show structured evidence dict (job counts, sites, errors)\n"
            "  /chart                Show ASCII bar chart of Harvester pilot counts by status\n"
            "  /faq [today|week|month]  Most frequently asked questions (default: all time)\n"
            "  /rates [today|week|month] Show rated responses as a table (default: all time)\n"
            "  /script [filename]       Write code block(s) from last response to file\n"
            "  /rate <1-5>           Rate the last response (1=poor, 5=excellent)\n"
            "  /tracing              Show timing + trace spans for last request\n"
            "  /history              Show turns currently held in context memory\n"
            "  /plugin <id>          Switch plugin (affects banner tool name)\n"
            "  /debug on|off         Toggle debug (shows tool calls + raw results)\n"
            "  /fastpath on|off      Toggle deterministic fast-path routing (off → use LLM planner)\n"
            "  /costs                Show estimated LLM token cost for the last request + session total\n"
            "  /clear                Clear transcript, context memory, and HTTP cache\n"
            "  /links [N]            List links from last response; /links N opens link N in browser\n"
            "  /superuser <pw>       Unlock developer mode (requires BAMBOO_SUPERUSER_PASSWORD)\n"
            "  /exit, /quit          Exit the app\n"
            "\n"
            "Tip: Use PageUp/PageDown to scroll. To copy text, hold Option (macOS) or\n"
            "     Shift (Linux/Windows) while selecting with the mouse.\n"
            "Use --no-inline to use the alternate screen."
        )

    async def _cmd_tools(self) -> None:
        """Refresh and display the list of tools registered on the MCP server."""
        if self._mcp_ready:
            await self._refresh_tools()
        if not self.tool_names:
            self._write_system("No tools returned by server.")
            return
        body = "\n".join(f"- `{n}`" for n in self.tool_names)
        self._write_panel(Markdown(body), title=f"{_now()}  tools")

    async def _cmd_plugin(self, args: List[str]) -> None:
        """Switch the active plugin and reload banner and tools.

        Args:
            args (List[str]): Remaining command tokens; first element is the
                new plugin ID when present.
        """
        if not args:
            self._write_system(f"Current plugin: {self.plugin_id}")
            return
        self.plugin_id = args[0]
        if self._mcp_ready:
            await self._refresh_tools()
            self._detect_answer_tool()
            await self._load_banner()
            self._render_banner()
        else:
            self._render_banner_placeholder()
        self._write_system(f"Switched plugin to: {self.plugin_id}")

    def _cmd_debug(self, args: List[str]) -> None:
        """Toggle or explicitly set debug mode.

        Args:
            args (List[str]): Remaining command tokens; first element may be
                ``"on"`` or ``"off"``.
        """
        if args and args[0].lower() in ("on", "off"):
            self.debug_mode = args[0].lower() == "on"
        else:
            self.debug_mode = not self.debug_mode
        self._write_system(f"Debug {'ON' if self.debug_mode else 'OFF'}.")

    def _cmd_fastpath(self, args: List[str]) -> None:
        """Toggle or explicitly set fast-path routing mode.

        When fast-path is ON (default) the deterministic routing intercepts
        handle pilot, jobs DB, and site-health questions without an LLM call.
        When OFF, those intercepts are skipped and the question falls through
        to the topic guard and LLM planner — useful for testing planner routing
        on questions that would normally be short-circuited.

        Args:
            args (List[str]): Remaining command tokens; first element may be
                ``"on"`` or ``"off"``.
        """
        if args and args[0].lower() in ("on", "off"):
            self.fast_path_mode = args[0].lower() == "on"
        else:
            self.fast_path_mode = not self.fast_path_mode
        status = "ON (deterministic routing)" if self.fast_path_mode else "OFF (LLM planner)"
        self._write_system(f"Fast-path routing {status}.")

    def _cmd_costs(self) -> None:
        """Show estimated LLM token cost for the last request and session total.

        Reads ``input_tokens`` and ``output_tokens`` from every ``llm_call``
        span in ``_last_spans``, prices them using :data:`_MODEL_COST_PER_MTOK`,
        and presents a per-call breakdown table plus a request total.  A second
        section shows the running session total accumulated across all completed
        requests since the TUI was started.  Models not in the pricing table
        fall back to the default rate and are flagged.
        """
        if not self._last_spans:
            self._write_system(
                "No request data yet — ask a question first, then type /costs."
            )
            return

        est = _estimate_cost(self._last_spans)
        calls = est["calls"]

        if not calls:
            self._write_system(
                "No LLM calls found in last request spans.\n"
                "This can happen if tracing is disabled (BAMBOO_TRACE must be set) "
                "or if the last request was answered from cache."
            )
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Input tok", justify="right", style="yellow")
        table.add_column("Output tok", justify="right", style="yellow")
        table.add_column("Rate ($/Mtok in/out)", style="dim")
        table.add_column("Cost (USD)", justify="right", style="green")

        for call in calls:
            rate_str = f"{call['rate_in']:.3f} / {call['rate_out']:.3f}"
            cost_str = f"${call['cost_usd']:.6f}"
            label = f"{call['provider']}/{call['model']}" if call["provider"] else call["model"]
            table.add_row(
                label,
                f"{call['input_tokens']:,}",
                f"{call['output_tokens']:,}",
                rate_str,
                cost_str,
            )

        table.add_section()
        total_cost = est["total_cost_usd"]
        table.add_row(
            "[bold]This request[/bold]",
            f"[bold]{est['total_input']:,}[/bold]",
            f"[bold]{est['total_output']:,}[/bold]",
            "",
            f"[bold green]${total_cost:.6f}[/bold green]",
        )

        # Session total row — accumulated across all requests since startup.
        if self._session_request_count > 0:
            table.add_section()
            req_word = "request" if self._session_request_count == 1 else "requests"
            table.add_row(
                f"[bold cyan]Session ({self._session_request_count} {req_word})[/bold cyan]",
                f"[bold cyan]{self._session_input_tokens:,}[/bold cyan]",
                f"[bold cyan]{self._session_output_tokens:,}[/bold cyan]",
                "",
                f"[bold cyan]${self._session_cost_usd:.6f}[/bold cyan]",
            )

        note = ""
        if est["unknown_models"]:
            unknown = ", ".join(est["unknown_models"])
            note = (
                f"\n⚠  Unknown model(s): {unknown}\n"
                f"   Rates defaulted to ${_DEFAULT_COST_PER_MTOK[0]:.2f}/"
                f"${_DEFAULT_COST_PER_MTOK[1]:.2f} per Mtok. "
                "Add the model to _MODEL_COST_PER_MTOK in chat.py for accurate pricing."
            )

        self._write_panel(table, title=f"{_now()}  costs (estimated)", border_style="green")
        if note:
            self._write_system(note)

    def _cmd_superuser(self, args: List[str]) -> None:
        """Attempt to unlock superuser / developer mode with a password.

        Checks the supplied password against :data:`_SUPERUSER_PASSWORD`
        using a constant-time compare.  On success, sets
        :attr:`_superuser` to ``True`` for the lifetime of the session.
        On failure, prints a generic rejection message that does not
        reveal whether a password is configured.

        Usage: ``/superuser <password>``

        Args:
            args (List[str]): Remaining command tokens; first element is
                the password attempt.
        """
        if not _SUPERUSER_PASSWORD:
            self._write_system(
                "Superuser mode is not configured "
                "(set BAMBOO_SUPERUSER_PASSWORD to enable)."
            )
            return
        if not args:
            self._write_system("Usage: /superuser <password>")
            return
        attempt = args[0]
        if hmac.compare_digest(attempt.encode(), _SUPERUSER_PASSWORD.encode()):
            self._superuser = True
            self._write_system(
                "🔓 Superuser mode unlocked. Developer tools are now active."
            )
        else:
            self._write_system("Incorrect password.")

    def _cmd_links(self, args: List[str]) -> None:
        """List links from the last assistant response, or open one by number.

        With no arguments, prints all ``(label, url)`` pairs found in the
        most recent AskPanDA response, numbered from 1.  With a numeric
        argument ``N``, opens link number N in the system default browser via
        :func:`webbrowser.open`.

        Args:
            args (List[str]): Parsed command arguments.  Empty for listing;
                one element (digit string) to open a specific link.
        """
        if not self._last_response_links:
            self._write_system(
                "No links found in the last response.\n"
                "Ask a question that returns links, then type /links."
            )
            return

        # Opening a specific link by 1-based index.
        if args:
            try:
                idx = int(args[0])
            except ValueError:
                self._write_system(f"Usage: /links [N] — N must be a number, got {args[0]!r}")
                return
            if idx < 1 or idx > len(self._last_response_links):
                self._write_system(
                    f"Link number {idx} is out of range "
                    f"(1–{len(self._last_response_links)}).  Type /links to list them."
                )
                return
            label, url = self._last_response_links[idx - 1]
            webbrowser.open(url)
            self._write_system(f"Opened in browser: {label}\n{url}")
            return

        # Listing mode — show all links numbered.
        lines_out = [f"Links in last response ({len(self._last_response_links)}):"]
        for i, (label, url) in enumerate(self._last_response_links, start=1):
            if label == url:
                lines_out.append(f"  {i}.  {url}")
            else:
                lines_out.append(f"  {i}.  {label}")
                lines_out.append(f"       {url}")
        lines_out.append("")
        lines_out.append("Type /links N to open link N in your browser.")
        self._write_system("\n".join(lines_out))

    def _handle_history_command(self) -> None:
        """Render the current in-context conversation turns as a Rich table.

        Shows each turn's role, a character count, and a truncated preview of
        the content.  Useful for debugging follow-up resolution and verifying
        that the history cap is working as expected.
        """
        if not self._history:
            turns_word = "turn" if _MAX_HISTORY_TURNS == 1 else "turns"
            self._write_system(
                f"No conversation history in context yet.\n"
                f"(Cap: {_MAX_HISTORY_TURNS} {turns_word} = {_MAX_HISTORY_TURNS * 2} messages. "
                f"Set BAMBOO_HISTORY_TURNS to change.)"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Role", style="cyan", no_wrap=True)
        table.add_column("Chars", justify="right", style="yellow", no_wrap=True)
        table.add_column("Preview", style="")

        for idx, msg in enumerate(self._history, start=1):
            role = str(msg.get("role", "?"))
            content = str(msg.get("content", ""))
            preview = content[:120].replace("\n", " ")
            if len(content) > 120:
                preview += "…"
            table.add_row(str(idx), role, str(len(content)), preview)

        pair_count = len(self._history) // 2
        remainder = len(self._history) % 2
        summary = f"{pair_count} complete pair(s)"
        if remainder:
            summary += " + 1 pending user turn"
        summary += f"  |  cap: {_MAX_HISTORY_TURNS} pairs ({_MAX_HISTORY_TURNS * 2} messages)"

        table.add_section()
        table.add_row("", "[bold]total[/bold]", f"[bold yellow]{len(self._history)}[/bold yellow]", summary)

        self._write_panel(table, title=f"{_now()}  context history", border_style="cyan")

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------

    def _snapshot_trace_position(self) -> int:
        """Return the current end-of-file byte position of the trace file.

        Called immediately before each MCP tool invocation so that
        :meth:`_collect_spans` can read only the spans produced by that
        specific request.

        Returns:
            Current file size in bytes, or 0 if tracing is not configured.
        """
        try:
            return os.path.getsize(self._trace_file)
        except OSError:
            return 0

    def _collect_spans(self, position: int) -> List[Dict[str, Any]]:
        """Read trace spans written to the trace file since *position*.

        Args:
            position: Byte offset obtained from :meth:`_snapshot_trace_position`
                before the request was issued.

        Returns:
            List of parsed span dicts in emission order.  Non-JSON lines and
            lines without the ``"bamboo_trace"`` sentinel are silently skipped.
        """
        spans: List[Dict[str, Any]] = []
        try:
            with open(self._trace_file, "r", encoding="utf-8") as fh:
                fh.seek(position)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and obj.get("bamboo_trace"):
                            spans.append(obj)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return spans

    def _handle_tracing_command(self) -> None:
        """Render a Rich table of trace spans for the most recent request.

        Displays event type, tool name, duration, and the most important
        event-specific fields (route, backend/hits, guard verdict, token
        counts) in a compact panel.  If no spans are available — because
        tracing is disabled or no request has been made yet — a helpful
        explanation is shown instead.
        """
        if not self._last_spans:
            if self.cfg.transport == "http":
                self._write_system(
                    "No trace spans available.\n\n"
                    "HTTP transport: set BAMBOO_TRACE=1 and BAMBOO_TRACE_FILE on the server "
                    "process, then tail the file manually:\n"
                    "  tail -f $BAMBOO_TRACE_FILE | grep bamboo_trace | jq ."
                )
            else:
                self._write_system(
                    "No trace spans yet — ask a question first, then type /tracing."
                )
            return

        # Build the summary table.
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Event", style="cyan", no_wrap=True)
        table.add_column("Tool", style="", no_wrap=True)
        table.add_column("ms", justify="right", style="yellow", no_wrap=True)
        table.add_column("Detail", style="dim")

        total_ms: float = 0.0
        for span_rec in self._last_spans:
            event = str(span_rec.get("event", ""))
            tool = str(span_rec.get("tool", ""))
            duration = float(span_rec.get("duration_ms", 0.0))
            if event == "tool_call":
                total_ms = duration  # outermost span — use as request total

            detail = _span_detail(span_rec)
            table.add_row(event, tool, f"{duration:.0f}", detail)

        # Footer row with total.
        table.add_section()
        table.add_row(
            "[bold]total[/bold]",
            "",
            f"[bold yellow]{total_ms:.0f}[/bold yellow]",
            "wall time for full tool_call span",
        )

        self._write_panel(table, title=f"{_now()}  tracing", border_style="cyan")

    def _write_panel(self, renderable: Any, title: str, border_style: str = "dim") -> None:
        """Write a panel to the transcript.

        Args:
            renderable (Any): Rich renderable content.
            title (str): Panel title.
            border_style (str): Rich border style name.
        """
        if not self.transcript:
            return

        # Width of the transcript region (fallback to app width)
        width = None
        try:
            width = max(10, self.transcript.size.width)
        except Exception:
            try:
                width = max(10, self.size.width)
            except Exception:
                width = None

        self.transcript.write(
            Panel(
                renderable,
                title=title,
                border_style=border_style,
                expand=True,
                width=width,  # <-- this is the key
                padding=(0, 1),
            )
        )

        self.transcript.scroll_end(animate=False)
        if self.input_widget:
            self.input_widget.focus()

    def _write_panel_old(self, renderable: Any, title: str, border_style: str = "dim") -> None:
        """Write a panel using the legacy rendering path.

        Args:
            renderable (Any): Rich renderable content.
            title (str): Panel title.
            border_style (str): Rich border style name.
        """
        if not self.transcript:
            return
        self.transcript.write(Panel(renderable, title=title, border_style=border_style))

        if self.transcript:
            self.transcript.scroll_end(animate=False)
        if self.input_widget:
            self.input_widget.focus()

        self.action_focus_input()

    async def _fetch_promptlog_events(self) -> None:
        """Poll the server for buffered OpenSearch prompt-log events and display them.

        Calls ``bamboo_promptlog_status`` on the MCP server, which drains the
        server-side event ring buffer and returns write confirmations and errors
        accumulated since the previous call.  Each event is rendered as a system
        or error panel in the transcript.

        ``log_prompt`` fires as a background ``asyncio.create_task`` inside
        ``call_llm`` and performs a synchronous OpenSearch network call via
        ``asyncio.to_thread``.  This method therefore retries up to
        ``_PROMPTLOG_POLL_ATTEMPTS`` times with ``_PROMPTLOG_POLL_INTERVAL_S``
        seconds between attempts so the background task has time to complete
        before we give up.

        The call is silently skipped when the tool is not registered on the
        server (i.e. prompt logging is disabled or the server is an older
        version without the tool).
        """
        _PROMPTLOG_POLL_ATTEMPTS = 6
        _PROMPTLOG_POLL_INTERVAL_S = 0.5

        if "bamboo_promptlog_status" not in self.tool_names:
            return
        try:
            for _ in range(_PROMPTLOG_POLL_ATTEMPTS):
                await asyncio.sleep(_PROMPTLOG_POLL_INTERVAL_S)
                res = await self._to_thread(
                    self.mcp.call_tool, "bamboo_promptlog_status", {}
                )
                text = _extract_text(res) or "{}"
                parsed = json.loads(text)
                events = parsed.get("events") or []
                if events:
                    for event in events:
                        severity = str(event.get("severity", "info"))
                        message = str(event.get("message", ""))
                        if not message:
                            continue
                        # Extract (index, doc_id) so /rate can reference
                        # the most recently indexed turn.
                        _m = re.search(
                            r"index='([^']+)'.*?id='([^']+)'",
                            message,
                        )
                        if _m:
                            self._last_rated_doc = (_m.group(1), _m.group(2))
                        if severity == "error":
                            self._write_error(message)
                        else:
                            self._write_system(message)
                    return  # events received — done
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Never let observability polling affect the main response path.

    #: Phase labels shown in sequence while the server is working.
    #: Each phase is shown for ``_THINKING_TICKS_PER_PHASE`` ticks before
    #: advancing to the next one (then cycling).  Dot count cycles every tick
    #: independently so the animation always appears live within each phase.
    _THINKING_PHASES: tuple[str, ...] = (
        "Routing question",
        "Retrieving evidence",
        "Synthesising answer",
    )
    #: Number of 1-second ticks to display each phase before advancing.
    _THINKING_TICKS_PER_PHASE: int = 4

    def _write_thinking(self) -> bool:
        """Start the animated thinking indicator below the transcript.

        Uses :meth:`set_interval` (Textual's own timer mechanism) to update
        the indicator every second.  Each tick advances the dot sequence
        (``·`` → ``··`` → ``···``) to confirm liveness.  The phase label
        (Routing / Retrieving / Synthesising) advances every
        :attr:`_THINKING_TICKS_PER_PHASE` ticks so the user sees a meaningful
        description of the current stage throughout the wait.

        :meth:`set_interval` integrates with Textual's event loop directly,
        avoiding the timing issues that ``asyncio.ensure_future`` + ``asyncio.sleep``
        can cause when the loop is shared with blocking worker threads.

        Returns:
            True if the indicator was started, False if unavailable.
        """
        if not self.thinking_widget:
            return False
        self.thinking_widget.add_class("active")

        dots = [".", "..", "..."]
        counter = [0]  # mutable cell so the closure can increment it

        def _tick() -> None:
            import datetime as _dt
            now = _dt.datetime.now().strftime("%H:%M:%S")
            tick = counter[0]
            phase = self._THINKING_PHASES[
                (tick // self._THINKING_TICKS_PER_PHASE) % len(self._THINKING_PHASES)
            ]
            dot = dots[tick % len(dots)]
            if self.thinking_widget:
                self.thinking_widget.update(
                    Text(f"{now}  {phase}{dot}", style="dim italic"),
                    layout=False,
                )
            counter[0] += 1

        _tick()  # Show the first frame immediately; timer fires subsequent ones
        self._thinking_task = self.set_interval(1.0, _tick)
        return True

    def _replace_thinking(self, indicator: bool, answer: Optional[str]) -> None:
        """Cancel the animated thinking indicator and optionally write the answer.

        Args:
            indicator: Value returned by _write_thinking (True if it was
                started).
            answer: The assistant response text (Markdown), or None to
                simply hide the indicator without writing a response.
        """
        if self._thinking_task is not None:
            self._thinking_task.stop()
            self._thinking_task = None
        if self.thinking_widget:
            self.thinking_widget.remove_class("active")
            self.thinking_widget.update("", layout=False)
        if answer is not None:
            self._write_assistant(answer)

    def _write_system(self, msg: str) -> None:
        """Write a system message.

        Args:
            msg (str): Message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  system", border_style="dim")

    def _write_user(self, msg: str) -> None:
        """Write a user message.

        Args:
            msg (str): Message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  you", border_style="dim")

    @staticmethod
    def _is_preformatted(msg: str) -> bool:
        """Return True when *msg* is a pre-formatted plain-text table, not Markdown.

        Pre-formatted responses (e.g. the direct CRIC full-list table) must be
        rendered with ``Text`` rather than ``Markdown`` to prevent Rich's paragraph
        reflow from collapsing multi-line plain text into a single wrapped block.
        A message is considered pre-formatted when its first line ends with ``":"``
        and the second line starts with two spaces (the indented site column), which
        is the signature layout produced by ``_format_cric_full_list``.

        Args:
            msg (str): Message text to inspect.

        Returns:
            True if the message should be rendered as pre-formatted text.
        """
        lines = msg.splitlines()
        if len(lines) < 2:
            return False
        return lines[0].rstrip().endswith(":") and lines[1].startswith("  ")

    @staticmethod
    def _extract_links(msg: str) -> List[Tuple[str, str]]:
        """Extract hyperlinks from a Markdown response string.

        Collects Markdown inline links ``[label](url)`` first, then picks up
        any bare ``https?://`` URLs that were not already captured as the
        target of a Markdown link.  Duplicates (same URL appearing more than
        once) are deduplicated while preserving first-occurrence order.

        Args:
            msg (str): Raw Markdown (or plain-text) response from the assistant.

        Returns:
            List of ``(label, url)`` tuples in order of first appearance.
            For bare URLs the label equals the URL.
        """
        seen: set[str] = set()
        results: List[Tuple[str, str]] = []

        # 1. Markdown inline links — [label](url)
        for label, url in _MD_LINK_RE.findall(msg):
            if url not in seen:
                seen.add(url)
                results.append((label.strip(), url))

        # 2. Bare URLs not already captured above
        # Build a set of all markdown-link URLs already consumed so we can skip them.
        md_urls: set[str] = {url for _, url in _MD_LINK_RE.findall(msg)}
        for url in _BARE_URL_RE.findall(msg):
            # Strip trailing punctuation that is unlikely to be part of the URL.
            url = url.rstrip(".,;:!?)")
            if url not in seen and url not in md_urls:
                seen.add(url)
                results.append((url, url))

        return results

    @staticmethod
    def _expand_links_for_display(msg: str) -> str:
        """Rewrite Markdown links inside a trailing Links section to show bare URLs.

        Rich's ``Markdown`` renderer displays ``[label](url)`` as underlined
        text with the URL invisible, which is useless in a terminal.  This
        method locates the last ``Links:`` heading in *msg* and rewrites every
        ``[label](url)`` entry within it to ``label — url`` so both the label
        and the full URL are always visible.  Content before the heading is
        left untouched so ordinary inline links in the prose still render with
        Rich Markdown styling.

        Args:
            msg (str): Raw assistant response text.

        Returns:
            Message text with the Links section rewritten for plain display.
        """
        # Find the last "Links:" heading (case-insensitive, on its own line).
        m = re.search(r"\n([^\n]*[Ll]inks:[^\n]*)\n", msg)
        if not m:
            return msg

        split_pos = m.start()
        prose = msg[:split_pos]
        links_block = msg[split_pos:]

        # Replace every [label](url) in the links block with "label — url".
        links_block = _MD_LINK_RE.sub(lambda mo: f"{mo.group(1)} — {mo.group(2)}", links_block)
        return prose + links_block

    def _write_assistant(self, msg: str) -> None:
        """Write an assistant message.

        Pre-formatted plain-text tables (e.g. the CRIC full-list output) are
        rendered with ``Text`` to prevent Rich's Markdown renderer from
        reflowing the table rows into a single paragraph.  All other responses
        are rendered as ``Markdown``.  Links are extracted from the original
        *msg* (before display rewriting) and stored in ``_last_response_links``
        so the user can open them with ``/links``.

        Args:
            msg (str): Message text (Markdown or pre-formatted plain text).
        """
        self._last_response_links = self._extract_links(msg)
        self._last_response_text = msg
        display_msg = self._expand_links_for_display(msg)
        renderable = Text(display_msg) if self._is_preformatted(display_msg) else Markdown(display_msg)
        self._write_panel(renderable, title=f"{_now()}  Bamboo MCP", border_style="dim")
        if self._last_response_links:
            n = len(self._last_response_links)
            word = "link" if n == 1 else "links"
            self._write_system(f"{n} {word} in this response — type /links to list, /links N to open.")

    @staticmethod
    def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
        """Extract fenced code blocks from a Markdown response.

        Matches blocks delimited by triple backticks, optionally with a
        language identifier.  Returns blocks in order of appearance.

        Args:
            text: Raw Markdown response text.

        Returns:
            List of ``(language, code)`` tuples.  ``language`` is the
            identifier after the opening fence (e.g. ``"python"``,
            ``"bash"``), or an empty string when none is specified.
            ``code`` is the block content with leading/trailing whitespace
            stripped.
        """
        import re as _re
        pattern = _re.compile(r"```(\w*)\s*\n(.*?)```", _re.DOTALL)
        return [(lang.lower(), code.strip()) for lang, code in pattern.findall(text)]

    @staticmethod
    def _extract_suggested_filename(text: str) -> str:
        """Extract a suggested script filename from a Markdown response.

        Recognises common patterns the LLM uses to label a script:

        - ``Script: calculate_pi.py``
        - ``**Script:** calculate_pi.py``
        - ``File: calculate_pi.py``
        - ``**File:** calculate_pi.py``
        - ``Filename: calculate_pi.py``
        - Code fence with inline filename: ` ```python calculate_pi.py `

        Args:
            text: Raw Markdown response text.

        Returns:
            The suggested filename string, or an empty string when none
            is found.
        """
        import re as _re
        # "Script: foo.py", "File: foo.py", "Filename: foo.py"
        # with optional Markdown bold markers
        # Match "Script: foo.py", "File: foo.py" etc. — allow any surrounding
        # whitespace, Unicode separators, and optional Markdown bold markers.
        label_re = _re.compile(
            r"(?:^|[\n\r])[ \t]*\*{0,2}(?:Script|File|Filename)\*{0,2}:[ \t]*([\w.][\w.\-]*)"
            r"[ \t]*(?:[\n\r]|$)",
            _re.IGNORECASE | _re.UNICODE,
        )
        m = label_re.search(text)
        if m:
            return m.group(1).strip()
        # Also try a simpler fallback: any word that looks like a filename
        # on a line by itself immediately after "Script" or "File"
        simple_re = _re.compile(
            r"(?:Script|File|Filename):\s*([\w.\-]+\.(?:py|sh|js|ts|cpp|c|java|go|rs|r|sql|yaml|yml|toml|json))"
            r"\b",
            _re.IGNORECASE,
        )
        m = simple_re.search(text)
        if m:
            return m.group(1).strip()
        # Code fence with inline filename: ```python calculate_pi.py
        fence_re = _re.compile(
            r"```\w*\s+([\w.\-]+\.\w+)\s*\n",
        )
        m = fence_re.search(text)
        if m:
            return m.group(1).strip()
        # "Save the script as random_numbers.C" / "name it foo.py" etc.
        save_re = _re.compile(
            r"(?:save|name|call)\s.*?\bas\s+([\w.][\w.\-]*\.\w+)",
            _re.IGNORECASE,
        )
        m = save_re.search(text)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _lang_to_extension(lang: str) -> str:
        """Map a code fence language identifier to a file extension.

        Args:
            lang: Language string from the code fence (e.g. ``"python"``).

        Returns:
            File extension including the leading dot (e.g. ``".py"``).
            Defaults to ``.txt`` for unknown or empty languages.
        """
        _MAP = {
            "python": ".py", "py": ".py",
            "bash": ".sh", "sh": ".sh", "shell": ".sh",
            "javascript": ".js", "js": ".js",
            "typescript": ".ts", "ts": ".ts",
            "json": ".json",
            "yaml": ".yaml", "yml": ".yaml",
            "toml": ".toml",
            "cpp": ".cpp", "c++": ".cpp", "c": ".c", "root": ".C",
            "java": ".java",
            "ruby": ".rb",
            "go": ".go",
            "rust": ".rs",
            "sql": ".sql",
            "r": ".r",
        }
        return _MAP.get(lang.lower(), ".txt")

    def _cmd_script(self, filename: str) -> None:
        """Write code block(s) from the last response to a local file.

        Resolution order for the output filename:
        1. Explicit filename supplied by the user (``/script foo.py``).
        2. Filename suggested in the response body (e.g. ``Script: foo.py``).
        3. Auto-generated name from language + timestamp.

        The proposed path is shown to the user before writing; the write
        proceeds immediately (TUI is single-user/expert context).

        Args:
            filename: Optional target filename supplied by the user.
        """
        import os as _os
        from datetime import datetime as _dt

        if not self._last_response_text:
            self._write_system("No previous response available for /script.")
            return

        blocks = self._extract_code_blocks(self._last_response_text)
        if not blocks:
            self._write_system(
                "No code blocks found in the last response.  "
                "The response must contain a fenced ``` code block."
            )
            return

        cwd = _os.getcwd()
        stamp = _dt.now().strftime("%H%M%S")
        # Try to honour a filename suggested in the response body
        suggested = self._extract_suggested_filename(self._last_response_text)
        written: list[str] = []

        for i, (lang, code) in enumerate(blocks):
            ext = self._lang_to_extension(lang)
            if filename and i == 0:
                # First block: use the user-supplied filename, adding the
                # detected extension if the user omitted one.
                if _os.path.splitext(filename)[1]:
                    target = filename
                else:
                    target = filename + ext
            elif filename and len(blocks) > 1:
                # Subsequent blocks: strip extension from user-supplied name
                # and append each block's own language extension.
                base = _os.path.splitext(filename)[0] or filename
                target = f"{base}_{i + 1}{ext}"
            elif suggested and i == 0:
                # First block: honour the LLM-suggested filename exactly.
                target = suggested
            elif suggested and len(blocks) > 1:
                # Subsequent blocks: use suggested base + own extension.
                base = _os.path.splitext(suggested)[0]
                target = f"{base}_{i + 1}{ext}"
            else:
                label = lang if lang else "script"
                suffix = f"_{i + 1}" if len(blocks) > 1 else ""
                target = f"{label}_{stamp}{suffix}{ext}"

            path = _os.path.join(cwd, target)
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(code + "\n")
                written.append(path)
            except OSError as exc:
                self._write_error(f"/script: could not write {path!r}: {exc}")
                return

        if len(written) == 1:
            self._write_system(f"Script written to: {written[0]}")
        else:
            joined = "\n  ".join(written)
            self._write_system(f"{len(written)} scripts written:\n  {joined}")

    def _write_warning(self, msg: str) -> None:
        """Write a warning message with yellow border.

        Args:
            msg: Warning message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  warning", border_style="yellow")

    def _write_error(self, msg: str) -> None:
        """Write an error message.

        Args:
            msg (str): Error message text.
        """
        self._write_panel(Text(msg), title=f"{_now()}  error", border_style="red")

    def _write_tool(self, tool: str, args: Dict[str, Any], res: Any) -> None:
        """Write a debug tool call block showing args and full raw result.

        Args:
            tool (str): Tool name.
            args (Dict[str, Any]): Tool input arguments.
            res (Any): Raw tool result.
        """
        body = (
            f"**tool:** `{tool}`\n\n"
            f"**args:**\n```json\n{_pretty(args)}\n```\n\n"
            f"**result:**\n```text\n{_extract_text(res) or _pretty(res)}\n```"
        )
        self._write_panel(Markdown(body), title=f"{_now()}  tool", border_style="cyan")

    async def _handle_json_command(self) -> None:
        """Display the verbatim BigPanDA API response for the last task or job.

        Calls ``bamboo_last_evidence`` with ``mode='raw'`` to retrieve the
        response from the server-side evidence store — no fresh HTTP request.
        Shows the full BigPanDA payload as returned by the API before any
        summarisation.  Use ``/inspect`` for the compact evidence dict instead.
        """
        if not self._mcp_ready:
            self._write_system("Not connected yet.")
            return
        try:
            res = await asyncio.wait_for(
                self._to_thread(self.mcp.call_tool, "bamboo_last_evidence", {"mode": "raw"}),
                timeout=30,
            )
            text = _extract_text(res)
            if not text:
                self._write_system("No evidence stored yet — ask about a task or job first.")
                return
            parsed = json.loads(text)
            if "error" in parsed:
                self._write_system(parsed["error"])
                return
            payload = parsed.get("evidence", parsed)
            tool = parsed.get("tool", "")
            title = f"{_now()}  raw JSON — {tool}" if tool else f"{_now()}  raw JSON"
            self._write_panel(
                Markdown(f"```json\n{_pretty(payload)}\n```"),
                title=title,
                border_style="yellow",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_error(f"Could not fetch raw JSON: {exc}")

    async def _handle_inspect_command(self) -> None:
        """Dump the structured evidence dict for the last task or job query.

        Calls ``bamboo_last_evidence`` with ``mode='evidence'`` to retrieve
        the compact evidence dict from the server-side store — the same data
        that was sent to the LLM: job counts, site breakdown, error tallies,
        sample job records, and the PanDA ID list.  No HTTP request is made.

        Use ``/json`` for the verbatim BigPanDA API response instead.
        """
        if not self._mcp_ready:
            self._write_system("Not connected yet.")
            return
        try:
            res = await asyncio.wait_for(
                self._to_thread(self.mcp.call_tool, "bamboo_last_evidence", {"mode": "evidence"}),
                timeout=30,
            )
            text = _extract_text(res)
            if not text:
                self._write_system("No evidence stored yet — ask about a task or job first.")
                return
            parsed = json.loads(text)
            if "error" in parsed:
                self._write_system(parsed["error"])
                return
            evidence = parsed.get("evidence", parsed)
            tool = parsed.get("tool", "")
            title = f"{_now()}  evidence — {tool}" if tool else f"{_now()}  evidence"
            self._write_panel(
                Markdown(f"```json\n{_pretty(evidence)}\n```"),
                title=title,
                border_style="cyan",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_error(f"Could not fetch evidence: {exc}")

    def _chart_width(self) -> int:
        """Return a usable character width for ASCII chart content.

        Reads the transcript widget width and subtracts space for the
        Panel border (2), padding (2), and the y-axis label area (8),
        leaving the remainder for the bar/column content.  Falls back
        to 80 when the transcript size is not yet available.

        Returns:
            Integer character width for the chart bar area, at least 20.
        """
        transcript = self.transcript
        try:
            w = transcript.size.width if transcript is not None else self.size.width
        except Exception:  # pylint: disable=broad-exception-caught
            try:
                w = self.size.width
            except Exception:  # pylint: disable=broad-exception-caught
                w = 80
        # 2 panel border + 2 panel padding + 8 y-axis labels + 2 safety margin
        return max(20, w - 14)

    async def _try_auto_chart(self) -> None:
        """Silently append an ASCII pilot chart after a Harvester answer.

        Called automatically at the end of :meth:`_handle_question`.  Fetches
        the latest evidence from the server store and renders a status bar
        chart only when both conditions are met:

        1. The last tool that ran was ``panda_harvester_workers``.
        2. ``nworkers_by_status`` contains more than one entry (a single-status
           result — e.g. a site-filtered snapshot that only saw ``running``
           workers — is not worth charting).

        All failures are swallowed silently so a chart rendering problem never
        disrupts the main answer display.
        """
        if not self._mcp_ready:
            return
        try:
            from askpanda_atlas.chart_utils import render_chart  # deferred — optional dep
        except ImportError:
            return
        try:
            res = await asyncio.wait_for(
                self._to_thread(
                    self.mcp.call_tool,
                    "bamboo_last_evidence",
                    {"mode": "evidence"},
                ),
                timeout=15,
            )
            text = _extract_text(res)
            if not text:
                return
            parsed = json.loads(text)
            # Structure: {"tool": "...", "mode": "...", "evidence": {"evidence": {...}}}
            tool = parsed.get("tool", "")
            if tool != "panda_harvester_workers":
                return
            snapshot_ev = (parsed.get("evidence") or {}).get("evidence") or {}
            by_status: dict = snapshot_ev.get("nworkers_by_status") or {}
            if len(by_status) <= 1:
                return

            # --- Status-bar chart (from snapshot evidence) ---
            chart_str = render_chart(snapshot_ev, width=self._chart_width(), mode="auto")
            self._write_panel(
                Text(chart_str),
                title=f"{_now()}  pilot chart",
                border_style="green",
            )

            # --- Timeseries chart (second MCP call to OpenSearch tool) ---
            # Extract status and time window from the last user question.
            last_question = (
                next(
                    (m["content"] for m in reversed(self._history)
                     if m.get("role") == "user"),
                    "",
                )
            )
            ts_status = _extract_status_from_question(last_question)
            window = _extract_window_from_question(last_question)
            ts_args: dict[str, str] = {"question": last_question, "status": ts_status}
            site = snapshot_ev.get("site_filter")
            if site:
                ts_args["site"] = site
            if window:
                ts_args["from_dt"], ts_args["to_dt"] = window
            else:
                # Fall back to the window the snapshot tool actually used.
                if snapshot_ev.get("from_dt"):
                    ts_args["from_dt"] = snapshot_ev["from_dt"]
                if snapshot_ev.get("to_dt"):
                    ts_args["to_dt"] = snapshot_ev["to_dt"]

            ts_res = await asyncio.wait_for(
                self._to_thread(
                    self.mcp.call_tool,
                    "atlas.harvester_timeseries",
                    ts_args,
                ),
                timeout=20,
            )
            ts_text = _extract_text(ts_res)
            if not ts_text:
                return
            ts_parsed = json.loads(ts_text)
            ts_ev = (ts_parsed.get("evidence") or ts_parsed)
            if ts_ev.get("error") or len(ts_ev.get("buckets") or []) < 2:
                return
            ts_chart = render_chart(ts_ev, width=self._chart_width(), mode="timeseries")
            self._write_panel(
                Text(ts_chart),
                title=f"{_now()}  pilot timeseries ({ts_status})",
                border_style="green",
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Never let a chart failure surface to the user

    async def _handle_chart_command(self) -> None:
        """Display an ASCII bar chart of Harvester pilot counts by status.

        Retrieves the most recent ``panda_harvester_workers`` evidence from
        the server-side store via ``bamboo_last_evidence(mode='evidence')``
        and renders it using :func:`askpanda_atlas.chart_utils.render_chart`.

        The chart is only meaningful after a pilot/Harvester question has been
        asked.  If the last tool was not ``panda_harvester_workers`` (e.g. the
        last query was a task or job lookup) a clear message is shown instead.

        The chart is rendered as pre-formatted ``Text`` (not Markdown) to
        prevent Rich from reflowing the bar characters.
        """
        if not self._mcp_ready:
            self._write_system("Not connected yet.")
            return
        try:
            from askpanda_atlas.chart_utils import render_chart  # deferred — optional dep
        except ImportError:
            self._write_system(
                "chart_utils not available — is askpanda_atlas installed?"
            )
            return
        try:
            res = await asyncio.wait_for(
                self._to_thread(
                    self.mcp.call_tool,
                    "bamboo_last_evidence",
                    {"mode": "evidence"},
                ),
                timeout=30,
            )
            text = _extract_text(res)
            if not text:
                self._write_system(
                    "No evidence stored yet — ask a pilot/Harvester question first."
                )
                return
            parsed = json.loads(text)
            if "error" in parsed:
                self._write_system(parsed["error"])
                return

            # bamboo_last_evidence wraps its payload as {"evidence": {"evidence": {...}, "tool": "..."}}.
            # Two unwrap steps are needed: first to get {"evidence": {...}, "tool": "..."}, then
            # Structure from bamboo_last_evidence: {"tool": "...", "mode": "...", "evidence": {"evidence": {...}}}
            # tool is at the top level; the actual evidence dict is two levels down.
            tool = parsed.get("tool", "")
            evidence = (parsed.get("evidence") or {}).get("evidence") or {}

            # Only render a chart when the last tool was the harvester tool.
            if tool and tool != "panda_harvester_workers":
                self._write_system(
                    f"Last query used '{tool}', not 'panda_harvester_workers'. "
                    "Ask a pilot/Harvester question first, then type /chart."
                )
                return

            if not evidence.get("nworkers_by_status") and not evidence.get("pivot"):
                self._write_system(
                    "No pilot counts in evidence — "
                    "the Harvester API may have returned no data for that window."
                )
                return

            chart_str = render_chart(evidence, width=self._chart_width(), mode="auto")
            self._write_panel(
                Text(chart_str),
                title=f"{_now()}  pilot chart",
                border_style="green",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._write_error(f"Could not render chart: {exc}")

    def _write_raw_json(self, res: Any) -> None:
        """Write the full raw BigPanDA JSON for the last tool result.

        Triggered by the ``/json`` command. Extracts the BigPanDA payload
        from the MCP result envelope, handling both dict and SDK object forms.

        Args:
            res (Any): Raw MCP tool result to display.
        """
        payload = self._extract_bigpanda_payload(res)
        if payload is None:
            self._write_system("Could not extract BigPanDA payload from last result.")
            return
        self._write_panel(
            Markdown(f"```json\n{_pretty(payload)}\n```"),
            title=f"{_now()}  raw JSON (BigPanDA)",
            border_style="cyan",
        )

    @staticmethod
    def _extract_bigpanda_payload(res: Any) -> Any:
        """Extract the raw BigPanDA payload from an MCP tool result.

        The MCP SDK wraps results in a ``CallToolResult`` object with a
        ``content`` list of ``TextContent`` items. The text itself is the
        LLM answer, which embeds structured data in a JSON prefix when the
        tool returns evidence. This method peels back all those layers.

        Args:
            res: Raw value returned by ``MCPClientSync.call_tool``.

        Returns:
            The BigPanDA payload dict, or ``None`` if it cannot be found.
        """
        # 1. Unwrap MCP SDK CallToolResult object → get text string.
        text: str | None = None
        content_attr = getattr(res, "content", None)
        if content_attr is not None:
            # SDK object: res.content is a list of TextContent items.
            items = content_attr if isinstance(content_attr, list) else [content_attr]
            for item in items:
                t = getattr(item, "text", None) or (
                    item.get("text") if isinstance(item, dict) else None
                )
                if isinstance(t, str) and t.strip():
                    text = t
                    break
        elif isinstance(res, list) and res:
            first = res[0]
            text = (
                first.get("text") if isinstance(first, dict)
                else getattr(first, "text", None)
            )
        elif isinstance(res, dict):
            text = res.get("text")
        elif isinstance(res, str):
            text = res

        if not isinstance(text, str):
            return None

        # 2. Try to parse the text as JSON — panda_task_status returns a
        # dict {"evidence": {"payload": {...}, ...}, "text": "..."} serialised
        # as JSON inside the TextContent.text field.
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                evidence = parsed.get("evidence")
                if isinstance(evidence, dict):
                    payload = evidence.get("payload")
                    if isinstance(payload, dict):
                        return payload
                    # No payload key — return the evidence dict itself.
                    return evidence
                return parsed
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # 3. Text is plain (LLM answer) — payload not embedded.
        return None


def main() -> None:
    """Parse CLI arguments and run the Textual chat app."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "http"], required=True)
    ap.add_argument("--http-url", default=os.getenv("MCP_URL", "http://localhost:8000/mcp"))
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    ap.add_argument("--terminate-on-close", action="store_true", default=True)

    inline_group = ap.add_mutually_exclusive_group()
    inline_group.add_argument("--inline", action="store_true", help="Run inline (default).")
    inline_group.add_argument("--no-inline", action="store_true", help="Run in alternate screen.")

    args = ap.parse_args()

    if args.transport == "http":
        cfg = MCPServerConfig(transport="http", http_url=args.http_url, terminate_on_close=args.terminate_on_close)
    else:
        _trace_file = os.path.join(tempfile.gettempdir(), f"bamboo_trace_{os.getpid()}.jsonl")
        _cric_table_file = os.path.join(tempfile.gettempdir(), f"bamboo_cric_table_{os.getpid()}.txt")
        _stdio_env = os.environ.copy()
        _stdio_env["BAMBOO_TRACE"] = "1"
        _stdio_env["BAMBOO_TRACE_FILE"] = _trace_file
        _stdio_env["BAMBOO_CRIC_TABLE_FILE"] = _cric_table_file
        _stdio_env["BAMBOO_QUIET"] = "1"  # redirect server stderr → /dev/null
        cfg = MCPServerConfig(
            transport="stdio",
            stdio_command=sys.executable,
            stdio_args=["-m", "bamboo.server"],
            stdio_env=_stdio_env,
        )

    app = BambooTui(cfg=cfg, plugin_id=args.plugin)

    # Textual's alternate-screen mode (--no-inline) uses absolute cursor
    # positioning without erasing lines before writing.  On SSH pseudo-TTYs
    # (e.g. lxplus) the terminal does not clear the alternate screen buffer on
    # entry, so every repaint overlays on previous content producing ghost
    # frames.  This is a fundamental limitation of Textual's non-inline
    # renderer on SSH — not fixable in application code without patching
    # Textual itself.
    #
    # When running over SSH we therefore force inline mode, which uses relative
    # cursor movement (delta updates only) and renders correctly on all
    # terminals.  Set BAMBOO_FORCE_NO_INLINE=1 to override this if your
    # terminal is known to handle alternate screen correctly over SSH.
    _ssh = bool(
        os.environ.get("SSH_CLIENT")
        or os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
    )
    _force_no_inline = os.environ.get("BAMBOO_FORCE_NO_INLINE", "").strip().lower() in (
        "1", "true", "yes",
    )
    if _ssh and args.no_inline and not _force_no_inline:
        import sys as _sys
        print(
            "[bamboo] SSH session detected — using inline mode "
            "(set BAMBOO_FORCE_NO_INLINE=1 to use alternate screen).",
            file=_sys.stderr,
        )
        inline = True
    else:
        inline = True if args.inline or not args.no_inline else False
    app.run(inline=inline, inline_no_clear=True, mouse=False)


if __name__ == "__main__":
    main()
