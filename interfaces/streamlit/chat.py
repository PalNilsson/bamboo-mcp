"""AskPanDA Streamlit Chat UI.

Connects to an AskPanDA MCP server via:
  - Streamable HTTP transport (production): connects to ``http://host:port/mcp``
  - STDIO transport (development): spawns ``python -m bamboo.server``

Run:
  streamlit run interfaces/streamlit/chat.py

Key design decisions
--------------------
- ``bamboo_answer`` is always the answer tool — ``_guess_auto_tool`` is gone.
- Fast-path routing defaults to ON (matches server default).
- Bearer token and URL are user-visible sidebar controls. Plugin is fixed
  to ``ASKPANDA_PLUGIN`` env var (default ``atlas``) — not user-selectable.
- After each response, expanders show Tracing, Costs, Evidence (inspect),
  and Raw JSON — equivalent to TUI /tracing, /costs, /inspect, /json.
- LLM info and experiment display name are fetched on connect via
  ``bamboo_health`` and ``<plugin>.ui_manifest`` respectively.
- Tracing works in stdio mode (server writes to a temp file we read back).
  In HTTP mode, span data is not available client-side; the expanders
  explain this and show what information is available.
"""
# pylint: disable=no-member  # streamlit uses dynamic attributes
from __future__ import annotations

import json
import hmac
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Sequence  # noqa: F401  (kept for type annotations in helpers)
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — makes ``interfaces`` importable when Streamlit runs this
# script directly (i.e. without ``pip install -e .`` at the repo root).
# Inserts the repo root (two levels up from this file) onto sys.path if it
# is not already present.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig  # noqa: E402
from interfaces.shared.deeplink import question_from_params  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY_TURNS = 10
try:
    _MAX_HISTORY_TURNS: int = int(os.getenv("BAMBOO_HISTORY_TURNS", str(_DEFAULT_HISTORY_TURNS)))
except ValueError:
    _MAX_HISTORY_TURNS = _DEFAULT_HISTORY_TURNS

_ANSWER_TOOL = "bamboo_answer"
_DEFAULT_PLUGIN = os.getenv("ASKPANDA_PLUGIN", "atlas")

#: Diagram rendering mode.  ``single-iframe`` (default) = text + diagrams
#: in one ``st.components.v1.html`` iframe with CSS-controlled portrait/landscape layout.
#: ``classic`` = one ``st.components.v1.html`` iframe per diagram, diagrams only (no inline text).
_DIAGRAM_MODE: str = os.getenv("BAMBOO_DIAGRAM_MODE", "single-iframe").lower()

# ---------------------------------------------------------------------------
# Mermaid diagram support
# ---------------------------------------------------------------------------

#: Set to False at startup if streamlit-mermaid cannot be imported.
#: When False we fall back to st.components.v1.html with the Mermaid CDN.
_MERMAID_AVAILABLE: bool = True
try:
    import streamlit_mermaid as stmermaid  # type: ignore[import]  # noqa: F401
except ImportError:
    _MERMAID_AVAILABLE = False

# ---------------------------------------------------------------------------
# Superuser mode
# ---------------------------------------------------------------------------

#: Plain-text password from env var.  When absent, superuser mode is hidden.
_SUPERUSER_PASSWORD: str = os.getenv("BAMBOO_SUPERUSER_PASSWORD", "")

#: Tool names that are only shown/accessible in superuser mode.
#: Imported from the shared guard module; extended by BAMBOO_SUPERUSER_TOOLS.
from interfaces.shared.superuser_guard import (  # noqa: E402
    SUPERUSER_TOOL_NAMES as _SUPERUSER_TOOL_NAMES,
    is_superuser_question as _is_superuser_question,
)

# Prices in USD per 1 million tokens: (input_rate, output_rate).
# Verify against current provider docs; unknown models fall back to _DEFAULT_COST.
_MODEL_COST_PER_MTOK: dict[str, tuple[float, float]] = {
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
_DEFAULT_COST: tuple[float, float] = (1.00, 3.00)

# ---------------------------------------------------------------------------
# Plot support
# ---------------------------------------------------------------------------

#: Tools whose evidence is rich-nested rather than flat columns/rows.
#: These never trigger the plot expander regardless of their evidence shape.
#: Add entries here when new non-tabular tools are introduced.
_PLOT_UNSUPPORTED_TOOLS: frozenset[str] = frozenset({
    "panda_task_status",
    "panda_job_status",
    "panda_log_analysis",
    "atlas.pilot_source_analysis",
    "atlas.core_dump_analysis",
    "code_query",
    "panda_server_health",
    "bamboo_health",
})

#: Known column names mapped to display units for axis labels.
#: Applied regardless of which tool produced the evidence.
_COLUMN_UNIT_MAP: dict[str, str] = {
    "duration": "s",
    "total_queue_time": "s",
    "resource_waiting_queue_time": "s",
    "file_transfer_queue_time": "s",
    "total_io_read_time": "s",
    "execution_time_ms": "ms",
    "size": "bytes",
    "bandwidth": "bytes/s",
    "speed": "FLOP/s",
    "flops": "FLOP",
    "site_cpu_util": "fraction",
    "grid_cpu_util": "fraction",
    "site_storage_util": "fraction",
    "grid_storage_util": "fraction",
    "nworkers": "workers",
    "duration_s": "s",
    "wall_time_s": "s",
    "queue_time_s": "s",
    "io_time_s": "s",
}

# ---------------------------------------------------------------------------
# Superuser helpers
# ---------------------------------------------------------------------------


def _check_superuser_password(attempt: str) -> bool:
    """Compare a password attempt against the configured superuser password.

    Uses :func:`hmac.compare_digest` to resist timing attacks.  Returns
    ``False`` immediately when no superuser password is configured.

    Args:
        attempt: The password string entered by the user.

    Returns:
        ``True`` when the attempt matches and a password is configured.
    """
    if not _SUPERUSER_PASSWORD:
        return False
    return hmac.compare_digest(attempt.encode(), _SUPERUSER_PASSWORD.encode())


# ---------------------------------------------------------------------------
# Mermaid helpers
# ---------------------------------------------------------------------------


def _extract_mermaid_blocks(text: str) -> tuple[str, list[str]]:
    r"""Extract fenced Mermaid code blocks from an LLM response.

    Scans ``text`` for fenced blocks that begin with ``\`\`\`mermaid``.
    Each matching block is removed from the text and collected separately
    so the caller can render it with the Mermaid component.

    Args:
        text: Raw assistant response text, potentially containing one or
            more ``\`\`\`mermaid`` ... ``\`\`\``` blocks.

    Returns:
        Tuple of ``(clean_text, diagram_defs)`` where ``clean_text`` is
        the response with Mermaid blocks stripped and ``diagram_defs`` is
        a list of raw Mermaid definition strings (one per block found).
    """
    import re
    pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
    diagrams: list[str] = []

    def _capture(m: re.Match[str]) -> str:
        diagrams.append(m.group(1).strip())
        return ""

    clean = pattern.sub(_capture, text)
    # Tidy up any double-blank lines left by removal.
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, diagrams


def _wrap_mermaid_labels(diagram: str, max_chars: int = 20) -> str:
    r"""Wrap long node labels in a Mermaid diagram using native ``\n`` breaks.

    Mermaid flowchart supports ``\n`` inside double-quoted node labels as a
    native line break without requiring ``htmlLabels: true``.  ``<br/>`` only
    works with ``htmlLabels`` enabled; since that config may not take effect
    in the embedded component, ``\n`` is more reliable.

    Labels exceeding ``max_chars`` are split at word boundaries.  Labels that
    already contain ``\n`` or ``<br`` are left unchanged.
    ``stateDiagram-v2`` diagrams are returned unmodified.

    Args:
        diagram: Raw Mermaid diagram definition string.
        max_chars: Maximum characters per line inside a node label.

    Returns:
        Diagram string with long labels wrapped using ``\n``.
    """
    if "stateDiagram" in diagram:
        return diagram

    def _wrap_text(text: str) -> str:
        r"""Wrap text at word boundaries using Mermaid native \n breaks.

        Splits on spaces and underscores.  Single tokens longer than
        ``max_chars`` are hard-cut at the limit so they never overflow
        the node box.
        """
        if len(text) <= max_chars or "\n" in text or "<br" in text:
            return text
        # Tokenise on spaces and underscores; preserve underscores as breaks
        raw_words = re.split(r"([ _]+)", text)
        words = [w for w in raw_words if w.strip("_ ")]
        # Hard-cut any single token that exceeds max_chars
        split_words: list[str] = []
        for word in words:
            while len(word) > max_chars:
                split_words.append(word[:max_chars])
                word = word[max_chars:]
            if word:
                split_words.append(word)
        lines: list[str] = []
        current = ""
        for word in split_words:
            candidate = (current + " " + word).strip() if current else word
            if current and len(candidate) > max_chars:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    pattern = re.compile(
        r'(\b\w+\b)'
        r'(\[+|\{+)'
        r'("?)'
        r'([^"\]\}\n]+?)'
        r'("?)'
        r'(\]+|\}+)',
        re.MULTILINE,
    )

    def _replace(m: re.Match[str]) -> str:
        node_id, open_b, q_open, label, q_close, close_b = m.groups()
        wrapped = _wrap_text(label.strip())
        if wrapped == label.strip():
            return m.group(0)
        if "\n" in wrapped and not q_open:
            q_open = q_close = '"'
        return f'{node_id}{open_b}{q_open}{wrapped}{q_close}{close_b}'

    result_lines = []
    for line in diagram.splitlines():
        if line.lstrip().startswith("%%"):
            result_lines.append(line)
        else:
            result_lines.append(pattern.sub(_replace, line))
    return "\n".join(result_lines)


def _normalise_latex(text: str) -> str:
    r"""Convert common LaTeX delimiter styles to Streamlit-compatible KaTeX syntax.

    Streamlit's ``st.markdown()`` renders LaTeX via KaTeX using ``$...$``
    (inline) and ``$$...$$`` (display) delimiters.  LLMs frequently produce
    other standard delimiter styles which render as raw text without conversion.

    Conversions applied (in order):

    - ``\[ ... \]``  →  ``$$...$$``  (standard LaTeX display math)
    - ``\( ... \)``  →  ``$...$``   (standard LaTeX inline math)
    - ``[ ... ]`` where content looks like a LaTeX expression  →  ``$$...$$``
      (bare bracket style sometimes emitted by LLMs)

    The bare-bracket heuristic matches ``[`` followed by content that contains
    a backslash (indicating a LaTeX command), ending at the next ``]``.  Plain
    prose in square brackets is not affected.

    Args:
        text: Raw assistant response text.

    Returns:
        Text with LaTeX delimiters normalised for Streamlit KaTeX rendering.
    """
    # \[ ... \] → $$...$$  (display math, one or two backslashes before bracket)
    text = re.sub(r'\\{1,2}\[(.+?)\\{1,2}\]', r'$$\1$$', text, flags=re.DOTALL)

    # \( ... \) → $...$  (inline math)
    # Use word-boundary assertion to avoid matching \left( and \right).
    text = re.sub(r'\\{1,2}\((?!left\b)(.+?)\\{1,2}\)(?!right\b)',
                  r'$\1$', text, flags=re.DOTALL)

    # [ ... ] → $$...$$ only when content contains a LaTeX command (backslash)
    # and the bracket is not already inside a $...$ block.
    text = re.sub(r'(?<!\$)\[([^\[\]]*\\[^\[\]]*)\](?!\$)', r'$$\1$$', text)

    return text


# ---------------------------------------------------------------------------
# Shared Mermaid JS snippets
# ---------------------------------------------------------------------------

_MERMAID_INIT_JS: str = """
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart:   { useMaxWidth: true, htmlLabels: false,
                   nodeSpacing: 40, rankSpacing: 50 },
    stateDiagram:{ useMaxWidth: true, htmlLabels: false },
    themeVariables: {
      fontSize: '14px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    }
  });
"""

_MERMAID_CDN: str = (
    "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"
)


def _mermaid_height_estimate(defn: str) -> int:
    """Estimate a generous iframe height for a Mermaid diagram.

    ``st.iframe`` defaults to ``height="content"`` which measures iframe
    content synchronously at render time.  Since Mermaid renders asynchronously
    (async JS), the measurement fires before the SVG exists and the iframe
    collapses to zero height.  Passing an explicit integer pixel height avoids
    this; the estimate therefore errs significantly on the tall side —
    whitespace below the diagram is harmless, clipping is not.

    Mermaid flowcharts with subgraphs render considerably taller than a
    naive line-count suggests because each subgraph adds its own padding,
    border, and label row.  The multipliers below were calibrated against
    typical LLM-generated diagrams (5-10 nodes, 0-3 subgraphs).

    Args:
        defn: Raw Mermaid definition string.

    Returns:
        Pixel height to pass as ``st.iframe(html, height=N)``.
        Must be an explicit integer — never rely on the default
        ``height="content"`` for async-rendered content.
    """
    first = defn.strip().splitlines()[0].lower() if defn.strip() else ""

    if first.startswith("statediagram"):
        lc = sum(1 for ln in defn.splitlines() if ln.strip())
        return min(2000, max(500, lc * 60 + 200))

    if first.startswith("sequencediagram"):
        lc = sum(1 for ln in defn.splitlines() if ln.strip())
        return min(2000, max(500, lc * 40 + 150))

    # Flowchart / graph.
    # Count subgraph blocks — each adds substantial vertical overhead.
    # Use a generous per-line multiplier as the baseline so that deeply
    # nested or wide fan-out diagrams are never clipped.
    lines = defn.splitlines()
    lc = sum(1 for ln in lines if ln.strip())
    subgraph_count = sum(1 for ln in lines if ln.strip().lower().startswith("subgraph"))
    # Base: 55px per non-empty line; subgraph overhead: 200px each; floor: 700px
    estimated = lc * 55 + subgraph_count * 200 + 150
    return min(2000, max(700, estimated))


def _render_mermaid_classic(diagram_defs: list[str]) -> None:
    r"""Render Mermaid diagrams — one ``st.iframe`` per diagram.

    Uses ``mermaid.render()`` (Promise-based async API).  Because Mermaid
    renders asynchronously, ``st.iframe`` must be called with an explicit
    integer ``height`` — the default ``height="content"`` measures the
    iframe synchronously before the SVG exists and collapses to zero.
    Height is estimated by :func:`_mermaid_height_estimate`.
    On parse error the raw definition is shown in a plain ``<pre>`` block.

    Args:
        diagram_defs: Raw Mermaid definition strings (one per block).
    """
    if not diagram_defs:
        return

    for i, defn in enumerate(diagram_defs):
        if len(diagram_defs) > 1:
            st.caption(f"Diagram {i + 1}")

        # Quote unquoted edge labels that contain special characters
        # ( ) < > which Mermaid v11 may tokenise as separate nodes.
        # Only labels between arrow tokens and a pipe | are affected.
        import re as _re
        defn = _re.sub(
            r'(-->|--[^>]*>|===+>?)\s*([^"|\n{\[\]]+?)\s*(\|)',
            lambda m: (
                '{} "{}" {}'.format(
                    m.group(1), m.group(2).strip(), m.group(3))
                if any(c in m.group(2) for c in "()<>")
                else m.group(0)
            ),
            defn,
        )
        safe_defn = defn.replace("&", "&amp;")
        uid = f"mermaid-{i}"

        html = f"""<!DOCTYPE html>
<html>
<head>
<script src="{_MERMAID_CDN}"></script>
<style>
  body {{ margin: 0; padding: 0; background: transparent; }}
  #container {{ width: 50%; max-width: 600px; background: white; padding: 4px 8px;
                box-sizing: border-box; }}
  #container svg {{ width: 100% !important; max-width: 100% !important;
                    height: auto !important; }}
  #container svg text {{ font-size: 14px !important;
                         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }}
  #fallback {{ display:none; color:#c00; font-family:monospace;
               font-size:13px; padding:8px; }}
</style>
</head>
<body>
<div id="container"><div id="{uid}"></div></div>
<div id="fallback">⚠ Diagram syntax error — raw definition:
<pre>{safe_defn}</pre></div>
<script>
{_MERMAID_INIT_JS}
(async function() {{
  try {{
    const {{ svg }} = await mermaid.render("{uid}-svg", `{safe_defn}`);
    const el = document.getElementById("{uid}");
    el.innerHTML = svg;
    const s = el.querySelector("svg");
    if (s) {{
      s.removeAttribute("width");
      s.removeAttribute("height");
      s.style.width  = "100%";
      s.style.height = "auto";
      // Override Mermaid's internal SVG <style> by setting font directly on
      // every text/tspan element — external CSS cannot override SVG <style>.
      s.querySelectorAll("text, tspan").forEach(function(t) {{
        t.style.fontSize = "13px";
        t.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      }});
      // Trigger iframeResizer to re-measure after SVG is in the DOM.
      window.dispatchEvent(new Event('resize'));
    }}
  }} catch(e) {{
    document.getElementById("{uid}").style.display = "none";
    document.getElementById("fallback").style.display = "block";
  }}

}})();
</script>
</body>
</html>"""
        # Explicit pixel height required — st.iframe's default height="content"
        # measures content synchronously, before async Mermaid JS has rendered
        # the SVG, resulting in a zero-height iframe.
        st.iframe(html, height=_mermaid_height_estimate(defn))


def _render_mermaid_single_iframe(
    diagram_defs: list[str],
    markdown_text: str,
) -> None:
    r"""Render text and Mermaid diagrams together in a single ``st.iframe``.

    Experimental renderer enabled by ``BAMBOO_DIAGRAM_MODE=single-iframe``.
    Laying text and SVG out in the same document gives full CSS control:

    - **Portrait diagram** (naturalHeight > naturalWidth after render):
      floated right at 38% width so prose wraps alongside it.
    - **Landscape diagram**: full width, placed after the text.
    - Multiple diagrams: all appended below the text section.

    ``st.iframe`` must be called with an explicit integer ``height``; the
    default ``height="content"`` measures the iframe synchronously before
    async Mermaid JS has rendered the SVG, collapsing to zero.
    Height is estimated by :func:`_mermaid_height_estimate`.

    Falls back to :func:`st.markdown` when ``diagram_defs`` is empty.

    Args:
        diagram_defs: Raw Mermaid definition strings.
        markdown_text: The assistant reply with Mermaid fences already
            stripped (i.e. ``clean_answer`` from
            :func:`_extract_mermaid_blocks`).
    """
    if not diagram_defs:
        st.markdown(markdown_text)
        return

    # Escape the markdown for injection into a JS string literal.
    # We hand it to marked.js for rendering inside the iframe.
    md_escaped = (
        markdown_text
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    # Build one render call per diagram.
    diagram_js_array = "[" + ",".join(
        f"`{d.replace('&', '&amp;').replace('`', chr(96))}`"
        for d in diagram_defs
    ) + "]"

    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="{_MERMAID_CDN}"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 15px; line-height: 1.6; background: white; padding: 8px 12px; }}
  #prose {{ overflow: hidden; }}
  #prose h1,h2,h3 {{ margin: 0.6em 0 0.3em; font-weight: 600; }}
  #prose p  {{ margin: 0.4em 0; }}
  #prose ul,ol {{ margin: 0.4em 0 0.4em 1.4em; }}
  #prose code {{ background:#f3f3f3; border-radius:3px; padding:1px 4px;
                 font-size:0.88em; }}
  #prose pre  {{ background:#f3f3f3; border-radius:4px; padding:8px;
                 overflow-x:auto; }}
  .diagram-wrap {{ margin: 12px 0; }}
  .diagram-wrap.portrait {{ float: right; width: min(38%, 380px);
                            margin: 0 0 12px 16px; clear: right;
                            overflow-x: auto; }}
  .diagram-wrap.landscape {{ width: 50%; max-width: 600px;
                              clear: both; overflow-x: auto; }}
  .diagram-wrap svg {{ width: 100% !important; max-width: 100% !important;
                       height: auto !important; display: block; }}
  .diagram-wrap svg text {{ font-size: 14px !important;
                             font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }}
  .diagram-error {{ color:#c00; font-family:monospace; font-size:12px;
                    padding:6px; border:1px solid #fcc; border-radius:4px; }}
  #clearfix {{ clear: both; }}
</style>
</head>
<body>
<div id="prose"></div>
<div id="diagrams"></div>
<div id="clearfix"></div>
<script>
{_MERMAID_INIT_JS}

document.getElementById("prose").innerHTML = marked.parse(`{md_escaped}`);

const defs = {diagram_js_array};

(async function() {{
  const prose    = document.getElementById("prose");
  const diagrams = document.getElementById("diagrams");
  for (let i = 0; i < defs.length; i++) {{
    const wrap = document.createElement("div");
    wrap.className = "diagram-wrap";
    diagrams.appendChild(wrap);
    try {{
      const id = "mermaid-si-" + i;
      const {{ svg }} = await mermaid.render(id, defs[i]);
      wrap.innerHTML = svg;
      const s = wrap.querySelector("svg");
      if (s) {{
        s.removeAttribute("width");
        s.removeAttribute("height");
        // Override Mermaid internal SVG <style> — external CSS cannot win.
        s.querySelectorAll("text, tspan").forEach(function(t) {{
          t.style.fontSize = "13px";
          t.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        }});
        const vb = s.getAttribute("viewBox");
        let isPortrait = false;
        if (vb) {{
          const parts = vb.split(" ").map(Number);
          isPortrait = (parts[3] || 1) > (parts[2] || 1) * 1.3;
        }}
        if (isPortrait) {{
          // Prepend portrait diagram into prose before text so float:right
          // makes the text wrap alongside it rather than below.
          wrap.classList.add("portrait");
          prose.insertBefore(wrap, prose.firstChild);
        }} else {{
          wrap.classList.add("landscape");
          // Already in diagrams — stays below prose.
        }}
      }}
    }} catch(e) {{
      wrap.innerHTML =
        "<div class=\"diagram-error\">⚠ Diagram error: " +
        e.toString().replace(/</g,"&lt;") + "</div>";
      wrap.classList.add("landscape");
    }}
  }}
  // Trigger iframeResizer to re-measure after all SVGs are in the DOM.
  window.dispatchEvent(new Event('resize'));
}})();
</script>
</body>
</html>"""

    # Explicit pixel height required — st.iframe's default height="content"
    # measures content synchronously, before async Mermaid JS has rendered
    # the SVG, resulting in a zero-height iframe.
    _h = max((_mermaid_height_estimate(d) for d in diagram_defs), default=700)
    st.iframe(html, height=_h)


def _render_mermaid_blocks(diagram_defs: list[str]) -> None:
    """Dispatch to the active diagram renderer (classic or single-iframe).

    .. note::
        As of the current implementation :func:`_render_chat` always calls
        :func:`_render_mermaid_classic` directly (prose is already rendered
        by ``st.markdown``), so this dispatcher is no longer invoked from the
        main render path.  It is kept for potential future use.

    In ``classic`` mode (``BAMBOO_DIAGRAM_MODE=classic``) each diagram is
    rendered in its own ``st.iframe`` via :func:`_render_mermaid_classic`.

    In ``single-iframe`` mode (default, ``BAMBOO_DIAGRAM_MODE=single-iframe``)
    this function is a no-op — callers must invoke
    :func:`_render_mermaid_single_iframe` directly with both prose and diagrams.

    Args:
        diagram_defs: Raw Mermaid definition strings extracted by
            :func:`_extract_mermaid_blocks`.
    """
    if _DIAGRAM_MODE != "classic":
        return  # handled at call site in _render_chat
    _render_mermaid_classic(diagram_defs)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _extract_text(content_items: Any) -> str:
    """Extract human-readable text from MCP content items.

    Args:
        content_items: Tool response — list of MCPContent dicts, a single
            dict, a string, or any object with a ``content`` attribute.

    Returns:
        Concatenated text content, or empty string.
    """
    if content_items is None:
        return ""
    if hasattr(content_items, "content"):
        content_items = getattr(content_items, "content")
    if isinstance(content_items, str):
        return content_items
    if isinstance(content_items, dict):
        if content_items.get("type") == "text":
            return str(content_items.get("text", ""))
        return json.dumps(content_items, indent=2)
    if isinstance(content_items, list):
        parts: list[str] = []
        for item in content_items:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, indent=2))
            elif hasattr(item, "type") and getattr(item, "type") == "text" and hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p.strip())
    try:
        return json.dumps(content_items, indent=2)
    except Exception:  # pylint: disable=broad-exception-caught
        return str(content_items)


def _tool_names(tools_result: Any) -> list[str]:
    """Extract tool names from ``session.list_tools()`` results.

    Args:
        tools_result: Return value of MCP list_tools.

    Returns:
        Sorted list of tool name strings.
    """
    if hasattr(tools_result, "tools"):
        tools_result = getattr(tools_result, "tools")
    names: list[str] = []
    if tools_result is None:
        return names
    if isinstance(tools_result, list):
        for t in tools_result:
            if isinstance(t, dict) and "name" in t:
                names.append(str(t["name"]))
            elif hasattr(t, "name"):
                names.append(str(getattr(t, "name")))
    elif isinstance(tools_result, dict):
        inner = tools_result.get("tools")
        if isinstance(inner, list):
            return _tool_names(inner)
    return sorted(set(names))


def _cap_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim messages to at most ``_MAX_HISTORY_TURNS`` user+assistant pairs.

    Args:
        messages: Full message list.

    Returns:
        Trimmed list keeping the most recent turns.
    """
    max_msgs = _MAX_HISTORY_TURNS * 2
    return messages[-max_msgs:] if len(messages) > max_msgs else messages


def _estimate_cost(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate LLM cost from trace spans.

    Args:
        spans: Parsed trace span dicts from the trace file.

    Returns:
        Dict with ``calls``, ``total_input``, ``total_output``,
        ``total_tokens``, ``total_cost_usd``, and ``unknown_models``.
    """
    calls: list[dict[str, Any]] = []
    unknown_models: list[str] = []
    total_input = total_output = 0
    total_cost = 0.0

    for span in spans:
        if span.get("event") != "llm_call":
            continue
        model = str(span.get("model", "unknown"))
        provider = str(span.get("provider", ""))
        inp = int(span.get("input_tokens") or 0)
        out = int(span.get("output_tokens") or 0)
        duration_ms = float(span.get("duration_ms", 0.0))

        rate = _MODEL_COST_PER_MTOK.get(model) or _MODEL_COST_PER_MTOK.get(model.lower())
        if rate is None:
            unknown_models.append(model)
            rate = _DEFAULT_COST

        call_cost = (inp / 1_000_000) * rate[0] + (out / 1_000_000) * rate[1]
        calls.append({
            "provider": provider, "model": model,
            "input_tokens": inp, "output_tokens": out,
            "duration_ms": duration_ms, "cost_usd": call_cost,
            "rate_in": rate[0], "rate_out": rate[1],
        })
        total_input += inp
        total_output += out
        total_cost += call_cost

    return {
        "calls": calls,
        "total_input": total_input, "total_output": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": total_cost,
        "unknown_models": list(set(unknown_models)),
    }


def _read_spans(trace_file: str, from_pos: int) -> list[dict[str, Any]]:
    """Read bamboo trace spans written since ``from_pos``.

    Args:
        trace_file: Path to the NDJSON trace file.
        from_pos: Byte offset to start reading from.

    Returns:
        List of parsed span dicts.
    """
    spans: list[dict[str, Any]] = []
    try:
        with open(trace_file, "r", encoding="utf-8") as fh:
            fh.seek(from_pos)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("bamboo_trace"):
                        spans.append(obj)
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return spans


def _trace_file_size(trace_file: str) -> int:
    """Return current byte size of the trace file, or 0 if absent.

    Args:
        trace_file: Path to the trace file.

    Returns:
        File size in bytes.
    """
    try:
        return os.path.getsize(trace_file)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Cached MCP client (NO widget calls inside)
# ---------------------------------------------------------------------------

#: Every :class:`MCPClientSync` this module has created, so that they can be
#: closed rather than orphaned.
#:
#: ``@st.cache_resource`` is the only other reference to a live client, and
#: ``st.cache_resource.clear()`` drops it without closing anything.  Each
#: orphaned client keeps a running event-loop thread, an open transport and, on
#: stdio, a server subprocess; one such orphan held a wedged anyio cancel scope
#: and spun a full CPU core for 17 hours on aipanda033.
#:
#: Deliberately a registry rather than a second lookup through
#: ``_get_mcp_client``: retrieving the cached instance requires the same five
#: arguments it was created with, and the natural way to use Reconnect is to
#: edit the URL or token first — which would build a *new* client, close that,
#: and leave the live one orphaned anyway.
_LIVE_CLIENTS: list[MCPClientSync] = []

#: Guards :data:`_LIVE_CLIENTS`.  Streamlit runs one script thread per session,
#: and both the cache and this registry are process-wide.
_LIVE_CLIENTS_LOCK = threading.Lock()


def _register_client(client: MCPClientSync) -> None:
    """Record a client so it can be closed later.

    Args:
        client: A freshly constructed client.
    """
    with _LIVE_CLIENTS_LOCK:
        _LIVE_CLIENTS.append(client)


def _close_live_clients() -> None:
    """Close and forget every client this module has created.

    Best-effort and idempotent: the registry is drained first, so a repeat
    call closes nothing, and a client whose ``close()`` raises does not stop
    the others from being closed.  A failure here must not abort a reconnect —
    the user clicked Reconnect precisely because something is already wrong.
    """
    with _LIVE_CLIENTS_LOCK:
        clients = list(_LIVE_CLIENTS)
        _LIVE_CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Reconnect: failed to close an MCP client.", exc_info=True)


@st.cache_resource
def _get_mcp_client(
    transport: str,
    http_url: str,
    bearer_token: str,
    stdio_command: str,
    trace_file: str,
) -> MCPClientSync:
    """Create and cache an MCPClientSync.

    All parameters are plain scalars so Streamlit can hash them correctly.
    No widgets may be called inside a ``@st.cache_resource`` function.

    Args:
        transport: ``"http"`` or ``"stdio"``.
        http_url: MCP endpoint URL (HTTP transport).
        bearer_token: Bearer token for auth, or empty string.
        stdio_command: Python executable for stdio transport.
        trace_file: Trace file path injected into stdio server env.

    Returns:
        Connected MCPClientSync instance.
    """
    if transport == "http":
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None
        cfg = MCPServerConfig(transport="http", http_url=http_url, http_headers=headers)
    else:
        env = os.environ.copy()
        env["BAMBOO_TRACE"] = "1"
        env["BAMBOO_TRACE_FILE"] = trace_file
        env["BAMBOO_QUIET"] = "1"
        cfg = MCPServerConfig(
            transport="stdio",
            stdio_command=stdio_command,
            stdio_args=["-m", "bamboo.server"],
            stdio_env=env,
        )
    client = MCPClientSync(cfg)
    _register_client(client)
    return client


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """Initialise all required session state keys on first run."""
    _fp_env = os.getenv("BAMBOO_FAST_PATH", "1").strip().lower()
    _fast_path_default: bool = _fp_env not in ("0", "off", "false")
    defaults: dict[str, Any] = {
        "messages": [],
        "fast_path": _fast_path_default,
        "tool_names": [],
        "display_name": "",
        "llm_info": "",
        "server_ok": False,
        "last_spans": [],
        "last_evidence": None,
        "last_raw": None,
        "last_tool": None,
        "last_plugin_id": None,
        "superuser": False,
        "superuser_warning": None,
        "last_diagrams": [],
        "last_clean_answer": "",
        "trace_file": os.path.join(
            tempfile.gettempdir(), f"bamboo_streamlit_{os.getpid()}.jsonl"
        ),
        "pending_question": None,
        # Set once the URL query parameters have been turned into an opening
        # question, so a rerun does not ask it again.
        "deeplink_consumed": False,
        "promptlog_notices": [],
        "poll_promptlog": False,
        "retry_promptlog": False,  # one deferred retry if first poll misses doc_id
        "last_doc_id": None,   # (index, doc_id) of most recent indexed turn
        "last_rating": None,   # rating submitted for that turn
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _seed_from_query_params() -> None:
    """Turn URL query parameters into an opening question, once per session.

    Lets the PanDA monitor link straight to an answer:
    ``?job_id=7272161793`` opens the chat with "Analyze job 7272161793 and
    explain the failure" already asked.  The same link is the "continue in
    Bamboo" target from the REST answer panel, and the fallback when the REST
    API is down.

    Guarded twice against asking the same question again.  ``deeplink_consumed``
    stops a Streamlit rerun resubmitting it, and the parameters are cleared from
    the URL so a browser refresh does not either — each repeat would be a
    silent LLM call the user did not ask for.
    """
    if st.session_state.get("deeplink_consumed"):
        return
    st.session_state["deeplink_consumed"] = True

    try:
        params = dict(st.query_params)
    except Exception:  # pylint: disable=broad-exception-caught
        # Older Streamlit builds, or a context without a script run.
        return

    question = question_from_params(params)
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["pending_question"] = question
    st.session_state["last_rating"] = None

    try:
        st.query_params.clear()
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _connect(mcp: MCPClientSync, plugin_id: str) -> None:
    """Fetch tool list, LLM info and display name from the server.

    Args:
        mcp: Connected MCP client.
        plugin_id: Active plugin namespace (e.g. ``"atlas"``).
    """
    try:
        tools = _tool_names(mcp.list_tools())
        st.session_state["tool_names"] = tools
        st.session_state["server_ok"] = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.session_state["server_ok"] = False
        st.session_state["tool_names"] = []
        raise exc

    # LLM info via bamboo_health
    try:
        health_raw = mcp.call_tool("bamboo_health", {})
        health_text = _extract_text(health_raw)
        for line in health_text.splitlines():
            if "llm_info:" in line:
                st.session_state["llm_info"] = line.split(":", 1)[1].strip()
                break
    except Exception:  # pylint: disable=broad-exception-caught
        st.session_state["llm_info"] = ""

    # LLM probe — check auth/connectivity before the user types a question.
    st.session_state["llm_probe"] = {"status": "", "detail": ""}
    try:
        probe_raw = mcp.call_tool("bamboo_llm_probe", {})
        probe_text = _extract_text(probe_raw)
        probe = json.loads(probe_text)
        if isinstance(probe, dict):
            st.session_state["llm_probe"] = probe
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # Tool absent on older server — degrade gracefully.

    # Display name and banner from ui_manifest
    manifest_tool = f"{plugin_id}.ui_manifest"
    if manifest_tool in tools:
        try:
            raw = mcp.call_tool(manifest_tool, {})
            manifest = json.loads(_extract_text(raw) or "{}")
            if isinstance(manifest, dict):
                st.session_state["display_name"] = str(
                    manifest.get("display_name") or plugin_id.upper()
                )
        except Exception:  # pylint: disable=broad-exception-caught
            st.session_state["display_name"] = plugin_id.upper()
    else:
        st.session_state["display_name"] = plugin_id.upper()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _reconnect() -> None:
    """Drop the current MCP client and server metadata, then rerun.

    Closing comes first and is not optional.  ``st.cache_resource.clear()``
    only drops the cache's reference; whatever it was holding carries on
    running with nothing left to close it.
    """
    _close_live_clients()
    st.cache_resource.clear()
    for key in ("server_ok", "tool_names", "display_name", "llm_info",
                "last_spans", "last_evidence", "last_raw", "last_tool"):
        st.session_state.pop(key, None)
    st.rerun()


def _render_sidebar() -> tuple[str, str, str, str, str]:
    """Render sidebar controls and return connection parameters.

    ``plugin_id`` is not user-selectable — it is fixed to the
    ``ASKPANDA_PLUGIN`` environment variable (default ``"atlas"``).

    Returns:
        Tuple of ``(transport, http_url, bearer_token, plugin_id,
        stdio_command)`` where ``plugin_id`` equals ``_DEFAULT_PLUGIN``.
    """
    st.sidebar.title(st.session_state.get("display_name", "AskPanDA"))

    # --- Connection ---
    st.sidebar.header("Connection")
    transport = st.sidebar.selectbox(
        "Transport", ["http", "stdio"], index=0,
        help="HTTP: connect to a running server. stdio: spawn a local server.",
    )

    http_url = st.sidebar.text_input(
        "Server URL",
        value=os.getenv("MCP_URL", "http://localhost:8000/mcp"),
        disabled=(transport != "http"),
        help="MCP endpoint URL, e.g. http://hostname:8000/mcp",
    )

    bearer_token = st.sidebar.text_input(
        "Bearer token (optional)",
        value=os.getenv("MCP_BEARER_TOKEN", ""),
        type="password",
        disabled=(transport != "http"),
        help="Leave empty if the server has no auth configured.",
    )

    # Plugin is fixed to the ASKPANDA_PLUGIN env var (default "atlas").
    # Not exposed as a UI control — switch experiments by restarting with a
    # different env var rather than hot-switching in the sidebar.
    plugin_id = _DEFAULT_PLUGIN

    stdio_command = sys.executable  # not exposed to users; used internally

    # --- Server status ---
    st.sidebar.header("Status")
    if st.session_state.get("server_ok"):
        st.sidebar.success("Connected")
        llm_info = st.session_state.get("llm_info", "")
        if llm_info:
            st.sidebar.caption(f"🤖 {llm_info}")
        n_tools = len(st.session_state.get("tool_names", []))
        st.sidebar.caption(f"{n_tools} tools registered")
        # Surface LLM probe warnings so auth problems are visible before
        # the user submits a question.
        _probe = st.session_state.get("llm_probe", {})
        _probe_status = _probe.get("status", "") if isinstance(_probe, dict) else ""
        if _probe_status and _probe_status not in ("ok", "not_configured", ""):
            _probe_detail = _probe.get("detail", "") if isinstance(_probe, dict) else ""
            if _probe_status == "auth_error":
                st.sidebar.warning(
                    "⚠️ LLM authentication failed. "                    "Check your API key and restart the server.",
                    icon="🔑",
                )
            elif _probe_status == "config_error":
                st.sidebar.warning(
                    f"⚠️ LLM configuration error: {_probe_detail}",
                    icon="⚙️",
                )
            else:
                st.sidebar.warning(
                    f"⚠️ LLM probe: {_probe_status} — {_probe_detail}",
                    icon="⚠️",
                )
    else:
        st.sidebar.warning("Not connected")

    # --- Settings ---
    st.sidebar.header("Settings")
    st.sidebar.toggle(
        "Fast-path routing",
        key="fast_path",
        help=(
            "ON: deterministic routing for task/job/pilot questions (faster). "
            "OFF: all questions go through the LLM planner."
        ),
    )

    n_turns = len(st.session_state.get("messages", [])) // 2
    st.sidebar.caption(
        f"Context: {n_turns} / {_MAX_HISTORY_TURNS} turns in memory"
    )

    # --- Actions ---
    st.sidebar.header("Actions")
    if st.sidebar.button("🔄  Reconnect", use_container_width=True):
        _reconnect()

    if st.sidebar.button("🗑  Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_spans"] = []
        st.session_state["last_evidence"] = None
        st.session_state["last_raw"] = None
        st.session_state["last_tool"] = None
        st.session_state["last_diagrams"] = []
        st.session_state["last_doc_id"] = None
        st.session_state["last_rating"] = None
        st.rerun()

    with st.sidebar.expander("Tools registered on server"):
        tools = st.session_state.get("tool_names", [])
        if tools:
            st.write("\n".join(f"- `{t}`" for t in tools))
        else:
            st.caption("Not connected yet.")

    # --- Developer / superuser access ---
    if _SUPERUSER_PASSWORD:
        st.sidebar.header("Developer access")
        if st.session_state.get("superuser"):
            st.sidebar.success("🔓 Superuser mode active")
            if st.sidebar.button("🔒  Lock", use_container_width=True):
                st.session_state["superuser"] = False
                st.rerun()
        else:
            pw = st.sidebar.text_input(
                "Password",
                type="password",
                key="_su_pw_input",
                help="Enter the superuser password to unlock developer tools.",
            )
            if st.sidebar.button("Unlock", use_container_width=True):
                if _check_superuser_password(pw):
                    st.session_state["superuser"] = True
                    st.session_state["superuser_warning"] = None
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect password.")

    return transport, http_url, bearer_token, str(plugin_id), stdio_command


# ---------------------------------------------------------------------------
# Response detail expanders
# ---------------------------------------------------------------------------

def _render_tracing_expander(
    spans: list[dict[str, Any]],
    transport: str,
) -> None:
    """Render tracing span data in a Streamlit expander.

    Args:
        spans: Trace spans collected for the last request.
        transport: ``"http"`` or ``"stdio"``.
    """
    with st.expander("⏱  Tracing", expanded=False):
        if not spans:
            if transport == "http":
                st.caption(
                    "Trace spans are not available for HTTP transport — the server "
                    "writes them to its own trace file. To inspect them, run on the server:\n\n"
                    "```bash\ntail -f $BAMBOO_TRACE_FILE | grep bamboo_trace | jq .\n```"
                )
            else:
                st.caption("No spans collected for this request.")
            return

        rows: list[dict[str, Any]] = []
        total_ms = 0.0
        for span in spans:
            event = str(span.get("event", ""))
            tool = str(span.get("tool", ""))
            duration_ms = float(span.get("duration_ms", 0.0))
            if event == "tool_call":
                total_ms = duration_ms
            # Build a short detail string
            detail_parts: list[str] = []
            if event == "llm_call":
                provider = span.get("provider", "")
                model = span.get("model", "")
                inp = span.get("input_tokens")
                out = span.get("output_tokens")
                detail_parts.append(f"{provider}/{model}")
                if inp is not None and out is not None:
                    detail_parts.append(f"tokens={inp}→{out}")
            elif event == "guard":
                allowed = span.get("allowed")
                reason = span.get("reason", "")
                detail_parts.append(f"allowed={allowed} reason={reason}")
            elif event == "retrieval":
                backend = span.get("backend", "")
                hits = span.get("hits")
                if backend:
                    detail_parts.append(f"backend={backend}")
                if hits is not None:
                    detail_parts.append(f"hits={hits}")
            elif event == "route":
                route = span.get("route", "")
                if route:
                    detail_parts.append(f"route={route}")
            rows.append({
                "event": event,
                "tool": tool,
                "ms": f"{duration_ms:.0f}",
                "detail": "  ".join(detail_parts),
            })

        rows.append({"event": "total", "tool": "", "ms": f"{total_ms:.0f}", "detail": "wall time"})
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_costs_expander(spans: list[dict[str, Any]]) -> None:
    """Render estimated LLM cost in a Streamlit expander.

    Args:
        spans: Trace spans collected for the last request.
    """
    with st.expander("💰  Estimated cost", expanded=False):
        if not spans:
            st.caption("No trace data — cost estimation requires tracing.")
            return

        est = _estimate_cost(spans)
        calls = est["calls"]
        if not calls:
            st.caption("No LLM calls found in spans (tracing may be disabled on the server).")
            return

        st.dataframe(
            [
                {
                    "model": c["model"],
                    "input tok": c["input_tokens"],
                    "output tok": c["output_tokens"],
                    "duration ms": f"{c['duration_ms']:.0f}",
                    "cost USD": f"${c['cost_usd']:.6f}",
                }
                for c in calls
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"**Total:** {est['total_input']:,} in + {est['total_output']:,} out = "
            f"{est['total_tokens']:,} tokens  |  **${est['total_cost_usd']:.6f}**"
        )
        if est["unknown_models"]:
            st.warning(
                f"Unknown model(s): {', '.join(est['unknown_models'])}. "
                f"Rates defaulted to ${_DEFAULT_COST[0]:.2f}/${_DEFAULT_COST[1]:.2f} per Mtok."
            )


def _render_evidence_expander(evidence: Any) -> None:
    """Render the compact evidence dict in a Streamlit expander.

    Args:
        evidence: Evidence dict from ``bamboo_last_evidence`` with ``mode='evidence'``.
    """
    with st.expander("🔬  Evidence (inspect)", expanded=False):
        if evidence is None:
            st.caption("No evidence stored — ask about a specific task or job first.")
            return
        st.json(evidence)


def _render_raw_expander(raw: Any) -> None:
    """Render the raw BigPanDA API response in a Streamlit expander.

    Args:
        raw: Raw payload from ``bamboo_last_evidence`` with ``mode='raw'``.
    """
    with st.expander("📄  Raw JSON", expanded=False):
        if raw is None:
            st.caption("No raw payload stored — ask about a specific task or job first.")
            return
        st.json(raw)


def _fetch_evidence(mcp: MCPClientSync) -> tuple[Any, Any, str | None]:
    """Fetch compact evidence and raw payload from the server.

    Calls ``bamboo_last_evidence`` twice — once for the compact evidence dict
    and once for the verbatim BigPanDA API response.

    Args:
        mcp: Connected MCP client.

    Returns:
        Tuple of ``(evidence_dict_or_None, raw_payload_or_None, tool_name_or_None)``.
    """
    evidence = None
    raw = None
    tool_name: str | None = None
    try:
        ev_raw = mcp.call_tool("bamboo_last_evidence", {"mode": "evidence"})
        ev_text = _extract_text(ev_raw)
        if ev_text:
            parsed = json.loads(ev_text)
            tool_name = parsed.get("tool") or None
            # bamboo_last_evidence wraps: {"tool":..., "evidence": {"evidence": {...}}}
            # Two unwraps are needed to reach the actual columns/rows dict.
            inner = parsed.get("evidence", parsed)
            inner = inner.get("evidence", inner) if isinstance(inner, dict) else inner
            if inner and not (isinstance(inner, dict) and inner.get("error")):
                evidence = inner
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    try:
        raw_result = mcp.call_tool("bamboo_last_evidence", {"mode": "raw"})
        raw_text = _extract_text(raw_result)
        if raw_text:
            parsed_raw = json.loads(raw_text)
            inner_raw = parsed_raw.get("evidence", parsed_raw)
            if inner_raw and not (isinstance(inner_raw, dict) and inner_raw.get("error")):
                raw = inner_raw
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return evidence, raw, tool_name


def _column_label(col: str) -> str:
    """Convert a snake_case column name to a readable axis label with units.

    Strips common SQL aggregate prefixes (``avg_``, ``sum_``, etc.) before
    looking up the bare name in :data:`_COLUMN_UNIT_MAP`.

    Args:
        col: Raw column name from a query result (e.g. ``"avg_duration_s"``).

    Returns:
        Human-readable label string, e.g. ``"Duration (s)"`` or ``"Site"``.
    """
    bare = col
    for prefix in ("avg_", "sum_", "min_", "max_", "count_", "total_"):
        if col.startswith(prefix):
            bare = col[len(prefix):]
            break
    readable = bare.replace("_", " ").title()
    unit = _COLUMN_UNIT_MAP.get(bare)
    return f"{readable} ({unit})" if unit else readable


def _detect_plot(
    columns: list[str],
    rows: list[dict[str, Any]],
) -> tuple[str, str, str | None, str | None] | None:
    """Detect what kind of Plotly chart can be drawn from tabular evidence.

    Inspects column names and row values to classify each column as text or
    numeric, then selects a chart type from recognised patterns.  Returns
    ``None`` when no pattern matches.

    Columns whose names end in ``_id`` or equal ``JOB_ID`` / ``_ID`` are
    excluded from classification — they are identifiers, not plottable axes.

    Recognised patterns (checked in priority order):

    1. **Scatter with colour**: one text column + exactly two numeric columns
       → scatter plot with the text column as colour grouping.
    2. **Bar**: one text column + exactly one numeric column → horizontal bar.
    3. **Histogram**: one numeric column with ≥ 4 rows → distribution histogram.
    4. **Scatter**: exactly two numeric columns → scatter plot.

    Args:
        columns: Ordered list of column names from the query result.
        rows: List of row dicts from the query result.

    Returns:
        4-tuple ``(chart_type, x_col, y_col, color_col)`` or ``None``.
    """
    if not columns or not rows:
        return None

    def _is_id_col(col: str) -> bool:
        """Return True when a column looks like an identifier, not a value.

        Args:
            col: Column name to test.

        Returns:
            True if the column should be excluded from chart axis selection.
        """
        lower = col.lower()
        return lower.endswith("_id") or lower in ("job_id", "_id", "id")

    text_cols: list[str] = []
    numeric_cols: list[str] = []
    for col in columns:
        if _is_id_col(col):
            continue
        for row in rows:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                numeric_cols.append(col)
            else:
                text_cols.append(col)
            break

    # Pattern 1: one text + two or more numerics → scatter coloured by the text
    # column, using the first two numeric columns as axes.  Checked before bar
    # so that multi-numeric queries (e.g. avg queue time AND avg execution time
    # per site) get a scatter rather than a bar showing only the first numeric.
    if len(text_cols) == 1 and len(numeric_cols) >= 2:
        return ("scatter", numeric_cols[0], numeric_cols[1], text_cols[0])

    # Pattern 2: one text + one numeric → horizontal bar chart.
    if len(text_cols) == 1 and len(numeric_cols) == 1:
        return ("bar", text_cols[0], numeric_cols[0], None)

    # Pattern 3: one numeric, enough rows → histogram of distribution.
    if len(text_cols) == 0 and len(numeric_cols) == 1 and len(rows) >= 4:
        return ("histogram", numeric_cols[0], None, None)

    # Pattern 4: two numerics, no label → scatter.
    if len(text_cols) == 0 and len(numeric_cols) == 2:
        return ("scatter", numeric_cols[0], numeric_cols[1], None)

    return None


def _build_plot_figure(
    chart_type: str,
    x_col: str,
    y_col: str | None,
    color_col: str | None,
    rows: list[dict[str, Any]],
) -> Any:
    """Build and return a Plotly figure from tabular evidence rows.

    Args:
        chart_type: One of ``"bar"``, ``"histogram"``, or ``"scatter"``.
        x_col: Column for the x-axis (value axis for bar; single axis for
            histogram).
        y_col: Column for the y-axis.  ``None`` for histograms.
        color_col: Column for colour grouping.  ``None`` when not wanted.
        rows: List of row dicts from the query result.

    Returns:
        Plotly figure object ready to pass to ``st.plotly_chart``.
    """
    import plotly.express as px  # type: ignore[import]

    if chart_type == "bar":
        fig = px.bar(
            rows,
            x=y_col,
            y=x_col,
            orientation="h",
            labels={x_col: _column_label(x_col), y_col: _column_label(y_col)},
            title=_column_label(y_col),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    elif chart_type == "histogram":
        fig = px.histogram(
            rows,
            x=x_col,
            labels={x_col: _column_label(x_col)},
            title=f"Distribution of {_column_label(x_col)}",
            nbins=min(20, len(rows)),
        )
    else:  # scatter
        colour_labels = {color_col: _column_label(color_col)} if color_col else {}
        fig = px.scatter(
            rows,
            x=x_col,
            y=y_col,
            color=color_col,
            labels={
                x_col: _column_label(x_col),
                y_col: _column_label(y_col),
                **colour_labels,
            },
            title=f"{_column_label(y_col)} vs {_column_label(x_col)}",
        )
        fig.update_traces(marker={"size": 10, "opacity": 0.85, "line": {"width": 1, "color": "white"}})
        # Hide the colour legend when it has many entries — it crowds the plot
        # area without adding information when points overlap or are uniform.
        n_legend = len(set(row.get(color_col) for row in rows)) if color_col else 0
        fig.update_layout(showlegend=(n_legend <= 8))

    fig.update_layout(margin={"t": 40, "b": 20, "l": 20, "r": 20}, height=400)
    return fig


def _render_plot_expander(evidence: Any, tool_name: str | None) -> None:
    """Render an interactive Plotly chart from flat tabular evidence.

    Appears as a collapsible **\u2603\ufe0f Plot** expander below the assistant reply
    whenever the evidence has a plottable ``columns``/``rows`` shape.  No
    plugin name is hard-coded: the expander fires for any tool whose evidence
    is flat-tabular and not in :data:`_PLOT_UNSUPPORTED_TOOLS`.

    Silently renders nothing when plotly is not installed, the tool is in
    :data:`_PLOT_UNSUPPORTED_TOOLS`, evidence is absent or erroneous, or
    the columns do not match a recognised chart pattern.

    Args:
        evidence: Evidence dict with ``columns``, ``rows``, ``truncated``,
            and optionally ``sql``.
        tool_name: Name of the tool that produced the evidence, or ``None``.
    """
    if tool_name in _PLOT_UNSUPPORTED_TOOLS:
        return
    try:
        import plotly.express  # type: ignore[import]  # noqa: F401
    except ImportError:
        return

    if not isinstance(evidence, dict):
        return
    if evidence.get("error"):
        return

    columns: list[str] = evidence.get("columns") or []
    rows: list[dict[str, Any]] = evidence.get("rows") or []
    truncated: bool = bool(evidence.get("truncated"))
    sql: str = str(evidence.get("sql") or "")

    if not rows:
        return

    chart_spec = _detect_plot(columns, rows)
    if chart_spec is None:
        return

    chart_type, x_col, y_col, color_col = chart_spec

    try:
        fig = _build_plot_figure(chart_type, x_col, y_col, color_col, rows)
        st.plotly_chart(fig, use_container_width=True)
        if truncated:
            st.caption(
                "⚠️ Chart shows a partial result — the query was capped. "
                "Refine your question to see the full dataset."
            )
        if sql:
            st.caption(f"SQL: `{sql}`")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.warning(f"Plot could not be rendered: {exc}")


# ---------------------------------------------------------------------------
# Main chat panel
# ---------------------------------------------------------------------------

def _poll_promptlog_events(mcp: MCPClientSync) -> None:
    """Call ``bamboo_promptlog_status`` and push any events to the notice queue.

    Drains the server-side prompt-log ring buffer and appends each event to
    ``st.session_state["promptlog_notices"]`` so that
    :func:`_render_promptlog_notices` can display them on the next render cycle.

    Silently skipped when the tool is not registered on the server (prompt
    logging disabled or older server version).

    Args:
        mcp: Connected MCP client.
    """
    if "bamboo_promptlog_status" not in st.session_state.get("tool_names", []):
        return
    try:
        raw = mcp.call_tool("bamboo_promptlog_status", {})
        text = _extract_text(raw) or "{}"
        parsed = json.loads(text)
        notices: list = st.session_state.get("promptlog_notices", [])
        for event in parsed.get("events") or []:
            severity = str(event.get("severity", "info"))
            message = str(event.get("message", ""))
            if message:
                notices.append((severity, message))
                # Extract (index, doc_id) for the rating widget.
                _m = re.search(
                    r"index='([^']+)'.*?id='([^']+)'",
                    message,
                )
                if _m:
                    st.session_state["last_doc_id"] = (_m.group(1), _m.group(2))
        st.session_state["promptlog_notices"] = notices
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # Never let observability polling affect the main response path.


def _render_promptlog_notices() -> None:
    """Drain and display any prompt-log notifications accumulated in session state.

    Notifications are written by a background thread (the OpenSearch write
    worker) and stored in ``st.session_state["promptlog_notices"]`` as
    ``(severity, message)`` tuples.  Calling this function renders them and
    clears the queue so they are shown exactly once.

    Severity mapping:

    * ``"error"``   → :func:`st.error` with a red circle icon.
    * ``"warning"`` → :func:`st.warning` with a warning icon.
    * ``"info"``    → :func:`st.toast` with a green check icon (non-blocking).
    """
    notices: list = st.session_state.get("promptlog_notices", [])
    if not notices:
        return
    st.session_state["promptlog_notices"] = []
    for severity, message in notices:
        if severity == "error":
            st.error(message, icon="🔴")
        elif severity == "warning":
            st.warning(message, icon="⚠️")
        else:
            st.toast(message, icon="✅")


def _render_chat(mcp: MCPClientSync, transport: str) -> None:  # noqa: C901
    """Render the main chat panel.

    Handles the two-rerun pattern required by Streamlit:
    - Rerun 1: append user message, set ``pending_question``, rerun.
    - Rerun 2: generate assistant response, clear ``pending_question``, rerun.

    During rerun 2 the full chat history is rendered *before* the spinner so
    that ``st.spinner("Thinking…")`` appears at the bottom of the page, just
    above the input box, rather than at the top where it would be invisible in
    long conversations.

    Args:
        mcp: Connected MCP client.
        transport: ``"http"`` or ``"stdio"`` — affects tracing availability.
    """
    messages: list[dict[str, str]] = st.session_state["messages"]

    # Generate assistant response for a pending question
    if st.session_state.get("pending_question"):
        question: str = st.session_state["pending_question"]
        st.session_state["pending_question"] = None

        # Pre-dispatch superuser guard: block questions that would route to
        # superuser-only tools when the session is not authenticated.
        if (
            _SUPERUSER_PASSWORD
            and not st.session_state.get("superuser", False)
            and _is_superuser_question(question, st.session_state.get("tool_names", []))
        ):
            # Remove the user message we just appended — it should not sit in
            # history without a corresponding answer.
            if messages and messages[-1]["role"] == "user":
                st.session_state["messages"] = messages[:-1]
            # Store the warning in session state so it survives st.rerun().
            # A plain st.warning() here is wiped by the rerun before it renders.
            st.session_state["superuser_warning"] = (
                "🔒 This question requires **superuser mode**. "
                "Enter your password in the **Developer access** section of the sidebar."
            )
            st.rerun()
            return

        trace_file: str = st.session_state["trace_file"]
        pre_pos = _trace_file_size(trace_file) if transport == "stdio" else 0

        # Render chat history first so the spinner appears at the bottom of
        # the page (just above the input box) rather than at the top.
        for _msg in messages:
            with st.chat_message(_msg["role"]):
                _rendered = (
                    _normalise_latex(_msg["content"])
                    if _msg["role"] == "assistant"
                    else _msg["content"]
                )
                st.markdown(_rendered)

        # Run mcp.call_tool on a background thread so the main thread can
        # drive a phase + dot animation on the visible st.status label.
        # st.status label updates ARE safe from the main script thread —
        # only widget writes from *non*-main threads are blocked by Streamlit.
        _call_result: dict = {}

        def _do_call() -> None:
            """Execute the MCP tool call and store the result."""
            try:
                _call_result["result"] = mcp.call_tool(
                    _ANSWER_TOOL,
                    {
                        "question": question,
                        "messages": list(messages),
                        "bypass_fast_path": not st.session_state.get("fast_path", True),
                    },
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _call_result["error"] = exc

        _call_thread = threading.Thread(target=_do_call, daemon=True)

        # Phase labels and dot sequence cycled in the main thread while the
        # MCP call runs in the background.  Each phase shows for
        # _TICKS_PER_PHASE seconds; dots cycle every second within the phase.
        _phases = ["Routing question", "Retrieving evidence", "Synthesising answer"]
        _dots = [".", "..", "..."]
        _TICKS_PER_PHASE = 4

        with st.status(_phases[0] + _dots[0], expanded=False) as _status:
            _call_thread.start()
            _tick = 0
            while _call_thread.is_alive():
                _phase = _phases[(_tick // _TICKS_PER_PHASE) % len(_phases)]
                _dot = _dots[_tick % len(_dots)]
                _status.update(label=f"{_phase}{_dot}")
                time.sleep(1.0)
                _tick += 1
            _call_thread.join()

            if "error" in _call_result:
                answer = f"⚠️ Error: {_call_result['error']}"
                _status.update(label="Error", state="error", expanded=False)
            else:
                answer = _extract_text(_call_result.get("result")) or "*(No text output.)*"
                _status.update(label="Done", state="complete", expanded=False)

        # Extract any Mermaid diagram blocks from the answer before storing.
        # The clean text (without fenced blocks) is stored in message history
        # so follow-up context is not polluted with raw Mermaid syntax.
        clean_answer, diagram_defs = _extract_mermaid_blocks(answer)
        st.session_state["last_diagrams"] = diagram_defs
        st.session_state["last_clean_answer"] = clean_answer

        # Collect spans (stdio only)
        spans: list[dict[str, Any]] = []
        if transport == "stdio":
            spans = _read_spans(trace_file, pre_pos)
        st.session_state["last_spans"] = spans

        # Fetch evidence from server store
        evidence, raw, last_tool = _fetch_evidence(mcp)
        st.session_state["last_evidence"] = evidence
        st.session_state["last_raw"] = raw
        st.session_state["last_tool"] = last_tool

        # Signal that the next render cycle should poll for prompt-log events.
        # We cannot poll now because log_prompt() fires as a background task
        # inside call_llm and may not have completed before we get here.
        # The st.rerun() below triggers another render pass where the poll
        # will run, by which time the OpenSearch write should be finished.
        st.session_state["poll_promptlog"] = True

        # Append assistant reply (clean, no Mermaid blocks) and cap history.
        messages.append({"role": "assistant", "content": clean_answer})
        st.session_state["messages"] = _cap_messages(messages)
        st.rerun()

    # Render persisted superuser guard warning (set on the previous rerun).
    # Must be shown here — after st.rerun() any ephemeral st.warning() is lost.
    if st.session_state.get("superuser_warning"):
        st.warning(st.session_state["superuser_warning"], icon="🔒")
        st.session_state["superuser_warning"] = None

    _render_promptlog_notices()

    # Deferred prompt-log poll: log_prompt() fires as a background task inside
    # call_llm and may not finish before the response rerun.  The flag set
    # during the response pass signals that we should poll on this render cycle,
    # by which time the OpenSearch write should have completed.
    #
    # If the first poll returns before OpenSearch finishes writing (last_doc_id
    # still None), set retry_promptlog so we poll once more on the next cycle.
    # This fixes the missing rating widget on the first question after restart.
    if st.session_state.pop("poll_promptlog", False):
        _poll_promptlog_events(mcp)
        if st.session_state.get("last_doc_id") is None:
            # OpenSearch write not yet flushed — retry up to 3 times with
            # 0.5 s intervals before giving up and showing no rating widget.
            for _ in range(3):
                time.sleep(0.5)
                _poll_promptlog_events(mcp)
                if st.session_state.get("last_doc_id") is not None:
                    break

    # Legacy single-retry flag — kept for safety but no longer set.
    if st.session_state.pop("retry_promptlog", False):
        time.sleep(0.5)
        _poll_promptlog_events(mcp)

    # Render chat history.
    # Always render the last assistant message via st.markdown so text is
    # never suppressed.  Diagrams are rendered separately below the message
    # loop using the appropriate renderer.
    _messages = st.session_state["messages"]
    _diagrams = st.session_state.get("last_diagrams") or []
    for _mi, msg in enumerate(_messages):
        with st.chat_message(msg["role"]):
            rendered = (
                _normalise_latex(msg["content"])
                if msg["role"] == "assistant"
                else msg["content"]
            )
            st.markdown(rendered)

    # After the last assistant reply: render diagrams (if any), plot, expanders.
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
        # Mermaid diagrams — rendered separately from the prose which is
        # already shown via st.markdown above.  Always use the classic
        # renderer (one st.iframe per diagram) regardless of _DIAGRAM_MODE:
        # the single-iframe mode was designed to co-render prose+diagrams
        # together, but since prose is handled by st.markdown, passing ""
        # as markdown_text produces an oversized mostly-blank iframe.
        _render_mermaid_classic(_diagrams)

        _render_plot_expander(
            st.session_state["last_evidence"],
            st.session_state.get("last_tool"),
        )
        # Superuser tools expose full source evidence; hide raw/evidence
        # expanders from non-superuser sessions for those tools to avoid
        # leaking potentially large source blobs to casual users.
        last_tool = st.session_state.get("last_tool")
        is_superuser_tool = last_tool in _SUPERUSER_TOOL_NAMES
        is_superuser = st.session_state.get("superuser", False)
        show_evidence_detail = (not is_superuser_tool) or is_superuser

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _render_tracing_expander(st.session_state["last_spans"], transport)
        with col2:
            _render_costs_expander(st.session_state["last_spans"])
        with col3:
            if show_evidence_detail:
                _render_evidence_expander(st.session_state["last_evidence"])
        with col4:
            if show_evidence_detail:
                _render_raw_expander(st.session_state["last_raw"])

        # Rating widget — shown after every assistant response.
        _render_rating_widget(mcp)

        # Script download button(s) — shown when the last response
        # contains one or more fenced code blocks.
        _render_script_download()

    # Chat input — must be the last widget
    question = st.chat_input(st.session_state.get("display_name", "Ask PanDA"))
    if question:
        expanded_q, help_md = _expand_slash_command(question)
        if help_md is not None:
            # Display help inline without submitting to the MCP server.
            st.session_state["messages"].append({"role": "user", "content": question})
            st.session_state["messages"].append(
                {"role": "assistant", "content": help_md}
            )
            st.rerun()
        else:
            submitted = expanded_q if expanded_q is not None else question
            if isinstance(submitted, str) and submitted.startswith("__rate__"):
                try:
                    rating_val = int(submitted.split("__rate__")[1])
                    doc_ref = st.session_state.get("last_doc_id")
                    tool_names = st.session_state.get("tool_names", [])
                    if doc_ref and "bamboo_promptlog_rate" in tool_names:
                        idx_name, doc_id = doc_ref
                        raw = mcp.call_tool(
                            "bamboo_promptlog_rate",
                            {"index": idx_name, "doc_id": doc_id,
                             "rating": rating_val},
                        )
                        parsed_r = json.loads(_extract_text(raw) or "{}")
                        if not parsed_r.get("error"):
                            st.session_state["last_rating"] = rating_val
                    else:
                        st.session_state["messages"].append(
                            {"role": "assistant",
                             "content": "No response to rate yet."}
                        )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                st.rerun()
            else:
                st.session_state["messages"].append(
                    {"role": "user", "content": question}
                )
                st.session_state["pending_question"] = submitted
                st.session_state["last_rating"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# Slash-command expansion (Streamlit)
# ---------------------------------------------------------------------------

#: Displayed when the user types /help in the Streamlit chat input.
_STREAMLIT_HELP = """**Bamboo slash commands**

| Command | Description |
|---|---|
| `/help` | Show this help |
| `/faq` | Most frequently asked questions — all time |
| `/faq today` | Most frequently asked questions today |
| `/faq week` | Most frequently asked questions this week |
| `/faq month` | Most frequently asked questions this month |
| `/rates` | Rated responses — all time, as a table |
| `/rates today` | Rated responses today |
| `/rates week` | Rated responses this week |
| `/rates month` | Rated responses this month |
| `/rate <1-5>` | Rate the last response (1=very poor 🔴, 5=excellent 💚) |
| `/script` | Download code block(s) from the last response |
| `/task <id>` | Summarise status of task *id* |
| `/job <id>` | Analyse failure of job *id* |

For all other questions, type naturally — no slash needed.
"""

_TASK_CMD_RE_ST = re.compile(r"^/task\s+(\d+)\s*$", re.IGNORECASE)
_JOB_CMD_RE_ST = re.compile(r"^/job\s+(\d+)\s*$", re.IGNORECASE)


def _expand_slash_command(raw: str) -> tuple[str | None, str | None]:  # noqa: C901
    """Expand a slash command into a question or a help string.

    Returns a tuple ``(question, help_markdown)`` where exactly one element
    is non-None.  If the input is not a slash command both elements are None
    and the caller should treat it as a plain question.

    Args:
        raw: Raw text submitted by the user.

    Returns:
        ``(question, None)`` when the command maps to a question to submit,
        ``(None, help_md)`` when the command should display help inline,
        ``(None, None)`` when the input is not a slash command.
    """
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return None, None

    parts = stripped.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/help", "/?"):
        return None, _STREAMLIT_HELP

    m_task = _TASK_CMD_RE_ST.match(stripped)
    if m_task:
        return (
            f"Summarize the status of task {m_task.group(1)} including dataset info.",
            None,
        )

    m_job = _JOB_CMD_RE_ST.match(stripped)
    if m_job:
        return (
            f"Analyse the failure of job {m_job.group(1)} and explain why it failed.",
            None,
        )

    if cmd == "/rates":
        scope = args[0].lower() if args else ""
        # Build the OpenSearch query here so the LLM never has to construct
        # or modify it — that was the source of malformed range queries.
        if scope == "today":
            label = "today"
            query_json = (
                '{"query":{"bool":{"must":[' +
                '{"exists":{"field":"rating"}},' +
                '{"range":{"@timestamp":{"gte":"now/d"}}}' +
                ']}},' +
                '"sort":[{"rating":{"order":"desc"}}]}'
            )
        elif scope == "week":
            label = "this week"
            query_json = (
                '{"query":{"bool":{"must":[' +
                '{"exists":{"field":"rating"}},' +
                '{"range":{"@timestamp":{"gte":"now-7d/d"}}}' +
                ']}},' +
                '"sort":[{"rating":{"order":"desc"}}]}'
            )
        elif scope == "month":
            label = "this month"
            query_json = (
                '{"query":{"bool":{"must":[' +
                '{"exists":{"field":"rating"}},' +
                '{"range":{"@timestamp":{"gte":"now-30d/d"}}}' +
                ']}},' +
                '"sort":[{"rating":{"order":"desc"}}]}'
            )
        else:
            label = "all time"
            query_json = (
                '{"query":{"exists":{"field":"rating"}},' +
                '"sort":[{"rating":{"order":"desc"}}]}'
            )
        return (
            f"Show all rated responses for {label} as a markdown table. "
            "Call the opensearch_promptlog_query tool. "
            "Pass max_hits=50 as a separate argument. "
            "Pass source_fields=[@timestamp,turn_number,session_id,"
            "model,tools_used,raw_question,rating] as a separate argument. "
            f"Pass this exact JSON as the query argument (nothing else): {query_json} "
            "Each hit includes _id automatically. "
            "Format as markdown table: "
            "Doc ID (full _id) | Time | Turn | Session (first 8 chars) "
            "| Model | Tools | Question (raw_question field, first 60 chars) | Rating. "
            "The Doc ID is the only way to retrieve the full response.",
            None,
        )

    if cmd == "/faq":
        scope = args[0].lower() if args else ""
        if scope == "today":
            return (
                "What are the most frequently asked questions in Bamboo today? "
                "Use a terms aggregation on raw_question.keyword filtered to gte:now/d.",
                None,
            )
        if scope == "week":
            return (
                "What are the most frequently asked questions in Bamboo this week? "
                "Use a terms aggregation on raw_question.keyword filtered to gte:now-7d/d.",
                None,
            )
        if scope == "month":
            return (
                "What are the most frequently asked questions in Bamboo this month? "
                "Use a terms aggregation on raw_question.keyword filtered to gte:now-30d/d.",
                None,
            )
        return (
            "What are the most frequently asked questions in Bamboo across all time? "
            "Use a terms aggregation on raw_question.keyword with no date filter.",
            None,
        )

    if cmd == "/rate":
        scope = args[0] if args else ""
        try:
            n = int(scope)
        except (ValueError, TypeError):
            return None, "❓ Usage: `/rate <1-5>` — e.g. `/rate 4`"
        if not 1 <= n <= 5:
            return None, "❓ Rating must be between 1 and 5."
        return f"__rate__{n}", None

    if cmd == "/script":
        return (
            None,
            "Use the **⬇ Download** button(s) below the last response "
            "to save code block(s) to your computer. "
            "The button appears automatically when the response contains a code block.",
        )

    # Unknown command — return a short inline error as help text
    return None, f"❓ Unknown command `{cmd}`. Type `/help` for available commands."


# ---------------------------------------------------------------------------
# Star rating widget
# ---------------------------------------------------------------------------

#: Star labels per rating value, colour-coded via emoji.
_RATING_STARS: dict[int, str] = {
    1: "⭐",
    2: "⭐⭐",
    3: "⭐⭐⭐",
    4: "⭐⭐⭐⭐",
    5: "⭐⭐⭐⭐⭐",
}

#: Button labels with colour-coded text shown inside each star button.
_RATING_LABELS: dict[int, str] = {
    1: "🔴 1",
    2: "🟠 2",
    3: "🟡 3",
    4: "🟢 4",
    5: "💚 5",
}


def _extract_code_blocks_st(text: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks from a Markdown response.

    Args:
        text: Raw Markdown response text.

    Returns:
        List of ``(language, code)`` tuples.
    """
    pattern = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
    return [(lang.lower(), code.strip()) for lang, code in pattern.findall(text)]


def _extract_suggested_filename_st(text: str) -> str:
    """Extract a suggested filename from a Markdown response.

    Recognises ``Script: foo.py``, ``File: foo.py``, ``**Script:** foo.py``
    and code fences with inline filenames.

    Args:
        text: Raw Markdown response text.

    Returns:
        Suggested filename or empty string.
    """
    label_re = re.compile(
        r"(?:^|[\n\r])[ \t]*\*{0,2}(?:Script|File|Filename)\*{0,2}:[ \t]*([\w.][\w.\-]*)"
        r"[ \t]*(?:[\n\r]|$)",
        re.IGNORECASE | re.UNICODE,
    )
    m = label_re.search(text)
    if m:
        return m.group(1).strip()
    simple_re = re.compile(
        r"(?:Script|File|Filename):\s*([\w.\-]+\.(?:py|sh|js|ts|cpp|c|java|go|rs|r|sql|yaml|yml|toml|json))"
        r"\b",
        re.IGNORECASE,
    )
    m = simple_re.search(text)
    if m:
        return m.group(1).strip()
    fence_re = re.compile(r"```\w*\s+([\w.\-]+\.\w+)\s*\n")
    m = fence_re.search(text)
    if m:
        return m.group(1).strip()
    # "Save the script as random_numbers.C" / "name it foo.py" etc.
    save_re = re.compile(
        r"(?:save|name|call)\s.*?\bas\s+([\w.][\w.\-]*\.\w+)",
        re.IGNORECASE,
    )
    m = save_re.search(text)
    if m:
        return m.group(1).strip()
    return ""


_LANG_EXT_MAP: dict[str, str] = {
    "python": ".py", "py": ".py",
    "bash": ".sh", "sh": ".sh", "shell": ".sh",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "json": ".json", "yaml": ".yaml", "yml": ".yaml",
    "toml": ".toml", "cpp": ".cpp", "c": ".c", "root": ".C",
    "java": ".java", "ruby": ".rb", "go": ".go",
    "rust": ".rs", "sql": ".sql", "r": ".r",
}


def _render_script_download() -> None:
    """Render download button(s) for code block(s) in the last response.

    Extracts all fenced code blocks from the last assistant message.
    For each block, renders a ``st.download_button`` so the user can
    save the script to their local machine directly from the browser.
    No server-side file is written — the content is streamed to the
    browser as a download.
    """
    messages = st.session_state.get("messages", [])
    last_assistant = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        None,
    )
    if not last_assistant:
        return

    blocks = _extract_code_blocks_st(last_assistant)
    if not blocks:
        return

    suggested = _extract_suggested_filename_st(last_assistant)

    for i, (lang, code) in enumerate(blocks):
        ext = _LANG_EXT_MAP.get(lang, ".txt")
        if suggested and i == 0:
            fname = suggested
        elif suggested and len(blocks) > 1:
            base = suggested.rsplit(".", 1)[0] if "." in suggested else suggested
            fname = f"{base}_{i + 1}{ext}"
        else:
            label = lang if lang else "script"
            suffix = f"_{i + 1}" if len(blocks) > 1 else ""
            fname = f"{label}{suffix}{ext}"

        label_text = f"⬇ Download {fname}"
        st.download_button(
            label=label_text,
            data=(code + "\n").encode("utf-8"),
            file_name=fname,
            mime="text/plain",
            key=f"dl_script_{i}_{hash(code) & 0xFFFFFF}",
        )


def _render_rating_widget(mcp: MCPClientSync) -> None:
    """Render the star rating widget below the last assistant response.

    Displays five colour-coded buttons (1–5) when no rating has been submitted
    for the current response.  Once the user clicks a button the rating is
    persisted via ``bamboo_promptlog_rate``, ``last_rating`` is stored in
    session state, and subsequent renders show only a static confirmation
    caption — preventing duplicate votes.

    Does nothing when:
    - No document has been indexed yet (``last_doc_id`` is None).
    - The ``bamboo_promptlog_rate`` tool is not registered on the server.

    Args:
        mcp: Connected MCP client.
    """
    doc_ref: tuple[str, str] | None = st.session_state.get("last_doc_id")
    if doc_ref is None:
        return
    tool_names: list[str] = st.session_state.get("tool_names", [])
    if "bamboo_promptlog_rate" not in tool_names:
        return

    current_rating: int | None = st.session_state.get("last_rating")
    index, doc_id = doc_ref

    # Once a rating has been submitted for this document, show only the
    # confirmation caption — do not render buttons again so the user cannot
    # vote more than once per response.
    if current_rating is not None:
        stars = _RATING_STARS.get(current_rating, str(current_rating))
        labels = {1: "Very poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
        st.caption(
            f"Your rating: {stars} — {labels.get(current_rating, '')} ({current_rating}/5)"
        )
        return

    st.markdown("**Rate this response:**")
    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        label = _RATING_LABELS[i]
        if col.button(label, key=f"rate_{i}_{doc_id}", use_container_width=True):
            try:
                raw = mcp.call_tool(
                    "bamboo_promptlog_rate",
                    {"index": index, "doc_id": doc_id, "rating": i},
                )
                text = _extract_text(raw) or "{}"
                parsed = json.loads(text)
                if parsed.get("error"):
                    st.error(f"Rating failed: {parsed['error']}", icon="🔴")
                else:
                    st.session_state["last_rating"] = i
                    st.rerun()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"Rating error: {exc}", icon="🔴")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit app entry point."""
    # set_page_config must be the first Streamlit call each run.
    st.set_page_config(
        page_title="Bamboo MCP",
        page_icon="🐼",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Base font size — Streamlit 1.58+ uses data-testid attributes;
           the old [class*="css"] selector no longer matches anything. */
        html, body {
            font-size: 16px !important;
        }
        /* Main content area */
        [data-testid="stMain"], [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"], [data-testid="stMarkdownContainer"] {
            font-size: 16px !important;
        }
        /* Chat messages */
        [data-testid="stChatMessage"], [data-testid="stChatMessageContent"] {
            font-size: 16px !important;
        }
        /* Prose inside chat messages */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] td {
            font-size: 16px !important;
        }
        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            font-size: 15px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_session()
    _seed_from_query_params()

    transport, http_url, bearer_token, plugin_id, stdio_command = _render_sidebar()

    # Set a pre-connection display name from plugin_id so the UI never shows
    # a stale or generic label before the server has been contacted.
    # _connect() will overwrite this with the ui_manifest display_name once
    # the server responds.
    if not st.session_state.get("display_name"):
        st.session_state["display_name"] = plugin_id.upper()

    # If the user switches plugin in the sidebar while connected, the cached
    # display_name belongs to the old plugin.  Clear it so the pre-connection
    # label updates immediately to the new plugin_id, then re-fetch the
    # new plugin's ui_manifest so the display_name is correct.
    _plugin_changed = st.session_state.get("last_plugin_id") != plugin_id
    if _plugin_changed:
        st.session_state["display_name"] = plugin_id.upper()
        st.session_state["last_plugin_id"] = plugin_id

    # Build (or retrieve cached) MCP client
    try:
        mcp = _get_mcp_client(
            transport=transport,
            http_url=http_url,
            bearer_token=bearer_token,
            stdio_command=stdio_command,
            trace_file=st.session_state["trace_file"],
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Failed to create MCP client: {exc}")
        st.code(traceback.format_exc())
        st.stop()

    # Re-fetch branding when the plugin changes while already connected,
    # now that mcp is in scope.
    if _plugin_changed and st.session_state.get("server_ok"):
        try:
            _connect(mcp, plugin_id)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # Connect / refresh server metadata if not yet done.
    # After a successful connect we immediately rerun so the sidebar updates
    # from "Not connected" to "Connected" without waiting for the first question.
    if not st.session_state.get("server_ok"):
        with st.spinner("Connecting to server…"):
            try:
                _connect(mcp, plugin_id)
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"Could not connect to MCP server: {exc}")
                st.info(
                    "Check that the server is running and the URL/token are correct, "
                    "then click **Reconnect** in the sidebar."
                )
                st.stop()

    # Page header
    st.title(st.session_state.get("display_name", plugin_id.upper()))
    llm_info = st.session_state.get("llm_info", "")
    if llm_info:
        st.caption(f"🤖 {llm_info}")

    _render_chat(mcp, transport)


if __name__ == "__main__":
    main()
