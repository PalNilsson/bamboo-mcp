"""LLM-backed planner tool.

This module provides an MCP tool that asks an LLM to return a *machine-parseable*
execution plan describing which Bamboo tool(s) to call for a given user question.

The plan is validated with Pydantic to ensure:
  * a stable JSON shape,
  * type correctness,
  * lightweight semantic constraints (e.g., confidence in [0, 1]).

The planner is intentionally **separate** from the main answer/orchestration
tooling so that Bamboo can implement a hybrid strategy:
  * Use deterministic routing for obvious cases.
  * Fall back to LLM planning when intent is ambiguous or multi-step.

Note:
  The planner only returns a plan. It does not execute tools.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator

from bamboo.llm.runtime import get_llm_manager, get_llm_selector
from bamboo.llm.types import GenerateParams, Message
from bamboo.tools.base import MCPContent, text_content
from bamboo.tools._tool_names import wire_tool_definitions
from bamboo.tools._tool_profiles import PROFILE_ORCHESTRATED, is_advertised
from bamboo.tracing import EVENT_LLM_CALL, span


#: Tool profiles the planner catalog draws from.
#:
#: Pinned to the orchestrated surface rather than following
#: ``BAMBOO_TOOL_PROFILE``, because the planner serves that surface by
#: construction: it proposes compound tools and ``bamboo_executor`` runs them
#: in-process.  Were this to follow the environment, a server started with
#: ``BAMBOO_TOOL_PROFILE=primitive`` would drop ``panda_log_analysis`` from the
#: catalog while ``bamboo_answer`` remained advertised and callable, so every
#: question routed through the planner would lose log analysis.
#:
#: It is also what keeps primitives out of the tool-retrieval index planned for
#: Track A unconditionally, rather than only when the server happens to be in
#: orchestrated mode.
_PLANNER_PROFILES: frozenset[str] = frozenset({PROFILE_ORCHESTRATED})


class PlanRoute(str, Enum):
    """Routing decision returned by the planner."""

    FAST_PATH = "FAST_PATH"
    PLAN = "PLAN"
    RETRIEVE = "RETRIEVE"


class ToolCall(BaseModel):
    """A single tool invocation proposed by the planner."""

    tool: str = Field(
        ..., min_length=1, description="Tool name as used by Bamboo (e.g., 'panda_task_status' or 'atlas.task_status')."
    )
    arguments: dict[str, Any] = Field(default_factory=dict, description="JSON arguments to pass to the tool.")
    namespace: str | None = Field(
        default=None,
        description=(
            "Optional namespace hint used when resolving tools via entry points (e.g., 'atlas'). "
            "If omitted, the executor may resolve by suffix/name."
        ),
    )


class RetrievalQuery(BaseModel):
    """Optional retrieval hint (future-proof; DB/vector store may implement this)."""

    type: str = Field(
        ..., description="Retrieval method (e.g., 'exact', 'by_entity', 'embedding')."
    )
    keys: dict[str, Any] = Field(default_factory=dict, description="Structured keys (e.g., task_id, job_id, error_signature).")


class ReusePolicy(BaseModel):
    """Policy hints about reusing past answers/evidence."""

    allow_final_answer_reuse: bool = Field(
        default=False,
        description="If true, a previously stored final answer may be returned without fresh tool calls (use carefully).",
    )
    allow_pattern_reuse: bool = Field(
        default=True,
        description="If true, previously observed troubleshooting patterns may be reused as suggestions.",
    )
    requires_fresh_evidence: bool = Field(
        default=True,
        description="If true, the executor should obtain fresh tool evidence before presenting a final answer.",
    )


class Plan(BaseModel):
    """Top-level plan object returned by the planner."""

    route: PlanRoute = Field(..., description="How the executor should proceed: FAST_PATH, PLAN, or RETRIEVE.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Planner confidence in [0, 1]."
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Ordered list of tool calls to execute. Empty is allowed for RETRIEVE-only decisions.",
    )
    retrieval_query: RetrievalQuery | None = Field(
        default=None,
        description="Optional retrieval instruction. Not required until DB/vector store integration.",
    )
    reuse_policy: ReusePolicy = Field(
        default_factory=ReusePolicy,
        description="Hints controlling reuse of past answers/patterns.",
    )
    explain: str = Field(
        default="",
        max_length=1000,
        description="Short human-readable rationale for debugging and test traces.",
    )

    @model_validator(mode="after")
    def _check_semantics(self) -> "Plan":
        """Validate cross-field semantics."""
        if self.route in (PlanRoute.FAST_PATH, PlanRoute.PLAN) and not self.tool_calls:
            raise ValueError("route FAST_PATH/PLAN requires at least one tool_calls entry")
        if self.route == PlanRoute.RETRIEVE and self.retrieval_query is None and not self.tool_calls:
            raise ValueError("route RETRIEVE requires retrieval_query and/or tool_calls")
        return self


def get_plan_json_schema() -> dict[str, Any]:
    """Return the exact JSON schema for the :class:`Plan` object.

    Returns:
        Dict[str, Any]: JSON Schema (draft-2020-12 compatible) generated by Pydantic.
    """
    return Plan.model_json_schema()


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_first_json_object(text: str) -> str:
    """Extract the first JSON object from an LLM response.

    The planner prompt requests *JSON only*, but models sometimes wrap output
    in code fences or include leading commentary. This helper extracts the
    first JSON object conservatively.

    Args:
        text: Raw model response text.

    Returns:
        str: A JSON object string.

    Raises:
        ValueError: If no JSON object could be extracted.
    """
    if not text:
        raise ValueError("Empty response")

    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Fallback: find first '{' and attempt to parse the largest valid object.
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")

    candidate = text[start:].strip()
    # Try progressively shorter suffixes to find a valid JSON object.
    for end in range(len(candidate), 1, -1):
        snippet = candidate[:end]
        try:
            json.loads(snippet)
            return snippet
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    raise ValueError("Unable to extract valid JSON object")


def _build_atlas_planner_prompt(schema_compact: str) -> str:
    """Return the planner system prompt for the ATLAS / ePIC plugin.

    Args:
        schema_compact: Compact JSON schema string for the Plan output format.

    Returns:
        System prompt string.
    """
    return (
        "You are a tool planner for an MCP server. "
        "Your job is to output a single JSON object that conforms exactly to the provided JSON Schema.\n\n"
        "Hard rules:\n"
        "- Output MUST be valid JSON (no trailing commas).\n"
        "- Output MUST be a single JSON object, and MUST NOT be wrapped in markdown fences.\n"
        "- Do not include any explanation outside the JSON object.\n"
        "- Only propose tools that appear in the provided tool catalog.\n"
        "- Be conservative: if uncertain, set route='PLAN' and confidence lower.\n\n"
        "Routing guidance:\n"
        "- If the question contains a task ID (hints.task_id present): "
        "use panda_task_status. route=FAST_PATH.\n"
        "- If the question asks to diagnose/analyse/why a job failed "
        "(hints.job_id present + failure keywords): "
        "use panda_log_analysis. route=FAST_PATH.\n"
        "- If panda_log_analysis has already been called and returned "
        "traceback_available=true with a non-null deepest_pilot_frame, AND the "
        "user wants to understand the pilot code or why the exception was raised "
        "(e.g. 'why did the pilot code fail', 'show me the source', 'what is "
        "wrong with the pilot', 'can this be fixed'): use atlas.pilot_source_analysis, "
        "passing job_id, the log_excerpt and the pilot_version from the prior "
        "evidence. route=PLAN.\n"
        "- If the question asks what a job was actually doing when it was "
        "killed or stalled, or explicitly asks for a core dump, gdb, or a "
        "backtrace (e.g. 'analyse the core dump of job 123', 'what was job 123 "
        "stuck on', 'why did job 123 hang'), and a job ID is present: "
        "use atlas.core_dump_analysis with action='start'. Pass mode='hang' "
        "when the job was killed as a looping job (pilot error code 1150), "
        "mode='crash' for a segfault or abort, and mode='auto' when unsure. "
        "This tool is ATLAS-only and takes about a minute; never propose it "
        "for ePIC or any other plugin. route=FAST_PATH.\n"
        "- If the question asks about a job (hints.job_id present, no failure keywords): "
        "use panda_job_status. route=FAST_PATH.\n"
        "- If the question asks about BOTH pilot counts/status AND job counts/failures "
        "at a site (e.g. 'pilots and jobs at BNL', 'site health'): "
        "use panda_harvester_workers AND panda_jobs_query together. "
        "Pass site= to panda_harvester_workers and queue= to panda_jobs_query. "
        "route=FAST_PATH.\n"
        "- If the question asks about pilot failure rates, failure percentages, or "
        "which sites had high failure rates over a time window "
        "(e.g. 'which sites had pilot failures above 20% today', "
        "'pilot failure rate at BNL this week', 'sites with most failed pilots'): "
        "use atlas.harvester_timeseries with status='failed'. "
        "Pass site= if a single site is mentioned; omit site= for cross-site queries. "
        "Always express from_dt/to_dt as absolute ISO-8601 strings "
        "(e.g. '2026-06-12T00:00:00'), never as relative expressions like 'now-6h'. "
        "route=FAST_PATH.\n"
        "- If the question asks about live pilot counts or Harvester worker status "
        "right now (e.g. 'how many pilots are running', 'how many pilots submitted', "
        "'pilot counts at X', 'pilots running at X'): "
        "use panda_harvester_workers. "
        "Always express from_dt/to_dt as absolute ISO-8601 strings "
        "(e.g. '2026-06-12T00:00:00'), never as relative expressions like 'now-6h'. "
        "route=FAST_PATH.\n"
        "- If the question asks about live job counts, job failures, job status, "
        "or error rates at a site (e.g. 'how many jobs failed', 'job failure rate', "
        "'top errors at X'): use panda_jobs_query. route=FAST_PATH.\n"
        "- If the question asks about job performance metrics from the "
        "historical OpenSearch index — stage-in time, stage-out time, wall-clock "
        "time, queue wait time, payload execution time, pilot setup time, "
        "memory usage (RSS/PSS/vmem/swap), memory-leak rate/diagnostics, "
        "CPU efficiency, HS06 accounting, I/O throughput or data volume, "
        "pilot/execution/DDM error codes, task campaign context, carbon "
        "footprint, software environment (ATLAS release, cmtconfig, "
        "lsetup time, OS version, Python version), or "
        "any pilottiming sub-field "
        "(e.g. 'average stage-in time at BNL', 'maximum queue time for failed jobs', "
        "'total wall-clock time at CERN today', 'average RSS memory at BNL', "
        "'CPU efficiency at IN2P3', 'total HS06-seconds at TRIUMF', "
        "'average write throughput at CERN', 'CO2 footprint per job', "
        "'average memory leak rate at CERN', 'which Python versions are used', "
        "'show me all Python versions used by jobs this week', "
        "'OS version breakdown at BNL', 'average lsetup time at CERN'): "
        "use atlas.job_stats. Pass site= if a site is mentioned. route=FAST_PATH.\n"
        "- If the question asks about a site's queue configuration: "
        "use panda_queue_info. route=FAST_PATH.\n"
        "- If the question asks whether the PanDA server is alive, OK, running, or "
        "healthy (e.g. 'is PanDA alive?', 'is the PanDA server OK?', 'is PanDA up?', "
        "'PanDA server status'): use panda_server_health. route=FAST_PATH.\n"
        "- code_query requires an EXPLICIT source file path in the question "
        "(e.g. 'pilot/util/processes.py', 'src/foo.py'). "
        "NEVER use code_query for conceptual questions like 'how does X work', "
        "'what is X', 'explain X', or 'tell me about X'. "
        "Those must use panda_doc_search instead.\n"
        "- For ALL other questions (general knowledge, concepts, how-to, 'what is'): "
        "use panda_doc_search AND panda_doc_bm25 together. route=RETRIEVE. "
        "Never answer general questions from the LLM alone — always retrieve first.\n\n"
        f"JSON Schema (must match exactly):\n{schema_compact}\n"
    )


def _build_cgsim_planner_prompt(schema_compact: str) -> str:
    """Return the planner system prompt for the CGSim plugin.

    Contains no knowledge of PanDA tools — only CGSim tools.

    Args:
        schema_compact: Compact JSON schema string for the Plan output format.

    Returns:
        System prompt string.
    """
    return (
        "You are a tool planner for the CGSim simulation assistant (Bamboo MCP). "
        "Your job is to output a single JSON object that conforms exactly to the provided JSON Schema.\n\n"
        "Hard rules:\n"
        "- Output MUST be valid JSON (no trailing commas).\n"
        "- Output MUST be a single JSON object, and MUST NOT be wrapped in markdown fences.\n"
        "- Do not include any explanation outside the JSON object.\n"
        "- Only propose tools that appear in the provided tool catalog.\n"
        "- Be conservative: if uncertain, set route='PLAN' and confidence lower.\n\n"
        "Routing guidance:\n"
        "- If the question asks about simulation results — job timings, execution durations, "
        "queue wait times, file transfer speeds, network congestion, site allocation, "
        "disk I/O, retry rates, CPU/storage utilisation, or any data recorded in the "
        "simulation database — use cgsim.sim_query. route=FAST_PATH. "
        "This includes generic questions like 'show me all jobs', 'list all job IDs', "
        "'what happened during the simulation', 'which site was busiest'.\n"
        "  IMPORTANT: pass the user's question to cgsim.sim_query VERBATIM — do not "
        "expand, rephrase, or add detail. The tool handles its own SQL generation.\n"
        "- If the question asks about how CGSim or SimGrid works, plugin APIs, "
        "configuration options, or concepts (e.g. 'what is a netzone?', "
        "'how do I write a plugin?', 'what does assignJob do?'): "
        "use cgsim.doc_search AND cgsim.doc_bm25 together. route=RETRIEVE.\n"
        "- When in doubt, prefer cgsim.sim_query over the doc tools — most questions "
        "in this context are about simulation results, not documentation.\n\n"
        f"JSON Schema (must match exactly):\n{schema_compact}\n"
    )


def build_planner_system_prompt(schema: dict[str, Any], plugin_id: str = "") -> str:
    """Build the system prompt for the planner.

    Selects a plugin-specific prompt when *plugin_id* is recognised, falling
    back to the ATLAS prompt for unknown plugins.

    Args:
        schema: JSON schema that the model output must conform to.
        plugin_id: Active plugin identifier (e.g. ``"cgsim"``, ``"atlas"``).

    Returns:
        str: System prompt text.
    """
    schema_compact = json.dumps(schema, ensure_ascii=False)
    if plugin_id == "cgsim":
        return _build_cgsim_planner_prompt(schema_compact)
    return _build_atlas_planner_prompt(schema_compact)


def build_planner_user_prompt(
    question: str,
    tool_catalog: list[dict[str, Any]],
    hints: dict[str, Any] | None = None,
) -> str:
    """Build the user prompt for the planner.

    Args:
        question: User question.
        tool_catalog: List of tool definition objects (name, description, inputSchema).
        hints: Optional structured hints from deterministic extraction.

    Returns:
        str: User prompt text.
    """
    payload = {
        "question": question,
        "tool_catalog": tool_catalog,
    }
    if hints:
        payload["hints"] = hints
    return (
        "Given the user's question, choose the best route and propose tool calls. "
        "Use the hints if they are present.\n\n"
        "Return only the JSON plan.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _tool_def_from_obj(
    obj: Any,
    fallback_name: str = "",
    profiles: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Return a compact tool definition dict from a tool object, or None.

    Args:
        obj: Tool object expected to have a ``get_definition`` method.
        fallback_name: Name to use if the definition doesn't include one.
        profiles: Optional tool profiles to filter by.  When given, a
            definition advertised under none of them yields ``None``.  The
            check happens here rather than in the caller because the raw
            definition is already in hand, and the returned dict drops the
            ``profiles`` key along with everything else outside the three
            fields below.

    Returns:
        Dict with ``name``, ``description``, and ``inputSchema`` keys, or None
        if the object has no usable definition or is filtered out by
        *profiles*.
    """
    get_def = getattr(obj, "get_definition", None)
    if not callable(get_def):
        return None
    try:
        raw = get_def()
        if not isinstance(raw, dict):
            return None
        d: dict[str, Any] = cast(dict[str, Any], raw)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if profiles is not None and not is_advertised(d, profiles):
        return None
    name: str = str(d.get("name") or fallback_name)
    if not name:
        return None
    return {
        "name": name,
        "description": str(d.get("description", "")),
        "inputSchema": d.get("inputSchema", {}),
    }


def _collect_tool_catalog(namespaces: list[str] | None = None) -> list[dict[str, Any]]:
    """Collect a compact tool catalog for the planner.

    Includes both the statically-registered core TOOLS (from ``bamboo.core``)
    and any tools discovered via Python entry points.  Core tools are always
    included regardless of the ``namespaces`` filter so the LLM is aware of
    the full built-in toolset.

    Tools restricted to a profile outside :data:`_PLANNER_PROFILES` are
    excluded from both sources.  This does not follow ``BAMBOO_TOOL_PROFILE``;
    see that constant for why.

    Args:
        namespaces: Optional list of namespaces to include for *entry-point*
            tools (e.g. ['atlas']). Core tools are always included.

    Returns:
        List[Dict[str, Any]]: Tool definitions suitable for prompt inclusion.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def _add(obj: Any, fallback_name: str = "") -> None:
        entry = _tool_def_from_obj(obj, fallback_name, profiles=_PLANNER_PROFILES)
        if entry and entry["name"] not in seen:
            seen.add(entry["name"])
            out.append(entry)

    # Internal/infrastructure tools that the planner must never propose —
    # they are synthesis helpers, not evidence sources.
    _INTERNAL_TOOLS: frozenset[str] = frozenset({
        "bamboo_llm_answer",
        "bamboo_answer",
        "bamboo_plan",
        "bamboo_health",
    })

    # Core tools that belong to the PanDA/ATLAS domain.  When a non-PanDA
    # namespace is active (e.g. "cgsim") these are excluded from the catalog
    # so the LLM planner cannot accidentally pick them.
    _PANDA_CORE_TOOLS: frozenset[str] = frozenset({
        "panda_task_status",
        "panda_job_status",
        "panda_log_analysis",
        "panda_jobs_query",
        "panda_harvester_workers",
        "panda_server_health",
        "panda_doc_search",
        "panda_doc_bm25",
        "panda_queue_info",
        "cric_query",
    })
    # pilot_source_analysis is deliberately absent: this set filters the
    # built-in TOOLS dict, and pilot source analysis is a plugin entry point
    # (wire name atlas.pilot_source_analysis), so the namespace filter below
    # already excludes it for non-PanDA plugins.  Listing it here matched
    # nothing.

    # Determine whether to exclude PanDA core tools.
    # "atlas" and "epic" are PanDA-family plugins; all others are not.
    _PANDA_PLUGIN_NAMESPACES: frozenset[str] = frozenset({"atlas", "epic", ""})
    _exclude_panda = bool(namespaces) and not any(
        ns in _PANDA_PLUGIN_NAMESPACES for ns in namespaces
    )

    # 1) Statically-registered core tools — always included, except internals
    # and PanDA tools when a non-PanDA namespace is active.
    try:
        from bamboo.core import TOOLS  # pylint: disable=import-outside-toplevel
        for tool_name, tool_obj in TOOLS.items():
            if tool_name in _INTERNAL_TOOLS:
                continue
            if _exclude_panda and tool_name in _PANDA_CORE_TOOLS:
                continue
            _add(tool_obj, fallback_name=tool_name)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    # 2) Plugin tools discovered via entry points, named as clients see them.
    #
    # wire_tool_definitions() applies bamboo.core's own registration rules, so
    # a name in this catalog is a name the planner can actually propose.  The
    # previous loop passed the entry-point key only as a *fallback* and let
    # get_definition()["name"] win, which advertised internal names the server
    # does not expose: the catalog said "core_dump_analysis" while the routing
    # guidance above says "atlas.core_dump_analysis" and the hard rule says to
    # propose only catalogued tools.  Faced with that contradiction the planner
    # dropped the guidance and fell back to panda_log_analysis, so an explicit
    # request to analyse a core dump was answered with a log analysis.
    for defn in wire_tool_definitions():
        name = str(defn.get("name") or "")
        if not name:
            continue
        if not is_advertised(defn, _PLANNER_PROFILES):
            continue
        if namespaces:
            ns = name.split(".", 1)[0] if "." in name else ""
            if ns not in namespaces:
                continue
        if name in seen:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "description": str(defn.get("description", "")),
            "inputSchema": defn.get("inputSchema", {}),
        })

    return out


class BambooPlannerTool:
    """LLM-backed planner that outputs a JSON plan."""

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool discovery definition.

        Returns:
            Dict[str, Any]: Tool definition compatible with MCP discovery.
        """
        return {
            "name": "bamboo_plan",
            "description": (
                "Decompose a complex question into a structured plan of tool calls. "
                "Use when a question requires multiple steps, combines task and "
                "job lookups, or when intent is ambiguous and deterministic routing "
                "is insufficient. Returns a validated JSON plan by default, or a "
                "synthesised natural-language answer when execute=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User question to plan for."},
                    "hints": {
                        "type": "object",
                        "description": "Optional structured hints from deterministic extraction.",
                    },
                    "namespaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of namespaces to include in the tool catalog (e.g. ['atlas']).",
                    },
                    "temperature": {"type": "number", "default": 0.0, "description": "Planner temperature (keep low)."},
                    "max_tokens": {"type": "integer", "default": 900, "description": "Max completion tokens."},
                    "execute": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, execute the plan and return a synthesised answer. "
                            "If false (default), return the raw JSON plan for inspection."
                        ),
                    },
                    "messages": {
                        "type": "array",
                        "description": (
                            "Optional full chat history as a list of {role, content} dicts. "
                            "Used to thread conversation context into the synthesised answer "
                            "when execute=true."
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
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:
        """Execute the planner call.

        When ``execute`` is ``False`` (the default) the validated plan is
        returned as a JSON text block.  When ``execute`` is ``True`` the plan
        is immediately executed by
        :func:`~bamboo.tools.bamboo_executor.execute_plan` and a synthesised
        natural-language answer is returned instead.

        Args:
            arguments: Tool arguments.  Must contain ``question``.  Optional
                keys: ``hints``, ``namespaces``, ``temperature``,
                ``max_tokens``, ``execute``, ``messages``.

        Returns:
            List[MCPContent]: Single text content block — either a JSON plan
            (``execute=False``) or a synthesised answer (``execute=True``).

        Raises:
            ValueError: If the question is missing.
            RuntimeError: If the LLM runtime is not initialised.
        """
        question = str(arguments.get("question", "") or "").strip()
        if not question:
            raise ValueError("'question' is required")

        namespaces = arguments.get("namespaces")
        namespaces_list = [str(x) for x in namespaces] if isinstance(namespaces, list) else None

        hints = arguments.get("hints")
        hints_dict = hints if isinstance(hints, dict) else None

        plugin_id: str = str(arguments.get("plugin_id", "") or "").strip().lower()

        schema = get_plan_json_schema()
        tool_catalog = _collect_tool_catalog(namespaces=namespaces_list)

        system = build_planner_system_prompt(schema, plugin_id=plugin_id)
        user = build_planner_user_prompt(question=question, tool_catalog=tool_catalog, hints=hints_dict)
        planner_messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        temperature = float(arguments.get("temperature", 0.0))
        max_tokens = int(arguments.get("max_tokens", 900))

        text = await _call_default_llm(planner_messages, temperature=temperature, max_tokens=max_tokens)

        # Validate, with one optional repair attempt.
        plan: Plan
        try:
            plan = Plan.model_validate_json(extract_first_json_object(text))
        except Exception as e:  # pylint: disable=broad-exception-caught
            repair_messages = planner_messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. "
                        "Return ONLY a corrected JSON object that matches the schema exactly.\n"
                        f"Validation error: {str(e)[:300]}"
                    ),
                }
            ]
            text2 = await _call_default_llm(repair_messages, temperature=0.0, max_tokens=max_tokens)
            try:
                plan = Plan.model_validate_json(extract_first_json_object(text2))
            except Exception as e2:  # pylint: disable=broad-exception-caught
                # Tools must never raise — return a structured error response.
                return text_content(json.dumps({
                    "error": "planner_parse_failure",
                    "message": (
                        "The LLM planner did not return a valid JSON plan after "
                        f"two attempts. Last error: {str(e2)[:300]}"
                    ),
                    "raw_output": text2[:500],
                }))

        # When execute=False (default), return the raw JSON plan for inspection.
        execute = bool(arguments.get("execute", False))
        if not execute:
            # Pydantic's JSON helpers intentionally limit json.dumps kwargs.
            # Use model_dump() + json.dumps to keep Unicode readable.
            return text_content(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))

        # When execute=True, run the plan and return a synthesised answer.
        # Extract conversation history from the optional ``messages`` argument
        # so the LLM synthesis step has context for follow-up questions.
        # Imports are deferred to avoid a circular-import cycle at module load.
        from bamboo.tools.base import coerce_messages  # pylint: disable=import-outside-toplevel
        from bamboo.tools.bamboo_answer import _extract_history  # pylint: disable=import-outside-toplevel
        from bamboo.tools.bamboo_executor import execute_plan  # pylint: disable=import-outside-toplevel

        messages_raw: list[Any] = arguments.get("messages") or []
        chat_messages = coerce_messages(messages_raw) if messages_raw else []
        history = _extract_history(chat_messages, question) if chat_messages else []

        return await execute_plan(plan, question, history, plugin_id=plugin_id or "atlas")


async def _call_default_llm(messages: list[Message], temperature: float, max_tokens: int) -> str:
    """Call the configured default LLM profile and emit a trace span.

    Args:
        messages: Chat messages.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.

    Returns:
        str: Raw model text response.
    """
    selector = get_llm_selector()
    manager = get_llm_manager()

    default_profile = getattr(selector, "default_profile", "default")
    registry = getattr(selector, "registry", None)
    if registry is None:
        raise RuntimeError("LLM selector does not expose a registry.")

    model_spec = registry.get(default_profile)
    client = await manager.get_client(model_spec)
    # Emit a llm_call span so token counts and wall-clock time appear in
    # /costs and /tracing.  generate() is called *inside* the span so the
    # duration reflects the real LLM latency, not zero.
    async with span(
        EVENT_LLM_CALL,
        tool="bamboo_plan",
        provider=getattr(model_spec, "provider", ""),
        model=getattr(model_spec, "model", ""),
    ) as s:
        resp = await client.generate(
            messages=messages,
            params=GenerateParams(temperature=temperature, max_tokens=max_tokens),
        )
        usage = resp.usage
        s.set(
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )

    return resp.text


bamboo_plan_tool = BambooPlannerTool()
