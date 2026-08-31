"""Plan executor for Bamboo — runs a validated :class:`~bamboo.tools.planner.Plan`.

This module provides the execution layer that sits between the planner and the
LLM synthesiser.  Given a ``Plan`` produced by :mod:`bamboo.tools.planner`, it:

1. Iterates ``plan.tool_calls`` in order.
2. Resolves each tool via the core ``TOOLS`` registry or the plugin entry-point
   loader.
3. Validates arguments with :func:`~bamboo.core._validate_arguments`.
4. Calls ``await tool.call(args)`` and collects ``list[MCPContent]``.
5. Unpacks JSON evidence from evidence tools.
6. Selects a synthesis system prompt based on which tools were called.
7. Synthesises a final natural-language answer via the LLM.

All functions are intentionally **pure orchestration** — no experiment-specific
logic lives here.  Synthesis prompts are kept as module-level constants so they
can be updated independently of routing logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from bamboo.llm.types import Message
from bamboo.session_scope import EVIDENCE_BUCKET, ScopedMapping
from bamboo.tools.base import MCPContent, text_content
from bamboo.tools.llm_passthrough import bamboo_llm_answer_tool
from bamboo.tools.loader import find_tool_by_name
from bamboo.tools._tool_names import canonical_tool_name
from bamboo.tools.planner import Plan
from bamboo.tracing import EVENT_PLAN, EVENT_RETRIEVAL, EVENT_SYNTHESIS, span

# ---------------------------------------------------------------------------
# Session-scoped evidence store
#
# Populated by execute_plan() after every successful tool call so that the
# TUI /json and /inspect commands can retrieve the last evidence dict without
# re-fetching from BigPanDA.  A separate "last_tool" key tracks which tool ran
# most recently so callers can retrieve the most relevant entry.
#
# Keys are **canonical wire names**, not whatever spelling the plan used —
# _execute_one_tool canonicalises before writing.  Readers may therefore match
# on an exact literal (_CORE_DUMP_TOOL, "panda_log_analysis") without also
# handling every alias _resolve_tool would have accepted.  Anything else
# writing to this store must canonicalise its key too.
#
# This looks like a module-level dict and behaves like one, but the storage
# belongs to the *active session* rather than the process — see
# bamboo.session_scope.  It has to, because two readers here consult the store
# across turns rather than within one:
#
#   * get_last_core_dump_offer() gates the bare-affirmative rule, so a
#     process-global store let one user's "yes" start a core-dump analysis on
#     another user's job;
#   * get_last_traceback_evidence() gates the rule 1b pilot-source route, so a
#     question could be answered against another user's traceback.
#
# Under stdio no session is bound and every access lands in the default
# bucket, which is byte-for-byte the old process-global behaviour.
# ---------------------------------------------------------------------------

_last_evidence_store: ScopedMapping = ScopedMapping(EVIDENCE_BUCKET)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthesis system prompt constants (moved from bamboo_answer.py)
# ---------------------------------------------------------------------------

_SYSTEM_LOG_ANALYSIS: str = (
    "You are AskPanDA for the ATLAS experiment.\n"
    "Given a user's question and a JSON evidence object containing PanDA job "
    "log analysis, write a concise diagnostic answer.\n"
    "Key evidence fields:\n"
    "- failure_type: Bamboo's classification of the failure.\n"
    "- exception_type / exception_message: the exception parsed from the log's "
    "Python traceback, when one was present.\n"
    "- deepest_pilot_frame: {pilot_path, lineno, func} — the innermost pilot3 "
    "frame in the traceback, i.e. the pilot code that was running when the "
    "exception surfaced.\n"
    "- exception_frames: the full call chain, including standard library frames.\n"
    "- log_excerpt: the extracted section of the log.\n"
    "- core_dump_probe_state: what the job-log file listing says about a core "
    "dump. One of 'present' (a core file with non-zero size), 'truncated' (a "
    "core file of zero length — the kernel was still writing it when the "
    "process was killed), 'timed_out' (no core file, and the pilot killed the "
    "job for looping), 'absent' (listing checked, no core file) or "
    "'not_probed' (the listing was not available — this means NOT LOOKED AT, "
    "not 'no core dump').\n"
    "- core_dump_available: true, false, or null. Null means not probed and "
    "must never be reported as 'no core dump'; it is the tri-state companion "
    "to core_dump_probe_state, which is more specific and should be preferred "
    "when the two are both present.\n"
    "- core_dump_candidates: the core files found, each with name, dirname, "
    "size_bytes and modification (UTC). core_dump_total_bytes is their sum.\n"
    "Rules:\n"
    "- State the failure classification clearly.\n"
    "- When exception_type is present, it is authoritative: build the diagnosis "
    "around it and around the call chain in exception_frames.\n"
    "- piloterrordiag is a summary written elsewhere in the pilot and can "
    "contradict the traceback. When they disagree, trust the traceback and say "
    "so explicitly. In particular, a diag mentioning 'payload execution' does "
    "NOT mean the payload ran — check where in the call chain the exception was "
    "actually raised.\n"
    "- Name the pilot file, line and function from deepest_pilot_frame.\n"
    "- Never infer a root cause from file names or file sizes you have read out "
    "of log_excerpt. If the evidence does not identify the cause, say that "
    "rather than guessing from which files exist or how large they are. This "
    "rule is about the raw excerpt: core_dump_probe_state and "
    "core_dump_candidates are a structured probe, not a directory listing you "
    "interpreted, and the next two rules govern them instead.\n"
    "- core_dump_probe_state is authoritative on whether a core dump exists. "
    "Never say there is no core dump when it is 'present' or 'truncated', and "
    "never say there is one when it is 'absent'. When it is 'not_probed', say "
    "the job log listing was unavailable rather than reporting either outcome. "
    "Report 'truncated' and 'timed_out' as themselves: a zero-length core and "
    "no core at all are different facts about what went wrong.\n"
    "- A core dump is evidence that the process died hard, not a diagnosis of "
    "why. Report that one exists, its name and size, and stop there. Do not "
    "construct a crash narrative from its modification timestamp — that "
    "timestamp records when the core was written, which for a job killed by "
    "the pilot is usually the kill itself, and reasoning backwards from it to "
    "an earlier crash time is guesswork. Only the analysed stack tells you "
    "where the process died.\n"
    "- Quote relevant log excerpts if present.\n"
    "- Suggest concrete next steps based on the failure type.\n"
    "- Do not offer to fetch the pilot source code; that offer is appended "
    "automatically when applicable.\n"
    "- Do not offer to fetch or analyse the core dump, and do not ask the user "
    "whether they want it analysed; that offer is appended automatically when "
    "applicable.\n"
    "- Do not include a Links section; links are appended automatically.\n"
    "- Keep it under ~10 bullet points.\n"
)

_SYSTEM_PILOT_SOURCE: str = (
    "You are AskPanDA for the ATLAS experiment.\n"
    "You have fetched the pilot3 source code from GitHub for the functions "
    "named in a job failure traceback.\n"
    "The evidence contains:\n"
    "- exception: the exception string that was raised.\n"
    "- traceback_frames: list of {pilot_path, func} dicts from the traceback.\n"
    "- source_snippets: dict keyed by 'path::function' containing the extracted "
    "source of each function.\n"
    "- github_urls: dict keyed by pilot path with links to the GitHub source.\n"
    "- missing_functions: functions named in the traceback but not found in source.\n"
    "- fetch_errors: any GitHub fetch failures.\n"
    "- github_repo / github_ref / pilot_version / ref_kind / ref_resolution: "
    "which pilot3 repository and ref the source was read from, and why.\n"
    "- line_verification: {checked, mismatches, version_skew} — whether the "
    "traceback line numbers agree with the fetched source.\n"
    "Rules:\n"
    "- Explain exactly which line in the deepest frame caused the exception and why.\n"
    "- ref_kind tells you how far to trust line numbers:\n"
    "  * 'release_tag' — the source is exactly what ran; quote line numbers freely.\n"
    "  * 'development_branch' — the job ran an unreleased pilot, so the source "
    "comes from a moving development branch. Describe functions by name, treat "
    "line numbers as approximate even when line_verification reports no skew, "
    "and state plainly that the code shown may differ from what ran.\n"
    "  * 'unknown_version' — the pilot version could not be determined; caveat "
    "line numbers the same way.\n"
    "- If line_verification.version_skew is true, the fetched source definitely "
    "does not match the build that ran the job: describe the function by name "
    "rather than by line number and say so. Mention ref_resolution so the "
    "reader knows what was read.\n"
    "- Quote the relevant source lines from source_snippets.\n"
    "- Describe whether this is a pilot infrastructure bug or a site configuration "
    "issue (e.g. missing UID in passwd/LDAP vs. a code defect).\n"
    "- If the fix is straightforward, suggest a concrete code change or workaround.\n"
    "- Include GitHub source links from github_urls so developers can navigate directly.\n"
    "- Do not include a Links section; links are appended automatically.\n"
    "- Keep it focused — this is a developer-level diagnosis, not a user-facing summary.\n"
)

# ---------------------------------------------------------------------------
# Mermaid diagram guidance — appended to synthesis prompts that benefit from
# diagrams (algorithms, flow descriptions, architecture explanations).
# ---------------------------------------------------------------------------

_MERMAID_GUIDANCE: str = (
    "\nDiagram rule:\n"
    "- If the answer describes a process, algorithm, state machine, or data flow "
    "that would be significantly clearer as a diagram, include exactly ONE Mermaid "
    "diagram in a fenced ```mermaid block immediately after your prose explanation.\n"
    "- Prefer 'flowchart TD' for algorithms and flows, 'sequenceDiagram' for "
    "protocols, and 'stateDiagram-v2' for state machines.\n"
    "- Only include a diagram when it genuinely adds clarity; omit it for "
    "simple status answers or factual lookups.\n"
    "- IMPORTANT: When a diagram is warranted, you MUST use a ```mermaid fenced "
    "block. Do NOT substitute an ASCII art box diagram, a plain text "
    "representation, or an indented text tree — these are not diagrams and "
    "will not render. If you cannot produce valid Mermaid syntax, omit the "
    "diagram entirely rather than falling back to text art.\n"
    "- Node label rules (strictly enforced — long labels are cut off in the renderer):\n"
    "  * Keep every node label to 20 characters or fewer.\n"
    "  * If a label needs more than 20 characters, split it across two lines "
    "using a Mermaid line break: A[\"First line<br/>second line\"].\n"
    "  * Do NOT use 'Component: Long Action Description' style labels — "
    "split them: A[\"Job Control<br/>Fetch Job\"].\n"
    "  * For stateDiagram-v2, keep state names short; add a note or description "
    "in the prose rather than cramming it into the node.\n"
    "- Mermaid syntax rules (these are strictly enforced by the renderer):\n"
    "  * stateDiagram-v2: transitions use '-->' not '->'. "
    "State labels with spaces must be quoted: state \"My State\" as s1. "
    "Do not use colons inside state names. "
    "Transition labels use colon syntax: s1 --> s2 : label.\n"
    "  * flowchart TD: node IDs must not contain spaces — use underscores or camelCase. "
    "Labels go in brackets: A[\"My Label\"] --> B[\"Other Label\"].\n"
    "  * All diagrams: no HTML tags other than <br/> inside node labels. "
    "No bare parentheses in node IDs. "
    "Test your syntax mentally — if unsure, use a simpler diagram type.\n"
    "  * If you include a %%{init}%% directive, use double-quoted JSON — "
    "single quotes are silently ignored by the Mermaid parser.\n"
)

_SYSTEM_CODE_QUERY: str = (
    "You are AskPanDA operating in superuser / developer mode.\n"
    "You have been given source code fetched from a GitHub repository.\n"
    "The evidence contains:\n"
    "- file_path: the file path within the repository (e.g. pilot/util/processes.py).\n"
    "- github_url: URL to browse the file on GitHub.\n"
    "- source: the module source text (may be the complete file, a function snippet, "
    "or a truncated file — see the 'truncated' flag).\n"
    "- truncated: true when the source was cut short because the file exceeds the "
    "context limit.  A '# --- TRUNCATED ---' comment at the end of the source marks "
    "the cut point and shows how many lines were omitted.\n"
    "- fetch_error: non-empty string when the file could not be retrieved.\n"
    "Rules:\n"
    "- Answer the user's specific question about the code directly and precisely.\n"
    "- Quote relevant source lines verbatim to support your answer.\n"
    "- If diagnosing a potential bug, identify the exact function and line number.\n"
    "- If explaining an algorithm, walk through it step by step.\n"
    "- Include the GitHub URL so the developer can navigate directly.\n"
    "- If fetch_error is non-empty, explain that the file could not be fetched "
    "and suggest checking the path or network access.\n"
    "- If truncated is false, the source is the COMPLETE file as fetched. "
    "You MUST NOT mention truncation, suggest the file is incomplete, or recommend "
    "fetching the full file. The file has been fully retrieved.\n"
    "- If truncated is true, note once at the end that the analysis covers only "
    "the retrieved portion and direct the user to the GitHub URL for the full file. "
    "Do NOT list this as a numbered finding or a code quality issue.\n"
    "- Do not fabricate source lines that are not in the evidence.\n"
    "- Do not claim an identifier, import, function, or variable is unused, missing, \n"
    "  or undeclared unless you have traced every occurrence of that name in the \n"
    "  source. If unsure, say so rather than asserting a false finding.\n"
    "- Do not invent bugs or issues not demonstrable from the source. Hedged \n"
    "  observations (\"may\", \"consider\", \"worth verifying\") are acceptable; \n"
    "  false definitive claims (\"is unused\", \"is missing\", \"is broken\") are not.\n"
)

_SYSTEM_JOB: str = (
    "You are AskPanDA for the ATLAS experiment.\n"
    "Given a user's question and a JSON evidence object from BigPanDA, "
    "write a concise, helpful answer about the job status.\n"
    "Rules:\n"
    "- If evidence.not_found is true: say the job was not found and suggest "
    "checking the ID.\n"
    "- Otherwise: summarise status, site, queue, pilot error, and timing.\n"
    "- Always include the BigPanDA monitor URL as plain text (not a Markdown "
    "hyperlink), e.g.: Monitor: https://bigpanda.cern.ch/job/12345/\n"
    "- Keep it under ~8 bullet points.\n"
)

_SYSTEM_TASK: str = (
    "You are AskPanDA, an assistant for the ATLAS experiment at CERN.\n"
    "You are given a user question and JSON metadata for a PanDA task fetched from BigPanDA.\n"
    "Answer the user's specific question using only data explicitly present in the metadata.\n"
    "Rules:\n"
    "- If evidence.not_found is true or evidence.http_status==404: clearly state that the task ID\n"
    "  was not found in BigPanDA. Say the task does not exist or the ID is incorrect. Do not\n"
    "  include a monitor link.\n"
    "- If evidence indicates a non-JSON or HTTP error (but not 404): explain that BigPanDA returned\n"
    "  an unexpected response and include the monitor_url so the user can check manually.\n"
    "- TASK STATUS: ``task_status`` is the ONLY authoritative source for the overall task\n"
    "  outcome. Report it verbatim (e.g. \'finished\', \'failed\', \'running\'). The jobs endpoint\n"
    "  returns only a SAMPLE of jobs (typically failed ones), so jobs_by_status reflects that\n"
    "  sample only — NOT all jobs. A task with task_status=\'finished\' completed successfully\n"
    "  even if the sample contains only failed jobs. NEVER infer the task outcome from\n"
    "  jobs_by_status. Always report task_status first, then the job breakdown.\n"
    "- JOB COUNTS: Use dsinfo[\'nfilesfinished\'] and dsinfo[\'nfilesfailed\'] for the\n"
    "  authoritative finished/failed counts when present — these are not inflated by retries.\n"
    "  Fall back to sum(jobs_by_piloterrorcode.values()) for failed count if dsinfo is absent.\n"
    "  jobs_by_status may be inflated by retries. A single job appears in BOTH\n"
    "  jobs_by_piloterrorcode AND errs_by_count (different views) — do NOT add them together.\n"
    "- TERMINOLOGY: When reporting nfilesfinished/nfilesfailed from dsinfo, call them JOBS not\n"
    "  files (e.g. \'49,988 jobs finished, 12 jobs failed\'). These are grid job counts, not\n"
    "  dataset file counts. Use \'files\' only when describing dataset contents.\n"
    "- PANDA IDs: The evidence has a ``failed_pandaids`` field — a plain list of integer\n"
    "  PanDA job IDs for the failed jobs (e.g. [7073513639, 7073514709, ...]). When the\n"
    "  user asks for job IDs, list every value in failed_pandaids. Note it is a sample\n"
    "  (up to 20) and point to the monitor URL for the complete list.\n"
    "- Provide a thorough summary covering: overall status, dataset name, job counts\n"
    "  (finished, failed, total), failure details (error codes and root cause), computing\n"
    "  sites involved, and any other fields relevant to the question. Use bullet points.\n"
    "- Otherwise: answer the question directly using only fields present in the metadata.\n"
    "- NEVER infer, guess, or derive values not explicitly in the data. If a requested value is\n"
    "  absent, say it is not available in the metadata rather than inventing it.\n"
    "- The Job list section below the metadata lists actual PanDA job IDs. Use ONLY those IDs\n"
    "  when answering questions about pandaids/job IDs. If the section says no jobs were\n"
    "  returned, say so — never derive job IDs from dataset IDs or any other field.\n"
    "- Include the BigPanDA monitor URL as plain text at the end in non-error cases,\n"
    "  e.g.: Monitor: https://bigpanda.cern.ch/task/12345/\n"
)

_SYSTEM_RAG: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You are given a user question and relevant excerpts retrieved from the "
    "PanDA/Bamboo documentation knowledge base.\n"
    "Rules:\n"
    "- Base your answer ONLY on the retrieved documentation excerpts.\n"
    "- If the excerpts fully answer the question, answer concisely from them.\n"
    "- If the excerpts are partially relevant, use only what is directly "
    "supported. Do NOT fill gaps with general knowledge.\n"
    "- If the excerpts do not answer the question, say so explicitly: tell the "
    "user the documentation did not contain enough information on this topic, "
    "and suggest they consult the official PanDA documentation or BigPanDA "
    "monitor. Do NOT invent or summarise from general knowledge.\n"
    "- Do not fabricate PanDA-specific details (task IDs, queue names, error "
    "codes, algorithm descriptions) that are not in the excerpts.\n"
    "- Be concise and precise. Prefer bullet points for multi-part answers.\n"
    "- When the user explicitly asks for a diagram, produce a Mermaid diagram. "
    "If the excerpts contain enough information, label it as based on "
    "documentation. If the excerpts are insufficient, draw from your general "
    "knowledge of PanDA/ATLAS and label it as based on general knowledge. "
    "Only omit the diagram entirely if the subject has no meaningful visual "
    "structure (e.g. a simple yes/no question).\n"
    + _MERMAID_GUIDANCE
)

_SYSTEM_RAG_NO_CONTEXT: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "No relevant documentation excerpts were retrieved for this question.\n"
    "Rules:\n"
    "- Answer from your general knowledge of PanDA, ATLAS, and related "
    "distributed computing systems. Clearly label your answer as based on "
    "general knowledge, not retrieved documentation.\n"
    "- Do NOT fabricate PanDA-specific runtime values such as specific task IDs, "
    "queue names, error codes, or live configuration settings.\n"
    "- Conceptual explanations (algorithms, workflows, architecture, terminology) "
    "are fair game — answer them directly and accurately.\n"
    "- If the question requires live operational data (job status, site health, "
    "pilot counts), explain that and suggest the BigPanDA monitor or the "
    "relevant Bamboo query.\n"
    "- Be concise and precise. Prefer bullet points for multi-part answers.\n"
    + _MERMAID_GUIDANCE
)

_SYSTEM_GENERIC: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have been given the results of one or more tool calls. Synthesise "
    "a clear, concise answer to the user's question based solely on the "
    "evidence provided.\n"
    "Rules:\n"
    "- Do not infer or fabricate values not present in the evidence.\n"
    "- If the evidence shows errors or empty results, explain that clearly.\n"
    "- Include relevant monitor URLs when available.\n"
    "- Be concise and prefer bullet points for multi-part answers.\n"
    + _MERMAID_GUIDANCE
)

# ---------------------------------------------------------------------------
# Per-plugin synthesis prompt overrides
# Keys are ASKPANDA_PLUGIN values; missing keys fall back to the defaults above.
# ---------------------------------------------------------------------------

_SYSTEM_RAG_CGSIM: str = (
    "You are Bamboo, an expert assistant for AskCGSim and SimGrid distributed "
    "computing simulation, with specific knowledge of the CGSim/PanDA integration.\n"
    "CGSim is a SimGrid-based framework for simulating large-scale computing grids "
    "such as the WLCG. It ingests historical PanDA job records for calibration and "
    "is designed to simulate infrastructures managed by PanDA. Questions about the "
    "PanDA/CGSim connection — such as how to simulate PanDA brokerage, how CGSim "
    "uses PanDA job logs for calibration, or how to model ATLAS/PanDA workloads in "
    "CGSim — are explicitly in scope and should be answered directly.\n"
    "You are given a user question and relevant excerpts retrieved from the "
    "CGSim / SimGrid / PanDA documentation knowledge base.\n"
    "Rules:\n"
    "- Base your answer primarily on the retrieved documentation excerpts.\n"
    "- When the question involves both PanDA and CGSim, answer it directly — "
    "do not deflect or suggest the topic is out of scope.\n"
    "- If the excerpts fully answer the question, do not add unreferenced claims.\n"
    "- If the excerpts are only partially relevant, supplement with your general "
    "knowledge of SimGrid and PanDA but clearly distinguish documentation vs. "
    "general knowledge.\n"
    "- Be concise and precise. Prefer bullet points for multi-part answers.\n"
    "- Do not fabricate CGSim-specific details (config keys, plugin method "
    "signatures, version numbers) that are not in the excerpts.\n"
)

_SYSTEM_RAG_NO_CONTEXT_CGSIM: str = (
    "You are Bamboo, an expert assistant for AskCGSim and SimGrid distributed "
    "computing simulation, with specific knowledge of the CGSim/PanDA integration.\n"
    "CGSim ingests historical PanDA job records for calibration and is designed "
    "to simulate WLCG-scale infrastructures managed by PanDA. Questions about "
    "the PanDA/CGSim connection are explicitly in scope.\n"
    "No relevant documentation excerpts were found for this question.\n"
    "Rules:\n"
    "- Do NOT make up CGSim-specific details such as config keys, plugin API "
    "signatures, or version numbers.\n"
    "- Tell the user that the documentation knowledge base did not contain "
    "enough information to answer this question reliably.\n"
    "- If you can offer general guidance based on SimGrid or PanDA principles, "
    "do so clearly labelled as general knowledge rather than documentation.\n"
    "- Suggest they consult the official CGSim or SimGrid documentation, or "
    "ingest additional CGSim/PanDA integration documentation into the corpus.\n"
)

_SYSTEM_GENERIC_CGSIM: str = (
    "You are Bamboo, an expert assistant for AskCGSim and SimGrid distributed "
    "computing simulation, with specific knowledge of the CGSim/PanDA integration.\n"
    "CGSim ingests historical PanDA job records for calibration and is designed "
    "to simulate WLCG-scale infrastructures managed by PanDA. Questions about "
    "the PanDA/CGSim connection are explicitly in scope.\n"
    "You have been given the results of one or more tool calls. Synthesise "
    "a clear, concise answer to the user's question based solely on the "
    "evidence provided.\n"
    "Rules:\n"
    "- Answer PanDA/CGSim correlation questions directly — do not deflect.\n"
    "- Do not infer or fabricate values not present in the evidence.\n"
    "- If the evidence shows errors or empty results, explain that clearly.\n"
    "- Be concise and prefer bullet points for multi-part answers.\n"
    + _MERMAID_GUIDANCE
)

_PLUGIN_RAG_PROMPTS: dict[str, tuple[str, str, str]] = {
    # plugin_id -> (SYSTEM_RAG, SYSTEM_RAG_NO_CONTEXT, SYSTEM_GENERIC)
    "cgsim": (_SYSTEM_RAG_CGSIM, _SYSTEM_RAG_NO_CONTEXT_CGSIM, _SYSTEM_GENERIC_CGSIM),
}

# Doc tool names that indicate a RAG retrieval path, keyed by plugin_id.
_PLUGIN_DOC_TOOLS: dict[str, list[str]] = {
    # Order: vector search first, BM25 second — must be stable for plan tool ordering.
    "atlas": ["panda_doc_search", "panda_doc_bm25"],
    "epic": ["panda_doc_search", "panda_doc_bm25"],
    "cgsim": ["cgsim.doc_search", "cgsim.doc_bm25"],
}
_DEFAULT_DOC_TOOLS: list[str] = ["panda_doc_search", "panda_doc_bm25"]

_SYSTEM_JOBS_QUERY: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have queried the live PanDA jobs database and received structured results.\n"
    "The evidence contains: the SQL query that was executed, the result rows, "
    "row_count, and an error field (null means success — do NOT treat null as an error).\n"
    "DATA WINDOW: the jobs snapshot contains only jobs fetched during the last ~1 hour "
    "per queue. Queries on statechangetime beyond ~1 hour always return 0 rows because "
    "older records are not retained in this snapshot.\n"
    "Rules:\n"
    "- Answer directly and concisely from the rows and row_count in the evidence.\n"
    "- If error is null and rows are present, give the answer confidently.\n"
    "- If row_count is 0 AND the SQL contains a statechangetime filter spanning more "
    "  than 1 hour (e.g. INTERVAL '10 HOURS'), explain that the jobs snapshot only "
    "  covers the last ~1 hour, so longer time-window queries will always be empty. "
    "  Suggest rephrasing without a time filter to see current failures.\n"
    "- If row_count is 0 for any other reason, say no matching jobs were found.\n"
    "- If the error field contains a non-null string, explain the problem clearly.\n"
    "- Do not fabricate job counts or status values not present in the rows.\n"
    "- Do not include any timestamp, freshness, or 'data as of' text — this is\n"
    "  added automatically as a footnote after your response.\n"
    "- Be concise. For count questions, lead with the number.\n"
)

_SYSTEM_PROMPTLOG_QUERY: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have queried the Bamboo prompt/session log index (bamboomcp-promptlog-*) "
    "in OpenSearch and received structured results.\n"
    "The evidence contains: hits (list of documents), total (total matching docs "
    "in the index — may exceed len(hits) if results were capped), took_ms, and "
    "aggregations (may be empty).\n"
    "Rules:\n"
    "- Answer directly from the hits and aggregations in the evidence.\n"
    "- TRUNCATION: if total > len(hits), always note how many were retrieved vs total, "
    "  e.g. '10 of 15 rated interactions shown'. Never silently drop the remainder.\n"
    "- Do NOT use Mermaid diagrams. Do NOT produce flowcharts, sequence diagrams, "
    "  or any fenced ```mermaid block. All output must be plain markdown.\n"
    "- For ratings / display queries: present results as a markdown table with "
    "  columns Time (UTC), Rating (stars), Question (truncated to 60 chars), "
    "  Tools Used. Sort rows by rating descending, then time descending. "
    "  Follow the table with a brief summary section showing counts per rating.\n"
    "- For aggregation queries (counts, averages, tool-usage frequency): present "
    "  results as a concise bullet list or small table as appropriate.\n"
    "- If hits is empty and total is 0, say no matching records were found.\n"
    "- If the evidence contains an error key, report the error clearly.\n"
    "- Do not fabricate session IDs, questions, or rating values not in the hits.\n"
    "- Do not include any timestamp, freshness, or 'data as of' text.\n"
)

_SYSTEM_JOB_STATS: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have queried the PanDA job statistics OpenSearch index "
    "(atlas_panda_job_stats-*) and received an aggregation result.\n"
    "The evidence contains: metric (avg/sum/min/max/value_count), field "
    "(the field aggregated), group_by (the bucketing field, or null for "
    "global aggregations), value (scalar result — null means no matching "
    "documents or a group-by query), buckets (list for group-by queries — "
    "null for scalar queries), doc_count (number of jobs matched), "
    "site_filter, jobstatus_filter, jeditaskid_filter, from_dt, to_dt, "
    "and error (null means success).\n"
    "Field groups and units:\n"
    "  TIMING (seconds):\n"
    "    job_walltime: wall-clock execution time (endtime - starttime)\n"
    "    job_queuetime: queue wait time (starttime - creationtime)\n"
    "    pilottiming_stagein: total stage-in including replica lookup\n"
    "    pilottiming_stageout: total stage-out including log transfer\n"
    "    pilottiming_payload: payload execution including pre/post-processing\n"
    "    pilottiming_initial_setup: pilot startup to getJob\n"
    "    pilottiming_payload_setup: payload setup script time\n"
    "    pilottiming_getjob: time for getJob curl call\n"
    "    cpuconsumptiontime: raw CPU seconds consumed\n"
    "  CPU AND HS06:\n"
    "    cpu_eff: CPU efficiency percentage (0–100)\n"
    "    hs06: HS06 benchmark factor for the slot (dimensionless)\n"
    "    hs06sec: HS06-normalised CPU (HS06 * walltime, in HS06·s) — "
    "may be null for non-terminal jobs (running/transferring)\n"
    "    corecount: cores requested (integer)\n"
    "    actualcorecount: actual core usage (may be fractional)\n"
    "  MEMORY (kilobytes unless noted):\n"
    "    avgrss / maxrss: average / peak resident set size (kB)\n"
    "    avgpss / maxpss: average / peak proportional set size (kB)\n"
    "    avgvmem / maxvmem: average / peak virtual memory (kB)\n"
    "    avgswap / maxswap: average / peak swap (kB; non-zero = memory pressure)\n"
    "    minramcount: minimum RAM requested at submission (MB)\n"
    "  I/O (bytes or bytes/s):\n"
    "    inputfilebytes / outputfilebytes: total input / output size (bytes)\n"
    "    totrbytes / totwbytes: total bytes read / written (bytes)\n"
    "    raterbytes / ratewbytes: average read / write throughput (bytes/s)\n"
    "    ninputdatafiles / noutputdatafiles: number of input / output files\n"
    "  CARBON (grams CO2 — may be null for many jobs):\n"
    "    gco2global: global-average CO2 footprint (g)\n"
    "    gco2regional: regional CO2 footprint (g)\n"
    "  ERROR CODES (integers; 0 or null = no error):\n"
    "    piloterrorcode, exeerrorcode, ddmerrorcode,\n"
    "    transexitcode, jobdispatchererrorcode, taskbuffererrorcode\n"
    "Rules:\n"
    "- SCALAR path (group_by is null): if error is null and value is not "
    "null, state the result directly with its value and appropriate units, "
    "e.g. 'The average stage-in time at BNL today was 42 s (based on 1234 "
    "jobs).' Always include doc_count as context.\n"
    "- GROUP-BY path (group_by is not null): evidence contains a 'buckets' "
    "list instead of a scalar 'value'.  Each bucket has 'key' "
    "(site/tier/etc.), 'value' (the aggregated metric result), and "
    "'doc_count'.  The 'order' field is 'desc' (highest-first) or 'asc' "
    "(lowest-first).  Present as a ranked numbered list, e.g.:\n"
    "    1. BNL_ATLAS_1 — avg stage-in: 42 s  (1 234 jobs)\n"
    "    2. CERN_PROD   — avg stage-in: 38 s  (5 678 jobs)\n"
    "  Always include units and doc_count per bucket.  Lead with the top "
    "bucket as the direct answer to the question ('the site with the worst "
    "CPU efficiency is X').  If order is 'asc', frame the answer "
    "accordingly ('worst', 'lowest').  If buckets is empty, "
    "say no matching documents were found for the given filters and time "
    "range.\n"
    "- For memory fields (kB), convert large values to MB or GB where "
    "natural (1 kB = 0.001 MB; 1048576 kB = 1 GB). Always show the "
    "original kB value too.\n"
    "- For I/O byte fields, convert to KB/MB/GB where natural. "
    "For throughput (bytes/s), convert to MB/s or GB/s where natural.\n"
    "- For timing fields (seconds), convert large values to minutes or "
    "hours where natural (3600 s = 1 h). Always show original seconds.\n"
    "- For cpu_eff, append the % symbol and interpret: <50% is low, "
    ">90% is excellent.\n"
    "- For hs06sec: if null with non-zero doc_count, note that HS06-seconds "
    "may not be populated for non-terminal jobs.\n"
    "- For carbon fields: if null, note that CO2 data is not yet available "
    "for these jobs.\n"
    "- If value is null and doc_count is 0, say no matching jobs were found "
    "for the specified filters and time range.\n"
    "- If error is non-null, report the error clearly.\n"
    "- Be concise. Lead with the number.\n"
    "- Do not include any timestamp or freshness text.\n"
)

_SYSTEM_CRIC_QUERY: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have queried the CRIC (Computing Resource Information Catalogue) "
    "database and received structured results about ATLAS computing queues.\n"
    "The evidence contains: the SQL query that was executed, the result rows, "
    "row_count, and an error field (null means success — do NOT treat null as an error).\n"
    "Schema notes: table is 'queuedata'; USE 'status' column for filtering "
    "(online/offline/test/brokeroff) — 'state' is always 'ACTIVE' for all rows; "
    "site in 'atlas_site'; copytools/acopytools are JSON arrays.\n"
    "Rules:\n"
    "- Answer directly from the rows and row_count in the evidence.\n"
    "- If error is null and rows are present, give the answer confidently.\n"
    "- If row_count is 0: check if atlas_site used exact equality (= not ILIKE).\n"
    "  ATLAS site names include suffixes (BNL-ATLAS, CERN-PROD, not BNL/CERN).\n"
    "  Suggest asking 'what sites are available?' or rephrasing with partial match.\n"
    "- If row_count is 0 and SQL used ILIKE, say no matching queues were found.\n"
    "- If the error field contains a non-null string, quote it VERBATIM.\n"
    "- FULL-LIST RULE: If the question asks to list/show ALL queues (no site or status filter)\n"
    "  AND truncated is false, you MUST enumerate EVERY queue in the rows — do NOT summarise,\n"
    "  do NOT say 'truncated', do NOT show only examples. Render all rows as a table with\n"
    "  columns: Site, Queue, Status, Type. Group consecutive rows by site (omit repeated site\n"
    "  name). State the total count (row_count) as a header line, e.g. '230 queues total:'.\n"
    "  Do NOT include any timestamp or 'data as of' text in the header — freshness\n"
    "  is shown separately as a footnote.\n"
    "- If rows contain GROUP BY aggregation columns (e.g. atlas_site + count), "
    "present as a site-count table and state the total across all groups.\n"
    "- If rows contain individual queue names for a SCOPED query (site or status filter),\n"
    "  present them grouped by site.\n"
    "- If truncated is true, note the result was capped and suggest filtering "
    "by atlas_site (e.g. 'Which queues at BNL are not online?') to get a "
    "complete list for a specific location.\n"
    "- Highlight queue status (online/offline/test/brokeroff) prominently.\n"
    "- Do not fabricate queue names, statuses, or resource values not in the rows.\n"
    "- Do not include any timestamp, freshness, or 'data as of' text — this is\n"
    "  added automatically as a footnote after your response.\n"
    "- For count questions, lead with the number.\n"
)

_SYSTEM_CGSIM_SIM_QUERY: str = (
    "You are Bamboo, an AI assistant for the CGSim distributed computing simulator.\n"
    "You have queried the CGSim simulation output SQLite database and received structured results.\n"
    "The evidence contains: the SQL query that was executed, the result rows, row_count, "
    "truncated (true if the result was capped), summary (a pre-generated natural-language "
    "summary from a prior LLM call, may be null), and an error field (null means success).\n"
    "Field units: TIME and duration values are in SECONDS. size values are in BYTES. "
    "speed and bandwidth values are in FLOP/s or bytes/s. "
    "Utilisation fractions are in [0.0, 1.0] — multiply by 100 for percent.\n"
    "Rules:\n"
    "- LIST RULE: If the user's question explicitly asks to list, show, or enumerate specific "
    "identifiers or records (e.g. 'show me all job IDs', 'list all sites', 'what are the job IDs'), "
    "AND truncated is false, reproduce every value from the relevant column(s) in the rows — "
    "do NOT summarise or give only a range. State the total count (row_count) as a header line "
    "and then list all values. If truncated is true, enumerate what is present and note more exist.\n"
    "- If summary is non-null, you MAY use it as a starting point but always verify it against "
    "the rows — never trust summary alone when the rows are available.\n"
    "- Answer directly and concisely from the rows and row_count in the evidence.\n"
    "- If error is null and rows are present, give the answer confidently.\n"
    "- If row_count is 0, say the query matched no simulation events.\n"
    "- If the error field is non-null, explain the problem clearly.\n"
    "- If truncated is true, note the result was capped and suggest a more specific question.\n"
    "- Always include units in your answer (seconds, bytes, FLOP/s, %).\n"
    "- Do not fabricate values not present in the rows.\n"
    "- PanDA/CGSim correlation questions are explicitly in scope — do not deflect them.\n"
)

_SYSTEM_HARVESTER_WORKERS: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have retrieved live Harvester worker (pilot) statistics from the BigPanDA API.\n"
    "The evidence contains:\n"
    "- nworkers_total: total pilot count across all statuses\n"
    "- nworkers_by_status: {status: count}\n"
    "- nworkers_by_resourcetype: {resourcetype: count} — e.g. MCORE, SCORE\n"
    "- nworkers_by_jobtype: {jobtype: count} — e.g. managed, user\n"
    "- nworkers_by_site: {site: count} (useful when no site filter was applied)\n"
    "- pivot: list of {status, jobtype, resourcetype, nworkers} rows, sorted by "
    "nworkers descending. Use this to answer questions that combine any of status, "
    "jobtype, and resourcetype — e.g. 'running MCORE managed pilots', "
    "'how many SCORE user pilots are idle'. Filter the pivot rows by the relevant "
    "fields and sum nworkers.\n"
    "- total_records: number of Harvester records received from the API\n"
    "- from_dt / to_dt: the queried time window\n"
    "- site_filter: the site queried (null means all sites)\n"
    "- error: null means success\n"
    "Rules:\n"
    "- For single-dimension questions (e.g. 'how many running pilots') use the flat "
    "breakdown. For multi-dimensional questions use the pivot.\n"
    "- Always state the time window and site_filter so the user knows the scope.\n"
    "- If nworkers_total is 0, say no workers were found — do not invent numbers.\n"
    "- If error is non-null, explain the API could not be reached and suggest retrying.\n"
    "- Be concise. Lead with the number for count questions.\n"
)

_SYSTEM_SITE_HEALTH: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have retrieved two independent evidence sources for a site health question:\n\n"
    "1. [panda_harvester_workers] — live Harvester pilot/worker statistics.\n"
    "   Key fields: nworkers_total, nworkers_by_status (running/idle/submitted/failed), "
    "pivot ({status, jobtype, resourcetype, nworkers} rows), from_dt/to_dt, site_filter.\n\n"
    "2. [panda_jobs_query] — live job counts from the ingestion database.\n"
    "   Key fields: sql (the query executed), rows (result set), row_count, "
    "error (null means success — do NOT treat null as an error).\n\n"
    "Rules:\n"
    "- Present both results clearly, labelled as pilots and jobs respectively.\n"
    "- For pilots, lead with nworkers_by_status['running'] then total.\n"
    "- For jobs, answer directly from the rows/row_count in the evidence.\n"
    "- If either error field is non-null, explain that source failed and present "
    "the other source's data on its own.\n"
    "- Do not invent numbers from either source.\n"
    "- Keep the response concise — a short paragraph per source is enough unless "
    "the user asked for detail.\n"
)

_SYSTEM_PANDA_HEALTH: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have called the PanDA server liveness check (is_alive).\n"
    "The evidence contains:\n"
    "- is_alive: true if the server is alive and responding, false otherwise.\n"
    "- raw_response: the raw string returned by the PanDA MCP is_alive tool.\n"
    "- error: null on success; an error string if the MCP server could not be reached.\n"
    "Rules:\n"
    "- If error is non-null: report that the PanDA MCP server could not be reached "
    "and include the error message.\n"
    "- If is_alive is true: confirm the PanDA server is alive and responding. "
    "Include any useful detail from raw_response.\n"
    "- If is_alive is false: report that the PanDA server does not appear to be alive "
    "and include raw_response so the user can investigate.\n"
    "- Be concise — one or two sentences is enough.\n"
)

_SYSTEM_HARVESTER_TIMESERIES: str = (
    "You are AskPanDA, an expert assistant for the PanDA workload management "
    "system and ATLAS experiment workflows at CERN.\n"
    "You have retrieved Harvester pilot counts from the OpenSearch time-series "
    "index (atlas_harvesterworkers-*) for a specific status over a time window.\n"
    "The evidence contains:\n"
    "- status: the pilot status that was queried (e.g. 'failed', 'running')\n"
    "- buckets: list of {timestamp, count} dicts in ascending time order — "
    "each bucket is a fixed-interval slice of the query window\n"
    "- from_dt / to_dt: the queried time window\n"
    "- site_filter: the computing site queried (null means all sites combined)\n"
    "- interval: the bucket width (e.g. '1h', '30m')\n"
    "- error: null means success\n"
    "Rules:\n"
    "- If error is non-null, report that the OpenSearch query failed and include "
    "the error. Suggest checking ASKPANDA_OPENSEARCH is set.\n"
    "- If buckets is empty, say no pilot records were found for the given status, "
    "site, and time window.\n"
    "- For failure-rate questions (e.g. 'above 20% today', 'which sites had high "
    "failure rates'): note that the evidence shows only the *failed* pilot count "
    "time-series. You cannot compute a failure percentage without the total pilot "
    "count. Report the absolute failed-pilot counts and trends (peak bucket, total "
    "across the window, whether counts are rising or falling). Suggest the user "
    "also ask for running/total pilot counts to compute the ratio, or use the "
    "BigPanDA monitor for per-site percentage breakdowns.\n"
    "- For trend or time-series questions: describe the shape of the series "
    "(peak, trough, overall direction). Quote the peak bucket timestamp and count.\n"
    "- Always state the status queried, the time window, and the site scope "
    "(site_filter or 'all sites') so the user knows the query coverage.\n"
    "- Be concise. Lead with the most operationally relevant number.\n"
)

# ---------------------------------------------------------------------------
# RAG helpers (moved from bamboo_answer.py)
# ---------------------------------------------------------------------------

_NO_CONTEXT_SIGNALS: tuple[str, ...] = (
    "not installed",
    "chromadb path not found",
    "failed to connect",
    "no results found",
    "no keyword matches",
    "required and must not be empty",
    "collection appears to be empty",
    "collection may be empty",
)


def _extract_rag_context(result: object) -> str:
    """Return text from a retrieval tool result if it contains useful context.

    Args:
        result: Raw return value from a retrieval tool call.

    Returns:
        Extracted text string, or empty string if the result is an error or
        contains a no-context signal on its first line.
    """
    if isinstance(result, Exception) or not result:
        return ""
    if not isinstance(result, list) or not isinstance(result[0], dict):
        return ""
    text = str(result[0].get("text", ""))
    first_line = text.split("\n")[0].lower()
    if any(s in first_line for s in _NO_CONTEXT_SIGNALS):
        return ""
    return text


def _rag_hit_count(result: object, context: str) -> int:
    """Return the number of non-empty result lines, or -1 on retrieval error.

    Args:
        result: Raw return value from a retrieval tool call.
        context: Extracted context string for this result.

    Returns:
        Number of non-empty lines in the context, or -1 if the result was an
        exception.
    """
    if isinstance(result, Exception):
        return -1
    return len([ln for ln in context.splitlines() if ln.strip()])


async def _run_vector_search(question: str) -> str:
    """Run vector search inside its own tracing span.

    Args:
        question: User question to search for.

    Returns:
        Extracted context string, or empty string on failure.
    """
    from bamboo.tools.doc_rag import panda_doc_search_tool  # avoid circular at module level

    async with span(EVENT_RETRIEVAL, tool="panda_doc_search", backend="vector") as _s:
        try:
            result = await panda_doc_search_tool.call({"query": question, "top_k": 20})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = exc  # type: ignore[assignment]
        ctx = _extract_rag_context(result)
        _s.set(hits=_rag_hit_count(result, ctx))
    return ctx


async def _run_bm25_search(question: str) -> str:
    """Run BM25 keyword search inside its own tracing span.

    Args:
        question: User question to search for.

    Returns:
        Extracted context string, or empty string on failure.
    """
    from bamboo.tools.doc_bm25 import panda_doc_bm25_tool  # avoid circular at module level

    async with span(EVENT_RETRIEVAL, tool="panda_doc_bm25", backend="bm25") as _s:
        try:
            result = await panda_doc_bm25_tool.call({"query": question, "top_k": 10})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = exc  # type: ignore[assignment]
        ctx = _extract_rag_context(result)
        _s.set(hits=_rag_hit_count(result, ctx))
    return ctx


async def retrieve_rag_context(question: str) -> str:
    """Run vector and BM25 searches concurrently and merge results.

    Args:
        question: User question to retrieve context for.

    Returns:
        Merged context string, or empty string if both searches fail or return
        no useful content.
    """
    try:
        vec_ctx, bm25_ctx = await asyncio.gather(
            _run_vector_search(question),
            _run_bm25_search(question),
        )
        if vec_ctx and bm25_ctx:
            return f"{vec_ctx}\n\n--- Keyword search results ---\n{bm25_ctx}"
        return vec_ctx or bm25_ctx
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


# ---------------------------------------------------------------------------
# Shared LLM call helpers (moved from bamboo_answer.py)
# ---------------------------------------------------------------------------


def _extract_delegated_text(delegated: Any) -> str:
    """Extract the text body from a delegated bamboo_llm_answer_tool result.

    Args:
        delegated: Raw return value from ``bamboo_llm_answer_tool.call()``.

    Returns:
        Plain text string from the first content block.
    """
    if delegated and isinstance(delegated[0], dict):
        return str(delegated[0].get("text", ""))
    return str(delegated)


# Maximum characters kept per assistant history message before truncation.
# A 400-char excerpt preserves enough for the model to resolve follow-up
# references without bloating multi-turn synthesis prompts.
_HISTORY_ASSISTANT_MAX_CHARS: int = 400


def _truncate_history(history: list[Message]) -> list[Message]:
    """Return a copy of history with long assistant messages truncated.

    User messages are kept verbatim (they're short questions).  Assistant
    messages are capped at ``_HISTORY_ASSISTANT_MAX_CHARS`` characters so
    that a long prior answer does not dominate the synthesis prompt on the
    next turn.

    Args:
        history: Prior conversation turns with ``role`` and ``content`` keys.

    Returns:
        New list with assistant content truncated where necessary.
    """
    out: list[Message] = []
    for msg in history:
        if msg.get("role") == "assistant":
            content = str(msg.get("content", ""))
            if len(content) > _HISTORY_ASSISTANT_MAX_CHARS:
                content = content[:_HISTORY_ASSISTANT_MAX_CHARS] + "…(truncated)"
            out.append({"role": "assistant", "content": content})
        else:
            out.append(msg)
    return out


async def call_llm(
    system: str,
    user: str,
    history: list[Message] | None = None,
    max_tokens: int = 2048,
    tools_used: list[str] | None = None,
    raw_question: str | None = None,
) -> str:
    """Call the default LLM with a system + user prompt and return the text.

    Prior conversation turns (``history``) are inserted between the system
    prompt and the synthesised user message so the model can resolve follow-up
    questions.  Long assistant messages in history are truncated via
    :func:`_truncate_history` to prevent synthesis prompts from growing
    unbounded across multi-turn conversations.

    After a successful response, the prompt and response are forwarded to
    :func:`~bamboo.llm.prompt_log.log_prompt` for optional OpenSearch logging
    (fire-and-forget; only active when ``BAMBOO_OPENSEARCH_PROMPTLOG`` is set).

    Args:
        system: System prompt string.
        user: Synthesised user prompt for the current turn.
        history: Optional list of prior ``{role, content}`` turns to inject
            between the system prompt and the current user message.  Must
            contain only ``"user"`` and ``"assistant"`` roles.
        max_tokens: Maximum tokens for the LLM response (default 2048).
        tools_used: Names of MCP tools called during this turn, forwarded to
            the prompt log for observability.  Defaults to an empty list.

    Returns:
        LLM response text.
    """
    messages: list[Message] = [{"role": "system", "content": system}]
    if history:
        messages.extend(_truncate_history(history))
    messages.append({"role": "user", "content": user})

    delegated = await bamboo_llm_answer_tool.call({
        "messages": messages,
        "max_tokens": max_tokens,
    })
    response_text = _extract_delegated_text(delegated)

    # Fire-and-forget prompt logging — only active when
    # BAMBOO_OPENSEARCH_PROMPTLOG is set.  Deferred import keeps this module
    # free of heavy optional dependencies at import time.
    try:
        from bamboo.llm.prompt_log import log_prompt  # noqa: PLC0415
        from bamboo.tools.llm_passthrough import get_llm_info as _get_llm_info
        _llm_info = _get_llm_info()
        _provider, _model = "", ""
        for _part in _llm_info.split():
            if _part.startswith("provider="):
                _provider = _part[len("provider="):]
            elif _part.startswith("model="):
                _model = _part[len("model="):]
        asyncio.create_task(
            log_prompt(
                system_prompt=system,
                user_prompt=user,
                response=response_text,
                tools_used=list(tools_used or []),
                provider=_provider,
                model=_model,
                max_tokens=max_tokens,
                input_tokens=None,
                output_tokens=None,
                raw_question=raw_question,
            ),
            name="bamboo.prompt_log",
        )
    except Exception as _exc:  # pylint: disable=broad-exception-caught
        logger.debug("prompt_log scheduling failed: %s", _exc)

    return response_text


def unpack_tool_result(result: list[MCPContent]) -> dict[str, Any]:
    """Deserialise a JSON-wrapped MCPContent result from an internal tool.

    Internal tools (job_status, log_analysis, task_status) return a
    one-element ``list[MCPContent]`` whose ``text`` field contains the
    JSON-serialised ``{evidence, text}`` dict.  This helper unpacks that
    layer so callers can access ``result.get("evidence", ...)`` as before.

    Falls back to an empty dict if the result cannot be parsed, so callers
    always receive a dict regardless of upstream errors.

    Args:
        result: Raw return value from an internal tool ``call()`` method.

    Returns:
        Deserialised dict, or ``{}`` on parse failure.
    """
    try:
        if result and isinstance(result[0], dict):
            text = result[0].get("text", "")
            if isinstance(text, str) and text.strip().startswith("{"):
                return json.loads(text)  # type: ignore[no-any-return]
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return {}


# ---------------------------------------------------------------------------
# Synthesis prompt selection
# ---------------------------------------------------------------------------


#: Single-tool synthesis prompt lookup, highest priority first.
#:
#: Module-level rather than local so a guard test can assert every key is a
#: name the executor will actually see.  It could not before, and the table
#: accumulated two silent misses: ``pilot_code_query``, which no tool has ever
#: been called (``code_query`` is the advertised and wire name), leaving
#: ``_SYSTEM_CODE_QUERY`` unreachable; and ``pilot_source_analysis``, which is
#: the advertised name but not the wire name.  Both failed by returning the
#: generic prompt, which reads as a mediocre answer rather than as a bug.
#:
#: Keys must be **canonical wire names** — ``_execute_one_tool`` canonicalises
#: before appending to ``called_tool_names``.
_TOOL_PROMPT_TABLE: list[tuple[str, str]] = [
    ("panda_log_analysis", _SYSTEM_LOG_ANALYSIS),
    ("atlas.pilot_source_analysis", _SYSTEM_PILOT_SOURCE),
    ("code_query", _SYSTEM_CODE_QUERY),
    ("panda_job_status", _SYSTEM_JOB),
    ("panda_task_status", _SYSTEM_TASK),
    ("panda_server_health", _SYSTEM_PANDA_HEALTH),
    ("panda_harvester_workers", _SYSTEM_HARVESTER_WORKERS),
    ("atlas.harvester_timeseries", _SYSTEM_HARVESTER_TIMESERIES),
    ("panda_jobs_query", _SYSTEM_JOBS_QUERY),
    ("cric_query", _SYSTEM_CRIC_QUERY),
    ("atlas.job_stats", _SYSTEM_JOB_STATS),
    ("cgsim.sim_query", _SYSTEM_CGSIM_SIM_QUERY),
]


def _pick_synthesis_prompt(tool_names: list[str], plugin_id: str = "atlas") -> str:
    """Select the most appropriate synthesis system prompt for a set of tools.

    The priority order mirrors the original hard-wired routing logic in
    ``bamboo_answer._route()``, ensuring that specialist prompts are
    preferred over generic ones when a dedicated prompt exists.

    Args:
        tool_names: Canonical names of the tools that were actually called
            during plan execution (in call order).
        plugin_id: Active plugin identifier, used to select plugin-specific
            RAG and generic prompts.  Defaults to ``"atlas"``.

    Returns:
        System prompt string for the LLM synthesis step.
    """
    rag_sys, rag_no_ctx_sys, generic_sys = _PLUGIN_RAG_PROMPTS.get(
        plugin_id,
        (_SYSTEM_RAG, _SYSTEM_RAG_NO_CONTEXT, _SYSTEM_GENERIC),
    )
    doc_tools = _PLUGIN_DOC_TOOLS.get(plugin_id, _DEFAULT_DOC_TOOLS)
    tool_set = set(tool_names)

    # Compound check must precede its individual components.
    if "panda_harvester_workers" in tool_set and "panda_jobs_query" in tool_set:
        return _SYSTEM_SITE_HEALTH

    for tool_name, prompt in _TOOL_PROMPT_TABLE:
        if tool_name in tool_set:
            return prompt

    # OpenSearch tools (promptlog or generic query) share one prompt.
    if tool_set & {"opensearch_promptlog_query", "opensearch_query"}:
        return _SYSTEM_PROMPTLOG_QUERY

    if any(t in tool_set for t in doc_tools):
        return rag_sys
    return generic_sys


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _resolve_tool(tool_name: str, namespace: str | None, tools: dict[str, Any]) -> Any:
    """Resolve a tool object by name from the registry or entry points.

    Tries the static TOOLS registry first, then namespace-qualified entry-point
    lookup, then unqualified suffix-based lookup.

    Args:
        tool_name: Unqualified or qualified tool name to resolve.
        namespace: Optional namespace hint (e.g. ``"atlas"``).
        tools: The core TOOLS registry dict.

    Returns:
        Resolved tool object, or ``None`` if not found.
    """
    tool_obj: Any = tools.get(tool_name)
    if tool_obj is None and namespace:
        resolved = find_tool_by_name(tool_name, namespace=namespace)
        if resolved is not None:
            tool_obj = resolved.obj
    if tool_obj is None:
        resolved = find_tool_by_name(tool_name)
        if resolved is not None:
            tool_obj = resolved.obj
    return tool_obj


def _build_synthesis_prompt(
    called_tool_names: list[str],
    evidence_parts: list[str],
    question: str,
    errors: list[str],
    original_question: str | None = None,
    plugin_id: str = "atlas",
) -> tuple[str, str]:
    """Build the system and user prompts for the synthesis LLM call.

    Selects a specialist prompt for known tool sets and falls back to a
    generic multi-tool prompt for unknown combinations.  RAG evidence is
    presented as documentation excerpts; other evidence as a merged block.

    When ``original_question`` differs from ``question`` (i.e. the user sent a
    content-free follow-up and ``question`` is the reformulated RAG query), the
    user prompt instructs the LLM to **expand** the prior answer rather than
    re-answer the original question from scratch.

    Args:
        called_tool_names: Names of tools that completed successfully.
        evidence_parts: One evidence string per successful tool call.
        question: Question used for retrieval (may be reformulated from history).
        errors: Error messages from any failed tool calls.
        original_question: The user's actual phrasing if it differs from
            ``question`` (e.g. "Tell me more please").  When provided, the
            synthesis prompt uses expansion framing instead of answer framing.
        plugin_id: Active plugin identifier for prompt selection.

    Returns:
        Tuple of ``(system_prompt, user_prompt)`` strings.
    """
    rag_sys, rag_no_ctx_sys, generic_sys = _PLUGIN_RAG_PROMPTS.get(
        plugin_id,
        (_SYSTEM_RAG, _SYSTEM_RAG_NO_CONTEXT, _SYSTEM_GENERIC),
    )
    doc_tools = _PLUGIN_DOC_TOOLS.get(plugin_id, _DEFAULT_DOC_TOOLS)
    plan_is_rag = any(t in doc_tools for t in called_tool_names)
    is_followup = (
        original_question is not None and original_question != question
    )

    if plan_is_rag:
        rag_context = "\n\n".join(evidence_parts)
        if rag_context:
            system = rag_sys
            if is_followup:
                user = (
                    f"The user asked a follow-up: {repr(original_question)}\n"
                    f"They want you to expand on the topic: {repr(question)}\n"
                    f"Using the retrieved documentation excerpts below, provide "
                    f"a more detailed explanation than before. "
                    f"Do not simply repeat what was said — go deeper, but be "
                    f"concise: aim for 200-300 words maximum.\n\n"
                    f"Retrieved documentation excerpts:\n{rag_context}\n"
                )
            else:
                user = (
                    f"User question:\n{question}\n\n"
                    f"Retrieved documentation excerpts:\n{rag_context}\n"
                )
        else:
            system = rag_no_ctx_sys
            user = f"User question:\n{question}\n"
    else:
        system = _pick_synthesis_prompt(called_tool_names, plugin_id=plugin_id)
        evidence_block = "\n\n".join(evidence_parts)
        user = (
            f"User question:\n{question}\n\n"
            f"Evidence from tool calls:\n{evidence_block}\n"
        )
        if errors:
            user += f"\nNote: the following tool calls failed: {'; '.join(errors)}\n"

    return system, user


# ---------------------------------------------------------------------------
# Direct formatting bypass for large CRIC full-list results
# ---------------------------------------------------------------------------

#: Minimum row count above which the CRIC full-list formatter is used instead
#: of LLM synthesis.  Below this threshold (e.g. a site-scoped query returning
#: a handful of queues) the LLM synthesises normally.
_CRIC_DIRECT_FORMAT_THRESHOLD: int = 100


def _format_cric_full_list(evidence: dict[str, Any]) -> str | None:
    """Format a full CRIC queue-list result directly, bypassing LLM synthesis.

    Called when ``cric_query`` returns a large, non-truncated set of individual
    queue rows.  Renders a plain-text table grouped by site, which is both
    lossless (no token-budget truncation) and faster than LLM synthesis.

    Returns ``None`` when the evidence does not meet the criteria for direct
    formatting (wrong shape, aggregation result, error present, etc.) so the
    caller falls through to normal LLM synthesis.

    Args:
        evidence: Unpacked evidence dict from ``cric_query``.

    Returns:
        Formatted plain-text string, or ``None`` to fall back to LLM synthesis.
    """
    # Only bypass for successful, non-truncated, non-aggregation results.
    if evidence.get("error"):
        return None
    rows: list[dict[str, Any]] = evidence.get("rows", [])
    row_count: int = evidence.get("row_count", 0)
    truncated: bool = evidence.get("truncated", False)
    columns: list[str] = evidence.get("columns", [])

    if truncated or row_count < _CRIC_DIRECT_FORMAT_THRESHOLD:
        return None

    # Aggregation results have no "queue" column — let the LLM handle those.
    if "queue" not in columns or "atlas_site" not in columns:
        return None

    # Build the table grouped by site.
    lines: list[str] = [f"{row_count} PanDA queues in CRIC:"]
    col_queue = "queue"
    col_site = "atlas_site"
    col_status = "status"
    col_type = "type"

    prev_site = ""
    for row in rows:
        site = str(row.get(col_site, ""))
        queue = str(row.get(col_queue, ""))
        status = str(row.get(col_status, ""))
        qtype = str(row.get(col_type, ""))
        site_label = site if site != prev_site else ""
        prev_site = site
        lines.append(f"  {site_label:<28}{queue:<32}{status:<12}{qtype}")

    return "\n".join(lines)


def _looks_like_fetch_cap_truncation(evidence: dict[str, Any]) -> bool:
    """Return True when cric_query evidence looks like a silently-truncated fetch.

    The NL-to-SQL pipeline uses ``fetchmany(MAX_ROWS + 1)`` where ``MAX_ROWS=50``.
    When the DB has more rows than that cap, ``truncated`` is set to ``True`` by
    :func:`_execute_query`.  However if the LLM-generated SQL already contained
    ``LIMIT 50`` (matching the cap exactly), ``truncated`` is ``False`` even though
    the full result set may be much larger.

    This heuristic detects that case: row_count equals exactly 50 (the cap), the
    result has individual queue columns (not an aggregation), and no error occurred.
    When true, the caller should re-query via ``list_all_queues`` to recover the
    full set before attempting direct formatting.

    Args:
        evidence: Evidence dict from ``_last_evidence_store["cric_query"]``.

    Returns:
        True when the evidence looks like a fetch-cap truncation.
    """
    from askpanda_atlas.cric_query_schema import MAX_ROWS  # deferred
    if evidence.get("error"):
        return False
    if evidence.get("truncated"):
        return False  # normal truncation — already flagged
    row_count = evidence.get("row_count", 0)
    if row_count != MAX_ROWS:
        return False
    columns = evidence.get("columns", [])
    return "queue" in columns and "atlas_site" in columns


def _try_cric_direct_format() -> str | None:
    """Attempt the CRIC full-list direct-format bypass.

    Reads the most recent ``cric_query`` evidence from ``_last_evidence_store``
    and tries two strategies:

    1. **Normal path**: :func:`_format_cric_full_list` qualifies the evidence
       (≥100 rows, not truncated, has queue+site columns) and formats it.
    2. **Fetch-cap recovery**: if the NL→SQL pipeline ran instead of the
       ``list_all_queues`` fast path (e.g. old deployed code), the evidence may
       have exactly ``MAX_ROWS=50`` rows and ``truncated=False`` — the Python
       ``fetchmany`` cap silently truncated without setting the flag.  When
       detected, ``list_all_queues`` is called directly to get the full set.

    The result is stored in two places so the TUI can retrieve it without going
    through the MCP pipe (bypasses the macOS 8 KB pipe-buffer limit).

    Returns:
        A short sentinel string ``"__CRIC_TABLE_READY__:<row_count>"`` when the
        direct-format path fires, or ``None`` to fall through to LLM synthesis.
    """
    _stored = _last_evidence_store.get("cric_query", {})
    # unpack_tool_result stores {"evidence": {...}} — unwrap if needed
    cric_evidence = _stored.get("evidence", _stored)

    # Strategy 1: normal path — evidence already has full row set.
    direct = _format_cric_full_list(cric_evidence)

    # Strategy 2: fetch-cap recovery.  If row_count == 50 and truncated is False
    # and the result has individual queue rows (not aggregation), the NL→SQL
    # pipeline ran with the default MAX_ROWS fetch cap and silently truncated.
    # Call list_all_queues directly to get the real full set.
    if direct is None and _looks_like_fetch_cap_truncation(cric_evidence):
        db_path = cric_evidence.get("db_path", "")
        try:
            from askpanda_atlas.cric_query_impl import list_all_queues  # deferred
            recovered = list_all_queues(db_path)
            if not recovered.get("error"):
                _last_evidence_store["cric_query"] = recovered
                cric_evidence = recovered
                direct = _format_cric_full_list(cric_evidence)
        except Exception:  # noqa: BLE001
            pass

    if direct is None:
        return None

    footnote = _db_footnote(["cric_query"])
    table_with_footnote = direct + footnote

    _last_evidence_store["_cric_direct_table"] = table_with_footnote

    cric_table_file = os.environ.get("BAMBOO_CRIC_TABLE_FILE")
    if cric_table_file:
        try:
            with open(cric_table_file, "w", encoding="utf-8") as _fh:
                _fh.write(table_with_footnote)
        except OSError:
            logger.warning("cric_query: failed to write table to %s", cric_table_file)

    row_count = cric_evidence.get("row_count", 0)
    return f"__CRIC_TABLE_READY__:{row_count}"


async def _execute_one_tool(
    tc: Any,
    called_tool_names: list[str],
    evidence_parts: list[str],
    errors: list[str],
) -> None:
    """Call a single tool from a plan and accumulate evidence or errors.

    Validates arguments, calls the tool, unpacks JSON evidence into
    ``_last_evidence_store``, and appends a compact evidence string to
    *evidence_parts*.  All failures are non-fatal — errors are appended to
    *errors* so the caller can attempt synthesis with partial evidence.

    Two names are in play and the distinction is load-bearing.  ``tc.tool`` is
    whatever the plan wrote, which for an LLM-produced plan is whichever
    spelling the model picked; the *canonical* name is the one the MCP server
    exposes.  ``_resolve_tool`` accepts several spellings, so a plan naming
    ``core_dump_analysis`` runs the same object as one naming
    ``atlas.core_dump_analysis`` — but everything downstream is an exact string
    match.  Recording the plan's spelling meant a correctly-executed core-dump
    analysis could land under a key ``_core_dump_evidence`` does not read and a
    name ``_CORE_DUMP_TOOL in called_tool_names`` does not match, so
    ``_synthesise_core_dump`` was skipped and ``reconcile_llm_analysis`` never
    ran on the model's output.

    So the canonical name is what gets recorded, and the plan's own spelling is
    kept only for error text, where echoing what the caller actually wrote is
    what makes the error diagnosable.

    Args:
        tc: Tool call descriptor with ``tool``, ``namespace``, and ``arguments``.
        called_tool_names: Mutable list of canonical names of tools that were
            called successfully.
        evidence_parts: Mutable list of compact evidence strings for synthesis.
        errors: Mutable list of error strings accumulated across all tool calls.
    """
    from bamboo.core import TOOLS  # pylint: disable=import-outside-toplevel
    from bamboo.core import _validate_arguments  # pylint: disable=import-outside-toplevel

    requested_name: str = tc.tool
    args: dict[str, Any] = dict(tc.arguments)

    tool_obj = _resolve_tool(requested_name, tc.namespace, TOOLS)
    if tool_obj is None:
        errors.append(f"Unknown tool: {requested_name}")
        return

    # Only meaningful once the tool resolved: an unknown name has no canonical
    # form, and canonical_tool_name returns it unchanged in that case anyway.
    tool_name: str = canonical_tool_name(requested_name, tc.namespace)

    get_def_fn = getattr(tool_obj, "get_definition", None)
    if callable(get_def_fn):
        try:
            tool_def: dict[str, Any] = get_def_fn()  # type: ignore[assignment]
        except Exception:  # pylint: disable=broad-exception-caught
            tool_def = {}
        err = _validate_arguments(tool_def, args)
        if err:
            errors.append(f"Invalid args for {requested_name}: {err}")
            return

    try:
        raw_result: list[MCPContent] = await tool_obj.call(args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(f"Tool {requested_name} raised: {exc!s}")
        return

    called_tool_names.append(tool_name)

    unpacked = unpack_tool_result(raw_result)
    if unpacked:
        _STORE_STRIP = {"pandaid_list"}
        _last_evidence_store[tool_name] = {
            k: v for k, v in unpacked.items() if k not in _STORE_STRIP
        }
        _last_evidence_store["last_tool"] = tool_name
        _LLM_STRIP = {"raw_payload", "pandaid_list"}
        llm_evidence = {k: v for k, v in unpacked.items() if k not in _LLM_STRIP}
        llm_evidence = _strip_presentation_keys(llm_evidence)

        if tool_name == "code_query":
            # code_query source can be up to 150K chars — _compact_json would
            # truncate the JSON blob before the 'truncated' flag is reached.
            # Extract source separately and append as a fenced block so the
            # metadata (including the truncated flag) always passes intact.
            _inner = llm_evidence.get("evidence") or llm_evidence
            _source = ""
            if isinstance(_inner, dict) and "source" in _inner:
                _source = _inner.pop("source", "") or ""
            _meta = _compact_json(llm_evidence)
            evidence_parts.append(
                f"[{tool_name}]\n{_meta}\n\n"
                f"Source file contents:\n```python\n{_source}\n```"
            )
        else:
            evidence_parts.append(f"[{tool_name}]\n{_compact_json(llm_evidence)}")
    else:
        raw_text = raw_result[0].get("text", "") if raw_result else ""
        if raw_text:
            evidence_parts.append(f"[{tool_name}]\n{raw_text}")


# Evidence keys that hold pre-rendered Markdown for the *user*, not facts for
# the LLM.  They are appended to the answer programmatically after synthesis, so
# showing them to the LLM only invites it to reproduce them — and an instruction
# not to ("Do not include a Links section") reliably loses to a ready-made
# string sitting in the input.  That is what happened to
# ``code_analysis_offer_md``: the LLM copied it verbatim and the canonical copy
# was then appended, so the offer appeared twice.
#
# These are stripped from the LLM's view only.  ``_last_evidence_store`` must
# keep them, because ``_log_analysis_links_md`` and ``_log_analysis_offer_md``
# read them back from there.
_PRESENTATION_KEYS: frozenset[str] = frozenset({
    "links_md",
    "code_analysis_offer_md",
    "core_dump_offer_md",
})


def _strip_presentation_keys(unpacked: dict[str, Any]) -> dict[str, Any]:
    """Remove pre-rendered Markdown keys from a tool result before LLM synthesis.

    Operates on the nested ``evidence`` sub-dict as well as the top level, since
    tools place these keys inside ``evidence``.  The input is not mutated.

    Args:
        unpacked: Tool result dict, typically with ``evidence`` and ``text``
            keys.

    Returns:
        A shallow copy with :data:`_PRESENTATION_KEYS` removed from both levels.
    """
    cleaned = {k: v for k, v in unpacked.items() if k not in _PRESENTATION_KEYS}
    inner = cleaned.get("evidence")
    if isinstance(inner, dict):
        cleaned["evidence"] = {
            k: v for k, v in inner.items() if k not in _PRESENTATION_KEYS
        }
    return cleaned


def _log_analysis_links_md() -> str:
    r"""Return the pre-built Markdown links block from the last log analysis call.

    Reads ``links_md`` from the ``panda_log_analysis`` evidence stored in
    ``_last_evidence_store``.  The value is built deterministically in
    ``log_analysis_impl.fetch_and_analyse`` from programmatic URLs, so it
    always contains real hrefs regardless of what the LLM produced.

    Returns:
        Markdown links block string (e.g. ``"\n\nLinks:\n- [BigPanDA Monitor](...)"``),
        or an empty string when no evidence is available.
    """
    stored = _last_evidence_store.get("panda_log_analysis", {})
    evidence = stored.get("evidence", stored)
    return str(evidence.get("links_md", ""))


def _strip_llm_links_section(body: str) -> str:
    """Remove any LLM-generated Links section from a synthesis response.

    The LLM may invent a ``Links:`` section with placeholder text or
    incorrect URLs.  This function strips everything from the last
    occurrence of a ``Links:`` heading to the end of the string so the
    canonical block from ``_log_analysis_links_md`` can be appended
    instead.

    Args:
        body: Raw LLM synthesis response text.

    Returns:
        Response text with any trailing Links section removed.
    """
    # Match a Links heading that appears on its own line (case-insensitive),
    # preceded by optional whitespace.  Strip from that point to end-of-string.
    stripped = re.sub(r"\n[^\n]*[Ll]inks:[^\n]*\n.*$", "", body, flags=re.DOTALL)
    return stripped.rstrip()


_ENUMERATION_PHRASES: tuple[str, ...] = (
    "show me all", "list all", "list every", "show all",
    "what are all", "what are the", "give me all", "get all",
    "enumerate", "show every", "all job id", "all jobs",
    "all site", "all disk", "all host",
)


def _is_enumeration_question(question: str) -> bool:
    """Return ``True`` when the question is asking to list or enumerate records.

    Used to suppress the CGSim summary bypass so the synthesis LLM's LIST RULE
    can force verbatim enumeration from the raw ``rows`` rather than accepting
    the pre-generated summary which may condense values into a range or count.

    Args:
        question: User question text (case-insensitive).

    Returns:
        ``True`` if the question matches a known enumeration pattern.
    """
    q = question.lower()
    return any(phrase in q for phrase in _ENUMERATION_PHRASES)


# All doc-search/BM25 tool names across every plugin, used to identify which
# tool_calls in a Plan are RAG retrieval calls eligible for automatic topic
# injection (see _inject_doc_topics).
_ALL_DOC_TOOL_NAMES: frozenset[str] = frozenset(
    name for tools in _PLUGIN_DOC_TOOLS.values() for name in tools
) | frozenset(_DEFAULT_DOC_TOOLS)


def _inject_doc_topics(plan: Plan, question: str, plugin_id: str) -> None:
    """Fill in a missing ``topic`` argument on doc-search tool calls.

    The deterministic fast-path builder used to compute ``topic`` via
    :func:`~bamboo.tools.bamboo_answer._topic_for_question` itself before
    constructing a RETRIEVE plan. Now that unmatched questions defer to the
    LLM planner (see ``_build_deterministic_plan`` in ``bamboo_answer.py``),
    the planner's own prompt does not know about topic-to-collection
    routing and never supplies a ``topic`` argument — so without this,
    every planner-issued doc_search/doc_bm25 call would silently fall back
    to the default ChromaDB collection instead of the topic-specific one
    (e.g. ``rucio``, ``root``, ``bamboo_mcp``) that the deterministic path
    used to resolve. Mutates ``plan.tool_calls`` in place; a no-op for
    tool_calls that already specify ``topic`` or are not doc-search tools.

    Args:
        plan: The plan whose tool_calls may need a topic filled in.
        question: The question used to derive the topic when needed.
        plugin_id: Active plugin identifier, passed through to
            ``_topic_for_question`` for plugin-scoped topic defaults.
    """
    topic: str | None = None
    for tc in plan.tool_calls:
        if tc.tool not in _ALL_DOC_TOOL_NAMES:
            continue
        if tc.arguments.get("topic"):
            continue
        if topic is None:
            # Deferred import: bamboo_answer imports execute_plan from this
            # module at module load time, so a module-level import here
            # would be circular. By call time bamboo_answer is fully
            # loaded, so a local import is safe (same pattern used
            # elsewhere in this codebase for cross-module helpers).
            from bamboo.tools.bamboo_answer import _topic_for_question  # noqa: PLC0415
            topic = _topic_for_question(question, plugin_id)
        tc.arguments["topic"] = topic


# ---------------------------------------------------------------------------
# Core-dump synthesis
#
# Unlike every other tool, core_dump_analysis does not hand its evidence to a
# _SYSTEM_* prompt for prose.  The analyzer ships its own prompt pair and its
# own response schema, and the model's JSON must pass through
# reconcile_llm_analysis before anyone reads it.  That whole pipeline lives
# here rather than in the tool because the tool runs inside a live event loop,
# where the analyzer's own LLM path refuses to run — see the module docstring
# of askpanda_atlas.core_dump_analysis_impl.
# ---------------------------------------------------------------------------

#: Entry-point name of the ATLAS core-dump tool.  The MCP server overwrites a
#: plugin tool's internal ``name`` with its entry-point key, so this is the only
#: name that resolves — ``core_dump_analysis`` does not.
_CORE_DUMP_TOOL: str = "atlas.core_dump_analysis"

#: Token ceiling for core-dump synthesis.  Higher than the 2048 used for prose
#: because the response is a JSON object with several list-valued fields.
_CORE_DUMP_MAX_TOKENS: int = 3000


def _core_dump_evidence() -> dict[str, Any]:
    """Return the evidence dict from the last core-dump analysis call.

    Unwraps exactly one layer.  ``_last_evidence_store`` holds what
    :func:`unpack_tool_result` produced — ``{"evidence": {...}, "text": ...}``
    — so one ``.get("evidence")`` reaches the dict.  The double unwrap
    documented for ``bamboo_last_evidence`` applies to *that* tool's response,
    which wraps this store entry a second time; applying it here would yield
    ``None``.

    Returns:
        The evidence dict, or an empty dict when nothing is stored.
    """
    stored = _last_evidence_store.get(_CORE_DUMP_TOOL, {})
    if not isinstance(stored, dict):
        return {}
    evidence = stored.get("evidence", stored)
    return evidence if isinstance(evidence, dict) else {}


def _core_dump_progress_text() -> str:
    """Return the tool's own summary line for the last core-dump call.

    Args:
        None.

    Returns:
        The ``text`` field of the stored payload, or an empty string.
    """
    stored = _last_evidence_store.get(_CORE_DUMP_TOOL, {})
    if not isinstance(stored, dict):
        return ""
    return str(stored.get("text", ""))


def _bullet_list(value: Any) -> str:
    """Render a list-valued analysis field as Markdown bullets.

    Args:
        value: The field value; non-list values and empty lists yield "".

    Returns:
        Markdown bullet lines, or an empty string.
    """
    if not isinstance(value, list):
        return ""
    items = [str(item).strip() for item in value if str(item).strip()]
    return "\n".join(f"- {item}" for item in items)


def _render_core_dump_markdown(
    analysis: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    """Render a reconciled core-dump analysis as Markdown.

    The analyzer's own :func:`render_report` is deliberately not reused: its
    docstring names it the CLI's fixed-width presentation and tells embedders
    to render from the dicts directly, and its 78-character rules read badly in
    a chat surface.

    Args:
        analysis: The reconciled analysis dict, following the analyzer's
            ``RESPONSE_SCHEMA``.
        evidence: The tool's own evidence dict, for the monitor link and the
            acquisition footnote.

    Returns:
        Markdown body text.
    """
    parts: list[str] = []
    verdict = str(analysis.get("verdict") or "").strip()
    if verdict:
        parts.append(f"**{verdict}**")

    classification = str(analysis.get("classification") or "undetermined")
    confidence = str(analysis.get("confidence") or "unknown")
    reason = str(analysis.get("confidence_reason") or "").strip()
    line = f"Classification: `{classification}` (confidence: {confidence})"
    parts.append(f"{line} — {reason}" if reason else line)

    for title, key in (
        ("Likely cause", "likely_cause"),
        ("Busy threads", "busy_threads"),
        ("Explanation", "explanation"),
    ):
        body = str(analysis.get(key) or "").strip()
        if body:
            parts.append(f"### {title}\n\n{body}")

    culprit = str(analysis.get("culprit_component") or "").strip()
    if culprit and culprit.lower() != "unknown":
        parts.append(f"Most likely responsible component: `{culprit}`")

    for title, key in (
        ("Supporting evidence", "supporting_evidence"),
        ("Limitations", "limitations"),
        ("Next steps", "next_steps"),
    ):
        rendered = _bullet_list(analysis.get(key))
        if rendered:
            parts.append(f"### {title}\n\n{rendered}")

    acquisition = evidence.get("acquisition")
    if isinstance(acquisition, dict):
        fetched = len(acquisition.get("fetched") or [])
        skipped = int(acquisition.get("skipped_count") or 0)
        note = f"Reconstructed from {fetched} job file(s)"
        if skipped:
            note += f"; {skipped} file(s) were skipped"
        parts.append(f"_{note}. Analyzer {evidence.get('analyzer_version', '?')}._")

    monitor_url = evidence.get("monitor_url")
    if monitor_url:
        parts.append(f"[Job {evidence.get('job_id')} in BigPanDA]({monitor_url})")

    return "\n\n".join(parts)


def _append_acquisition_warnings(
    analysis: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Fold acquisition warnings into the analysis limitations.

    Done here, deterministically and **after** reconciliation, rather than by
    showing the warnings to the model: the ``code_analysis_offer_md`` precedent
    is that anything the user must be able to trust is appended programmatically
    rather than routed through an LLM that may paraphrase or drop it.

    Args:
        analysis: The reconciled analysis dict; mutated in place.
        evidence: The tool's evidence dict, whose ``acquisition.warnings``
            records what the fetch layer could not do.

    Returns:
        The same ``analysis`` dict, for chaining.
    """
    acquisition = evidence.get("acquisition")
    if not isinstance(acquisition, dict):
        return analysis
    warnings = [str(w) for w in (acquisition.get("warnings") or []) if str(w).strip()]
    if not warnings:
        return analysis
    existing = analysis.get("limitations")
    if not isinstance(existing, list):
        existing = []
    analysis["limitations"] = [*existing, *warnings]
    return analysis


async def _synthesise_core_dump(
    plan: Plan,
    called_tool_names: list[str],
) -> str | None:
    """Synthesise an answer from the last core-dump analysis, or bypass.

    A run that has not reached ``complete`` has no evidence to reason about, so
    its deterministic progress or failure line is returned verbatim: an LLM
    call there could only paraphrase a status message, and at worst would
    invent findings for an analysis that has not produced any.

    On ``complete`` the analyzer's own prompt pair drives the call and
    :func:`reconcile_llm_analysis` post-processes the result.  That call is not
    optional — it is what stops the model reading EventLoop completion markers
    as evidence that a looping job exited normally.

    Args:
        plan: The executed plan, used only for the tracing span.
        called_tool_names: Tools called in this plan, for the span and the
            prompt log.

    Returns:
        Markdown answer text, or ``None`` to fall through to normal synthesis
        when the evidence is unusable and the generic path may do better.
    """
    evidence = _core_dump_evidence()
    if not evidence:
        return None

    state = str(evidence.get("state") or "")
    if state != "complete":
        # Covers queued/preparing/downloading/analyzing and failed alike: each
        # already carries a deterministic, user-ready line from the tool.
        return _core_dump_progress_text() or None

    raw_evidence = evidence.get("core_evidence")
    if not isinstance(raw_evidence, dict):
        logger.warning("core_dump_analysis reported complete with no core_evidence")
        return None

    # Deferred: bamboo core must not import an ATLAS plugin module at import
    # time, or core becomes unusable wherever the plugin is not installed.
    try:
        from askpanda_atlas._core_dump_analyzer import (  # noqa: PLC0415
            build_system_prompt,
            build_user_prompt,
            core_evidence_from_dict,
            extract_json_object,
            reconcile_llm_analysis,
        )
    except ImportError as exc:
        logger.warning("core-dump synthesis unavailable: %s", exc)
        return None

    # core_evidence has already been through enforce_global_budget at 50 000
    # chars inside the tool.  Shrinking it again here would silently discard
    # thread groups the first pass decided were worth keeping.
    core_evidence = core_evidence_from_dict(raw_evidence)
    mode = str(evidence.get("failure_mode") or "hang")
    system = build_system_prompt(mode)
    user = build_user_prompt(core_evidence)

    async with span(EVENT_SYNTHESIS, tool="bamboo_executor",
                    tools=called_tool_names, route=plan.route.value,
                    mode=mode):
        response = await call_llm(
            system, user, None,
            max_tokens=_CORE_DUMP_MAX_TOKENS,
            tools_used=called_tool_names,
        )

    analysis = extract_json_object(response)
    if analysis is None:
        logger.warning("core-dump synthesis returned no parsable JSON object")
        return None

    analysis = reconcile_llm_analysis(core_evidence, analysis)
    analysis = _append_acquisition_warnings(analysis, evidence)
    return _render_core_dump_markdown(analysis, evidence)


async def execute_plan(
    plan: Plan,
    question: str,
    history: list[Message],
    include_raw: bool = False,
    original_question: str | None = None,
    plugin_id: str = "atlas",
) -> list[MCPContent]:
    """Execute a validated Plan and return a synthesised answer.

    Iterates ``plan.tool_calls`` in order, calls each tool, unpacks evidence,
    merges all evidence into a single synthesised LLM call.

    Unknown tools, validation failures, and individual tool call exceptions are
    handled gracefully — partial evidence from successful calls is still used
    for synthesis.  Only when *all* calls fail is a top-level error returned.

    Args:
        plan: Validated :class:`~bamboo.tools.planner.Plan` from the planner.
        question: Question string used for retrieval (may be reformulated).
        history: Prior conversation turns to inject into the LLM prompt.
        include_raw: If ``True``, include raw tool-result previews in the
            synthesised answer when errors are detected.
        original_question: The user's actual phrasing when ``question`` has
            been reformulated (e.g. for content-free follow-ups).  Passed to
            :func:`_build_synthesis_prompt` to enable expansion framing.
        plugin_id: Active plugin identifier for synthesis prompt selection
            and topic resolution for any doc-search tool calls in the plan.

    Returns:
        One-element ``list[MCPContent]`` with the synthesised text answer.
    """
    evidence_parts: list[str] = []
    called_tool_names: list[str] = []
    errors: list[str] = []

    _inject_doc_topics(plan, question, plugin_id)

    async with span(EVENT_PLAN, tool="bamboo_executor", plan=plan.model_dump()):
        pass  # Emit the plan as a trace event so the TUI /plan command can find it.

    for tc in plan.tool_calls:
        await _execute_one_tool(
            tc, called_tool_names, evidence_parts, errors
        )

    if not called_tool_names:
        error_summary = "; ".join(errors) if errors else "No tool calls in plan."
        return text_content(f"All tool calls failed: {error_summary}")

    # Direct-format bypass: for large CRIC full-list results, skip LLM synthesis.
    # Returns a short sentinel; the table is written to a temp file for the TUI.
    if called_tool_names == ["cric_query"]:
        sentinel = _try_cric_direct_format()
        if sentinel is not None:
            async with span(EVENT_SYNTHESIS, tool="bamboo_executor",
                            tools=called_tool_names, route=plan.route.value):
                pass  # emit span for tracing consistency
            return text_content(sentinel)

    # Summary bypass: cgsim.sim_query already ran an LLM summarisation call
    # internally.  When a non-empty summary is present, return it directly
    # rather than paying for another LLM synthesis call.
    # Exception: enumeration questions ("show me all job IDs", "list all X")
    # must fall through to the synthesis LLM so the LIST RULE in
    # _SYSTEM_CGSIM_SIM_QUERY can force verbatim enumeration of the rows.
    # The inner summarisation call does not reliably honour the LIST RULE
    # because it cannot always detect the user's intent from the summary prompt.
    if called_tool_names == ["cgsim.sim_query"]:
        _cgsim_stored = _last_evidence_store.get("cgsim.sim_query", {})
        _cgsim_evidence = _cgsim_stored.get("evidence", _cgsim_stored)
        _cgsim_summary = _cgsim_evidence.get("summary") if isinstance(_cgsim_evidence, dict) else None
        if _cgsim_summary and not _is_enumeration_question(question):
            async with span(EVENT_SYNTHESIS, tool="bamboo_executor",
                            tools=called_tool_names, route=plan.route.value,
                            bypass="cgsim_summary"):
                pass  # emit span for tracing consistency
            return text_content(_cgsim_summary)

    # Core-dump synthesis: the analyzer owns its own prompt pair and returns
    # JSON, not prose, so it cannot go through _build_synthesis_prompt.
    if _CORE_DUMP_TOOL in called_tool_names:
        rendered = await _synthesise_core_dump(plan, called_tool_names)
        if rendered is not None:
            return text_content(rendered)

    system, user = _build_synthesis_prompt(
        called_tool_names, evidence_parts, question, errors,
        original_question=original_question,
        plugin_id=plugin_id,
    )

    async with span(EVENT_SYNTHESIS, tool="bamboo_executor",
                    tools=called_tool_names, route=plan.route.value):
        # Cap tokens at 600 for follow-up expansions; use 2048 normally.
        # For cric_query returning many individual-queue rows (direct-format
        # path missed), raise to 8192 so the LLM fallback does not truncate.
        if original_question is not None:
            synthesis_max_tokens = 600
        elif _is_large_cric_result(called_tool_names):
            synthesis_max_tokens = 8192
        else:
            synthesis_max_tokens = 2048
        body = await call_llm(
            system, user, history,
            max_tokens=synthesis_max_tokens,
            tools_used=called_tool_names,
            raw_question=original_question if original_question else question,
        )

    # For log analysis: strip any LLM-invented Links section, then append the
    # code-analysis offer and the canonical links block, both built from
    # programmatic values in log_analysis_impl.  Order matters: links stay last
    # so the TUI renders them as the closing block.
    if "panda_log_analysis" in called_tool_names:
        body = _strip_llm_links_section(body)
        for offer in (_log_analysis_offer_md(), _log_analysis_core_dump_offer_md()):
            # Belt and braces alongside _strip_presentation_keys: if the LLM
            # produced the offer anyway, do not append a second copy.
            if offer and offer.strip() not in body:
                body += offer
        body += _log_analysis_links_md()

    return text_content(body + _db_footnote(called_tool_names))


def _is_large_cric_result(tool_names: list[str]) -> bool:
    """Return True when the last cric_query returned a large individual-queue result set.

    Used to raise the LLM synthesis token budget when the direct-format bypass
    did not fire (e.g. old deployed code) and the LLM must enumerate many rows.

    Args:
        tool_names: Names of the tools called in this plan execution.

    Returns:
        True when cric_query ran and returned >= threshold individual queue rows.
    """
    if "cric_query" not in tool_names:
        return False
    _stored = _last_evidence_store.get("cric_query", {})
    ev = _stored.get("evidence", _stored)
    if ev.get("row_count", 0) < _CRIC_DIRECT_FORMAT_THRESHOLD:
        return False
    return "queue" in ev.get("columns", [])


def _db_footnote(tool_names: list[str]) -> str:
    r"""Return a "Database last updated" footnote for DB-backed tool responses.

    Reads ``db_last_modified`` from ``_last_evidence_store`` for each tool in
    *tool_names* and returns a formatted footnote line.  Returns an empty string
    when no timestamp is available (errors, in-memory test DBs, non-DB tools).

    Args:
        tool_names: Names of the tools whose evidence to inspect.

    Returns:
        Footnote string like ``"\n\nDatabase last updated: 2026-04-07 10:31 UTC"``,
        or an empty string when no timestamp is found.
    """
    for name in tool_names:
        _stored = _last_evidence_store.get(name, {})
        evidence = _stored.get("evidence", _stored)
        ts = evidence.get("db_last_modified")
        if ts:
            return f"\n\nDatabase last updated: {ts}"
    return ""


def get_last_traceback_evidence() -> dict[str, Any] | None:
    """Return the stored panda_log_analysis evidence if it contains a pilot traceback.

    Used by ``bamboo_answer._build_deterministic_plan`` to detect whether a
    follow-up question should be routed to ``pilot_source_analysis`` rather
    than the default job-status path.

    The gate is ``traceback_available`` plus a resolved ``deepest_pilot_frame``,
    not a specific ``failure_type``.  It used to require
    ``failure_type == "pilot_monitoring_error"``, which made source analysis
    unreachable for every other kind of pilot exception — including the
    transform-download timeouts that motivated the traceback-first extractor.
    Requiring ``deepest_pilot_frame`` matters because ``pilot_source_analysis``
    fetches pilot3 modules from GitHub: a traceback with no pilot frames (a pure
    Athena payload traceback) gives it nothing to fetch.

    ``failure_type == "pilot_monitoring_error"`` with a ``log_excerpt`` is still
    accepted so that evidence produced by an older deployment, which predates
    the ``traceback_available`` key, continues to route correctly.

    Returns:
        The ``evidence`` sub-dict from the last ``panda_log_analysis`` call when
        source-level analysis is possible; ``None`` otherwise.
    """
    stored = _last_evidence_store.get("panda_log_analysis", {})
    evidence = stored.get("evidence", stored)
    if not isinstance(evidence, dict):
        return None
    if evidence.get("traceback_available") and evidence.get("deepest_pilot_frame"):
        return evidence
    # Legacy path: evidence from a deployment without traceback_available.
    if (
        evidence.get("failure_type") == "pilot_monitoring_error"
        and evidence.get("log_excerpt")
    ):
        return evidence
    return None


# Backwards-compatible alias for the pre-widening name.  Retained so external
# callers and tests that import the old symbol keep working.
get_last_pilot_monitoring_evidence = get_last_traceback_evidence


def _log_analysis_offer_md() -> str:
    """Return the pre-built code-analysis follow-up offer from the last log analysis.

    Like ``links_md``, the offer is built deterministically in
    ``log_analysis_impl.fetch_and_analyse`` from the parsed traceback frame, so
    the pilot path, line number and function name reaching the user are always
    the real ones rather than whatever the LLM reproduced from memory.

    Returns:
        Markdown offer string, or an empty string when the last failure had no
        pilot traceback to analyse.
    """
    stored = _last_evidence_store.get("panda_log_analysis", {})
    evidence = stored.get("evidence", stored)
    if not isinstance(evidence, dict):
        return ""
    return str(evidence.get("code_analysis_offer_md", ""))


def _log_analysis_core_dump_offer_md() -> str:
    """Return the pre-built core-dump follow-up offer from the last log analysis.

    Built deterministically in ``log_analysis_impl._build_core_dump_offer``
    from the job-log listing, and deliberately narrow: it is emitted only for a
    looping-job kill (pilot code 1150) that left a non-empty core file.  Like
    ``code_analysis_offer_md`` it is a presentation key, hidden from the
    synthesis LLM and appended here instead, so the file name and size the user
    sees are the listing's own rather than the model's recollection.

    Returns:
        Markdown offer string, or an empty string when the last failure had no
        analysable core dump.
    """
    stored = _last_evidence_store.get("panda_log_analysis", {})
    evidence = stored.get("evidence", stored)
    if not isinstance(evidence, dict):
        return ""
    return str(evidence.get("core_dump_offer_md", ""))


def get_last_core_dump_offer() -> dict[str, Any] | None:
    """Return the stored core-dump offer from the last log analysis, if any.

    This is the gate for ``bamboo_answer``'s affirmative follow-up rule: a bare
    "yes" only means "analyse the core dump" when an offer to do so is actually
    outstanding.  The job ID is recovered from the evidence rather than from
    the user's message, since the affirmative itself carries no ID.

    The offer string is the gate rather than ``core_dump_available``, because
    ``_build_core_dump_offer`` applies conditions this function should not
    duplicate — the pilot error code, a non-empty core, and whether the tool is
    installed at all.  An empty string means no offer was made, whatever the
    reason.

    Returns:
        ``{"job_id": int, "offer_md": str}`` when the last ``panda_log_analysis``
        offered a core-dump analysis and recorded a usable job ID; ``None``
        otherwise.
    """
    stored = _last_evidence_store.get("panda_log_analysis", {})
    evidence = stored.get("evidence", stored)
    if not isinstance(evidence, dict):
        return None
    offer = str(evidence.get("core_dump_offer_md", ""))
    if not offer.strip():
        return None
    try:
        job_id = int(evidence["job_id"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"job_id": job_id, "offer_md": offer}


def _compact_json(obj: Any, limit: int = 12000) -> str:
    """Compact JSON for prompts, bounded to ``limit`` characters.

    Args:
        obj: Any JSON-serialisable object.
        limit: Maximum character count before truncation.

    Returns:
        Compact JSON string, truncated with an ellipsis if over ``limit``.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:  # pylint: disable=broad-exception-caught
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(truncated)"
    return s


__all__ = [
    "execute_plan",
    "call_llm",
    "unpack_tool_result",
    "retrieve_rag_context",
    "_pick_synthesis_prompt",
    "get_last_pilot_monitoring_evidence",
    "get_last_core_dump_offer",
    "_CORE_DUMP_TOOL",
    "_render_core_dump_markdown",
    "_append_acquisition_warnings",
    "_synthesise_core_dump",
    # Private, but reached by name from outside this module (tests, and
    # bamboo_answer for the evidence store).  Listing them keeps the export
    # surface honest: a py.typed marker would otherwise make pyright reject
    # exactly these while accepting their neighbours, which reads as a bug in
    # the caller rather than as a missing export.
    "_core_dump_evidence",
    "_log_analysis_core_dump_offer_md",
    "_strip_presentation_keys",
    "_last_evidence_store",
    "_TOOL_PROMPT_TABLE",
    "_PLUGIN_DOC_TOOLS",
    "_DEFAULT_DOC_TOOLS",
    "_SYSTEM_LOG_ANALYSIS",
    "_SYSTEM_PILOT_SOURCE",
    "_SYSTEM_CODE_QUERY",
    "_MERMAID_GUIDANCE",
    "_SYSTEM_JOB",
    "_SYSTEM_TASK",
    "_SYSTEM_RAG",
    "_SYSTEM_RAG_NO_CONTEXT",
    "_SYSTEM_GENERIC",
    "_SYSTEM_JOB_STATS",
    "_SYSTEM_JOBS_QUERY",
    "_SYSTEM_PROMPTLOG_QUERY",
    "_SYSTEM_CRIC_QUERY",
    "_format_cric_full_list",
    "_CRIC_DIRECT_FORMAT_THRESHOLD",
    "_SYSTEM_HARVESTER_WORKERS",
    "_SYSTEM_HARVESTER_TIMESERIES",
    "_SYSTEM_SITE_HEALTH",
    "_SYSTEM_PANDA_HEALTH",
    "bamboo_last_evidence_tool",
    "bamboo_promptlog_status_tool",
]


class BambooLastEvidenceTool:
    """MCP tool that returns the evidence dict from the most recent tool call.

    ``execute_plan`` stores the unpacked evidence from every tool call in
    ``_last_evidence_store``.  This tool exposes that store so the TUI
    ``/json`` and ``/inspect`` commands can retrieve it without making a
    fresh HTTP request to BigPanDA.

    Two modes (controlled by the ``mode`` argument):

    ``"evidence"`` (default)
        The compact structured evidence dict — job counts, site breakdown,
        error tallies, sample job records.  This is what was sent to the LLM.

    ``"raw"``
        The verbatim BigPanDA API response stored under
        ``evidence["raw_payload"]``, if present.  Falls back to the full
        evidence dict if ``raw_payload`` is absent.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool definition for ``bamboo_last_evidence``.

        Returns:
            Tool definition dict compatible with MCP discovery.
        """
        return {
            "name": "bamboo_last_evidence",
            "description": (
                "Return the evidence dict from the most recent panda_task_status "
                "or panda_log_analysis call.  Use mode='evidence' (default) for "
                "the compact structured summary, or mode='raw' for the verbatim "
                "BigPanDA API response."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["evidence", "raw", "table"],
                        "description": (
                            "'evidence' returns the compact LLM-facing evidence dict; "
                            "'raw' returns the verbatim BigPanDA API payload; "
                            "'table' returns the pre-formatted CRIC full-list table text."
                        ),
                    },
                    "tool": {
                        "type": "string",
                        "description": (
                            "Optional tool name to retrieve evidence for "
                            "(e.g. 'panda_task_status').  Defaults to the "
                            "most recently called tool."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:
        """Return stored evidence from the last tool execution.

        Args:
            arguments: Dict with optional ``"mode"`` (``"evidence"`` or
                ``"raw"``) and optional ``"tool"`` name.

        Returns:
            One-element MCP content list with the JSON-serialised evidence,
            or an error message if no evidence is stored yet.
        """
        mode: str = str(arguments.get("mode") or "evidence")
        requested_tool: str | None = arguments.get("tool") or None

        if not _last_evidence_store:
            return text_content(json.dumps({
                "error": "No evidence stored yet — ask about a task or job first."
            }))

        # The store is keyed on canonical wire names, so a caller asking for
        # "core_dump_analysis" must reach the entry written under
        # "atlas.core_dump_analysis".  A name with no canonical form passes
        # through unchanged and simply misses, as before.
        if requested_tool:
            tool_name: str | None = canonical_tool_name(requested_tool)
        else:
            tool_name = _last_evidence_store.get("last_tool")
        evidence = _last_evidence_store.get(str(tool_name), {}) if tool_name else {}

        if not evidence:
            return text_content(json.dumps({
                "error": f"No evidence stored for tool {tool_name!r}.",
                "available_tools": [k for k in _last_evidence_store if k != "last_tool"],
            }))

        if mode == "table":
            # Return the pre-formatted CRIC full-list table stored by the
            # direct-format bypass in execute_plan.  The table is keyed
            # separately so it survives independent of which tool ran last.
            table_text = _last_evidence_store.get("_cric_direct_table")
            if not table_text:
                return text_content(json.dumps({
                    "error": "No CRIC table available — ask to list all queues first.",
                }))
            return text_content(json.dumps({"table": table_text}))

        if mode == "raw":
            payload = evidence.get("raw_payload")
            result = payload if isinstance(payload, dict) else evidence
        else:
            # Return evidence without the raw_payload to keep it compact.
            result = {k: v for k, v in evidence.items() if k != "raw_payload"}

        return text_content(json.dumps({
            "tool": tool_name,
            "mode": mode,
            "evidence": result,
        }))


bamboo_last_evidence_tool = BambooLastEvidenceTool()


class BambooPromptLogStatusTool:
    """MCP tool that drains and returns buffered OpenSearch prompt-log events.

    ``_write_document`` in :mod:`bamboo.llm.prompt_log` appends an event to a
    process-local ring buffer on every successful write and on every failure.
    This tool exposes that buffer so the TUI and Streamlit interfaces can poll
    for write confirmations and errors after each response — without requiring
    in-process callbacks, which do not work across the stdio subprocess
    boundary.

    Calling this tool is destructive: the buffer is cleared on each call so
    events are delivered exactly once.  Returns an empty list when
    ``BAMBOO_OPENSEARCH_PROMPTLOG`` is not set or no events have accumulated
    since the last poll.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool definition for ``bamboo_promptlog_status``.

        Returns:
            Tool definition dict compatible with MCP discovery.
        """
        return {
            "name": "bamboo_promptlog_status",
            "description": (
                "Return and clear buffered OpenSearch prompt-log write events "
                "(confirmations and errors) accumulated since the last call.  "
                "Returns an empty list when prompt logging is disabled or no "
                "events have been buffered."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[MCPContent]:  # noqa: ARG002
        """Drain and return buffered prompt-log events.

        Args:
            arguments: Unused; accepted for MCP tool interface compatibility.

        Returns:
            One-element MCP content list with JSON-serialised event list.
            Each event has ``"turn"`` (int), ``"severity"``
            (``"info"`` / ``"warning"`` / ``"error"``), and ``"message"``
            (str) keys.
        """
        from bamboo.llm.prompt_log import drain_events  # noqa: PLC0415
        events = drain_events()
        return text_content(json.dumps({"events": events}))


bamboo_promptlog_status_tool = BambooPromptLogStatusTool()


class BambooPromptlogRateTool:
    """MCP tool for rating a Bamboo prompt-log entry (1–5 stars).

    Updates the ``rating`` field of an existing OpenSearch document using
    the ``update`` API.  The ``index`` and ``doc_id`` are obtained from the
    system panel notification shown after each indexed turn
    (``index='…'`` and ``id='…'``).
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool definition for ``bamboo_promptlog_rate``.

        Returns:
            Tool definition dict compatible with MCP discovery.
        """
        return {
            "name": "bamboo_promptlog_rate",
            "description": (
                "Rate a Bamboo LLM response by updating its OpenSearch prompt-log "
                "document with a star rating (1–5). "
                "1 = very poor (red), 5 = excellent (green). "
                "Requires the index name and document id from the system panel "
                "notification (e.g. index='bamboomcp-promptlog-2026.05.26' "
                "id='abc123'). "
                "Requires BAMBOO_OPENSEARCH_PROMPTLOG to be set."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": (
                            "OpenSearch index name, e.g. "
                            "'bamboomcp-promptlog-2026.05.26'."
                        ),
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "OpenSearch document _id from the system panel.",
                    },
                    "rating": {
                        "type": "integer",
                        "description": "Star rating 1 (very poor) to 5 (excellent).",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["index", "doc_id", "rating"],
                "additionalProperties": False,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Apply the star rating to the specified prompt-log document.

        Args:
            arguments: MCP tool argument dict with ``index``, ``doc_id``,
                and ``rating`` keys.

        Returns:
            One-element MCP content list with a JSON confirmation or error.
        """
        from bamboo.llm.prompt_log import update_rating  # local import

        index: str = arguments.get("index", "").strip()
        doc_id: str = arguments.get("doc_id", "").strip()
        rating_raw = arguments.get("rating")

        if not index or not doc_id:
            return text_content(json.dumps(
                {"error": "Both 'index' and 'doc_id' are required."}
            ))
        try:
            rating = int(rating_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return text_content(json.dumps(
                {"error": f"rating must be an integer 1–5, got {rating_raw!r}"}
            ))

        try:
            resp = await asyncio.to_thread(
                update_rating, index, doc_id, rating
            )
            result = resp.get("result", "?") if isinstance(resp, dict) else str(resp)
            return text_content(json.dumps({
                "rated": True,
                "index": index,
                "doc_id": doc_id,
                "rating": rating,
                "result": result,
            }))
        except ValueError as exc:
            return text_content(json.dumps({"error": str(exc)}))
        except RuntimeError as exc:
            return text_content(json.dumps({"error": str(exc)}))
        except ImportError:
            return text_content(json.dumps(
                {"error": "opensearch-py not installed: pip install opensearch-py"}
            ))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return text_content(json.dumps({"error": str(exc)}))


bamboo_promptlog_rate_tool = BambooPromptlogRateTool()
