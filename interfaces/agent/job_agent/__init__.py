r"""Deterministic job-failure analysis agent over the ``atlas.log.*`` primitives.

Bamboo ships two surfaces over the same log-analysis implementation: the
compound ``panda_log_analysis`` that its own planner selects, and the five
``atlas.log.*`` *code-mode* primitives a caller composes itself.  This package
is the reference composition of the second — the loop
``docs/code-mode.md`` writes out, driven against a
live Bamboo MCP server and finished with one synthesis call.

It is deliberately **not** a ReAct agent.  Every decision on the diagnosis path
is a server-side domain rule — ``plan_fetch`` chooses the files, ``fetch_text``
derives the character budget from the role it was given, ``classify`` picks
which traceback to trust — and an LLM placed inside that loop would add
latency, cost and non-determinism while buying nothing.  It would also break
the property that makes the composition worth having: equivalence with
``panda_log_analysis`` is a property of *that* loop, not of any loop.

Only :mod:`~interfaces.agent.job_agent.composer` is re-exported here.  The CLI
is imported separately so this package stays importable without the MCP SDK
installed, which is what lets the Track A evaluation harness consume
:func:`analyse_job` in-process against its own fake client.

Run it with::

    python -m interfaces.agent.job_agent --panda-id 6799893074 \\
        --host aipanda033.cern.ch --port 8000
"""
from __future__ import annotations

from interfaces.agent.job_agent.composer import (
    MAX_PLAN_CALLS,
    OUTCOME_ANALYSED,
    OUTCOME_ERROR,
    OUTCOME_NO_LOGS,
    PRIMITIVE_TOOLS,
    JobAgentError,
    JobAnalysisResult,
    MCPCallable,
    PrimitiveError,
    ToolCallError,
    ToolUnavailableError,
    analyse_job,
    analyse_jobs,
    build_synthesis_brief,
    missing_primitives,
)

__all__ = [
    "MAX_PLAN_CALLS",
    "OUTCOME_ANALYSED",
    "OUTCOME_ERROR",
    "OUTCOME_NO_LOGS",
    "PRIMITIVE_TOOLS",
    "JobAgentError",
    "JobAnalysisResult",
    "MCPCallable",
    "PrimitiveError",
    "ToolCallError",
    "ToolUnavailableError",
    "analyse_job",
    "analyse_jobs",
    "build_synthesis_brief",
    "missing_primitives",
]
