"""ATLAS log-fetch primitives — askpanda_atlas plugin package.

Delegates to the canonical implementation in
``askpanda_atlas.log_primitives_impl``.  Mirrors the wrapper convention the
other plugin tools follow, so the entry point names a stable module path that
does not move if the implementation is reorganised.

Unlike ``log_analysis.py`` there is no fallback implementation: the primitives
exist to be composed by a code-mode agent talking to a running Bamboo server,
so there is no isolated-exercise case for them to serve.
"""
from __future__ import annotations

from askpanda_atlas.log_primitives_impl import (  # noqa: F401
    AtlasLogPlanFetchTool,
    get_definition,
    plan_fetch,
    plan_fetch_tool,
)

__all__ = [
    "AtlasLogPlanFetchTool",
    "get_definition",
    "plan_fetch",
    "plan_fetch_tool",
]
