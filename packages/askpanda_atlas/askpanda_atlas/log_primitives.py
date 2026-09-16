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
    AtlasLogClassifyTool,
    AtlasLogFetchMetadataTool,
    AtlasLogFetchTextTool,
    AtlasLogListFilesTool,
    AtlasLogPlanFetchTool,
    classify,
    classify_tool,
    fetch_metadata,
    fetch_metadata_tool,
    fetch_text,
    fetch_text_tool,
    get_classify_definition,
    get_definition,
    get_fetch_text_definition,
    get_list_files_definition,
    get_metadata_definition,
    list_files,
    list_files_tool,
    plan_fetch,
    plan_fetch_tool,
)

__all__ = [
    "AtlasLogClassifyTool",
    "AtlasLogFetchMetadataTool",
    "AtlasLogFetchTextTool",
    "AtlasLogListFilesTool",
    "AtlasLogPlanFetchTool",
    "classify",
    "classify_tool",
    "fetch_metadata",
    "fetch_metadata_tool",
    "fetch_text",
    "fetch_text_tool",
    "get_classify_definition",
    "get_definition",
    "get_fetch_text_definition",
    "get_list_files_definition",
    "get_metadata_definition",
    "list_files",
    "list_files_tool",
    "plan_fetch",
    "plan_fetch_tool",
]
