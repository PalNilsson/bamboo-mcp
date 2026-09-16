#!/usr/bin/env bash
#
# Example environment configuration for AskPanDA LLM support
# Copy this file, remove `_example`, and fill in the API keys as needed.
# Remember to add this file to your .gitignore to avoid committing sensitive information.

########################################
# PANDA RELATED
########################################

export PANDA_BASE_URL="https://bigpanda.cern.ch"
export ASKPANDA_PANDA_RETRIES="2"
export ASKPANDA_PANDA_BACKOFF_SECONDS="0.8"

# Path to the DuckDB file written by the ingestion agent.
# Used by the panda_jobs_query tool (atlas.jobs_query).
# Defaults to "jobs.duckdb" in the current working directory if unset.
export PANDA_DUCKDB_PATH="jobs.duckdb"

# Optional: maximum rows returned by panda_jobs_query (default: 500).
# export PANDA_JOBS_QUERY_MAX_ROWS="500"

# Path to the CRIC queuedata DuckDB file written by the cric_agent.
# Used by the cric_query tool (atlas.cric_query).
# Defaults to "cric.duckdb" in the current working directory if unset.
export CRIC_DUCKDB_PATH="${HOME}/.askpanda/cric.duckdb"

# Optional: maximum rows returned by cric_query (default: 200).
# export CRIC_QUERY_MAX_ROWS="200"

########################################
# PANDA MCP (external PanDA MCP server)
########################################

# Full URL of the PanDA MCP HTTP endpoint.
# If unset, PanDA MCP tools return a graceful "server not connected" error.
export PANDA_MCP_BASE_URL="https://aipanda120.cern.ch:8443/mcp"

# OIDC token cache file written by `get-panda-token` (panda-mcp-client).
# Bamboo reads the `id_token` field from this file at session startup.
# Run `uvx --from panda-mcp-client get-panda-token` once to populate it.
# Token renewal is handled separately (via a Bamboo MCP agent service).
# Defaults to ~/.panda_id_token when unset.
# export PANDA_MCP_TOKEN_FILE="${HOME}/.panda_id_token"

# Explicit bearer token override (takes priority over PANDA_MCP_TOKEN_FILE).
# Leave unset to use the token file instead.
# export PANDA_MCP_TOKEN=""

# Optional virtual-organisation name sent as Origin: <vo> header.
# export PANDA_MCP_ORIGIN="atlas"

# TLS certificate verification.
# httpx and requests both honour the standard SSL_CERT_FILE env var.
#
# On lxplus (recommended): point at the system CA bundle — no certifi
# modifications needed, survives venv recreations and certifi upgrades:
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
#
# Outside CERN: build a combined bundle (system CAs + CERN CAs) and point
# SSL_CERT_FILE at it, or append the CERN CAs to your system bundle.
# Note: files from cafiles.cern.ch are DER-encoded — they need
# `openssl x509 -inform DER` conversion before use in a PEM bundle.
# export SSL_CERT_FILE="/path/to/ca-bundle-with-cern-cas.pem"
#
# Or point PANDA_MCP_CA_BUNDLE at a combined PEM bundle (Bamboo-specific,
# used only for the PanDA MCP connection):
# export PANDA_MCP_CA_BUNDLE="/path/to/ca-bundle.pem"
#
# Development/testing only — disables TLS verification entirely:
# export PANDA_MCP_TLS_VERIFY=0

########################################
# LLM PROFILE SELECTION
########################################

# Which profile names the selector will use
export LLM_DEFAULT_PROFILE="default"
export LLM_FAST_PROFILE="fast"
export LLM_REASONING_PROFILE="reasoning"

########################################
# DEFAULT PROFILE (used if nothing else matches)
########################################

export LLM_DEFAULT_PROVIDER="mistral"
export LLM_DEFAULT_MODEL="mistral-large-latest"

########################################
# FAST PROFILE (classification, routing, lightweight tasks)
########################################

export LLM_FAST_PROVIDER="mistral"
export LLM_FAST_MODEL="mistral-large-latest"

########################################
# REASONING PROFILE (log analysis, synthesis, RAG answers)
########################################

export LLM_REASONING_PROVIDER="mistral"
export LLM_REASONING_MODEL="mistral-large-latest"

########################################
# MISTRAL CONFIGURATION
########################################

# Required when using provider="mistral"
export MISTRAL_API_KEY=""

# Optional concurrency / retry tuning
export ASKPANDA_MISTRAL_CONCURRENCY="4"
export ASKPANDA_MISTRAL_RETRIES="3"
export ASKPANDA_MISTRAL_BACKOFF_SECONDS="1.0"

########################################
# OPENAI CONFIGURATION
########################################

# Required when using provider="openai" or provider="openai_compat".
# Install: pip install -r requirements-openai.txt
export OPENAI_API_KEY=""

# Optional tuning for the OpenAI provider.
# export ASKPANDA_OPENAI_CONCURRENCY="8"
# export ASKPANDA_OPENAI_RETRIES="3"
# export ASKPANDA_OPENAI_BACKOFF_SECONDS="1.0"

########################################
# ANTHROPIC CONFIGURATION
########################################

# Required when using provider="anthropic".
# Install: pip install -r requirements-anthropic.txt
export ANTHROPIC_API_KEY=""

# Optional tuning for the Anthropic provider.
# export ASKPANDA_ANTHROPIC_CONCURRENCY="4"
# export ASKPANDA_ANTHROPIC_RETRIES="3"
# export ASKPANDA_ANTHROPIC_BACKOFF_SECONDS="1.0"

########################################
# GEMINI CONFIGURATION
########################################

# Required when using provider="gemini".
# Install: pip install -r requirements-gemini.txt
export GEMINI_API_KEY=""

# Optional tuning for the Gemini provider.
# export ASKPANDA_GEMINI_CONCURRENCY="4"
# export ASKPANDA_GEMINI_RETRIES="3"
# export ASKPANDA_GEMINI_BACKOFF_SECONDS="1.0"

########################################
# OPENAI-COMPATIBLE ENDPOINT (Llama / Mistral via vLLM, Ollama, etc.)
########################################

# Required when using provider="openai_compat".
# Uses the same openai SDK as the OpenAI provider.
# Install: pip install -r requirements-openai.txt
export ASKPANDA_OPENAI_COMPAT_BASE_URL=""
export OPENAI_COMPAT_API_KEY=""

# Optional tuning.
# export ASKPANDA_OPENAI_COMPAT_CONCURRENCY="8"
# export ASKPANDA_OPENAI_COMPAT_RETRIES="3"
# export ASKPANDA_OPENAI_COMPAT_BACKOFF_SECONDS="1.0"

########################################
# PLUGIN SELECTION
########################################

# Active experiment plugin for the TUI and Streamlit interfaces.
# Controls which ui_manifest tool is loaded and which banner/accent is shown.
# Options: atlas | epic | cgsim   (default: atlas)
# Must be explicitly exported (not commented out) so that sourcing this file
# clears any stale value left in the shell from a previous cgsim/epic session.
export ASKPANDA_PLUGIN="atlas"

########################################
# TOOL PROFILE
########################################

# Which advertised tool surface the MCP server publishes.
# Options: orchestrated | primitive | both   (default: orchestrated)
#
#   orchestrated  Compound tools, driven by Bamboo's own planner. Today's
#                 behaviour, and what the Textual/Streamlit interfaces expect.
#   primitive     Fine-grained primitives for an external code-mode agent that
#                 composes them itself.
#   both          Publish both surfaces and let the client choose.
#
# Only tools that opt in via their definition are affected; everything else is
# advertised under every profile. Affects advertising (tools/list and the
# planner catalog) only — any registered tool remains callable. An unrecognised
# value logs a warning and falls back to orchestrated.
#
# Tools currently opting in to "primitive":
#   atlas.log.plan_fetch      Decides which of a failed job's logs to download
#                             next. Re-entrant: fetch what "next" names, pass
#                             the signals back as "observed", until "done".
#   atlas.log.fetch_metadata  Job metadata subset: status, site, error codes,
#                             timing. Facts only, no decisions.
#   atlas.log.list_files      The job's log tarball with sizes, root files
#                             first.
#   atlas.log.fetch_text      Downloads one log file and returns its
#                             diagnostic excerpt, plus the setup_has_error
#                             signal plan_fetch consumes.
#   atlas.log.classify        Joins the fetched excerpts and returns the
#                             failure category. Pure — no network.
#
# The five compose into the loop panda_log_analysis runs internally:
#   fetch_metadata -> plan_fetch <-> fetch_text -> classify
#
# Primitives declare an outputSchema and return structured content, so they
# require mcp >= 1.10.0 at runtime — below that floor the SDK ignores the
# schema and the structured half of the result is silently dropped rather than
# failing loudly. The pin in requirements.txt enforces this; check the
# deployment venv after upgrading.
#
# Left commented out: the default is the correct setting for every interface
# shipped in this repo.
# export BAMBOO_TOOL_PROFILE="orchestrated"

# Character budget for what a primitive returns: the excerpt atlas.log.fetch_text
# extracts from a log file, and the verbatim traceback carried alongside it.
#
# Deliberately separate from the monolith's internal _MAX_EXCERPT_CHARS, which
# it happens to equal by default (8000). A code-mode agent may have a far
# larger context than Bamboo's own synthesis step; raising this budget for one
# must not move the other, so panda_log_analysis is unaffected by any value set
# here.
#
# On the payload path the budget is split as the monolith splits it:
# payload.stdout is excerpted against (budget - 2000) and payload.stderr
# against 2000, so the joined excerpt stays within budget. setup.stdout and
# pilotlog.txt get the whole figure.
#
# An unset, unparseable or non-positive value logs a warning and falls back to
# 8000, rather than failing every call over a configuration typo.
# export BAMBOO_PRIMITIVE_MAX_CHARS="8000"

########################################
# RAG / CHROMADB (doc_search / doc_bm25 tools)
########################################

# Path to the ChromaDB persistent directory created by the ingestion script.
export BAMBOO_CHROMA_PATH="./chroma_db"

# Multi-collection map: JSON object mapping topic keys to logical collection
# names.  The document-monitor agent in bamboo-mcp-services ingests each
# source repository into a separate named collection; this map tells Bamboo
# MCP which collection to query for each topic.
#
# Adding a new collection requires only updating this JSON string — no code
# changes or new environment variables.
#
# Built-in topic defaults (used when a topic key is absent from the map):
#   panda          → panda_docs          atlas    → atlas_docs
#   bamboo         → bamboo_docs (legacy alias; see note below)
#   bamboo_mcp     → bamboo_mcp_docs     bamboo_services → bamboo_services_docs
#   rucio          → rucio_docs          root     → root_docs
#   epic           → epic_docs           cgsim    → cgsim_docs
#
# The "bamboo" key is a legacy alias retained for single-collection
# deployments.  Deployments that have split Bamboo documentation into two
# separate collections (bamboo-mcp repo docs vs bamboo-mcp-services repo docs)
# should add "bamboo_mcp" and "bamboo_services" entries explicitly to avoid
# install/setup docs from one component polluting answers about the other.
#
# Uncomment and adjust to match your bamboo-mcp-services deployment:
# export BAMBOO_CHROMA_COLLECTION_MAP='{
#     "panda":           "panda_docs",
#     "atlas":           "atlas_docs",
#     "bamboo_mcp":      "bamboo_mcp_docs",
#     "bamboo_services": "bamboo_services_docs",
#     "rucio":           "rucio_docs",
#     "root":            "root_docs",
#     "epic":            "epic_docs",
#     "cgsim":           "cgsim_docs"
# }'
#
# Legacy single-collection layout (bamboo-mcp + bamboo-mcp-services docs mixed):
# export BAMBOO_CHROMA_COLLECTION_MAP='{
#     "panda":  "panda_docs",
#     "atlas":  "atlas_docs",
#     "bamboo": "bamboo_docs",
#     "rucio":  "rucio_docs",
#     "root":   "root_docs",
#     "epic":   "epic_docs",
#     "cgsim":  "cgsim_docs"
# }'

# Scalar fallback: used for all topics when BAMBOO_CHROMA_COLLECTION_MAP is
# absent or has no entry for the requested topic.  Also preserves backward
# compatibility with single-collection deployments.
export BAMBOO_CHROMA_COLLECTION="atlas_docs"

########################################
# DEBUG / SAFETY
########################################

# Fast-path routing: deterministic regex-based routing for task/job/pilot
# questions (faster, no LLM planner call).  Set to 0 to always use the LLM
# planner instead — useful when experimenting with new plugins like CGSim
# where the fast-path rules are not yet tuned for the domain.
# Options: 1 (on, default when unset) | 0 (off — use LLM planner)
export BAMBOO_FAST_PATH="0"

# Uncomment for verbose debug logs if needed
# export ASKPANDA_DEBUG="1"

########################################
# SUPERUSER / DEVELOPER MODE
########################################

# Plain-text password to unlock superuser mode in the Streamlit and TUI
# interfaces.  When set, a "Developer access" section appears in the
# Streamlit sidebar and the /superuser <password> command is active in
# the TUI.  Superuser mode exposes developer tools such as pilot_code_query.
# Leave unset (or empty) to disable superuser mode entirely.
# export BAMBOO_SUPERUSER_PASSWORD="changeme"

# Additional tool names to treat as superuser-gated (comma-separated).
# The defaults (code_query, atlas.code_query) are always included.
# Example: bamboo_code_query,atlas.bamboo_code_query
# export BAMBOO_SUPERUSER_TOOLS=""

# Additional routing-signal regex patterns for the pre-dispatch superuser
# guard (comma-separated, Python re syntax, case-insensitive).
# Any question matching one of these patterns is blocked until the user
# authenticates as a superuser.  The defaults cover *.py filenames and
# code-inspection verb + keyword combinations.
# Example: bamboo/.*\.py,core/bamboo/.*
# export BAMBOO_SUPERUSER_PATTERNS=""

########################################
# CODE QUERY (superuser tool)
########################################

# GitHub repository to fetch source files from.
# Format: owner/repo  (default: PanDAWMS/pilot3 — override for any other codebase)
# export BAMBOO_CODE_QUERY_REPO="PanDAWMS/pilot3"

# Branch or tag to fetch from (default: master).
# Example: export BAMBOO_CODE_QUERY_REPO="my-org/my-repo"
# export BAMBOO_CODE_QUERY_BRANCH="master"

########################################
# TRACING
########################################

# Set to 1 to enable structured request/response tracing.
# When BAMBOO_TRACE_FILE is set, spans are written only to that file (stderr
# is left clean — required when running under the Textual TUI).
# When BAMBOO_TRACE_FILE is not set, spans are written to stderr instead.
# See docs/tracing.md for the full event schema and jq recipes.
# export BAMBOO_TRACE="1"
# export BAMBOO_TRACE_FILE="/tmp/bamboo_trace.jsonl"

# OpenTelemetry export (optional — requires pip install -r requirements-otel.txt).
# When set, spans are also exported via OTLP/gRPC to the given endpoint
# (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc.) as a parent/child tree.
# export BAMBOO_OTEL_ENDPOINT="http://localhost:4317"
# export BAMBOO_OTEL_SERVICE_NAME="bamboo"   # default: bamboo
# export BAMBOO_OTEL_INSECURE="1"            # set to 0 to enable TLS

# Set to 1 to redirect the server's stderr to /dev/null.
# The Textual TUI sets this automatically when launching via stdio transport.
# Useful if running the server as a background subprocess in other contexts.
# export BAMBOO_QUIET="1"

# ---------------------------------------------------------------------------
# Context memory (multi-turn chat history)
# ---------------------------------------------------------------------------
# Maximum number of user+assistant turn *pairs* to keep in context per session.
# Each pair = 1 user message + 1 assistant reply (2 messages total).
# Default: 10 pairs (20 messages).  Set lower to reduce LLM token usage.
# History is held in-memory in the TUI only; the server is always stateless.
# export BAMBOO_HISTORY_TURNS="10"

# Maximum tokens for LLM synthesis responses.
# Raise these for longer, more detailed answers — at the cost of higher latency.
# export BAMBOO_SYNTHESIS_MAX_TOKENS="2048"   # fresh questions (default: 2048)
# export BAMBOO_FOLLOWUP_MAX_TOKENS="600"     # follow-up expansions (default: 600)

echo "AskPanDA LLM environment variables loaded (example configuration)."

########################################
# HTTP SERVER (bamboo.entrypoints.http)
########################################

# Bind host for the HTTP MCP server (python -m bamboo.server_http).
# 127.0.0.1  → localhost only (default, safe for local development)
# 0.0.0.0    → all interfaces (required for remote clients)
# export BAMBOO_HTTP_HOST="127.0.0.1"

# Bind port (default: 8000)
# export BAMBOO_HTTP_PORT="8000"

# Uvicorn log level: debug | info | warning | error (default: info)
# export BAMBOO_HTTP_LOG_LEVEL="info"

# Bearer token auth — leave unset for open access (local/testbed use).
# Option A: tokens file (one entry per line, format: client_id: token)
# export BAMBOO_MCP_TOKENS_FILE="/etc/bamboo/tokens.txt"
# Option B: inline comma-separated client_id:token pairs
# export BAMBOO_MCP_TOKENS="alice:token-abc,bob:token-xyz"

########################################
# STREAMLIT / HTTP CLIENT
########################################

# Default MCP server URL for the Streamlit app and TUI in HTTP transport mode.
# export MCP_URL="http://localhost:8000/mcp"

# Bearer token for authenticating to a Bamboo HTTP server.
# export MCP_BEARER_TOKEN=""

# Timeouts in seconds for MCP tool calls.  These are two INDEPENDENT ceilings
# on the same call and the lower one silently wins, so raising one alone has no
# effect.  Both default to 300 s.
#
#   BAMBOO_MCP_CLIENT_TIMEOUT  client-side deadline on the call's future
#   BAMBOO_MCP_HTTP_TIMEOUT    per-tool-call deadline on the HTTP transport
#                              (not a connection timeout)
#
# Long-running tools need headroom under BOTH: large task status fetches take
# 60-90 s for tasks with thousands of jobs, and a tool that fetches and analyses
# a job's files takes longer still.  Pinning either below the default will cut
# such a call short while the work continues server-side.
# export BAMBOO_MCP_CLIENT_TIMEOUT="300"
# export BAMBOO_MCP_HTTP_TIMEOUT="300"

########################################
# CORE-DUMP ANALYSIS (atlas.core_dump_analysis)
########################################

# Root directory for analysis workspaces.  Each job gets one directory here,
# holding the reconstructed job tree, the core file, the gdb evidence and the
# worker log.  /tmp is adequate on aipanda033.
#
# NOTHING IS DELETED, EVER — not partial downloads, not failed runs, not
# superseded evidence.  Reaping belongs to a separate service script, so the
# quota below is what stops this directory growing without bound.
# export BAMBOO_CORE_ANALYSIS_ROOT="/tmp/bamboo/core-analysis"

# How long a 'start' call waits inline before handing back a request ID.  An
# analysis takes about a minute, so most calls return the full result in the
# same turn and the caller never sees a handle.  Must stay comfortably below
# BAMBOO_MCP_CLIENT_TIMEOUT above, which is the real ceiling.
# export BAMBOO_CORE_ANALYSIS_INLINE_WAIT="120"

# Age at which a run that is still not finished is declared failed.  This is
# the backstop for a worker that is alive but wedged; a worker that has *died*
# is detected at once from its pid and does not wait for this.
# export BAMBOO_CORE_ANALYSIS_HARD_TIMEOUT="900"

# Whole-container deadline passed to the analyzer as --container-timeout.  Well
# below the analyzer's own 1800 s default, which assumes a patient CLI user
# rather than an interactive session.
# export BAMBOO_CORE_ANALYSIS_CONTAINER_TIMEOUT="600"

# Byte ceiling for everything under BAMBOO_CORE_ANALYSIS_ROOT.  At or above it,
# a new analysis is refused and reports what is being held.
# export BAMBOO_CORE_ANALYSIS_MAX_BYTES="53687091200"

# The analysis needs CVMFS on the host: it reconstructs the job's own ATLAS
# release container rather than using the host's gdb.  There is no option to
# fall back to a local gdb, because a mismatched release resolves the payload's
# symbols against the wrong binaries and produces a confident, wrong answer.
# Set ATLAS_LOCAL_ROOT_BASE if CVMFS is mounted somewhere unusual.
# export ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"

# A container runtime is needed too, but not necessarily one installed on the
# host: the preflight looks for apptainer or singularity on this process's
# PATH, then on a *login* shell's PATH (which is what atlasLocalSetup.sh runs
# under, and is usually wider than a daemon's), then for ALRB's own apptainer
# under the CVMFS repository.  Set this only when the runtime lives somewhere
# none of those three find it.
# export BAMBOO_CORE_DUMP_APPTAINER="/opt/apptainer/bin/apptainer"

# Escape hatch: skip the runtime check entirely and let the analyzer report a
# missing runtime itself, a few seconds later.  The three CVMFS checks still
# run.  Use this if detection ever refuses an analysis that would have worked.
# export BAMBOO_CORE_DUMP_SKIP_RUNTIME_CHECK="1"

# Character budget for the evidence handed to the synthesis model.  The
# analyzer's own CLI default is 50000, sized for a small model; that is far too
# tight here, because the last stage of the reduction cascade is the primary
# thread's backtrace — so a job with many shared libraries and several distinct
# thread stacks spends its budget on cheaper evidence and then truncates the one
# field worth reading.  Nothing is lost from disk either way: evidence.json and
# gdb_raw.txt in the workspace are never budgeted.
# export BAMBOO_CORE_ANALYSIS_MAX_EVIDENCE_CHARS="120000"

# CPython gdb helper(s), enabling py-bt inside the release container.  Normally
# unnecessary — the analyzer searches next to every libpython object loaded from
# the core — but ATLAS/LCG releases do not always ship one, and gdb's auto-load
# then has no candidate to find.
#
# The helper reads CPython's own struct layouts, which change between MINOR
# versions, so it must match the interpreter in the core: a 3.12 helper cannot
# read a 3.11 process.  Point this at a DIRECTORY of per-version helpers rather
# than a single file, laid out as <version>/python-gdb.py; the analyzer detects
# the version from the core and picks the matching one, and declines rather than
# loading a mismatched helper.  A single file still works when every job you
# analyse uses the same interpreter.
#
#   /data/bamboo/tools/cpython-gdb/3.11/python-gdb.py
#   /data/bamboo/tools/cpython-gdb/3.12.13/python-gdb.py
#
# The path is read on the HOST and the helper is copied into the job directory,
# which the container sees at /srv.  It is not passed as an environment
# variable: ALRB launches apptainer with --cleanenv and binds only /cvmfs, the
# user's home, the job directory and a scratch path, so a helper anywhere else
# does not exist as far as the container is concerned.
# export BAMBOO_CORE_DUMP_PYTHON_GDB="/data/bamboo/tools/cpython-gdb"

# Directory of separate .debug files, for when a release ships libpython and
# the analysis libraries stripped of DWARF — which is what stops py-bt even
# when the correct helper loads, and what leaves optimised XrdCl frames without
# argument or local-variable data.
#
# LEAVE THIS UNSET unless your site actually publishes debug trees. ATLAS
# releases under /cvmfs/atlas.cern.ch/repo/sw/software do not ship them in any
# location known to this tool, so for a stock deployment there is nothing to
# point at and no Python-level backtrace is obtainable.
#
# If your site does publish them, write a TEMPLATE, not a path: that directory
# holds a hundred-odd releases and each job names its own, so a fixed path is
# wrong for every job but one. {project}, {release}, {platform} and {base} are
# filled from the payload log's setup banner —
#
#   Using AnalysisBase/25.2.103 [cmake] with platform x86_64-el9-gcc15-opt
#           at /cvmfs/atlas.cern.ch/repo/sw/software/25.2
#
# giving AnalysisBase, 25.2.103, x86_64-el9-gcc15-opt and the base path. The
# expanded directory must exist and be visible inside the container; it is
# dropped with a warning when it is not, because gdb accepts a missing path
# silently and loads nothing.
# export BAMBOO_CORE_DUMP_DEBUG_DIR="{base}/{project}/{release}/InstallArea/{platform}/.debug"
