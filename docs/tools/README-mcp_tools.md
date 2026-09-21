# Bamboo MCP tools reference

This directory contains one document per MCP tool exposed by the Bamboo server.

---

## Orchestration layer

These tools handle routing, planning, and synthesis. They are called by clients (including the TUI) rather than being called directly for data.

| Tool | Document | Description |
|---|---|---|
| `bamboo_answer` | [bamboo_answer.md](bamboo_answer.md) | Primary entry point — routes questions and returns synthesised answers. |
| `bamboo_plan` | [bamboo_plan.md](bamboo_plan.md) | LLM-backed planner that decomposes questions into tool call plans. |
| `bamboo_last_evidence` | [bamboo_last_evidence.md](bamboo_last_evidence.md) | Returns the evidence dict from the most recent data tool call. |

---

## Operational data tools

These tools fetch live PanDA data from BigPanDA, Harvester, OpenSearch, or local databases.

| Tool | Document | Description |
|---|---|---|
| `panda_log_analysis` | [panda_log_analysis.md](panda_log_analysis.md) | Job failure diagnosis — downloads pilot log, classifies failure, returns evidence. |
| `panda_job_status` | [panda_job_status.md](panda_job_status.md) | Status and metadata for a single PanDA job. |
| `panda_task_status` | [panda_task_status.md](panda_task_status.md) | Status, progress, and dataset info for a PanDA task. |
| `panda_harvester_workers` | [panda_harvester_workers.md](panda_harvester_workers.md) | Harvester pilot/worker counts from the BigPanDA API. |
| `panda_harvester_timeseries` | [panda_harvester_timeseries.md](panda_harvester_timeseries.md) | Per-bucket pilot counts from OpenSearch for time-series charts. |
| `panda_jobs_query` | [panda_jobs_query.md](panda_jobs_query.md) | Aggregate job statistics via natural-language SQL against the ingestion DB. |
| `cric_query` | [cric_query.md](cric_query.md) | Queue and site configuration via natural-language SQL against the CRIC DB. |
| `panda_server_health` | [panda_server_health.md](panda_server_health.md) | PanDA server liveness check. |

---

## Documentation retrieval tools

These tools search the PanDA/Bamboo documentation corpus for conceptual questions.

| Tool | Document | Description |
|---|---|---|
| `panda_doc_search` | [panda_doc_search.md](panda_doc_search.md) | Vector similarity search (ChromaDB) — ATLAS / ePIC corpus. |
| `panda_doc_bm25` | [panda_doc_bm25.md](panda_doc_bm25.md) | BM25 keyword search over the same ATLAS / ePIC corpus. |
| `cgsim.doc_search` | [cgsim_doc_search.md](cgsim_doc_search.md) | Vector similarity search over the CGSim / SimGrid corpus. |
| `cgsim.doc_bm25` | [cgsim_doc_bm25.md](cgsim_doc_bm25.md) | BM25 keyword search over the same CGSim corpus. |

---

## CGSim simulation data tools

These tools query the SQLite database produced by a CGSim simulation run.

| Tool | Document | Description |
|---|---|---|
| `cgsim.sim_query` | [cgsim_sim_query.md](cgsim_sim_query.md) | Natural-language to SQL against the CGSim EVENTS database. |

---

## Source code analysis tools

These tools fetch and analyse PanDA Pilot source code from GitHub.

| Tool | Document | Description |
|---|---|---|
| `pilot_source_analysis` | [pilot_source_analysis.md](pilot_source_analysis.md) | Traceback-driven analysis — fetches pilot3 functions named in a job failure exception. |
| `atlas.core_dump_analysis` | [core_dump_analysis.md](core_dump_analysis.md) | **ATLAS only.** gdb against a failed job's core dump inside the matching release container; answers what the payload was doing when it was killed. |
| `code_query` | [code_query.md](code_query.md) | **Superuser.** On-demand fetch of any source file or function from a configurable repository; targeted Q&A, algorithm explanation, Mermaid diagrams. |

---

## Self-observability tools

These tools query Bamboo's own prompt/response log index in OpenSearch.

| Tool | Description |
|---|---|
| `opensearch_query` | General-purpose read-only DSL query against any allowed OpenSearch index. Accepts `index_pattern`, `query` (JSON DSL string), `max_hits`, `source_fields`. |
| `opensearch_promptlog_query` | Convenience wrapper pre-wired to `bamboomcp-promptlog-*`. Schema-aware: documents turn counts, session replay, FAQ analysis, tool usage, ratings. Uses `raw_question.keyword` for accurate frequency aggregations. |
| `bamboo_promptlog_status` | Drains the server-side event ring buffer of OpenSearch write notifications (destructive read, delivered exactly once). Used by TUI and Streamlit to surface write confirmations. |
| `bamboo_promptlog_rate` | Updates the `rating` field (1–5) on an existing prompt-log document via partial OpenSearch `update`. Uses the write credential (`BAMBOO_OPENSEARCH_PROMPTLOG`). |

Environment variables: `ASKPANDA_OPENSEARCH` (read), `BAMBOO_OPENSEARCH_PROMPTLOG` (write),
`BAMBOO_OPENSEARCH_ALLOWED_INDICES` (read allow-list, default: `atlas_harvesterworkers-*,bamboomcp-promptlog-*`).

See [`docs/opensearch.md`](../opensearch.md) for the full schema, DSL examples, and rating query patterns.

---

## Infrastructure tools

| Tool | Document | Description |
|---|---|---|
| `bamboo_health` | [bamboo_health.md](bamboo_health.md) | Bamboo server health — version, LLM config, integration flags. |
| `bamboo_llm_answer` | [bamboo_llm_answer.md](bamboo_llm_answer.md) | Direct LLM passthrough — no routing or data tools. |

---

## Code-mode primitive surface

**ATLAS only, and not advertised by default.** These five tools decompose
`panda_log_analysis` into composable steps for an agentic framework that
writes code against small primitives rather than selecting one compound tool
per turn. A server advertises them instead of the monolith when
`BAMBOO_TOOL_PROFILE=primitive`, and alongside it under `both`; the default
(`orchestrated`) withholds them entirely, so they never enter Bamboo's own
planner catalog.

They are documented together rather than one per file: they are only
meaningful composed, and the loop, the budgets and the equivalence contract
with `panda_log_analysis` belong in one place.

| Tool | Description |
|---|---|
| `atlas.log.fetch_metadata` | Job metadata subset — status, site, error codes and diagnoses, timing, the pilot version implied by `pilotid`. Facts, not decisions. |
| `atlas.log.plan_fetch` | Which log files to download next, and in what order. Re-entrant: hand back the `signals` from each fetch as `observed`. At most three calls. |
| `atlas.log.fetch_text` | Downloads one file and returns its diagnostic excerpt, budgeted by the role `plan_fetch` assigned, plus the signals the next plan needs. |
| `atlas.log.classify` | The verdict, from the metadata and the fetched excerpts. Pure — no network, safe to call twice. |
| `atlas.log.list_files` | The job's log tarball with sizes, job-root files first. Not needed to diagnose a failure. |

See [`docs/code-mode.md`](../code-mode.md) for the loop, the output shapes,
`BAMBOO_TOOL_PROFILE`, `BAMBOO_PRIMITIVE_MAX_CHARS`, and the equivalence
contract — including the three differences from `panda_log_analysis` that are
intended.

---

## Stub tools (not production)

These tools exist in the codebase but are not connected to live data. They are superseded by the production tools listed above.

| Tool | Document | Superseded by |
|---|---|---|
| `panda_pilot_status` | [panda_pilot_status.md](panda_pilot_status.md) | `panda_harvester_workers` |
| `panda_queue_info` | [panda_queue_info.md](panda_queue_info.md) | `cric_query` |

---

## Plugin differences

### Tools only in `askpanda_atlas`

The following tools have no ePIC equivalent and are absent from the `askpanda_epic` package:

| Tool | Reason |
|---|---|
| `panda_harvester_workers` | ATLAS-specific Harvester deployment |
| `panda_harvester_timeseries` | ATLAS OpenSearch cluster (`os-atlas.cern.ch`) |
| `panda_jobs_query` | ATLAS-specific ingestion database |
| `cric_query` | CRIC is an ATLAS computing resource catalogue |
| `panda_server_health` | ATLAS PanDA MCP session wiring |
| `atlas.log.*` (five primitives) | Code-mode surface; nothing would replace `panda_log_analysis` on an ePIC server, so the ePIC copy stays profile-agnostic |

`code_query` and `pilot_source_analysis` are built-in core tools (not plugin-specific) and are available to all experiments.

### `panda_task_status` implementation differences

| Aspect | `askpanda_atlas` | `askpanda_epic` |
|---|---|---|
| Endpoints fetched | Two: `/jobs/?jeditaskid=` **and** `/task/<id>/` | One: `/jobs/?jeditaskid=` only |
| Evidence: `failed_pandaids` | Present — flat list of failed job IDs | Absent |
| Evidence: task metadata fields | `status`, `superstatus`, `taskname`, `username`, `creationdate` from task endpoint | Not available (jobs endpoint only) |
| Tracing | `_trace()` writes to `BAMBOO_TRACE_FILE` | Not present |

### `panda_log_analysis` differences

| Aspect | `askpanda_atlas` | `askpanda_epic` |
|---|---|---|
| Monitor link label | `BigPanDA Monitor` | `PanDA Monitor` |
| Tool tags | `"atlas"`, `"bigpanda"` | `"epic"`, `"eic"` |

The log extraction, failure classification, stderr fetching, and evidence structure are identical in both packages.

### `cgsim` plugin tools

The `cgsim` package provides documentation search and simulation database query
tools for the CGSim / SimGrid distributed computing simulator.

| Entry point | Tool name | Description |
|---|---|---|
| `cgsim.doc_search` | `cgsim.doc_search` | Vector similarity search over CGSim / SimGrid documentation |
| `cgsim.doc_bm25` | `cgsim.doc_bm25` | BM25 keyword search over the same corpus |
| `cgsim.ui_manifest` | `cgsim.ui_manifest` | TUI branding (banner, accent `green`, display name) |
| `cgsim.sim_query` | `cgsim.sim_query` | Natural-language to SQL against the CGSim simulation output SQLite database |

Set `ASKPANDA_PLUGIN=cgsim` and `CGSIM_DB_PATH=/path/to/cgsim.db` when
running a CGSim deployment.  Set `BAMBOO_CHROMA_COLLECTION=cgsim_docs` if
using the RAG documentation tools.

### Entry point naming

All plugin tools are loaded via Python entry points. The MCP server overwrites `get_definition()["name"]` with the entry point key at load time — for example, `panda_harvester_timeseries` is registered as `atlas.harvester_timeseries` and must be called by that name.

---

## Superuser tools

Tools tagged `superuser` in their definition are always registered on the MCP server and callable by any MCP client. The Streamlit and TUI interfaces use the `BAMBOO_SUPERUSER_PASSWORD` env var to gate their evidence panels in non-authenticated sessions — this is a UI convenience, not a security boundary.

| Tool | Tag | UI behaviour |
|---|---|---|
| `code_query` | `superuser`, `developer` | Evidence and Raw JSON panels hidden until superuser unlock |

See [`docs/interfaces.md`](../interfaces.md#superuser-mode) for the full superuser setup guide.
