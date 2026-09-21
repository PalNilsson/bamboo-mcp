# MCP workflow, tool orchestration, and LLM usage

## What MCP provides

Bamboo runs an MCP server that publishes a catalog of tools and executes tool
calls over JSON-RPC 2.0.  Two transports are supported:

- **stdio** (default, preferred) — the server is spawned as a subprocess; the
  TUI and MCP Inspector use this mode.
- **Streamable HTTP** — the server listens on an HTTP endpoint; the Streamlit
  UI and multi-user deployments use this mode.

## How LLMs are used

Bamboo's own UIs (TUI and Streamlit) call `bamboo_answer` directly by name.
`bamboo_answer` is a server-side orchestrator: it decides which evidence tools
to call, calls them, and then uses an LLM to turn the results into a
natural-language answer.

There are three places in this pipeline where an LLM may be invoked.  Each is
optional — the common case requires only the third.

### 1. Topic guard (optional, fast model)

Before any tool call, a two-stage guard checks whether the question is
on-topic for PanDA/ATLAS.

- **Stage 1 — keyword check** (free, synchronous): the question is matched
  against allow and deny term lists.  Most domain questions are allowed
  immediately by a keyword hit (`reason=keyword_allow`).
- **Stage 2 — LLM classifier** (fast model, ~5 output tokens): fires only
  when Stage 1 cannot reach a verdict — e.g. a short or ambiguous question
  with no domain keywords.  The model replies with a single word, `ALLOW` or
  `DENY`.

Two additional cases bypass the guard entirely:

- **Content-free follow-ups** ("Tell me more", "Elaborate") — trivially
  on-topic when prior history exists; recorded as `reason=followup_allow`.
- **Keyword allow** — as above, no LLM needed.

If the LLM classifier fails for any reason, the guard fails open (the
question is allowed through) to avoid blocking legitimate users.

### 2. Tool selection (optional, default model)

`bamboo_answer` decides which evidence tool(s) to call using one of two
mechanisms, tried in order:

**Deterministic routing** — `_build_deterministic_plan()` inspects the
question for a task ID, a job ID, or failure-analysis keywords using regex
patterns.  This produces a `Plan` object.  For combined site-health questions,
the two-tool plan is built directly in `_run_fast_path_intercepts()` rather
than `_build_deterministic_plan()`.  The full priority order is:

| Question contains | Tool(s) called | Route |
|---|---|---|
| job ID + failure keywords (analyse, why, fail, log…) | `panda_log_analysis` | `FAST_PATH` |
| job ID (no task ID) | `panda_job_status` | `FAST_PATH` |
| task ID | `panda_task_status` | `FAST_PATH` |
| pilot signals + job-specific signals, no IDs | `panda_harvester_workers` + `panda_jobs_query` | `FAST_PATH` |
| pilot signals only, no IDs | `panda_harvester_workers` | `FAST_PATH` |
| jobs DB signals only, no IDs | `panda_jobs_query` | `FAST_PATH` |
| neither | `panda_doc_search` + `panda_doc_bm25` | `RETRIEVE` |

**No LLM is involved here.** This covers the overwhelming majority of
questions with zero routing cost.

The site-health row (fourth) fires when a question contains signals from both
`_PILOT_SIGNALS` and `_JOBS_DB_SPECIFIC_SIGNALS` — a job-specific subset that
excludes generic phrases like `"how many"` and `"count"` to avoid false
positives on pure pilot questions such as `"how many pilots are running?"`.
Questions with `"task"` are excluded from site-health routing.

The pilot and site-health checks both bypass the topic guard entirely (saving
~3 s).  Contextual ID resolution from history runs first so pronouns like
`"how many of those failed?"` still route correctly to `panda_task_status`.

**LLM planner fallback** — if the deterministic step cannot produce a plan
(currently reserved for future multi-step questions), `bamboo_plan` is called.
This uses the default LLM to select tools from a curated catalog (internal
infrastructure tools are excluded) guided by explicit routing rules in the
system prompt.  In practice this path is not triggered by normal usage today.

### 3. Answer synthesis (always, default model)

After the evidence tool(s) have run, the LLM synthesises a natural-language
answer.  It receives:

- A route-specific system prompt (different rules for RAG documentation
  questions, task status, job status, and log analysis)
- Prior conversation history, with long assistant messages truncated to
  400 characters to prevent prompt growth across multi-turn conversations
- A synthesised user message containing the question and the tool evidence

For content-free follow-ups ("Tell me more"), the user message uses
**expansion framing** — the LLM is told to go deeper than the prior answer
with a 200-300 word target, and `max_tokens` defaults to 600 to keep
response time predictable.  For all other questions `max_tokens` defaults
to 2048.  Both values are configurable:

```bash
export BAMBOO_SYNTHESIS_MAX_TOKENS=4096   # fresh questions (default 2048)
export BAMBOO_FOLLOWUP_MAX_TOKENS=1024    # follow-up expansions (default 600)
```

The LLM's role here is purely presentational: it summarises and explains
evidence but does not select tools, invent facts, or call further tools.

---

### Comparison with standard MCP tool selection

In standard MCP usage a *host LLM* (e.g. Claude Desktop, Cursor) calls
`tools/list` and uses the tool descriptions to decide which tool to call.
**Bamboo's server fully supports this** — the catalog is published and an
external host can drive everything.

Bamboo's own UIs differ in two ways: tool selection happens server-side (not
in the client), and the common case uses deterministic regex routing rather
than an LLM at all.  When the LLM planner is invoked as a fallback, it is
constrained by routing rules rather than left to reason freely from tool
descriptions alone.



---

## Tool discovery

Tools are discovered via Python entry points in the `bamboo.tools` group (with
an optional legacy fallback to `askpanda.tools`).

Example entry point:

```toml
[project.entry-points."bamboo.tools"]
"atlas.task_status"       = "askpanda_atlas.task_status:panda_task_status_tool"
"atlas.harvester_workers" = "askpanda_atlas.harvester_worker:panda_harvester_workers_tool"
```

Built-in tools are also registered directly in `bamboo.core.TOOLS`.

> **`panda_harvester_workers`** is registered as a plugin tool via the
> `askpanda_atlas` package entry point `atlas.harvester_workers`.  It fetches
> live Harvester pilot/worker counts from the BigPanDA API endpoint
> `/harvester/getworkerstats/`.  See the
> [CLAUDE.md Harvester Workers section](../CLAUDE.md#harvester-workers-harvester_worker_implpy)
> for full implementation details.

### Tool profiles

Which discovered tools are *advertised* is filtered by `BAMBOO_TOOL_PROFILE`
(`orchestrated`, the default; `primitive`; or `both`), read per `tools/list`.
A definition may restrict itself with a `profiles` key; one that names no
profile is advertised under all of them, which is every tool in the tree
except the log-analysis compound/primitive pair.

The filter is **advertising only** — `call_tool` is not gated by profile, and
neither is `bamboo_executor`'s in-process resolution nor the `_tool_names`
alias map — and **fail-open**: an unrecognised profile name, in the variable
or in a definition, warns and degrades rather than making a tool vanish from
the listing.

Because the variable is read at listing time, a tool definition may differ
between listings. `panda_log_analysis` rebuilds its own per call for exactly
that reason; anything caching a definition across a profile change is wrong.

See [`docs/code-mode.md`](code-mode.md) for the surface this exists to serve.

## Tool execution contract

Each tool provides:

- `get_definition() -> dict` — MCP tool definition including `name`,
  `description`, and `inputSchema` (JSON Schema with
  `"additionalProperties": false`).
- `async call(arguments: dict) -> list[MCPContent]` — returns a non-empty
  list of MCP content dicts, always with at least `{"type": "text", "text": "..."}`.

Tools that carry structured evidence (task status, job status, log analysis)
JSON-serialise their payload into the `text` field:

```python
return text_content(json.dumps({"evidence": {...}, "text": "Summary..."}))
```

Callers that need the raw evidence parse it back with
`json.loads(result[0]["text"])`.  The executor does this automatically via
`unpack_tool_result()` in `bamboo.tools.bamboo_executor`.

**Tools must never raise.** All error conditions are returned as
`text_content(error_message)` so the MCP client always receives a well-formed
response.

### Tools that declare an `outputSchema`

The five `atlas.log.*` code-mode primitives are the exception to the shape
above.  Each declares an `outputSchema` and returns the two-element
`(content, structured)` tuple the SDK validates against it, so a client reads
`structuredContent` rather than re-parsing JSON out of a text block.  This
requires `mcp >= 1.10.0`: below that floor the SDK ignores `outputSchema` and
drops the structured half *silently*.

Two consequences, both deliberate:

- **No such schema carries a top-level `required`, and every one declares
  `error`.**  The SDK rejects a result with no structured content from a tool
  advertising an `outputSchema`, so a failure has to be expressible under the
  schema too; one that demanded the success keys would turn every error path
  into an opaque *"Output validation error"*.  `_validate_arguments` failures
  in `core.py` carry structured content for the same reason.
- **They do not reach `unpack_tool_result`.**  That unpacker expects the
  list-of-content-dicts shape above.  The primitives are not planner-visible,
  so nothing in Bamboo's own pipeline unpacks them; a code-mode client reads
  the structured half directly.

See [`docs/code-mode.md`](code-mode.md).

## Server-side argument validation

Before dispatching to a tool, `core.py`'s `call_tool` handler runs
`_validate_arguments()` against the tool's `inputSchema`.  This checks:

- `anyOf` satisfaction (e.g. `question` OR `messages` required)
- Required fields present and non-null
- No unknown keys when `additionalProperties: false`

Validation failures return a descriptive `text_content` error rather than
raising, so clients always receive a response.

## Orchestration model

Bamboo supports two complementary orchestration styles, and the tool catalog
serves both:

1. **External-LLM-driven (standard MCP)**: a host LLM (Claude Desktop,
   Cursor, or any MCP client) reads `tools/list`, uses the tool `name`,
   `description`, and `inputSchema` to decide which tool to call, and sends
   a `tools/call` request directly.  The individual tools (`panda_task_status`,
   `panda_job_status`, `panda_log_analysis`, etc.) are designed to be called
   this way.  Their schemas are strict (`additionalProperties: false`,
   `required` fields declared) precisely so an external LLM can reason about
   them reliably.

2. **Server-driven (`bamboo_answer`)**: Bamboo's own UIs call `bamboo_answer`
   directly.  `bamboo_answer` selects and calls the individual tools itself
   using the routing pipeline described below.

> **Note on tool descriptions**: because the tool catalog is exposed to
> external LLM clients, the `description` field of each tool definition is
> part of the public contract.  Descriptions should be written from the
> perspective of an LLM deciding whether to call a tool, not from an
> implementer's perspective.

### bamboo_answer routing pipeline

`bamboo_answer` is the primary entry point for both UIs.  It accepts
`question` (string), `messages` (full chat history for multi-turn context),
and `bypass_routing` (flag to skip routing and go directly to the LLM),
and `bypass_fast_path` (flag to skip only the deterministic fast-path
intercepts and `_build_deterministic_plan`, falling through to the topic
guard and LLM planner — useful for testing planner routing on questions that
would normally be short-circuited).  Exposed as the TUI `/fastpath off` command.

**Step 1 — Argument validation** (in `core.py`, before dispatch)

The `inputSchema` is checked against the supplied arguments.

**Step 2 — History extraction**

Prior user/assistant turns are extracted from `messages` via
`_extract_history()`, stripping system messages and the current question so
they can be injected into later LLM calls.

**Step 3 — Follow-up detection and RAG query reformulation**

`_is_content_free_followup()` checks whether the question is a content-free
phrase ("Tell me more", "Elaborate", "Go on", etc.).  When it matches and
history is present:

- The topic guard (Step 4) is bypassed entirely, recorded as
  `reason=followup_allow` in the trace.
- The RAG query (`rag_query`) is substituted with the last meaningful user
  question from history via `_last_user_question()` so retrieval targets the
  actual topic rather than the follow-up phrase.
- `original_question` is set to the user's literal phrasing and passed
  through to the synthesis step, which uses expansion framing.

**Step 4 — Topic guard** (`bamboo.tools.topic_guard`)

Skipped for content-free follow-ups (Step 3).  Otherwise, a two-stage guard
checks the question: keyword allow/deny first, then LLM classifier for
ambiguous cases.  If the question is rejected, a polite message is returned
immediately and no downstream calls are made.

**Step 5 — Deterministic routing** (`_build_deterministic_plan()`)

Regex patterns extract `task_id` and `job_id` from `rag_query`.  The
combination of IDs and keywords builds a `Plan` object with no LLM call:

| Condition | Route | Tool(s) called |
|---|---|---|
| job_id + analysis keywords | `FAST_PATH` | `panda_log_analysis` |
| job_id only (no task_id) | `FAST_PATH` | `panda_job_status` |
| task_id | `FAST_PATH` | `panda_task_status` |
| pilot + job-specific signals, no IDs | `FAST_PATH` | `panda_harvester_workers` + `panda_jobs_query` |
| pilot signals only, no IDs | `FAST_PATH` | `panda_harvester_workers` |
| jobs DB signals only, no IDs | `FAST_PATH` | `panda_jobs_query` |
| no ID | `RETRIEVE` | `panda_doc_search` (top_k=5) + `panda_doc_bm25` (top_k=5) |

This covers all common questions with **zero LLM cost**.

**Step 6 — LLM planner fallback** (`bamboo_plan`)

Reserved for questions where `_build_deterministic_plan()` returns `None`.
Currently unreachable in normal usage but available for future multi-step or
ambiguous question patterns.  The planner receives a tool catalog that excludes
internal infrastructure tools (`bamboo_llm_answer`, `bamboo_answer`,
`bamboo_plan`, `bamboo_health`) and routing guidance in its system prompt.

**Step 7 — Plan execution** (`bamboo.tools.bamboo_executor.execute_plan()`)

The executor iterates the plan's `tool_calls` in order.  For each call it:

1. Resolves the tool from `TOOLS` or plugin entry points.
2. Validates arguments against the tool's `inputSchema`.
3. Calls `await tool.call(args)` and unpacks JSON evidence.
4. Accumulates evidence strings for the synthesis step.

Unknown tools, validation failures, and tool exceptions are handled
gracefully — partial evidence from successful calls is still synthesised.
Only when all calls fail is a top-level error returned.

An `EVENT_PLAN` trace span is emitted at the start of execution containing
the full plan JSON, which the TUI's `/plan` command reads.

**Step 8 — LLM synthesis** (`bamboo.tools.bamboo_executor.call_llm()`)

See [Answer synthesis](#answer-synthesis) above.  The synthesis prompt is
selected by `_pick_synthesis_prompt()` based on which tools ran:

| Tools called | System prompt used |
|---|---|
| `panda_log_analysis` | Log-analysis diagnostic prompt |
| `panda_job_status` | Job-status summary prompt |
| `panda_task_status` | Task-metadata summary prompt |
| `panda_harvester_workers` + `panda_jobs_query` | Site-health prompt (two labelled evidence sources: pilots and jobs) |
| `panda_harvester_workers` alone | Pilot stats prompt (pivot table, flat breakdowns, time window) |
| `panda_jobs_query` alone | Jobs DB prompt (explicitly tells the LLM `"error": null` means success) |
| `panda_doc_search` or `panda_doc_bm25` | RAG documentation prompt |
| other / mixed | Generic multi-tool prompt |

When `original_question` is set (content-free follow-up), the user message
uses expansion framing with a 200-300 word target and `max_tokens` defaulting
to 600 (configurable via `BAMBOO_FOLLOWUP_MAX_TOKENS`).

### bamboo_plan (LLM-backed planner)

`bamboo_plan` is a standalone MCP tool that can be called directly by
external clients or used internally by `bamboo_answer` as a routing fallback.

When called with `execute=False` (default) it returns a validated JSON plan.
When called with `execute=True` it executes the plan via `execute_plan()` and
returns a synthesised answer directly.

The planner system prompt includes routing guidance:

- task ID present → `panda_task_status`
- job ID + failure keywords → `panda_log_analysis`
- job ID alone → `panda_job_status`
- **pilot AND job signals** at a site → `panda_harvester_workers` + `panda_jobs_query` together (pass `site=` and `queue=`)
- **pilot signals only** → `panda_harvester_workers`
- **job/failure signals only** → `panda_jobs_query`
- **queue configuration** → `panda_queue_info`
- all other questions → `panda_doc_search` + `panda_doc_bm25` (always
  retrieve first; never answer general knowledge questions from parametric
  memory alone)

Infrastructure tools are excluded from the catalog to prevent the planner
from routing general questions to the raw LLM passthrough.

The planner validates output against the `Plan` Pydantic model and performs
at most one repair attempt.  If both attempts fail, a structured error is
returned (not raised).

Each planner LLM call now emits an `llm_call` trace span with
`tool="bamboo_plan"`, so token counts appear correctly in the TUI `/costs`
output when the planner is active (i.e. when `/fastpath off` is set or for
questions that bypass the deterministic fast-path).

The authoritative plan schema:

```python
from bamboo.tools.planner import get_plan_json_schema
schema = get_plan_json_schema()
```

## Context memory (multi-turn chat)

Both UIs maintain an in-memory conversation history (capped at
`BAMBOO_HISTORY_TURNS` user+assistant pairs, default 10).  On each question
the full history is sent as `messages` to `bamboo_answer`.  The server
extracts prior turns and injects them into the synthesis LLM call.  The
server is stateless — history lives in the client.

To prevent synthesis prompts from growing unbounded across long conversations,
long assistant messages in the injected history are truncated at
`_HISTORY_ASSISTANT_MAX_CHARS` characters (default 400).  This keeps a
sufficient excerpt for follow-up resolution without bloating the prompt.

## Sequence diagram

<img src="images/mcp_sequence_diagram.png" alt="Diagram" width="100%" />

---

## See also

- [`docs/architecture.md`](architecture.md) — process boundary, MCP wire, and `bamboo_answer` routing flow with diagrams
- [`docs/code-mode.md`](code-mode.md) — the `atlas.log.*` primitive surface, `BAMBOO_TOOL_PROFILE`, and the equivalence contract with `panda_log_analysis`
