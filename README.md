# Bamboo MCP

**Bamboo MCP** is a lightweight MCP-based runtime with a plugin architecture for
AI-assisted scientific tools, targeting PanDA/ATLAS workflows, ePIC/EIC
experiment operations, and CGSim distributed computing simulation.

LLMs are used for *summarisation and explanation*, not as sources of truth.
Structured evidence is always returned alongside natural-language answers.

> **Status (August 2026):** core infrastructure is stable; latest release
> **v1.0.8**. The newest addition is `atlas.core_dump_analysis`, which runs gdb
> against a failed ATLAS job's core dump inside the matching release container
> and reports what the payload was actually doing when it was killed — the
> question log analysis structurally cannot answer, since a looping-job kill
> happens precisely because the payload stopped producing output.
>
> Earlier additions include a multi-step AI Agent (`scripts/bamboo_agent.py`)
> for complex multi-hop queries,
> full OpenSearch self-observability — Bamboo logs every prompt/response turn and
> can query its own logs for FAQ analysis, session replay, tool-usage analytics,
> and per-model token cost breakdowns. Users can rate responses (1–5 stars) from
> both the TUI (`/rate N`) and Streamlit (star buttons). LaTeX formulas render
> natively in Streamlit. Slash commands (`/faq`, `/rates`, `/script`, `/rate`)
> are available in both interfaces. The `code_query` developer tool fetches and
> analyses source files from any GitHub repository (`BAMBOO_CODE_QUERY_REPO`);
> the `cric_query` tool answers natural-language questions about ATLAS queue and
> site configuration; `cgsim.sim_query` queries CGSim simulation output.

---

## Contributing

### Repository setup

The canonical repository is at **https://github.com/BNLNPPS/bamboo-mcp**. Development follows a standard fork-and-pull-request workflow.

**First-time setup:**

```bash
# Clone your fork
git clone https://github.com/<your-username>/bamboo-mcp.git
cd bamboo-mcp

# Add the canonical repo as upstream
git remote add upstream https://github.com/BNLNPPS/bamboo-mcp.git

# Verify
git remote -v
# origin    https://github.com/<your-username>/bamboo-mcp.git (fetch)
# origin    https://github.com/<your-username>/bamboo-mcp.git (push)
# upstream  https://github.com/BNLNPPS/bamboo-mcp.git (fetch)
# upstream  https://github.com/BNLNPPS/bamboo-mcp.git (push)
```

**Day-to-day workflow:**

```bash
# Push your changes to your fork
git push origin master

# Open a pull request from your fork to BNLNPPS/bamboo-mcp via GitHub

# Keep your fork in sync with upstream
git fetch upstream
git merge upstream/master
```

---

## Quick start

### 1. Create a virtual environment

```bash
python3 -m venv ~/Development/venv-bamboo
source ~/Development/venv-bamboo/bin/activate
```

### 2. Install the packages

```bash
# Core MCP server — required
pip install -r requirements.txt
pip install -e ./core

# ATLAS / PanDA plugin
pip install -e ./packages/askpanda_atlas

# ePIC / EIC plugin
pip install -e ./packages/askpanda_epic

# AskCGSim plugin
pip install -e ./packages/askcgsim

# Root package — required for the TUI and Streamlit UI
pip install -e .

# TUI interface
pip install -r requirements-textual.txt

# Streamlit web UI
pip install -r requirements-ui.txt

# HTTP server (for shared/testbed deployments)
pip install -r requirements-http.txt

# RAG tools (ChromaDB vector search + BM25)
pip install -r requirements-rag.txt
```

Install **one** LLM provider (Mistral is the default):

```bash
pip install -r requirements-mistral.txt    # Mistral (default)
pip install -r requirements-openai.txt     # OpenAI / OpenAI-compatible
pip install -r requirements-anthropic.txt  # Anthropic
pip install -r requirements-gemini.txt     # Google Gemini
```

See [`docs/developer.md`](docs/developer.md) for the full list of optional
feature packages (tracing, Streamlit UI, etc.).

### 3. Configure environment

```bash
cp bamboo_env_example.sh bamboo_env.sh
# Edit bamboo_env.sh: set your API key and preferred provider/model
source bamboo_env.sh
```

The minimum you need to set:

```bash
export LLM_DEFAULT_PROVIDER="mistral"          # or openai, anthropic, gemini
export LLM_DEFAULT_MODEL="mistral-large-latest"
export MISTRAL_API_KEY="your-key-here"         # whichever provider you chose
```

### 4. Launch

**Textual TUI (stdio — recommended for local use):**

```bash
# From core/ directory
cd core
python ../interfaces/textual/chat.py --transport stdio --no-inline
```

**Streamlit web UI:**

```bash
cd core
streamlit run ../interfaces/streamlit/chat.py
```

**HTTP server (shared / testbed deployments):**

```bash
# Install deps first: pip install -r requirements-http.txt
python -m bamboo.server_http --host 0.0.0.0 --port 8000
# MCP endpoint: http://<host>:8000/mcp
# Health check: curl http://<host>:8000/healthz
```

Then connect the TUI or Streamlit to the running server:

```bash
export MCP_URL="http://<host>:8000/mcp"
python interfaces/textual/chat.py --transport http
```

See [`docs/http-server.md`](docs/http-server.md) for auth, firewall, and
persistent-mode configuration.

**AI Agent (multi-step reasoning, requires HTTP server):**

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --question "Which sites had the highest pilot failure rate today?" \
    --verbose
```

See [`docs/agent.md`](docs/agent.md) for options, tuning, and testing instructions.

**Running in AskCGSim mode:**

```bash
export ASKPANDA_PLUGIN="cgsim"
export BAMBOO_FAST_PATH="0"             # recommended for CGSim
export BAMBOO_CHROMA_COLLECTION="cgsim_docs"
export BAMBOO_CHROMA_PATH="/path/to/chromadb-cgsim"
cd core
python ../interfaces/textual/chat.py --transport stdio --no-inline
```

Type any question and press Enter.

---

## TUI slash commands

| Command | What it does |
|---|---|
| `/help` | Show all commands |
| `/task <id>` | Shorthand for "summarise task \<id\>" |
| `/job <id>` | Shorthand for "analyse failure of job \<id\>" |
| `/faq [today\|week\|month]` | Most frequently asked questions from prompt logs (default: all time) |
| `/rates [today\|week\|month]` | Rated responses as a markdown table (default: all time) |
| `/rate <1-5>` | Rate the last response (1 = poor 🔴, 5 = excellent 💚) |
| `/script [filename]` | Write code block(s) from the last response to a local file |
| `/tracing` | Show timing and trace spans for the last request |
| `/costs` | Show estimated LLM token cost for the last request |
| `/json` | Show raw BigPanDA JSON for the last query |
| `/inspect` | Show compact evidence dict (what the LLM saw) for the last query |
| `/history` | Show turns currently held in context memory |
| `/fastpath on\|off` | Toggle deterministic fast-path routing (off → use LLM planner) |
| `/debug on\|off` | Toggle verbose tool call output |
| `/tools` | List tools registered on the server |
| `/links [N]` | List links from the last response; `/links N` opens link N in browser |
| `/superuser <pw>` | Unlock developer mode (requires `BAMBOO_SUPERUSER_PASSWORD`) |
| `/clear` | Clear transcript, context memory, and HTTP cache |
| `/exit` | Quit |

`PageUp`/`PageDown` to scroll · `Ctrl+Q` to quit ·
Hold **Option** (macOS) or **Shift** (Linux/Windows) to select text with the mouse.

See [`docs/question-cheatsheet.md`](docs/question-cheatsheet.md) for ready-to-paste test questions.

---

## Streamlit slash commands

All slash commands are available in the Streamlit chat input box too:

| Command | What it does |
|---|---|
| `/help` or `/?` | Show available commands inline |
| `/faq [today\|week\|month]` | Most frequently asked questions from prompt logs |
| `/rates [today\|week\|month]` | Rated responses as a markdown table |
| `/rate <1-5>` | Rate the last response |
| `/script` | Show instructions for downloading code blocks |
| `/task <id>` | Shorthand for "summarise task \<id\>" |
| `/job <id>` | Shorthand for "analyse failure of job \<id\>" |

The Streamlit interface also shows **⬇ Download** button(s) automatically below
any response that contains a fenced code block, and **star rating buttons**
(🔴 1 – 💚 5) after every response. No slash command is needed for either.

---

## Self-observability

Bamboo logs every prompt/response turn to an OpenSearch index
(`bamboomcp-promptlog-YYYY.MM.DD`). The following questions are answered
directly from the prompt log without touching PanDA or documentation tools:

- "How many turns have I had today?"
- "What are the most frequently asked questions?" → `/faq`
- "Show all rated responses this week" → `/rates week`
- "Which tools were used most often today?"
- "Show me the full response for doc id `<id>`"
- "What is the average rating per model?"

Ratings (1–5 stars) are stored on each logged document and queryable via
`opensearch_promptlog_query`. See [`docs/opensearch.md`](docs/opensearch.md)
for the full schema, DSL query examples, and architecture diagram.

The `raw_question` field stores the user's original question as typed — use
`raw_question.keyword` for accurate frequency aggregations (FAQ analysis).
The `user_prompt` field stores the full synthesis prompt (question + evidence
context) and is unsuitable for frequency analysis.

---

## Key ideas

- **Tool-first** — tools are authoritative; LLMs only summarise their output
- **Plugin architecture** — experiment-specific logic lives in plugins, not in core
- **Narrow waist** — every tool returns `list[MCPContent]`; the MCP wire format is JSON-RPC 2.0
- **Context memory** — multi-turn chat history is maintained in the client and threaded into every LLM call
- **Configurable routing** — `bamboo_answer` uses deterministic fast-path routing by default; set `BAMBOO_FAST_PATH=0` to route all questions through the LLM planner (recommended for CGSim)
- **Superuser / developer mode** — set `BAMBOO_SUPERUSER_PASSWORD` to enable a password-protected developer tier in both UIs; unlocks `code_query` and future developer tools

---

## Inspecting and running the server

```bash
# List available tools
python -m bamboo tools list

# Start the MCP server (stdio — spawned automatically by TUI)
python -m bamboo.server

# Start the HTTP server (shared deployments)
python -m bamboo.server_http --host 0.0.0.0 --port 8000

# Health check (HTTP server only)
curl http://localhost:8000/healthz   # → ok

# Interactive inspection via MCP Inspector (stdio)
npx @modelcontextprotocol/inspector python3 -m bamboo.server

# Interactive inspection via MCP Inspector (HTTP)
npx @modelcontextprotocol/inspector --url http://localhost:8000/mcp
```

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/developer.md`](docs/developer.md) | Full setup, editable installs, testing, linting |
| [`docs/http-server.md`](docs/http-server.md) | Running the HTTP server for shared/testbed deployments |
| [`docs/rest-api.md`](docs/rest-api.md) | REST analysis API — endpoints, polling, caching, budgets, PanDA monitor integration |
| [`docs/mcp.md`](docs/mcp.md) | MCP protocol, tool contracts, LLM roles, orchestration |
| [`docs/architecture.md`](docs/architecture.md) | Process boundary, MCP wire, `bamboo_answer` routing flow |
| [`docs/interfaces.md`](docs/interfaces.md) | TUI, Streamlit UI, HTTP transport, context memory |
| [`docs/plugins.md`](docs/plugins.md) | Writing and registering plugins |
| [`docs/jobs-database.md`](docs/jobs-database.md) | Live PanDA jobs DB queries — schema, examples, guard rules, routing |
| [`docs/cric-database.md`](docs/cric-database.md) | CRIC queuedata queries — schema, examples, guard rules, routing, disambiguation |
| [`docs/cgsim-database.md`](docs/cgsim-database.md) | CGSim simulation DB queries — EVENTS schema, METADATA fields, example SQL, security model |
| [`docs/harvester-workers.md`](docs/harvester-workers.md) | Harvester pilot/worker counts — API, evidence structure, routing, time windows |
| [`docs/rag.md`](docs/rag.md) | RAG pipeline (ChromaDB + BM25) |
| [`docs/tracing.md`](docs/tracing.md) | Structured tracing and OpenTelemetry |
| [`docs/opensearch.md`](docs/opensearch.md) | OpenSearch integration — prompt logging, read queries, self-observability, rating, GDPR pseudonymisation |
| [`docs/security.md`](docs/security.md) | Authentication and token management |
| [`docs/agent.md`](docs/agent.md) | AI Agent — multi-step reasoning loop, CLI options, testing, tuning |
| [`docs/question-cheatsheet.md`](docs/question-cheatsheet.md) | Ready-to-paste test questions, including code review question patterns and `code_query` examples |
| [`docs/tools/README-mcp_tools.md`](docs/tools/README-mcp_tools.md) | MCP tools reference — one document per tool, with inputs, outputs, routing, and design notes |
| [`docs/tools/code_query.md`](docs/tools/code_query.md) | `code_query` tool reference — evidence pipeline, routing, LLM quality guidance, example sessions |
| [`docs/tools/core_dump_analysis.md`](docs/tools/core_dump_analysis.md) | `atlas.core_dump_analysis` tool reference — execution model, synthesis boundary, routing rules 1c/1d |
| [`scripts/README-core_dump_analysis.md`](scripts/README-core_dump_analysis.md) | `analyze_core_dump.py` CLI — gdb invocation, executable resolution, LLM backends |

---

## Plugins

| Package | Status | Description |
|---|---|---|
| `askpanda_atlas` | Active | ATLAS / PanDA workflows |
| `askpanda_epic` | Active | ePIC / EIC experiment at BNL |
| `askpanda_verarubin` | Planned | Vera Rubin Observatory |
| `cgsim` | Active | CGSim / SimGrid distributed computing simulator |

### ATLAS plugin tools

| Entry point | Tool name | Description |
|---|---|---|
| `atlas.task_status` | `panda_task_status` | Task metadata and job-level detail |
| `atlas.log_analysis` | `panda_log_analysis` | Pilot/payload log download and failure classification |
| `atlas.core_dump_analysis` | `atlas.core_dump_analysis` | gdb analysis of a failed job's core dump, in the matching ATLAS release container |
| `atlas.doc_search` | `panda_doc_search` | Vector similarity search over ATLAS documentation |
| `atlas.doc_bm25` | `panda_doc_bm25` | BM25 keyword search over ATLAS documentation |
| `atlas.jobs_query` | `panda_jobs_query` | Natural language → SQL against the ingestion DuckDB |
| `atlas.harvester_workers` | `panda_harvester_workers` | Live Harvester pilot/worker counts |
| `atlas.harvester_timeseries` | `panda_harvester_timeseries` | Per-bucket pilot counts from OpenSearch (timeseries charts) |
| `atlas.panda_server_health` | `panda_server_health` | PanDA server liveness via PanDA MCP |
| `atlas.cric_query` | `cric_query` | Natural language → SQL against the CRIC queuedata DuckDB |
| `atlas.ui_manifest` | `atlas.ui_manifest` | TUI branding (banner, accent colour, display name) |

Set `BAMBOO_CHROMA_COLLECTION=atlas_docs` when running the ATLAS deployment to
point the doc tools at the ATLAS vector store.

### Built-in developer tools (all experiments)

| Tool name | Description |
|---|---|
| `code_query` | **Superuser.** Fetches any source file from a configurable GitHub repository for code review, algorithm explanation, and Mermaid diagram generation. Default repo: `PanDAWMS/pilot3`. Configured via `BAMBOO_CODE_QUERY_REPO` and `BAMBOO_CODE_QUERY_BRANCH`. |

### ePIC plugin tools

| Entry point | Tool name | Description |
|---|---|---|
| `epic.task_status` | `panda_task_status` | Task metadata and job-level detail |
| `epic.log_analysis` | `panda_log_analysis` | Pilot/payload log download and failure classification |
| `epic.doc_search` | `panda_doc_search` | Vector similarity search over ePIC documentation |
| `epic.doc_bm25` | `panda_doc_bm25` | BM25 keyword search over ePIC documentation |
| `epic.ui_manifest` | `epic.ui_manifest` | TUI branding (banner, accent colour, display name) |

Set `BAMBOO_CHROMA_COLLECTION=epic_docs` when running the ePIC deployment to
point the doc tools at the ePIC vector store.

### AskCGSim plugin tools

| Entry point | Tool name | Description |
|---|---|---|
| `cgsim.doc_search` | `cgsim.doc_search` | Vector similarity search over CGSim / SimGrid documentation |
| `cgsim.doc_bm25` | `cgsim.doc_bm25` | BM25 keyword search over CGSim / SimGrid documentation |
| `cgsim.ui_manifest` | `cgsim.ui_manifest` | TUI branding (banner, accent colour, display name) |
| `cgsim.sim_query` | `cgsim.sim_query` | Natural language → SQL against the CGSim simulation output SQLite database |

Set `ASKPANDA_PLUGIN=cgsim` and `CGSIM_DB_PATH=/path/to/cgsim.db` when
running the AskCGSim deployment.  Set `BAMBOO_CHROMA_COLLECTION=cgsim_docs`
to point the doc tools at the CGSim vector store.
