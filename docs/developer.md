# Developer guide

This guide covers local development setup, editable installs, testing, and
linting.

---

## Development environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools
```

---

## Editable installs

Bamboo uses **editable installs** so that plugins register their entry points
correctly with `importlib.metadata`.

```bash
# Core server + ATLAS plugin (minimum for development)
pip install -e ./core
pip install -e ./packages/askpanda_atlas

# Additional plugins
pip install -e ./packages/askcgsim
pip install -e ./packages/askpanda_verarubin
pip install -e ./packages/askpanda_epic
```

**You must re-run `pip install -e` after changing:**

- any `pyproject.toml`
- entry-point definitions (`bamboo.tools`)
- plugin dependencies
- package layout / module paths

You do NOT need to reinstall for plain Python source file changes.

### Stale `*.egg-info` shadows the real entry points

A setuptools build leaves an `*.egg-info/` directory **inside the package
source tree** — `packages/askpanda_atlas/askpanda_atlas.egg-info/`,
`core/bamboo_core.egg-info/`. These are gitignored build artifacts, but
`tests/conftest.py` puts those same directories at the front of `sys.path`, so
`importlib.metadata` discovers them as installed distributions. An
`entry_points.txt` left over from an earlier version then shadows the current
one.

The failure is quiet and misleading. The tools whose entry points are missing
from the stale file simply never appear in `tools/list`, so
`BAMBOO_TOOL_PROFILE=primitive` advertises no primitives at all and
`BAMBOO_TOOL_PROFILE=both` is indistinguishable from `orchestrated` —
which reads exactly like a bug in the profile switch. Six tests in
`test_tool_profiles.py` and `test_tool_name_canon.py` fail, and none of their
messages points here.

Check it before believing any entry-point symptom:

```bash
grep -i '^Version' packages/askpanda_atlas/askpanda_atlas.egg-info/PKG-INFO
grep 'atlas.log' packages/askpanda_atlas/askpanda_atlas.egg-info/entry_points.txt
```

If the version is behind the one in `pyproject.toml`, delete and reinstall:

```bash
rm -rf packages/*/[a-z]*.egg-info core/*.egg-info
pip install -e ./core -e ./packages/askpanda_atlas
```

---

## Optional feature dependencies

Bamboo uses separate requirements files for optional features so the core
package stays lightweight.

| File | Install when… |
|---|---|
| `requirements.txt` | Always — base MCP server dependencies |
| `requirements-dev.txt` | Running tests or linting |
| `requirements-mistral.txt` | Using `LLM_DEFAULT_PROVIDER=mistral` |
| `requirements-openai.txt` | Using `LLM_DEFAULT_PROVIDER=openai` or `openai_compat` |
| `requirements-anthropic.txt` | Using `LLM_DEFAULT_PROVIDER=anthropic` |
| `requirements-gemini.txt` | Using `LLM_DEFAULT_PROVIDER=gemini` |
| `requirements-rag.txt` | Using the RAG pipeline (`panda_doc_search`, `panda_doc_bm25`) |
| `requirements-otel.txt` | Exporting traces via OpenTelemetry (`BAMBOO_OTEL_ENDPOINT`) |
| `requirements-textual.txt` | Running the Textual TUI |
| `requirements-ui.txt` | Running the Streamlit UI |

---

## Running tests

Install dev dependencies first:

```bash
pip install -r requirements-dev.txt
```

This installs `pytest`, `pytest-asyncio>=0.21`, `flake8`, `pylint`, and the
circular-import detector.  **`pytest-asyncio>=0.21` is required** — the test
suite uses `asyncio_mode = "strict"` (set in `pyproject.toml`) and most tests
are `async def`.  Without it every async test will fail with
`"async def functions are not natively supported"`.

Run all tests from the repo root:

```bash
pytest tests/
# or quieter:
pytest -q tests/
# single file:
pytest tests/test_task_status.py
# single test:
pytest tests/test_task_status.py::test_task_status_success_json
```

All 324 tests run fully offline — no API keys, no network, no ChromaDB
instance required.

---

## Test suite layout

```
tests/
├── conftest.py                      # sys.path setup for non-installed runs
│
├── test_task_status.py              # panda_task_status — BigPanDA HTTP tool
├── test_job_status.py               # panda_job_status
├── test_log_analysis.py             # panda_log_analysis — failure classification
├── test_doc_rag.py                  # panda_doc_search — ChromaDB vector search
├── test_doc_bm25.py                 # panda_doc_bm25 — BM25 keyword search
├── test_topic_guard.py              # two-stage topic guard
├── test_bamboo_answer_helpers.py    # helper functions (_extract_task_id, _compact_json, etc.)
├── test_bamboo_answer_rag.py        # bamboo_answer — routing, follow-up detection, guard bypass
├── test_bamboo_executor.py          # execute_plan — tool resolution, evidence merging, synthesis
├── test_llm_error_handling.py       # friendly LLM error messages, all routes
├── test_planner.py                  # bamboo_plan — LLM planner tool
├── test_context_memory.py           # multi-turn history threading across all routes
├── test_narrow_waist.py             # list[MCPContent] contract enforced by all tools
│
├── test_llm_providers.py            # OpenAI / Anthropic / Gemini / compat clients
│
├── test_tracing.py                  # bamboo.tracing — NDJSON spans, file output
├── test_tracing_otel.py             # bamboo.tracing — OpenTelemetry integration
│
├── test_panda_http_sync.py          # _panda_http / _fallback_http parity
├── test_loader.py                   # plugin entry-point loader
└── test_cli.py                      # bamboo CLI
```

### Testing strategy

- **Unit-test tools by mocking external services** — BigPanDA HTTP, LLM
  providers, ChromaDB, and upstream MCP servers are all mocked with
  `unittest.mock`.
- **No real credentials needed** — all async tests use `AsyncMock`; all
  provider tests mock the vendor SDK at the module level.
- **Tracing tests** mock `_start_otel_span` and `_get_otel_tracer` directly
  rather than patching `sys.modules`, which is more reliable across pytest
  isolation modes.
- **`bamboo.tools` rule**: tools must always return a result and never raise.
  Error-handling tests verify this contract end-to-end.

---

## Environment configuration

Copy the example file and fill in your credentials:

```bash
cp bamboo_env_example.sh bamboo_env.sh
source bamboo_env.sh
```

Key variables:

| Variable | Purpose |
|---|---|
| `LLM_DEFAULT_PROVIDER` | `mistral`, `openai`, `anthropic`, `gemini`, `openai_compat` |
| `LLM_DEFAULT_MODEL` | Model string for the chosen provider |
| `MISTRAL_API_KEY` / `OPENAI_API_KEY` / … | Provider API keys |
| `ASKPANDA_ENABLE_REAL_PANDA` | `1` to use real BigPanDA API |
| `PANDA_MONITOR_TOKEN` | BigPanDA access token — required for log analysis |
| `PANDA_MONITOR_TOKEN_SCHEME` | Authorization scheme for it (default `Bearer`) |
| `BAMBOO_TRACE` | `1` to enable structured tracing |
| `BAMBOO_TRACE_FILE` | Write trace NDJSON to file (required for TUI) |
| `BAMBOO_OTEL_ENDPOINT` | OTLP/gRPC endpoint for OpenTelemetry export |

See `bamboo_env_example.sh` for the full list and `docs/tracing.md` for
tracing details.

---

## CLI

```bash
# List all registered tools
python -m bamboo tools list
python -m bamboo tools list --json

# Start the MCP server (stdio)
python -m bamboo.server

# Inspect with MCP Inspector
npx @modelcontextprotocol/inspector python3 -m bamboo.server
```

---

## Linting

```bash
# Flake8 (max line length 200, complexity 15)
flake8 .

# Pylint
pylint core/ interfaces/ packages/

# Type checking
pyright .

# Pre-commit (runs flake8 + circular import detection on changed files)
pre-commit run --all-files
```

The pre-commit hook checks for circular imports using
`circular-import-detector==1.0.18`.  Run it before opening a PR.

### pyright on a CERN machine: keep the Node bootstrap off AFS

`pyright` on PyPI is a launcher, not a type checker — the checker is
JavaScript. With no usable Node it downloads one and unpacks it into
`~/.cache/pyright-python/nodeenv`, which on lxplus and `aipanda033` is AFS.
An unpacked Node is several hundred megabytes and the AFS home quota is not,
so the install dies partway through with `[Errno 122] Disk quota exceeded`
under a wall of `nodeenv` tracebacks, having already consumed whatever quota
was left:

```
OSError: [Errno 122] Disk quota exceeded:
  '.../.cache/pyright-python/nodeenv/include/node/openssl/archs/BSD-x86/...'
RuntimeError: nodeenv failed; for more reliable node.js binaries try
  `pip install pyright[nodejs]`
```

Take that suggestion — it is the right fix here, not a workaround.
`pyright[nodejs]` pulls in `nodejs-wheel-binaries`, which ships Node as a
wheel into `site-packages`; the venv lives on `/data`, so nothing touches AFS.
`pyright/node.py` checks for `nodejs_wheel` *before* a global `node` and
before building a nodeenv, so it is used as soon as it is installed:

```bash
rm -rf ~/.cache/pyright-python      # reclaim the half-written nodeenv first
pip install 'pyright[nodejs]'
fs lq ~                             # confirm the quota came back
```

If you would rather keep the nodeenv, redirect it instead — but off AFS:

```bash
export PYRIGHT_PYTHON_ENV_DIR=/data/bamboo/.cache/pyright-nodeenv
```

`PYRIGHT_PYTHON_CACHE_DIR` and `XDG_CACHE_HOME` move the whole cache root and
work equally well.

### Keep the pyright version pinned

`requirements-dev.txt` pins pyright exactly, and the pin is load-bearing
rather than tidiness. pyright ships its own copy of typeshed, so a version
bump can change how a symbol narrows and turn clean code into an error with
nothing in the repository having changed.

One that actually happened: `inspect.isclass(X)` narrows `X` to
`type[object]` when `X`'s own type is unknown, which is the case wherever
`mcp` is unresolvable — an empty environment, the no-pytest venv below.
`object.__init__` takes no keyword arguments, so `ListToolsResult(tools=...)`
behind an `isclass` guard became *"No parameter named 'tools'"* under 1.1.411
while 1.1.408 said nothing. The code was correct both times; the branch is
dynamic by design, so it now binds through a local `Any` and reads the same
under 1.1.408, 1.1.411 and 1.1.414.

Install the dev tools from the file rather than by hand, so the pin applies:

```bash
pip install -r requirements-dev.txt
```

That also brings in `pytest-asyncio`, without which a seventh of the suite
reports *"async def functions are not natively supported"* — see
`tests/test_async_plugin_available.py`.

### pyright sees more where more is installed

pyright reports on what it can resolve, so a clean run means "clean given the
packages present in this environment". `anthropic`, `opensearch-dsl` and
`pysqlite3` are optional dependencies: where they are absent their imports
resolve to `Unknown`, every attribute on them is permitted, and the modules
that use them are effectively unchecked. Install them and real errors appear —
not new ones, just ones that were always there and invisible.

Run pyright somewhere with the optional extras installed before trusting a
clean result, or treat a clean run on a minimal environment as weaker evidence
than it looks:

```bash
pip install anthropic opensearch-dsl pysqlite3-binary
```

**Check the quota even if you do not care about pyright.** The same AFS home
holds the sentence-transformers model ChromaDB loads from
`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`.
A full quota there does not raise — the RAG stack falls back to
`DummyEmbedder` and its 8-dimensional vectors, and retrieval quietly becomes
noise. A failed pyright run is the cheapest possible warning about that.

---

## Adding a new LLM provider

1. Create `core/bamboo/llm/providers/<name>_client.py` following the pattern
   of `mistral_client.py` — lazy SDK import, async semaphore, retry loop,
   `LLMConfigError` / `LLMRateLimitError` escape the retry loop immediately.
2. Register it in `core/bamboo/llm/factory.py` → `_PROVIDER_MAP`.
3. Add `requirements-<name>.txt` with the SDK dependency.
4. Add tests in `tests/test_llm_providers.py` — happy path, missing API key,
   missing SDK, rate limit, timeout, retry.
5. Document the new env vars in `bamboo_env_example.sh`.

## Adding a new tool

1. Create `core/bamboo/tools/<name>.py` implementing `get_definition()` and
   an async `call(arguments)` that **always returns a result** (never raises).
2. Register the singleton in `core/bamboo/core.py` → `TOOLS`.
3. Wrap any LLM call sites with the tracing `span()` context manager.
4. Add tests.

## Writing a plugin

See `docs/plugins.md`.
