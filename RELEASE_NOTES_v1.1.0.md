# Bamboo MCP v1.1.0

*21 September 2026 — successor to v1.0.8 (20 August 2026)*

Two substantial new surfaces — a **code-mode primitive surface** for agentic
frameworks and a **REST analysis facade** for the PanDA monitor — plus spend
accounting, per-session conversational state, and a round of MCP client
reliability work that closes the wedged-session failure mode seen on
`aipanda033`.

`CHANGELOG.md` carries the full per-change record with root-cause analysis.
This document is the summary.

---

## Highlights

- **Code mode.** Five `atlas.log.*` primitives decompose `panda_log_analysis`
  into composable steps for an agent that writes code against small tools
  rather than selecting one compound tool per turn. Off by default; selected
  with `BAMBOO_TOOL_PROFILE=primitive`. See
  [`docs/code-mode.md`](docs/code-mode.md).
- **REST analysis facade** at `/api/v1`, behind the PanDA monitor's "Analyse
  failure" button. Asynchronous by default, disk-backed, cached and
  single-flighted. Off unless `BAMBOO_REST_ENABLED` is set. See
  [`docs/rest-api.md`](docs/rest-api.md).
- **Spend accounting and admission control** (`cost_guard`), so a shared
  deployment cannot be run into an unbounded LLM bill by one caller.
- **Session-scoped conversational state.** Concurrent clients no longer share
  one another's context.
- **The MCP client no longer wedges.** The specific failure that spun a full
  CPU core for 17 hours is fixed, along with four neighbouring defects on the
  same path.
- **A failed job with a clean setup log and empty payload logs now arrives
  with context** instead of with nothing. This one is visible to end users —
  see *Behaviour changes* below.

---

## Upgrading

**The `mcp` floor has moved from 0.9.0 to 1.10.0, and this one is blocking.**
Structured tool output (`CombinationContent`, `outputSchema` validation,
`validate_input`) arrived in 1.10.0. Below that floor the SDK does not reject
an `outputSchema` declaration — it advertises the schema and never enforces
it, and the mismatch surfaces as a malformed result at the client rather than
an error at the server.

```bash
pip install --ignore-installed PyJWT "mcp>=1.10.0,<2.0.0" --break-system-packages
python -c "import importlib.metadata as m; print(m.version('mcp'))"
```

Then reinstall the plugin packages editable — the new entry points do not
register otherwise, even for an existing editable install — and clear stale
bytecode:

```bash
pip install -e core/ -e packages/askpanda_atlas/ -e packages/askpanda_epic/
find . -name "__pycache__" -type d -exec rm -rf {} +
```

Nothing else is required. Every new surface in this release is opt-in:
`BAMBOO_TOOL_PROFILE` defaults to `orchestrated`, `BAMBOO_REST_ENABLED`
defaults to off, and a deployment that sets neither behaves as v1.0.8 did
apart from the behaviour changes listed below.

---

## New in this release

### Code-mode primitive surface

`panda_log_analysis` is a compound tool: one call runs metadata fetch, file
listing, log download, excerpt, classification and evidence bundling. Agentic
frameworks built around code mode need a finer-grained surface, so Bamboo now
advertises a second one over the same implementation functions:

| Tool | Does |
|---|---|
| `atlas.log.fetch_metadata` | The job's metadata subset — facts, not decisions |
| `atlas.log.plan_fetch` | Which files to read next, re-entrant, at most three calls |
| `atlas.log.fetch_text` | One file's diagnostic excerpt, budgeted server-side |
| `atlas.log.classify` | The verdict, pure and idempotent |
| `atlas.log.list_files` | The log tarball with sizes |

Every domain decision stays server-side: the agent carries opaque dicts
between calls and never evaluates a predicate. The two surfaces **partition** —
`orchestrated` (the default) advertises the compound tool and withholds the
primitives, `primitive` does the reverse, `both` is the union — and the switch
is advertising only: `call_tool` is not gated by profile.

The composed loop and one `fetch_and_analyse` call are held to the same
verdict, excerpt, exception, pilot version, URLs and **download order** by a
nineteen-scenario equivalence walkthrough. Three differences are intended and
are documented rather than left to be discovered.

ATLAS only. Not visible to Bamboo's own planner, so the orchestrated tool
catalog is unchanged at 26 tools.

### REST analysis facade

A small HTTP surface at `/api/v1`, served by the same uvicorn process as the
MCP endpoint, so the PanDA monitor's backend can ask Bamboo why a job failed
and render the answer. The browser never talks to Bamboo and never holds the
token.

Asynchronous by default with polling, a disk-backed record store, caching and
single-flight so a page refresh does not start a second analysis, deep links
into the Streamlit chat for a user who wants to continue the conversation, and
per-caller admission limits.

### Spend accounting and admission control

`cost_guard` tracks LLM spend and refuses new work past a configured budget,
rather than discovering the overrun on an invoice. Prompt-log documents now
also carry token counts for the synthesis call, so per-model cost breakdowns
are answerable from Bamboo's own OpenSearch index.

### Session-scoped conversational state

Conversational context is now keyed per session. Previously two clients
talking to one server could see each other's context — a correctness problem
as soon as more than one person uses a shared deployment.

### Core-dump analysis

`--debug-file-directory` / `BAMBOO_CORE_DUMP_DEBUG_DIR` for per-release debug
symbols, explicit discovery of the CPython gdb helper so `py-bt` works,
correct container mount, better interpreter detection, and a failed run now
keeps the evidence of why it failed instead of deleting it.

---

## Behaviour changes

**A clean setup log is no longer thrown away.** When a pilot-1305 job's
`setup.stdout` was read successfully, reported no setup error, and the payload
logs then yielded nothing — both zero-length or both undownloadable — the
analysis previously reported no log at all. The fallback read a field that is
only ever assigned on a branch that returns early, so it could only ever
resolve to the empty string.

Such a job now reaches the TUI, the Streamlit interface, the REST facade and
the PanDA monitor's "Analyse failure" button with the setup log's excerpt, its
exception and its traceback count. `setup_log_excerpt` is unchanged and still
means "setup.stdout reported an error".

**`tools/list` no longer publishes Bamboo's internal tool metadata.** The
`tags` key every definition carries and the `examples` key nine of them carry
were being accepted by `mcp.types.Tool` (which allows extra fields), retained
and serialised to every client on every listing. Nothing read either back.
Definitions are now projected onto the fields the MCP `Tool` type declares. A
client that was reading `tags` off the wire — none is known — would need
changing.

**MCP client error messages now name the right machine.** Every HTTP
connection failure previously advised checking a server subprocess, which does
not exist on that transport. Failures are now classified on the originating
error: an unreachable endpoint points at the server and the tunnel, a timeout
names its own budget and the variable that sets it, a 401 points at the bearer
token.

---

## Fixed

Selected; the CHANGELOG has the rest.

- **Reconnect orphaned a live MCP client on every click**, each orphan keeping
  an event-loop thread, a transport and possibly a server subprocess. This is
  the mechanism behind the client that spun a full CPU core for 17 hours.
- **A failed connect left an established transport behind with no reference to
  close it**, and a failing session close abandoned the rest of the teardown.
- **A failed HTTP connect reported a cancellation instead of the error that
  caused it** — a `CancelledError` with no cause, which also slipped past
  `except Exception` guards in the agent.
- **A refused endpoint took the full per-call budget to report itself.**
- **The MCP session was opened and closed by different tasks**, which can wedge
  an anyio cancel scope.
- **Conversational state leaked between concurrent clients.**
- **A completed analysis was replayed indefinitely** and said nothing about it.
- **The planner catalog advertised tool names the server does not expose**, and
  a plan's spelling of a tool name decided whether its evidence was found.
- **`atlas.log.list_files` could emit a payload its own schema rejects** —
  found by an end-to-end run against the real SDK, not by a unit test, which is
  why the regression test validates with real `jsonschema`.

---

## Documentation

- [`docs/code-mode.md`](docs/code-mode.md) (new) — the primitive surface, the
  profile switch, budgets, the equivalence contract.
- [`docs/rest-api.md`](docs/rest-api.md) (new) — the full `/api/v1` contract,
  polling, caching, budgets, monitor integration.
- [`docs/mcp.md`](docs/mcp.md) — tool profiles, and the `outputSchema`
  execution contract for tools that return structured content.
- [`docs/tools/panda_log_analysis.md`](docs/tools/panda_log_analysis.md) — the
  primitive surface, the clean-setup fallback, and corrected excerpt budgets.
- [`docs/security.md`](docs/security.md), [`docs/http-server.md`](docs/http-server.md),
  [`docs/opensearch.md`](docs/opensearch.md) — the REST surface, its shared
  token, and the scope of the new token counts.

---

## Test suites

| Suite | Tests |
|---|---|
| `pytest tests` | 1582 |
| `pytest packages/askpanda_atlas packages/askpanda_epic` | 1732 |

The two invocations must stay separate; combining them causes collection
errors.
