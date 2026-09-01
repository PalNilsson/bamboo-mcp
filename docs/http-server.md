# HTTP server deployment

This guide covers running Bamboo as a shared HTTP server so multiple users can
connect without running their own local server process.  The primary use case
is a testbed deployment at BNL or a similar facility.

The stdio transport (used by the TUI by default) spawns a private server
subprocess per user session and needs no network configuration.  The HTTP
transport runs a single persistent server process that any number of clients
can connect to over the network.

---

## Architecture

```
┌─────────────────────┐        HTTP/MCP         ┌──────────────────────────┐
│  User A — TUI       │ ──────────────────────► │                          │
├─────────────────────┤                          │  Bamboo HTTP server      │
│  User B — TUI       │ ──────────────────────► │  (uvicorn ASGI app)      │
├─────────────────────┤                          │                          │
│  User C — curl /    │ ──────────────────────► │  bamboo.entrypoints.http │
│  MCP Inspector      │                          │                          │
├─────────────────────┤         REST            │    /mcp      MCP         │
│  PanDA monitor      │ ──────────────────────► │    /api/v1   REST        │
│  (Django backend)   │                          │    /healthz  liveness    │
└─────────────────────┘                          └──────────────────────────┘
```

The server holds **one shared LLM configuration and one shared PanDA
connection**.  Each client gets an isolated MCP session (its own session ID
and conversation state) but shares the underlying resources.

The `/api/v1` REST surface is off unless `BAMBOO_REST_ENABLED` is set.  It
exists so a web page can ask Bamboo why a job failed without speaking MCP; see
[`docs/rest-api.md`](rest-api.md).

---

## Prerequisites

Install `uvicorn` — the ASGI server used to run the HTTP entrypoint:

```bash
pip install uvicorn
```

All other dependencies are the same as for stdio mode.  Make sure the full
environment is configured (LLM keys, `PANDA_BASE_URL`, etc.) before starting.

---

## Starting the server

There are two equivalent ways to start the HTTP server:

### Option A — `python -m bamboo.server_http` (recommended)

A thin wrapper that reads host/port from env vars or CLI flags and prints a
startup banner:

```bash
# Defaults: localhost:8000
python -m bamboo.server_http

# Custom host and port
python -m bamboo.server_http --host 0.0.0.0 --port 9000

# All options
python -m bamboo.server_http --help
```

Startup banner written to stderr:
```
Bamboo MCP HTTP server  v1.1.0
  MCP endpoint : http://0.0.0.0:8000/mcp
  Health check : http://0.0.0.0:8000/healthz
  Workers      : 1
  Auth         : disabled (open access)
  Log level    : info
```

### Option B — uvicorn directly (more control)

```bash
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` binds to all network interfaces so remote clients can
  connect.  Use `--host 127.0.0.1` to restrict to localhost only.
- `--port 8000` — change if the port is already in use.

The MCP endpoint is at:

```
http://<your-hostname-or-ip>:8000/mcp
```

### Finding your address

```bash
hostname -f                                          # FQDN (recommended at BNL)
ip addr show | grep "inet " | grep -v 127.0.0.1     # IP address
```

### Running persistently

For a testbed that should survive terminal disconnection:

```bash
# With nohup — logs go to bamboo_server.log
nohup python -m bamboo.server_http \
  --host 0.0.0.0 --port 8000 \
  > bamboo_server.log 2>&1 &

echo "Server PID: $!"   # note this to kill it later

# With screen — easier to inspect and reattach
screen -S bamboo-server
python -m bamboo.server_http --host 0.0.0.0 --port 8000
# Ctrl+A D to detach
# screen -r bamboo-server to reattach
```

### Multiple workers

**Run one worker.  `--workers > 1` is not supported.**

```bash
# Correct.
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000

# Broken: MCP sessions break under round-robin dispatch.
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000 --workers 4
```

`--workers` forks separate processes, and the per-session MCP state in
`bamboo/entrypoints/http.py` — `_transports`, `_tasks`, `_ready` — is a
module-level dict, so it is per-process.  A client establishes a session
against worker 1, its next request is dispatched to worker 2, and worker 2
builds a *second* transport for a session that already exists elsewhere.  The
symptom is intermittent and confusing rather than a clean failure.  Making this
work needs sticky session routing in front of uvicorn, which nothing here does.

Concurrency within the single worker is fine and is the intended answer.
Blocking work is offloaded to threads, the RAG cache is lock-protected, and
DuckDB deliberately runs on the loop thread because CRIC queries take 2–15 ms.
The practical ceilings are the default thread pool and the LLM provider's rate
limit, not the process count.  The REST facade bounds its own share with
`BAMBOO_ANALYSIS_MAX_CONCURRENCY`.

The REST analysis store is on disk rather than in memory specifically so that
multiple workers remain possible later — but the MCP session problem above has
to be solved first.

---

## Authentication

If the server is reachable from outside your team, configure Bearer token
authentication.  When no tokens are configured, the server accepts all
requests (suitable for an isolated testbed subnet, not for public exposure).

### Option A — tokens file (recommended for multiple users)

Create a file with one entry per line:

```
# /etc/bamboo/tokens.txt
# Format: client_id: token   (or just: client_id token)
alice: s3cr3t-token-for-alice
bob:   s3cr3t-token-for-bob
ci:    s3cr3t-token-for-ci-runner
```

Then set the env var before starting the server:

```bash
export BAMBOO_MCP_TOKENS_FILE="/etc/bamboo/tokens.txt"
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

### Option B — inline token list (quick setup)

```bash
export BAMBOO_MCP_TOKENS="alice:s3cr3t-alice,bob:s3cr3t-bob"
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

Format: comma-separated `client_id:token` pairs.  If only a token (no colon)
is given, the client ID is recorded as `"unknown"`.

### Generating secure tokens

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Run once per user and distribute the token out-of-band.

---

## Health endpoint

The server exposes a lightweight liveness probe at `GET /healthz` — no
authentication required, no MCP protocol involved:

```bash
curl http://localhost:8000/healthz
# → ok

# Non-zero exit code if server is down (useful in scripts)
curl --fail --silent http://localhost:8000/healthz && echo "server up"
```

Response: `200 ok` (plain text) when the process is alive.  Any path that is
not `/healthz`, `/mcp`, or under `/api/v1/` returns a plain-text `404`.

This endpoint is suitable for:
- **Kubernetes liveness/readiness probes** — `httpGet: path: /healthz`
- **Load balancer health checks**
- **Simple monitoring scripts** (`curl --fail`)
- **Manual verification** that the server started correctly

Note: `/healthz` confirms the process is running and the ASGI app is
responsive, but does not verify LLM connectivity or PanDA MCP session status.
Use the `bamboo_health` MCP tool for a deeper status check.

---

## REST analysis API

Alongside `/mcp`, the same process can serve a small REST surface at `/api/v1`
so a web page can ask Bamboo why a PanDA job failed.  It backs the "Analyze
failure" button on a failed job page in the PanDA monitor.

It is off by default.  A deployment that does not set `BAMBOO_REST_ENABLED`
behaves exactly as before it existed.

```bash
export BAMBOO_REST_ENABLED=1
export BAMBOO_REST_STORE_ROOT=/var/lib/bamboo/rest-analysis
export BAMBOO_COST_STATE_ROOT=/var/lib/bamboo/cost

python -m bamboo.server_http --host 127.0.0.1 --port 8000
```

Check it is up:

```bash
curl -sS http://127.0.0.1:8000/api/v1/capabilities \
  -H "Authorization: Bearer $TOKEN"
```

Three deployment points specific to this surface:

- **Bind to localhost** when the only client is a monitor on the same node.  An
  unauthenticated request that reaches this port can spend the day's LLM
  budget, so there is no reason to expose it more widely than it is used.
- **Move the state roots off `/tmp`.**  Both default under `/tmp`, which does
  not survive a reboot: the answer cache is lost and the day's recorded spend
  resets to zero.
- **Give the monitor its own token line** (`panda-monitor: <token>`) so its
  traffic is distinguishable from a human's in the logs.  The REST surface
  shares one allowlist and one policy with `/mcp`.

The full contract — endpoints, response envelope, error codes, polling, caching,
budgets, and the monitor integration guide — is in
[`docs/rest-api.md`](rest-api.md).

---

## Firewall

At BNL you may need to open the port through the host firewall:

```bash
# RHEL / Rocky Linux (firewalld)
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload

# Ubuntu (ufw)
sudo ufw allow 8000/tcp
```

Verify the port is reachable from another machine:

```bash
curl http://<your-hostname>:8000/mcp
# Expected: HTTP 405 (correct — GET is not a valid MCP request)
```

---

## Connecting with the TUI

Users on other machines run:

```bash
python interfaces/textual/chat.py \
  --transport http \
  --http-url http://<your-hostname>:8000/mcp
```

Or set `MCP_URL` in the environment to avoid repeating the URL:

```bash
export MCP_URL="http://<your-hostname>:8000/mcp"
python interfaces/textual/chat.py --transport http
```

### With authentication

Pass the Bearer token via `--token` or set it in `bamboo_env.sh`:

```bash
# Via flag
python interfaces/textual/chat.py \
  --transport http \
  --http-url http://<your-hostname>:8000/mcp \
  --token s3cr3t-token-for-alice

# Via environment variable (add to bamboo_env.sh)
export MCP_BEARER_TOKEN="s3cr3t-token-for-alice"
python interfaces/textual/chat.py --transport http \
  --http-url http://<your-hostname>:8000/mcp
```

The token is sent as `Authorization: Bearer <token>` on every request.

---

## Connecting with MCP Inspector

Useful for debugging tool calls directly:

```bash
npx @modelcontextprotocol/inspector \
  --url http://<your-hostname>:8000/mcp
```

---

## Verifying the server is running

### Quick liveness check

```bash
curl http://localhost:8000/healthz
# → ok
```

### Full MCP handshake verification

A three-step sequence that confirms the MCP protocol is working end-to-end:

```bash
# Pick a session ID (any UUID — the client chooses it)
SESSION="bamboo-test-$$"

# Step 1 — initialize the MCP session
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2024-11-05",
        "capabilities":{},
        "clientInfo":{"name":"curl","version":"0.1"}}}'
# → event: message
# → data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",
#           "capabilities":{"prompts":...,"tools":...},"serverInfo":{...}}}

# Step 2 — send the required initialized notification
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'

# Step 3 — list all registered tools
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
# → event: message
# → data: {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"bamboo_health",...},...]}}
```

The `notifications/initialized` step (step 2) is required by the MCP spec —
the server will reject `tools/list` with "received request before
initialization was complete" if it is skipped.

The session ID is chosen by the client and passed in the `mcp-session-id`
header on every request in the session.

### Inspecting tool descriptions

The MCP `tools/list` method returns the full tool descriptions that the LLM
uses for tool selection — useful for verifying that a plugin's tools are
registered with the expected descriptions after deployment.

Note: the response only contains tools for the **active plugin** (set by
`ASKPANDA_PLUGIN`). This is intentional — sending all plugins' tool
descriptions to the LLM wastes tokens. Set `ASKPANDA_PLUGIN` before starting
the server to control which plugin's tools are returned:

```bash
SESSION="bamboo-inspect-$$"

# Initialize session
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0.1"}}}' > /dev/null

# Send required initialized notification
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' > /dev/null

# Fetch and print tool names and descriptions
curl -s -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | grep "^data: " \
  | sed 's/^data: //' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for tool in data['result']['tools']:
    print(f'=== {tool[\"name\"]} ===')
    print(tool['description'])
    print()
"
```

The `grep "^data: "` step strips the SSE framing (`event: message\ndata: {...}`)
so that `python3` receives clean JSON. The `notifications/initialized` step is
required by the MCP spec — omitting it causes the server to reject `tools/list`
with "received request before initialization was complete".

```bash
# Check the process
ps aux | grep uvicorn

# Check the port
ss -tlnp | grep 8000          # Linux
lsof -i :8000                  # macOS

# Tail the log (if using nohup)
tail -f bamboo_server.log
```

---

## Stopping the server

```bash
# If started with nohup, use the PID you noted earlier
kill <PID>

# Or find and kill by port
kill $(lsof -ti :8000)         # macOS
kill $(fuser 8000/tcp)         # Linux

# If running in screen
screen -r bamboo-server
# then Ctrl+C
```

---

## Environment variables reference

All standard Bamboo env vars apply.  HTTP-specific additions:

| Variable | Default | Purpose |
|---|---|---|
| `BAMBOO_HTTP_HOST` | `127.0.0.1` | Bind host for `python -m bamboo.server_http` |
| `BAMBOO_HTTP_PORT` | `8000` | Bind port for `python -m bamboo.server_http` |
| `BAMBOO_HTTP_LOG_LEVEL` | `info` | Uvicorn log level (`debug`/`info`/`warning`/`error`) |
| `BAMBOO_MCP_TOKENS_FILE` | — | Path to Bearer token allowlist file |
| `BAMBOO_MCP_TOKENS` | — | Inline comma-separated `client_id:token` list |
| `MCP_URL` | — | Default server URL read by TUI (`--http-url` default) |

REST analysis surface — all off or safe by default.  The full table, including
the analysis store and budget settings, is in
[`docs/rest-api.md`](rest-api.md#environment-variables):

| Variable | Default | Purpose |
|---|---|---|
| `BAMBOO_REST_ENABLED` | `0` | Master switch for `/api/v1/*` |
| `BAMBOO_REST_STORE_ROOT` | `/tmp/bamboo/rest-analysis` | Records, cache, claims — move off `/tmp` |
| `BAMBOO_COST_STATE_ROOT` | `/tmp/bamboo/cost` | Daily spend counters — move off `/tmp` |
| `BAMBOO_REST_INLINE_WAIT_S` | `8.0` | Wait before answering 202 |
| `BAMBOO_ANALYSIS_MAX_CONCURRENCY` | `4` | Concurrent analyses |
| `BAMBOO_ANALYSIS_DAILY_BUDGET_USD` | `0` (off) | Daily LLM ceiling in USD |

See `bamboo_env_example.sh` for the full list of LLM, PanDA, and tracing
variables that also need to be set on the server.

---

## Troubleshooting

**`Connection refused` on the client**
: The server is not running, or the firewall is blocking the port.  Check
  `ps aux | grep uvicorn` on the server and verify the firewall rules.

**`HTTP 401 Unauthorized`**
: Auth is enabled on the server but the client sent no token or the wrong one.
  Verify `BAMBOO_MCP_TOKENS` / `BAMBOO_MCP_TOKENS_FILE` and the token the
  client is sending.

**`HTTP 405 Method Not Allowed` on GET**
: This is correct MCP behaviour — the endpoint only accepts POST.  The server
  is running fine.

**`HTTP 404` on `/api/v1/...` with `The REST analysis API is disabled`**
: `BAMBOO_REST_ENABLED` is not set in the environment the *server process*
  inherited.  Exporting it in your shell after uvicorn started has no effect —
  restart the server.  A `404` on `/api/v1` with no trailing segment is
  different: the prefix needs a path, so use `/api/v1/capabilities`.

**Cached analyses disappear after every reboot**
: `BAMBOO_REST_STORE_ROOT` is still at its `/tmp` default.  The same applies to
  `BAMBOO_COST_STATE_ROOT` and the day's recorded spend.

**LLM errors on first question**
: The LLM environment variables (`MISTRAL_API_KEY` etc.) are not set in the
  server process environment.  Make sure you `source bamboo_env.sh` before
  starting uvicorn, or export the variables in the systemd unit / nohup
  invocation.

**`SSL_CERT_FILE` errors**
: If `SSL_CERT_FILE` is set in the shell environment and points to a
  non-existent file, the Mistral HTTP client will fail with
  `[Errno 2] No such file or directory`.  Run `unset SSL_CERT_FILE` before
  starting the server.
