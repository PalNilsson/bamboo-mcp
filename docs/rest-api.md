# REST analysis API

A small HTTP surface at `/api/v1`, served by the same uvicorn process as the
MCP endpoint, so a web page can ask Bamboo why a PanDA job failed and render
the answer.  It exists for the "Analyze failure" button on a failed job page in
the PanDA monitor.

Everything here is off unless `BAMBOO_REST_ENABLED` is set.  A deployment that
does not set it behaves exactly as v1.0.8 did.

- Implementation: [`core/bamboo/entrypoints/rest.py`](../core/bamboo/entrypoints/rest.py)
- Record store: [`core/bamboo/analysis_store.py`](../core/bamboo/analysis_store.py)
- Spend and admission: [`core/bamboo/cost_guard.py`](../core/bamboo/cost_guard.py)
- Deployment of the process itself: [`docs/http-server.md`](http-server.md)

---

## Why REST and not MCP

The browser cannot speak MCP sensibly.  Streamable HTTP means JSON-RPC, plus a
session handshake, plus SSE, plus a bearer token — and putting that token in
page JavaScript hands it to every visitor of the monitor, which is the whole
security model gone in one view-source.

So the browser never talks to Bamboo:

```
┌─────────┐   HTML/JSON   ┌──────────────────────┐   REST    ┌──────────────┐
│ browser │ ────────────► │  PanDA monitor       │ ────────► │  Bamboo      │
│         │ ◄──────────── │  (Django, same node) │ ◄──────── │  /api/v1     │
└─────────┘               └──────────────────────┘           └──────────────┘
                            holds the bearer token             127.0.0.1:8000
                            knows who the user is
```

The monitor backend is the only party that holds the service token, and it is
also the only party that knows the authenticated identity of the person who
clicked.  Both facts point the same way.

The contract below is deliberately plain HTTP with no Bamboo-specific concepts
in it, so the facade can move behind the PanDA Gateway later without the
monitor noticing.

---

## Enabling it

Four things, in this order.

**1. Switch it on.**

```bash
export BAMBOO_REST_ENABLED=1
```

Without this, every request under `/api/v1/` returns `404` with the error code
`not_found` and the message `The REST analysis API is disabled; set
BAMBOO_REST_ENABLED=1 to enable it.`  That message is the one reliable way to
tell "the flag is unset" from "the route is wrong", so check for it first when
a monitor integration returns 404.

**2. Bind to localhost.**

```bash
python -m bamboo.server_http --host 127.0.0.1 --port 8000
```

The monitor runs on the same node.  Nothing else needs to reach this port, and
an unauthenticated request that reaches it can spend the day's LLM budget.

**3. Configure a token for the monitor.**

The REST surface shares one token allowlist and one policy with `/mcp` —
`rest.authenticate()` is called by both.  Give the monitor its own client id so
its traffic is distinguishable in the logs:

```
# /etc/bamboo/tokens.txt
panda-monitor: <token>
```

See [`docs/security.md`](security.md) for the file format and
`scripts/generate_tokens.py` for generating values.

If no tokens are configured at all, authentication is disabled and every caller
is recorded as `auth-disabled`.  That is acceptable on a laptop.  It is not
acceptable on a shared node, even one bound to localhost, because any local
account can then spend the budget.

**4. Point the state directories somewhere persistent.**

```bash
export BAMBOO_REST_STORE_ROOT=/var/lib/bamboo/rest-analysis
export BAMBOO_COST_STATE_ROOT=/var/lib/bamboo/cost
```

Both default under `/tmp`, which does not survive a reboot.  Losing the store
loses the answer cache; losing the cost state resets the day's recorded spend
to zero, which matters if a budget is enforced.

---

## Endpoints

```
POST /api/v1/analysis            {job_id, mode?, user?}  -> 200 done | 202 running
GET  /api/v1/analysis/{id}                               -> 200
POST /api/v1/analysis/{id}/rating {rating: 1..5}         -> 204
GET  /api/v1/capabilities                                -> 200
```

Note the prefix must be followed by a path segment.  A request to exactly
`/api/v1` is not routed here; it falls through to the HTTP entrypoint's
plain-text 404.

Request bodies are capped at 64 KiB (`MAX_BODY_BYTES`).  A body that exceeds it
is a `400 invalid_request`.

### `POST /api/v1/analysis`

Start an analysis, or pick up one that already exists.

```json
{
  "job_id": 7272161793,
  "mode": "failure",
  "user": "pnilsson"
}
```

| Field | Required | Meaning |
|---|---|---|
| `job_id` | yes | PanDA job id.  Must be a positive integer. |
| `mode` | no | Analysis flavour.  Only `"failure"` is supported; defaults to it. |
| `user` | no | Identity for attribution, recorded as `requested_by`.  Falls back to the authenticated client id. |

`user` is an attribution field and nothing more.  It is not trusted, it grants
nothing, and it is not used in the cache key.  The monitor forwards the
authenticated username here so a record can be traced back to a person; the
authorisation decision was already made by the monitor before the request was
sent.

`mode` exists so the vocabulary is in place before there is a second flavour.
Core-dump analysis is deliberately *not* one of them — see
[Deliberate omissions](#deliberate-omissions).

Responses:

- `200` — an answer is ready.  Either it came from cache (`cached: true`) or it
  finished inside the inline wait.
- `202` — accepted and running.  Poll the `analysis_id`.
- `400 invalid_request` — missing or non-integer `job_id`, unsupported `mode`,
  malformed JSON, oversized body.
- `401` / `403 unauthorized` — no token, or an unknown one.
- `429 budget_exhausted` — the daily budget is spent.  Carries `Retry-After`.

### `GET /api/v1/analysis/{id}`

Poll a running analysis, or re-read a finished one.  Same envelope as the POST.

- `200` — the record, in whatever state it is in.
- `400 invalid_request` — the id is not a bare identifier.  Ids arrive straight
  from a URL path, so anything that could climb out of the store directory is
  refused rather than sanitised.
- `404 not_found` — no such record.  Records are swept after
  `BAMBOO_ANALYSIS_RETENTION_S`, so a very old id gives this too.

### `POST /api/v1/analysis/{id}/rating`

```json
{"rating": 4}
```

Writes the rating onto the prompt-log document for this analysis, which is the
same field the TUI's `/rate N` and the Streamlit star buttons write.  Ratings
from the monitor therefore land in the same OpenSearch index and the same
analyses as ratings from chat.

- `204` — stored.  No body.
- `400 invalid_request` — rating absent, non-integer, or outside 1–5.
- `404 not_found` — no such analysis.
- `409 no_promptlog` — the analysis has no prompt-log document to rate.  Either
  prompt logging is disabled on this deployment, or the fire-and-forget write
  had not reported its document id before the answer was returned.  A rating
  widget should be hidden, not shown-then-failing, when `promptlog` is `null`
  in the record.
- `502 rating_failed` — OpenSearch refused or was unreachable.

### `GET /api/v1/capabilities`

What this deployment can do and how much of its budget is left.  Useful as a
readiness check for the monitor: if this returns 200, the token is good and
the feature is on.

```json
{
  "modes": ["failure"],
  "model": "claude-sonnet-4-6",
  "plugin": "atlas",
  "inline_wait_s": 8.0,
  "poll_after_s": 2,
  "limits": {"max_concurrency": 4, "max_queue": 20, "in_flight": 1},
  "budget": {
    "daily_usd": 0.0,
    "spent_usd": 1.8342,
    "calls_today": 214,
    "unpriced_calls_today": 0
  }
}
```

`daily_usd: 0.0` means no budget is enforced; accounting still runs.
`unpriced_calls_today` counts calls whose model has no entry in the price
table — they are counted in tokens but contribute nothing to `spent_usd`, so a
non-zero value here means the reported spend is an undercount rather than the
truth.

---

## The response envelope

Identical for POST and GET, so the monitor has one shape to render whether the
answer arrived inline, from cache, or after polling.

```json
{
  "analysis_id": "a1b2c3d4e5f6",
  "job_id": 7272161793,
  "mode": "failure",
  "state": "complete",
  "cached": false,
  "elapsed_s": 21.418,
  "poll_after_s": null,
  "answer_markdown": "The job failed with pilot error 1150 ...",
  "evidence": {"...": "..."},
  "promptlog": {"index": "bamboomcp-promptlog-2026.08.31", "doc_id": "xY7..."},
  "error": null
}
```

| Field | Type | Notes |
|---|---|---|
| `analysis_id` | string | Opaque, `[A-Za-z0-9_-]{1,64}`.  Poll and rate with it. |
| `job_id` | int | Echoed back. |
| `mode` | string | Echoed back. |
| `state` | string | `queued`, `running`, `complete`, `failed`. |
| `cached` | bool | True when served from a previous identical analysis. |
| `elapsed_s` | float | Creation to terminal state.  Still climbing while running. |
| `poll_after_s` | int \| null | Suggested poll interval, `null` once terminal. |
| `answer_markdown` | string \| null | The answer.  Markdown.  **See the rendering warning below.** |
| `evidence` | object \| null | Structured evidence behind the answer. |
| `promptlog` | object \| null | `{index, doc_id}`, or `null` when there is nothing to rate. |
| `error` | string \| null | Set when `state` is `failed`. |

### States

```
queued ──► running ──┬──► complete
                     └──► failed
```

`complete` and `failed` are terminal; `poll_after_s` is `null` in both, and
that is the signal to stop polling, not `state == "complete"`.  A client that
polls on state alone will spin forever on a failure.

A record whose owning process died is reported as `failed` rather than left
pending, so a server restart mid-analysis surfaces to the client as an error it
can retry rather than as a poll that never terminates.

### `evidence`

The structured evidence dict from the tool that ran, read out of the caller's
own session bucket rather than a process-global "last tool" store.  That
distinction is why session scoping had to land before this facade could: under
concurrent requests, the last tool that ran belongs to whoever ran it.

Two shapes to expect beyond the ordinary one:

- `null` — no evidence was recorded.
- `{"truncated": true, "reason": "..."}` — the serialised evidence exceeded
  `BAMBOO_ANALYSIS_MAX_RECORD_CHARS` and was replaced by this marker.  The
  answer text is unaffected; only the stored evidence was dropped.

A `log_available: false` key inside the evidence means the analysis ran but
found no log to read, usually because a just-failed job is still uploading.
That result is cached for five minutes rather than a week, so the next caller
gets a real answer.

---

## Asynchronous by default

An analysis takes tens of seconds.  An nginx in front of the monitor typically
gives up at sixty.  So the POST starts the work and hands back an identifier,
and a short inline wait means the quick cases and every cache hit still finish
in one round trip.

```
POST /api/v1/analysis
   │
   ├── cache hit ──────────────────────────────► 200, cached: true
   │
   ├── someone else is already running this ───► 202, their analysis_id
   │
   ├── budget spent ───────────────────────────► 429, Retry-After
   │
   └── start work, wait up to inline_wait_s
           ├── finished in time ───────────────► 200
           └── still running ──────────────────► 202, poll_after_s: 2
```

Client contract for polling:

- Poll `GET /api/v1/analysis/{id}` every `poll_after_s` seconds.
- Stop when `poll_after_s` is `null`.
- Back off on repeated non-terminal responses rather than polling flat out.
- Give up after a hard stop of roughly three minutes and tell the user, rather
  than polling until the browser tab closes.

The inline wait is `BAMBOO_REST_INLINE_WAIT_S`, default 8 s.  Setting it to `0`
makes every fresh analysis return 202, which is a reasonable choice if the
monitor's own request timeout is tight.  Note it does not cancel anything: the
analysis continues after the response is sent, and the record is there when the
client polls.

---

## Caching and single-flight

Two different problems, solved separately.

**Cache** answers "has this exact question already been answered".  Job logs are
immutable once uploaded, so a completed analysis stays valid for a long time.
The key folds in job id, mode, model, and prompt version, because an answer
produced by a different model or a different synthesis prompt is a different
answer:

```
cache_key = f(job_id, mode, model, BAMBOO_ANALYSIS_PROMPT_VERSION)
```

Bump `BAMBOO_ANALYSIS_PROMPT_VERSION` to invalidate every cached answer without
deleting anything.  Default TTL is one week; the no-log case is five minutes.

**Single-flight** stops twenty people clicking the same button from starting
twenty analyses.  The first caller takes a claim; everyone else is handed the
winner's `analysis_id` with a 202 and polls that one.  Claims are published by
hard-linking a fully written temporary file into place, which on POSIX both
fails when the destination exists and makes the file visible complete.

A claim whose owning process is gone is taken over rather than blocking the job
until someone clears the directory by hand.

`sweep()` removes records and pointers older than
`BAMBOO_ANALYSIS_RETENTION_S`.  Nothing calls it on a timer yet — run it from
cron if the store grows.

---

## Admission and limits

In order: cache, then single-flight claim, then budget, then a concurrency
slot.  Refusing early is the design.  A 429 before anything runs is a clean
answer; discovering the budget is gone halfway through an analysis wastes the
tokens it took to find out.

The budget is `BAMBOO_ANALYSIS_DAILY_BUDGET_USD`, default `0` meaning no
ceiling.  Spend is a per-UTC-day counter in a `flock`-protected file, shared
across processes so a detached worker's spend lands in the same total.

**Verify the price table before enabling a budget.**
`cost_guard.DEFAULT_MODEL_PRICES` is a starting point taken from model
knowledge, not an authority — provider prices change with no signal reaching
this repository.  Check the current figures against the provider's pricing page
and override with `BAMBOO_MODEL_PRICES` rather than editing code:

```bash
export BAMBOO_MODEL_PRICES='{"anthropic/claude-sonnet-4-6": [3.0, 15.0]}'
```

Values are USD per million tokens, `[input, output]`.  A model absent from the
table is still counted in tokens and shows up in `unpriced_calls_today`, so an
unpriced model is a visible gap rather than free.

Concurrency is `BAMBOO_ANALYSIS_MAX_CONCURRENCY` slots with a
`BAMBOO_ANALYSIS_MAX_QUEUE` waiting list.  A cap on the queue as well as on the
slots is the point: an unbounded queue turns a spike into a slow-motion outage
where everyone waits, nobody is told to go away, and the ones at the back gave
up long ago.

One rough edge to know about.  The slot is acquired inside the analysis task,
not at admission, so a queue overflow does not produce a 429.  It produces a
record in state `failed` with an error like `20 request(s) already waiting for
one of 4 slots; try again shortly`, delivered as an ordinary `200`.  A monitor
that treats any `failed` state as "Bamboo could not answer, try again" handles
this correctly without special-casing it, but it is not the response code you
would design if you were doing it again.

---

## Integrating the PanDA monitor

The Bamboo side is done.  This is what the `panda-bigmon-core` side needs.  It
is a specification rather than a file list — names must be checked against that
repository.

### 1. Template block

On the job page, rendered when the job has failed and the user is in the pilot
group.  A button, an empty panel, and a hidden rating widget.

### 2. Django proxy view

The browser calls the monitor; the monitor calls Bamboo.  The view:

- holds the bearer token in settings, never in a template or in JavaScript
- forwards the authenticated username as `X-Bamboo-User` and in the `user` body
  field
- enforces a per-user rate limit of its own, because Bamboo's limits are
  global and cannot tell one impatient user from a spike
- passes the response envelope through largely unchanged, minus anything the
  page does not need

### 3. Polling JavaScript

2 s interval with backoff, hard stop around three minutes, stop on
`poll_after_s === null`.

### 4. Rendering the answer — the one that matters

**Render `answer_markdown` as server-side markdown through bleach with an
allowlist.  Never assign it to `innerHTML`.**

The answer embeds job log text.  Log text is attacker-influenceable: a payload
writes to stdout, that lands in a pilot log, the log is fed to a model, and the
model's output quotes it.  So the string is untrusted content wrapped in more
untrusted content, arriving at a page that is authenticated as the viewing
user.  Markdown rendered client-side into `innerHTML` is stored XSS in the
monitor with extra steps.

Hide the rating widget when `promptlog` is `null`; there is nothing to rate and
the endpoint will return 409.

### 5. Continue in Bamboo

A link to the Streamlit app carrying `?job_id=N`, which opens the chat with
"Analyze job N and explain the failure" already asked — the same sentence this
facade sends, so the conversation continues from the same answer rather than
restarting from a different one.  See
[`interfaces/shared/deeplink.py`](../interfaces/shared/deeplink.py).

### End-to-end example

```bash
TOKEN=<the panda-monitor token>
BASE=http://127.0.0.1:8000/api/v1

# Start an analysis.
curl -sS -X POST "$BASE/analysis" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Bamboo-User: pnilsson" \
  -H "Content-Type: application/json" \
  -d '{"job_id": 7272161793, "user": "pnilsson"}'

# → 202
# {"analysis_id":"a1b2c3d4e5f6","state":"running","poll_after_s":2,
#  "answer_markdown":null,"error":null, ...}

# Poll until poll_after_s is null.
curl -sS "$BASE/analysis/a1b2c3d4e5f6" -H "Authorization: Bearer $TOKEN"

# → 200
# {"analysis_id":"a1b2c3d4e5f6","state":"complete","poll_after_s":null,
#  "elapsed_s":21.418,"answer_markdown":"The job failed with ...",
#  "promptlog":{"index":"bamboomcp-promptlog-2026.08.31","doc_id":"xY7..."}, ...}

# Rate it.
curl -sS -X POST "$BASE/analysis/a1b2c3d4e5f6/rating" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"rating": 4}'

# → 204, no body
```

---

## Error codes

Machine-readable `code`, because the monitor renders a different panel for
"budget spent" than for "job unknown", and matching on prose breaks the first
time the wording is improved.

```json
{"error": {"code": "budget_exhausted", "message": "...", "retry_after_s": 3600.0}}
```

| Status | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | Bad or missing `job_id`, unsupported `mode`, malformed or oversized body, bad rating, malformed analysis id. |
| 401 | `unauthorized` | No `Authorization` header, or a malformed one. |
| 403 | `unauthorized` | Token present but not in the allowlist. |
| 404 | `not_found` | Unknown analysis id, unrouted path, or the API is disabled. |
| 405 | `method_not_allowed` | Right path, wrong verb. |
| 409 | `no_promptlog` | Rating an analysis with no prompt-log document. |
| 429 | `budget_exhausted` | Daily budget spent.  `Retry-After` header set. |
| 502 | `rating_failed` | OpenSearch refused or was unreachable. |

An analysis that fails after admission is not an HTTP error.  It is a `200`
carrying `state: "failed"` and an `error` string.

---

## Deliberate omissions

**Core-dump analysis is not a mode.**  It holds a single global slot and
refuses rather than queues (`core_dump_analysis_impl.py`), so it cannot back a
button that appears on every failed job page.  Twenty simultaneous clicks would
give one gdb run and nineteen refusals.  Escalating to a core dump from the
answer panel is a Phase 2 question with a different concurrency design behind
it.

**No streaming.**  Poll for now; SSE progress is Phase 2.

**One uvicorn worker.**  The analysis store is on disk specifically so multiple
workers stay possible later, but the MCP session state in
`entrypoints/http.py` is per-process and would need sticky routing first.  See
[`docs/http-server.md`](http-server.md#multiple-workers).

**No write operations.**  Nothing here retries a task or kills a job.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BAMBOO_REST_ENABLED` | `0` | Master switch for `/api/v1/*` |
| `BAMBOO_REST_STORE_ROOT` | `/tmp/bamboo/rest-analysis` | Records, cache, claims |
| `BAMBOO_REST_INLINE_WAIT_S` | `8.0` | Wait before answering 202 |
| `BAMBOO_ANALYSIS_MAX_CONCURRENCY` | `4` | Concurrent analyses |
| `BAMBOO_ANALYSIS_MAX_QUEUE` | `20` | Queue depth before refusal |
| `BAMBOO_ANALYSIS_CACHE_TTL_S` | `604800` | Cached answer lifetime (one week) |
| `BAMBOO_ANALYSIS_RETENTION_S` | `1209600` | Sweep age (two weeks) |
| `BAMBOO_ANALYSIS_PROMPT_VERSION` | `1` | Bump to invalidate the cache |
| `BAMBOO_ANALYSIS_MAX_RECORD_CHARS` | `1000000` | Evidence cap per record |
| `BAMBOO_ANALYSIS_DAILY_BUDGET_USD` | `0` (off) | Daily ceiling in USD |
| `BAMBOO_COST_STATE_ROOT` | `/tmp/bamboo/cost` | Daily counter files |
| `BAMBOO_MODEL_PRICES` | unset | JSON price overrides |
| `BAMBOO_COST_ENFORCE` | `0` | Refuse LLM calls when over budget |
| `BAMBOO_SESSION_BUCKETS` | `128` | Session bucket cap |
| `BAMBOO_SESSION_TTL_S` | `7200` | Session bucket idle TTL |
| `BAMBOO_DEEPLINK_ALLOW_QUESTION` | `0` | Honour `?q=` free text in deep links |

The two that matter most on a real deployment are `BAMBOO_REST_STORE_ROOT` and
`BAMBOO_COST_STATE_ROOT`, because their defaults live under `/tmp`.

---

## Troubleshooting

**Every request returns 404 with `The REST analysis API is disabled`**
: `BAMBOO_REST_ENABLED` is not set in the *server* process environment.  Setting
  it in your shell after uvicorn started does nothing.

**404 on `/api/v1` exactly**
: The prefix needs a path segment.  Use `/api/v1/capabilities`.

**401 from the monitor but curl works**
: The proxy view is not forwarding `Authorization`, or is forwarding the user's
  own credentials instead of the service token.

**Answers are always `cached: true` and stale**
: The synthesis prompt or the model changed without the cache key changing.
  Bump `BAMBOO_ANALYSIS_PROMPT_VERSION`.

**Ratings return 409**
: `BAMBOO_OPENSEARCH_PROMPTLOG` is not set, so there is no document to rate.
  Check `promptlog` in the record before showing the widget.

**Spend looks too low**
: Check `unpriced_calls_today` in `/api/v1/capabilities`.  A model missing from
  the price table contributes zero dollars.

**A restart lost every cached answer**
: `BAMBOO_REST_STORE_ROOT` is still under `/tmp`.

---

## See also

- [`docs/http-server.md`](http-server.md) — running the process, auth, firewall
- [`docs/security.md`](security.md) — token management
- [`docs/tools/panda_log_analysis.md`](tools/panda_log_analysis.md) — the tool
  that does the actual work behind `mode: "failure"`
- [`docs/opensearch.md`](opensearch.md) — the prompt log a rating writes to
