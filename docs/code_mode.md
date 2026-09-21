# Code mode — the `atlas.log.*` primitive surface

Bamboo's `panda_log_analysis` is a *compound* tool: one call fetches job
metadata, lists the log tarball, downloads the right files in the right order,
excerpts them, classifies the failure and bundles the evidence.  That shape
suits Bamboo's own planner, which selects one tool per turn.

Agentic frameworks built around **code mode** — where the model writes code
that composes small tool primitives rather than picking one compound tool —
need a finer-grained surface.  Bamboo provides one: five `atlas.log.*` tools
over the *same* implementation functions, advertised instead of the monolith
when `BAMBOO_TOOL_PROFILE=primitive`.

The monolith is untouched.  This is a second advertised surface, not a
rewrite, and the two are held to producing the same answer by a
scenario-by-scenario equivalence walkthrough — see
[The equivalence contract](#the-equivalence-contract).

- Implementation: [`packages/askpanda_atlas/askpanda_atlas/log_primitives_impl.py`](../packages/askpanda_atlas/askpanda_atlas/log_primitives_impl.py)
- Entry points: [`packages/askpanda_atlas/askpanda_atlas/log_primitives.py`](../packages/askpanda_atlas/askpanda_atlas/log_primitives.py)
- Profile switch: [`core/bamboo/tools/_tool_profiles.py`](../core/bamboo/tools/_tool_profiles.py)
- Equivalence walkthrough: [`packages/askpanda_atlas/tests/test_log_equivalence.py`](../packages/askpanda_atlas/tests/test_log_equivalence.py)
- Scenario table: [`packages/askpanda_atlas/tests/log_scenarios.py`](../packages/askpanda_atlas/tests/log_scenarios.py)
- The compound tool these decompose: [`docs/tools/panda_log_analysis.md`](tools/panda_log_analysis.md)

---

## Why a second surface rather than a split

Splitting the monolith into primitives and rebuilding it from them was the
obvious alternative and was rejected for two reasons.

**The orchestrated path must not move.** `panda_log_analysis` is what the TUI,
the Streamlit interface, the REST facade at `/api/v1` and the PanDA monitor's
"Analyse failure" button all reach.  A refactor of its internals to serve a
different consumer puts that path at risk for no benefit to the people using
it.

**A code-mode agent and Bamboo's planner want opposite things.** The planner
wants few, powerful tools: every additional name in the catalog is another
way for tool selection to go wrong.  A code-mode agent wants many, small,
composable ones.  A single surface cannot be both, so there are two, and a
server advertises one of them.

---

## Turning it on

`BAMBOO_TOOL_PROFILE` selects which surface the server advertises.

| Value | Advertises | Tools |
|---|---|---|
| unset / `orchestrated` | `panda_log_analysis` | **26** |
| `primitive` | the five `atlas.log.*` | **30** |
| `both` | the union | **31** |

Counts are for the ATLAS plugin with every optional dependency installed;
they are what `create_server()`'s `list_tools` handler actually returns.  The
two log surfaces **partition**: `primitive` withholds the monolith, so an
agent given the primitives cannot sidestep them, and `orchestrated` withholds
the primitives, so Bamboo's planner never sees four names it cannot select.

Three properties of the switch are load-bearing:

- **Advertising only.**  `call_tool` is *not* gated by profile, and neither is
  `bamboo_executor`'s in-process resolution nor the `_tool_names` alias map.
  The profile controls catalog size and selection accuracy, not access.
  Gating dispatch would risk the orchestrated path for no gain.
- **Fail-open.**  An unrecognised value (`BAMBOO_TOOL_PROFILE=primitve`) logs
  one warning and falls back to `orchestrated` rather than refusing to list
  tools.  Likewise a definition whose `profiles` key cannot be parsed is
  treated as profile-agnostic, so a typo costs an extra advertised tool
  rather than silently removing one.
- **Read at listing time.**  The variable is read per `tools/list`, not
  snapshotted at import, so `panda_log_analysis` rebuilds its definition on
  every call.  Anything that caches a tool definition across a profile change
  is wrong.

Tools that name no profile are advertised under all of them.  That is every
other tool in the tree, deliberately: `panda_job_status`, `cric_query` and
`bamboo_health` are as useful to a code-mode agent as to the planner.  Only
the log-analysis compound/primitive pair is ever split.

**Requirements.** `mcp >= 1.10.0` at runtime.  Below that floor the SDK
ignores `outputSchema` and drops the structured half of every primitive's
result *silently* — the call appears to succeed and the agent gets text it
then has to parse.  `requirements.txt` pins both bounds.

**ATLAS only.** There is no ePIC mirror of these tools.  The ePIC copy of
`log_analysis_impl.py` stays profile-agnostic precisely because nothing would
replace it: an ePIC server under `BAMBOO_TOOL_PROFILE=primitive` must not end
up with no log analysis at all.

---

## The loop

Five tools, but only four are on the diagnosis path.  This is the whole
composition:

```python
meta = atlas.log.fetch_metadata(job_id)

fetched, observed = [], {"fetched": []}
plan = atlas.log.plan_fetch(job_id)
while True:
    for entry in plan["next"]:
        got = atlas.log.fetch_text(job_id, entry["filename"], entry["role"])
        fetched.append(got)
        observed["fetched"].append(entry["filename"])
        observed.update(got["signals"])
    if plan["done"]:
        break
    plan = atlas.log.plan_fetch(job_id, observed)

verdict = atlas.log.classify(meta, fetched)
```

`atlas.log.list_files` is not on that path; it answers "what is in this job's
tarball" for a caller that wants the listing itself.

**Every decision stays server-side.** `plan_fetch` chooses the files,
`fetch_text` derives the character budget from the `role` `plan_fetch`
assigned and computes the `setup_has_error` signal, `classify` joins the
excerpts and picks which traceback to trust.  The agent carries opaque dicts
between calls; it never evaluates a domain predicate.  That is the property
that makes the composition equivalent to the monolith rather than
approximately like it.

### Why planning is a tool

`_fetch_logs_payload` does not follow a static plan.  For pilot error code
1305 it reads `setup.stdout` first and *only* skips `payload.stdout` and
`payload.stderr` if that file turns out to contain a setup error — a decision
taken mid-flight, on file **content**.

A one-shot `plan_fetch(job_id) -> [ordered files]` could only express that by
exporting the predicate into a tool description, where it would drift from
the implementation that has to agree with it.  So `plan_fetch` is
**re-entrant**: fetch what `next` names, hand back the `signals` you got as
`observed`, call again.

### Loop-shape rules

- `done` means *no further `plan_fetch` call is required*.  It may be `true`
  alongside a **non-empty** `next` — that is the terminal plan: fetch those
  files and stop.
- A caller that re-plans anyway is answered `{"next": [], "done": true}`, so
  the naive `while not done:` loop terminates too.
- **At most three `plan_fetch` calls** for any job, and the bound is asserted
  by a test.
- Re-planning costs no HTTP round trip.  Both the metadata fetch and the file
  listing go through the in-process caches that the first call populated.
- `plan_fetch` will not re-offer a file whose name is in `observed["fetched"]`,
  so a caller that fetches off-plan and reports it does not get the same file
  twice.

---

## The five tools

Every tool takes `job_id` (integer, required) and an optional `timeout`
(integer, seconds, default 60) unless stated otherwise, declares
`additionalProperties: false` on its input, and returns **structured
content** validated by the SDK against a declared `outputSchema`.

### `atlas.log.fetch_metadata`

Facts about the job.  No decisions: which file to read and whether the job has
logs at all belong to `plan_fetch`, because a second place to learn "which
file" is a second place for that rule to drift.

| Key | Meaning |
|---|---|
| `job_id` | The job described. |
| `monitor_url` | BigPanDA page for the job. |
| `piloterrorcode` | Coerced to an integer; `0` when absent or unparseable. |
| `piloterrordiag` | Pilot error diagnosis; `""` when absent. |
| `pilotid` | Raw `pilotid` field; `""` when absent. |
| `pilot_version_from_pilotid` | Version parsed out of `pilotid`. The **fallback** when no pilot log was read. |
| 19 pass-through fields | `jobstatus`, `jobsubstatus`, `computingsite`, `cloud`, `atlasrelease`, `jeditaskid`, `attemptnr`, `maxattempt`, `transformation`, `exeerrorcode`, `exeerrordiag`, `taskbuffererrorcode`, `taskbuffererrordiag`, `ddmerrorcode`, `ddmerrordiag`, `starttime`, `endtime`, `duration`, `commandtopilot` — BigPanDA's values, verbatim and untyped. |

Pass-through fields are present as `null` when absent rather than omitted, so
a consumer can rely on the shape instead of probing with `in`.  They are
deliberately **untyped** in the schema: BigPanDA decides those types, and
declaring one here would reject a job whose field came back as a string where
another job's came back as an integer.

Two of them are in the subset for reasons that are not obvious.
`commandtopilot` is what `classify_failure` searches for the
JEDI-reassignment signal, and `pilotid` carries the pilot version when no
pilot log was downloaded.  A subset modelled on the monolith's evidence keys
alone would have misclassified exactly the jobs that never really failed.

Pass the whole result to `classify` as `job`, unmodified.

### `atlas.log.plan_fetch`

Which files to download next, and in what order.

Input: `job_id`, optional `observed` (the `signals` mapping from a previous
`fetch_text`, optionally with a `fetched` list of filenames), optional
`timeout`.

| Key | Meaning |
|---|---|
| `job_id` | The job the plan applies to. |
| `strategy` | `payload_1305`, `pilotlog` or `metadata_only`. |
| `next` | Files to fetch now — each `{filename, role, url, reason}`. May be empty. |
| `done` | `true` when no further `plan_fetch` call is required. |
| `notes` | Remarks, typically recording a skipped file. |

The three strategies:

- **`payload_1305`** — pilot error 1305, the payload failed.  A failed release
  or container setup produces the same code with empty payload logs, so
  `setup.stdout` is offered first, alone, with `done: false`.  If the caller
  reports `setup_has_error: true`, the plan ends there with a note and no
  further files.  Otherwise `payload.stdout` (role `primary`) and
  `payload.stderr` (role `secondary`) follow in one terminal plan.
- **`pilotlog`** — every other pilot error code.  One file, chosen by the
  monolith's own `_select_log_filename`, normally `pilotlog.txt`, in a single
  terminal plan.
- **`metadata_only`** — the job's status is not one of `failed`, `holding` or
  `cancelled`, so no logs are downloaded at all.  `next` is empty, `done` is
  true, and `notes` says why.  Call `classify` with `fetched: []`.

A file the listing confirms to be zero-length is skipped, with a note.  An
**unavailable listing** is fail-open: the file is offered anyway, because
"the listing could not be fetched" is not the same as "the file is empty".

`role` maps one-to-one onto the monolith's `_LogFetchResult` URL fields —
`setup` → `setup_log_url`, `primary` → `log_url`, `secondary` → `stderr_url`.
That is deliberate, so comparing a composed loop against one
`fetch_and_analyse` call is a field comparison rather than an interpretation.

### `atlas.log.fetch_text`

Downloads one file and returns its **diagnostic excerpt**, not the raw file.
A pilot log is routinely tens of megabytes and the useful part of it is chosen
by a rule — traceback first, then a pilot-code anchor, then the tail — that
the agent must not have to reimplement.

Input: `job_id`, `filename` (required), `role` (`setup`/`primary`/`secondary`,
default `primary`), optional `timeout`.  Pass `filename` and `role` exactly as
`plan_fetch` gave them: the role sets the budget.

| Key | Meaning |
|---|---|
| `job_id`, `filename`, `url` | What was read, and where a human can click it. |
| `role` | The role after validation. An unrecognised role degrades to `primary` with a note rather than failing the call. |
| `available` | `false` when the file was empty or could not be downloaded. |
| `bytes` | Size of the **whole file** in bytes, not of the excerpt. Comparable with `list_files`' `size_bytes`. |
| `truncated` | `true` when the excerpt is shorter than the whole file. |
| `context` | `{excerpt, exception, traceback_count}` — pass through to `classify` unchanged. |
| `signals` | Domain predicates computed server-side. Merge into `observed`. |
| `pilot_version` | Parsed from `pilotlog.txt` **only**; `""` for any other file. |
| `notes` | Remarks, typically recording an unreadable file. |

Three rules here are worth knowing before composing off-plan:

**The excerpt is computed over the full text, not over a capped slice.**
Traceback anchoring searches the whole file in the monolith, so excerpting at
the tool boundary is what keeps the two paths comparable — and it means no
uncapped log is ever serialised across the wire.

**`signals` carries `setup_has_error` only when the file *is* `setup.stdout`.**
`plan_fetch` counts the key's presence as "the setup log has been read", so a
`setup_has_error: false` picked up from `payload.stdout` and merged into
`observed` would make the next plan skip the setup log entirely.

**`pilot_version` is keyed on the filename, not the role.**  The monolith
parses a version only on the pilotlog path; on the payload path it falls back
to `pilotid` without looking at `payload.stdout` at all.  `parse_pilot_version`
matches its pattern anywhere in a file, so a payload log that echoed a version
line would otherwise make a composed loop report a version the monolith does
not, with no way for the caller to know which to trust.

### `atlas.log.classify`

The verdict.  **Pure**: no network, no job ID, nothing cached — everything it
needs has already been fetched.  Cheap to call, trivial to test, safe to call
twice.

Input: `job` (the `fetch_metadata` result, required) and `fetched` (the
`fetch_text` results, **in fetch order**).

| Key | Meaning |
|---|---|
| `failure_type` | Short category — `stagein_timeout`, `payload_error`, `reassigned_by_jedi`, …; `unknown` when nothing matched. |
| `context` | The combined `{excerpt, exception, traceback_count}` the verdict was taken from. |
| `notes` | Remarks about how the contexts were combined. |

It joins the excerpts rather than leaving that to the caller, because both
join rules are domain rules: the separator written between `payload.stdout`
and `payload.stderr`, and the precedence that prefers the **stderr**
traceback when both files have one — Python tracebacks and segfault reports
go to stderr, so that is the exception which actually terminated the payload.
Put either rule in a tool description and it drifts from the implementation
that has to agree with it.

A setup context is used only when no payload content was supplied.  That
covers the erroring `setup.stdout` (where the loop ended early) and the clean
`setup.stdout` whose payload logs then turned out to be empty — the same two
cases `_fetch_logs_payload` covers in its own fallback branch.

Call it with `fetched: []` for a metadata-only job; it classifies from
metadata alone, exactly as the monolith does.

### `atlas.log.list_files`

What is in the job's log tarball, job-root files first, with sizes.  Not
needed to diagnose a failure — `plan_fetch` already consults the listing — but
useful when the question *is* "what did this job produce".

| Key | Meaning |
|---|---|
| `listing_available` | `false` means the listing could not be fetched. **Unknown, not empty.** |
| `files` | Up to 500 entries of `{relative_path, name, dirname, size_bytes, modification}`. |
| `total` | Entries in the full listing, before truncation. |
| `truncated` | `true` when `files` holds fewer than `total`. |
| `notes` | Remarks, typically recording truncation. |

An unavailable listing is reported as `listing_available: false` with an empty
`files`, **not** as an `error`: the monolith treats the same condition
fail-open and attempts the download anyway, and a primitive that called it
fatal would disagree with `plan_fetch` about the same job in the same session.
A `size_bytes` of 0 means the file is empty and not worth fetching.

---

## Budgets

`BAMBOO_PRIMITIVE_MAX_CHARS` caps what a primitive returns: the excerpt
`fetch_text` extracts, and the verbatim traceback carried alongside it.
Default **8000**.

It is read **independently** of the monolith's internal `_MAX_EXCERPT_CHARS`,
which it happens to equal by default.  A code-mode agent may have a far larger
context window than Bamboo's own synthesis step; raising the budget for one
must not move the other, so `panda_log_analysis` is unaffected by any value
set here.

The budget is split per role exactly as `_fetch_logs_payload` splits it:

| Role | Budget |
|---|---|
| `secondary` (`payload.stderr`) | `min(2000, budget)` |
| `primary`, on a 1305 job | `budget - 2000` |
| `primary`, any other job | `budget` |
| `setup` | `budget` |

The reduction for the payload primary is **unconditional**, taken before it is
known whether `payload.stderr` has any content, because that is what the
monolith does — so the joined excerpt stays within budget either way.  A
budget at or below the stderr reservation is not reduced further: an operator
who set it that low did not mean to disable excerpting.

An unset, unparseable or non-positive value logs a warning and falls back to
8000, for the same reason an unrecognised profile does — a configuration typo
must not fail every call.

`context.exception.raw` is capped separately, at `min(5000, budget)`, through
`truncate_traceback` rather than a slice: a slice discards the terminal
exception line, which is the part worth keeping.

---

## Errors and the output-schema contract

Every primitive declares an `outputSchema` and every return path yields the
`(content, structured)` tuple the SDK expects.  Two consequences for callers:

**Failures arrive as data, not as protocol errors.**  A failed call returns a
normal structured result carrying an `error` string:

```json
{"job_id": 6789012345, "error": "Failed to fetch job metadata from BigPanDA"}
```

Check for `error` before reading anything else.

**No output schema carries a top-level `required`, and all of them declare
`error`.**  The SDK rejects a result with no structured content from a tool
advertising an `outputSchema`, so a failure has to be expressible under that
schema too.  A schema demanding the success keys would turn every error path
into an opaque *"Output validation error"* instead of the precise message
above.

The primitives never reach `bamboo_executor.unpack_tool_result`, which
expects the orchestrated surface's list-of-content-dicts shape.  They are not
planner-visible, so nothing in Bamboo's own pipeline unpacks them — a
code-mode client reads `structuredContent` directly.

---

## The equivalence contract

The composed loop and one `fetch_and_analyse` call must reach the same answer
for the same job.  That is not a docstring claim; it is
[`test_log_equivalence.py`](../packages/askpanda_atlas/tests/test_log_equivalence.py),
which drives both paths over nineteen scenarios — same metadata, same
listing, same downloaded text — and compares eight projections:

| Key | `fetch_and_analyse` | the primitive loop |
|---|---|---|
| `failure_type` | `evidence["failure_type"]` | `verdict["failure_type"]` |
| `excerpt` | `evidence["log_excerpt"]` | the combined `context["excerpt"]` |
| `exception_type` | `evidence["exception_type"]` | `context["exception"]["exc_type"]` |
| `traceback_count` | `evidence["traceback_count"]` | `context["traceback_count"]` |
| `pilot_version` | `evidence["pilot_version"]` | first non-empty fetched, else `pilot_version_from_pilotid` |
| `log_available` | `evidence["log_available"]` | any fetched file came back |
| `urls` | the three URL evidence keys | each fetched file's `url`, by role |
| `fetched` | download order, from a spy | its own spy |

Plus, on every scenario: the `fetch_metadata` subset equals the monolith's
evidence key by key over the intersection, and the search text built from the
subset equals the one built from the full job dict.

`fetched` is the assertion with the most teeth.  An agreeing excerpt says the
two paths arrived somewhere together; an agreeing **download order** says
`plan_fetch` transcribed `_fetch_logs_payload`'s control flow — setup first,
the early return on a setup error, the zero-length skips — rather than
approximating it.

### Three differences that are intended

An agent author who does not know these will discover them by being
surprised.

1. **Evidence bundling has no counterpart.**  The primitives stop at
   `classify` by design: the link block, the follow-up offer, the core-dump
   probe and the metadata pass-through that `fetch_and_analyse` performs
   afterwards are not part of this surface.  The metadata subset is compared
   against the evidence separately.
2. **`context.exception.raw` is capped** at the tool boundary where the
   monolith carries it whole.  `exception_type` and `traceback_count` agree;
   `raw` is a truncation of the monolith's.  A study that treats the excerpt
   as ground truth without knowing this will measure the cap.
3. **URLs for files that were never read.**  `_fetch_logs_payload` assigns
   `log_url` before it knows whether `payload.stdout` is worth downloading, so
   its evidence can link a file neither path read — deliberately, since the
   link is for a human to click.  The primitives produce a URL only for a file
   they fetched.

### Lockstep drift

Two paths can agree and both be wrong.  A rule changed in the monolith and
transcribed faithfully into `plan_fetch` keeps every comparison above green,
so the scenario table also pins the expected **fetch order** and **verdict**
outright.  Such a change then has to be made in the table, where it is
reviewable.

The table in
[`log_scenarios.py`](../packages/askpanda_atlas/tests/log_scenarios.py) is
plain importable data rather than pytest fixtures, so it can be consumed
outside pytest:

```python
import pathlib, sys
sys.path.insert(0, "packages/askpanda_atlas/tests")
from log_scenarios import SCENARIOS  # noqa: E402
```

Adding a row is cheap.  Changing an existing row's `expect_*` fields is a
behaviour change and should be argued for.  Its fixtures are deliberately
**wide as well as long** — the context window either side of a traceback is
capped at a line count before the character budget applies, so a narrow log
excerpts identically under a 2000-character budget and an 8000-character one.
A "simplified" fixture silently stops discriminating the budget rules.

---

## Composing off-plan

The loop above is the supported composition, but the point of shipping
primitives is that an agent can compose them its own way.  Four properties
hold regardless of how you call them — and, because the loop never exercises
them, each is covered in
[`test_log_primitives.py`](../packages/askpanda_atlas/tests/test_log_primitives.py)
and
[`test_log_primitives_text.py`](../packages/askpanda_atlas/tests/test_log_primitives_text.py)
rather than by the equivalence walkthrough:

- **`plan_fetch` will not re-offer a file you report as fetched.**  Put its
  name in `observed["fetched"]`.
- **The whole-file rule for an erroring `setup.stdout` keys on the
  filename**, not the role you assigned.  Setup failures are shell output
  rather than tracebacks, so when no traceback is present the whole capped
  file is kept instead of an anchored window — fetch `setup.stdout` under any
  role and you still get that.
- **`setup_has_error` is emitted only for `setup.stdout`**, whatever role you
  fetched it under.
- **A pilot version comes only from `pilotlog.txt`**, whatever role you
  fetched it under.  For any job whose pilot log you did not read, use
  `fetch_metadata`'s `pilot_version_from_pilotid`.

Arguments are read tolerantly throughout, in the style of the `from_state`
codecs: a non-mapping `observed`, a wrong-typed `fetched`, a malformed entry
in the list, or an unknown key degrades to "nothing observed" rather than
raising.  One malformed entry costs its own content, not the whole
classification.

---

## Worked example

Against a real pilot-1305 job, through Bamboo's own MCP client.  Any MCP
client works; this one is in-tree.

```python
from interfaces.shared.mcp_client import MCPClientSync, MCPServerConfig

client = MCPClientSync(MCPServerConfig(...))  # server started with
                                              # BAMBOO_TOOL_PROFILE=primitive

def call(name: str, **args: object) -> dict:
    """Call one primitive and return its structured payload."""
    result = client.call_tool(name, args)
    payload = result.structuredContent
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload

job_id = 6789012345
meta = call("atlas.log.fetch_metadata", job_id=job_id)
# {'piloterrorcode': 1305, 'jobstatus': 'failed',
#  'pilot_version_from_pilotid': '3.14.0.22', ...}

fetched, observed = [], {"fetched": []}
plan = call("atlas.log.plan_fetch", job_id=job_id)
# {'strategy': 'payload_1305', 'done': False,
#  'next': [{'filename': 'setup.stdout', 'role': 'setup', ...}]}

while True:
    for entry in plan["next"]:
        got = call("atlas.log.fetch_text", job_id=job_id,
                   filename=entry["filename"], role=entry["role"])
        fetched.append(got)
        observed["fetched"].append(entry["filename"])
        observed.update(got["signals"])
    if plan["done"]:
        break
    plan = call("atlas.log.plan_fetch", job_id=job_id, observed=observed)

# Round 2, setup.stdout having reported no error:
# {'strategy': 'payload_1305', 'done': True, 'next': [
#     {'filename': 'payload.stdout', 'role': 'primary', ...},
#     {'filename': 'payload.stderr', 'role': 'secondary', ...}]}

verdict = call("atlas.log.classify", job=meta, fetched=fetched)
# {'failure_type': 'payload_error',
#  'context': {'excerpt': '...', 'exception': {'exc_type': 'ValueError', ...},
#              'traceback_count': 2},
#  'notes': []}
```

Three calls to `plan_fetch` at most; here, two.  The agent wrote no domain
logic: it did not decide to read the setup log first, did not decide whether
the payload logs were worth reading, did not choose a character budget, did
not pick which of the two tracebacks to trust, and did not join the excerpts.

---

## Limits and non-goals

- **Not planner-visible.**  The primitives declare `profiles: ["primitive"]`
  and Bamboo's own catalog is pinned to `orchestrated`, so they never enter
  the planner prompt, cannot dilute its tool-selection accuracy, and will not
  enter the RAG tool-retrieval index.  This is also why adding them cost none
  of the usual `bamboo_answer.py` / `planner.py` / `bamboo_executor.py` edits
  a planner-visible tool requires.
- **No synthesis.**  These tools return evidence.  Turning evidence into prose
  is the calling agent's job — that is the point of code mode.
- **No core-dump probe, no follow-up offer.**  Those live in
  `fetch_and_analyse`; see [`docs/tools/core_dump_analysis.md`](tools/core_dump_analysis.md)
  for the tool that answers what a killed payload was doing.
- **ATLAS only**, as above.
- **One job at a time.**  There is no batch form; a loop over job IDs is the
  intended shape, and the metadata and listing caches make it cheap.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `BAMBOO_TOOL_PROFILE` | `orchestrated` | `orchestrated`, `primitive` or `both`. Unrecognised values warn once and fall back. |
| `BAMBOO_PRIMITIVE_MAX_CHARS` | `8000` | Character budget for what a primitive returns. Independent of the monolith's excerpt budget. |

Both are documented with their full rationale in
[`bamboo_env_example.sh`](../bamboo_env_example.sh).  Production leaves both
unset.

---

## See also

- [`docs/tools/panda_log_analysis.md`](tools/panda_log_analysis.md) — the compound tool these decompose
- [`docs/mcp.md`](mcp.md) — tool discovery, the execution contract, argument validation
- [`docs/plugins.md`](plugins.md) — writing and registering plugin tools
- [`docs/developer.md`](developer.md) — editable installs, running the test suites
