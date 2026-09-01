# `panda_log_analysis`

**Package:** `askpanda_atlas`, `askpanda_epic`
**Modules:** `askpanda_atlas.log_analysis_impl`, `askpanda_epic.log_analysis_impl`
**Type:** Operational data — job failure diagnosis

---

## Purpose

`panda_log_analysis` diagnoses why a specific PanDA job failed. It fetches job metadata and the pilot log from the PanDA monitor, extracts the most relevant failure context from the log, classifies the failure type, and returns structured evidence for LLM synthesis.

This is the primary tool for questions like:
- "Why did job 7099498577 fail?"
- "What error caused job 6837798305 to fail?"
- "Analyse the failure of job 7100840246."

---

## Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| `job_id` | integer | Yes | PanDA job ID (`pandaid`) to analyse. |
| `query` | string | No | Original user query (passed to the LLM synthesiser). |
| `context` | string | No | Optional additional context (site, task ID, release). |

---

## Output

A JSON-serialised evidence dict with the following keys:

| Key | Description |
|---|---|
| `job_id` | The queried job ID. |
| `monitor_url` | BigPanDA / PanDA monitor URL for this job. |
| `jobstatus` | Final job status (`failed`, `holding`, `cancelled`, etc.). |
| `jobsubstatus` | Sub-status if available. |
| `computingsite` | Site where the job ran. |
| `cloud` | Cloud region. |
| `atlasrelease` | ATLAS software release (key name is from the PanDA API; present in both ATLAS and ePIC). |
| `jeditaskid` | Parent task ID. |
| `attemptnr` / `maxattempt` | Retry attempt number and maximum allowed. |
| `transformation` | Payload transformation script (e.g. `Reco_tf.py`). |
| `piloterrorcode` / `piloterrordiag` | Numeric pilot error code and diagnostic string. |
| `exeerrorcode` / `exeerrordiag` | Execution error code and diagnostic string. |
| `taskbuffererrorcode` / `taskbuffererrordiag` | Task buffer error code and diagnostic string. |
| `ddmerrorcode` / `ddmerrordiag` | DDM (data management) error code and diagnostic string. |
| `starttime` / `endtime` / `duration` | Job timing information. |
| `failure_type` | Classified failure category (see below). |
| `log_url` | URL to the primary log file (`payload.stdout` for code 1305; `pilotlog.txt` otherwise). `null` for non-failed jobs. |
| `stderr_url` | URL to `payload.stderr`. Populated only when `payload.stderr` is non-empty for code 1305 jobs; `null` otherwise. |
| `setup_log_url` | URL to `setup.stdout`. Populated for code 1305 jobs where `setup.stdout` was non-empty; `null` otherwise. |
| `setup_log_excerpt` | Budget-capped content of `setup.stdout` when a fatal setup error was found. `null` if setup completed successfully or the file was absent. |
| `log_available` | Whether at least one log file was successfully downloaded. |
| `log_excerpt` | Most relevant section of the log, extracted by pattern matching. For code 1305 jobs where a setup error was found this is the `setup.stdout` content; otherwise it combines stdout and stderr (separated by `--- payload.stderr ---`). |
| `links_md` | Pre-built Markdown links block appended verbatim after LLM synthesis. |

---

## Failure classification

The tool classifies failures into categories using pattern matching against error fields and log content:

| Category | Signals |
|---|---|
| `reassigned_by_jedi` | "reassigned by jedi", "toreassign" |
| `stagein_timeout` | "file transfer timed out", "timeout during stage-in", "cp_timeout" |
| `stageout_timeout` | "timed out during stage-out", "timeout during stage-out" |
| `timeout` | "timeout", "timed out", "walltime", "cpu time exceeded", "tobekilled" |
| `segfault` | "segmentation fault", "sigsegv", "signal 11" |
| `disk_full` | "no space left", "disk quota", "disk full", "work directory too large" |
| `memory` | "out of memory", "oom killer", "memory limit", "job has exceeded the memory" |
| `network` | "connection refused", "network unreachable", "dns failure", "socket error" |
| `input_missing` | "no such file", "file not found", "input file missing" |
| `stagein_failed` | "failed to stage-in", "stage-in failed" |
| `pilot_monitoring_error` | "getpwuid", "uid not found", "list_processes_and_threads" |
| `setup_release_not_found` | "no matched release is found", "!!!error!!!" |
| `payload_error` | "athena", "traceback", "exception", "abort", "core dump" |
| `pilot_error` | "piloterrorcode" |
| `unknown` | No pattern matched |

Pattern matching is evaluated in the order above — the first match wins.

`pilot_monitoring_error` intentionally appears before `payload_error` because the pilot WARNING log lines that accompany these errors contain the word "exception", which would otherwise trigger a false `payload_error` classification.

`setup_release_not_found` intentionally appears before `payload_error` because when the Apptainer/Athena environment setup fails the `setup.stdout` excerpt contains `!!!ERROR!!!`, which would otherwise match the `payload_error` "exception" / "abort" keywords.

`pilot_monitoring_error` signals a pilot infrastructure issue, not a user payload failure. The canonical example is pilot error code 1354 (`getpwuid(): uid not found`), which occurs when the pilot's CPU monitoring code calls `getpass.getuser()` and the current process UID is not present in the worker node's user database. This is a site configuration issue, not a problem with the user's job.

`setup_release_not_found` signals that the container/software environment could not be configured before the payload was launched. The canonical example is an architecture mismatch: `aarch64` release requested on an `x86_64` worker, resulting in `!!!ERROR!!! No matched release is found` in `setup.stdout`. The payload never ran in this case.

---

## Log extraction

Logs are fetched only for jobs in `failed`, `holding`, or `cancelled` states.

### File-size index

Before downloading any log file the tool fetches the filebrowser directory listing
(`/filebrowser/?pandaid={job_id}&json`, no `filename=` parameter) to obtain a
`{filename: size_bytes}` index. Any file confirmed to have `size == 0` is skipped
without issuing a download request — the file contains no diagnostic information
and fetching it wastes a round-trip.

If the index endpoint returns an error the tool falls back to attempting all files
(fail-open), preserving the behaviour of older versions.

### File selection and download order for code 1305

For `piloterrorcode == 1305` (payload setup verification error) the download order is:

1. **`setup.stdout` (always first)** — If non-empty per the file-size index, download it and scan for a recognisable setup error (see `_SETUP_ERROR_PATTERNS` below).
2. **If a setup error is found** — use `setup.stdout` as the primary excerpt and **stop**. Do not attempt `payload.stdout` or `payload.stderr`. The payload never ran, so those files will be empty and contain no additional diagnostic information.
3. **If no setup error is found** (setup succeeded, or the file is absent/empty) — fall through to `payload.stdout` → `payload.stderr`, skipping any file confirmed zero-length by the index.

#### Setup error patterns (`_SETUP_ERROR_PATTERNS`)

| Pattern | Description |
|---|---|
| `!!!error!!!` (case-insensitive) | Generic AtlasSetup / Apptainer fatal error prefix |
| `no matched release is found` (case-insensitive) | Athena release not found for the requested platform/arch |
| `asetup.*failed` (case-insensitive) | `asetup` command failure |
| `error.*release.*not.*found` (case-insensitive) | Generic release-not-found phrasing |

### File selection for all other error codes

| Condition | Files fetched |
|---|---|
| `piloterrorcode == 1305`, setup error found | `setup.stdout` only |
| `piloterrorcode == 1305`, no setup error | `payload.stdout` + `payload.stderr` (if non-empty) |
| All other codes | `pilotlog.txt` only (or `payload.stdout` if `_select_log_filename` returns it) |

The stderr fetch for code 1305 (no-setup-error path) is intentional: Python tracebacks, C++ exceptions, and segfaults frequently appear only on stderr and would be missed if only stdout were examined.

### Context window extraction for `pilotlog.txt`

This is a three-level priority cascade:

**Level 1 — hardcoded pattern for known codes (`_PILOT_CODE_PATTERNS`)**

Eight pilot error codes have a hardcoded regex pattern that is known to appear near the failure in the log:

| Code | Pattern used |
|---|---|
| 1099 | `"Failed to stage-in file"` |
| 1104 | `r"work directory .* is too large"` |
| 1150 | `"pilot has decided to kill looping job"` |
| 1151 | `"File transfer timed out"` |
| 1201 | `"caught signal"` |
| 1235 | `"job has exceeded the memory limit"` |
| 1324 | `"Service not available"` |
| 1354 | `"getpwuid"` (uses trailing-context extraction — see below) |

When a match is found, the 40 lines immediately preceding (and including) the matching line are returned. The pilot writes thousands of lines; without anchoring, the relevant 40 lines would be buried.

**Trailing-context extraction for code 1354**

For pilot error code 1354 the diagnostic content *follows* the anchor line: the pilot writes a `WARNING | Exception caught: 'getpwuid()...'` line first, and then the multi-line Python traceback appears on subsequent lines. Standard preceding-context extraction would stop at the `WARNING` line and miss the traceback entirely.

For codes in `_TRAILING_CONTEXT_CODES` (currently `{1354}`), `_extract_context_window_with_trailing` is used instead. After matching the anchor line it continues collecting up to 30 further lines, stopping early on a blank line (which reliably signals the end of a Python traceback block in pilot logs).

**Level 2 — `piloterrordiag` as a fallback pattern (the scalability mechanism)**

For the 100+ pilot error codes not in `_PILOT_CODE_PATTERNS`, the tool uses the first 40 characters of `piloterrordiag` (regex-escaped) as the search pattern. This works because the pilot generates `piloterrordiag` directly from the log message it just wrote — the same text that becomes the diagnostic string was written to the log moments earlier. Searching for it therefore finds the right line without requiring a handcrafted entry per error code.

**Level 3 — tail fallback**

If the `piloterrordiag` pattern fails to match, the last 40 lines of the log are returned. This is a last resort that ensures something is always sent to the LLM rather than an empty excerpt.

### Context window extraction for `setup.stdout` (code 1305, setup error)

No pattern matching is attempted. The full content of `setup.stdout` is used, capped at `_MAX_EXCERPT_CHARS` (6 000 characters). Setup logs for jobs with fatal errors are typically short (a few hundred lines of `asetup`/Apptainer output ending with `!!!ERROR!!!`), so the cap is rarely reached.

### Context window extraction for payload logs (code 1305, no setup error)

No pattern matching is attempted. The excerpt is built with a **split budget** to guarantee both the relevant stdout errors and the stderr traceback are always visible:

| Section | Budget | Method |
|---|---|---|
| `payload.stdout` | up to 4 000 characters | **Character-based tail** (last 4 000 chars) |
| `payload.stderr` | up to 2 000 characters | Full content |
| **Total** | **6 000 characters** | |

**Why character-based (not line-based) for stdout:** Payload logs from frameworks like EventLoop and TopCPToolkit are often hundreds of thousands of characters of verbose `INFO` tool-initialisation messages, followed by a compact block of `ERROR` lines at the very end. A line-count tail would land in the middle of INFO messages and miss the ERROR block. A char-based tail always captures the final ERROR cascade.

### Character cap

The total excerpt sent to the LLM is capped at **6 000 characters**. The split budget above is how that cap is distributed for payload failures. For `pilotlog.txt` the full 6 000 characters are available for the context window.

---

## Links

The `links_md` evidence field contains a Markdown links block constructed from programmatic URLs — not from LLM text — so the URLs are always correct. `bamboo_executor` strips any LLM-invented links section and appends this block verbatim to the synthesised answer.

The block always starts with the monitor link. Additional links are included in the following order, when present:

1. **Setup Log** (`setup.stdout`) — included when a setup error was found and `setup.stdout` was fetched
2. **Payload Log** (`payload.stdout`) — included for code 1305 jobs where the payload-log path was taken
3. **Payload stderr** (`payload.stderr`) — included when `payload.stderr` was fetched
4. **Pilot Log** (`pilotlog.txt`) — included for all other error codes

Example — code 1305 with setup failure:

```
Links:
- BigPanDA Monitor — https://bigpanda.cern.ch/job?pandaid=7106290502
- Setup Log — https://bigpanda.cern.ch/filebrowser/?pandaid=7106290502&json&filename=setup.stdout
```

Example — code 1305 with payload failure (no setup error):

```
Links:
- BigPanDA Monitor — https://bigpanda.cern.ch/job?pandaid=7099498577
- Payload Log — https://bigpanda.cern.ch/filebrowser/?pandaid=7099498577&json&filename=payload.stdout
- Payload stderr — https://bigpanda.cern.ch/filebrowser/?pandaid=7099498577&json&filename=payload.stderr
```

Example — pilot log failure:

```
Links:
- BigPanDA Monitor — https://bigpanda.cern.ch/job?pandaid=7099498577
- Pilot Log — https://bigpanda.cern.ch/filebrowser/?pandaid=7099498577&json&filename=pilotlog.txt
```

---

## Internal structure

The synchronous `fetch_and_analyse` function is decomposed into three units to keep cyclomatic complexity within the project limit (≤ 15):

| Function | Responsibility |
|---|---|
| `_fetch_logs_payload` | Code 1305 path: setup.stdout first, payload fallback, zero-length guards |
| `_fetch_logs_pilotlog` | All other codes: pilotlog.txt selection, zero-length guard, context extraction |
| `fetch_and_analyse` | Orchestration: metadata fetch, dispatch to helper, evidence assembly |

Both helpers return a `_LogFetchResult` dataclass carrying the six log-fetch outputs (`log_excerpt`, `log_url`, `log_available`, `stderr_url`, `setup_log_url`, `setup_log_excerpt`).

---

## Plugin differences

| Aspect | `askpanda_atlas` | `askpanda_epic` |
|---|---|---|
| Monitor label | `BigPanDA Monitor` | `PanDA Monitor` |
| Cache module | `askpanda_atlas._cache` | `askpanda_epic._cache` |
| Tool tags | includes `"atlas"` | includes `"epic"`, `"eic"` |

The analysis logic (log extraction, failure classification, evidence structure) is identical in both packages.

---

## Routing

`bamboo_answer` routes to this tool deterministically when a job ID and failure-analysis keywords are both present in the question. The routing uses `_is_log_analysis_request`, which matches patterns like "analyse", "analyze", "why", "fail", "log", "diagnos" near a job ID.

---

## See also

- [`panda_job_status`](panda_job_status.md) — job status without log analysis (for status/metadata questions)
- [`pilot_source_analysis`](pilot_source_analysis.md) — follow-up tool for `pilot_monitoring_error`: fetches the relevant pilot3 source modules from GitHub and extracts the functions named in the traceback for LLM analysis
- [`bamboo_last_evidence`](bamboo_last_evidence.md) — inspect the raw evidence dict via `/inspect` or `/json`
- [`docs/rest-api.md`](../rest-api.md) — the REST facade behind the PanDA monitor's "Analyze failure" button, which reaches this tool by asking `bamboo_answer` "Analyze job N and explain the failure"
