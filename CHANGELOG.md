# Changelog

All notable changes to Bamboo are documented here.

---

## [Unreleased]

### Changed
- **The evidence budget was truncating the one field worth reading**
  (`packages/askpanda_atlas/askpanda_atlas/core_dump_analysis_impl.py`).
  `load_core_evidence` used the analyzer's CLI default of 50 000 characters —
  roughly 12 500 tokens against a 200 000-token context — and the last stage of
  `enforce_global_budget` is `primary_thread.backtrace`. A job with many shared
  libraries and several distinct thread stacks spends the budget on cheaper
  evidence and then cuts the most valuable field.

  Job 7272161793 did exactly that: the XRootD shutdown chain from `Py_Exit`
  through `DefaultEnv::Finalize` to `PollerBuiltIn::Stop` was cut out of the
  model's copy, and the report said so under Limitations. Nothing was lost from
  disk — with `--no-llm` the analyzer never calls `enforce_global_budget` at
  all, so `evidence.json` carried the full per-section evidence and
  `gdb_raw.txt` is not budgeted at any point.

  The default is now 120 000 characters, overridable with
  `BAMBOO_CORE_ANALYSIS_MAX_EVIDENCE_CHARS` for a deployment on a smaller
  model. An unparsable or non-positive value falls back to the default rather
  than shrinking the evidence to nothing.

### Added
- **REST analysis facade for the PanDA monitor**
  (`core/bamboo/entrypoints/rest.py`, `core/bamboo/entrypoints/http.py`). Four
  endpoints under `/api/v1`, served by the existing uvicorn process, so a
  button on a failed job page can ask Bamboo why the job failed.

  A REST surface rather than MCP because the browser cannot speak MCP
  sensibly: Streamable HTTP means JSON-RPC plus a session handshake plus SSE
  plus a bearer token, and putting that token in page JavaScript hands it to
  every monitor visitor. The browser talks to the monitor's own backend, which
  talks to this facade with a service token held server-side.

  ```
  POST /api/v1/analysis            {job_id, mode?}  -> 200 done | 202 running
  GET  /api/v1/analysis/{id}                        -> 200
  POST /api/v1/analysis/{id}/rating {rating: 1..5}  -> 204
  GET  /api/v1/capabilities                         -> 200
  ```

  The work is `bamboo_answer` asked "Analyze job N and explain the failure" —
  the same sentence a chat user would type, taking the same code path, so the
  button cannot drift from chat behaviour. `bypass_fast_path` is pinned to
  `False` rather than left to the environment, because `BAMBOO_FAST_PATH` is
  read only by the Streamlit and Textual interfaces and a REST caller should
  not inherit an interface's debugging switch. `tests/test_rest_routing.py`
  pins the phrasing to `panda_log_analysis`, including a check that the
  f-string in `_execute` still matches, so a routing change breaks a test
  rather than the button.

  Admission order is cache, claim, budget, then concurrency slot. Refusing
  early is the point: a 429 before anything runs is a clean answer, whereas
  discovering the budget is gone halfway through wastes the tokens it took to
  find out. Errors carry a machine-readable `code`, because the monitor renders
  a different panel for "budget spent" than for "job unknown" and matching on
  prose breaks the first time the wording improves.

  Core-dump analysis is deliberately not offered: it holds a single global slot
  and serialises, so it cannot back a button that appears on every failed job
  page.

  Auth is shared with `/mcp` through one `authenticate()` helper rather than a
  second copy of the same check. Everything is off unless `BAMBOO_REST_ENABLED`
  is set, so deploying this changes nothing until somebody decides otherwise.
  `BAMBOO_REST_INLINE_WAIT_S` (default 8 s) controls how long a request waits
  before answering 202, so quick analyses and every cache hit still complete in
  one round trip.

- **Disk-backed store for asynchronous analyses**
  (`core/bamboo/analysis_store.py`). Records, an answer cache and single-flight
  claims, all as small JSON files under `BAMBOO_REST_STORE_ROOT` (default
  `/tmp/bamboo/rest-analysis`).

  On disk rather than in a dict because a log analysis takes tens of seconds,
  which is longer than a browser request should wait behind an nginx proxy, so
  the REST facade has to hand back an identifier to poll — and that identifier
  must survive three things a process global does not: a restart, which would
  otherwise strand every client mid-poll with nothing anywhere saying why; a
  second uvicorn worker, which would round-robin a poll to a process that never
  saw the request; and a concurrent caller asking the same question, who needs
  to see the claim.

  The cache key folds in the model and a prompt version alongside the job and
  mode, because an answer produced by a different model or a since-revised
  prompt is a different answer and serving it would make a model change
  invisible. `BAMBOO_ANALYSIS_PROMPT_VERSION` invalidates every cached answer
  without deleting anything. Job logs are immutable once uploaded, so the
  default TTL is a week — except when the analysis found no log, which gets 300
  seconds, since a job that has just failed may still be uploading. Failures
  are never cached: most are transient rather than properties of the job.

  Claims are published by writing a temporary file and hard-linking it into
  place, so twenty clicks on the same failed job start one analysis and the
  other nineteen are handed the winner's id. `O_CREAT | O_EXCL` alone is not
  enough: it creates a zero-length file and the body lands on the next line, so
  a caller arriving inside that window reads nothing, concludes the claim is
  corrupt and takes it over, producing two winners. A claim that is unreadable
  is therefore re-read a few times before being declared abandoned, and a
  take-over is confirmed by reading back who owns the file. A claim or record
  whose owning process is gone is taken over or marked failed rather than
  blocking that job until somebody clears the directory by hand.

  Records tolerate unknown fields on read, so a rolling upgrade with old code
  reading newer manifests does not strand a poller. Analysis ids are validated
  before they touch a path, since they arrive from a URL. `sweep()` enforces
  `BAMBOO_ANALYSIS_RETENTION_S` (default two weeks).

- **Spend accounting and admission control** (`core/bamboo/cost_guard.py`,
  `core/bamboo/llm/metered.py`, `core/bamboo/llm/factory.py`). A daily USD
  counter, a price table, and a concurrency limiter, ahead of the PanDA
  monitor's "Analyze failure" button turning LLM spend from something a few
  chat users generate into something a page view generates.

  Token usage was already normalised as `TokenUsage` on every `LLMResponse`,
  but it was read only for tracing spans and then discarded, so nothing could
  meter it and the prompt-log's `input_tokens` and `output_tokens` fields were
  always null. `MeteredLLMClient` now wraps every client returned by
  `build_client`. That seam was chosen over any individual call site because
  `client.generate()` is called from at least ten places — synthesis, the LLM
  planner, the topic guard, the promptlog NL-to-DSL translator, the
  connectivity probe, and several ATLAS plugin implementations — and metering
  one would miss the rest. The planner is not a rounding error: its prompt
  carries the whole tool catalogue.

  The counter is a JSON file per UTC date updated under `flock`, not a process
  global, for three reasons: a restart must not reset the day's spend; the
  core-dump analyzer builds its own client in a detached worker process and its
  spend must land in the same total; and multiple uvicorn workers, if ever
  enabled, must share one budget.

  Accounting is always on; enforcement is not. `check_budget()` is for
  admission time, so a refusal is a clean early answer rather than a failure
  halfway through an analysis. Interactive chat is deliberately not gated by
  default — cutting a user off mid-conversation to save a fraction of a cent is
  the worse outcome — but `BAMBOO_COST_ENFORCE=1` makes the metered client
  refuse calls once the budget is gone.

  A model with no price entry has its tokens counted and its calls counted
  under `unpriced_calls`, so it shows up as a visible gap rather than as free,
  and logs once per process rather than once per call. Prices are overridable
  at runtime through `BAMBOO_MODEL_PRICES` because model prices change without
  any signal reaching this repository; the built-in table is a starting point
  and must be verified against the provider's pricing page before a budget is
  enforced.

  `ConcurrencyLimiter` bounds both slots and the queue waiting for them. A
  semaphore alone gives an unbounded queue, which converts a spike into a
  slow-motion outage: every caller waits, none is told to go away, and the ones
  at the back have long since given up by the time they are served.

  New environment variables: `BAMBOO_COST_STATE_ROOT` (default
  `/tmp/bamboo/cost` — note `/tmp` does not survive a reboot, so set a
  persistent path where the budget matters), `BAMBOO_ANALYSIS_DAILY_BUDGET_USD`
  (default `0`, meaning no ceiling), `BAMBOO_MODEL_PRICES`,
  `BAMBOO_COST_ENFORCE`, `BAMBOO_ANALYSIS_MAX_CONCURRENCY` (default 4) and
  `BAMBOO_ANALYSIS_MAX_QUEUE` (default 20).

- **Session-scoped conversational state** (`core/bamboo/session_scope.py`). A
  context variable naming the active session plus a bounded LRU registry of
  named per-session buckets. The session id is bound once per session at the
  transport boundary — `bamboo.entrypoints.http._run_session` sets it as its
  first statement, and because a task owns its own context and every tool call
  for that session runs inside that task, the one assignment covers the
  session's whole lifetime.

  `ScopedMapping` is a `MutableMapping` view onto one bucket of the active
  session, so a module-level name can keep its dict syntax while the storage
  follows the session. That indirection is deliberate: rewriting each of the
  two dozen `_last_evidence_store` call sites to a helper leaves room for a
  missed site to silently reintroduce leakage, and a missed site still
  type-checks and still passes any test that runs unscoped.

  Bounded by `BAMBOO_SESSION_BUCKETS` (default 128 non-default buckets) and
  `BAMBOO_SESSION_TTL_S` (default 7200 s idle). The default bucket is exempt
  from both, since evicting it would break cross-turn follow-ups in a long
  stdio session. An unparsable value falls back to the default rather than
  raising, so a typo in a deployment environment cannot stop the server
  starting.

  With no scope bound every access lands in the default bucket, which
  reproduces the previous process-global behaviour exactly. stdio, the TUI and
  the existing suite are unaffected.

- **Prompt-log documents carry the transport session id**
  (`core/bamboo/llm/prompt_log.py`). `session_id` was the process-wide
  `_SESSION_ID`, correct under stdio but not on the shared HTTP server, where
  every connected client was indexed under one id. That makes a conversation
  impossible to reconstruct from the index and distorts any per-session
  aggregation, which matters for the retrieval- and faithfulness-evaluation
  work. `_effective_session_id()` now prefers the bound session id and falls
  back to `_SESSION_ID` when no scope is active.

- **Explicit search for the CPython gdb helper, enabling `py-bt`**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`). `py-bt`
  needs CPython's `python-gdb.py`, which gdb auto-loads as `<objfile>-gdb.py`
  next to each loaded object. ATLAS/LCG releases do not reliably ship one: for
  job 7272161793 gdb's `info auto-load python-scripts` listed exactly one
  script, libstdc++'s, so auto-load was healthy and had no libpython candidate
  to find — the report's "python-gdb.py was not loaded" was accurate but gave
  the reader nothing to act on.

  A bootstrap now runs before `py-bt` and searches: an explicit
  `--python-gdb-helper` first, then `<objfile>-gdb.py` and
  `share/gdb/auto-load/` for every loaded `libpython*` or `python3*` object.
  It is rendered as a single `python exec(…)` command because the phase runner
  pairs one `-ex` per command and cannot carry a multi-line `python … end`
  block. A helper that fails to source is skipped rather than aborting the
  phase.

  What was searched is recorded in `gdb_metadata.python_helper` and named in
  the `python.reason` text, so "not found here" is distinguishable from "not
  looked for". `BAMBOO_CORE_DUMP_PYTHON_GDB` threads an override through
  `build_analyzer_argv` as an *argument* — ALRB launches apptainer with
  `--cleanenv`, so no host environment variable reaches the container.

### Fixed
- **Conversational state leaked between concurrent clients**
  (`core/bamboo/tools/bamboo_executor.py`, `core/bamboo/llm/prompt_log.py`).
  `_last_evidence_store` was a module-level dict keyed by tool name alone,
  with no session dimension. Two readers consult it *across* turns rather than
  within one, and both are routing gates:

  - `get_last_core_dump_offer()` decides whether a bare "yes" means "analyse
    the core dump", and recovers the job id from the stored evidence rather
    than from the user's message. On the shared HTTP server one process serves
    every client, so user B's "yes" could start a gdb run against user A's job.
  - `get_last_traceback_evidence()` gates the rule 1b pilot-source route, so a
    question could be answered against another user's traceback.

  `bamboo_last_evidence` had the same defect in visible form: it returned
  whatever tool ran last anywhere in the process.

  `prompt_log._last_doc_store` was a single-slot process-wide deque, so under
  concurrency `get_last_doc_id()` could hand one client's rating to another
  client's document. `record_last_doc()` now files coordinates per session and
  `get_last_doc_id()` confines the lookup to the caller's session. Attribution
  happens in a new `_index_and_record()` coroutine rather than inside
  `_write_document`, because that function runs in a worker thread where the
  session context variable is invisible; the session id is captured on the
  event loop and passed explicitly. `_write_document` now returns
  `(index, doc_id)` on success so the wrapper has something to file, and keeps
  its single-argument signature so existing callers and test doubles are
  unaffected.

  Latent rather than dormant: concurrent chat use was light enough to hide it,
  and the arrival rate a "Analyze failure" button on every failed job page
  implies is what makes it certain.

- **Synthesis asserted lock ownership it had just said it could not read**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`). Job
  7272161793's analysis described a three-way cycle in which the XRootD timer
  thread both held `StreamMutex` and was blocked acquiring it — a
  contradiction, not a deadlock — while its own limitations section correctly
  said the lock-ownership graph could not be read from the optimised frames.
  A confident body over an honest limitation is the worst combination: the
  reader acts on the body.

  The frames supported a simpler reading. `Stream::OnReadTimeout` takes the
  stream lock on entry, so thread 3 held it and then called
  `PollerBuiltIn::ShutdownEvents`, waiting for an acknowledgement that only
  thread 3 — the poller — could send. A self-deadlock, with threads 1 and 2 as
  consequences rather than participants. The first analysis of the same job had
  it right before richer evidence encouraged a more elaborate story.

  `SYSTEM_PROMPT_HANG` now forbids writing a cycle in which one thread both
  holds and waits for a mutex, requires a named holder *and* the frame showing
  it holds the lock (entering a function documented to take it counts; being
  blocked in the acquire path does not), permits "holder unidentified" as an
  answer, and directs the model to the smallest cycle the frames support.

- **`next_steps` offered a check that could not check anything** (same module).
  The report advised re-running the job to verify `output.root` had been
  written — but a re-run produces a different `output.root` and says nothing
  about the original. It also cited `XRD_RUNFORKHANDLER`, which governs fork
  safety rather than shutdown ordering. Verification steps must now name the
  artifact and how to inspect it, and a configuration setting may not be cited
  unless the model can say what it does and how it bears on this failure.

- **"py-bt produced nothing" guessed between two causes instead of testing**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`). When the
  helper loaded but produced no frames, the reason text offered a
  minor-version mismatch *or* missing DWARF as possibilities. For job
  7272161793 the version had already matched exactly — a 3.13 helper for a 3.13
  core — so naming the version as a candidate sent the reader looking for a
  different helper when the helper was correct.

  The bootstrap now runs `gdb.lookup_type("PyObject")` after sourcing the
  helper and reports the outcome as `interpreter_types`. That settles it: gdb
  either has the interpreter's type information or it does not, and without it
  the helper has nothing to walk regardless of version. The three outcomes are
  now reported distinctly — no DWARF, types present but no Python on the stack
  (expected for a process captured inside a C-level shutdown), and no helper at
  all.

### Added
- **`--debug-file-directory` / `BAMBOO_CORE_DUMP_DEBUG_DIR`, as a per-release
  template** (same module, and `core_dump_analysis_impl.py`). ATLAS releases
  ship libpython and the analysis libraries stripped of DWARF, which is what
  stops `py-bt` even with the correct helper and what leaves optimised XrdCl
  frames without argument data. Where a site publishes matching `.debug` trees,
  pointing gdb at them recovers both.

  It takes a template rather than a path. `/cvmfs/atlas.cern.ch/repo/sw/software`
  holds a hundred-odd releases and each job names its own, so a fixed path is
  wrong for every job but one — the first cut of this setting shipped with a
  hardcoded `25.2` example that did not exist. `{project}`, `{release}`,
  `{platform}` and `{base}` are filled by `_parse_release_info` from the setup
  banner in `payload.stdout` (`Using AnalysisBase/25.2.103 [cmake] with platform
  x86_64-el9-gcc15-opt` / `at /cvmfs/.../25.2`), which is also recorded in
  `gdb_metadata.release` so a derived path can be traced back to what it was
  built from.

  The expanded directory must exist and is dropped with a warning when it does
  not: `set debug-file-directory` on a missing path is not an error in gdb, it
  simply loads nothing, so validating is the difference between a setting that
  quietly does nothing and one that says so. Placeholders with no release
  banner, and templates that fail to expand, are dropped the same way. A
  literal path still works for a single-release site.

  Applied as `-iex`, before the core is loaded, since gdb binds separate debug
  files when an objfile is first read. Not staged into the job directory: debug
  trees run to hundreds of megabytes and the path is a CVMFS one, already
  visible in the container.

  **Documented as leave-unset for a stock ATLAS deployment.** No debug trees
  are published under `/cvmfs/atlas.cern.ch/repo/sw/software` in any location
  this tool knows of.

- **A completed analysis was replayed indefinitely and said nothing about it**
  (`packages/askpanda_atlas/askpanda_atlas/core_dump_analysis_impl.py`,
  `core/bamboo/tools/bamboo_answer.py`). `start_analysis` returns the stored
  evidence when a manifest is `complete` — correct by default, since refetching
  a gigabyte core to answer the same question twice is wasteful — but two
  things made it a trap. The payload was worded identically to a fresh run, and
  no phrasing could reach the `restart` argument, so a job analysed before a fix
  kept reporting the behaviour that fix addressed with no indication that gdb
  had not run.

  Job 7272161793 spent several rounds of debugging in exactly that state: its
  `evidence.json` had no `python_helper` key at all, because the analyzer had
  not run since that key was introduced, while each new question produced
  freshly-worded prose from the same stale evidence.

  `build_response` now takes `replayed` and marks it in both the evidence and
  the text, naming the timestamp and how to force a fresh run. Rule 1c reads
  restart phrasing — "re-run", "again", "redo", "from scratch" — and sets
  `restart` only from explicit wording, never inferred, because it spends a
  gigabyte transfer and the single analysis slot.

- **Interpreter detection missed everything but a versioned basename**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`). The
  bootstrap matched loaded objects on basename alone, against `libpython*` or a
  `python3*` prefix. An interpreter packaged as a plain `python` executable
  matched nothing, so no version was detected, no per-version directory was
  tried, and the search came back empty on a release that is plainly 3.13.

  Detection now also matches a version carried in the object *path*
  (`.../lib/python3.13/lib-dynload/...`), and falls back to
  `_python_version_hint`, which reads the version out of the job directory —
  the payload's own stdout carries `PYTHONPATH`, and an ATLAS release names
  `lib/python3.13/site-packages` in it. The hint is a fallback only; what the
  core says wins.

  `libpython.py` is now accepted alongside `python-gdb.py`, both when searching
  and when staging: that is the name CPython's source tree uses and therefore
  the name most helpers are fetched under. The interpreter objects that were
  recognised are recorded in `gdb_metadata.python_helper.python_objects`, so
  "nowhere to look" and "found nothing to try for 3.13" are distinguishable in
  the reason text.

- **The CPython gdb helper override could never be found**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`).
  `BAMBOO_CORE_DUMP_PYTHON_GDB` was threaded through to the bootstrap as the
  *host* path it was set to. The bootstrap runs inside the release container,
  which ALRB launches with `--cleanenv` and `--contain`, binding only `/cvmfs`,
  the user's home, the job directory at `/srv` and a scratch path. A helper at
  `/data/bamboo/tools/cpython-gdb/...` does not exist there, so `os.path.isfile`
  returned False and the search fell through silently — the same class of
  mistake as assuming the working directory survives `bash -lc`.

  The helper is now staged into the job directory, which the container already
  sees, and the `/srv` form is passed onward. Staged copies join the worker and
  runner scripts in the existing cleanup: removed on success, kept on failure.

- **A single helper path was the wrong shape** (same module). The helper reads
  CPython's own struct layouts, which change between *minor* versions — 3.12's
  frame representation is not 3.11's — so one fixed file cannot serve a
  deployment that analyses jobs from several releases.
  `BAMBOO_CORE_DUMP_PYTHON_GDB` now also accepts a directory of per-version
  helpers laid out as `<version>/python-gdb.py`. The bootstrap detects the
  version from the core's own `libpython` object, takes the matching
  subdirectory (exact `X.Y`, then any `X.Y.*`), and declines rather than loading
  a mismatched helper: no helper is better than the wrong helper. The detected
  version is recorded in `gdb_metadata.python_helper` and named in the reason
  text, so a reader who needs to supply a helper knows which one to fetch.

- **"Not a Python process" was reported for Python processes** (same module).
  When the helper loaded but `py-bt` produced no frames, `_summarise_python`
  fell through to its non-Python branch — plainly wrong for an Athena payload.
  That case now names the loaded helper and the two usual causes: a
  minor-version mismatch, or a `libpython` with no DWARF for the helper to read
  interpreter structures from.

- **Synthesis invented operational mechanisms in `next_steps`**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`). The job
  7272161793 report advised harvesting output files after a looping-job kill
  (the pilot does not stage out after a 1150 kill) and flagging the job so it
  would not count against "the user's error quota" (no such mechanism exists).
  Both are the model reaching past a single core file into systems it has no
  evidence about, and both read as authoritative to someone who does not know
  better. `SYSTEM_PROMPT_BASE` now confines `next_steps` to further diagnosis,
  a change in the culprit component, or a check the reader can perform, and
  names those two inventions specifically.

- **The core-dump container mounted the wrong directory at `/srv`**
  (`packages/askpanda_atlas/askpanda_atlas/_core_dump_analyzer.py`).
  `atlasLocalSetup.sh` binds `$PWD` at `/srv`, and `_container_path` maps every
  host path under `--job-dir` to `/srv/<rel>` on that assumption. The launcher
  established the working directory with `subprocess.run(..., cwd=job_dir)` and
  then ran the setup through `bash -lc` — a login shell, which sources
  `/etc/profile` and the user's profile *before* reaching the command string.
  On a CERN AFS account that chain ends in the home directory:

  ```
  $ cd /tmp/bamboo/core-analysis/job-7272161793/job && bash -lc 'echo $PWD'
  /afs/cern.ch/user/a/atlpan
  ```

  ALRB therefore bound `$HOME` at `/srv` and refused with `Error: unable to
  source setupfile /srv/my_release_setup.sh` — which reads as a missing file
  and was in fact a wrong mount. The file was on disk in the job directory
  throughout; `_job_prep` and the launcher's own `is_file()` check had both
  already passed.

  The `cd` now happens inside the command string, after the profile has had its
  say. A `test -f` guard on the release setup runs between the `cd` and the
  source, so any future drift between the working directory and the `/srv`
  assumption fails on one legible line instead of ALRB's command menu.

  `--contain` was ruled out by probe: `-e "-c -i"` and `-e "-i"` both mount the
  job directory correctly, so the default container arguments are unchanged.

- **A failed container run deleted the evidence of why it failed**
  (same module). The `finally` block unlinked the staged worker, runner and
  evidence files unless `--keep-container-artifacts` was set, so the exact
  command ALRB was handed was gone by the time anyone read the error. Artifacts
  are now kept whenever the run fails, and their location is printed.

- **ALRB's message-of-the-day buried the reason for a failed analysis**
  (same module, and `_core_dump_worker.py`). The analyzer put 6000 characters
  of container output into its exception verbatim, and `worker_log_tail` then
  kept the last twenty *lines* of that — which for a container failure are the
  ROOT security notice and the `lsetup` menu, never the cause. The one line
  that explained job 7272161793 had scrolled off the top before the message
  reached the TUI.

  New `_last_error_line()` scans the whole output for the last `Error:` /
  `Exception:` line and both the analyzer's exception and the worker's failure
  message now lead with it, with the raw tail kept underneath and trimmed to
  2000 characters. The last match wins rather than the first, because a nested
  failure reports its outermost cause last. No match falls back to the tail
  alone, which is the pre-existing behaviour.

- **`/fastpath off` silently disarmed core-dump offer acceptance**
  (`core/bamboo/tools/bamboo_answer.py`). Rule 1d was gated on
  `not bypass_fast_path`, alongside the deterministic routing rules. It does
  not belong there. `bypass_fast_path` exists to hand a *question* to the LLM
  planner instead of a deterministic rule, but a bare "yes" is not a question
  the planner can route: `_run_topic_guard` reformulates content-free
  follow-ups into a documentation query before any planner sees them. Skipping
  rule 1d therefore did not reroute the turn, it destroyed it.

  Observed on job 7272161793. `panda_log_analysis` diagnosed a looping-job kill
  and appended its deterministic offer — "A core dump is present in the job
  log: `core.1178643` (1.1 GB) … Analyse it?" — and the reply "yes please" was
  answered with *"the documentation search did not return relevant results for
  'yes please'"*. Nothing was wrong with the stored offer: the same evidence
  dict that produced the appended offer is the one `get_last_core_dump_offer()`
  reads. The turn never reached it.

  Rule 1d now runs in both routing modes, in the same class as the social
  intercepts beside it: it resolves state this process created in an earlier
  turn, and both of its conditions — a stored offer *and* a message that is
  nothing but an affirmative — are unchanged. Rule 1c stays gated, because an
  explicit request *is* a question and the planner can name the tool itself.
  `tests/test_core_dump_routing.py` now covers acceptance under
  `bypass_fast_path=True`, the unarmed affirmative under the same flag, and
  that 1c remains gated.

- **Core-dump analysis refused on hosts where it would have worked**
  (`packages/askpanda_atlas/askpanda_atlas/core_dump_analysis_impl.py`).
  `preflight_atlas_environment` asked `shutil.which("apptainer")` in the MCP
  server process. The analyzer does not use that PATH:
  `_collect_evidence_atlas_container` starts the release with
  `bash -lc "… source atlasLocalSetup.sh -c …"`, under a login shell whose PATH
  is assembled from `/etc/profile.d` and is typically far wider than the one a
  daemon inherits from systemd. ALRB also ships its own apptainer under CVMFS
  and supplies it to the container setup when the host has none, so "no
  apptainer on this host" never implied "no apptainer for the analysis" in the
  first place.

  On aipanda033 all three CVMFS checks passed and the tool still refused with
  "apptainer is not on PATH", for job 7272161793.

  Detection now runs four avenues in `find_container_runtime`: the new
  `BAMBOO_CORE_DUMP_APPTAINER` override, the process PATH, a login shell's PATH
  (probed once per process with `bash -lc 'command -v …'`, negative results
  cached, every failure mode contained to `None`), and ALRB's bundled apptainer
  under the CVMFS repository. `singularity` is accepted alongside `apptainer`
  for older EL7 deployments. The refusal now names all three discovery avenues
  and both escape hatches, so a reader cannot conclude they should fix the
  wrong PATH. `BAMBOO_CORE_DUMP_SKIP_RUNTIME_CHECK=1` drops the runtime check
  while leaving the CVMFS checks in place.

  The no-fallback-to-local-gdb rule is untouched. It governs *fallback*; this
  governs *detection*, and the two failure modes are independent — a detection
  false negative refuses an analysis that would have been correct, which is its
  own kind of wrong answer.

- **The log-analysis synthesis prompt was blind to the core-dump probe**
  (`core/bamboo/tools/bamboo_executor.py`). Rounds 2 and 4a added
  `core_dump_probe_state`, `core_dump_available`, `core_dump_candidates` and
  `core_dump_total_bytes` to the log-analysis evidence, but `_SYSTEM_LOG_ANALYSIS`
  was never updated to describe any of them. Worse, it carried the rule
  *"never infer a root cause from a directory listing or from file sizes in the
  log excerpt"* — and `core_dump_candidates` is a list of records with a `name`
  and a `size_bytes`. The model was being told to disregard the exact shape of
  what the probe emits.

  The result was not a consistent failure but an inconsistent one. Job
  7263015032 was answered once with a section headed "Root Cause — Core Dump"
  built around `core.20115`, and thirteen minutes later with "No core dump is
  visible in the pilot log evidence — the title mentions 'core dump' but the
  actual failure mode is a looping/hung job." Same job, same probe output,
  opposite conclusions, neither flagged as uncertain.

  The prompt now documents all four keys and all five probe states, and the
  listing prohibition is scoped to what the model reads out of `log_excerpt`,
  with an explicit note that the probe keys are structured evidence governed by
  the rules that follow. Those rules make `core_dump_probe_state` authoritative
  on existence, require `truncated` and `timed_out` to be reported as
  themselves rather than flattened into "no core dump", and require
  `not_probed` / `core_dump_available: null` to be reported as *not looked at*
  rather than as absence.

  A separate rule stops the model treating the core dump as the diagnosis. The
  earlier answer reasoned from the core's modification timestamp to a crash
  "approximately 2 hours before the pilot killed the job", which the timestamp
  does not support — it records when the core was written, which for a
  pilot-killed job is usually the kill. Only the analysed stack says where the
  process died.

  Also added: an instruction not to write its own "shall I analyse it?" offer,
  matching the existing one for pilot source. The offer is appended from
  `core_dump_offer_md` after synthesis, and a model-invented offer has nothing
  behind it — the follow-up intercept keys on the stored offer, not on the
  answer text, so "yes please" against an invented offer finds nothing.

- **`_SYSTEM_CODE_QUERY` was unreachable** (`core/bamboo/tools/bamboo_executor.py`).
  The synthesis prompt table keyed on `pilot_code_query`, a name no tool has
  ever been called — `code_query` is both the advertised and the wire name, and
  `pilot_code_query` survives only in a docstring. Every `code_query` answer has
  been synthesised with the generic prompt rather than the specialist one.
  Found by the guard test below, not by inspection: the failure mode is a
  slightly worse answer, which nothing reports.

  The table is now a module-level `_TOOL_PROMPT_TABLE` rather than a local
  rebuilt on each call, so a test can assert its keys are reachable at all. It
  could not before, which is how the table accumulated two dead entries — this
  one and `pilot_source_analysis`, fixed earlier in this change.

- **A plan's spelling of a tool name decided whether its evidence was
  readable** (`core/bamboo/tools/bamboo_executor.py`,
  `core/bamboo/tools/_tool_names.py`). `_execute_one_tool` keyed
  `_last_evidence_store` and `called_tool_names` on the literal `tc.tool`
  string. `_resolve_tool` accepts several spellings of the same tool, so a plan
  naming `core_dump_analysis` ran exactly the same object as one naming
  `atlas.core_dump_analysis` — and then recorded it under a key no reader looks
  up. `_core_dump_evidence()` returned `{}`, `_CORE_DUMP_TOOL in
  called_tool_names` was False, `_synthesise_core_dump` was skipped, and the
  analyzer's JSON went to generic prose synthesis with `reconcile_llm_analysis`
  never called. Nothing raised; the answer was simply worse, and worse in the
  specific way reconciliation exists to prevent.

  The name recorded is now the canonical wire name. Exact-literal readers
  (`_CORE_DUMP_TOOL`, `"panda_log_analysis"`, `_pick_synthesis_prompt`'s table)
  are correct again without any of them having to enumerate aliases. The plan's
  own spelling is retained for error text, where echoing what the caller wrote
  is what makes `Unknown tool: …` diagnosable.

  `bamboo_last_evidence` canonicalises its `tool` argument for the same reason:
  the store is keyed on wire names, so a caller asking for
  `core_dump_analysis` must reach the entry written under
  `atlas.core_dump_analysis`.

  `alias_map()` is memoised. Building it loads every entry point in the group —
  ~230 ms — and canonicalisation now sits on the execution path once per tool
  call; measured 229 ms → 0.9 µs warm. Entry points cannot change without a
  process restart, so there is no natural invalidation; `reset_alias_cache()`
  exists for tests that patch discovery.

  New `tests/test_tool_name_canon.py` pins the agreement rather than the
  mechanism: both spellings of the core-dump tool converge on one store key,
  catalog names are a subset of wire names, and every namespaced name in the
  planner prompt is catalogued.

- **The planner catalog advertised tool names the server does not expose**
  (`core/bamboo/tools/_tool_names.py`, new; `core/bamboo/core.py`,
  `core/bamboo/tools/planner.py`). `_collect_tool_catalog` took a plugin tool's
  name from its own `get_definition()["name"]`, passing the entry-point key only
  as a fallback. `bamboo.core` does the opposite: it *overwrites* the definition
  name with the entry-point key, unless the advertised name is already a
  built-in `TOOLS` registration, in which case the entry point is dropped
  entirely. The two rules disagreed for four of the five plugin tools on the
  wire.

  The user-visible consequence was on `atlas.core_dump_analysis`. The planner's
  routing guidance names it correctly, but the catalog offered
  `core_dump_analysis` and the prompt's hard rule is to propose only catalogued
  tools. Faced with a contradiction the planner discarded the guidance and fell
  back to `panda_log_analysis` — which is catalogued, and which independently
  matches "diagnose/why a job failed" — so an explicit request to analyse a core
  dump was answered with a log analysis.

  The near miss is worse than the miss. Had the planner instead proposed the
  catalogued `core_dump_analysis`, `_resolve_tool` would have resolved it and
  the tool would have run, but `_execute_one_tool` keys `_last_evidence_store`
  and `called_tool_names` on the literal string the plan used. `_CORE_DUMP_TOOL
  in called_tool_names` would have been False, `_synthesise_core_dump` would
  have been skipped, and the analyzer's JSON would have gone to generic prose
  synthesis with `reconcile_llm_analysis` never called — the one step that stops
  a model reading EventLoop completion markers as evidence that a looping
  payload exited normally. Step 2 of this change closes that path directly.

  `wire_tool_definitions()` is now the single implementation of the naming and
  de-duplication rules; `bamboo.core` and the planner both call it, so the
  catalog equals the wire surface by construction rather than by agreement.
  Definitions are copied before renaming — a tool whose `get_definition`
  returned a module-level constant was previously renamed in place by the first
  call.

- **`pilot_source_analysis` was named inconsistently against the wire**
  (`core/bamboo/tools/bamboo_answer.py`, `core/bamboo/tools/bamboo_executor.py`,
  `core/bamboo/tools/planner.py`, `interfaces/streamlit/chat.py`). Rule 1b, the
  `_pick_synthesis_prompt` table, the planner guidance and the Streamlit
  plot-exclusion set all used the unqualified name, which is what the tool
  advertises but not what clients can call: it is not in built-in `TOOLS`, so
  its wire name is `atlas.pilot_source_analysis`. Internally self-consistent, so
  nothing failed — but an external MCP client calling the tool by the name
  `list_tools` gave it got generic synthesis instead of `_SYSTEM_PILOT_SOURCE`.
  All four sites now use the wire name.

  This was scheduled as a later step and pulled forward because it stopped being
  separable: once the catalog reports wire names, leaving the other sites on the
  advertised name would have broken planner-routed pilot source analysis at the
  commit boundary.

  The tool's own `get_definition()["name"]` is deliberately unchanged. It is the
  input to the rename, not a duplicate of it, and the ePIC mirror pins it.

- Removed a dead `pilot_source_analysis` entry from planner
  `_PANDA_CORE_TOOLS`. That set filters the built-in `TOOLS` dict, which has
  never contained the key, so the entry matched nothing; non-PanDA plugins are
  excluded from the tool by the namespace filter instead.

### Added
- **Guards tying the log-analysis prompt to the evidence it receives**
  (`tests/test_log_analysis_prompt.py`). A prompt cannot be unit-tested for
  whether a model obeys it, but it can be tested that every probe key reaching
  the model and every state value the probe can emit is at least named in it.
  The key guard derives its expectations from
  `_build_core_dump_evidence()` minus `_PRESENTATION_KEYS` rather than from a
  hand-written list, so a new key is covered the moment it is emitted — it
  caught `core_dump_available` missing from the first draft of the prompt
  rewrite above.

  Verified by mutation: reverting the listing prohibition, renaming a probe
  state, or dropping the offer rule each fails the corresponding guard. Skipped
  via `importorskip` where the ATLAS plugin is absent, since the probe
  constants live there.

- **Drift guards for tool-name literals** (`tests/test_tool_name_canon.py`).
  Three sites in core compare exact strings against `called_tool_names`, and a
  literal that is nearly right selects no specialist behaviour and falls
  through to a generic path without raising. The guards assert
  `_TOOL_PROMPT_TABLE` keys, the doc-tool sets, and every
  `ToolCall(tool="...")` literal in `bamboo_answer` are names the MCP server
  actually exposes.

  The `bamboo_answer` literals are collected by walking the module's AST rather
  than by exercising the routing functions. Those rules are gated on question
  text, history and stored evidence, so a behavioural test pins only the
  branches someone remembered to write a case for; the AST sees all 20
  literals, including in rules added later. This is the guard for the
  "three-file edit problem" — adding a plugin tool means touching
  `bamboo_answer`, `planner` and `bamboo_executor`, and a wrong name in one of
  them is silent.

  **Membership, not idempotence.** The first draft asserted each literal was a
  fixed point under `canonical_tool_name`, and mutation testing showed it did
  not catch `pilot_code_query` at all: an unknown name canonicalises to itself
  by design, so a fixed-point check passes for a name that is not a tool. The
  guards assert membership in the wire surface instead. Verified by mutation —
  reverting the `code_query` key, the `pilot_source_analysis` key, rule 1b's
  `ToolCall` name, or the planner catalog to advertised names each fails the
  corresponding guard and nothing else.

  A namespaced literal whose plugin is absent is filtered out, not skipped:
  `pytest.skip` from inside the loop would abandon the whole test, leaving
  every ATLAS name unchecked while the run still looked green. Each guard also
  asserts it checked something, so it cannot go inert.

---

## v1.0.8 — 2026-08-20

### Added
- **Core-dump acquisition layer**
  (`packages/askpanda_atlas/askpanda_atlas/_job_prep.py`, new). Reconstructs a
  PanDA job directory from BigPanDA's unauthenticated media path so the vendored
  analyzer can run against it. `select_files_for_fetch()` is pure policy and
  mirrors the analyzer's own `discover_job_logs` rules against the *listing*, so
  the rebuilt directory contains what discovery would have chosen and nothing
  else — on the reference looping job that is 774 kB out of a 119 MB tarball,
  with five copies of `output.root`, 50 cmake modules under `workDir/usr`, and
  the pilot log all skipped and each recorded in `FetchPlan.skipped` with its
  reason.

  Three properties are load-bearing and each is pinned by a test:

  - **Media URLs are constructed, never read from `media_link`.** BigPanDA omits
    the `dirname`/`name` separator for nested entries, so its own link for
    `workDir` + `in.txt` is `.../workDirin.txt`. The link is correct only when
    `dirname` is empty, which is exactly the case where it adds nothing.
  - **Modification times are restored with `os.utime()` from the listing,
    parsed as UTC.** For a looping job the strongest deterministic observation
    is how long the payload had been silent when the core was captured, and it
    is computed purely from mtimes. A directory rebuilt with "now" as every
    timestamp loses that observation *silently* — the analysis still runs and
    simply omits it. The round-trip test therefore pins the reference value
    exactly (7774.0 s / `"2h 09m 34s"`) rather than within a tolerance, and was
    verified to fail when `os.utime()` is removed.
  - **The `workDir` recency window is anchored on the payload streams, not the
    core.** A looping job's payload is silent for hours before the core is
    written, so a window measured backwards from the core discards precisely the
    files that were live when the payload stopped. Empty streams are excluded
    from the anchor: a zero-length `payload.stderr` carries no activity
    information and would move the cutoff for no reason.

  `_find_core_dump_candidates` is imported from `log_analysis_impl` rather than
  reimplemented, so the core that gets analysed is necessarily the one the probe
  named when it offered the analysis.

- **Uncached binary media primitives**
  (`packages/askpanda_atlas/askpanda_atlas/_cache.py`). `head_remote_file()` for
  authoritative size, byte-range support and modification time; `stream_to_file()`
  for streaming a core straight to disk. Neither touches the cache store: routing
  a gigabyte-scale core through `cached_fetch_log()` would decode it into a `str`
  via `resp.text` and pin it under `LOG_TTL` (`math.inf`) for the lifetime of the
  process.

  Three guards, each answering an observed failure mode. A `text/html` body is
  rejected **before any byte is written** — the SSO-gated endpoint answers an
  unauthenticated request with an HTTP **200** login page, so the status code
  alone cannot distinguish it from success and `curl -f` does not catch it. The
  transferred length is verified against the expected size, because a short
  transfer that ends cleanly is otherwise indistinguishable from a complete one.
  Bytes land in `<dest>.part` and are renamed onto the destination only once both
  guards pass, so the real path never exists in a truncated state. A failed
  transfer deliberately keeps its `.part` file: it is the input to a resumed
  retry, and nothing in this package deletes files.

  Resume support is opt-in and refuses to engage where it would be unsafe —
  without a known final size a partial file cannot be told from a complete one,
  and a server that answers **200** to a `Range` request has sent the whole body,
  so the carried-over prefix is discarded rather than appended to (which would
  silently produce a corrupt file of plausible length).

- **Core-dump analysis routing, synthesis and documentation** — step 5, the
  final step of the core-dump work. Registers `atlas.core_dump_analysis` at the
  five points a plugin tool must appear, and adds the synthesis path the tool
  itself deliberately does not have.

  **Synthesis lives in `bamboo_executor`, not in the tool.**
  `_complete_via_bamboo` refuses to run inside a live event loop and its own
  error text names the alternative: collect with `--no-llm`, synthesise through
  the caller's provider stack. `_synthesise_core_dump()` is that caller. It is
  the only tool whose evidence bypasses `_build_synthesis_prompt` entirely,
  because the analyzer ships its own prompt pair and its own response schema and
  returns JSON rather than prose.

  Four properties are load-bearing:

  - **A non-terminal run is answered with its own progress line, no LLM call.**
    There is no evidence to reason about while a run is still downloading; an
    LLM call there could only paraphrase a status message, and at worst would
    invent findings for an analysis that has not produced any. Follows the
    `_try_cric_direct_format` and cgsim-summary bypass precedents.
  - **`reconcile_llm_analysis()` is always called on the model output.** It is
    what stops the model reading EventLoop completion markers as evidence that
    a looping payload exited normally — those markers describe event-processing
    state, not payload exit.
  - **`acquisition.warnings` are appended to `analysis["limitations"]` after
    reconciliation**, deterministically, never through the model. The ordering
    is not cosmetic: `reconcile_llm_analysis` drops list entries that read as
    claims of normal job success, so appending first would let a legitimate
    acquisition warning be filtered back out. Verified by mutation — swapping
    the two lines fails exactly one test.
  - **`core_evidence` is not shrunk a second time.** It has already been through
    `enforce_global_budget` at 50 000 chars inside the tool; a second pass would
    discard thread groups the first pass decided were worth keeping.

  The analyzer imports are deferred inside the function, so bamboo core stays
  importable wherever the ATLAS plugin is not installed. `render_report` is
  deliberately not reused — its own docstring names it the CLI's fixed-width
  presentation and tells embedders to render from the dicts directly.

  **Routing: two entry points, in two different layers.** Rule 1c (explicit
  request naming a job) is a `_build_deterministic_plan` rule, ahead of rule 1
  because "analyse the core dump of job X" independently matches
  `_is_log_analysis_request`. Rule 1d (a bare affirmative accepting the stored
  offer) is a `_route` branch instead, because both layers below it would
  consume the affirmative first: `_is_ack` matches "ok", "okay", "great" and
  "perfect", and the topic guard rewrites content-free follow-ups before the
  deterministic planner sees them. The promptlog fast path sits early for
  exactly this reason. Verified by mutation — moving rule 1d after the social
  intercept fails exactly one test.

  Rule 1c passes `mode="auto"` since an explicit request may name any job; rule
  1d pins `mode="hang"`, since the offer is only ever made for a looping-job
  kill (pilot code 1150), which removes a metadata round trip and the failure
  mode where that fetch fails and leaves the framing unresolved.

  **"Why did job X hang" is deliberately not a rule 1c signal.** It is a
  diagnosis request, and routing it straight to a multi-gigabyte core fetch
  holding the single analysis slot is the wrong opening move for a question
  `panda_log_analysis` usually answers outright — and when it does not, that
  same log analysis emits the offer rule 1d catches. The expensive path is
  reached by naming it or by accepting it, never by guessing.

- **`tests/test_core_dump_routing.py`** (new, 39 tests) and
  **`tests/test_core_dump_synthesis.py`** (new, 19 tests). Rule 1d is tested
  end-to-end through `_route` rather than at the plan layer, because a plan-layer
  test would pass while the shipped path still answered "You're welcome".

- **`packages/askpanda_atlas/tests/test_job_prep.py`** (new, 60 tests) and 16
  further tests in `tests/test_cache.py`. Sockets are blocked by an autouse
  fixture so any path reaching the real HTTP layer fails loudly instead of
  falling through to a 403 and passing for the wrong reason.

### Fixed
- **`core_dump_offer_md` was built but never reached the user.**
  `log_analysis_impl._build_core_dump_evidence` has emitted the key since round
  2, but it was absent from `_PRESENTATION_KEYS` and nothing appended it after
  synthesis — so it was passed to the synthesis LLM as ordinary evidence. That
  is precisely the failure the `code_analysis_offer_md` comment documents: a
  ready-made offer string sitting in the model's input reliably beats an
  instruction not to reproduce it, and the canonical copy is then appended on
  top. The offer is now a presentation key, hidden from the model and appended
  programmatically, so the file name and size the user sees are the listing's
  own. Rule 1d had nothing to fire on until this was fixed.

- **`packages/askpanda_atlas/askpanda_atlas/core_dump_worker.py` was missing its
  leading underscore.** `WORKER_MODULE` names `askpanda_atlas._core_dump_worker`,
  the test suite imports `_core_dump_worker`, and the module's own docstring
  documents `python -m askpanda_atlas._core_dump_worker`. Every other private
  module in the package keeps the underscore. Renamed. Left unfixed this fails
  at runtime in `spawn_worker()` with no test catching it, because the suite
  fails at import instead.

- **`BambooAnswerTool._route` was at the `max-complexity = 15` ceiling exactly**,
  so any added branch broke lint. The social intercept and rule 1d are now
  grouped in `_run_early_intercepts()`, which also makes their ordering explicit
  rather than incidental; `_route` drops to 14. Behaviour unchanged.

- **`scripts/README-core_dump_analysis.md` now links to
  `docs/tools/core_dump_analysis.md`.** The CLI README had carried a plain
  sentence in place of the link while the tool doc did not yet exist.

- **`test_saved_looping_cases_share_family_but_have_distinct_subtypes` read
  fixtures from `/mnt/data/`**, a path outside the repository, so the test could
  only pass on the machine that happened to hold the two saved evidence bundles
  and failed everywhere else. Both bundles are now committed as
  `packages/askpanda_atlas/tests/fixtures/core-analysis{7,9}.json` and the test
  resolves them relative to its own location. They are the only regression
  evidence for the `post-event-processing-xrootd-shutdown-hang` family and its
  `poller-finalization` / `remote-file-close` subtypes, which are derived from
  real validated looping jobs and cannot be reconstructed synthetically.

- **`scripts/analyze_core_dump.py` pointed at `scripts/README-analyze_core_dump.md`**,
  which has never existed; the file is `scripts/README-core_dump_analysis.md`.

- **`bamboo_env_example.sh` still documented `BAMBOO_MCP_CLIENT_TIMEOUT="120"`**,
  the pre-300 s default. Commented out, so nothing was pinned — but as
  documentation it contradicted the code and would have restored the 30/120 s
  ceiling for anyone who uncommented it. Now documents both
  `BAMBOO_MCP_CLIENT_TIMEOUT` and `BAMBOO_MCP_HTTP_TIMEOUT` at 300 s, and states
  that they are independent ceilings on the same call where the lower one
  silently wins.

### Fixed (earlier)
- **ePIC delegation test's fetch mock had the pre-`repo` signature**
  (`packages/askpanda_epic/tests/test_pilot_source_analysis_epic.py`).
  `TypeError: _fetch() takes 2 positional arguments but 4 were given` — the ATLAS
  mocks were updated when `fetch_pilot_module` gained `ref` and `repo`, but the
  ePIC delegation test's was not.

  It slipped through because the ePIC suite cannot run this test unless
  `askpanda_atlas` is importable; without it the delegation falls back to a stub
  and the test is already failing for an unrelated reason. Verification that
  compared *failure lists* therefore treated a genuinely broken test as
  pre-existing noise. Confirmed fixed with `askpanda_atlas` on the path, where the
  full ePIC suite is 89/89 in both this tree and a pristine baseline.

- **`test_pilot_source_analysis_epic.py` reached the real GitHub.** The module
  docstring claims "All external HTTP calls are patched; no network access is
  required", but three tests patched only `fetch_pilot_module`, leaving
  `resolve_source_ref`'s probe to hit `raw.githubusercontent.com` — and because
  the probe seeds the source cache, a test could pass on live source while
  asserting fetches had failed. Added the same autouse `_no_raw_github_fetch`
  fixture already guarding the ATLAS suite.

- **Stale `resolve_github_ref` references** in the
  `fetch_and_analyse_pilot_source` docstring and a comment in
  `test_pilot_source_analysis.py`, left by the rename to `resolve_source_ref`.
  The docstring also still described the old master-fallback behaviour rather than
  release-tag vs development-branch selection.

- **mcp pinned below 2.0.0 — the low-level Server decorator API was removed**
  (`requirements.txt`, `requirements-ui.txt`, `pyproject.toml`). Surfaced as four
  pyright `reportAttributeAccessIssue` errors in CI on
  `core/bamboo/core.py:280,318,370,393` (`list_tools`, `call_tool`,
  `list_prompts`, `get_prompt` unknown on `Server[Any]`).

  This was a true positive, not type-checker noise. `core.build_server()`
  registers handlers with the low-level decorator API; mcp 2.0.0 removed it.
  Verified directly: under mcp 1.29.0 all four attributes exist and pyright is
  clean, under 2.0.0 none exist.

  The requirement was `mcp>=0.9.0` with no upper bound, so a fresh CI install
  resolved 2.0.0 while developer machines kept an older 1.x — which is why the
  error appeared only in CI. Worse, the failure is near-silent: those decorators
  run inside `build_server()`, which the test suite never calls, so all 1125
  tests passed against an mcp version that cannot start the server at all. Only
  pyright caught it. Suppressing the errors inline would have hidden a
  production start-up failure, so the fix is the pin.

- **Renamed routing gate left five tests patching a nonexistent attribute**
  (`tests/test_bamboo_answer_helpers.py`). `get_last_pilot_monitoring_evidence`
  → `get_last_traceback_evidence` was applied to `bamboo_answer`'s import but not
  to the tests that `patch.object` it, so all five raised `AttributeError`.
  Repointed at the new name.

  Deliberately *not* fixed by re-importing the old alias into `bamboo_answer`:
  the routing code calls the new name, so a patched alias would be silently
  ineffective and the tests would pass while intercepting nothing. The
  `AttributeError` was the correct outcome.

### Added
- **`tests/test_mcp_server_api_compat.py`.** Verifies the installed mcp is one
  whose `Server` API `core.build_server()` can use, so lifting the `mcp<2.0.0`
  pin without porting `core.py` fails a test instead of failing at server
  start-up.

  The check deliberately does not `import mcp`. `tests/conftest.py` assigns
  `MagicMock` to `mcp.server.Server`, so an in-process import inside the test
  session returns the stub and the attribute assertions fail for the wrong
  reason — the first version of this test did exactly that. Instead it reads the
  version from distribution metadata and probes the real API in a subprocess with
  a clean interpreter, matching the conditions under which `build_server()`
  actually runs. Both checks earn their place: on a half-upgraded install the
  metadata reported 1.29.0 while the subprocess correctly found all four
  decorators missing.

  Confirmed to pass under mcp 1.29.0 and fail under 2.0.0 *with conftest's stub
  active*, which is the condition that matters.

- **Corrected the mcp stubbing comment in `tests/conftest.py`.** It claimed
  "setdefault is a no-op when the real package is already present, so a genuine
  mcp installation always wins". That is wrong: `sys.modules.setdefault` only
  no-ops if the module has already been *imported*, not merely installed, so in a
  fresh session the stub shadows a real mcp too. No test in the suite exercises
  the real mcp API, which is the second half of why the 2.0.0 breakage stayed
  invisible behind 1132 passing tests. The stubbing behaviour itself is left
  unchanged — altering it would change the import environment for the whole suite
  — but the comment now describes what actually happens.
- **Routing coverage for the widened gate**
  (`tests/test_bamboo_answer_helpers.py`): a case using the modern evidence shape
  (`traceback_available` + `deepest_pilot_frame` with an unrelated
  `failure_type`) that would not have routed under the old
  `pilot_monitoring_error` gate, asserting `pilot_version` is threaded through to
  `pilot_source_analysis`; and a case confirming `_PILOT_SOURCE_SIGNALS` covers
  the wording of the offer `panda_log_analysis` appends, so accepting the offer
  by echoing it is not a dead end. Neither the widened gate nor the
  `pilot_version` argument had any test coverage before.

- **ePIC plugin copies had drifted from their ATLAS originals**
  (`packages/askpanda_epic/askpanda_epic/log_analysis_impl.py`,
  `packages/askpanda_epic/askpanda_epic/_traceback_parse.py`). Surfaced as a
  pyright `reportAttributeAccessIssue` on `_TRACEBACK_TRAILING_LINES` in
  `test_log_analysis_epic.py`.

  Root cause: both ePIC files were mirrored from their ATLAS counterparts at a
  point in the session, and both ATLAS files were then changed again without
  re-mirroring. Two fixes never reached ePIC:
  `_strip_directory_listing`/`_LISTING_LINE_RES` (so the ePIC excerpt builder
  went on pulling `ls -l` directory listings into the LLM context — the very
  artifact that caused the original misdiagnosis), and the chained-traceback fix
  in `find_traceback_blocks` (so the ePIC copy still split
  `During handling of the above exception...` chains into two blocks and
  reported the first exception rather than the one that propagated).

  Neither plugin's own test suite could catch this, because each suite passes
  against its own copy; only a cross-package comparison can. Both copies are
  regenerated, and drift is now a test failure rather than a downstream
  type-check error — see Tests.

### Added
- **`tests/plugin_mirror_spec.py` and `tests/test_plugin_mirror_parity.py`.**
  The atlas→epic naming substitutions are now recorded as data in
  `plugin_mirror_spec.MIRRORS`, so the ePIC copies can be regenerated
  mechanically instead of by hand. Three guards: the ePIC copy must equal the
  ATLAS source with substitutions applied (failing with a unified diff and
  regeneration instructions); the ePIC copy must not `import` from
  `askpanda_atlas`, checked via `ast` rather than substring search because the
  docstrings legitimately mention `askpanda_atlas` in prose; and every
  substitution must still match the ATLAS source, so the table cannot rot
  silently while hiding a real unrecorded difference.

- **Code-analysis offer rendered twice** (`core/bamboo/tools/bamboo_executor.py`).
  Observed on job 7261310898: the answer ended with the "Ask me to show the
  pilot source" offer two times over, the LLM's copy without its backticks.

  Root cause: `_LLM_STRIP` filters only the *top level* of the unpacked tool
  result (`evidence`, `text`), but `code_analysis_offer_md` and `links_md` sit
  inside the nested `evidence` dict. Both were therefore handed to the synthesis
  LLM as literal Markdown strings; the LLM copied the offer into its answer and
  `bamboo_executor` then appended the canonical copy. The prompt instruction
  "Do not offer to fetch the pilot source code; that offer is appended
  automatically" lost to a ready-made string sitting in the input — the same
  reason `_strip_llm_links_section` exists rather than trusting the parallel
  "Do not include a Links section" instruction.

  New `_strip_presentation_keys()` removes `_PRESENTATION_KEYS` from both the
  top level and the nested `evidence` dict before synthesis, without mutating
  the input. The strip applies to the LLM's view only: `_last_evidence_store`
  must retain these keys because `_log_analysis_offer_md` and
  `_log_analysis_links_md` read them back from there. The append site also gains
  an idempotence guard. Removing `links_md` from the LLM's view should
  additionally reduce the invented `Links:` sections that
  `_strip_llm_links_section` currently mops up.

- **Duplicate fetch of the probed pilot3 module.** `resolve_github_ref` fetched
  `unique_paths[0]` to test a candidate ref and discarded the content, after
  which the main loop downloaded the same file again. The probe response is now
  retained in `SourceRef.probe_text` and seeds the source cache.

### Changed
- **Pilot3 source selection distinguishes released from unreleased builds**
  (`packages/askpanda_atlas/askpanda_atlas/pilot_source_analysis_impl.py`,
  `core/bamboo/tools/bamboo_executor.py`).

  Job 7261310898 ran pilot 3.14.1.27, which has no release tag. The previous
  resolver fell back to `PanDAWMS/pilot3@master` — which by definition is *not*
  what an unreleased build ran, so it would have presented released code as the
  source of a development-build traceback.

  Released and unreleased pilots live in different places and the two paths are
  now mutually exclusive:
  - A version with a matching tag is a released build: read the tag in
    `PanDAWMS/pilot3`. The development branch is never consulted.
  - A version with no tag is an unreleased build: read `PalNilsson/pilot3@next`.
    `master` is never consulted.
  - `master` is used only when `pilot_version` is unknown, since the build then
    cannot be classified at all and reaching into the development fork on a
    guess would be worse.

  `resolve_github_ref` becomes `resolve_source_ref` returning a `SourceRef`
  (repo, ref, kind, resolution, probe_text, reachable); `_raw_url`/`_browse_url`
  and `fetch_pilot_module` gain a `repo` parameter. New evidence keys
  `github_repo` and `ref_kind`. Because `next` is a *moving* branch, line
  numbers are indicative even when `verify_frame_lines` detects no skew — a
  function can shift lines without changing name — so `ref_resolution`, the text
  summary and `_SYSTEM_PILOT_SOURCE` all say so, and the prompt now branches on
  `ref_kind` rather than parsing prose. Locations are overridable via
  `BAMBOO_PILOT3_REPO`, `BAMBOO_PILOT3_BRANCH`, `BAMBOO_PILOT3_DEV_REPO` and
  `BAMBOO_PILOT3_DEV_BRANCH`, following the `BAMBOO_CODE_QUERY_*` convention;
  the development fork is expected to move to a BNLNPPS organisation.

### Tests
- Presentation-key stripping tests in `tests/test_log_analysis.py`, including
  that the input is not mutated (the evidence store depends on it).
- `TestResolveGithubRef` replaced by `TestResolveSourceRef`, asserting the
  mutual exclusivity directly: a released version never probes `PalNilsson` or
  `next`, and an untagged version never probes `master`. Plus dev-branch
  resolution, unreachable-candidate reporting, env overrides, probe reuse, and
  an end-to-end unreleased-version case built from job 7261310898.
- **New autouse `_no_raw_github_fetch` fixture** in
  `test_pilot_source_analysis.py`. Seeding the cache from the probe means a test
  that patches only `fetch_pilot_module` now reaches the real
  `raw.githubusercontent.com` through `_fetch_raw` — `test_fetch_error_recorded`
  did exactly that and passed for the wrong reason, extracting a snippet from
  live GitHub source while asserting all fetches had failed. The fixture makes
  "no network" the default so no future test can leak silently; tests needing a
  successful probe patch `_fetch_raw` explicitly.

### Added
- **Traceback-first log excerpt extraction and pilot exception evidence**
  (`packages/askpanda_atlas/askpanda_atlas/_traceback_parse.py` — new,
  `packages/askpanda_epic/askpanda_epic/_traceback_parse.py` — new,
  `log_analysis_impl.py` in both plugins, `core/bamboo/tools/log_analysis.py`,
  `core/bamboo/tools/bamboo_executor.py`, `core/bamboo/tools/bamboo_answer.py`,
  `core/bamboo/tools/planner.py`,
  `packages/askpanda_atlas/askpanda_atlas/pilot_source_analysis_impl.py`).

  **Root cause.** `extract_log_excerpt()` anchored its context window on a
  per-error-code search string from `_PILOT_CODE_PATTERNS`, falling back to
  `re.escape(piloterrordiag[:40])` used as a literal regex. `piloterrordiag`
  is a summary written by a different pilot code path than the log record, so
  the wordings routinely differ. For pilot error code 1310 the metadata reads
  `"Exception caught during payload execution"` while the log record reads
  `"execute payloads caught an exception (cannot recover): timed out,
  Traceback ..."` — no match. Extraction then fell back to
  `_extract_tail(log_text, 40)`, and the last 40 lines of a failed job's
  `pilotlog.txt` are stage-out and log-archiving boilerplate: `removed
  /tmp/...` lines, an `ls -lF` directory listing and a `tar cvfz` command.

  Given only that, the synthesis LLM inferred a root cause from the *file
  sizes* in the directory listing and reported job 7261310898 as a "remote
  file open failure / stage-in problem". The actual cause was a
  `TimeoutError` fetching the runGen transform over HTTP inside
  `get_analysis_trf` → `download_transform` → `download_file`: the payload
  never started. `failure_type` also came out as `"timeout"` — correct by
  accident, matched from the substring `"using timeout=90 s"` in the pilot's
  own `tar` command.

  Adding `1310` to `_PILOT_CODE_PATTERNS` would have fixed that one job and
  left the next unmatched code broken, so extraction was re-anchored on two
  format-level invariants that hold regardless of error code, pilot version or
  experiment: the `YYYY-MM-DD HH:MM:SS,mmm | LEVEL |` pilot log record prefix
  (whose *absence* identifies continuation lines, making the record — not a
  line count — the right extraction unit), and the `Traceback (most recent
  call last):` → indented `File "...", line N, in func` frames → unindented
  `ExceptionType: message` shape.

  - New `_traceback_parse.py` provides `find_traceback_blocks`,
    `select_primary_traceback`, `parse_exception`, `find_primary_exception`,
    `parse_pilot_version`, `parse_pilot_version_from_pilotid` and
    `truncate_traceback`. Handles chained (`During handling of the above
    exception...`) tracebacks as a single block, Python 3.11+ `~~~^^^` column
    markers, and both prefixed pilot logs and unprefixed payload logs through
    one code path. `Frame` carries `file`, `lineno`, `func`, `pilot_path` and
    `is_pilot`, so CVMFS/stdlib frames are retained for diagnosis while
    `deepest_pilot_frame` still resolves the pilot-owned failure locus.
  - New `extract_failure_context()` returns a `FailureContext` (excerpt +
    parsed exception + traceback count); `extract_log_excerpt()` retained as a
    thin wrapper so existing callers and tests are unaffected.
  - Traceback-first extraction applies to `pilotlog.txt`, `payload.stdout`,
    `payload.stderr` and `setup.stdout` — the traceback format is identical in
    all four, so Athena payload failures benefit as much as pilot failures.
    When both payload files contain a traceback, stderr's wins (that is where
    Python tracebacks and abort reports are written).
  - New evidence keys: `traceback_available`, `exception_type`,
    `exception_message`, `exception_frames`, `deepest_pilot_frame`,
    `traceback_count`, `pilot_version`, `code_analysis_offer_md`. All keys are
    always present (`None`/`False` when absent) so consumers can rely on the
    shape rather than probing with `in`.
  - New `_classify_from_exception()` runs *before* the `_FAILURE_PATTERNS`
    substring table, keyed on exception type and the pilot call chain rather
    than substring presence anywhere in the excerpt. New categories:
    `transform_download_timeout`, `transform_download_failed` and
    `pilot_exception` — the last preferred over `payload_error`, which
    actively misleads when the exception was raised while *building* the
    payload command and the payload never ran. `pilot_monitoring_error` is
    preserved by name because `bamboo_answer` and `planner` route on it.
  - `_SYSTEM_LOG_ANALYSIS` now states that the parsed exception is
    authoritative over `piloterrordiag`, instructs the LLM to say so
    explicitly when they disagree, and forbids inferring a root cause from
    directory listings or file sizes — the exact failure mode above.

- **Pilot source analysis pinned to the job's pilot release tag**
  (`packages/askpanda_atlas/askpanda_atlas/pilot_source_analysis_impl.py`).
  Pilot releases are tagged after release (e.g. tag `3.14.0.22`), so a
  traceback's line numbers are only meaningful against the tag the job ran;
  fetching `master` silently misreports them for any module changed since,
  which for an actively developed file like `pilot/util/https.py` is most of
  the time. `resolve_github_ref()` probes the bare tag then a `v`-prefixed
  variant with a real content fetch, falling back to `master` and recording
  why in `ref_resolution`. `function_at_line()` and `verify_frame_lines()` use
  `ast` to confirm the function at each traceback line matches the frame's
  name, reporting `line_verification.version_skew`; `_SYSTEM_PILOT_SOURCE`
  instructs the LLM to describe functions by name rather than line number when
  skew is detected. `pilot_version` added to the tool input schema and
  threaded through from `panda_log_analysis` evidence via `bamboo_answer`.

### Changed
- **`pilot_source_analysis` is reachable for any pilot traceback, not only
  `pilot_monitoring_error`** (`core/bamboo/tools/bamboo_executor.py`,
  `core/bamboo/tools/bamboo_answer.py`, `core/bamboo/tools/planner.py`).
  `get_last_pilot_monitoring_evidence()` gated on `failure_type ==
  "pilot_monitoring_error"`, which made source-level analysis unreachable for
  every other kind of pilot exception — including the transform-download
  timeouts above. Renamed to `get_last_traceback_evidence()` and re-gated on
  `traceback_available` plus a non-null `deepest_pilot_frame` (required
  because the tool fetches pilot3 modules from GitHub: a pure Athena payload
  traceback gives it nothing to fetch). The old `pilot_monitoring_error` +
  `log_excerpt` path is still accepted so evidence produced by a deployment
  predating `traceback_available` continues to route correctly, and the old
  symbol is kept as an alias. Planner hint and tool description updated to
  match.
- **Deterministic code-analysis follow-up offer.** When a traceback reaches
  pilot3 code, `panda_log_analysis` evidence now carries
  `code_analysis_offer_md` naming the exact frame
  (`pilot/util/https.py:2301 (download_file)`), appended verbatim by
  `bamboo_executor` before the links block — like `links_md`, built from
  programmatic values so the LLM cannot garble the path or line number.
  `_PILOT_SOURCE_SIGNALS` extended to cover the offer's own wording so a user
  who accepts it by echoing it routes to `pilot_source_analysis`. Chaining is
  deliberately *not* automatic: the offer keeps the extra 5 GitHub fetches
  opt-in.
- **`_MAX_EXCERPT_CHARS` raised from 6000 to 8000**, with
  `_TRACEBACK_RESERVED_CHARS = 5000` allocated to the traceback *first* and
  surrounding context taking the remainder. `truncate_traceback()` keeps the
  head and tail with an elision marker rather than slicing from the front,
  because a plain `text[:n]` discards the terminal `ExceptionType: message`
  line — the single most diagnostic line in the traceback. Budgets are now
  passed *into* `extract_failure_context()` rather than applied as a
  post-hoc slice by the caller: `_fetch_logs_payload` previously sliced the
  stdout excerpt to `stdout_budget` after extraction, which could decapitate a
  traceback. `_STDOUT_CHAR_TAIL` is retained but documented as vestigial.
- **Directory-listing lines stripped from traceback context windows**
  (`_strip_directory_listing`). `ls -l` entries, `total N` headers and the
  pilot's `list_work_dir`/`print_executable` records are removed from the
  context either side of a traceback. They are never diagnostic and are
  actively harmful: a listing plus no real error is what led the LLM to
  diagnose job 7261310898 from the fact that `remote_open.stderr` was 28 kB.

### Fixed
- **Chained-traceback detection** (`_traceback_parse.find_traceback_blocks`).
  The first implementation closed a block at the terminal exception line, so a
  following `During handling of the above exception, another exception
  occurred:` marker was never seen and the chain was split into two blocks —
  reporting the *first* exception as the failure when the last one is what
  actually propagated. Now looks ahead past the exception line for a chain
  marker and absorbs the chained traceback into the same block.

### Tests
- New `packages/askpanda_atlas/tests/test_traceback_parse.py` (35 tests):
  record-prefix parsing, block boundaries, chained and truncated tracebacks,
  severity-based selection, pilot-vs-stdlib frame discrimination (including
  that the `pilot3/` scratch directory is not mistaken for the `pilot/`
  package), dotted custom exception types, colons in indented source lines,
  version detection and budget-aware truncation.
- Job 7261310898 regression fixture added to
  `packages/askpanda_atlas/tests/test_log_analysis.py`: asserts the excerpt
  contains `TimeoutError`/`download_transform`, contains no `tar cvfz` or
  `ls` listing lines, classifies as `transform_download_timeout`, exposes the
  correct `deepest_pilot_frame`, detects `pilot_version` (and falls back to
  `pilotid` on the 1305 path), and reports the exception rather than the
  misleading diag in the text summary. Plus exception-driven classification
  tests, including that a parsed exception overrides the `timeout=90 s`
  substring noise.
- Ref-resolution and version-skew tests added to
  `test_pilot_source_analysis.py`. Note these must patch `_fetch_raw`, not
  `fetch_pilot_module`, since `resolve_github_ref` probes through the former.
- **Docstrings of the two `test_log_analysis.py` files disambiguated.** The
  repo-root `tests/test_log_analysis.py` and
  `packages/askpanda_atlas/tests/test_log_analysis.py` carried identical
  docstrings, which made them indistinguishable when opened side by side. They
  are near-duplicates: 20 of the root file's 21 tests share names with the
  package file and 18 of those bodies are byte-identical. Each docstring now
  states what its file actually covers (core-shim reachability vs. the canonical
  implementation suite) and warns that shared behaviour must be changed in both
  places. The duplication itself is left in place — see Notes.
- Two existing fixtures updated because they encoded the old budgets, not
  because behaviour regressed: the root payload-tail test now derives its log
  size from the live `_MAX_EXCERPT_CHARS` (at 8000 its hardcoded 700-line log
  fit entirely in budget, so the "beginning absent" assertion was passing
  vacuously), and the 1354 test's blanket "no tail lines" assertion became a
  bounded one (the traceback must precede the trailing window, and at most
  `_TRACEBACK_TRAILING_LINES` lines may follow) because a bounded trailing
  window is now included by design — the pilot logs the resulting error code
  and state transition there. Mirrored into the ePIC copy.

### Notes
- `_traceback_parse.py` is duplicated in `askpanda_atlas` and `askpanda_epic`
  rather than shared. Plugin packages must stay independently installable and
  must not import each other or bamboo core at module scope, and
  `log_analysis_impl.py` is already duplicated the same way. The two copies
  differ only in the module docstring and the `askpanda_*` import prefix; any
  change to one must be mirrored to the other.
- **Unresolved duplication:** `tests/test_log_analysis.py` (root) is a 95%
  duplicate of `packages/askpanda_atlas/tests/test_log_analysis.py`. This work
  required fixing the same excerpt-budget fixture in both files independently,
  and the root file lacks the 1354 traceback test entirely, so the two have
  already drifted. Only one test in the root file is unique
  (`test_extract_log_excerpt_uses_tail_for_payload`), and its sole distinct
  value is proving the tool is reachable through the `bamboo.tools.log_analysis`
  shim. Reducing the root file to just that assertion would remove a standing
  maintenance trap, but deleting tests is a judgement call and was left for
  review rather than done here.
- `resolve_github_ref`, `function_at_line` and `verify_frame_lines` are
  deliberately *not* re-exported through the `pilot_source_analysis` shims:
  they are internal helpers, and exporting them would force matching stub
  definitions into `_fallback_pilot_source_analysis.py` in both packages for
  no benefit.

### Added
- **`atlas.job_stats`: memory-leak diagnostics and software-environment
  fields** (`packages/askpanda_atlas/askpanda_atlas/job_stats_schema.py`,
  `core/bamboo/tools/bamboo_answer.py`, `docs/question-cheatsheet.md`):
  Sasha's ingestion pipeline now parses the raw `jobmetrics` VARCHAR string
  upstream and exposes six previously-nested sub-fields as flat top-level
  fields: `lsetup_time` (s), `os_version` (keyword), `python_version`
  (keyword), and the memory-usage linear-fit parameters `leak_slope` (kB/s),
  `leak_intersect` (kB), `leak_chi2` (dimensionless goodness-of-fit) — renamed
  from PanDA's raw `leak`/`intersect`/`chi2` for clarity, since the fit model
  is `memory(t) ≈ leak_slope * t + leak_intersect`. Also added
  `task_container_name` (keyword), a previously-unregistered task-context
  field observed in the same sample record. `os_version` and `python_version`
  were added to `KEYWORD_GROUP_BY_FIELDS` for platform-breakdown queries.
  `_JOB_STATS_SIGNALS` gained fast-path routing tokens for all six jobmetrics
  fields plus natural-language phrases ("memory leak", "leak rate") for the
  leak-fit fields specifically — natural phrasing for `os_version` /
  `python_version` (e.g. "python version") was deliberately excluded as too
  ambiguous with generic non-job-stats questions, left to the LLM planner
  instead, same treatment as `atlasrelease`/`cmtconfig`/`homepackage`.
  `task_container_name` was not added as a `group_by` target pending a use
  case. Test coverage added in `test_job_stats.py`
  (`TestJobMetricsFields`, plus `parse_llm_params`/`group_by` round-trips)
  and `test_bamboo_answer_helpers.py` (`TestIsJobStatsQuestion`).
- **`atlas.job_stats`: `python_version` / `os_version` equality filters**
  (`packages/askpanda_atlas/askpanda_atlas/job_stats_schema.py`,
  `packages/askpanda_atlas/askpanda_atlas/job_stats_impl.py`,
  `core/bamboo/tools/bamboo_answer.py`):
  Previously the only filterable fields were `site`, `jobstatus`, and
  `jeditaskid` — `group_by=python_version` could show a breakdown, but
  there was no way to filter to a specific version while grouping by a
  different field (e.g. "which sites are still running python 3.7").
  Reported live: routing correctly deferred to `atlas.job_stats` but the
  tool itself had no way to express the query.
  Added `python_version` and `os_version` as new filter keys throughout
  the stack: the sub-LLM's JSON extraction schema (`_SYSTEM_TEMPLATE`),
  `parse_llm_params`, `fetch_job_stats` (new `wildcard` filter clauses,
  prefix-match not substring-match, so `"3.7"` matches stored `"3.7.9"`
  but not `"13.7.0"`), `_error_evidence`/`_cannot_answer_evidence` (new
  `python_version_filter`/`os_version_filter` evidence keys), and the
  tool's `inputSchema` (mandatory since `additionalProperties: false` is
  set — an argument override without a matching schema property would be
  rejected outright).
  Also added `bamboo_answer.py`-level deterministic extraction
  (`_extract_python_version_from_question`, recognizing "python 3.7"/
  "python3.7"/"python 2"; `_extract_os_version_from_question`, recognizing
  "EL7"/"EL9" and "os version 7"/"OS 9.7") as an argument-level safety net
  on top of the sub-LLM's own extraction — mirrors the existing `site`
  override pattern. Consolidated into a new `_build_job_stats_args` helper
  shared by both job-stats fast-path call sites (`_run_fast_path_intercepts`
  and `_build_deterministic_plan`) to avoid duplicating the extraction
  logic and to keep `_run_fast_path_intercepts` under the flake8
  max-complexity threshold (adding the two extractions inline pushed it
  from 15 to 17).
  Test coverage: `packages/askpanda_atlas/tests/test_job_stats.py` gained
  wildcard-filter assertions (prefix match verified directly against the
  mocked `Search.filter()` calls) and a combined filter+group_by test
  matching the original report's exact use case.
  Full suite: 2025 tests pass (1082 core + 760 askpanda_atlas + 89
  askpanda_epic + 94 askcgsim).

### Fixed
- **`_is_job_stats_question` didn't recognise bare OS version mentions
  either** (`core/bamboo/tools/bamboo_answer.py`): found while writing
  cheat-sheet examples for the new `os_version` filter — "Which sites are
  still on EL7?" and "...os version 9.7..." both routed to
  `panda_jobs_query` (no `os_version` column) via the same
  `_is_jobs_db_question`-wins-first mechanism as the Python version bug
  above. Fixed by having `_is_job_stats_question` reuse
  `_extract_python_version_from_question` /
  `_extract_os_version_from_question` directly instead of a separate
  detection-only regex, so routing detection and argument extraction share
  one source of truth and can't drift apart again. Added symmetric
  regression tests (`test_el7_shorthand_site_question`,
  `test_el9_shorthand_is_a_signal`, `test_os_version_word_phrase_is_a_signal`).
- **`_is_job_stats_question` didn't recognise bare Python version mentions**
  (`core/bamboo/tools/bamboo_answer.py`): "Which sites are still using
  python 3.7?" matched `_is_jobs_db_question` (via "sites") before
  `_is_job_stats_question` ever had a chance, since job_stats only
  recognised the literal `python_version` token or "memory leak"-style
  phrases — not a bare version number. This routed the question into the
  jobs/CRIC database-disambiguation prompt, or (with "running jobs" added)
  into `panda_jobs_query`, which has no `python_version` column at all.
  Fixed with a new regex signal (`_JOB_STATS_VERSION_RE`,
  `\bpython\s*[23](?:\.\d+)?\b`) checked alongside the existing literal-token
  signals in `_is_job_stats_question`. Known trade-off: an extremely
  contrived phrase like "python 2 days ago" would also match — accepted as
  low-risk given the domain.
- **Deterministic fast-path silently defaulted every unmatched question to
  RAG, never reaching the LLM planner** (`core/bamboo/tools/bamboo_answer.py`,
  `core/bamboo/tools/bamboo_executor.py`, `core/bamboo/tools/planner.py`):
  Reported live as "What was the average queuing time at CERN yesterday?"
  and "Which python versions are used on the sites?" both answering with a
  RAG "documentation doesn't cover this" response instead of routing to
  `atlas.job_stats`.
  Immediate trigger for the first report: `_JOB_STATS_SIGNALS` had
  `"queue time"`/`"queuetime"` but not the `-ing` form — fixed by adding
  `"queuing time"`, `"queueing time"`, `"queue wait time"`, `"queue wait"`.
  Immediate trigger for the second report: `python_version`/`os_version`/
  `lsetup_time`/the memory-leak fit fields were added to
  `job_stats_schema.py` and `bamboo_answer.py`'s fast-path signals in an
  earlier change, but the LLM planner's own `atlas.job_stats` routing
  description in `planner.py` — the third file in the "three-file edit
  problem" — was never updated to mention them, so even when a question
  reached the planner it had no guidance to route on. Fixed by extending
  the ATLAS routing prompt with explicit mentions and example phrasing
  (e.g. "which python versions are used").
  Root cause underneath both: `_build_deterministic_plan`'s final fallback
  (reached whenever no ID and no fast-path domain signal matched, across
  every plugin — job_stats, jobs_db, cric, pilot, promptlog, code_query)
  unconditionally built a RETRIEVE plan; it never returned `None` to defer
  to `bamboo_plan_tool` (the LLM planner), despite its own docstring and
  `_route()`'s comments claiming it did. This was previously intentional
  (see the old `test_bamboo_answer_rag.py` docstring) as a zero-LLM-cost
  optimization, but meant any phrasing gap in any signal set was a
  guaranteed wrong answer with no chance for the planner's broader semantic
  understanding to help — even where the planner's own prompt already
  documented the correct routing (e.g. "queue wait time" for
  `atlas.job_stats`). Fixed by returning `None` from that branch. The
  planner is already fully capable of choosing `route=RETRIEVE` with
  `doc_search`/`doc_bm25` itself for genuine documentation questions (see
  `_build_atlas_planner_prompt` / `_build_cgsim_planner_prompt`), so no RAG
  capability is lost — it's now reached via the planner instead of
  guaranteed deterministically.
  Two companion fixes found while implementing this: (1) `execute_plan` now
  auto-injects a `topic` argument (via `_inject_doc_topics`, a new helper)
  into any doc-search tool call that omits one, since the planner's routing
  prompt has no concept of ChromaDB-collection topic routing and previously
  would have silently used the default collection for every deferred
  question. (2) `PlannerTool.call` never threaded `plugin_id` through to
  `execute_plan` when executing its own plan (`execute=True`), so synthesis
  always used the "atlas" default regardless of the active plugin — a
  latent bug that matters more now that this path sees far more traffic.
  Test impact: rewrote `tests/test_bamboo_answer_rag.py` (its docstring
  documented the old always-RAG behavior as intentional) plus ripple
  effects in `test_bamboo_answer_helpers.py`, `test_context_memory.py`,
  `test_llm_error_handling.py`, and `test_topic_guard.py` — each had tests
  mocking `execute_plan` directly for no-ID questions that now defer to
  `bamboo_plan_tool` instead. Added new coverage in `test_bamboo_executor.py`
  (topic injection) and `test_planner.py` (plugin_id threading, routing
  prompt content). Full suite: 2014 tests pass across core + all plugins.
- **`panda_jobs_query` histogram returns 0 rows for multi-hour time windows**
  (`packages/askpanda_atlas/askpanda_atlas/jobs_query_schema.py`,
  `core/bamboo/tools/bamboo_executor.py`):
  Two compounding root causes. (1) The SQL generation prompt had no date/time
  anchor — the LLM had no temporal reference point. `build_sql_prompt` now
  injects `TODAY=` and `NOW=` (current UTC) on line 1 of the system prompt at
  call time, following the same pattern already used in `job_stats_schema.py`.
  (2) The prompt never explained that `statechangetime` data spans only the
  last ~1 hour per queue snapshot — longer `INTERVAL` filters always produced
  0 rows because older rows simply aren't retained. A `DATE RULE` block now
  states this constraint explicitly and instructs the LLM to drop the time
  filter for windows that exceed the snapshot. (3) The `_queue` rule was
  changed from "Always filter by `_queue`" to "filter ONLY when a specific
  site is mentioned", which is correct for global queries like histograms
  (the LLM was already omitting it correctly, but the contradictory rule
  invited future regressions). Three global-query examples added to the
  prompt (histogram, global count, cross-queue status breakdown). The
  `_SYSTEM_JOBS_QUERY` synthesis prompt was also extended: when `row_count`
  is 0 and the SQL contains a `statechangetime` interval filter spanning
  more than 1 hour, the synthesiser now explains the data-window limitation
  and suggests rephrasing without a time filter.

- **`opensearch_promptlog_query` / ratings query fails with "not valid JSON"**
  (`core/bamboo/tools/opensearch_promptlog_query.py`):
  The Bamboo fast-path router (`bamboo_answer.py`) passes the raw
  natural-language question as the `query` argument when routing to
  `opensearch_promptlog_query`. The tool forwarded this directly to
  `opensearch_query_tool.call()`, which does `json.loads(query)` and raised
  `'query' is not valid JSON: Expecting value: line 1 column 1 (char 0)`.
  Fixed by adding a `_generate_dsl(question)` async helper that calls the
  Bamboo LLM to translate a natural-language question into an OpenSearch DSL
  body (same LLM-generation pattern as `_call_llm_for_sql` in
  `jobs_query_impl.py`). `call()` now detects non-JSON `query` values and
  invokes `_generate_dsl` transparently before forwarding to
  `opensearch_query_tool`. A dedicated DSL generation system prompt
  (`_DSL_GENERATION_SYSTEM_PROMPT`) documents the full promptlog schema
  including the `rating` field with worked examples for ratings queries,
  FAQ frequency aggregations, and session replay.

- **`opensearch_promptlog_query` / "show me all ratings" returns unrated records**
  (`core/bamboo/tools/opensearch_promptlog_query.py`):
  Two follow-up defects surfaced after live testing. (1) `DEFAULT_SOURCE_FIELDS`
  did not include `rating`, so the `rating` field was absent from every returned
  hit — the synthesiser saw `NULL` on all records and concluded no ratings
  existed. `rating` added to `DEFAULT_SOURCE_FIELDS`. (2) The `_generate_dsl`
  LLM was not reliably including `{"range":{"rating":{"gte":1}}}` in the
  generated DSL for ratings questions, returning all 14 unrated turns instead of
  only rated ones. The advisory "For rating queries: filter with…" rule was
  replaced with a `RATING QUERIES — CRITICAL: MUST include…` imperative rule and
  a second ratings example (all-time, no date restriction) was added alongside
  the existing "ratings today" example, so the LLM sees the filter used
  consistently across both forms.

- **Bug 1 — Date hallucination in `atlas.job_stats` (Mistral-specific)**
  (`job_stats_schema.py`): Restructured the LLM prompt so the current date
  is unmissable.  The system prompt now opens with a terse
  `TODAY=<date>  NOW=<datetime>` anchor on line 1, followed by a `DATE RULE:`
  block that contains the pre-computed concrete timestamps for "today",
  "last hour", and "last 7 days" — no arithmetic left for the LLM to perform.
  The verbose prose `Current UTC date and time:` block is removed.  The
  user-message prefix is strengthened to `"TODAY IS <date>. NOW IS <datetime>
  UTC. USE THESE DATES ONLY.\n\n<question>"` (imperative anchor Mistral
  honours reliably).  The `one_hour_ago` timestamp is now pre-computed in
  `build_query_prompt` and injected into the template.
- **Bug 2 — "Which site has the highest X?" returned global max, not per-site**
  (`job_stats_impl.py`, `job_stats_schema.py`, `bamboo_executor.py`): Added
  optional `group_by` (keyword field to bucket by) and `top_n` (bucket count,
  1–20, default 5) parameters throughout the pipeline:
  - `parse_llm_params` now extracts and validates `group_by` against the new
    `KEYWORD_GROUP_BY_FIELDS` constant; invalid values are silently rejected.
  - `fetch_job_stats` gains a **terms path**: when `group_by` is set it
    executes a terms + sub-aggregation query ordered descending by the
    sub-metric, and returns a `buckets` list
    (`[{"key": ..., "value": ..., "doc_count": ...}]`) instead of a scalar
    `value`.  The existing scalar path is unaffected.
  - Evidence dict now always contains `group_by`, `top_n`, `buckets`, and
    `value` keys regardless of path, so the executor can inspect them reliably.
  - `_error_evidence` and `_cannot_answer_evidence` updated to include the new
    keys for structural consistency.
  - `PandaJobStatsTool.call()` passes `group_by` and `top_n` through to
    `fetch_job_stats`.
  - Cache key updated to include `group_by` and `top_n`.
  - `_SYSTEM_JOB_STATS` synthesis prompt updated to describe the buckets path
    and how to present ranked results.

- **Bug 3 — "Which site has the worst X?" returned the best performers**
  (`job_stats_impl.py`, `job_stats_schema.py`, `bamboo_executor.py`): The
  terms aggregation was hardcoded to `order={"sub_metric": "desc"}`, so
  questions asking for worst/lowest/least always returned the top performers.
  Added an explicit `order` parameter (`"asc"` / `"desc"`, default `"desc"`)
  throughout the group-by pipeline:
  - `job_stats_schema.py`: `"order"` key added to the LLM response spec with
    a clear rule — use `"asc"` for worst/lowest/least/bottom/poorest, `"desc"`
    for highest/best/most/top/greatest (or omit).  The
    `"Which site has the worst CPU efficiency today?"` example now correctly
    includes `"order": "asc"`.  A second `"asc"` example added
    (`"Which site has the lowest average stage-in time today?"`).
  - `job_stats_impl.py`: `parse_llm_params` extracts and validates `order`
    (invalid values fall back to `"desc"`); `fetch_job_stats` accepts `order`,
    passes it to `bucket(..., order={"sub_metric": order})`, includes it in
    the cache key and evidence dict (scalar path sets `order: None`);
    `_error_evidence` and `_cannot_answer_evidence` include `order` key;
    `PandaJobStatsTool.call()` threads `order` through.
  - `bamboo_executor.py`: `_SYSTEM_JOB_STATS` group-by rule now references
    the `order` field and tells the synthesiser to frame `"asc"` results as
    "worst/lowest".
- **Test cache-key bug** (`test_job_stats.py`): `TestFetchJobStats._run` was
  building cache keys without the `group_by`/`top_n`/`order` suffixes added
  in the previous session, causing `test_none_value_when_no_docs` to receive
  a stale hit from `test_avg_returns_value` and assert `42.5 is None`.

### Added
- **`KEYWORD_GROUP_BY_FIELDS`** (`job_stats_schema.py`): frozenset of 12
  keyword fields permitted as `group_by` targets:
  `computingsite`, `jobstatus`, `tier`, `task_campaign`, `task_type`,
  `task_workinggroup`, `prodsourcelabel`, `transfertype`, `inputfiletype`,
  `atlasrelease`, `country`, `atlas_resource_type`.
- **46 new tests** (`test_job_stats.py`): 8 date-anchor tests
  (`TestBuildQueryPromptDateAnchor`), 17 group-by parse tests
  (`TestParseLlmParamsGroupBy`, including 6 `order` cases), 15 group-by
  fetch tests (`TestGroupByFetchJobStats`, including 4 `order` cases),
  4 end-to-end tool group-by tests (`TestPandaJobStatsToolGroupBy`), plus
  4 new `TestSchemaConstants` assertions for `KEYWORD_GROUP_BY_FIELDS`.
  Total test count: **167** (was 121).
- **"Per-site and grouped breakdowns" section** (`docs/question-cheatsheet.md`):
  group-by example questions, permitted `group_by` field list, and note on
  invalid-field fallback behaviour.

### Added
- **`atlas.job_stats` tool** (`packages/askpanda_atlas`): new OpenSearch-backed
  tool replacing `atlas.job_timing`, targeting the richer
  `atlas_panda_job_stats-*` index.  Field coverage expands from 18 to 73
  confirmed fields across seven groups: timing (batch 1), I/O and data
  transfer, errors, task/campaign context, software environment, CPU/HS06
  accounting, memory, carbon footprint, and infrastructure traceability.
  New numeric aggregation targets include `avgrss`, `maxrss`, `avgpss`,
  `maxpss`, `avgvmem`, `maxvmem`, `avgswap`, `maxswap`, `minramcount`,
  `cpuconsumptiontime`, `cpu_eff`, `hs06`, `hs06sec`, `corecount`,
  `actualcorecount`, `inputfilebytes`, `outputfilebytes`, `totrbytes`,
  `totwbytes`, `raterbytes`, `ratewbytes`, `ninputdatafiles`,
  `noutputdatafiles`, `gco2global`, `gco2regional`, `piloterrorcode`,
  `exeerrorcode`, `ddmerrorcode`, `transexitcode`, `task_nattempts`, and more.
- **`_JOB_STATS_SIGNALS`** frozenset (`bamboo_answer.py`): replaces the
  former `_JOB_TIMING_SIGNALS`.  Expanded to cover memory (`"memory usage"`,
  `"rss memory"`, `"avgrss"`, `"maxrss"`, `"resident set"`, …), CPU/HS06
  (`"cpu efficiency"`, `"cpu_eff"`, `"hs06"`, `"hs06sec"`, …), I/O
  throughput (`"write throughput"`, `"ratewbytes"`, `"inputfilebytes"`, …),
  carbon footprint (`"carbon"`, `"co2"`, `"gco2"`, …), and error codes
  (`"pilot error"`, `"ddm error"`, `"exe error"`, …).  Ambiguous signals
  that overlap with `_JOBS_DB_SIGNALS` (e.g. `"failed jobs"`, `"error rate"`)
  are intentionally excluded and handled by the LLM planner.
- **`_SYSTEM_JOB_STATS`** synthesis prompt (`bamboo_executor.py`): replaces
  `_SYSTEM_JOB_STATS`.  Describes all new field groups with their units
  (kB for memory, bytes/s for throughput, HS06·s for HS06-normalised CPU)
  and includes guidance on null values for `hs06sec` and carbon fields on
  non-terminal jobs.
- **`atlas_panda_job_stats-*`** added to `_DEFAULT_ALLOWED_PATTERNS` in
  `opensearch_query.py`; `atlas_panda_job_timing-*` removed.
- **`test_job_stats.py`** (110 tests): full test coverage for the new tool,
  including `TestNewNumericFields` (32 field-presence assertions) and
  `TestParseLlmParamsNewFields` (12 round-trip tests for batch-2 fields).

### Changed
- **`atlas.job_timing` entry point removed** from `pyproject.toml`;
  replaced by `atlas.job_stats = askpanda_atlas.job_stats:panda_job_stats_tool`.
  The old `job_timing_*.py` source files are retained in the repository for
  reference but are no longer registered.
- **Planner routing hint updated** (`planner.py`): `atlas.job_timing` →
  `atlas.job_stats`; description expanded to cover memory, CPU/HS06, I/O,
  error codes, and carbon queries.
- **Fast-path intercepts updated** (`bamboo_answer.py`): both intercept
  sites now call `_is_job_stats_question()` and route to `atlas.job_stats`.
- **`docs/question-cheatsheet.md`**: `atlas.job_timing` section replaced by
  `atlas.job_stats` section with new question groups for memory, CPU/HS06,
  I/O, errors, task/campaign context, and carbon footprint.


  keys in `_chroma_routing.py` that map to dedicated logical collection names
  (`bamboo_mcp_docs` and `bamboo_services_docs` respectively).  Deployments
  that split the Bamboo documentation into two separate ChromaDB collections
  — one for the `bamboo-mcp` repository and one for `bamboo-mcp-services` —
  now get correctly isolated retrieval: questions about installing or
  configuring the MCP core never return Services agent documentation, and
  vice versa.  The legacy `"bamboo"` → `"bamboo_docs"` entry is retained for
  backward compatibility with single-collection deployments.
- **`_BAMBOO_SERVICES_SIGNALS`** frozenset (`bamboo_answer.py`): keyword
  phrases that unambiguously refer to the `bamboo-mcp-services` component
  (`"bamboo mcp services"`, `"bamboo-mcp-services"`, `"bamboo services"`,
  `"supervisor agent"`, `"ingestion agent"`, `"cric agent"`,
  `"document monitor"`, etc.).  Checked *before* `_BAMBOO_SIGNALS` so that
  the more specific match wins when both would match.
- **Expanded `_BAMBOO_SIGNALS`** (`bamboo_answer.py`): added
  `"bamboo install"`, `"install bamboo"`, `"bamboo plugin"`,
  `"bamboo interface"`, `"bamboo ui"`, `"bamboo tui"`, `"bamboo cli"`,
  `"bamboo streamlit"` — previously these fell through to `topic="atlas"`,
  causing the wrong collection to be queried.
- **`bamboo_env_example.sh` updated**: RAG collection map example now shows
  both the recommended two-collection layout (`bamboo_mcp` / `bamboo_services`)
  and the legacy single-collection layout as an alternative comment block.

### Added
- **Multi-collection RAG support**: `doc_rag.py` and `doc_bm25.py` now accept
  an optional `topic` parameter (e.g. `"panda"`, `"atlas"`, `"rucio"`,
  `"root"`, `"bamboo"`, `"epic"`, `"cgsim"`) that selects which ChromaDB
  collection to query, enabling the five separate collections produced by
  `bamboo-mcp-services` to be queried correctly per question domain.
- **`BAMBOO_CHROMA_COLLECTION_MAP`** env var: JSON object mapping topic keys
  to logical collection names (e.g.
  `'{"panda":"panda_docs","atlas":"atlas_docs","rucio":"rucio_docs"}'`).
  Adding a new collection requires only updating this string — no code
  changes needed.  Falls back to the existing `BAMBOO_CHROMA_COLLECTION`
  scalar, then to built-in per-topic defaults (`panda_docs`, `atlas_docs`,
  `bamboo_docs`, `rucio_docs`, `root_docs`, `epic_docs`, `cgsim_docs`).
- **`resolve_collection_for_topic()`** (`_chroma_routing.py`): new helper that
  maps a topic string → logical collection name (via
  `BAMBOO_CHROMA_COLLECTION_MAP`) → physical blue/green slot (via the
  existing `resolve_collection()`).  All RAG tools now route through this
  single function.
- **`_topic_for_question()`** (`bamboo_answer.py`): lightweight keyword
  classifier that infers the correct topic from the user question and active
  plugin (Rucio signals → `"rucio"`, ROOT signals → `"root"`, Bamboo meta
  → `"bamboo"`, atlas plugin → `"atlas"`, etc.).  Result is injected into
  both `panda_doc_search` and `panda_doc_bm25` plan tool call arguments by
  `_build_deterministic_plan()`.

### Changed
- **Subclass simplification**: `AtlasDocSearchTool`, `AtlasDocBM25Tool`,
  `EpicDocSearchTool`, `EpicDocBM25Tool`, `CgsimDocSearchTool`,
  `CgsimDocBM25Tool` — the full copy-paste `_ensure_collection()` /
  `_ensure_index()` overrides have been removed from all six package
  subclasses.  Each subclass now only overrides `get_definition()` and sets a
  `_default_topic` class attribute (`"atlas"`, `"epic"`, `"cgsim"`).  All
  collection resolution logic lives exclusively in the base class.
- **Reranking workaround removed**: the `_is_bamboo_internal()` source-priority
  reranking in `doc_rag.py` and `doc_bm25.py` (which deprioritised
  `PalNilsson/*` chunks) has been removed now that Bamboo-internal
  documentation lives in its own dedicated `bamboo_docs` collection.
- **`bamboo_env_example.sh`**: RAG section updated to document
  `BAMBOO_CHROMA_COLLECTION_MAP`; the old per-plugin collection comment block
  replaced with JSON map format.

### Tests
- `tests/test_chroma_routing.py`: `TestResolveCollectionForTopic` (9 tests)
  covering map lookup, blue/green sidecar traversal, scalar fallback,
  built-in defaults, unknown topics, case-insensitivity, corrupt map,
  and adding new collections via env only.
- `tests/test_doc_rag.py`: 4 new tests for `topic` argument passthrough,
  `_default_topic` class attribute, and `get_definition` schema shape.
- `tests/test_doc_bm25.py`: 4 new tests for `topic` argument passthrough,
  cache invalidation on topic change, and `get_definition` schema shape.
- `tests/test_bamboo_answer_rag.py`: 13 new tests for `_topic_for_question()`
  and `_build_deterministic_plan()` topic injection.


  that orchestrates MCP tool calls in a Reason → Act → Observe → Evaluate loop.
  Bypasses the single-pass `bamboo_answer`/`bamboo_executor` pipeline and is
  intended for complex, multi-hop queries.  Key types: `AgentMemory`,
  `AgentStep`, `AgentResult`, `BambooAgent`.  Uses `reasoning` LLM profile for
  tool selection and synthesis; `fast` profile for the per-step sufficiency
  evaluator.  All LLM calls are routed through the `bamboo_llm_answer` MCP tool,
  so no additional LLM initialisation is required.  Maximum steps (default 6),
  confidence threshold (default 0.80), and synthesis token budget (default 2048)
  are all configurable via `BAMBOO_AGENT_MAX_STEPS`, `BAMBOO_AGENT_CONFIDENCE`,
  and `BAMBOO_AGENT_MAX_TOKENS` environment variables.
- **Agent CLI** (`scripts/bamboo_agent.py`): standalone script wrapping
  `BambooAgent`.  Supports single-shot (`--question`), stdin-pipe, and
  interactive REPL (`--interactive`) modes.  Outputs formatted text (with
  optional `--verbose` trace) or machine-readable JSON (`--output-json`).
  Compatible with both HTTP and STDIO MCP transports.  Bearer token auth via
  `--token` or `BAMBOO_MCP_TOKEN`.
- **Agent tests** (`tests/test_agent.py`): full test coverage for
  `AgentMemory`, `_ToolSelection`, `_EvalResult`, `_extract_json_block`,
  `_truncate_observation`, `_observation_from_result`, and `BambooAgent`
  (single-step success, two-step completion, early `should_synthesise` flag,
  max-steps truncation, tool call failure, reasoning parse error, eval parse
  error, field type assertions, zero-tools edge case).
- **Agent prompt log stub**: `BambooAgent._synthesise` contains a fully
  commented-out `log_prompt` call (`# AGENT_LOG`) targeting the future
  `bamboomcp-agentlog-YYYY.MM.DD` index.  Uncomment once the index template
  is provisioned in OpenSearch.

### Fixed
- **Relative `from_dt`/`to_dt` expressions crash the Harvester API**:
  the LLM planner occasionally emits OpenSearch-style relative timestamps
  (`"now-6h"`, `"now/d"`) instead of absolute ISO-8601 strings.  The BigPanDA
  Harvester HTTP API does not understand these and returns an error, producing a
  zeroed-out evidence dict and a misleading "API unavailable" response.  Fixed by
  adding `_resolve_dt()` to `harvester_worker_impl.py`, which intercepts any
  non-ISO argument and resolves it to an absolute UTC timestamp before the HTTP
  call.  The planner routing prompt for `panda_harvester_workers` and
  `atlas.harvester_timeseries` also now explicitly instructs the LLM to use
  absolute ISO-8601 strings, not relative expressions.
- **Pilot failure-rate routing misclassification**: questions such as "which sites
  had pilot failures above 20% today?" were incorrectly routed to
  `panda_harvester_workers` (the BigPanDA HTTP snapshot tool) instead of
  `atlas.harvester_timeseries` (the OpenSearch time-series tool).  The planner
  routing guidance now has a dedicated rule for failure-rate and
  failure-percentage questions that explicitly selects `atlas.harvester_timeseries`
  with `status='failed'`, while the live-count rule is tightened to snapshot
  queries only ("how many pilots are running right now").
- **`atlas.harvester_timeseries` tool description**: the description previously
  read "used to render ASCII time-series charts in the TUI" — making the planner
  LLM treat it as an internal charting helper rather than a query tool.  The
  description now explicitly lists failure-rate and cross-site trend questions as
  primary use cases, with concrete examples.
- **`bamboo_executor._pick_synthesis_prompt`**: `atlas.harvester_timeseries` now
  selects `_SYSTEM_HARVESTER_TIMESERIES` (a new specialist prompt) instead of
  falling through to `_SYSTEM_GENERIC`.  The new prompt instructs the LLM to
  report absolute failed-pilot counts and trends, and to explain why it cannot
  compute a failure *percentage* without the total pilot count (cross-referencing
  `failed` vs `total` requires two queries).

### Added
- **`panda_job_timing` tool** (`packages/askpanda_atlas`): new OpenSearch-backed
  MCP tool that answers natural-language questions about PanDA job timing against
  the `atlas_panda_job_timing-*` index.  Uses a single LLM call to extract
  structured aggregation parameters (metric, field, filters, time range) from the
  user's question, then executes a single-value OpenSearch metric aggregation
  (`avg`, `sum`, `min`, `max`, `value_count`) and returns a compact evidence dict
  for Bamboo's central synthesiser.
- **`job_timing_schema.py`**: schema registry for the confirmed batch-1 fields
  (10 core identifier/status fields + 10 timing fields including all six parsed
  `pilottiming_*` sub-fields), field validation helpers, and the LLM prompt
  template for query-parameter extraction.
- **`job_timing_impl.py`**: full tool implementation with `fetch_job_timing()`
  (synchronous OpenSearch query, cached 120 s), `parse_llm_params()` (validates
  LLM JSON output against schema), `_default_window()` (24-hour look-back), and
  structured error/cannot-answer evidence constructors.
- **`job_timing.py`**: thin entry-point wrapper with `ImportError` fallback
  (mirrors `harvester_timeseries.py`).
- **`tests/test_job_timing.py`**: 34 tests covering schema constants, prompt
  builder, `parse_llm_params`, `_default_window`, error constructors,
  `fetch_job_timing` with mocked OpenSearch, and `PandaJobTimingTool.call()`
  end-to-end.
- `atlas.job_timing` entry point registered in `pyproject.toml`.
- `atlas_panda_job_timing-*` added to `_DEFAULT_ALLOWED_PATTERNS` in
  `core/bamboo/tools/opensearch_query.py` so the generic `opensearch_query`
  tool can also reach this index without config changes.

### Added

- **Blue/green ChromaDB slot routing — live re-resolution without server restart.**
  The `bamboo-mcp-services` document-monitor agent now stores vectors in two
  physical ChromaDB collections per logical name (`atlas_docs__a` /
  `atlas_docs__b`) and swaps between them atomically.  Bamboo MCP now resolves
  the logical collection name (e.g. `atlas_docs`) to the currently live
  physical slot on **every RAG query** by reading the routing sidecar
  (`<BAMBOO_CHROMA_PATH>/collection_routing.json`).  When the document-monitor
  agent completes an update cycle the next query automatically picks up the new
  slot with no server restart required.

  If the sidecar is absent or has no entry for the configured logical name
  Bamboo falls back to using the logical name directly, so deployments that
  have not yet upgraded to the blue/green agent are unaffected.

  **New module** `core/bamboo/tools/_chroma_routing.py` — standalone
  `resolve_collection(chroma_path, logical_name)` helper.  Does not import
  from `bamboo-mcp-services`; Bamboo MCP remains fully independent.

  **Changed:** `core/bamboo/tools/doc_rag.py` (`PandaDocSearchTool`) and
  all three plugin overrides (`askpanda_atlas`, `askpanda_epic`, `askcgsim`)
  — `_ensure_collection` now re-reads the sidecar on every call and
  invalidates the cached collection handle when the physical name changes.
  A new `_resolved_physical` attribute tracks the currently open physical
  slot; `_reset()` clears it alongside `_client` and `_collection`.

  **Scripts** `probe_rag.py` and `inspect_chroma.py` both resolve the
  logical name via the sidecar and print the resolved physical slot name in
  their output headers.

  **New tests** `tests/test_chroma_routing.py` — 11 tests covering
  `resolve_collection` (sidecar present, absent, corrupt, missing entry,
  mid-run update) and `PandaDocSearchTool` live re-resolution (correct slot
  opened, cache invalidated on swap, no unnecessary reopens, pre-blue/green
  fallback, `_reset` clears resolved name).

  **Docs** `docs/rag.md` — new *Blue/green slot routing* section explaining
  the sidecar format, live re-resolution, fallback behaviour, and how to
  diagnose the active slot with `inspect_chroma.py` and `probe_rag.py`.

 The spinner
  is rendered after the full chat history during the pending-question pass, so
  it appears at the bottom of the page just above the input box rather than
  at the top where it was invisible in long conversations.
- **RAG synthesis: prohibit general-knowledge fallback when excerpts are insufficient.**
  ``_SYSTEM_RAG`` now instructs the LLM to tell the user the documentation
  did not contain enough information rather than supplementing with general
  knowledge.  The previous wording ("supplement with your general knowledge
  but clearly distinguish...") gave the LLM a loophole to produce fully
  hallucinated answers dressed as general knowledge when the retrieved
  excerpts were topically adjacent but not actually relevant.
- **Streamlit: sidebar shows "Connected" immediately on startup.** Added
  ``st.rerun()`` after a successful first ``_connect()`` call so the
  sidebar status updates from "Not connected" to "Connected" as soon as
  the server handshake completes, without waiting for the first question.
- **Streamlit: remove Experiment/plugin selector from sidebar.** The plugin
  is now fixed to the ``ASKPANDA_PLUGIN`` environment variable (default
  ``atlas``).  Switching experiments requires restarting the server with a
  different env var rather than hot-switching in the UI, which avoids
  confusing mid-session state resets.
- **Streamlit: rating poll retries up to 3×0.5 s.** Replaces the
  single-retry flag with a tight loop that polls ``bamboo_promptlog_status``
  up to three times at 0.5 s intervals, stopping as soon as ``last_doc_id``
  is set.  Fixes intermittent missing rating buttons when OpenSearch flushes
  slowly.
- **Streamlit: rating widget retry on first question after restart.**
  If the deferred prompt-log poll fires before OpenSearch has flushed the
  background write (``last_doc_id`` still ``None``), a ``retry_promptlog``
  flag triggers one additional poll after a 0.5 s sleep on the following
  render cycle.  Fixes the missing rating buttons on the first response
  after a server restart.
- **Streamlit: one-shot rating widget.** After a user submits a star rating,
  the five rating buttons are replaced by a static confirmation caption for
  the remainder of the session, preventing duplicate votes on the same
  response.
- **Streamlit: retry prompt-log poll for first-question rating.** If the
  deferred `poll_promptlog` pass completes before the OpenSearch background
  write finishes (`last_doc_id` still `None`), a `retry_promptlog` flag
  is set and a second poll runs 0.5 s later on the next render cycle.
  This fixes the missing rating widget on the first question after a server
  restart.
- **`docs/remote-testing.md`:** Step-by-step guide for running the Bamboo MCP
  server and Streamlit UI on lxplus and accessing them from home via an SSH
  port-forwarding tunnel over the CERN VPN.  Covers SSH key setup, tunnel
  command, server and Streamlit startup, health-check verification, and a
  troubleshooting table for common failure modes.

### Fixed

- **`panda_job_status`: MCPCaller server name mismatch caused "server not
  connected" errors.** `job_status.py` used `_SERVER = "bigpanda-downloader"`
  but `panda_mcp_session.py` registers the session under
  `PANDA_MCP_SERVER_NAME = "panda"`.  The lookup always returned `None`,
  so every job-status query failed regardless of whether the PanDA MCP
  connection was healthy.  Fixed `_SERVER` to `"panda"` and updated the
  stale server name in `_mcp_caller.py` docstring,
  `docs/tools/panda_job_status.md`, and `docs/mcp_sequence_diagram.mmd`.

- **Streamlit: Mermaid diagram height and rendering.** Height estimation
  now uses non-empty line count (`line_count * 20 + 80`, capped at 800 px)
  instead of arrow count, which overcounted for state diagrams and produced
  oversized iframes that pushed nodes below the visible area.  Mermaid CDN
  bumped from `@10` to `@11` for improved state diagram rendering.
- **Streamlit: single-iframe mode no longer duplicates text.** The chat
  history render loop now skips ``st.markdown()`` for the last assistant
  message when ``BAMBOO_DIAGRAM_MODE=single-iframe`` and diagrams are
  present — ``_render_mermaid_single_iframe()`` renders both text and
  diagrams together.
- **Streamlit: classic mode sanitises edge labels with special chars.**
  A ``re.sub`` pass quotes unquoted Mermaid edge labels containing ``(``,
  ``)``, or ``<`` before rendering, preventing Mermaid v11 from tokenising
  them as separate nodes (e.g. ``(loop counter < max)``) .
- **Streamlit: dual Mermaid renderer via ``BAMBOO_DIAGRAM_MODE``.** The
  existing per-diagram ``components.html`` renderer is refactored into
  ``_render_mermaid_classic()`` and now uses ``mermaid.render()``
  (Promise API) instead of ``startOnLoad`` for precise post-render SVG
  attribute cleanup.  A new experimental ``_render_mermaid_single_iframe()``
  renderer (``BAMBOO_DIAGRAM_MODE=single-iframe``) places text and all
  diagrams into a single iframe with CSS layout: portrait diagrams float
  right at 38% width so prose wraps alongside them; landscape diagrams
  span full width below the text.  ``_render_mermaid_blocks()`` dispatches
  to the active mode; ``last_clean_answer`` is stored in session state so
  the single-iframe renderer can access the stripped markdown text.
- **Streamlit: Mermaid syntax errors show plain-text fallback.** Added a
  ``mermaid.parseError`` handler that hides the Mermaid error graphic and
  displays the raw diagram definition in a styled ``<pre>`` block instead,
  making it easy to see what the LLM generated without a jarring full-page
  error image.
- **Streamlit: Mermaid diagrams scale and auto-size correctly.** A
  ``MutationObserver`` strips Mermaid's inline ``width``/``height``
  attributes from the SVG after render (they override CSS and cause
  oversized output), then posts the actual rendered height to Streamlit
  so the iframe auto-resizes to fit.  A 600 ms fallback ``setTimeout``
  handles edge cases where the observer fires before layout settles.
  ``scrolling=False`` — no scroll bar needed once the iframe matches the
  diagram height.
- **Streamlit: Mermaid diagrams scale to fit iframe width.** Switched to
  ``useMaxWidth: true`` with ``width: 100% !important`` on the SVG so
  diagrams shrink to fit rather than rendering at natural size and pushing
  content off-screen.  Reduced node/rank spacing (40/50 px) and tightened
  the height estimate to 14 px per line (cap 600 px) so diagrams are
  compact.  Mermaid CDN bumped to v11.
- **Streamlit: `/rates` date-filtered queries no longer fail.** The
  ``/rates today``, ``/rates week``, and ``/rates month`` slash commands
  now pass a fully pre-built OpenSearch ``bool/must`` query with both the
  ``exists`` on ``rating`` and the ``range`` on ``@timestamp`` baked in,
  leaving nothing for the LLM to construct or modify.  Previously the LLM
  generated a malformed ``range`` query combining multiple fields.
- **Streamlit: `st.components.v1.html` deprecation noted.** `st.iframe`
  (the advertised replacement) accepts a URL `src`, not raw HTML, so it
  cannot replace `components.v1.html` for inline Mermaid rendering.
  A ``.. note::`` has been added to the docstring documenting this
  constraint.  The call site is unchanged pending a Streamlit fix or
  alternative approach.
- **Prompt log: suppress 403 index-template spam.** `_ensure_index_template`
  now detects `AuthorizationException(403)` responses, sets the
  ``_template_applied`` flag to prevent retries, and logs at ``INFO`` rather
  than ``WARNING``.  The OpenSearch ``pilot-monitor-agent`` user lacks
  ``indices:admin/index_template/put`` permission; retrying on every server
  start was pointless and noisy.  Document writes are unaffected.


- **PanDA MCP OIDC token file support.** `panda_mcp_session.py` now reads
  the `id_token` field from the OIDC token cache file written by
  `get-panda-token` (from the `panda-mcp-client` package).  Token resolution
  order: (1) `PANDA_MCP_TOKEN` env var, (2) `id_token` from the file at
  `PANDA_MCP_TOKEN_FILE` (default `~/.panda_id_token`), (3) no token for
  public endpoints.  A new `_read_token_file()` helper handles JSON parsing
  and all failure modes (missing file, malformed JSON, missing field) with
  WARNING-level log messages rather than crashes.  Token renewal will be
  handled by a forthcoming Bamboo MCP agent service.
- **`panda_server_health`: error diagnosis.** When `system_is_alive` returns
  an error string, a new `_diagnose_error()` helper maps known patterns to
  human-readable explanations included in the evidence (`error_explanation`
  field) and appended to the summary text.  Covers: server-side SSL failure
  on port 25443, Bamboo-side CA bundle issues, connection refused/timeout,
  and auth/token errors.  No second LLM call required — diagnosis is
  deterministic.
- **TLS docs — use system CA bundle via `SSL_CERT_FILE`**: The correct
  approach on lxplus is `export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`.
  Both `httpx` and `requests` honour this standard env var automatically.
  Modifying the certifi bundle is fragile (DER files or HTML redirect pages
  silently corrupt it; changes lost on `pip upgrade certifi`) and is no
  longer recommended.  Updated `CLAUDE.md`, `bamboo_env_example.sh`, and
  `docs/question-cheatsheet.md`.

### Fixed

- **`panda_server_health`: correct upstream tool name.** The tool name used
  to call the PanDA MCP server was `is_alive`; the actual tool name exposed
  by the server is `system_is_alive`.  Updated `_TOOL` constant and all
  docstring references accordingly.
- **`panda_mcp_session`: surface inner exception from `ExceptionGroup`.**
  The session failure handler previously logged only the top-level
  `ExceptionGroup` message, hiding the root cause.  It now iterates
  `exc.exceptions` and logs each inner exception with a full traceback via
  `exc_info=`.
- **PanDA MCP TLS on lxplus**: The certifi bundle in the virtualenv does not
  include the CERN Grid CA or CERN Root CA 2 even on lxplus, where the
  system CA store does.  `PANDA_MCP_BASE_URL` must omit the trailing slash
  (use `…/mcp` not `…/mcp/`) to avoid a 307 redirect that the MCP client
  does not follow.  Updated `bamboo_env_example.sh` and docs accordingly.

- **Mermaid diagram rendering in Streamlit.** The LLM can now return a
  ` ```mermaid ``` ` block alongside prose when a question calls for a diagram
  (algorithms, state machines, protocols, flows).  The Streamlit UI extracts
  the block before storing the response in chat history (keeping history
  clean), then renders it inline.  Multiple diagrams per response are
  supported.  The TUI strips diagram blocks before storing history.
  The `_MERMAID_GUIDANCE` constant is appended to `_SYSTEM_RAG`,
  `_SYSTEM_GENERIC`, `_SYSTEM_RAG_CGSIM`, `_SYSTEM_GENERIC_CGSIM`, and
  `_SYSTEM_CODE_QUERY` synthesis prompts.
  Added `streamlit-mermaid>=0.2.0` to `requirements-ui.txt`.

- **Mermaid rendering — CDN-based renderer replacing streamlit-mermaid.**
  `streamlit-mermaid` uses `svgPanZoom` which scales the entire diagram SVG
  down to fit the Streamlit column width, making node text unreadably small.
  Replaced with `st.components.v1.html` embedding Mermaid 10 from CDN with
  `useMaxWidth: false` and `htmlLabels: true` — no SVG scaling, nodes render
  at their natural size, wide diagrams scroll horizontally.
  `_wrap_mermaid_labels()` post-processor wraps long node labels using native
  Mermaid `\n` line breaks, splits on underscores, and hard-cuts tokens that
  exceed the limit, ensuring all text is readable regardless of LLM output.
  `_MERMAID_GUIDANCE` updated with strict node label rules (≤20 chars/line,
  `\n` not `<br/>`, no long identifiers in nodes), anti-hallucination rules
  for static analysis findings, and explicit Mermaid syntax rules per diagram
  type.  `streamlit-mermaid` is retained as an optional dependency.

- **Superuser / developer mode.** A password-protected developer tier is now
  available in both the Streamlit and TUI interfaces.

  - Set `BAMBOO_SUPERUSER_PASSWORD` in `bamboo_env.sh` to enable the feature.
  - **Streamlit:** a "Developer access" section appears in the sidebar.  After
    entering the correct password, the session is flagged as a superuser and
    developer tools become active.  A 🔓/🔒 toggle controls the lock state.
  - **TUI:** `/superuser <password>` unlocks the session; listed in `/help`.
  - **Pre-dispatch guard:** questions that would route to superuser-only tools
    are blocked before the server call when the session is unauthenticated.
    Guard logic lives in `interfaces/shared/superuser_guard.py` and is shared
    by both interfaces.  Configurable via `BAMBOO_SUPERUSER_PATTERNS` and
    `BAMBOO_SUPERUSER_TOOLS` env vars.  Superuser tools are always registered
    on the MCP server; the guard is a UI-layer gate only.
  - Added `BAMBOO_SUPERUSER_PASSWORD`, `BAMBOO_SUPERUSER_TOOLS`, and
    `BAMBOO_SUPERUSER_PATTERNS` documentation to `bamboo_env_example.sh`.

- **`code_query` tool (superuser / developer).** A new built-in MCP tool that
  fetches an arbitrary source file from any configured GitHub repository and
  returns it for LLM analysis.  Replaces the earlier `pilot_code_query` design.

  - Input: `file_path` (e.g. `pilot.py`, `pilot/util/processes.py`), `question`,
    and optional `function_name` for targeted function extraction using AST.
  - Repository and branch configurable via `BAMBOO_CODE_QUERY_REPO` and
    `BAMBOO_CODE_QUERY_BRANCH` (defaults: `PanDAWMS/pilot3`, `master`).
  - Source limit raised to **150 000 characters** (from 12 000) with
    line-boundary truncation — rare for typical files.
  - Evidence pipeline fix: source is extracted from the evidence dict *before*
    `_compact_json` (12K limit) and appended as a plain fenced block, ensuring
    the `truncated` flag and all metadata always reach the LLM intact regardless
    of file size.
  - Dedicated synthesis prompt `_SYSTEM_CODE_QUERY` with rules against
    fabricating source lines, claiming identifiers are unused without tracing
    them, and inventing bugs not demonstrable from the source.
  - Tagged `superuser`; evidence expanders hidden from non-superuser sessions.
  - Registered as `"code_query"` in `core.py` `TOOLS` dict.
  - Full test coverage in `tests/test_code_query.py` (25 tests).
  - `docs/tools/code_query.md` — new tool reference documentation.
  - `docs/tools/pilot_code_query.md` — superseded; delete from repo.

- **`code_query` — fast-path routing.** `_build_deterministic_plan` now
  includes a rule (priority 6, after ID-driven rules, before RAG fallback) that
  routes to `code_query` when the question contains a `*.py` filename/path or an
  inspection verb combined with a repository keyword.  The first `*.py` token is
  extracted from the question and passed as `file_path`.

- **`code_query` — follow-up routing.** After a `code_query` response,
  content-free affirmatives (`"yes please"`, `"ok"`, `"continue"`) and
  code-review continuation phrases (`"verify the full file"`, `"show the
  remaining code"`, `"get the complete source"`) are automatically re-routed to
  `code_query` with the file path recovered from history.  Bypasses the topic
  guard.  Detection via `_last_tool_was_code_query` (scans prior assistant
  message for code-specific vocabulary) and `_is_code_review_continuation`
  (regex matching continuation words + repository keywords).

- **Streamlit — plugin display name fix.** Switching the plugin dropdown while
  connected now immediately re-fetches `ui_manifest` for the new plugin, so the
  display name updates correctly without requiring a reconnect.

- **Streamlit — generic inline plot.** After any tool response with flat tabular
  evidence (`columns` + `rows`), an interactive Plotly chart is rendered directly
  in the main chat area.  Chart type is chosen automatically based on column
  types.  Requires `plotly>=5.0` (added to `requirements-ui.txt`).

### Added

- **OpenSearch prompt-log UI notifications.** When ``BAMBOO_OPENSEARCH_PROMPTLOG``
  is set, write confirmations and errors from the OpenSearch indexing background
  task are now surfaced directly inside the running interface.

  Architecture: ``log_prompt()`` is called from ``call_llm()`` in
  ``bamboo_executor.py`` as ``asyncio.create_task`` (fire-and-forget).
  ``_write_document`` appends each outcome to a process-local ring buffer
  (``deque(maxlen=20)``) in ``prompt_log.py``.  A new built-in MCP tool
  ``bamboo_promptlog_status`` exposes the buffer via a destructive drain
  (events delivered exactly once per poll).  Both interfaces poll the tool
  after each response and display results in the UI.

  - ``core/bamboo/llm/prompt_log.py``: added ``_event_log`` ring buffer,
    ``drain_events()``, ``NotifyFn``, ``register_notify_callback()``,
    ``clear_notify_callback()``, and ``_notify()``.  ``_write_document``
    appends ``{"turn", "severity", "message"}`` events on success, warning,
    error, and ``ImportError`` (opensearch-py not installed).  The
    ``ImportError`` case now produces a visible ``"warning"`` event with an
    install instruction rather than silently logging at DEBUG.
  - ``core/bamboo/tools/bamboo_executor.py``: ``call_llm()`` gains a
    ``tools_used`` parameter and fires ``log_prompt`` via
    ``asyncio.create_task`` after each LLM response.  New
    ``BambooPromptLogStatusTool`` / ``bamboo_promptlog_status_tool`` calls
    ``drain_events()`` and returns events as JSON.
  - ``core/bamboo/core.py``: ``bamboo_promptlog_status`` registered in
    ``TOOLS``.
  - **TUI** (``interfaces/textual/chat.py``): ``_fetch_promptlog_events()``
    polls ``bamboo_promptlog_status`` after every response with a retry loop
    (6 × 0.5 s) to allow the background OpenSearch write to complete.
    ``"error"`` events render as error panels; ``"info"``/``"warning"`` as
    system panels.
  - **Streamlit** (``interfaces/streamlit/chat.py``):
    ``_poll_promptlog_events()`` polls on the render cycle *after* the
    response rerun (deferred via ``poll_promptlog`` session-state flag) so
    the background write has time to complete.  Events are pushed to
    ``promptlog_notices`` and rendered by ``_render_promptlog_notices()`` as
    ``st.error`` / ``st.warning`` / ``st.toast``.

  Requires ``pip install opensearch-py`` and write permission for the
  configured user on ``bamboomcp-promptlog-*`` in OpenSearch.

### Fixed

- **TUI — alternate-screen rendering on SSH (lxplus).** Textual's `--no-inline`
  (alternate screen) renderer uses absolute cursor positioning without erasing
  lines, relying on the terminal clearing the alternate screen buffer on entry.
  SSH pseudo-TTYs (e.g. lxplus accessed via Claude Code) do not reliably do
  this, causing every repaint to accumulate as ghost frames — visible as stacked
  bordered panels during long requests or a screen full of blue lines at
  startup.  Confirmed via debug instrumentation that `_write_panel` is called
  the correct number of times; the ghosting was a pure terminal rendering
  artifact.  This behaviour is consistent across all Textual versions tested
  (0.86–8.2.5) and cannot be fixed in application code without patching Textual.

  When an SSH session is detected (`SSH_CLIENT` / `SSH_TTY` /
  `SSH_CONNECTION` env vars, set automatically by OpenSSH), `--no-inline` is
  silently overridden and inline mode is used instead.  Inline mode uses delta
  updates (relative cursor movement) and renders correctly on all terminals.
  Set `BAMBOO_FORCE_NO_INLINE=1` to override the auto-switch for terminals
  known to handle alternate screen correctly over SSH.

- **`_compact_json` truncating `code_query` evidence.** `_compact_json` has a
  12 000-character limit applied to all evidence blobs.  A `code_query` result
  with a 23K source file produced a 25K JSON blob, truncated mid-source before
  the `"truncated": false` flag was reached.  The LLM received genuinely
  truncated content and correctly reported it.  Fixed in `_call_tool_and_collect`:
  `code_query` evidence is now handled specially — `source` is extracted before
  `_compact_json` and appended as a plain fenced block, so metadata always passes
  intact.

- **Diagram questions routing to `code_query` instead of RAG.**
  Questions like *"show me a diagram of the pilot states"* matched
  `_is_code_query_question` (verb `show me` + keyword `pilot`) and were
  routed to `code_query`, which returned an error because no `.py` file
  was present.  Fixed with a two-tier verb model: tier-1 source-access verbs
  (`download`, `fetch`, `look at`, …) match with any domain keyword; tier-2
  conceptual verbs (`show me`, `explain`, `describe`, …) only match with
  structural code keywords (`function`, `class`, `source`, …), not concept
  words like `pilot` alone.  Applied consistently in both
  `bamboo_answer.py` and `superuser_guard.py`.

- **LLM refusing to draw diagrams from partial RAG context.**
  `_SYSTEM_RAG`'s *"don't add unreferenced claims"* rule prevented the LLM
  from generating a Mermaid diagram when the documentation excerpts described
  states and transitions but didn't contain a pre-drawn diagram.  Added an
  explicit exception: when the user asks for a diagram and the LLM has enough
  knowledge to draw one, it should do so and label it as general knowledge.

- **Superuser gate not blocking questions without a file path.** The original
  guard regex required a `pilot/...py` path separator, so `"Look at pilot.py"`
  slipped through.  Replaced with a two-signal detector: any `*.py` token
  (bare filename or slash-path) OR an inspection verb combined with a repository
  keyword.  Both `bamboo_answer.py` and `superuser_guard.py` now use the same
  detection logic, kept in sync.

- **`"Look at pilot.py"` not routing to `code_query`.** Without a fast-path
  rule, the LLM planner failed to route bare filenames to `code_query` and fell
  back to monitoring evidence.  Fixed by adding `code_query` to the deterministic
  fast-path (see above).

- **Follow-up phrases hitting the topic guard.** After a `code_query` response,
  natural follow-ups such as `"yes please"`, `"please verify the full file"`, and
  `"download the full file"` were blocked by the topic guard as off-topic.
  Fixed by `_is_content_free_followup` pattern extension and the new
  `_is_code_review_continuation` interceptor in `_run_fast_path_intercepts`.

- **Streamlit — `stmermaid.mermaid()` AttributeError.** `streamlit-mermaid`
  0.3.0 exposes `st_mermaid(code=...)`, not `mermaid()`.  Fixed.

- **`ASKPANDA_PLUGIN` not cleared when sourcing env file.** The line was
  commented out in `bamboo_env_example.sh`, so sourcing it left a stale
  `cgsim` value from a previous session in the shell.  Now explicitly exported.

- **Plugin tool isolation — PanDA tools no longer visible to non-PanDA plugins.**
  `_build_deterministic_plan`, `_run_fast_path_intercepts`, and the LLM planner
  all gate PanDA-specific rules behind `plugin_id in _PANDA_PLUGINS`.

- **`cgsim.sim_query` — SQL generation token cap, enumeration, routing.**
  See v1.0.7 for full details; these fixes were backported into this release.

- **Streamlit `_fetch_evidence` — double-nesting and null-error guard.** Fixed.

- **`interfaces/shared/mcp_client.py` — mcp SDK 1.x compatibility.** Fixed.

- **`requirements-rag.txt` — `pysqlite3-binary` restricted to Linux.** Fixed.

### Changed

- **`pilot_code_query` → `code_query`.** The tool, module, test file, env vars,
  synthesis prompt, and all documentation have been renamed for generality.
  The tool now targets any GitHub repository (not just pilot3) and uses `file_path`
  (not `pilot_path`) as the input parameter.

  | Old | New |
  |---|---|
  | `pilot_code_query` | `code_query` |
  | `bamboo.tools.pilot_code_query` | `bamboo.tools.code_query` |
  | `PilotCodeQueryTool` | `CodeQueryTool` |
  | `fetch_pilot_code()` | `fetch_source_file()` |
  | `pilot_path` parameter | `file_path` parameter |
  | `BAMBOO_PILOT_REPO` | `BAMBOO_CODE_QUERY_REPO` |
  | `BAMBOO_PILOT_BRANCH` | `BAMBOO_CODE_QUERY_BRANCH` |
  | `_SYSTEM_PILOT_CODE_QUERY` | `_SYSTEM_CODE_QUERY` |
  | `tests/test_pilot_code_query.py` | `tests/test_code_query.py` |
  | `docs/tools/pilot_code_query.md` | `docs/tools/code_query.md` |


### Added

- **OpenSearch read-query tools.** Bamboo can now query any index on the CERN
  OpenSearch cluster directly from the TUI, Streamlit, or any MCP client —
  without needing the OpenSearch Dashboards web UI.

  - **`opensearch_query`** — general-purpose MCP tool that executes an
    OpenSearch DSL query (supplied as a JSON string) against any index pattern
    in a configurable allow-list.  Arguments: `index_pattern`, `query` (DSL
    JSON string), optional `max_hits` (1–100, default 10), optional
    `source_fields` projection.  Returns `{"hits": [...], "total": N,
    "took_ms": N, "aggregations": {...}}`.  Uses `ASKPANDA_OPENSEARCH`
    (shared read credential with harvester timeseries).  Allow-list controlled
    by `BAMBOO_OPENSEARCH_ALLOWED_INDICES` (default:
    `atlas_harvesterworkers-*,bamboomcp-promptlog-*`).

  - **`opensearch_promptlog_query`** — convenience wrapper pre-wired to
    `bamboomcp-promptlog-*` with the three large text fields
    (`system_prompt`, `user_prompt`, `response`) excluded from results by
    default.  Rich schema description in the tool definition lets the LLM
    construct useful queries without knowing the field names.  Supports
    session replay, tool-usage analytics, token cost comparisons, and
    per-provider breakdowns.

  - **`core/bamboo/llm/opensearch_client.py`** — new shared client factory
    (`create_os_client(password)`) used by all three OpenSearch paths (prompt
    log write, harvester timeseries read, general read).  Eliminates the
    duplicate connection logic previously duplicated between `prompt_log.py`
    and `harvester_timeseries_impl.py`.

  - Registered in `core.py` `TOOLS` dict.  25 new unit tests in
    `tests/test_opensearch_query.py` covering allow-list logic, error
    handling, max_hits clamping, aggregation passthrough, and the
    promptlog-query projection defaults.

  - `docs/opensearch.md` extended with a "Read queries from Bamboo" section:
    example DSL queries, architecture diagram, and "Adding a new index"
    instructions.

### Changed

- **`prompt_log._create_os_client` and
  `harvester_timeseries_impl.create_os_client` now delegate to the shared
  `bamboo.llm.opensearch_client.create_os_client` factory.**  Both functions
  are preserved at their original names for backward compatibility; no
  call-site or test changes are required.



### Added

- **OpenSearch prompt-log self-observability and analytics.**  Bamboo can now
  answer questions about its own usage — turn counts, session replay, FAQ
  analysis, tool call frequency, model/provider breakdowns — by querying the
  `bamboomcp-promptlog-*` index directly from the TUI or Streamlit.

  - **`bamboo_promptlog_rate` MCP tool.**  Rates a logged response (1–5 stars)
    by applying a partial `update` to the existing OpenSearch document.
    `prompt_log.py` gains `_last_doc_store` (deque maxlen=1),
    `get_last_doc_id()`, and `update_rating(index, doc_id, rating)`.
    The `rating` field (integer, nullable) is included in the index template
    mapping.  Uses the write credential (`BAMBOO_OPENSEARCH_PROMPTLOG`).

  - **`prompt_log.py` — index mapping and timestamp fixes.**
    `_ensure_index_template()` applies a `bamboomcp-promptlog` index template
    on the first write of each process, ensuring `@timestamp` is always mapped
    as `date` (not auto-detected as `text`).  This fixes date-range queries
    such as `gte:now/d` that silently returned zero results when the mapping
    was wrong.  Timestamps changed from `isoformat()+00:00` to explicit
    `Z`-suffix `strftime` format (`strict_date_optional_time` canonical form).
    Notification messages now include `session=` so the UUID is visible
    directly in the TUI system panel for use in session-replay queries.

  - **Promptlog fast-path routing in `bamboo_answer.py`.**
    `_is_promptlog_question()` detects self-observability queries
    (FAQ, session replay, tool-usage analytics, turn counts, model queries)
    and routes them directly to `opensearch_promptlog_query` via a new rule 7
    in `_build_deterministic_plan`, before the doc-search RAG fallback (which
    becomes rule 8).  `_build_promptlog_plan()` helper extracted consistent
    with `_build_code_query_plan`.  `# noqa: C901` added to
    `_build_deterministic_plan` (intentional dispatcher).

  - **Topic guard self-observability terms.**  `topic_guard.py` gains a
    `# Bamboo self-observability` block in `_ALLOW_TERMS` covering `session`,
    `turns`, `bamboo`, `opensearch`, `which model`, `tool usage`, `faq`, and
    related phrases.  These now fast-path to `keyword_allow` without invoking
    the LLM classifier, preventing prompt-log queries from being incorrectly
    rejected as off-topic.

  - **`opensearch_promptlog_query` description improvements.**  Accumulated
    fixes to the LLM-facing tool description across multiple iterations:
    OpenSearch date-math rules (`now/d`, `now-7d/d`); `size:0` rules
    (display queries must omit `size`; aggregation-only queries use
    `size:0 + source_fields=[]`); `total` field semantics (pre-size-limit
    document count, not value_count result); `session_id.keyword` fallback
    for indices created before the template; mandatory `user_prompt.keyword`
    for terms aggregations (without `.keyword` the field is tokenised,
    producing word-level buckets instead of full-question buckets);
    multi-user deployment note (cross-session queries must omit
    `session_id` filter); explicit FAQ examples.

- **LaTeX formula rendering in Streamlit.**  `_normalise_latex()` in
  `interfaces/streamlit/chat.py` converts common LLM LaTeX delimiter styles
  (`\[ \]`, `\( \)`, bare `[ ]` with a backslash in the content) to the
  `$$...$$` / `$...$` forms that Streamlit's built-in KaTeX renderer
  understands.  Applied to every assistant message before `st.markdown()`.
  No new dependencies — KaTeX is bundled in Streamlit.
  12 unit tests in `tests/test_normalise_latex.py`.

- **Slash commands — TUI.**  New commands added to `/help`:

  | Command | Action |
  |---|---|
  | `/faq [today\|week\|month]` | Most frequently asked questions from prompt logs; default scope is all time |
  | `/rate <1-5>` | Rate the most recent response; submits `bamboo_promptlog_rate` with the `(index, doc_id)` extracted from the last notification |

  `/rate` confirmation displays `★☆☆☆☆`–`★★★★★` stars inline.

- **Slash commands — Streamlit.**  `_expand_slash_command()` intercepts slash
  commands at the `st.chat_input` level before submission to the MCP server.
  `/help` and unknown commands render as inline assistant messages with no
  server round-trip.  Commands supported:

  | Command | Action |
  |---|---|
  | `/help`, `/?` | Formatted markdown command reference |
  | `/faq [today\|week\|month]` | Most frequently asked questions |
  | `/task <id>` | Summarise task status |
  | `/job <id>` | Analyse job failure |
  | `/rate <1-5>` | Rate the last response |

- **Star rating widget — Streamlit.**  `_render_rating_widget()` displays five
  colour-coded buttons below each assistant response: 🔴 1, 🟠 2, 🟡 3, 🟢 4,
  💚 5.  Clicking submits `bamboo_promptlog_rate` and reruns; the selected star
  is shown bold and a caption confirms the rating (e.g. "Your rating: ⭐⭐⭐⭐
  — Good (4/5)").  Widget is suppressed when `bamboo_promptlog_rate` is not
  registered on the server or no document has been indexed yet.
  `(index, doc_id)` is extracted from `bamboo_promptlog_status` notification
  events and stored in `st.session_state["last_doc_id"]`.

### Fixed

- **Prompt-log queries routing to RAG instead of OpenSearch.**  Questions such
  as *"show me the frequently asked questions"*, *"how many turns today?"*, and
  *"what was the last question I asked?"* were falling through to the doc-search
  fallback because `_build_deterministic_plan` had no promptlog routing rule.
  Fixed by the new rule 7 described above.

- **FAQ aggregation returning wrong counts.**  `terms` aggregations on the
  `user_prompt` field (a `text` type) bucket on individual tokens rather than
  full question strings, causing *"What is PanDA?"* (asked 4 times) to appear
  as three separate single-occurrence buckets.  Fixed by enforcing
  `user_prompt.keyword` in the tool description and in the `/faq` command
  question text.

- **Session replay and turn queries returning zero results.**  Two causes:
  (1) The `@timestamp` field was auto-mapped as `text` on indices created
  before the template fix, silently breaking `range`/date-math queries.
  Fixed by the index template.  (2) `term:{session_id:...}` queries returned
  zero on such indices because `session_id` was also mapped as `text`.
  `session_id.keyword` fallback documented in the tool description.

- **`test_superuser_guard.py` failures when run after `test_normalise_latex`.**
  `_import_normalise_latex` was leaving `interfaces.shared` stubbed as a plain
  `types.ModuleType` in `sys.modules`, causing subsequent imports of
  `interfaces.shared.superuser_guard` to return `MagicMock` objects.  Fixed by
  wrapping stub injection in a `try/finally` that restores `sys.modules` to its
  original state after the import.



### Added

- **`/script [filename]` TUI command.**  Extracts fenced code blocks from the
  last assistant response and writes them to the current working directory.
  Filename resolution order: (1) user-supplied argument, (2) label in the
  response body (`Script: foo.py`, `File: foo.py`, `Save the script as foo.C`,
  code fence with inline filename), (3) auto-generated from language + timestamp.
  Multiple blocks are written with numeric suffixes, each using its own detected
  language extension (`.py`, `.cpp`, `.sh`, `.C`, etc.).  If the user supplies
  a filename without an extension, the first block's language extension is appended
  automatically.  Added to `/help`.

- **Streamlit download button.**  `_render_script_download()` detects fenced code
  blocks in the last assistant message and renders `st.download_button` for each.
  File content is streamed directly to the browser — no server-side file is written.
  Correct approach for browser-based deployment.  Suggested filename is honoured
  using the same resolution order as the TUI `/script` command.

- **`/script` in Streamlit slash commands.**  Typing `/script` in the Streamlit
  chat input displays an explanation of the download button mechanism.

### Fixed

- **`/rates` missing entries (exists filter).**  `range:{rating:{gte:1}}` silently
  skips documents where the `rating` field is absent.  Changed to
  `exists:{field:rating}` so all rated documents are returned regardless of how the
  field was mapped.  `max_hits` set explicitly to 50.

- **`/rates` 400 parsing_exception.**  The submitted question was embedding `max_hits`
  and `source_fields` inside the JSON query body, causing OpenSearch to return
  `Unknown key for a START_OBJECT in [bool]`.  Rewrote the question text to label
  each argument separately and provide the DSL body as a standalone JSON string.

- **`/rates` showing only top entries.**  `user_prompt` (full synthesis prompt) was
  included in `source_fields`, consuming large amounts of context window and causing
  early truncation.  Replaced with `raw_question` (short original question).

- **`/rates` wrong extensions on multi-block output.**  When `/script calc.py` was
  used with a response containing multiple code blocks (Python + C++ + bash), all
  subsequent blocks inherited the `.py` extension from the user-supplied filename
  instead of using their own detected language extension.  Each block now uses its
  own `_lang_to_extension` result for blocks after the first.

- **`/script` missing extension when user omits it.**  `/script rnd` produced a file
  named `rnd` with no extension.  If the user-supplied filename has no extension,
  the first block's detected language extension is now appended automatically.

- **`/script` not honouring "Save the script as X.C" pattern.**  The
  `_extract_suggested_filename` function did not match the LLM's common phrasing
  *"Save the script as random_numbers.C"*.  Added a `save_re` pattern matching
  `save/name/call ... as <filename.ext>` (case-insensitive) as a fourth extraction
  strategy, after `Script:/File:` labels and inline fence filenames.

- **ROOT `.C` extension missing.**  Added `"root": ".C"` to both the TUI
  `_lang_to_extension` map and the Streamlit `_LANG_EXT_MAP` so ROOT macro
  code blocks get the correct `.C` extension in auto-named output.

- **`/rates` and "show me all the rates" misrouted to PanDA jobs.**  Promptlog
  routing was checked after the topic guard, which could substitute the original
  question with a prior PanDA-domain turn (context bleed).  Moved
  `_is_promptlog_question` into `_run_fast_path_intercepts`, before the topic
  guard, so the original question text is always used.

- **Rating vocabulary not in routing signals.**  "Show me all the rates from today"
  was misrouted to PanDA job tools because "rates" is ambiguous.  Added `rating`,
  `ratings`, `rated`, `star rating`, `lowest rated`, `highest rated`, `average
  rating` to `_PROMPTLOG_SIGNALS`, `_PROMPTLOG_PHRASES`, and `topic_guard`
  `_ALLOW_TERMS`.

- **`raw_question.keyword` for accurate FAQ aggregations.**  `user_prompt` contains
  the full synthesis context (question + evidence) and is unique per turn even for
  identical questions — aggregating on `user_prompt.keyword` produces no useful
  frequency data.  Added `raw_question: str | None` to `log_prompt()` and threaded
  it from `execute_plan()` via `call_llm()`.  `raw_question` stores the user's
  original typed question and its `.keyword` sub-field enables correct `terms`
  aggregations for `/faq`.  Updated all FAQ examples and `/faq` command text to
  use `raw_question.keyword`.



## v1.0.7 — 2026-05-15

### Added

- **`cgsim.sim_query` — natural-language to SQL tool for the CGSim simulation
  output database (`packages/askcgsim/`).** Answers questions about a CGSim
  simulation run by translating natural-language questions into SQL, executing
  them read-only against the local SQLite database, and summarising the results
  in natural language via a second LLM call.

  New files:

  | File | Purpose |
  |---|---|
  | `askcgsim/sim_query_schema.py` | SQL guard (AST allow-list), schema context string, and LLM prompt builders for both the SQL-generation and summarisation calls. Zero bamboo-core dependency. |
  | `askcgsim/sim_query_impl.py` | Full NL→SQL→execute→NL pipeline. Both LLM calls are async; SQLite execution runs synchronously on the event loop thread (consistent with the DuckDB precedent in `panda_jobs_query`). |
  | `askcgsim/sim_query.py` | Thin re-export wrapper with `ImportError` fallback if `sqlglot` is absent. |
  | `askcgsim/cgsim_reader.py` | `cgsim_reader.py` vendored from the `sqlite-reader` repository. Provides typed structured access to the EVENTS table via `CGSimReader` and `EventRow`. |
  | `tests/test_sim_query.py` | 64 unit tests covering the guard (every rejection rule, LIMIT injection, aggregation cap, CTE allowance), the full pipeline (happy path, cannot-answer, guard rejection, execution error, wrong database, summarisation failure, truncation), `CgsimSimQueryTool.call()`, schema context caching, and prompt builder shape. |

  Security — four independent read-only layers:

  1. SQLite URI `file:{path}?mode=ro` — the driver refuses any write at the OS level.
  2. `PRAGMA query_only = ON` — a second enforcement inside the SQLite library.
  3. sqlglot AST guard (`validate_and_guard`) — parses with the SQLite dialect;
     enforces single statement, SELECT-only root, no forbidden constructs at any
     AST depth, no system tables (`sqlite_master`, `sqlite_sequence`, …), and a
     table allow-list (`events` only). Queries without a LIMIT get `LIMIT 200`
     injected; aggregation queries (`GROUP BY`) get `LIMIT 1000`.
  4. Local-only deployment — `CGSIM_DB_PATH` is a local filesystem path.

  `pyproject.toml` changes: `cgsim.sim_query` entry point uncommented;
  `sqlglot>=25.0` added as a package dependency.

- **`cgsim.sim_query` documentation** — `docs/cgsim-database.md`,
  `docs/tools/cgsim_sim_query.md`; updated `docs/tools/README-mcp_tools.md`,
  `docs/question-cheatsheet.md`, `README.md`.

### Fixed

- **`cgsim.sim_query` routing** — `plugin_id` was not reaching the fast-path
  interceptors. `_run_fast_path_intercepts` and `_run_db_query_fast_path` had no
  `plugin_id` parameter, so every `_build_deterministic_plan` call inside them
  defaulted to `"atlas"` and routed simulation questions to `panda_jobs_query`.
  `plugin_id` is now threaded through the full chain:
  `_route()` → `_run_fast_path_intercepts` → `_run_db_query_fast_path` →
  `_build_deterministic_plan`. The CGSim branch in `_build_deterministic_plan`
  was also moved before the `_is_jobs_db_question` check so it takes priority.

- **LLM planner routing for CGSim (fast-path off)** — the planner had no
  plugin awareness, causing it to select `panda_jobs_query` for simulation
  questions when `BAMBOO_FAST_PATH=0`. Two changes:
  - `plugin_id` is now passed from `bamboo_answer.call()` through `plan_args`
    to `bamboo_plan_tool.call()` and into `build_planner_system_prompt()`.
  - `build_planner_system_prompt` now dispatches to a plugin-specific prompt
    builder. The CGSim prompt (`_build_cgsim_planner_prompt`) contains no PanDA
    vocabulary — it only knows `cgsim.sim_query`, `cgsim.doc_search`, and
    `cgsim.doc_bm25`, with clear guidance to prefer `cgsim.sim_query` for any
    simulation-data question.

- **Wrong-database error handling** — when the file at `CGSIM_DB_PATH` exists
  but contains no `EVENTS` table (empty file, wrong database), the tool now
  returns a specific `_wrong_database_evidence` error message rather than the
  generic "query could not be executed" message.

- **SQL generation prompt — ambiguous follow-up questions** — added explicit
  examples for "show me all jobs" / "list all job IDs" (`SELECT DISTINCT
  JOB_ID FROM EVENTS`) and strengthened the prompt to state that `EVENTS` is
  the only permitted table. Without this, follow-up questions like "show me all
  jobs" were generating PanDA-schema SQL (`SELECT * FROM jobs`) which the AST
  guard correctly blocked.

- **Summarisation prompt — tie/uniform distribution** — added an explicit
  instruction to report when all rows share the same ranked value (e.g. all
  sites have the same job count) rather than reporting only the top row as if
  it were uniquely the winner.

### Improved

- **`cgsim.sim_query` tracing** — three sub-spans are now emitted inside
  `fetch_and_analyse` so `/tracing` shows the breakdown between the two LLM
  calls and the SQLite execution:
  - `cgsim.sim_query/sql_generation` (`llm_call`) — SQL generation latency and
    token counts.
  - `cgsim.sim_query/sqlite_execute` (`tool_call`) — SQLite execution time and
    row count.
  - `cgsim.sim_query/summarisation` (`llm_call`) — summarisation latency and
    token counts.
  All three spans correctly wrap the operation they measure (the `generate()`
  call is now *inside* the `async with span(...)` block).

- **Planner tracing fix** — the `bamboo_plan` `llm_call` span previously
  recorded 0 ms because `client.generate()` was called *before* the span
  opened. Fixed by moving `generate()` inside the span, consistent with the
  corrected `cgsim.sim_query` spans.

- **`cgsim.sim_query` synthesis bypass** — when `cgsim.sim_query` returns a
  non-null `summary`, the executor now returns it directly without a redundant
  `bamboo_llm_answer` synthesis call. This saves ~3 seconds per query (one
  full LLM round-trip). The synthesis span is still emitted with
  `bypass="cgsim_summary"` for tracing consistency. The bypass falls through to
  normal synthesis if `summary` is null (e.g. summarisation LLM failure).

- **TUI fallback banner** — replaced the AskPanDA ASCII art in `FALLBACK_BANNER`
  with "Bamboo MCP" ASCII art (standard figlet font, 5-line layout matching the
  original height). Also updated all transient UI strings that previously said
  "AskPanDA": the default `display_name`, input placeholder, response panel
  title, and both error-fallback display names in `_load_banner`. Plugin-specific
  banners (e.g. AskCGSim) still override the fallback once `ui_manifest` loads.

- **Agent live progress callback** (`interfaces/agent/agent.py`,
  `scripts/bamboo_agent.py`): `BambooAgent` now accepts an optional
  `progress_callback: Callable[[str], None]` parameter.  The callback is
  invoked at each key moment in the ReAct loop — tool discovery, step start,
  tool call, observation size, eval result, and synthesis — so callers can
  display live status without coupling the agent to any output channel.
  `AgentResult` gains a new `llm_calls: int` field counting every
  `bamboo_llm_answer` MCP call made during the run (reason + evaluate +
  synthesise combined).  Step progress display changed from `Step X/Y` to
  `Step X (max Y)` to clarify that `max_steps` is a ceiling, not a target.
  A `✔  Done — N step(s), M LLM call(s), confidence=F` line is emitted after
  synthesis completes.
- **Agent CLI progress display and `--quiet` flag** (`scripts/bamboo_agent.py`):
  live progress is now shown on stderr by default using `\r` overwrite (single
  tidy line, no scrolling).  Line width is capped to the terminal width via
  `shutil.get_terminal_size()`.  The `--quiet` flag suppresses all progress
  output, useful when piping stdout or capturing JSON output.  `llm_calls=N`
  is included in both the printed footer line and `--output-json` output.
  The module docstring now contains a full annotated table of the MCP HTTP
  message sequence (POST/GET/DELETE) so operators understand what each server
  log line represents.

---

## 2026-05-12  Security — four independent read-only layers:

  1. SQLite URI `file:{path}?mode=ro` — the driver refuses any write at the OS level.
  2. `PRAGMA query_only = ON` — a second enforcement inside the SQLite library.
  3. sqlglot AST guard (`validate_and_guard`) — parses with the SQLite dialect;
     enforces single statement, SELECT-only root, no forbidden constructs at any
     AST depth, no system tables (`sqlite_master`, `sqlite_sequence`, …), and a
     table allow-list (`events` only). Queries without a LIMIT get `LIMIT 200`
     injected; aggregation queries (`GROUP BY`) get `LIMIT 1000`.
  4. Local-only deployment — `CGSIM_DB_PATH` is a local filesystem path.

  Pipeline: LLM call 1 (temperature 0.0, 512 tokens) generates SQL using a
  system prompt that embeds the full EVENTS schema, all METADATA fields by
  event type, `json_extract()` guidance, the `CANNOT_ANSWER` sentinel, explicit
  exclusion of the uncalibrated `cost` field, and eight worked example patterns.
  The generated SQL is fence-stripped, checked for refusals, and passed through
  the AST guard before execution. LLM call 2 (temperature 0.2, 1024 tokens)
  receives the original question, the executed SQL, and the raw results as JSON,
  and returns a natural-language summary with correct units. LLM call 2 is
  non-fatal: if it fails, the raw evidence dict is still returned with
  `summary: null`.

  `pyproject.toml` changes: `cgsim.sim_query` entry point uncommented;
  `sqlglot>=25.0` added as a package dependency.

- **`cgsim.sim_query` documentation.**

  | File | Description |
  |---|---|
  | `docs/cgsim-database.md` | Full reference: EVENTS schema, all METADATA fields by event type, total wall-clock time formula, eight example questions with generated SQL, four-layer security model, two-LLM-call pipeline diagram, and configuration. |
  | `docs/tools/cgsim_sim_query.md` | Tool reference card: purpose, inputs, data source, pipeline summary, guard rules table, full output key reference, configuration, key design notes. |

  Updated files:

  - `docs/tools/README-mcp_tools.md` — new "CGSim simulation data tools"
    section with `cgsim.sim_query`; CGSim plugin table updated.
  - `docs/question-cheatsheet.md` — new `cgsim.sim_query` section with six
    themed question groups (job timing, site analysis, network congestion, I/O
    bottleneck, job health, full job timeline).
  - `README.md` — `docs/cgsim-database.md` added to the docs table;
    `cgsim.sim_query` added to the AskCGSim plugin tools table; status blurb
    updated to reflect the new tool.

---

## 2026-05-12

### Fixed

- **`panda_jobs_query`: site-scoped queries returned 0 rows (bamboo_answer.py,
  jobs_query_impl.py, jobs_query_schema.py).** Two bugs combined to produce
  empty results for any site-scoped jobs query such as "Show me 10 jobs at BNL
  that failed with pilot error code 1324".

  Bug 1 (bamboo_answer.py): the solo `panda_jobs_query` fast-path never
  extracted the site name from the question and never populated the `queue`
  argument, even though the combined site-health path (panda_harvester_workers
  + panda_jobs_query) already did this correctly. The fix calls
  `_extract_site_from_question()` and sets `jobs_args["queue"] = site`
  in the fast-path, mirroring the site-health path.

  Bug 2 (jobs_query_schema.py): the SQL system prompt examples used exact
  equality (`_queue = 'BNL'`) for site filtering, but the actual `_queue`
  column values are full queue names such as `BNL_ATLAS_TIER1` and
  `BNL_ATLAS_TIER1-condor`. The LLM faithfully followed the examples and
  generated non-matching WHERE clauses. Fixed by updating all prompt examples
  and rules to use `ILIKE 'SITE%'` prefix matching, and by changing the queue
  hint appended in `jobs_query_impl.call()` from `(focus on queue: SITE)` to
  the explicit SQL instruction `(filter _queue ILIKE 'SITE%')`.

- **`panda_jobs_query`: site error counts were wrong when querying
  `errors_by_count` for site-scoped questions (jobs_query_schema.py,
  docs/jobs-database.md).** `errors_by_count` is populated from a separate
  BigPanDA summary endpoint and its `count` values do not match `COUNT(*)`
  on the `jobs` table. For example, "most common failures at BNL" via
  `errors_by_count` reported pilot:1150 as 7 jobs, while aggregating the
  `jobs` table directly found 42.

  Fixed by updating the SQL system prompt to always use `COUNT(*) GROUP BY`
  on the `jobs` table for site-scoped failure frequency questions, and to
  reserve `errors_by_count` only for global cross-queue rankings (no site
  filter). New example queries for "most common failures at SITE" and "top
  errors at SITE" now use `jobs` with `GROUP BY piloterrorcode, exeerrorcode`.
  The fallback schema description for `errors_by_count.count` is updated to
  document the separate-source semantics.

- **`panda_jobs_query`: "most common failures" questions routed to RAG instead
  of the jobs DB (bamboo_answer.py).** Phrases like "most common job failures
  at BNL" and "top failures at AGLT2" were not in `_JOBS_DB_SIGNALS` so they
  fell through to RAG retrieval, returning documentation text instead of live
  DB results. Added `"failures at"`, `"top failures"`, `"job failure"`,
  `"job failures"`, `"job error"`, `"job errors"`, `"common failure"`, and
  `"common error"` to both `_JOBS_DB_SIGNALS` and `_JOBS_DB_SPECIFIC_SIGNALS`.

- **`cric_query`: copytool follow-up questions routed to RAG instead of CRIC
  (bamboo_answer.py).** Questions like "Are any other sites using object
  stores?" or "Which sites use rucio?" were not recognised as CRIC questions
  because copytool names and object-store vocabulary were absent from
  `_CRIC_SIGNALS`. Added `"objectstore"`, `"object store"`, `"gfalcopy"`,
  `"rucio copytool"`, `"using rucio"`, `"using objectstore"`, and
  `"using gfal"` to `_CRIC_SIGNALS` so these route directly to `cric_query`
  without depending on the narrower follow-up regex.

---

## 2026-05-11

### Fixed
- ChromaDB RAG tools (panda_doc_search, panda_doc_bm25, and their ePIC and
  CGSim equivalents) now work on systems with SQLite < 3.35.0, such as CERN
  lxplus (AlmaLinux 9 / RHEL 9). A new compatibility shim
  (bamboo/tools/_sqlite_compat.py) monkey-patches pysqlite3-binary into
  sys.modules before ChromaDB is imported when the system SQLite is too old.
  The fix is a no-op on systems where the system SQLite is already sufficient.
  Add pysqlite3-binary to your environment: pip install -r requirements-rag.txt

## 2026-04-29

### Added

- **CGSim plugin (`packages/askcgsim/`).** A new Bamboo MCP plugin for the
  CGSim / SimGrid distributed computing simulator. CGSim is a SimGrid-based
  framework for simulating large-scale computing grids such as the WLCG; it
  ingests historical PanDA job records for calibration and is designed to
  simulate infrastructures managed by PanDA.

  Entry points registered under `bamboo.tools`:

  | Entry point | Tool name | Description |
  |---|---|---|
  | `cgsim.doc_search` | `cgsim.doc_search` | ChromaDB vector similarity search over CGSim / SimGrid documentation |
  | `cgsim.doc_bm25` | `cgsim.doc_bm25` | BM25 keyword search over the same corpus |
  | `cgsim.ui_manifest` | `cgsim.ui_manifest` | TUI branding: block-letter banner, green accent, "Bamboo – AskCGSim" display name |

  The default ChromaDB collection name is `cgsim_docs`, distinct from
  `atlas_docs` and `epic_docs` so all three corpora can coexist in the same
  ChromaDB directory. Tool names use dot notation throughout (matching the
  entry point key), which is a requirement for all Bamboo plugins — using
  underscores in `get_definition()["name"]` causes "Unknown tool" errors
  because core overwrites the name with the entry point key.

  Future tools are stubbed and commented out in `pyproject.toml`:
  `cgsim.sim_query`, `cgsim.site_status`, `cgsim.calibration_results`,
  `cgsim.event_monitor` — all planned as read-only SQLite interfaces to the
  CGSim simulation output database.

- **`cgsim.sim_query` security model documented.** The planned SQLite tool
  will enforce read-only access at four independent layers: SQLite URI
  `mode=ro` flag, `PRAGMA query_only = ON`, sqlglot AST validation against a
  CGSim table allow-list, and local-only filesystem access via `CGSIM_DB_PATH`.
  This mirrors the security pattern of `panda_jobs_query` (DuckDB) but uses
  SQLite since that is what CGSim produces.

- **Plugin-aware synthesis prompts.** `bamboo_executor.py` now selects
  synthesis system prompts based on the active plugin (`ASKPANDA_PLUGIN`).
  Three CGSim-specific prompts were added: `_SYSTEM_RAG_CGSIM`,
  `_SYSTEM_RAG_NO_CONTEXT_CGSIM`, and `_SYSTEM_GENERIC_CGSIM`. These identify
  the assistant as Bamboo (not AskPanDA), state that CGSim/PanDA correlation
  questions are explicitly in scope, and instruct the LLM not to deflect
  cross-domain questions. The `plugin_id` parameter is now threaded through the
  full call chain: `bamboo_answer.call()` -> `_route()` ->
  `_build_deterministic_plan()` -> `execute_plan()` ->
  `_build_synthesis_prompt()` -> `_pick_synthesis_prompt()`.

- **Plugin-aware identity in `templates.py`.** `get_bamboo_system_prompt()`
  now accepts a `plugin_id` parameter and returns a plugin-appropriate identity
  string from `_PLUGIN_IDENTITY`. For CGSim the identity names the assistant
  Bamboo, describes the CGSim/SimGrid/PanDA domain, and explicitly welcomes
  PanDA/CGSim correlation questions. `llm_passthrough.py` reads
  `ASKPANDA_PLUGIN` and passes it through.

- **Plugin-aware doc tool routing.** `_PLUGIN_DOC_TOOLS` and
  `_DEFAULT_DOC_TOOLS` in `bamboo_executor.py` are now ordered lists (not
  sets) mapping plugin IDs to their doc tool pair, ensuring stable plan
  ordering (vector search always before BM25). `_build_deterministic_plan()`
  uses the plugin-appropriate doc tools for the fallback RAG route.

- **`BAMBOO_FAST_PATH` environment variable.** Fast-path routing can now be
  enabled or disabled at startup via the `BAMBOO_FAST_PATH` env var. Set to
  `0`, `off`, or `false` to start with the LLM planner handling all routing;
  any other value (or unset) leaves fast-path on. Both the Textual TUI and
  Streamlit interface read this at startup. The default in
  `bamboo_env_example.sh` is `0` (off), recommended for CGSim where fast-path
  intercepts are tuned for PanDA/ATLAS patterns.

- **`ASKPANDA_PLUGIN` environment variable documented.** Added to
  `bamboo_env_example.sh` with `atlas`, `epic`, and `cgsim` as documented
  choices. Added to env var tables in `docs/interfaces.md` and `CLAUDE.md`.

- **CGSim topic guard terms.** `topic_guard.py` now includes CGSim and
  SimGrid terms in `_ALLOW_TERMS` (`cgsim`, `simgrid`, `assignjob`,
  `getresourceinformation`, `onjobend`, `onsimulationend`, `netzone`,
  `calibration`, `job wall time`, `job queue time`, `simulation`, `simulator`,
  `computing grid`, `distributed computing`). The rejection message and LLM
  classifier system prompt were updated to name CGSim and SimGrid as in-scope
  domains.

- **Dynamic banner height in the Textual TUI.** `_render_banner()` and
  `_render_banner_placeholder()` now set the `#banner` container height
  programmatically after rendering using `len(banner_lines) + 4` (2 Panel
  borders + 2 CSS padding rows). This ensures the bottom border is never
  clipped regardless of plugin banner height. The CGSim block-letter banner is
  6 lines tall vs the 5-line ATLAS/ePIC banners, which triggered the bug.

- **`python -m bamboo.server_http` entry point** (`core/bamboo/server_http.py`).
  A dedicated HTTP server launcher that reads `BAMBOO_HTTP_HOST` (default
  `127.0.0.1`), `BAMBOO_HTTP_PORT` (default `8000`), and
  `BAMBOO_HTTP_LOG_LEVEL` (default `info`) from environment variables or CLI
  flags, and prints a startup banner to stderr showing the MCP endpoint URL,
  health check URL, worker count, and auth status. This replaces the need to
  memorise the `uvicorn bamboo.entrypoints.http:app` invocation.

- **`requirements-http.txt`** — `uvicorn>=0.29` and `starlette>=0.36`
  extracted as a named dependency group for the HTTP server transport.

- **`GET /healthz` documented.** The existing liveness endpoint in
  `bamboo.entrypoints.http` is now prominently documented in
  `docs/http-server.md`, `README.md`, `CLAUDE.md`, and `bamboo_env_example.sh`.
  Suitable for Kubernetes liveness/readiness probes (`httpGet: path: /healthz`),
  load balancer health checks, and `curl --fail` monitoring scripts.

- **Plugin-aware tool list filtering (`core/bamboo/core.py`).** The
  `list_tools` MCP handler now only exposes tools whose entry-point namespace
  matches the active plugin (`ASKPANDA_PLUGIN`). Core tools in the `TOOLS`
  dict (`bamboo_health`, `bamboo_answer`, etc.) are always included.

  Before this change, all installed plugins' tool descriptions were sent to the
  LLM on every call — an ATLAS user was paying token cost for CGSim tool
  descriptions and vice versa. With three plugins at roughly three tools each,
  this was approximately nine wasted tool descriptions per call.

  The filtering applies only to `list_tools`. `call_tool` is unaffected — all
  plugin tools remain callable regardless of `ASKPANDA_PLUGIN`. The namespace
  used for filtering is the part of the entry-point key before the first dot
  (`atlas.task_status` → namespace `atlas`). This means the namespace in the
  entry-point key must exactly match the value set in `ASKPANDA_PLUGIN`; if
  they differ the plugin's tools will never appear in `list_tools`.

- **`tests/test_plugin_tool_filter.py`** — 10 tests covering the filtering
  logic: correct tools included per plugin, cross-plugin tools excluded,
  unknown plugin returns empty, env var drives filter, default is `atlas`.

- **Streamlit plugin selectbox extended.** The sidebar plugin selector now
  includes `cgsim` alongside `atlas` and `epic`. The default index is derived
  dynamically from `ASKPANDA_PLUGIN` rather than a hardcoded position.

### Changed

- **`_PLUGIN_DOC_TOOLS` and `_DEFAULT_DOC_TOOLS` changed from sets to lists.**
  Python sets have no guaranteed iteration order; using `list(set)[0]` to pick
  doc tools produced non-deterministic plan ordering. Both constants are now
  ordered lists with vector search (`doc_search`) always at index 0 and BM25
  (`doc_bm25`) at index 1.

- **AskCGSim synthesis prompts updated to welcome PanDA/CGSim correlation.**
  The initial CGSim prompts instructed the LLM to avoid framing answers in
  terms of PanDA or ATLAS. This was over-cautious: CGSim ingests PanDA job
  records for calibration and users legitimately ask about the integration.
  All three AskCGSim synthesis prompts and the `_PLUGIN_IDENTITY["cgsim"]` string
  in `templates.py` now explicitly state that CGSim/PanDA correlation questions
  are in scope and should be answered directly.

- **`bamboo_env_example.sh` RAG section updated.** The default
  `BAMBOO_CHROMA_COLLECTION` value changed from `document_monitor_agent` to
  `atlas_docs`, matching the ATLAS plugin default. A new comment lists all
  three per-plugin defaults (`atlas_docs`, `epic_docs`, `cgsim_docs`).

### Fixed

- **All plugins' tool descriptions sent to LLM on every call (token waste).**
  `list_tools` was returning entry-point tools from all installed plugins
  regardless of `ASKPANDA_PLUGIN`. With ATLAS, ePIC, and CGSim all installed,
  every LLM call received approximately nine extra tool descriptions it would
  never use. Fixed by filtering in `list_tools` to the active plugin's
  namespace only.

- **"Unknown tool" errors for CGSim doc tools.** `get_definition()["name"]`
  in `cgsim/doc_rag.py` and `cgsim/doc_bm25.py` returned underscore names
  (`cgsim_doc_search`, `cgsim_doc_bm25`). Core overwrites the definition name
  with the entry point key (dot notation: `cgsim.doc_search`,
  `cgsim.doc_bm25`), so the LLM was trying to call the underscore names while
  the server only exposed the dot names. Fixed by aligning `get_definition()`
  to return dot-notation names matching the entry point keys.

- **PanDA/ATLAS framing in CGSim answers.** Synthesis prompts in
  `bamboo_executor.py` were hardcoded for PanDA/ATLAS regardless of the active
  plugin, causing the LLM to begin every CGSim answer with "in the context of
  PanDA/ATLAS workflows". Fixed by making `_build_synthesis_prompt()`,
  `_pick_synthesis_prompt()`, and `execute_plan()` plugin-aware, and by adding
  CGSim-specific prompt constants.

- **CGSim questions rejected by topic guard.** "How does CGSim work?" reached
  the LLM classifier stage and was denied because `cgsim` and `simgrid` were
  not in `_ALLOW_TERMS`. Fixed by adding a CGSim/SimGrid keyword section to
  the allow list.

- **Banner bottom border clipped for CGSim.** The `#banner` CSS rule had a
  hardcoded `height: 9` sized for the 5-line ATLAS/ePIC banners. The CGSim
  block-letter banner is 6 lines, causing the bottom border to be cut off.
  Fixed by computing the height dynamically in `_render_banner()`.

### New files

| File | Purpose |
|---|---|
| `packages/askcgsim/askcgsim/__init__.py` | AskCGSim plugin package |
| `packages/askcgsim/askcgsim/doc_rag.py` | `cgsim.doc_search` tool |
| `packages/askcgsim/askcgsim/doc_bm25.py` | `cgsim.doc_bm25` tool |
| `packages/askcgsim/askcgsim/ui_manifest.py` | `cgsim.ui_manifest` tool |
| `packages/askcgsim/askcgsim/banner.txt` | 6-line block-letter CGSim banner |
| `packages/askcgsim/pyproject.toml` | Plugin entry points and metadata |
| `packages/askcgsim/tests/test_cgsim_plugin.py` | 30 tests covering all three tools |
| `core/bamboo/server_http.py` | `python -m bamboo.server_http` entry point |
| `requirements-http.txt` | HTTP server dependencies (uvicorn, starlette) |
| `tests/test_prompt_templates.py` | 9 tests for plugin-aware system prompts |
| `tests/test_plugin_tool_filter.py` | 10 tests for plugin-aware tool list filtering |
| `docs/tools/cgsim_doc_search.md` | Per-tool reference for `cgsim.doc_search` |
| `docs/tools/cgsim_doc_bm25.md` | Per-tool reference for `cgsim.doc_bm25` |

---



## 2026-04-08

### Added Bamboo MCP can now be built and distributed
  as a Docker image, enabling deployment on Kubernetes and easy distribution
  to users who want a self-contained environment.

  The image supports three runtime modes selected via the container command:

  | Command | Mode | Use case |
  |---|---|---|
  | *(default)* `server` | HTTP MCP server on port 8000 | Kubernetes, Docker Compose |
  | `tui` | Interactive Textual TUI | `docker run -it` for end users |
  | `stdio` | stdio MCP server | Claude Desktop integration |

  The Textual TUI is always installed in the image so that interactive use
  requires no separate build variant.

- **Multi-stage `Dockerfile`** (`docker/Dockerfile`). A `builder` stage
  installs all packages into `/opt/venv`; the `final` stage copies only the
  venv (no build tools, no source tree). Key properties:

  - Base image: `python:3.11-slim`.
  - Non-root user `bamboo` (UID 1000) for Kubernetes PSA compliance.
  - Well-known volume mount points at `/data/jobs`, `/data/cric`,
    `/data/chroma`, and `/data/trace`.
  - Default LLM provider set to **Google Gemini** (`gemini-2.0-flash`) for
    all three profiles (default, fast, reasoning).
  - `HEALTHCHECK` via `GET /healthz` (the existing endpoint in
    `bamboo.entrypoints.http`).

- **Build arguments** for optional dependency groups:

  | Argument | Default | Controls |
  |---|---|---|
  | `INSTALL_GEMINI` | `true` | Google Generative AI SDK |
  | `INSTALL_ANTHROPIC` | `false` | Anthropic SDK |
  | `INSTALL_OPENAI` | `false` | OpenAI SDK |
  | `INSTALL_RAG` | `false` | ChromaDB + BM25 |
  | `INSTALL_OTEL` | `false` | OpenTelemetry OTLP exporter |
  | `INSTALL_CERN_CA` | `true` | CERN Grid CA appended to certifi |

- **CERN Grid CA baked into the image.** When `INSTALL_CERN_CA=true` (the
  default), the builder stage downloads the CERN Root CA 2 and CERN Grid CA 2
  from `cafiles.cern.ch`, converts them from DER to PEM, and appends both to
  the certifi bundle. This allows `httpx` to verify the PanDA MCP server
  (`aipanda120.cern.ch:8443`) without setting `PANDA_MCP_TLS_VERIFY=0`.
  If `cafiles.cern.ch` is unreachable during the build (air-gapped
  environment), the build continues and the CA step is silently skipped.

- **`docker/entrypoint.sh`** — dispatch script that maps the container
  command to the correct Python invocation (`uvicorn`, Textual TUI, or
  `bamboo.server` stdio). Unknown commands fall through to `exec "$@"` for
  one-off debugging (e.g. `docker run bamboo-mcp python -m bamboo tools list`).

- **`docker/docker-compose.yml`** — local development and integration testing
  configuration. Defines two services: `bamboo-server` (HTTP server, always
  started) and `bamboo-tui` (interactive TUI, under the `tui` Compose
  profile). The TUI service connects to the server via `MCP_URL`. Host paths
  for DuckDB files are configured via `PANDA_DUCKDB_HOST_PATH` and
  `CRIC_DUCKDB_HOST_PATH` environment variables.

- **`docker/kubernetes/bamboo-mcp.yaml`** — Kubernetes deployment skeleton
  including Deployment, Service, ConfigMap, and PersistentVolumeClaims for
  the jobs and CRIC DuckDB volumes. The manifest uses the existing `/healthz`
  endpoint for both liveness and readiness probes. Includes a note on
  sticky-session requirements when scaling beyond one replica (the HTTP server
  holds in-process MCP session state).

- **`docker/docs/docker.md`** — usage documentation covering build arguments,
  all three runtime modes, Docker Compose workflow, Kubernetes quick-start,
  the CERN CA setup, and a one-liner for converting `bamboo_env.sh` to a
  Docker-compatible `bamboo.env.docker` file.

- **`.dockerignore`** — excludes test artefacts, `__pycache__`, secrets
  (`bamboo_env.sh`, `*.env`), DuckDB/ChromaDB files, docs, and log files
  from the build context.

### New files

| File | Purpose |
|---|---|
| `docker/Dockerfile` | Multi-stage container image definition |
| `docker/entrypoint.sh` | Runtime mode dispatcher |
| `docker/docker-compose.yml` | Local development / integration testing |
| `docker/kubernetes/bamboo-mcp.yaml` | Kubernetes Deployment + Service + PVCs |
| `docker/docs/docker.md` | Usage documentation |
| `.dockerignore` | Build context filter |


---

## 2026-04-07

### Added

- **ASCII charts in the Textual TUI.** Pilot/Harvester answers now
  automatically display two chart panels below the text response.

  - **Status bar** (`pilot chart`) — horizontal bar chart of worker counts
    per status (running, submitted, finished, failed, etc.) with the time
    window and grand total. Rendered from the existing
    `panda_harvester_workers` snapshot evidence; no extra API call.

  - **Timeseries** (`pilot timeseries (<status>)`) — vertical bar chart
    showing Harvester worker update events per bucket over the query time
    window. Status and time window are extracted from the user's question
    automatically. Bars fill the full terminal width. Rendered via the new
    `panda_harvester_timeseries` tool (see below).

  > **Note on timeseries counts:** the timeseries shows *update events per
  > bucket* — workers that reported a status change in that window — not the
  > total number of active pilots. The OpenSearch index is a stream of change
  > events, not a snapshot. The status bar remains the authoritative source
  > for total pilot counts.

  Both charts are suppressed when only one status is present. The `/chart`
  slash command re-displays the most recent chart after scrolling. Charts
  degrade gracefully when OpenSearch is unavailable.

- **`panda_harvester_timeseries` MCP tool** (`atlas.harvester_timeseries`).
  Queries the OpenSearch `atlas_harvesterworkers-*` index for per-bucket
  worker counts. Bucket interval is derived automatically from the query
  window (≤30 min → `1m`, ≤3 h → `5m`, ≤12 h → `15m`, else `1h`).
  Requires `ASKPANDA_OPENSEARCH` and CERN network access (VPN or lxplus).
  Gracefully skipped when `opensearch-py`/`opensearch-dsl` are not installed.

- **New slash command `/chart`** — re-displays the ASCII pilot chart for
  the last Harvester query.

- **`docs/harvester-workers.md`** — reference documentation for the
  `panda_harvester_workers` tool.

- **New environment variables** for OpenSearch connectivity:

  | Variable | Purpose |
  |---|---|
  | `ASKPANDA_OPENSEARCH` | Password for OpenSearch HTTP Basic auth. Required for timeseries charts. |
  | `ASKPANDA_OPENSEARCH_HOST` | OpenSearch cluster URL (default: `https://os-atlas.cern.ch/os`) |
  | `ASKPANDA_OPENSEARCH_USER` | HTTP auth username (default: `pilot-monitor-agent`) |
  | `ASKPANDA_OPENSEARCH_CA` | Path to CA bundle (default: `/etc/pki/tls/certs/CERN-bundle.pem`) |
  | `ASKPANDA_OPENSEARCH_VERIFY_CERTS` | Set to `false` to disable TLS verification for local dev |

### Fixed

- **Linux TUI banner** — the banner panel was collapsing to zero height on
  Linux before the first render due to `height: auto` not measuring multiline
  content correctly before layout. Fixed with `height: 9; min-height: 9`.

### New files

| File | Location |
|---|---|
| `chart_utils.py` | `packages/askpanda_atlas/askpanda_atlas/` |
| `harvester_timeseries_impl.py` | `packages/askpanda_atlas/askpanda_atlas/` |
| `harvester_timeseries.py` | `packages/askpanda_atlas/askpanda_atlas/` |
| `test_chart_utils.py` | `packages/askpanda_atlas/tests/` |
| `test_harvester_timeseries.py` | `packages/askpanda_atlas/tests/` |
| `harvester-workers.md` | `docs/` |

### Dependencies

```bash
pip install opensearch-py opensearch-dsl
```

Required for timeseries charts. Optional — the TUI starts normally without
them and timeseries charts are silently skipped.

### Configuration

Add to `packages/askpanda_atlas/pyproject.toml`:

```toml
[project.entry-points."bamboo.tools"]
"atlas.harvester_timeseries" = "askpanda_atlas.harvester_timeseries:panda_harvester_timeseries_tool"
```

## Fix for read-only DuckDB connections

`cric_query_impl.py` and `jobs_query_impl.py` now open on-disk DuckDB files with `read_only=True` (via `database=` keyword), allowing the MCP query tools to coexist with the agent writer processes without triggering DuckDB's single-writer lock. In-memory connections (`:memory:`) remain read-write for tests. Three call sites updated: `_execute_query` in both files, `_probe_table_names` in `cric_query_impl`. Docstrings updated to document the policy. Flake8 clean.
