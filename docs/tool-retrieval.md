# Tool retrieval

The LLM planner used to be shown every tool on every question: 22 definitions,
about 29 kB of JSON, of which `opensearch_promptlog_query` alone is a third.
That cost was paid per call regardless of what was asked, and selection accuracy
degrades as a flat list grows.

Bamboo now scores the catalog against the question and shows the planner only
the tools it plausibly needs. This page covers how to configure it, how to see
what it chose, and what the measurements say.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `BAMBOO_TOOL_RETRIEVAL` | `lexical` | `off`, `lexical`, `embedding`, `hybrid` |
| `BAMBOO_TOOL_RETRIEVAL_K` | `10` | Tool budget, **pinned tools included** |
| `BAMBOO_TOOL_RETRIEVAL_MIN_CATALOG` | `12` | Catalogs this size or smaller pass through unnarrowed |
| `BAMBOO_TOOL_RETRIEVAL_LOG` | unset | Truthy raises the per-question selection line from DEBUG to INFO |
| `BAMBOO_TOOL_RETRIEVAL_RRF_K` | `10` | Rank-fusion constant, `hybrid` only |

All are read at call time, not at import, so changing one takes effect without
a restart in any process that re-reads the environment. All fail open to the
default with a warning logged once per distinct bad value: a configuration typo
must not become a server that errors on every question, particularly when the
fallback is simply the behaviour that shipped for the last two years.

**`BAMBOO_TOOL_RETRIEVAL=off` is the kill switch.** It restores the previous
planner exactly — same catalog, same prompt, same routing guidance.

---

## Seeing what was selected

Every decision is logged, and mirrored into the trace stream as an
`EVENT_RETRIEVAL` record so selections appear in `/tracing`.

```
BAMBOO_TOOL_RETRIEVAL_LOG=1 python -m bamboo.server
```

```
tool retrieval: backend=lexical k=10 kept 10/22
[panda_doc_bm25(pinned), panda_doc_search(pinned), panda_log_analysis=4.87,
 atlas.pilot_source_analysis=2.92, atlas.harvester_timeseries=2.10,
 panda_job_status=1.85, atlas.core_dump_analysis=1.61, panda_task_status=1.52,
 atlas.job_stats=1.42, panda_jobs_query=0.92]
withheld [bamboo_llm_probe=0.00, ..., panda_harvester_workers=0.83, ...]
```

Withheld tools carry their scores too. "It scored 0.83 and missed the cut" and
"it scored nothing at all" need different fixes — a budget change versus a
description that does not say what the tool is for — and a list of survivors
alone cannot tell them apart.

Without the flag the same line is logged at DEBUG. The flag exists because
turning on DEBUG for the whole process to read one line buries it in HTTP and
SDK chatter.

### When retrieval did not run

The line says so, and says why. The reasons are deliberately distinct:

| Reason | Meaning |
|---|---|
| `disabled` | `BAMBOO_TOOL_RETRIEVAL=off`, or an unrecognised value |
| `no_question` | No question to score against |
| `catalog_small` | Catalog no larger than the budget; narrowing could only lose a tool |
| `backend_unavailable` | `embedding`/`hybrid` selected but no model is installed or cached |
| `backend_error` | The retriever raised, or returned a name absent from the catalog |
| `empty_result` | Every tool scored zero |

A retrieval layer that has quietly stopped retrieving presents as nothing worse
than a larger prompt, so none of these is silent. `backend_unavailable` and
`backend_error` are separated because "install `requirements-rag.txt`" and "the
retriever is broken" call for different responses.

---

## What gets scored

For each tool: its name split on `.` and `_` (so "harvester workers" in a
question matches `panda_harvester_workers`), the first 800 characters of its
description, and its top-level parameter *names*.

Parameter descriptions are not indexed — names like `site`, `queue` and `job_id`
are short and discriminating, while their prose would multiply the indexed text
for little added signal. The description cap matters more than it looks:
`opensearch_promptlog_query`'s description is 8,151 characters, and indexed
whole it would dominate the term statistics of a 22-document corpus and pull
unrelated questions towards itself.

### Pinned tools

`panda_doc_search` and `panda_doc_bm25` are never scored and never withheld.
They back the "for ALL other questions" route, which is the planner's only
fallback. They count against `k`, so `k=10` means two pins plus eight retrieved
— a pin outside the budget would make `k` a budget in name only.

The pins are load-bearing, not caution: without them recall at `k=10` drops from
0.992 to 0.892, because the fallback tools are individually weak lexical matches
for the general-knowledge questions they exist to answer.

### Routing guidance

The planner prompt pairs a hard rule — *"Only propose tools that appear in the
provided tool catalog"* — with guidance naming specific tools. Narrowing the
catalog without narrowing the guidance makes the prompt contradict itself, and
the planner has been observed resolving that by discarding the guidance
wholesale.

So guidance clauses are filtered in lockstep: a clause is emitted only when
every tool it names survived. A clause naming several tools is therefore also a
co-occurrence unit — the site-health clause keeps `panda_harvester_workers` and
`panda_jobs_query` together rather than hoping both survive independently. See
`_ATLAS_ROUTING_RULES` in `bamboo/tools/planner.py`.

Guidance is filtered **only when retrieval actually narrowed the catalog**, not
whenever a tool happens to be absent. A host without DuckDB loses
`panda_jobs_query` for unrelated reasons, and withholding the site-health clause
there is a separate decision from this one.

---

## Scope

Only the planner's catalog is narrowed. These paths are untouched:

- `bamboo_answer`'s deterministic fast path, which never consults the catalog
- `bamboo_executor`'s in-process tool resolution
- the MCP `tools/list` surface, which a directly-attached client still sees in
  full — it is a session-level RPC made before any question exists, so there is
  nothing to condition on

---

## Measurements

`scripts/eval_tool_retrieval.py` evaluates any backend against
`tests/data/tool_selection_corpus.json`: 120 labelled questions, 59 of them
flagged `hard` because they pair a question with a tool it is easy to confuse
for a neighbour.

```bash
python scripts/eval_tool_retrieval.py --retriever lexical --k 10 --k 12
python scripts/eval_tool_retrieval.py --json > baseline.json
```

Measured on a 22-tool ATLAS catalog (fingerprint in the report header):

| backend | k | recall | hard | guidance | payload |
|---|---|---|---|---|---|
| null (no retrieval) | — | 1.000 | 1.000 | 1.000 | 1.000 |
| **lexical** | **10** | **0.992** | **0.983** | **1.000** | **0.389** |
| lexical | 12 | 0.992 | 0.983 | 1.000 | 0.453 |
| embedding | 10 | 0.975 | 0.949 | 0.971 | 0.442 |
| embedding | 12 | 0.992 | 0.983 | 0.990 | 0.553 |
| hybrid | 10 | 0.992 | 0.983 | 1.000 | 0.395 |

**Always quote the catalog fingerprint with a number.** The same command on two
hosts gave 0.983 and 0.992 because their plugin descriptions differed; the
fingerprint is what distinguishes that from a change in the retriever.

### Why lexical rather than embeddings

The catalog is 22 short strings of rare domain jargon — `pilottiming`, `HS06`,
`cmtconfig`, `netzone`, `queuedata`. That is close to the best case for exact
term matching and close to the worst for a small general-purpose embedding
model, whose weakest tokens are precisely those.

The measurement agreed. The embedding backend's three failures at `k=10` were
`panda_queue_info` for *"Is BNL accepting MCORE jobs?"*, `atlas.job_stats` for
*"maximum queue time for failed jobs"*, and `code_query` for *"Show me the retry
logic in `pilot/util/https.py`"* — a literal source path, the most lexically
unambiguous token in the corpus. It also costs more payload at every `k`, plus a
one-to-three second model load and a dependency on `requirements-rag.txt`.

Lexical costs nothing, loads nothing, and runs in microseconds.

### Why hybrid is not the default

Hybrid fuses the two backends by reciprocal rank, and on paper should rescue the
one case lexical misses — *"Which questions received the lowest ratings last
month?"*, whose wording shares no term with `opensearch_promptlog_query`'s
description. The embedding backend alone does solve it. Hybrid does not.

That is arithmetic, not luck. With `RRF_K` and *N* scorable tools, a tool ranked
first by one backend scores `1/(RRF_K+1)`, and a tool ranked *last by both*
scores `2/(RRF_K+N)`. At the literature's default of 60 with N=20 those are
0.0164 and 0.0250, so **every** tool both backends rank beats **every** tool only
one ranks, whatever the ranks. Fusion degenerates into "the intersection, then
the leftovers", and the single-source find hybrid exists to rescue is exactly
what it structurally cannot rescue.

`RRF_K=60` assumes rankings over thousands of documents. A sweep confirmed the
prediction exactly:

| `RRF_K` | 3 | 5 | 10 | 20 | 60 |
|---|---|---|---|---|---|
| recall@10 | 1.000 | 1.000 | 0.992 | 0.992 | 0.992 |
| payload | 0.404 | 0.399 | 0.394 | 0.395 | 0.395 |

The default is now 5. **Treat that 1.000 with suspicion rather than pride.**
`RRF_K` was chosen on the same 120 cases it is reported against, the gain is a
single case, and it is the case the sweep went looking for. That is a fit, not a
validated improvement.

Hybrid is still not the default, for a reason unrelated to its score. It makes
an embedding model a hard dependency of the planner path, and when the model is
missing the fallback is the *whole* catalogue — a host without the model loses
100% of the benefit, where lexical would have lost none. One case in 120 does not
buy that.

### The case hybrid was chasing, and what actually fixed it

The question was *"Which questions received the lowest ratings last month?"*,
which needs `opensearch_promptlog_query`. Lexical missed it, and the cause was
not a semantic gap — it was this page's own 800-character indexing window. That
tool's description is 8,151 characters; the word "question" first appeared at
character 917 and "rating" at 6,903. BM25 never saw the words it needed.

Raising the cap does not fix it. At 9,000 characters the tool is retrieved, but
its 8 kB of text then crowds `panda_jobs_query` out of two site-health questions
— the same attention-dilution problem this whole feature exists to solve,
reproducing inside the ranker.

What fixed it was rewriting the first paragraph of the description so the
vocabulary a user would actually use appears in the indexed window. Lexical went
to **1.000 recall with no model, no tuning and no new dependency**.

**The lesson generalises: a tool's opening sentences are its retrieval surface.**
A description written for a reader who already knows what the tool is for will
underperform one that states the words a user would type. This is also free —
the same sentences are what the planner reads first.

---

## Extending it

### Adding a tool

Nothing is required — a new tool is scored like any other. But add questions for
it to `tests/data/tool_selection_corpus.json`, or
`tests/test_tool_selection_corpus.py` will fail: every catalog tool needs at
least three labelled cases or a written exemption. The exemptions are for tools
the interface invokes rather than a user asking for them in words
(`bamboo_llm_probe`, `atlas.ui_manifest`).

If a tool is retrieved poorly, the fix is almost always its description — see
the worked example above, where a description change did what an embedding model
could not. Retrieval only matches words inside the first 800 characters, so check
there first:

```python
from bamboo.tools.tool_retrieval import index_terms
print(" ".join(index_terms(entry)))
```

If the words a user would type are not in that output, no retriever will find
the tool.

### Adding a backend

Implement `ToolRetriever` — one `retrieve(question, catalog, k)` method — and
register it in `_build_retriever()` and in the eval script's `RETRIEVERS`. A
`score()` method is optional but recommended: it is what populates the per-tool
scores in the debug line.

Measure it before proposing it. The harness exists so that a backend choice is
settled by recall rather than by argument, and the null baseline is there so a
candidate has something to be worse than.
