# AI Agent

The Bamboo AI Agent is a multi-step reasoning interface for complex queries
that go beyond what the single-pass planner/executor pipeline handles well.
Instead of producing one plan and executing it once, the agent works
iteratively: it calls tools one at a time, evaluates whether the evidence so
far is sufficient, and keeps going until it is confident enough to synthesise
a final answer.

---

## When to use it

The standard TUI and Streamlit interfaces handle the majority of questions
well. Use the agent when the question requires correlating information from
multiple sources — for example, identifying high-failure-rate sites *and*
checking their current queue status, or asking for a root-cause analysis that
spans job logs, pilot errors, and site configuration.

For simple factual lookups the agent adds unnecessary latency (multiple round
trips to the LLM). Prefer the TUI or Streamlit for those.

### Not the job agent

Bamboo ships a second agent, and the two are easy to confuse because both
drive MCP tools against a running server from the command line.

| | `BambooAgent` (this document) | `interfaces.agent.job_agent` |
|---|---|---|
| Question | anything | "why did job *N* fail" |
| Loop | the LLM picks the next tool each step | fixed; every decision is a server-side rule |
| LLM calls | three per step, plus synthesis | one, at the end — or none |
| Tools | whatever `tools/list` advertises | the five `atlas.log.*` primitives |
| Output | prose | prose **and** structured evidence |
| Reproducible | no | yes, with `--no-synthesis` |

Use this one for open questions that span several data sources.  Use the job
agent when the question is a job ID, when the answer has to be the same twice
running, or when something downstream needs the evidence rather than the
prose.  `BambooAgent` is deliberately not the engine underneath it: its
observations come from text content blocks truncated at 6 000 characters,
which cannot carry the primitives' structured payloads intact.  See
[`docs/code-mode.md`](code-mode.md#the-reference-agent).

---

## Architecture

The agent implements a **Reason → Act → Observe → Evaluate** loop:

1. **Reason** — the LLM is given the question, the list of available tools,
   and all evidence gathered so far.  It returns a structured JSON object
   naming the next tool to call (or signalling that synthesis can begin).
2. **Act** — the chosen tool is called via the MCP HTTP transport.
3. **Observe** — the tool result is recorded in agent memory.
4. **Evaluate** — a second LLM call judges whether the accumulated evidence
   is sufficient.  If confidence is above the threshold the loop exits;
   otherwise it continues to the next step.
5. **Synthesise** — a final LLM call produces the natural-language answer
   from all accumulated observations.

All LLM calls go through `bamboo_llm_answer` on the running MCP server, so
the agent inherits the server's configured LLM provider and model without any
additional setup.

---

## Prerequisites

A running Bamboo HTTP server is required:

```bash
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
python -m bamboo.server_http --host localhost --port 8000 &
```

See [`docs/http-server.md`](http-server.md) for full server setup details.

---

## Running the agent

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --question "Your question here"
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--question` / `-q` | — | Question to answer (single-shot mode) |
| `--interactive` / `-i` | off | Start a REPL session |
| `--transport` | `http` | `http` or `stdio` |
| `--http-url` | `http://localhost:8000/mcp` | MCP server endpoint |
| `--max-steps` | 6 | Maximum reasoning iterations before forced synthesis |
| `--confidence` | 0.80 | Evaluator confidence threshold in [0, 1] |
| `--max-tokens` | 2048 | Token budget for the final synthesis call |
| `--verbose` / `-v` | off | Print the full reasoning trace |
| `--output-json` | off | Emit `AgentResult` as JSON (useful for scripting) |
| `--token` | — | Bearer token for authenticated servers |

All options can also be set via environment variables — see
[Environment variables](#environment-variables) below.

---

## Testing

### Simple query

A single-hop factual question.  Good for verifying the agent connects,
discovers tools, and produces an answer:

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --question "What is PanDA?" \
    --verbose
```

Expected behaviour: one or two tool calls (`panda_doc_bm25` or
`bamboo_llm_answer`), evaluator satisfied quickly, clean synthesised answer.

---

### Multi-hop query

A question that naturally requires correlating two data sources — the kind
the agent is designed for:

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --question "Which ATLAS sites had the highest pilot failure rate in the last 24 hours, and are those sites currently showing any queue or resource issues in CRIC?" \
    --max-steps 8 \
    --confidence 0.75 \
    --verbose
```

Expected behaviour: step 1 calls a Harvester timeseries or jobs tool for
failure rates; step 2 calls `atlas.cric_query` to check queue/site status for
those specific sites; the evaluator is satisfied once both datasets are in
context; synthesis correlates the two.

`--confidence 0.75` (slightly below the default 0.80) gives the agent a bit
more room to proceed when partial evidence overlaps between steps.

---

### Interactive REPL

For exploratory sessions where you want to ask several related questions
without restarting the agent:

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --interactive \
    --verbose
```

Type `exit` or press `Ctrl+D` to quit.  Each question opens a fresh
`AgentMemory`; there is no cross-question memory within the REPL.

---

### JSON output (scripting / notebooks)

Dumps the full `AgentResult` — answer, step trace, confidence, tools used —
as JSON to stdout:

```bash
python scripts/bamboo_agent.py \
    --transport http \
    --http-url http://localhost:8000/mcp \
    --question "What is the average job stagein time at BNL this week?" \
    --output-json
```

---

## Reading the verbose trace

With `--verbose` each step is printed as it completes:

```
─── Step 1/6 ───
  Thought: <LLM's rationale for the tool choice>
  Action:  <tool_name>(<arguments>)
  Obs:     <first 300 chars of the tool result>
  Eval:    sufficient=False  confidence=0.80  missing='...'
```

After all steps, the full trace is reprinted in a box followed by the final
answer and a one-line metadata summary:

```
(steps=3, confidence=1.00, tools=[panda_doc_bm25, bamboo_llm_answer])
```

`truncated` appears in the metadata line when the agent hit `--max-steps`
before the evaluator was satisfied.  In that case the synthesis prompt
instructs the LLM to state clearly what could not be determined.

---

## Tuning

**`--max-steps`** — increase for questions that need more than two or three
tool calls (e.g. a full RCA spanning logs, pilot errors, and site config).
Decreasing it saves latency for questions where one or two calls suffice.

**`--confidence`** — lower values (e.g. 0.70) allow the agent to proceed
with less certainty about sufficiency; useful when evidence sources overlap
and the evaluator is conservative.  Higher values (e.g. 0.90) require more
complete evidence before synthesis.

Both defaults are conservative and suitable for most queries.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `BAMBOO_AGENT_MAX_STEPS` | `6` | Maximum reasoning iterations |
| `BAMBOO_AGENT_CONFIDENCE` | `0.80` | Evaluator confidence threshold |
| `BAMBOO_AGENT_MAX_TOKENS` | `2048` | Synthesis token budget |
| `BAMBOO_MCP_HTTP_URL` | `http://localhost:8000/mcp` | Default HTTP endpoint |
| `BAMBOO_MCP_TOKEN` | — | Bearer token for authenticated servers |

---

## Limitations (current)

- **Sequential steps only** — tool calls are executed one at a time.  Parallel
  dispatch is planned for a future iteration.
- **No cross-question memory in REPL** — each question starts a fresh
  `AgentMemory`.  The REPL shares the MCP connection but not conversation state.
- **Single LLM profile** — all calls use the server's configured default LLM.
  Separate fast/reasoning profiles are not yet exposed by `bamboo_llm_answer`.
- **No prompt logging** — agent runs are not yet logged to OpenSearch.  The
  logging call is stubbed (`# AGENT_LOG`) pending index template provisioning.

---

## Files

| Path | Description |
|---|---|
| `interfaces/agent/agent.py` | Core agent: `BambooAgent`, `AgentMemory`, `AgentStep`, `AgentResult` |
| `scripts/bamboo_agent.py` | CLI entry point |
| `tests/test_agent.py` | Unit tests (36 tests) |
