---
name: agentfield-use
description: Discover and call agents already running on a local AgentField control plane. Use when the user asks to use, call, query, run, or delegate work to an installed AgentField agent (swe-planner, pr-af, sec-af, …), to list what agents or reasoners are available, or to check on an execution. Not for building new agents — that is the agentfield skill.
---

# Using AgentField agents

A machine with AgentField has a **control plane** (default `http://localhost:8080`,
override via `AGENTFIELD_SERVER`) and **agent nodes** installed under
`~/.agentfield`. Each node exposes **reasoners** — typed functions you call over
HTTP. You never talk to an agent's own port: every call goes through the control
plane, which routes it, records the workflow, and returns the result.

In local mode there is no auth. If the server has an API key configured, send it
as `X-API-Key: <key>` on every request.

## The flow

1. Health-check the control plane.
2. Discover what agents and reasoners exist.
3. Execute — async for anything nontrivial. Fire independent calls concurrently.
4. Poll (or stream) until the execution finishes — and watch for wedged runs.

## 1. Is the control plane up?

```bash
curl -s http://localhost:8080/health
```

Healthy: `200` with `{"status":"healthy", ...}`. Connection refused means no
control plane is running — the user can open the AgentField desktop app, or you
can start one in the background (`af server` blocks, so background it and poll
`/health` until healthy).

## 2. Discover agents and reasoners

```bash
curl -s "http://localhost:8080/api/v1/discovery/capabilities?include_input_schema=true"
```

This is the durable discovery endpoint. Reasoner names are `.reasoners[].id`
(NOT `.name`), and `include_input_schema=true` adds each reasoner's JSON input
schema — read it before calling so your `input` matches.

Don't assume `jq` exists (fresh Windows boxes lack it) — parse with what's
installed, e.g.:

```bash
curl -s "http://localhost:8080/api/v1/discovery/capabilities?include_input_schema=true" -o caps.json
python -c "
import json
for c in json.load(open('caps.json'))['capabilities'] or []:  # null when no agents registered
    print(c['agent_id'], c.get('health_status'), [r['id'] for r in c.get('reasoners',[])])"
```

Three gotchas:

- The response's `invocation_target` field uses a **colon** (`agent:reasoner`).
  The execute URL uses a **dot**. Build the target yourself: `<agent_id>.<reasoner_id>`.
- Discovery lists **every registered agent, including dead ones** — check
  `health_status` and only dispatch to `"active"` agents. Dispatching to an
  `inactive`/`unknown` agent queues work that never runs.
- Installed-but-never-started agents may not appear at all. The local registry
  is the source of truth for what's installed: `af list`, start with
  `af run <name>` (it detaches; the agent keeps running after the CLI exits).

### Too many reasoners to scan? Search, don't dump

When a box has more than ~20 reasoners installed, ranked search beats reading
the whole capabilities payload into context:

```bash
af agent search "review a pull request"     # BM25-ranked; --agent <id>, --limit N (max 50)
# or: curl -s "http://localhost:8080/api/v1/agentic/reasoners?q=review+pull+request"
```

Each hit carries `reasoner_id`, `agent_id`, `invocation_target`, `tags`,
`score`, and `agent_health` — everything you need to dispatch with no second
lookup. Build the execute target straight from `invocation_target` (colon → dot)
and only dispatch to hits whose `agent_health` is `"active"`.

## 3. Call a reasoner

Input kwargs are ALWAYS nested under `"input"` — never raw at the top level.

**Async — the default for real work.** Returns `202` immediately:

```bash
curl -s -X POST http://localhost:8080/api/v1/execute/async/swe-planner.plan \
  -H 'Content-Type: application/json' \
  -d '{"input": {"task": "add rate limiting to the API"}}'
# -> {"execution_id":"...", "run_id":"...", "status":"queued", ...}
```

**Sync — only for calls that finish fast** (hard 90s timeout, response carries
`result` directly):

```bash
curl -s -X POST http://localhost:8080/api/v1/execute/swe-planner.plan \
  -H 'Content-Type: application/json' \
  -d '{"input": {"task": "..."}}'
```

### Concurrency — use it

Async dispatch is cheap: fire all independent calls up front, then poll them
together. Do NOT serialize multi-agent work — the whole point of the control
plane is managing many agents at once. What to know:

- Concurrent calls to the **same reasoner** are safe when the agent is (e.g.
  pr-af isolates concurrent reviews per PR). If an agent's docs don't say it's
  parallel-safe, assume same-target calls may contend on shared state and
  stagger them; different agents never contend.
- Each call fans out inside the agent (one review ≈ dozens of sub-executions,
  several LLM CLI processes). 3–4 heavy runs per node is a sensible ceiling
  unless the agent documents otherwise.
- Save every `execution_id` you dispatch. Group related calls with an
  `X-Session-ID` header so they're queryable as one batch later.

**Check the load before piling on.** Every `af agent` / agentic response carries
`meta.load`: `{running_agents, total_agents, active_executions, cpu_cores,
recommended_max_concurrent}` (the recommendation is CPU-based). Read it before
launching more heavy runs — if `active_executions >= recommended_max_concurrent`,
finish or await in-flight work first rather than starting more, and tell the
user you're throttling to avoid overloading the machine.

## 4. Get the result

**What's in flight right now** — no IDs needed (also answers "how many agents
are running something"):

```bash
curl -s http://localhost:8080/api/v1/executions/active
# {"count":2,"runs":[{"run_id":"...","target":"pr-af-go.review","root_status":"running",
#   "active_executions":4,"total_executions":27,"started_at":"...","latest_activity":"..."}]}
```

Filters: `?agent_id=<node>`, `?session_id=<your session>`. CLI equivalent: `af ps`.

**One execution** — poll until `status` is terminal (`succeeded` / `failed`,
also `cancelled` / `timeout`):

```bash
curl -s http://localhost:8080/api/v1/executions/<execution_id>
```

Long-running agents can take tens of minutes — poll with backoff (start ~5s,
settle at ~30s) and tell the user what is in flight. For live progress, stream
Server-Sent Events from `GET /api/v1/executions/<execution_id>/events`.

**Several at once:** `POST /api/v1/executions/batch-status` with
`{"execution_ids": [...]}`. Terminal entries embed the FULL result payload —
responses can be large (100KB+), so write to a file and parse from there; never
pass the response through a command-line argument (Windows caps argv ~32KB).

There is **no** `GET /api/v1/executions` list endpoint — use `/executions/active`
for in-flight work and `POST /api/v1/agentic/query` (body:
`{"resource":"runs","filters":{"status":"..."},"limit":20}`) for history.

### Wedge protocol — "running" is not proof of progress

An execution can report `running` indefinitely after its agent silently dies or
deadlocks. Treat a run as suspect when `/executions/active` shows
`latest_activity` **more than ~10 minutes old** while `active_executions > 0`
AND `af logs <agent>` shows nothing new for that run. (A quiet log alone is not
proof — one long LLM completion can be minutes of legitimate silence.) Then:

1. Cancel the WHOLE run, not just the root:
   `POST /api/v1/workflows/<run_id>/cancel-tree` (bottom-up, cancels children
   too). Plain `/executions/<id>/cancel` cancels ONLY that execution — children
   keep "running" and must be cancelled individually.
2. Restart the agent if it's wedged: `af stop <name> && af run <name>`.
3. Re-submit the work.

## Sessions and multi-call work

- `X-Session-ID: <your-id>` on execute requests groups multi-turn work; the
  control plane forwards it to the agent and scopes session memory by it.
- Reuse `X-Run-ID` across several execute calls to group them into one
  workflow; each response also returns its `run_id`.

Agents share state through control-plane memory if you need to pass artifacts
around: `POST /api/v1/memory/set` with `{"key": ..., "data": <any>, "scope":
"global"}` and `POST /api/v1/memory/get` with `{"key": ...}` (non-global scopes
resolve from the `X-Workflow-ID` / `X-Session-ID` / `X-Actor-ID` headers).

## When things fail

| Symptom | Meaning | Fix |
|---|---|---|
| connection refused on :8080 | control plane not running | desktop app, or background `af server` and poll `/health` |
| agent `inactive` in discovery / missing | node installed but not running (or not installed) | `af list`, then `af run <name>` — or `af install <source>` |
| `missing required environment variables: X` from `af run` | required key not configured | `af secrets set X` (value via stdin/arg; `--node <name>` for node-scoped) — or desktop app → Agents → Keys |
| HTTP 502 with `error_message` | the agent itself errored | read `af logs <name>`, fix, retry |
| execution `running` but latest_activity stale & logs quiet | wedged run | wedge protocol above: cancel-tree → restart agent → re-submit |
| result claims success with zero findings/output on nontrivial input | possible silent tool failure inside the agent | check `af logs <name>` for that run before trusting it |

## Local ops cheat sheet (af CLI)

```bash
af list                    # installed agents + status
af ls [query]              # search reasoners across running agents (NOT the install registry)
af ps                      # in-flight runs across all agents (af ps --agent <name>)
af run <name>              # start (detached); af stop <name>
af logs <name>             # agent logs (-f follows; no per-run filter — grep by run_id)
af secrets set KEY         # store an API key (encrypted; prompts for value)
af secrets ls              # what's configured (values never shown)
af install <git-url>       # install a new agent node
```

## Audit trail

Every execution is recorded. When provenance matters (or the user asks "what
did the agents actually do"), fetch the verifiable-credential chain for a
workflow: `GET /api/v1/did/workflow/<run_id>/vc-chain` (available when DID/VC
is enabled), and verify offline with `af verify audit.json`.

## Hard rules

- Every call goes through the control plane — never POST to an agent's own port.
- Kwargs live under `"input"`. Empty input is `{"input": {}}`.
- Async + poll for anything that might exceed a few seconds; sync is for quick
  lookups only. Independent async calls go out together, not one at a time.
- Only dispatch to agents whose discovery `health_status` is `"active"`.
- Don't guess endpoints. The surface above is the contract; if something is
  missing, ask `GET /api/v1/agentic/discover?q=<keyword>` before inventing a route.
- Building or modifying an agent (new reasoners, scaffolds, deploys) is the
  **agentfield** skill's job — switch to it for that.
