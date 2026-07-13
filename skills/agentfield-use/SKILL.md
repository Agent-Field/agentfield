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
3. Execute — async for anything nontrivial.
4. Poll (or stream) until the execution finishes.

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
curl -s "http://localhost:8080/api/v1/discovery/capabilities?include_input_schema=true" \
  | jq '.capabilities[] | {agent: .agent_id, health: .health_status, reasoners: [.reasoners[].id]}'
```

This is the durable discovery endpoint. Reasoner names are `.reasoners[].id`
(NOT `.name`), and `include_input_schema=true` adds each reasoner's JSON input
schema — read it before calling so your `input` matches.

Two gotchas:

- The response's `invocation_target` field uses a **colon** (`agent:reasoner`).
  The execute URL uses a **dot**. Build the target yourself: `<agent_id>.<reasoner_id>`.
- Discovery only lists agents that are **running and registered**. Installed but
  stopped agents live in the local registry — check with `af ls`, start with
  `af run <name>` (it detaches; the agent keeps running after the CLI exits).

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

## 4. Get the result

Poll the execution until `status` is terminal (`succeeded` / `failed`, also
`cancelled` / `timeout`):

```bash
curl -s http://localhost:8080/api/v1/executions/<execution_id> \
  | jq '{status, result, error}'
```

Long-running agents can take minutes — poll with backoff (2s → 5s → 10s) and
tell the user what is in flight. For live progress, stream Server-Sent Events
from `GET /api/v1/executions/<execution_id>/events`. To check several at once:
`POST /api/v1/executions/batch-status` with `{"execution_ids": [...]}`.

There is **no** `GET /api/v1/executions` list endpoint — do not invent one.
Cancel with `POST /api/v1/executions/<id>/cancel`.

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
| agent missing from discovery | node installed but not running (or not installed) | `af ls`, then `af run <name>` — or `af install <source>` |
| `missing required environment variables: X` from `af run` | required key not configured | `af secrets set X` (value via stdin/arg; `--node <name>` for node-scoped) — or desktop app → Agents → Keys |
| HTTP 502 with `error_message` | the agent itself errored | read `af logs <name>`, fix, retry |
| execution stuck in `queued`/`running` | agent wedged or overloaded | `af stop <name> && af run <name>`, then re-submit |

## Local ops cheat sheet (af CLI)

```bash
af ls                      # installed agents + status
af run <name>              # start (detached); af stop <name>
af logs <name>             # agent logs
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
  lookups only.
- Don't guess endpoints. The surface above is the contract; if something is
  missing, say so instead of inventing a route.
- Building or modifying an agent (new reasoners, scaffolds, deploys) is the
  **agentfield** skill's job — switch to it for that.
