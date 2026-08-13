---
name: agentfield-use
version: 0.6.0
description: "Discover and call agents already running on a local or cloud AgentField control plane. Use when the user asks to use, call, query, run, or delegate work to an installed AgentField agent (swe-planner, pr-af, sec-af, …), to list what agents or reasoners are available, or to check on an execution. Resolves the right control plane (desktop-configured cloud first), fetches the reasoner's exact contract before dispatching, and only calls entry-point reasoners. Not for building new agents — that is the agentfield skill."
---

# Using AgentField agents

A machine with AgentField has one or more **control planes** — a local one
(default `http://localhost:8080`) and possibly a **cloud deployment**
configured in AgentField Desktop — plus **agent nodes** installed under
`~/.agentfield`. Each node exposes **reasoners** — typed functions you call
over HTTP. You never talk to an agent's own port: every call goes through the
control plane, which routes it, records the workflow, and returns the result.

**Resolve which control plane you are targeting before anything else (§0).**
The local and cloud fleets are disjoint: different agents, different versions,
different filesystems, different run history. Nothing ever falls back from one
to the other on its own.

A local server in local mode has no auth. A cloud deployment (and any server
with an API key configured) requires `X-API-Key: <key>` on every request.

## 0. Resolve the server first (local vs cloud)

Resolution order — stop at the first match:

1. **Explicit wins.** The user named a server, or `AGENTFIELD_SERVER` is set in
   the environment → use that.
2. **Read the desktop cloud config.** Check every path that applies to this
   machine — a file that exists but declares no enabled cloud does NOT end
   the search:
   - macOS: `~/Library/Application Support/agentfield-desktop/settings.json`
   - Windows: `%APPDATA%/agentfield-desktop/settings.json`
   - Linux: `~/.config/agentfield-desktop/settings.json`
   - WSL (detect: `grep -qi microsoft /proc/version`): the Linux path above
     first, then the Windows side, where the desktop app usually lives:
     `/mnt/c/Users/*/AppData/Roaming/agentfield-desktop/settings.json`.
     A Linux-side file with no `cloud` key shadowing a Windows file that
     holds the real cloud config is the common split-brain — the enabled
     cloud wins, whichever side declares it.
   The first file declaring `cloud.enabled: true` with a non-empty
   `cloud.serverUrl` makes the cloud the target: strip any trailing slash
   from the URL and take `cloud.apiKey` as the key. Health-check it
   (`GET <url>/health` with `X-API-Key`).
   - Healthy → use the cloud for everything below.
   - Unreachable → **stop and tell the user their cloud control plane is
     configured but not responding.** Do NOT silently fall back to local:
     work dispatched there lands on a different fleet with different
     filesystems, which is worse than no dispatch.
3. **Otherwise use local:** `http://localhost:8080`.

Then pass the target **explicitly on every call**: `af --server <url> -k <key>`
(every `af` command accepts `-s/--server` and `-k/--api-key`), or the URL plus
`-H 'X-API-Key: <key>'` for curl. Do not export `AGENTFIELD_SERVER` yourself to
switch targets — a global default leaks into other processes and outlives the
task; explicit per-call flags are the contract. If you'd rather not pass `-k`
each time, `af auth login --server <url>` stores a key per server in
`~/.agentfield/credentials.json`.

## MCP (zero-setup)

The control plane serves a built-in **MCP server at `<server>/mcp`** (default
`http://localhost:8080/mcp`) — same port, no extra process, on by default. If
your harness speaks MCP, this is the fastest way in.

Claude Code:

```bash
claude mcp add --transport http agentfield http://localhost:8080/mcp
# cloud target (§0):
claude mcp add --transport http agentfield https://<cloud-host>/mcp --header "X-API-Key: <key>"
```

Other MCP clients: point them at the same streamable-HTTP URL
(`http://<server>/mcp`, transport `http`). It's stateless JSON-RPC — no session
setup. If the server has an API key, pass it as an `X-API-Key: <key>` header in
the client's MCP config.

Five tools are exposed: `discover_agents`, `get_reasoner_schema`,
`execute_reasoner` (starts an async run, returns a `run_id`), `get_run`, and
`wait_run`. Disable with `AGENTFIELD_MCP_ENABLED=false` (the route then 404s).

The MCP tools cover the common discover → execute → poll loop. The `af` CLI and
the raw HTTP API below remain the full-power path (sessions, streaming,
cancel-tree, secrets, load-aware pacing); reach for them when a task needs more
than the five tools give you.

## The flow

0. Resolve the server (§0) — desktop-configured cloud first, explicit
   `--server`/URL on every call.
1. Health-check the control plane.
2. Discover what agents and reasoners exist, and fetch the target reasoner's
   exact contract before the first call.
3. Execute — async for anything nontrivial. Fire independent calls concurrently.
4. Poll (or stream) until the execution finishes — and watch for wedged runs.

## 1. Is the control plane up?

```bash
curl -s http://localhost:8080/health
```

Healthy: `200` with `{"status":"healthy", ...}`. Connection refused on the
**local** target means no control plane is running — the user can open the
AgentField desktop app, or you can start one in the background (`af server`
blocks, so background it and poll `/health` until healthy). If the resolved
target is the desktop-configured **cloud** and this check fails, stop and
report it (§0) — do not retarget local.

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

### Fetch the exact contract before you dispatch — never guess inputs

Search and discovery tell you a reasoner exists; they do not license a call.
Before the first call to any reasoner, read its contract:

```bash
af agent agent-summary --id <agent_id> -s <server>   # all of an agent's reasoners: descriptions + input/output schemas + health + 24h metrics
# single reasoner via MCP: get_reasoner_schema
# or the fleet at once: curl -s "<server>/api/v1/discovery/capabilities?include_input_schema=true"
```

Read BOTH the description and the input schema, and follow them literally:

- A schema of `{"type":"object"}` with no properties is NOT "anything goes" —
  it means the agent registered no schema and **the description text is the
  entire contract**. Field names, required-ness, and types stated there are
  binding (e.g. swe-pro's `code_task`: `goal` and an **absolute** `dir` are
  required; model pools are comma-separated strings, not arrays).
- Result semantics live in the description too. Some agents report a failed
  job in the RESULT (`status: "fail"`) while the execution itself reads
  `succeeded` — check the result's own status field, not just the execution's.

### Entry points only — undescribed reasoners are internal

Agents register their internal pipeline stages alongside their public flows,
and discovery lists all of them. Dispatch ONLY to reasoners that carry the
`entrypoint` tag or a description. A reasoner with no description (e.g.
swe-planner's `run_*` stages) or tagged `internal` is plumbing invoked by an
orchestrator — calling it directly fails or corrupts a run. `af ls -e` lists
tagged entry points; when in doubt, pick the described reasoner whose
description names your use case.

### No coverage: offer to build it

Only decide that there is **no coverage** after completing the health check,
capability discovery (including each candidate's description and input schema),
and a ranked search for the requested job. Coverage requires a healthy active
installed agent whose reasoner description **and** input schema support that
job; a similar name or tag alone is not coverage.

If discovery finds a stopped-but-capable installed agent, explain that it can be
started with `af run <name>`; do not offer a replacement build. If those checks
establish that no installed reasoner supports the requested job, say explicitly:
**"No capable installed agent was found for this job."** Then offer to build the
missing capability: with the `agentfield-personal` skill when the user wants an
agent installed on this machine, or with the `agentfield` skill for a standalone
project repository.

A completed no-coverage result is evidence for the offer, not authorization to
create anything. List, inspect, and diagnose-only requests never authorize
building an agent. Hand off to a builder skill only when the original request
already authorized creating an agent, or when the user explicitly accepts this
offer.

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
plane is managing many agents at once. When a batch of independent jobs arrives
(ten PRs to review, five repos to scan), the default is to dispatch the whole
batch now and poll as a group — not one-at-a-time. What to know:

- Concurrent calls to the **same reasoner** are safe when the agent is (e.g.
  pr-af isolates concurrent reviews per PR). If an agent's docs don't say it's
  parallel-safe, assume same-target calls may contend on shared state and
  stagger them; different agents never contend. Some agents serialize ALL
  executions process-wide (swe-pro queues concurrent `code_task` calls behind
  one lock) — the reasoner description says so when known; dispatching more
  than one heavy call to such a node just builds a queue.
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

**Canary after reconfiguration, then fan out.** The one exception to
fire-everything-up-front: you just changed a node's runtime config (provider,
model, bin path — `af secrets set` + restart). A misconfigured harness can fail
*silently* — the run reports `succeeded` with empty results in seconds, and an
agent that posts externally (GitHub reviews, Slack, tickets) will publish that
garbage under the user's identity, once per dispatched call. So after any
config change: send ONE representative call, confirm it did real work (nonzero
cost/duration, plausible output — not just `succeeded`), then fan out the rest
at full width. This is a gate on the first call after a config change, not a
reason to serialize steady-state work.

## 4. Get the result

**What's in flight right now** — no IDs needed (also answers "how many agents
are running something"):

```bash
curl -s http://localhost:8080/api/v1/executions/active
# {"count":2,"runs":[{"run_id":"...","target":"pr-af.review","root_status":"running",
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

### If the result carries a `workspace_handle`, you can read the files

Some agents (SWE-AF) mirror the workspace they are building in, so you can open
the actual files instead of reasoning from the summary — including uncommitted
edits and untracked files that no git push would carry. You do not ask whether
this is available and there is nothing to configure: the handle is in the result
when it works and absent when it doesn't.

```json
"workspace_handle": {"v":1, "remote":"ssh://host:port"|"dir:/path",
                     "namespace":"...", "key":"<64 hex>", "token":"..."}
```

`furrow` is rarely on PATH. AgentField installs it to `$AGENTFIELD_HOME/bin/`
(default `~/.agentfield/bin/`), and a node that ships its own copy keeps it
inside the installed package. Resolve it from those; do not try to install it
yourself. POSIX sh only — no brace expansion, so the package dirs are spelled
out.

```sh
os=$(uname -s | tr A-Z a-z)
af_home=${AGENTFIELD_HOME:-$HOME/.agentfield}
furrow_bin() {  # $1 = furrow | furrow-dial
  command -v "$1" 2>/dev/null && return
  for c in "$af_home/bin/$1" \
           "$af_home"/packages/*/bin/"$1"-"$os"-* \
           "$af_home"/packages/*/go/bin/"$1"-"$os"-*; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
}
FURROW=$(furrow_bin furrow) DIAL=$(furrow_bin furrow-dial)
```

A `dir:` handle needs only `$FURROW`; an `ssh://` handle needs `$DIAL` too. If
either is missing, say so plainly and carry on from the result — the mirror is
fine, this machine just has no client for it.

```bash
# ssh:// handle — furrow-dial carries the protocol; nothing else changes
export FURROW_SSH_COMMAND="$DIAL" FURROW_DIAL_TOKEN=<token> FURROW_DIAL_INSECURE=1
FURROW_RECOVERY_KEY=<key> "$FURROW" clone <remote>/<namespace> ./run-workspace --no-watch

# dir: handle (same machine) — clone rejects directory remotes, so pair instead.
# The path is the handle's remote with the "dir:" prefix removed; don't append
# anything to it.
git init -q run-workspace && "$FURROW" --repo run-workspace watch --no-daemon
"$FURROW" --repo run-workspace pair <path> --name <namespace> --key <key>
"$FURROW" --repo run-workspace sync --pull --bootstrap
```

`"$FURROW" --repo run-workspace sync --follow` keeps it current while the run
works. Read and diff freely. Treat it as a mirror, not a shared drive: it is
one-writer, and edits go back as a merge (`furrow merge <fork> --check "<cmd>"`),
so change files between issues or on a fork rather than while the agent writes.

`get_workspace_handle` re-fetches a handle mid-run:
`POST /api/v1/execute/<agent>.get_workspace_handle` with `{"input":{"run_id":"..."}}`.
`{"available": false}` means no mirror — carry on without it. Not every build
ships this reasoner: check the agent's reasoner list (discovery or
`agent-summary`) before calling it; if it's absent, the node predates the
mirror feature and results simply never carry a handle.

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
| connection refused on :8080 | local control plane not running | desktop app, or background `af server` and poll `/health` |
| desktop-configured cloud unreachable | cloud deployment down, or URL/key stale | stop and tell the user (§0) — never silently retarget local |
| 401/403 from a cloud target | missing or wrong `X-API-Key` | key from desktop `settings.json` `cloud.apiKey`, or `af auth login --server <url>` |
| agent `inactive` in discovery / missing | node installed but not running (or not installed) | `af list`, then `af run <name>` — or `af install <source>` |
| `missing required environment variables: X` from `af run` | required key not configured | `af secrets set X` (value via stdin/arg; `--node <name>` for node-scoped) — or desktop app → Agents → Keys |
| HTTP 502 with `error_message` | the agent itself errored | read `af logs <name>`, fix, retry |
| execution `running` but latest_activity stale & logs quiet | wedged run | wedge protocol above: cancel-tree → restart agent → re-submit |
| result claims success with zero findings/output on nontrivial input | possible silent tool failure inside the agent | check `af logs <name>` for that run before trusting it |

## Local ops cheat sheet (af CLI)

All commands accept `-s/--server <url>` and `-k/--api-key <key>` — required on
every invocation when the resolved target is the cloud (§0).

```bash
af list                    # installed agents + status
af ls [query]              # search reasoners across running agents (NOT the install registry)
af ls -e                   # only entry-point reasoners — the callable surface
af agent agent-summary --id <name>   # full contract: reasoners, schemas, health, 24h metrics
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

- Resolve the server per §0 and pass it explicitly (`--server` / full URL) on
  every call. A desktop-configured cloud beats the local default; an
  unreachable configured cloud is a stop-and-report, never a silent fallback.
- Fetch the reasoner's contract before the first call. A vacuous schema means
  the description is the contract — follow it literally.
- Dispatch only to `entrypoint`-tagged or described reasoners. Undescribed or
  `internal`-tagged reasoners are pipeline stages — never call them directly.
- Every call goes through the control plane — never POST to an agent's own port.
  The one exception is a `workspace_handle`: its `ssh://` endpoint is a furrow
  transport, not the agent's HTTP port, and the per-run token in the handle is
  what authorizes it. Reading files there is not an agent call.
- Kwargs live under `"input"`. Empty input is `{"input": {}}`.
- Async + poll for anything that might exceed a few seconds; sync is for quick
  lookups only. Independent async calls go out together, not one at a time.
- Only dispatch to agents whose discovery `health_status` is `"active"`.
- Don't guess endpoints. The surface above is the contract; if something is
  missing, ask `GET /api/v1/agentic/discover?q=<keyword>` before inventing a route.
- Building or modifying an agent (new reasoners, scaffolds, deploys) is the
  **agentfield** skill's job — switch to it for that.
