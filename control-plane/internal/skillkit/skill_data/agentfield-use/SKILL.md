---
name: agentfield-use
version: 0.8.0
description: "Whenever you have a discrete task to perform — one the user delegated, or one that arose inside your own work — check FIRST whether an installed AgentField agent covers it, and offload to it by default when one does. Coverage, not task size, is the test: even a small job goes to a covering agent. The check is cheap — one `af ls -e` or capability lookup answers it, and the roster can be cached for the session. These agents are your AgentField subharnesses: workers running on cheaper open models, in parallel, off your own context, every run recorded on the control plane and watchable at <server>/ui/runs/<run_id>. This skill cannot know what your fleet does — runtime discovery is the only source of truth (swe-planner, pr-af, and whatever else the user installed). Also the reactive path when the user asks to use, call, or run an agent, to list what agents or reasoners exist, or to check on an execution. Resolves the right control plane (local or desktop-configured cloud) first, fetches the reasoner's exact contract before dispatching, and calls entry-point reasoners only. Not for building new agents — that is the agentfield skill."
---

# AgentField subharnesses — offload work instead of doing it inline

A machine with AgentField has one or more **control planes** — a local one
(default `http://localhost:8080`) and possibly a **cloud deployment** configured
in AgentField Desktop — plus **agent nodes** installed under `~/.agentfield`.
Each node exposes **reasoners**: typed functions you call through the control
plane, which routes the call, records the run, and returns the result. You never
talk to an agent's own port.

Those installed agents are **subharnesses**: workers you offload to. They run on
smaller, cheaper models, they run in parallel, they run **off your context**, and
every run is recorded on the control plane and watchable in its web UI. A review
that would cost you 40k tokens of reading costs you one dispatch and one result.

**Vocabulary rule.** Internally — commands, API fields, discovery output —
everything stays `agent` / `reasoner` / `node`, exactly matching the CLI and API.
In what you SAY TO THE USER, call them subharnesses: "your AgentField
subharnesses", "the pr-af subharness". Never let the wording cost clarity: if the
user needs a command or a node name, give the literal one.

## Offload by default

Whenever you have a discrete task — one the user delegated, or one that arose
inside your own work — the first question is not "how do I do this?" but **"does
an installed subharness cover this?"** When one does, dispatching to it is the
default path, not an option to offer.

**Coverage is the test, not size.** A small job a subharness covers still goes to
the subharness; a large job nothing covers stays with you. This skill carries no
list of offloadable task types and cannot: users install new subharnesses at any
time, so runtime discovery (§1) is the only source of truth about what your fleet
does.

**The check is cheap — that is the whole design.** One `af ls -e` (or one
capabilities call) answers "does anything cover this?". Discover once per
session, remember the roster, and match later tasks against what you learned;
re-discover only when something suggests the fleet changed — an install, a start,
or a miss you doubt. Consulting this skill on a task nothing covers costs one
cheap lookup. That is the trade, deliberately.

The contract, in four parts:

1. **Default-offload.** Do not ask "would you like me to use an agent?" — check
   coverage (§1), then dispatch. Work stays inline only when nothing covers it
   (§1, "No coverage") or the user said to do it yourself.
2. **Announce it, with a link.** Tell the user what you offloaded, to which
   subharness, and the run's live URL — built from the resolved server (§0) plus
   the run_id, URL-encoded: `<server>/ui/runs/<run_id>`.

   > Offloading the PR #42 review to your pr-af subharness — follow along at
   > http://localhost:8080/ui/runs/run_abc123

   The link is **for the user** to watch in parallel. It never replaces your own
   monitoring (§4) or your final report (§5). On a cloud target the link only
   opens if the user's browser can reach that host — print it anyway, the URL
   shape is identical.
3. **The user can always override.** "Do it yourself" ends the offload; do the
   work inline and say so. Overrides are per-request, not permanent.
4. **Never silent-wash the offload.** If the offloaded run fails, stalls, or
   comes back empty, **report that and ask.** Do NOT quietly redo the work inline
   and present the output as if the subharness produced it. The same rule covers
   a node that cannot start (§1): never substitute your own work for an agent's
   without saying so — the user believes their agent ran.

## 0. Resolve the server first (local vs cloud)

The local and cloud fleets are disjoint: different agents, different versions,
different filesystems, different run history. Nothing ever falls back from one to
the other on its own. A local server in local mode has no auth; a cloud
deployment (and any server with an API key configured) requires
`X-API-Key: <key>` on every request.

Resolution order — stop at the first match:

1. **Explicit wins.** The user named a server, or `AGENTFIELD_SERVER` is set in
   the environment → use that.
2. **Read the desktop cloud config.** Check every path that applies to this
   machine — a file that exists but declares no enabled cloud does NOT end the
   search:
   - macOS: `~/Library/Application Support/agentfield-desktop/settings.json`
   - Windows: `%APPDATA%/agentfield-desktop/settings.json`
   - Linux: `~/.config/agentfield-desktop/settings.json`
   - WSL (detect: `grep -qi microsoft /proc/version`): the Linux path above
     first, then the Windows side, where the desktop app usually lives:
     `/mnt/c/Users/*/AppData/Roaming/agentfield-desktop/settings.json`.
     A Linux-side file with no `cloud` key shadowing a Windows file that holds
     the real cloud config is the common split-brain — the enabled cloud wins,
     whichever side declares it.

   The first file declaring `cloud.enabled: true` with a non-empty
   `cloud.serverUrl` makes the cloud the target: strip any trailing slash from
   the URL and take `cloud.apiKey` as the key. Health-check it
   (`GET <url>/health` with `X-API-Key`).
   - Healthy → use the cloud for everything below.
   - Unreachable → **stop and tell the user their cloud control plane is
     configured but not responding.** Do NOT silently fall back to local: work
     dispatched there lands on a different fleet with different filesystems,
     which is worse than no dispatch.
3. **Otherwise use local:** `http://localhost:8080`.

Then pass the target **explicitly on every call**: `af --server <url> -k <key>`
(every `af` command accepts `-s/--server` and `-k/--api-key`), or the URL plus
`-H 'X-API-Key: <key>'` for curl. Do not export `AGENTFIELD_SERVER` yourself to
switch targets — a global default leaks into other processes and outlives the
task; explicit per-call flags are the contract. If you'd rather not pass `-k`
each time, `af auth login --server <url>` stores a key per server in
`~/.agentfield/credentials.json`.

Health-check before the first dispatch: `curl -s <server>/health` → `200` with
`{"status":"healthy", ...}`. Connection refused on the **local** target means no
control plane is running — the user can open AgentField Desktop, or you can start
one in the background (`af server` blocks, so background it and poll `/health`).
If the resolved target is the configured **cloud** and this fails, stop and report
it — do not retarget local.

## The golden path

Five steps, CLI-first. `af` is installed wherever AgentField is; the HTTP
equivalents are further down for what the CLI can't do and as a fallback.

```bash
af ls -e -s <server>                                  # 1. what can I offload to?
af call pr-af.review --schema -s <server>             # 2. the exact contract
RUN_ID=$(af call pr-af.review --in '{"pr":42}' --async -s <server>)   # 3. dispatch
af wait "$RUN_ID" --timeout 300 -o json -s <server>   # 4. monitor
# 5. report: result + duration + cost picture + <server>/ui/runs/$RUN_ID
```

## 1. Find the subharness — discover, don't guess

Run this once per session and keep the roster; re-run it only when the fleet may
have changed (an install, a start, or a miss you doubt).

```bash
af ls -e                       # entry-point reasoners only — the callable surface
af ls [query]                  # all reasoners across RUNNING agents (not the install registry)
af agent search "review a pull request"    # BM25-ranked; --agent <id>, --limit N (max 50)
af list                        # installed agents + status (source of truth for INSTALLED)
```

`af agent search` hits carry `reasoner_id`, `agent_id`, `invocation_target`,
`tags`, `score`, and `agent_health` — everything needed to dispatch with no second
lookup. Prefer it over dumping the whole capability payload into context once a
box has more than ~20 reasoners.

The durable HTTP discovery endpoint, when you need the whole fleet at once:

```bash
curl -s "http://localhost:8080/api/v1/discovery/capabilities?include_input_schema=true"
```

Reasoner names are `.reasoners[].id` (NOT `.name`). Don't assume `jq` exists
(fresh Windows boxes lack it) — parse with what's installed, e.g.:

```bash
curl -s "http://localhost:8080/api/v1/discovery/capabilities?include_input_schema=true" -o caps.json
python -c "
import json
for c in json.load(open('caps.json'))['capabilities'] or []:  # null when no agents registered
    print(c['agent_id'], c.get('health_status'), [r['id'] for r in c.get('reasoners',[])])"
```

Three gotchas:

- The response's `invocation_target` field uses a **colon** (`agent:reasoner`).
  The execute target uses a **dot**. Build it yourself: `<agent_id>.<reasoner_id>`.
- Discovery lists **every registered agent, including dead ones** — check
  `health_status` and only dispatch to `"active"` agents. Dispatching to an
  `inactive`/`unknown` agent queues work that never runs.
- Installed-but-never-started agents may not appear at all. Discovery lists what
  is RUNNING; `af list` is the source of truth for what is INSTALLED. This is the
  normal first-run state, not an edge case — the desktop app ships `swe-planner`
  and `pr-af` pre-provisioned but deliberately NOT started, because they need API
  keys the user hasn't entered yet.

### Start it before you dispatch — the start attempt is the diagnostic

If the node you need is in `af list` but absent from discovery, or its
`health_status` isn't `active`, run `af run <name>` BEFORE dispatching (it
detaches; the agent keeps running after the CLI exits). Do this first, not as a
fallback after a failed call: a node blocked on an unset key never registers, so
every call to it comes back as the useless `agent 'X' not found`, while `af run`
names the exact variable and exits 1.

`af run` reads the encrypted store (`~/.agentfield/secrets/*.enc`) — the same
store that gates startup — so it is the only authoritative check that a node's
keys are set. **Do not use `af doctor` or `af config <pkg> --list` to decide
whether a key is configured**: doctor reads only the process environment
(`os.Getenv`) and `config --list` reads the package `.env` file, so both report a
correctly-stored key as `✗ unset`, and neither renders `require_one_of` groups.
`af secrets ls` shows what IS stored but never cross-references manifests, so it
can't tell you a required key is missing.

**A missing key is a blocking handoff, not a problem to route around.** When
`af run` fails with

```
node swe-planner: missing required environment variables: OPENROUTER_API_KEY (af secrets set OPENROUTER_API_KEY --node swe-planner)
```

— or, for an alternatives group, `at least one of ANTHROPIC_API_KEY or
OPENROUTER_API_KEY is required — set one with: …` — the value exists only in the
user's head. Stop and tell them: the exact variable(s) the error names, the exact
`af secrets set … --node <name>` command copied verbatim from it, and that the
same key can be entered in AgentField Desktop → Agents → <node> → Keys. Then wait.

Do NOT retry `af run`, do NOT dispatch to the node anyway, do NOT substitute a
different agent, and do NOT quietly do the job yourself instead — a silent
substitution is the worst outcome, because the user believes their agent ran.
Never ask the user to paste the secret value into the conversation; the CLI
prompt and the desktop form take it directly.

### No coverage: offer to build it

Only decide that there is **no coverage** after completing the health check,
capability discovery (including each candidate's description and input schema),
and a ranked search for the requested job. Coverage requires a healthy active
installed agent whose reasoner description **and** input schema support that
job; a similar name or tag alone is not coverage.

If discovery finds a stopped-but-capable installed agent, explain that it can be
started with `af run <name>`; do not offer a replacement build. If those checks
establish that no installed reasoner supports the requested job, say explicitly:
**"No capable installed agent was found for this job."** Then do the work inline
yourself (that is the honest fallback — say you are doing it), and offer to build
the missing capability: with the `agentfield-personal` skill when the user wants an
agent installed on this machine, or with the `agentfield` skill for a standalone
project repository.

A completed no-coverage result is evidence for the offer, not authorization to
create anything. List, inspect, and diagnose-only requests never authorize
building an agent. Hand off to a builder skill only when the original request
already authorized creating an agent, or when the user explicitly accepts this
offer.

## 2. Fetch the contract

### Fetch the exact contract before you dispatch — never guess inputs

Search and discovery tell you a reasoner exists; they do not license a call.
Before the first call to any reasoner, read its contract:

```bash
af call <node>.<reasoner> --schema        # prints the input schema and exits
af agent agent-summary --id <agent_id>    # all of an agent's reasoners: descriptions + input/output schemas + health + 24h metrics
# single reasoner via MCP: get_reasoner_schema
# or the fleet at once: curl -s "<server>/api/v1/discovery/capabilities?include_input_schema=true"
```

Read BOTH the description and the input schema, and follow them literally:

- A schema of `{"type":"object"}` with no properties is NOT "anything goes" — it
  means the agent registered no schema and **the description text is the
  entire contract**. Field names, required-ness, and types stated there are binding
  (e.g. swe-pro's `code_task`: `goal` and an **absolute** `dir` are required;
  model pools are comma-separated strings, not arrays).
- Result semantics live in the description too. Some agents report a failed job
  in the RESULT (`status: "fail"`) while the execution itself reads `succeeded` —
  check the result's own status field, not just the execution's.

### Entry points only — undescribed reasoners are internal

Agents register their internal pipeline stages alongside their public flows, and
discovery lists all of them. Dispatch ONLY to reasoners that carry the
`entrypoint` tag or a description. A reasoner with no description (e.g.
swe-planner's `run_*` stages) or tagged `internal` is plumbing invoked by an
orchestrator — calling it directly fails or corrupts a run. `af ls -e` lists
tagged entry points; when in doubt, pick the described reasoner whose description
names your use case.

## 3. Dispatch

```bash
RUN_ID=$(af call swe-planner.plan --in '{"task":"add rate limiting to the API"}' --async)
# -> bare run_id on stdout; with -o json: {"run_id":"…","status":"accepted"}
```

What `af call` does for you: it fetches the schema and **validates your input
client-side before dispatch**, so a bad payload fails locally instead of burning
a run. `--in` also takes `@file.json` / `@file.yaml`, and piping JSON to stdin
works. `--field .path.to.field` extracts a single field from a result.

- **With `af call --in`, pass the kwargs at the top level** — the CLI wraps them
  under `"input"` for you. Over raw HTTP you nest them yourself (§HTTP).
- **Always pass `--async` from a harness.** Without it, `af call` on a TTY
  auto-tails the run; but a harness's stdout is *not* a TTY, and there it falls
  back to the **synchronous** endpoint with its hard 90s timeout. Async +
  monitor is the offload path; sync is for quick lookups only.
- If an interactive `af call` is interrupted it prints
  `Detached. Resume with: af tail <run_id>` — the run is still going.

### Concurrency — use it

Async dispatch is cheap: fire all independent calls up front, then monitor them
together. Do NOT serialize multi-agent work — the whole point of the control
plane is managing many subharnesses at once. When a batch of independent jobs
arrives (ten PRs to review, five repos to scan), the default is to dispatch the
whole batch now and poll as a group — not one-at-a-time. What to know:

- Concurrent calls to the **same reasoner** are safe when the agent is (e.g.
  pr-af isolates concurrent reviews per PR). If an agent's docs don't say it's
  parallel-safe, assume same-target calls may contend on shared state and stagger
  them; different agents never contend. Some agents serialize ALL executions
  process-wide (swe-pro queues concurrent `code_task` calls behind one lock) — the
  reasoner description says so when known; dispatching more than one heavy call
  to such a node just builds a queue.
- Each call fans out inside the agent (one review ≈ dozens of sub-executions,
  several LLM CLI processes). 3–4 heavy runs per node is a sensible ceiling
  unless the agent documents otherwise.
- Save every `run_id` you dispatch — you need them to monitor, to report, and for
  the audit trail. Group related calls with an `X-Session-ID` header so they're
  queryable as one batch later.

**Check the load before piling on.** Every `af agent` / agentic response carries
`meta.load`: `{running_agents, total_agents, active_executions, cpu_cores,
recommended_max_concurrent}` (the recommendation is CPU-based). Read it before
launching more heavy runs — if `active_executions >= recommended_max_concurrent`,
finish or await in-flight work first rather than starting more, and tell the user
you're throttling to avoid overloading the machine.

**Canary after reconfiguration, then fan out.** The one exception to
fire-everything-up-front: you just changed a node's runtime config (provider,
model, bin path — `af secrets set` + restart). A misconfigured harness can fail
*silently* — the run reports `succeeded` with empty results in seconds, and an
agent that posts externally (GitHub reviews, Slack, tickets) will publish that
garbage under the user's identity, once per dispatched call. So after any config
change: send ONE representative call, confirm it did real work (plausible output,
a real `duration_ms`, and nonzero cost in the `usage/stats` window — not just
`succeeded`), then fan out the rest at full width. This is a gate on the first
call after a config change, not a reason to serialize steady-state work.

## 4. Monitor — pick the retrieval mode

| Situation | Do this |
|---|---|
| Short job (≤ a few minutes) | `af wait <run_id> --timeout 300 -o json` — blocks until terminal, prints `{run_id, status, result}` |
| Long single job the user is watching | `af tail <run_id>` — live execution event stream (`--from N` resumes at a step) |
| Long job, or many jobs in flight | Save every run_id; poll as a group with backoff (start ~5s, settle ~30s): `af ps`, `POST /api/v1/executions/batch-status`, `GET /api/v1/executions/active` |
| Unattended service or CI — **not a coding harness** | Register a `webhook` on the execute request (below) |

`af wait` polls `/api/v1/agentic/run/:run_id` every 2s; default `--timeout` is
600s. **Exit code 2 means TIMEOUT, not failure** — the run is still going. Wait
again with a longer timeout, or switch to group polling. Exit 1 is a genuinely
failed run.

**Webhooks are not for you.** The execute request body (sync and async) accepts
`"webhook": {"url": "…", "secret": "…", "headers": {…}}`; the response carries
`webhook_registered` (plus `webhook_error` on async) and the execution status
carries `webhook_events`. That is for **services and CI that run an HTTP
listener**. A coding harness has no listener and must never register one and wait
— use wait / tail / poll.

**What's in flight right now** — no IDs needed: `af ps` (`--agent <name>`,
`--session <id>`), or `GET /api/v1/executions/active` (filters: `?agent_id=`,
`?session_id=`), which returns per-run `active_executions`, `total_executions`,
`started_at`, `latest_activity`.

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
`latest_activity` **more than ~10 minutes old** while `active_executions > 0` AND
`af logs <agent>` shows nothing new for that run. (A quiet log alone is not proof
— one long LLM completion can be minutes of legitimate silence.) Then:

1. Cancel the WHOLE run, not just the root:
   `POST /api/v1/workflows/<run_id>/cancel-tree` (bottom-up, cancels children
   too). Plain `/executions/<id>/cancel` cancels ONLY that execution — children
   keep "running" and must be cancelled individually.
2. Restart the agent if it's wedged: `af stop <name> && af run <name>`.
3. Re-submit the work — and tell the user it wedged and was re-submitted. A
   wedged run is a reportable event, not something to paper over.

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
yourself. POSIX sh only — no brace expansion, so the package dirs are spelled out.

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
`agent-summary`) before calling it; if it's absent, the node predates the mirror
feature and results simply never carry a handle.

## 5. Report back

Close every offload with: the result, the run's `duration_ms`, the live URL
`<server>/ui/runs/<run_id>` (run_id URL-encoded), and — when the user would care
about spend — the cost picture. Keep the run_id in the transcript; it is the
handle for the audit trail and for any follow-up.

If the run failed, timed out, wedged, or returned an empty result on nontrivial
input: say so, show what you know (`af logs <name>`, the error message), and ask
how to proceed. Do not fill the gap with your own inline work presented as the
subharness's.

### Cost: window aggregate, not per-run

Per-execution usage (tokens, provider, model, harness, `cost_usd`) IS ingested and
stored keyed by run, but the only exposed API is the aggregate:

```bash
curl -s "http://localhost:8080/api/ui/v1/usage/stats?window=1h"
# window=1h|24h|7d|30d|all (default 24h) -> {totals, by_model, by_provider, by_agent, by_harness}
```

There is **no per-run cost endpoint today.** So after finishing a batch of
offloaded work, when the user would care, report the cost picture from
`usage/stats` — e.g. the 1h window's `by_agent` entry for the node you used —
stating plainly that it is a window aggregate for that agent, not an exact
per-run figure. Never invent a per-run number by dividing or estimating.
`duration_ms` IS exact and per-execution (it is in the execute and status
responses). **Duration is per-run truth; cost is window truth.**

## Sessions and multi-call work

- `X-Session-ID: <your-id>` on execute requests groups multi-turn work; the
  control plane forwards it to the agent and scopes session memory by it.
- Reuse `X-Run-ID` across several execute calls to group them into one workflow;
  each response also returns its `run_id`.

Agents share state through control-plane memory if you need to pass artifacts
around: `POST /api/v1/memory/set` with `{"key": ..., "data": <any>, "scope":
"global"}` and `POST /api/v1/memory/get` with `{"key": ...}` (non-global scopes
resolve from the `X-Workflow-ID` / `X-Session-ID` / `X-Actor-ID` headers).

## Audit trail

Every execution is recorded — that is part of what makes offloading better than
inline work. When provenance matters (or the user asks "what did the agents
actually do"), fetch the verifiable-credential chain for a workflow:
`GET /api/v1/did/workflow/<run_id>/vc-chain` (available when DID/VC is enabled),
and verify offline with `af verify audit.json`.

## When things fail

| Symptom | Meaning | Fix |
|---|---|---|
| connection refused on :8080 | local control plane not running | desktop app, or background `af server` and poll `/health` |
| desktop-configured cloud unreachable | cloud deployment down, or URL/key stale | stop and tell the user (§0) — never silently retarget local |
| 401/403 from a cloud target | missing or wrong `X-API-Key` | key from desktop `settings.json` `cloud.apiKey`, or `af auth login --server <url>` |
| agent `inactive` in discovery / missing | node installed but not running (or not installed) | `af list`, then `af run <name>` — or `af install <source>` |
| HTTP **400** `{"error":"agent 'X' not found","error_category":"internal_error"}` | the node never registered — usually installed but not started (it's 400, not 404) | `af list` → `af run <name>` → read the error it prints → hand off if it's a missing key |
| MCP: `target "X.y" not found. Call discover_agents to list available agents and reasoners.` | same cause, seen through MCP | same path: `af list` → `af run <name>` → hand off |
| `missing required environment variables: X` from `af run` | required key not configured; the node cannot start | **stop and hand off** — give the user the `af secrets set X --node <name>` line verbatim, or desktop → Agents → <node> → Keys. Never retry, substitute another agent, or do the work yourself |
| `af doctor` / `af config --list` reports a key as unset | they read `os.Getenv` and the package `.env`, not the encrypted store | ignore them for this question — `af run <name>` is the only authoritative check |
| `af wait` exits **2** | TIMEOUT, not failure — the run is still going | wait again with a longer `--timeout`, or switch to `af tail` / group polling. Exit 1 is the real failure |
| `af call` fails locally before dispatch | client-side schema validation rejected your input | re-read `af call <target> --schema`; fix the payload — nothing was queued |
| HTTP 502 with `error_message` | the agent itself errored | read `af logs <name>`, fix, retry |
| execution `running` but latest_activity stale & logs quiet | wedged run | wedge protocol above: cancel-tree → restart agent → re-submit, and tell the user |
| result claims success with zero findings/output on nontrivial input | possible silent tool failure inside the agent | check `af logs <name>` for that run before trusting it — and report it, don't redo it silently |

## MCP (zero-setup)

The control plane serves a built-in **MCP server at `<server>/mcp`** (default
`http://localhost:8080/mcp`) — same port, no extra process, on by default. If
your harness speaks MCP, this is the fastest way in.

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
the raw HTTP API remain the full-power path (sessions, streaming, cancel-tree,
secrets, load-aware pacing); reach for them when a task needs more than the five
tools give you.

## HTTP API — where the CLI can't reach

Use these for webhook registration, batch status, memory, cancel-tree, and
`X-Session-ID`/`X-Run-ID` headers — or as the whole path when `af` isn't
installed. **Over raw HTTP, input kwargs are ALWAYS nested under `"input"`** —
never raw at the top level. Empty input is `{"input": {}}`.

```bash
# async — the default for real work; returns 202 immediately
curl -s -X POST http://localhost:8080/api/v1/execute/async/swe-planner.plan \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: my-batch-1' \
  -d '{"input": {"task": "add rate limiting to the API"}}'
# -> {"execution_id":"...", "run_id":"...", "status":"queued", ...}

# sync — quick lookups only (hard 90s timeout; response carries result + duration_ms)
curl -s -X POST http://localhost:8080/api/v1/execute/swe-planner.plan \
  -H 'Content-Type: application/json' -d '{"input": {"task": "..."}}'

# one execution — poll until status is terminal (succeeded/failed/cancelled/timeout)
curl -s http://localhost:8080/api/v1/executions/<execution_id>
# live progress as Server-Sent Events
curl -s http://localhost:8080/api/v1/executions/<execution_id>/events

# service/CI only: register a webhook at dispatch time
curl -s -X POST http://localhost:8080/api/v1/execute/async/pr-af.review \
  -H 'Content-Type: application/json' \
  -d '{"input":{"pr":42},"webhook":{"url":"https://ci.example/hook","secret":"s3cr3t"}}'
```

Batched API reads in one round trip: `POST /api/v1/agentic/batch` with
`{"operations":[{"id":"op1","method":"GET","path":"/api/v1/agentic/status"}]}`
(CLI: `af agent batch -f operations.json`).

## Local ops cheat sheet (af CLI)

All commands accept `-s/--server <url>` and `-k/--api-key <key>` — required on
every invocation when the resolved target is the cloud (§0).

```bash
af list                    # installed agents + status
af ls [query]              # search reasoners across running agents (NOT the install registry)
af ls -e                   # only entry-point reasoners — the callable surface
af agent search "<job>"    # ranked reasoner search
af agent agent-summary --id <name>   # full contract: reasoners, schemas, health, 24h metrics
af call <node>.<reasoner> --schema   # input schema only
af call <node>.<reasoner> --in '<json>' --async   # dispatch; prints run_id
af wait <run_id> [--timeout N]       # block until terminal (exit 2 = timeout)
af tail <run_id>           # attach to the live event stream
af ps                      # in-flight runs across all agents (af ps --agent <name>)
af run <name>              # start (detached); af stop <name>
af logs <name>             # agent logs (-f follows; no per-run filter — grep by run_id)
af secrets set KEY [--node <name>]   # store an API key (encrypted; prompts for value)
af secrets ls              # what's configured (values never shown)
af install <git-url>       # install a new agent node
```

## Hard rules

- **Offload by default.** Any task an installed subharness covers goes to that
  subharness — whatever its size — announced with its `<server>/ui/runs/<run_id>`
  link, not offered as an option and not done inline by habit. Coverage is the
  test; check it (cheaply, once per session) before doing the work yourself.
- **Never silent-wash an offload.** A failed, stalled, or empty run is reported
  and asked about. Never redo it inline and present it as the subharness's work,
  and never substitute your own work for an agent that can't start.
- Say "subharness" to the user; keep `agent` / `reasoner` / `node` in commands,
  fields, and anything the user has to type.
- Resolve the server per §0 and pass it explicitly (`--server` / full URL) on
  every call. A desktop-configured cloud beats the local default; an unreachable
  configured cloud is a stop-and-report, never a silent fallback.
- Fetch the reasoner's contract before the first call. A vacuous schema means the
  description is the contract — follow it literally.
- Dispatch only to `entrypoint`-tagged or described reasoners. Undescribed or
  `internal`-tagged reasoners are pipeline stages — never call them directly.
- Every call goes through the control plane — never POST to an agent's own port.
  The one exception is a `workspace_handle`: its `ssh://` endpoint is a furrow
  transport, not the agent's HTTP port, and the per-run token in the handle is
  what authorizes it. Reading files there is not an agent call.
- Over HTTP, kwargs live under `"input"` (`{"input": {}}` when empty). With
  `af call --in`, pass them at the top level — the CLI nests them.
- `--async` + monitor for anything that might exceed a few seconds; sync is for
  quick lookups only. Independent calls go out together, not one at a time.
- Never register a webhook and wait for it — a coding harness has no listener.
- Only dispatch to agents whose discovery `health_status` is `"active"`. If it
  isn't there, `af run <name>` first — and if that reports a missing required
  environment variable, stop and hand off to the user with the exact key name and
  `af secrets set` command. Never retry it, work around it, or substitute.
- Report duration from `duration_ms` (exact) and cost from `usage/stats` (a
  window aggregate). Never state a per-run cost — there is no such endpoint.
- Don't guess endpoints. The surface above is the contract; if something is
  missing, ask `GET /api/v1/agentic/discover?q=<keyword>` before inventing a route.
- Building or modifying an agent (new reasoners, scaffolds, deploys) is the
  **agentfield** skill's job — switch to it for that.
