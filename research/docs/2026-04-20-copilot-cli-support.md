---
date: "2026-04-20 22:30:00 UTC"
researcher: "Copilot CLI"
git_commit: "0f42e612"
branch: "copilot-cli-support"
repository: "agentfield"
topic: "Adding GitHub Copilot CLI support to AgentField"
tags: [research, codebase, copilot-cli, sdk, skillkit, auth]
status: complete
last_updated: "2026-04-20"
last_updated_by: "Copilot CLI"
---

# Research — Adding GitHub Copilot CLI Support to AgentField

## Research Question

> Create a new branch `copilot-cli-support`, research and figure out how to add support for GitHub Copilot CLI.
>
> Clarified scope (via ask_user):
> 1. **Wrap Copilot CLI as an AgentField agent** — Copilot becomes a reasoner/skill callable by the control plane.
> 2. Ship **Copilot skills** as part of the AgentField installer.
> 3. Ensure **Copilot CLI authentication** works with AgentField.

This document is intentionally a *documentation-of-current-state* plus an integration map. No code has been written; follow-up work will produce a spec/plan and then an implementation PR.

---

## Summary

AgentField already has every primitive needed to wrap GitHub Copilot CLI as an agent, ship Copilot-facing skill packets through its installer, and share authentication with the host user's `copilot` invocation — **without any new architecture**. Specifically:

- **(a) Copilot-as-agent** maps cleanly onto the existing Python/Go SDK pattern: one agent node, one or more reasoners, each of which shells out to `copilot -p "<prompt>" --allow-all-tools --log-level none` (one-shot mode) and returns structured output. The SDK's `@app.reasoner()` decorator and `RegisterReasoner` already support subprocess wrapping; `ExecutionContext` lets us propagate `X-Run-ID` / `X-Session-ID` so each AgentField run gets a dedicated Copilot `--session-id` or `COPILOT_HOME`. See [§2.1](#21-copilot-as-agent).
- **(b) Skillkit integration** is a drop-in: AgentField's skillkit already writes a canonical `SKILL.md` into per-tool target dirs (`~/.claude/skills/…`, `~/.codex/AGENTS.override.md`, etc.). Copilot CLI looks for skills at `~/.copilot/skills/<name>/SKILL.md` — the exact same shape. Adding Copilot as a **new target** under `control-plane/internal/skillkit/target_copilot.go` plus one entry in `detectedTargets` is the full footprint. See [§2.2](#22-skills-installation).
- **(c) Authentication reuse** works through env var inheritance: Copilot CLI honors `COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN` (in that precedence) and falls back to the `gh` CLI's stored token. AgentField already lets agents read their own env, so `af run copilot` can inherit the user's Copilot auth with zero extra plumbing. For stronger isolation we can spawn Copilot with `COPILOT_HOME=$AGENTFIELD_HOME/copilot-<node>` to sandbox state. See [§2.3](#23-authentication-reuse).

MCP is *not* a viable integration path today: all AgentField MCP code was removed in commit `f732ed5e` (2026-04-07). The integration must go through the SDK/HTTP surface, not MCP.

There is currently **one trivial mention** of "GitHub Copilot" in the repo (`docs/CONTRIBUTING.md:101`, a list of AI tools contributors may use). No Copilot-specific code exists.

---

## Detailed Findings

### 1. Current state of AgentField (as of `0f42e612`)

#### 1.1 SDK agent shape (target surface for Copilot-as-agent)

**Python SDK** ([`sdk/python/agentfield/agent.py`](../../sdk/python/agentfield/agent.py)): `Agent` is a FastAPI subclass. `@app.reasoner()` (`agent.py:~1693`) registers a function and auto-mounts `POST /reasoners/<id>`. `ExecutionContext.from_request()` at `agent.py:1843` parses correlation headers (`X-Run-ID`, `X-Execution-ID`, `X-Session-ID`, `X-Parent-Execution-ID`, `X-Workflow-ID`, `X-Actor-ID`, plus DID headers) into a context object that can be injected into handlers by declaring a parameter `execution_context: ExecutionContext`. Control-plane URL resolution order (`agent.py:556-558`): explicit `agentfield_server` arg → `AGENTFIELD_SERVER` → `AGENTFIELD_SERVER_URL` → `http://localhost:8080`. On `serve()` the server binds to `PORT` (injected by `af run`).

Minimal Python agent (verbatim shape from research):
```python
from agentfield import Agent

app = Agent(node_id="copilot", agentfield_server=os.getenv("AGENTFIELD_SERVER"))

@app.reasoner()
async def run_command(prompt: str, execution_context) -> dict:
    # shell out to `copilot -p prompt …`
    ...

if __name__ == "__main__":
    app.serve()
```

**Go SDK** ([`sdk/go/agent/agent.go`](../../sdk/go/agent/agent.go)): `agent.New(Config{...})` + `app.RegisterReasoner(name, handler, opts...)` (`agent.go:~522`). Handler signature is `func(ctx context.Context, input map[string]any) (any, error)`; retrieve `ExecutionContext` via `agent.ExecutionContextFrom(ctx)` (`agent.go:2042`). `agent.Call(ctx, "other-node.reasoner", input)` (`agent.go:1570`) forwards all correlation + DID headers automatically.

> See [`research/docs/2026-04-20-sdk-patterns.md`](2026-04-20-sdk-patterns.md) for a complete walk-through of both SDKs, `af init` templates, and `af run` env-var injection.

#### 1.2 `af init` templates

Located at `control-plane/internal/templates/{python,go,typescript}/`. `templates.go:52-61` lists the template files. Each shipped template already includes an `echo` reasoner. For Copilot we can either:
- Provide a fourth template (`control-plane/internal/templates/copilot/`) generated by `af init my-copilot --language=copilot`, **or**
- Ship a standalone built-in agent node definition that the server registers automatically (similar to how the server already provides `/api/v1/nodes/register` and the Go SDK can register itself on startup).

The simpler ergonomics are "`af init copilot-agent`" → generates a ready-to-run Python project that wraps `copilot -p`.

#### 1.3 Skillkit — the right hook for "ship Copilot skills with AgentField"

AgentField's skillkit ([`control-plane/internal/skillkit/`](../../control-plane/internal/skillkit/)) is explicitly an **install-time Markdown distribution mechanism**. `Skill` struct at `skillkit/catalog.go:13-19`:

```go
type Skill struct {
    Name        string  // e.g. "agentfield-multi-reasoner-builder"
    Version     string
    Description string
    EmbedRoot   string  // "skill_data/<name>"
    EntryFile   string  // "SKILL.md"
}
```

Install pipeline (`skillkit/install.go:47-153`):
1. Resolve skill from `Catalog` by name.
2. `writeCanonical()` extracts embedded files to `~/.agentfield/skills/<name>/<version>/`.
3. `updateCurrentLink()` creates `~/.agentfield/skills/<name>/current → ./<version>/`.
4. For each selected target, call `t.Install(skill, currentLink)`.
5. Persist `~/.agentfield/skills/.state.json`.

Target implementations (one file each under `control-plane/internal/skillkit/`):
- `target_claude_code.go` — symlinks `~/.claude/skills/<name>` → canonical `current/`.
- `target_codex.go` — appends a `<!-- agentfield-skill:<name> -->` marker block to `~/.codex/AGENTS.override.md`.
- `target_gemini.go`, `target_opencode.go`, `target_aider.go`, `target_windsurf.go` — marker-block variants.
- `target_cursor.go` — "manual" mode: prints paste-in instructions.

CLI surface for skills is already complete (`control-plane/internal/cli/skill.go`):

| Command | Behavior |
|---|---|
| `af skill install [name]` | `--skill`, `--version`, `--target`, `--all`, `--all-targets`, `--force`, `--dry-run` |
| `af skill list` | Read `.state.json` |
| `af skill update [name]` | Re-install at embedded version with `Force: true` |
| `af skill uninstall [name]` | `--remove-canonical` also deletes `~/.agentfield/skills/<name>/` |
| `af skill print [name]` | Stdout the `SKILL.md` |
| `af skill path` | Print `~/.agentfield/skills` |
| `af skill catalog` | List skills embedded in the binary |

The installer's Phase-2 hook (`scripts/install.sh:536-568`) already calls `"$af_bin" skill install --all-targets` — adding Copilot to `detectedTargets()` means it ships automatically.

Currently only one skill is embedded: `skills/agentfield-multi-reasoner-builder/` (with `SKILL.md` and `references/*.md`). Adding new skills requires (a) `skills/<name>/`, (b) running `scripts/sync-embedded-skills.sh` (which uses `rsync --delete`), (c) adding a `go:embed` directive in `skillkit/embed.go`, (d) adding a `Catalog` entry.

> See [`research/docs/2026-04-20-install-skills.md`](2026-04-20-install-skills.md) *(also captured inline in the install-skills-flow sub-agent report — see [§5](#5-agent-reports) below)*.

#### 1.4 Authentication model

**Control-plane auth** ([`control-plane/internal/server/middleware/auth.go`](../../control-plane/internal/server/middleware/auth.go), full walk-through in `research/docs/2026-04-20-auth-model.md`):
- Single static API key on all routes; accepted as `X-API-Key`, `Authorization: Bearer`, or `?api_key=`.
- Empty key disables auth (`auth.go:26-29`). Sourced from `AGENTFIELD_API_KEY` or `api.auth.api_key` YAML.
- Optional DID-based signature layer (`DIDAuthMiddleware`, off by default).

**Agent → control-plane** (Python `agent.py:606-608`, Go `agent.go:415`): agent registers with `X-API-Key` on startup, receives an `IdentityPackage` (Ed25519 JWK) from `POST /api/v1/did/register`, then signs subsequent requests with `X-Caller-DID` + `X-DID-Signature` headers.

**No external secrets store exists.** External creds (OpenAI, GitHub, Copilot) are consumed by agent processes directly from **their own environment variables** — which is exactly how Copilot CLI expects to find `COPILOT_GITHUB_TOKEN` / `GH_TOKEN`. This is the integration surface for auth reuse.

`af run <agent>` injects a minimal env (`agent_service.go:452`: `PORT`, `AGENTFIELD_SERVER_URL`, and inherits parent env). So the Copilot agent, when launched by `af run`, will naturally inherit the host user's `GH_TOKEN` / `COPILOT_GITHUB_TOKEN` / `gh` cached auth.

#### 1.5 MCP — removed

`control-plane/internal/mcp/` and all related SDK modules were removed in commit `f732ed5e` (2026-04-07, "refactor: remove all MCP code from codebase (#359)"). The `test-mcp-endpoints.sh` scripts at `control-plane/scripts/` reference endpoints that no longer exist. **Do not plan any Copilot integration via MCP.**

---

### 2. GitHub Copilot CLI surface (external)

Full report with inline doc links: [`research/docs/2026-04-20-copilot-cli-external-research.md`](2026-04-20-copilot-cli-external-research.md). Verified against `copilot --help` v1.0.34 installed at `~/.local/bin/copilot`.

Key facts that shape the integration:

| Facet | Mechanism | Docs |
|---|---|---|
| **One-shot invocation** | `copilot -p "<prompt>" --allow-all-tools --log-level none` prints model output to stdout and exits 0. Stdin mode: `cat prompt.md \| copilot -p -`. | [programmatic interface](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli#programmatic-interface) |
| **JSON output** | `--json` flag emits a single NDJSON stream (turns, tool calls, token usage). | [CLI reference](https://docs.github.com/en/free-pro-team@latest/copilot/reference/copilot-cli-reference/cli-command-reference) |
| **Auth precedence** | `COPILOT_GITHUB_TOKEN` > `GH_TOKEN` > `GITHUB_TOKEN` > `gh` CLI cache > OS keychain entry `copilot-cli` > `~/.copilot/config.json`. | see external report §2 |
| **Token type** | Fine-grained PAT with **"Copilot Requests"** permission. Classic `ghp_` tokens **not supported**. | ibid |
| **State dir** | `COPILOT_HOME` (default `~/.copilot/`) contains `config.json`, `sessions/`, `skills/`, `plugins/`. `--config-dir` is an alias. | ibid |
| **Skills** | User-level skills: `~/.copilot/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`). Same open-standard shape as Claude Code. | [agentskills spec](https://github.com/agentskills/agentskills) |
| **Plugins** | `copilot plugin install <dir>`; a plugin is `plugin.json` + optional `skills/`, `agents/`, `hooks.json`, `.mcp.json`. One-shot alternative: `copilot --plugin-dir ./pkg`. | ibid |
| **Sandboxing** | Copilot restricts file writes to its `--cwd` (default: process cwd). Pass `--cwd /agent/workdir`. | ibid |
| **Observability** | `OTEL_EXPORTER_OTLP_ENDPOINT` or `COPILOT_OTEL_FILE_EXPORTER_PATH` emit OTel spans for every LLM call and tool use. | ibid |

---

### 3. Integration design

### 2.1 Copilot-as-agent

**Goal:** a new agent node `copilot` (language: Python, bundled) exposing reasoners such as `ask`, `review`, `plan`, `run_task` that are thin wrappers around `copilot -p`.

**Directory (proposed):** `examples/copilot-agent/` scaffolded by `af init copilot-agent` *or* a first-party tree at `sdk/python/examples/copilot/`. Code shape:

```python
# main.py (illustrative)
import asyncio, json, os, shlex, subprocess
from agentfield import Agent, ExecutionContext

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "copilot"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
)

COPILOT_BIN = os.getenv("COPILOT_BIN", "copilot")

async def _run_copilot(prompt: str, cwd: str, ctx: ExecutionContext) -> dict:
    # Per-run state isolation → own COPILOT_HOME keyed on run_id.
    home = os.path.join(os.getenv("AGENTFIELD_HOME", "~/.agentfield"),
                        "copilot-home", ctx.run_id or "default")
    os.makedirs(home, exist_ok=True)
    env = {**os.environ, "COPILOT_HOME": home}
    # Token inheritance: COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN > gh auth cache.
    cmd = [COPILOT_BIN, "-p", prompt, "--allow-all-tools",
           "--log-level", "none", "--json", "--cwd", cwd]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env)
    out, err = await proc.communicate()
    turns = [json.loads(l) for l in out.splitlines() if l.strip()]
    return {"turns": turns, "exit_code": proc.returncode,
            "stderr": err.decode("utf-8", "replace")}

@app.reasoner()
async def ask(prompt: str, cwd: str = None,
              execution_context: ExecutionContext = None) -> dict:
    return await _run_copilot(prompt, cwd or os.getcwd(), execution_context)

# Additional thin wrappers: review(diff), plan(task), run_task(task), apply_patch(...)
```

Key design points:
- **Session-per-run**: set `COPILOT_HOME=$AGENTFIELD_HOME/copilot-home/<run_id>` so concurrent runs don't trample each other's `sessions/` dir.
- **CWD discipline**: pass `--cwd` from input; never inherit the agent-server cwd.
- **Output shape**: return the raw NDJSON turns so upstream workflows can do their own summarization. Add a `summary` field by joining assistant text.
- **Timeouts / cancel**: implement a `timeout` input (default 300 s); use `asyncio.wait_for`.
- **DID-signed cross-agent calls**: automatic — the SDK injects headers; Copilot calls itself via `agent.Call("copilot.ask", ...)`.

Alternative surface: a **Go** implementation under `sdk/go/examples/copilot/` for users who want a single static binary. Same shape (`RegisterReasoner`, `os/exec`).

### 2.2 Skills installation

**Goal:** when users run `curl ... install.sh | bash`, the AgentField `SKILL.md` for "multi-reasoner builder" (and any future agent-facing skills) gets installed into Copilot CLI alongside Claude Code / Codex / Gemini.

**Change set (illustrative):**

1. **New target** at `control-plane/internal/skillkit/target_copilot.go`:
   - Detection: `~/.copilot/` exists *or* `copilot` binary on PATH (`exec.LookPath("copilot")`).
   - Install method: symlink `~/.copilot/skills/<name>` → canonical `~/.agentfield/skills/<name>/current/`. This is the cleanest option because Copilot treats `~/.copilot/skills/*/SKILL.md` as first-class user skills.
   - State: record `method: "symlink"`, `path: "~/.copilot/skills/<name>"` in `.state.json`.
2. **Register target** in `skillkit/install.go` (the `detectedTargets` / target registry).
3. **Doctor probe**: add a `copilot` CLI entry to `control-plane/internal/cli/doctor.go:90-133` so `af doctor --json` tells consumers whether Copilot is present.
4. **Install script**: no change needed — `scripts/install.sh:556` already runs `af skill install --all-targets`.

Plugin option (future): bundle AgentField's skills as a Copilot CLI *plugin* at `~/.agentfield/copilot-plugin/` (`plugin.json` + symlinks into canonical skill dirs) and run `copilot plugin install ~/.agentfield/copilot-plugin` from the target handler. Gives us one registration point rather than N per-skill symlinks, and a clean uninstall (`copilot plugin uninstall agentfield`).

### 2.3 Authentication reuse

**Default path (zero config):** Copilot CLI already checks `COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN`, and falls back to `gh` CLI's stored token. `af run copilot` spawns the agent as a child process which inherits the user's env, so any of those is picked up automatically.

**Enhanced path (explicit, per-agent):** allow the Copilot agent's config (or `af run copilot --env COPILOT_GITHUB_TOKEN=…`) to set a dedicated token without leaking it to sibling agents. Concretely, extend `af run` (`control-plane/internal/services/agent_service.go:452`) to accept `--env KEY=VAL` pass-through flags (the spec work should confirm this doesn't already exist in some form).

**Explicit non-goal:** storing Copilot tokens inside AgentField. The research confirms there is no secrets store, and design philosophy (`research/docs/2026-04-20-auth-model.md`) keeps external creds in the agent's env. We should preserve that invariant.

**Doctor integration:** `af doctor` already probes `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc. Extend it to report:
- Whether `copilot` is on PATH and its version (`copilot --version`).
- Whether `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` is set, *or* whether `gh auth status` reports an authenticated user.
- Location of `~/.copilot/config.json` and whether it contains a `lastLoggedInUser`.

---

## Code References

- [`control-plane/internal/skillkit/catalog.go:13-19`](../../control-plane/internal/skillkit/catalog.go) — `Skill` struct (target for Copilot target additions)
- [`control-plane/internal/skillkit/install.go:47-153`](../../control-plane/internal/skillkit/install.go) — install pipeline; add `target_copilot` to registry
- `control-plane/internal/skillkit/target_claude_code.go` — reference pattern for a new `target_copilot.go`
- `control-plane/internal/skillkit/target_codex.go` — alternative (marker-block) pattern
- `control-plane/internal/skillkit/embed.go:34-36` — `go:embed` entries for skill data
- [`control-plane/internal/cli/skill.go:23-57`](../../control-plane/internal/cli/skill.go) — CLI surface (no changes needed; generic over targets)
- [`control-plane/internal/cli/doctor.go:90-133`](../../control-plane/internal/cli/doctor.go) — add Copilot probes
- [`scripts/install.sh:536-568`](../../scripts/install.sh) — Phase-2 skill install hook (no changes needed)
- [`scripts/sync-embedded-skills.sh:26-28`](../../scripts/sync-embedded-skills.sh) — `SKILLS=(…)` list, add new skills here
- `skills/agentfield-multi-reasoner-builder/SKILL.md` — existing skill format reference
- [`sdk/python/agentfield/agent.py:1693,1843,606-608`](../../sdk/python/agentfield/agent.py) — `@app.reasoner()`, `ExecutionContext.from_request`, URL resolution
- [`sdk/go/agent/agent.go:~522,2042,1570`](../../sdk/go/agent/agent.go) — `RegisterReasoner`, `ExecutionContextFrom`, `Call`
- [`control-plane/internal/templates/python/`](../../control-plane/internal/templates/python/) — add `copilot/` sibling if we want `af init --language=copilot`
- [`control-plane/internal/services/agent_service.go:452`](../../control-plane/internal/services/agent_service.go) — env-var injection point for `af run`
- [`control-plane/internal/server/middleware/auth.go:26-29,38-69`](../../control-plane/internal/server/middleware/auth.go) — API-key auth, no changes needed
- `docs/CONTRIBUTING.md:101` — the only pre-existing mention of Copilot

## Architecture Documentation

- **SDK convention:** reasoners are pure async functions; subprocess wrapping is already the pattern for anything that isn't a network call. See `sdk/python/agentfield/ai/` (LLM helpers) for the precedent of shelling out to external services.
- **Skillkit convention:** each target is a separate Go file implementing the `Target` interface; state is a JSON blob at `~/.agentfield/skills/.state.json` with per-skill per-target records. Adding Copilot is additive and follows the existing pattern exactly.
- **No-secrets-store invariant:** external credentials are *always* read from the agent process's env. Anything proposing to store Copilot tokens inside AgentField is off-pattern.
- **MCP is gone:** do not plan integration through MCP; it was removed in `f732ed5e`.
- **Doctor is machine-consumable:** `af doctor --json` is the probe layer; new capabilities should have a doctor entry.

## Historical Context (from research/)

- [`research/docs/2026-04-20-sdk-patterns.md`](2026-04-20-sdk-patterns.md) — deep-dive on Python + Go agent construction, `af init` templates, `af run` env injection.
- [`research/docs/2026-04-20-auth-model.md`](2026-04-20-auth-model.md) — full auth layering (API key, DID/VC, connector token, admin token) and confirmation of the no-secrets-store design.
- [`research/docs/2026-04-20-copilot-cli-external-research.md`](2026-04-20-copilot-cli-external-research.md) — GitHub Copilot CLI v1.0.34 surface: `-p`, `--json`, `COPILOT_HOME`, plugins, skills, auth precedence, OTel hooks.
- **install-skills-flow** sub-agent report (content in §5 below, not yet materialized as a separate file) — skillkit internals and CLI surface.

## Open Questions

1. **Distribution of the Copilot agent**: first-party tree under `sdk/python/examples/copilot/`, or a separate repo (e.g. `agentfield-copilot-agent`), or a new `af init --language=copilot` template, or a bundled node auto-registered by the server? The repo seems to prefer `examples/` for third-party/user-facing flows and `sdk/.../examples/` for SDK showcases. Recommend: **both** — a scaffold template for local authoring, and an optional built-in mode.
2. **Session continuity across calls**: Copilot CLI supports `--resume <session-id>` in v1.0.34. Should we bind `ExecutionContext.SessionID` → Copilot `--resume`? That would give us multi-turn reasoning per AgentField session. Needs verification against the `--help` output (the external report notes it, but we should confirm the flag name).
3. **Per-skill vs per-plugin install strategy for Copilot target**: per-skill symlinks are simpler; a single plugin is more discoverable. Choose after a quick prototype.
4. **Output schema stability**: Copilot's `--json` NDJSON format isn't versioned. Do we want to post-process into a stable AgentField-native schema (e.g. `{turns, summary, usage, tool_calls}`) and keep the raw NDJSON as a side channel? Recommend: yes, add a versioned adapter.
5. **OTel pass-through**: AgentField's control plane has its own execution tracing. Should Copilot's OTel traces be ingested (via `OTEL_EXPORTER_OTLP_ENDPOINT` pointed at the control plane), or kept separate? Out of scope for first integration.
6. **Does `af run` currently accept `--env KEY=VAL` pass-through?** The research didn't verify this; the spec/plan stage should check `control-plane/internal/cli/commands/run.go`.

---

## 5. Agent reports

Full sub-agent outputs for reference:

- [`research/docs/2026-04-20-copilot-cli-external-research.md`](2026-04-20-copilot-cli-external-research.md) — external (Copilot CLI).
- [`research/docs/2026-04-20-sdk-patterns.md`](2026-04-20-sdk-patterns.md) — SDK agent patterns.
- [`research/docs/2026-04-20-auth-model.md`](2026-04-20-auth-model.md) — auth model.
- Install / skills flow details are folded into this synthesis (§1.3 and Code References); if a standalone file is desired later, the raw content is reproducible from the `install-skills-flow` sub-agent.
