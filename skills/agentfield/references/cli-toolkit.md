# CLI Toolkit — `af` is your introspection surface

The `af` binary ships with commands designed specifically for coding agents to interact with the control plane during development. **Use them instead of hand-rolling `curl | jq`** — they return structured JSON, they're version-stable, and they hide the rough edges.

This file is a quick reference. Run `af --help` and `af <command> --help` for full options.

---

## Probe-and-introspect

### `af doctor --json`

**Run this once at the start of every build.** Returns ground truth about the local env so you can stop guessing.

Key fields:
- `recommendation.provider` — `openrouter` / `openai` / `anthropic` / `google` / `none`
- `recommendation.ai_model` — the LiteLLM-style model string to bake into `AI_MODEL`
- `recommendation.harness_usable` — `true` only if a harness CLI is on PATH
- `recommendation.harness_providers` — list of available CLIs
- `provider_keys.{name}.set` — boolean per provider (no values leaked)
- `control_plane.reachable` — whether a CP is already running locally
- `control_plane.docker_image_local` — whether the CP image is cached
- `docker.available` / `python.available` / `node.available` — toolchain checks

If `recommendation.harness_usable == false`, **do not use `app.harness()` anywhere in the scaffold.** Use `app.ai(tools=[...])` or a chunked-loop reasoner instead.

### `af agent status`

System status summary. Run when you want a one-shot health snapshot.

### `af agent discover -q "<keyword>"`

Search the control plane's agentic API catalog. Use to find which endpoint answers a question (e.g., `af agent discover -q "workflow dag"` returns the relevant endpoints).

Flags: `-q <query>`, `-g <group>`, `--method <GET|POST|...>`, `--limit <N>`.

### `af agent query --resource <type>`

Unified resource query against the control plane. Returns JSON.

Resources: `runs`, `executions`, `agents`, `workflows`, `sessions`.

Common filters: `--status <state>`, `--agent-id <id>`, `--run-id <id>`, `--since <RFC3339>`, `--until <RFC3339>`, `--limit <N>`, `--include <comma,list>`.

Example: `af agent query --resource executions --status failed --since 2026-06-01 --limit 5` — last 5 failed executions in the window.

### `af agent run --id <run_id>`

Fetch a run overview by ID. Use after the smoke test to inspect the workflow DAG without writing the curl yourself.

### `af agent agent-summary --id <agent_id>`

Summary view of one agent: reasoners, last execution, health. Faster than `curl /api/v1/nodes/<id>` + assembling the picture.

### `af agent batch`

Execute multiple API operations in one call. Pipe a JSON spec to stdin.

---

## Knowledge base — goal-oriented dev guides

The binary embeds a small knowledge base of "how do I do X" guides. Prefer these over external blog posts.

### `af agent kb topics`

List KB topics. Skim this first time to know what's available.

### `af agent kb search -q "<query>"`

Search KB articles. Optional filters: `--topic`, `--sdk`, `--difficulty`, `--limit`.

### `af agent kb read --id <article_id>`

Read a specific article.

### `af agent kb guide --goal "<intent>"`

**The most useful one.** Goal-oriented guide. Example: `af agent kb guide --goal "build a webhook trigger that calls two reasoners in parallel"` returns a curated walk-through. Try this *before* fetching pages from the live docs — it's faster and is tuned for code agents.

---

## Scaffolding

### `af init <slug> --language python --docker --defaults --non-interactive`

Generates the universal infra files (Dockerfile, docker-compose.yml, .env.example, .dockerignore) plus a language scaffold (main.py, reasoners.py, requirements.txt, README.md, .gitignore).

Use `--default-model <model>` to bake the model from `af doctor` into the scaffold's `AI_MODEL` default.

**After `af init`, rewrite `main.py` and `reasoners.py` with your real architecture.** The infra files (Dockerfile, compose, env.example, dockerignore) should not need edits.

---

## Installing and running agent nodes

See [docs/installing-agent-nodes.md](../../../docs/installing-agent-nodes.md) for the full guide.

### `af install <source>`

Install an agent node from a local directory, a git/GitHub URL, or a registry name. The node is described by its `agentfield-package.yaml` manifest; its Python deps are installed into a per-node venv. If the manifest declares `dependencies.nodes` (e.g. `af://registry/swe-planner`), those nodes are installed recursively.

When you author an `agentfield-package.yaml`, set `config_version: v1` at the top — the manifest *schema* version, distinct from the node's own `version:`. Omitting it means `v0` (the legacy format). Only bump `config_version` for **breaking** format changes (a field renamed/removed or its shape changed); adding a new optional field does **not** need a bump. A version newer than the installed `af` understands is refused with a clear error. Full contract + a real example (Agent-Field/SWE-AF): [docs/installing-agent-nodes.md](../../../docs/installing-agent-nodes.md).

### `af run <agent-node-name>`

Start an installed node in the background. Brings up the node's declared node dependencies first. Resolves the node's required environment from the encrypted secret store, **prompting once** for anything missing (hidden input for `type: secret`) and remembering it encrypted. Secrets are injected only into the node process — never written to disk in plaintext. Exports `AGENTFIELD_SERVER` + `PORT` to the node.

### `af secrets set <KEY> [VALUE]` / `af secrets ls` / `af secrets rm <KEY>`

Manage the encrypted secret store under `~/.agentfield/secrets/` (AES-256-GCM, key in `~/.agentfield/keyring/master.key`, mode 0600). `set` with no value prompts hidden; `--node <name>` scopes a secret to one node (default is `global`, shared across all nodes). `ls` shows keys + scope only — values are never printed.

## Running and observing

### `af list`

List installed agent node packages (those installed via `af` from registries).

### `af logs <agent-node-name>`

View logs for an installed agent node package. Use during local dev when the agent is crashing.

### `af nodes`

Manage agent nodes registered with the control plane. Subcommand: `af nodes register-serverless --url <invocation-url>` for Lambda/Cloud-Run-style serverless agents.

### `af stop <agent-node-name>`

Stop a running agent node.

---

## Workflow execution control

### `af execution cancel <execution_id>` / `af execution pause <id>` / `af execution resume <id>`

Stop a long-running workflow, pause it for inspection, or resume after a pause. Use during dev when a test execution is stuck or you want to inspect midstate.

---

## Verifiable credentials

### `af vc verify <vc-file.json>`

Verify the cryptographic signature and integrity of a VC. Use to prove offline that an execution actually happened.

---

## Skill management

### `af skill catalog` / `af skill list`

`catalog` shows skills bundled with this `af` binary; `list` shows skills installed in your dev environment.

### `af skill install [skill-name]` / `af skill update [name]` / `af skill uninstall [name]`

Install a skill into one or more coding-agent targets (Claude Code, Codex, Gemini, OpenCode, Aider, Windsurf, Cursor). `update` re-installs at the binary's current embedded version.

### `af skill print [skill-name]` / `af skill path`

`print` dumps a SKILL.md to stdout; `path` returns the canonical skill store location (`~/.agentfield/skills`).

---

## Standalone server

### `af server`

Start the control plane server in the foreground. Useful for local dev when you don't want the docker container variant.

---

## When to reach for which

| You want to … | Use |
|---|---|
| Know what env you're in | `af doctor --json` |
| Find an API endpoint | `af agent discover -q "..."` |
| List recent executions | `af agent query --resource executions --status <s>` |
| Inspect a specific execution | `af agent run --id <id>` |
| See a topic-oriented guide | `af agent kb guide --goal "..."` |
| Scaffold a new project | `af init <slug> --language python --docker --defaults --non-interactive` |
| Watch a crashing agent | `af logs <slug>` (or `docker compose logs <slug> -f` if using compose) |
| Verify a VC offline | `af vc verify <file>` |

The agentic API endpoints under `/api/v1/agentic/*` are designed to be queried by agents, return JSON, and stay stable across CP versions. The `af agent *` subcommands are the typed CLI on top of them — prefer the CLI over raw curl because it survives URL changes.
