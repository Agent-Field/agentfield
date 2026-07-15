# Installing & Running Agent Nodes

`af install` and `af run` turn a published agent-node repository into a locally
running agent that connects to your control plane. Nodes are described by an
`agentfield-package.yaml` manifest, their dependencies (Python packages **and**
other agent nodes) are resolved automatically, and the secrets they need are
stored encrypted and injected only at runtime.

```bash
# Install a node straight from GitHub (pulls in its node dependencies too)
af install https://github.com/Agent-Field/pr-af

# Start it — you'll be prompted once for any missing required secrets
af run pr-af
```

Nodes can be written in **Python** (venv + pip) or **Go** (compiled binary).
`af install`/`af run` pick the right toolchain per node; the port, health check,
secret injection, and control-plane wiring are identical either way. See
[Language: Python or Go](#language-python-or-go) for the Go specifics.

Everything lives under `~/.agentfield/` (override with `AGENTFIELD_HOME`):

```
~/.agentfield/
├── installed.yaml          # registry of installed nodes + runtime state
├── packages/<node>/        # the installed node + its Python venv or built Go binary
├── logs/<node>.log         # process logs (af logs <node>)
├── keyring/master.key      # 0600 — local key that decrypts your secrets
└── secrets/
    ├── global.enc          # secrets shared across all nodes (encrypted)
    └── <node>.enc          # secrets scoped to one node (encrypted)
```

## The manifest: `agentfield-package.yaml`

Every installable node has this file at its repo root. Only `name`, `version`,
and a way to start (an `entrypoint.start`, a top-level `main.py` for a Python
node, or a `go.mod` for a Go node) are required; everything else is optional.

```yaml
config_version: v1            # manifest *schema* version (see below). Omit = v0 (legacy).
name: pr-af
version: 0.1.0                # the node's own release version — unrelated to config_version
description: Opens draft PRs from a task description
author: Agent-Field

# How to launch the node. The first token is run inside the node's venv when it
# is "python"/"python3". Omit this only if the repo has a top-level main.py.
entrypoint:
  start: python -m pr_af.app
  healthcheck: /health        # polled after launch to confirm readiness

agent_node:
  node_id: pr-af
  default_port: 8004

dependencies:
  python:                     # extra pip installs (requirements.txt is also honored)
    - httpx>=0.27
  nodes:                      # other agent nodes this one calls — installed recursively
    - af://registry/swe-planner

# Variables the node needs. Required ones are prompted for on first run and
# remembered (encrypted). type: secret hides the input and stores it encrypted.
user_environment:
  required:                     # every one of these must be set
    - name: GH_TOKEN
      description: GitHub token
      type: secret
      scope: global           # global (default) = shared across nodes; node = this node only
  require_one_of:               # each group needs AT LEAST ONE option set
    - id: llm_provider
      description: an LLM provider key
      options:
        - name: ANTHROPIC_API_KEY
          description: Anthropic key (Claude)
          type: secret
        - name: OPENROUTER_API_KEY
          description: OpenRouter key (DeepSeek/Qwen/Llama/…)
          type: secret
  optional:
    - name: PR_AF_MODEL
      description: Override the default model
      default: openrouter/moonshotai/kimi-k2
```

A published, real-world manifest to copy from:
[Agent-Field/SWE-AF `agentfield-package.yaml`](https://github.com/Agent-Field/SWE-AF/blob/main/agentfield-package.yaml).

### Manifest schema version (`config_version`)

`config_version` declares which version of the **manifest format** your file was
written against, so the control plane knows how to read it as the format evolves —
you're never locked into whatever shape shipped the day you authored the file.

- It is **not** the same as `version`. `version` is your node's own release
  (semver of the agent); `config_version` is the schema version of this file.
- **Omitting it means `v0`** — the original, pre-versioning format. Existing
  manifests keep working untouched.
- The current version is **`v1`**. New manifests should set `config_version: v1`.
- The `v` prefix is optional and case-insensitive (`v1`, `V1`, and `1` are equal).
  A value the control plane doesn't recognize (a typo, or a version **newer** than
  your `af` binary understands) fails the install with a clear message rather than
  being silently mis-read — upgrade `af` to install a node authored for a newer
  schema.

**When does `config_version` get bumped?** Only for **breaking** changes to the
format — a field renamed or removed, or its shape/meaning changed such that an old
reader would mis-handle a new file (or vice-versa). **Adding** a new optional field
is *not* breaking and does **not** bump the version: unknown keys are ignored by
older readers, and newer readers fall back to defaults. So most format growth needs
no bump at all; you only stamp a new `config_version` when the structure of an
existing config actually changes.

| `config_version` | Reader behavior                                             |
| ---------------- | ---------------------------------------------------------- |
| absent / `v0`    | Legacy format, read leniently. Every field below is optional except the manifest basics. |
| `v1`             | Same fields as v0, now explicitly versioned. Current default for new manifests. Later *additive* keys (e.g. `language`, `entrypoint.build` for Go nodes) live here too — they did not bump the version. |

### `require_one_of` — "at least one of these"

Some nodes accept alternatives — e.g. either an Anthropic key **or** an
OpenRouter key. List them under `require_one_of` as a group of `options`. A group
is satisfied as soon as **one** option resolves (from the process environment,
the secret store, or a manifest default).

On `af run`, if no option of a group is set, you're asked to fill in one and
leave the rest blank — the value you enter is validated and stored encrypted,
exactly like a required secret. In a non-interactive session an unsatisfied group
is a clean error listing the alternatives (`at least one of [ANTHROPIC_API_KEY |
OPENROUTER_API_KEY] is required`) instead of a runtime failure inside the node.

`required` (all must be set), `require_one_of` (at least one per group), and
`optional` (falls back to `default`) can all be used together in one manifest.

### Python dependencies

On install, a node's Python dependencies are installed into a per-node virtual
environment under `~/.agentfield/packages/<node>/venv`. Sources are honored in
order: `requirements.txt`, then `pip install .` for a `pyproject.toml`/`setup.py`
project, then any packages listed under `dependencies.python` in the manifest.
`af run` uses this venv automatically.

The venv is built with the `python3`/`python` on your `PATH`. If a node declares
`requires-python` (e.g. `>=3.11`) that your interpreter doesn't satisfy, `pip`
reports it and install fails — point `af` at a compatible interpreter (e.g. via
`pyenv`/`PATH`) and reinstall.

### Language: Python or Go

A node's implementation language is set by the optional top-level `language`
field: `python` (the default) or `go`. When `language` is omitted, `af` detects
a Go node by the presence of a `go.mod` at the package root; anything else is
treated as Python. Existing Python manifests need no changes.

```yaml
language: go
entrypoint:
  build: ./cmd/swe-planner   # Go package to compile at install time
  start: bin/swe-planner     # resulting binary, launched by `af run`
  healthcheck: /health
```

At install time a Go node is **compiled**, not pip-installed:

- The `go` toolchain is discovered on `PATH`. A missing `go` is an actionable
  error (how to install it); a `go` older than the module's `go.mod` directive is
  refused with an upgrade hint — the Go analogue of the `requires-python` check.
- With `entrypoint.build` set, `af` runs `go build -o <start> <build>`, leaving a
  runnable binary at the `entrypoint.start` path. `af run` launches that binary
  directly — same `PORT`, health check, secrets, and control-plane env as a
  Python node.
- Alternatively, use a `go run` entrypoint (`start: go run ./cmd/swe-planner`)
  or omit `entrypoint.start` entirely (defaults to `go run .`). Install then only
  compile-checks (`go build ./...`) and the binary is built on launch. This is
  simpler but recompiles each start, so a prebuilt binary is preferred for large
  nodes.

A Go node reads the same runtime environment as a Python node — `PORT`,
`AGENTFIELD_SERVER`, and any declared `user_environment` secrets are injected
into the process identically, and readiness is confirmed by polling
`entrypoint.healthcheck`.

#### `replace` directives and vendoring

Go modules that use a **local `replace` directive pointing outside the package**
(e.g. `replace example.com/sdk => ../../other/sdk`, as the SWE-AF Go port does for
the AgentField Go SDK) will not build after install, because the node is copied
into `~/.agentfield/packages/<node>/` and the relative path no longer resolves.
`af install` detects this and refuses with guidance rather than a confusing raw
build failure. Fix it one of these ways:

- **Vendor the module** (recommended): run `go mod vendor` in the node repo and
  commit the `vendor/` directory. It ships with the package, so the build is
  hermetic (`go build -mod=vendor`) regardless of `replace` targets.
- **Publish/tag the replaced module** and use a versioned `require` instead of a
  local `replace`.
- **Override at install time** with `AGENTFIELD_GO_REPLACE` — a comma-separated
  list of `go mod edit -replace` specs applied before building, e.g.
  `AGENTFIELD_GO_REPLACE="example.com/sdk=/abs/path/to/sdk" af install <src>`.

In-tree replaces (pointing inside the package) and module-version replaces are
always fine.

### Node dependencies

`dependencies.nodes` lets one node declare that it needs others. Each entry is
an installable reference:

| Reference                                   | Resolves to                                   |
| ------------------------------------------- | --------------------------------------------- |
| `af://registry/<name>`                      | `https://github.com/Agent-Field/<name>`       |
| `af://registry/<name>@<version>`            | same (version constraint is recorded, not yet enforced) |
| `https://github.com/<org>/<repo>`           | used as-is                                     |

`af install <node>` installs the node **and** any declared node dependencies it
doesn't already have, recursively. Already-installed nodes are skipped, which
also breaks dependency cycles. `af run <node>` starts a node's installed
dependencies first (in dependency order) before the node itself.

## One repo, many nodes: `--path`

By default `af install <src>` looks for the `agentfield-package.yaml` at the root
of the source (a git repo or a local directory). When a single repository ships
more than one installable node — for example a Python node at the root and a Go
port under `go/` — use `--path` to select the subdirectory to install:

```bash
# Install the node whose manifest lives at go/agentfield-package.yaml
af install https://github.com/Agent-Field/SWE-AF --path go

# Composes with an @ref pin on the URL
af install https://github.com/Agent-Field/SWE-AF@v1.2.3 --path go

# Also works for a local source
af install ./SWE-AF --path go
```

The subdirectory must contain its own `agentfield-package.yaml`; that subtree
becomes the package root that is copied to `~/.agentfield/packages/<name>` and
installed (a Go node builds relative to it). Because registry entries are keyed by
the manifest `name`, the root node and a `--path` node from the same repo coexist
as separate installs. `--path` is a path **relative to the source root**: absolute
paths and paths that escape the root with `..` are rejected, and a missing manifest
is reported with the full expected path. A bare `af install <src>` (no `--path`) is
unchanged — the root manifest is always what you get by default.

## Secrets: encrypted, shared, runtime-only

Secrets are never written to disk in plaintext and never baked into the package.
They are encrypted with AES-256-GCM under a random 32-byte key kept in
`~/.agentfield/keyring/master.key` (mode `0600`). At start time they are decrypted
straight into the child process' environment — nowhere else.

When `af run` needs a required variable, it resolves it in this order:

1. **Process environment** — if `OPENROUTER_API_KEY` is already exported, it's
   used as-is and **not** persisted.
2. **Node-scoped store** (`secrets/<node>.enc`), then **global store**
   (`secrets/global.enc`).
3. **Manifest `default`**.
4. **Prompt** (required variables only, when attached to a terminal). The value
   is validated against the manifest `validation` regex, then saved encrypted to
   the variable's scope.

Because most provider keys are `scope: global`, you enter `OPENROUTER_API_KEY`
once and every node reuses it. A `scope: node` variable is stored per-node and
overrides the global value for that node.

### Managing secrets directly

```bash
af secrets set OPENROUTER_API_KEY            # prompts, hidden input, stored global
af secrets set GH_TOKEN ghp_xxx              # value inline
af secrets set PR_AF_MODEL ... --node pr-af  # node-scoped override
af secrets ls                                # keys + scope only (values never shown)
af secrets rm GH_TOKEN                        # remove from global
af secrets rm PR_AF_MODEL --node pr-af        # remove a node-scoped secret
```

In a non-interactive session (CI, no TTY), missing required secrets are reported
as an error listing the variable names instead of hanging on a prompt — set them
ahead of time with `af secrets set` or by exporting them.

## Control-plane connection

`af run` exports `AGENTFIELD_SERVER` (the variable the SDK reads) and the legacy
`AGENTFIELD_SERVER_URL`, plus the assigned `PORT`, into the node process. The
server URL is resolved from your local configuration.

## Lifecycle reference

| Command                     | Does                                                            |
| --------------------------- | -------------------------------------------------------------- |
| `af install <src>`          | Install from a local path, git URL, or registry name + node deps |
| `af install <src> --path <subdir>` | Install the node in `<subdir>` (one repo shipping multiple nodes) |
| `af run <node>`             | Start a node (and its node deps) in the background              |
| `af list`                   | Show installed nodes and runtime state                         |
| `af logs <node>`            | Tail a node's process log                                      |
| `af stop <node>`            | Stop a running node                                            |
| `af uninstall <node>`       | Stop and remove a node                                         |
| `af secrets set/ls/rm`      | Manage the encrypted secret store                              |
