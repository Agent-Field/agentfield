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

Everything lives under `~/.agentfield/` (override with `AGENTFIELD_HOME`):

```
~/.agentfield/
├── installed.yaml          # registry of installed nodes + runtime state
├── packages/<node>/        # the installed node + its Python venv
├── logs/<node>.log         # process logs (af logs <node>)
├── keyring/master.key      # 0600 — local key that decrypts your secrets
└── secrets/
    ├── global.enc          # secrets shared across all nodes (encrypted)
    └── <node>.enc          # secrets scoped to one node (encrypted)
```

## The manifest: `agentfield-package.yaml`

Every installable node has this file at its repo root. Only `name`, `version`,
and a way to start (either `entrypoint.start` or a top-level `main.py`) are
required; everything else is optional.

```yaml
name: pr-af
version: 0.1.0
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
  required:
    - name: OPENROUTER_API_KEY
      description: LLM provider key
      type: secret
      scope: global           # global (default) = shared across nodes; node = this node only
  optional:
    - name: PR_AF_MODEL
      description: Override the default model
      default: openrouter/moonshotai/kimi-k2
```

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
| `af run <node>`             | Start a node (and its node deps) in the background              |
| `af list`                   | Show installed nodes and runtime state                         |
| `af logs <node>`            | Tail a node's process log                                      |
| `af stop <node>`            | Stop a running node                                            |
| `af uninstall <node>`       | Stop and remove a node                                         |
| `af secrets set/ls/rm`      | Manage the encrypted secret store                              |
