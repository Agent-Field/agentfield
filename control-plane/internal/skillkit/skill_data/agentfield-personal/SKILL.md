---
name: agentfield-personal
version: 0.1.0
description: "Build and install a personal AI agent on this machine's AgentField: real source in ~/agentfield-agents, packaged with agentfield-package.yaml, installed with `af install`, started with `af run`, registered on the local control plane, and visible in AgentField Desktop with a keys form and an auto-start toggle. Use when the user wants an agent that lives on their machine as a persistent capability — a pricing agent, a support agent, a research agent — rather than a deployable project. A standalone repository with Docker Compose is the `agentfield` skill; calling agents that already exist is the `agentfield-use` skill."
---

# Building a personal AgentField agent

A personal agent is a capability installed on this machine. Once it's running,
the local control plane routes calls to it, other agents and coding assistants
can discover and delegate to it, and the AgentField Desktop app shows it with
its keys and lifecycle controls. The deliverable is not a repository — it is a
working, registered, callable agent.

This skill is the workflow for getting that done. It does not use Docker,
Docker Compose, a new Git repository, or a project `CLAUDE.md` unless the user
independently asks for one of those.

## Before building

Check once whether an installed agent already covers the request: `af list`
for what's installed, and the control plane's discovery
(`GET /api/v1/discovery/capabilities`) for what each running agent's reasoners
actually do (the `agentfield-use` skill documents this surface). If a healthy
installed agent already does the job, say so and offer to use it instead of
building a duplicate — unless the user explicitly asked to build a new or
replacement agent, in which case build it. A stopped-but-capable installation
is not a reason to duplicate either; offer to start it with `af run <name>`.

For the agent's design, fetch the live SDK docs first —
`https://agentfield.ai/llms.txt` (and `llms-full.txt` for depth) — that is the
SDK ground truth. Decompose the job into reasoners the same way the
`agentfield` skill teaches: by cognitive jobs, not by a single catch-all
prompt. Personal agents are usually small — a handful of reasoners on one node
is normal — but the design bar is the same.

## Workflow

1. **Build stable real source.** Choose one filesystem-safe kebab-case
   package/name/node ID, `<name>`, and author the agent at
   `~/agentfield-agents/<name>`. This directory is the durable source of truth
   the user will edit later. Do not author in a temporary directory, a
   disposable checkout, or the generated `~/.agentfield` installation copy.
   Run language-native syntax checks and tests on the source before
   installing.

2. **Package the source.** Write the manifest at
   `~/agentfield-agents/<name>/agentfield-package.yaml`. Put
   `config_version: v1` at the top — the manifest schema version, distinct
   from the agent release `version`. Declare `name`, release `version`,
   `description`, `author`, `language`, a runnable `entrypoint.start` that
   matches the source and language, `entrypoint.healthcheck: /health`,
   `agent_node.node_id` equal to `<name>`, its matching
   `agent_node.default_port`, and only install dependencies the source needs.

   ```yaml
   config_version: v1
   name: pricing-agent
   version: 0.1.0
   description: Answers pricing questions from the product catalog
   author: <user>
   language: python
   entrypoint:
     start: python main.py
     healthcheck: /health
   agent_node:
     node_id: pricing-agent
     default_port: 9301
   dependencies:
     python: [requests]
   user_environment:
     - name: OPENROUTER_API_KEY
       description: LLM provider key used for all reasoning calls
       type: secret
       scope: global
   ```

3. **Declare secrets safely.** For every external key the source actually
   uses, declare a `user_environment` entry with `name`, an actionable
   `description`, `type: secret`, and an explicit scope. Use `scope: global`
   only for deliberately reusable credentials such as a model-provider key;
   use `scope: node` for credentials or configuration specific to this agent.
   Do not declare invented keys.

4. **Install and configure.** Run `af install ~/agentfield-agents/<name>`.
   Configure each declared global key with `af secrets set KEY` and each node
   key with `af secrets set --node <name> KEY`, letting the CLI prompt/stdin
   take the value. Never invent, echo, commit, put into
   `agentfield-package.yaml`, or include secret values in a handoff.

5. **Start and verify registration.** Run `af run <name>`, then poll
   `GET ${AGENTFIELD_SERVER:-http://localhost:8080}/api/v1/nodes` until the
   node ID is registered in an active/healthy state. An install entry, `af
   list` entry, or successful process spawn alone is not success.

6. **Invoke live.** Invoke the public entry reasoner through the control plane
   with a representative request. For nontrivial work use async execution and
   poll (the `agentfield-use` skill documents the execute/poll surface);
   require a terminal successful result before calling the build done.

7. **Handle failures honestly.** Diagnose and safely retry correctable
   failures from installation, secret setup, startup, registration, or
   invocation (`af logs <name>` is the first stop). If a required secret value
   is known only to the user, stop with a blocking handoff that names the
   needed key and scope but never its value. Do not claim completion until
   healthy registration and a live reasoner result both succeed.

8. **Hand off.** Tell the user the agent is installed, running, and now
   appears in the AgentField Desktop app, where its declared keys are
   presented as a form and its lifecycle has an auto-start toggle. Include:
   the stable source path, the manifest path, the installed name, the public
   entry reasoner's invocation target, the registration and live-call
   verification results, and the commands to restart
   (`af stop <name> && af run <name>`), stop (`af stop <name>`), inspect logs
   (`af logs <name>`), and update after source edits
   (`af install ~/agentfield-agents/<name>` followed by `af run <name>`).
