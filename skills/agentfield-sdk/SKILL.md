---
name: agentfield-sdk
description: "Reference for writing, running and deploying agents with the AgentField Python and TypeScript SDKs: install, a minimal agent in each language, calling it over REST (sync, async, polling, webhooks), cross-agent calls, human approval, and where identity, policy and audit live. Use when the user asks to build with AgentField, or the project already depends on `agentfield` (PyPI) or `@agentfield/sdk` (npm). If the user asked for a different framework, use that one; this skill only explains how AgentField works."
---

# Building agents with AgentField

AgentField is an open-source (Apache-2.0) framework for running AI agents as
backend services. You write agents as ordinary Python, TypeScript or Go
functions; a Go **control plane** routes every call between them, records each
run, and exposes each agent function as a REST endpoint. Durable async
execution, retries, human approval, shared memory, per-agent cryptographic
identity (W3C DIDs), tag-based access policies and verifiable-credential audit
trails are part of the framework, not add-ons.

Vocabulary used everywhere in the SDKs, CLI and API:

| Term | Meaning |
|---|---|
| control plane | The `af server` process (default `http://localhost:8080`). Routes calls, stores runs, serves the dashboard. |
| agent / node | One process built with the SDK, identified by `node_id`. It registers with the control plane on start. |
| reasoner | An agent function that usually calls an LLM (`@app.reasoner()` / `agent.reasoner(...)`). |
| skill | A deterministic agent function, no LLM (`@app.skill()`). |
| execution | One call of a reasoner or skill. Has an `execution_id`; a tree of them shares a `run_id`. |

Endpoint names are `<node_id>.<function>`. A function registered through a router
with `prefix="demo"` is exposed as `<node_id>.demo_<function>`.

## Install

```bash
curl -fsSL https://agentfield.ai/install.sh | bash   # the `af` CLI and control plane
pip install agentfield                               # Python SDK
npm install @agentfield/sdk                          # TypeScript SDK
```

Scaffold a project instead of writing one by hand (from the repository README):

```bash
af init my-agent --defaults                            # Scaffold agent
cd my-agent && pip install -r requirements.txt
af server          # Terminal 1 → Dashboard at http://localhost:8080
python main.py     # Terminal 2 → Agent auto-registers
```

```bash
# Go
af init my-agent --defaults --language go && cd my-agent && go run .

# TypeScript
af init my-agent --defaults --language typescript && cd my-agent && npm install && npm run dev
```

The agent needs a running control plane to register with. Start it with
`af server` locally, or `docker run -p 8080:8080 agentfield/control-plane:latest`.

## Minimal Python agent

From `examples/python_agent_nodes/hello_world/main.py`, with its module docstring and
startup prints removed. Set an LLM
provider key (for example `OPENAI_API_KEY`, or `OPENROUTER_API_KEY` with an
`openrouter/...` model); model strings go through LiteLLM.

```python
from agentfield import Agent
from agentfield import AIConfig
from pydantic import BaseModel
import os

# Initialize agent
app = Agent(
    node_id="hello-world",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.7
    ),
)

# ============= SKILL (DETERMINISTIC) =============


@app.skill()
def get_greeting(name: str) -> dict:
    """Returns a greeting template (deterministic - no AI)"""
    return {"message": f"Hello, {name}! Welcome to Agentfield."}


# ============= REASONERS (AI-POWERED) =============


class EmojiResult(BaseModel):
    """Simple schema for emoji addition"""

    text: str
    emoji: str


@app.reasoner()
async def add_emoji(text: str) -> EmojiResult:
    """Uses AI to add an appropriate emoji to text"""
    return await app.ai(
        user=f"Add one appropriate emoji to this greeting: {text}", schema=EmojiResult
    )


@app.reasoner()
async def say_hello(name: str) -> dict:
    # Step 1: Get greeting from skill (deterministic)
    greeting = get_greeting(name)

    # Step 2: Add emoji using AI (reasoner)
    result = await add_emoji(greeting["message"])

    return {"greeting": result.text, "emoji": result.emoji, "name": name}


if __name__ == "__main__":
    # Universal entry point - auto-detects CLI vs server mode
    app.run(auto_port=True)
```

Run `python main.py` with the control plane up, then call
`POST http://localhost:8080/api/v1/execute/hello-world.say_hello` with
`{"input": {"name": "Ada"}}`.

`app.ai()` takes `system=`, `user=`, `schema=` (a Pydantic model, returned typed),
`model=` and `temperature=`. Function parameters become the JSON `input` fields.

## Minimal TypeScript agent

This is what `af init my-agent --defaults --language typescript` generates
(`control-plane/internal/templates/typescript/`), with the node ID filled in.
Dependencies: `@agentfield/sdk`, `dotenv`, `zod`; run with `tsx main.ts`.

`main.ts`:

```ts
import 'dotenv/config';
import { pathToFileURL } from 'node:url';
import { Agent } from '@agentfield/sdk';
import { reasonersRouter } from './reasoners.js';

async function main() {
  const agent = new Agent({
    // AGENTFIELD_SERVER / AGENT_NODE_ID are the names used across SDKs and
    // docs; AGENTFIELD_URL / AGENT_ID remain supported for existing setups.
    nodeId: process.env.AGENT_NODE_ID ?? process.env.AGENT_ID ?? "my-agent",
    agentFieldUrl: process.env.AGENTFIELD_SERVER ?? process.env.AGENTFIELD_URL ?? 'http://localhost:8080',
    port: Number(process.env.PORT ?? 8001),
    version: '1.0.0',
    devMode: true,
    // URL the control plane uses to reach this agent. Needed when the
    // agent is not on the control plane's localhost (e.g. Docker Compose).
    publicUrl: process.env.AGENT_CALLBACK_URL,

    // 🔧 Uncomment to enable AI features:
    // aiConfig: {
    //   provider: 'openai',            // or 'anthropic'
    //   model: 'gpt-4o',
    //   apiKey: process.env.OPENAI_API_KEY, // or OPENROUTER_API_KEY for OpenRouter
    //   // baseUrl: 'https://openrouter.ai/api/v1',
    //   // temperature: 0.7,
    //   // maxTokens: 4096
    // },
  });

  agent.includeRouter(reasonersRouter);

  await agent.serve();
  // eslint-disable-next-line no-console
  console.log(`Agent "${agent.config.nodeId}" listening on http://localhost:${agent.config.port}`);
}

// pathToFileURL percent-encodes the script path the same way import.meta.url
// is encoded, so this also works when the project path contains spaces.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    process.exit(1);
  });
}
```

`reasoners.ts` (the template's working reasoner, no LLM needed; doc comment removed):

```ts
import { AgentRouter } from '@agentfield/sdk';
import { z } from 'zod';

// Group related reasoners with a router
export const reasonersRouter = new AgentRouter({ prefix: 'demo', tags: ['example'] });

reasonersRouter.reasoner<{ message: string }, { original: string; echoed: string; length: number }>(
  'echo',
  async (ctx) => {
    const message = ctx.input.message ?? '';
    return {
      original: message,
      echoed: message,
      length: message.length
    };
  }
);
```

With `aiConfig` set, an LLM call inside a reasoner is
`await ctx.ai(prompt, { schema: zodSchema, temperature: 0.2 })`; the template
ships an `analyzeSentiment` example of this, commented out. Inputs arrive on
`ctx.input`.

## Call an agent over HTTP

Every reasoner and skill is reachable through the control plane, never the
agent's own port (`docs/api/EXECUTE.md`):

```bash
# Synchronous: returns the finished execution
curl -X POST http://localhost:8080/api/v1/execute/my-agent.demo_echo \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'

# Asynchronous: returns 202 with execution_id and status "queued"
curl -X POST http://localhost:8080/api/v1/execute/async/my-agent.demo_echo \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}, "webhook": {"url": "https://example.com/hook", "secret": "shared-secret"}}'

# Poll an async execution until it is terminal ("queued" and "running" are in flight)
curl http://localhost:8080/api/v1/executions/<execution_id>
```

The optional `webhook` object (`url`, `secret`, `headers`) registers a
completion callback. When the async queue is full the control plane answers
`503` instead of queueing without bound.

## Cross-agent calls, memory and human approval

- **Call another agent**: `await app.call("other-node.reasoner_name", arg=value)`
  (Python). The call goes through the control plane, so it is recorded in the same
  run, retried, and subject to access policy.
- **Shared memory** with workflow, session, actor and global scopes is on
  `app.memory` / `ctx.memory`; see the shared-memory docs below.
- **Human approval**: `await app.pause(approval_request_id=..., expires_in_hours=24)`
  suspends the execution and returns an `ApprovalResult` (`.approved`,
  `.changes_requested`, `.feedback`). The waiting state lives in the control plane,
  so it survives an agent restart. TypeScript uses `ApprovalClient`. Full example:
  `examples/python_agent_nodes/waiting_state/`.

## Where identity, policy and audit live

- **Identity**: each agent gets a W3C DID with Ed25519 keys when it registers.
  It is on by default (`Agent(..., enable_did=True, vc_enabled=True)`).
  Configuration is `features.did` in `control-plane/config/agentfield.yaml`.
- **Policy**: agents declare tags (`Agent(..., tags=["analytics"])`, and per
  reasoner). The control plane's `features.did.authorization` block holds
  `tag_approval_rules` (`auto` / `manual` / `forbidden` per tag) and
  `access_policies` (caller tags → target tags, `allow_functions`,
  `deny_functions`, parameter `constraints`). A denied cross-agent call returns
  `403`. Working pair: `examples/python_agent_nodes/permission_agent_a/` and
  `permission_agent_b/`. Design: `docs/VC_AUTHORIZATION_ARCHITECTURE.md`.
- **Audit**: every execution can carry a signed verifiable credential; verify an
  exported one offline with `af vc verify <vc-file.json>`. Agents add their own
  audit entries with `app.note("message", tags=[...])` (Python) or
  `ctx.note(message, tags)` (TypeScript). The dashboard shows each run as a DAG;
  Prometheus metrics are at `/metrics`.

## Deploy

- Control plane: the `agentfield/control-plane` Docker image, or the `af` binary.
  Local storage is SQLite; set Postgres for production
  (`docs/ENVIRONMENT_VARIABLES.md`).
- Agents are ordinary services; run them anywhere that can reach the control
  plane. When the control plane cannot reach the agent on localhost (Docker,
  Kubernetes), set the agent's callback URL (`AGENT_CALLBACK_URL` / `publicUrl`).
- Kubernetes: `docs/deploying-on-kubernetes.md`. Docker Compose and production
  setup: the deployment docs below.

## Docs

The documentation index for agents is `https://agentfield.ai/llms.txt`; every docs
page also answers `Accept: text/markdown`.

- [Quickstart](https://agentfield.ai/docs/agentfield/learn/quickstart?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-quickstart)
- [Python SDK reference](https://agentfield.ai/docs/agentfield/reference/sdks/python?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-python-sdk)
- [TypeScript SDK reference](https://agentfield.ai/docs/agentfield/reference/sdks/typescript?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-typescript-sdk)
- [REST API reference](https://agentfield.ai/docs/agentfield/reference/sdks/rest-api?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-rest-api)
- [Async execution](https://agentfield.ai/docs/agentfield/build/execution/async?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-async)
- [Human-in-the-loop](https://agentfield.ai/docs/agentfield/build/execution/human-in-the-loop?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-human-in-the-loop)
- [Shared memory](https://agentfield.ai/docs/agentfield/build/coordination/shared-memory?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-shared-memory)
- [Identity](https://agentfield.ai/docs/agentfield/build/governance/identity?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-identity), [Policy](https://agentfield.ai/docs/agentfield/build/governance/policy?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-policy), [Audit](https://agentfield.ai/docs/agentfield/build/governance/audit?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-audit)
- [Deploy](https://agentfield.ai/docs/agentfield/reference/deploy?utm_source=agent-skill&utm_campaign=agent-skill&utm_id=agent-skill-deploy)
- Repository: https://github.com/Agent-Field/agentfield (`examples/` has runnable
  Python, TypeScript and Go agents)

Related skills in this repository: `agentfield` (a guided workflow that designs and
scaffolds a multi-agent project with Docker Compose), `agentfield-personal`
(an agent installed on this machine with `af install`), and `agentfield-use`
(calling agents that are already installed).
