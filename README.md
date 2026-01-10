<div align="center">

<img src="assets/github hero.png" alt="AgentField - Kubernetes, for AI Agents" width="100%" />

# Kubernetes for AI Agents

### **Deploy, Scale, Observe, and Prove.**

*When AI moves from chatbots into backends, making decisions, not just answering questions, it needs infrastructure, not frameworks.*

[![License](https://img.shields.io/badge/license-Apache%202.0-7c3aed.svg?style=flat&labelColor=1e1e2e)](LICENSE)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fsantoshkumarradha%2Fd98e2ad73502b4075f6a5f0ae4f5cae5%2Fraw%2Fbadge.json&style=flat&logo=download&logoColor=white&labelColor=1e1e2e&cacheSeconds=3600)](https://github.com/Agent-Field/agentfield)
[![Last Commit](https://img.shields.io/github/last-commit/Agent-Field/agentfield?style=flat&logo=git&logoColor=white&color=7c3aed&labelColor=1e1e2e)](https://github.com/Agent-Field/agentfield/commits/main)
[![Go](https://img.shields.io/badge/go-1.21+-00ADD8.svg?style=flat&labelColor=1e1e2e&logo=go&logoColor=white)](https://go.dev/)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat&labelColor=1e1e2e&logo=python&logoColor=white)](https://www.python.org/)
[![Deploy with Docker](https://img.shields.io/badge/deploy-docker-2496ED.svg?style=flat&labelColor=1e1e2e&logo=docker&logoColor=white)](https://docs.docker.com/)
[![Discord](https://img.shields.io/badge/discord-join%20us-5865F2.svg?style=flat&labelColor=1e1e2e&logo=discord&logoColor=white)](https://discord.gg/aBHaXMkpqh)

**[Docs](https://agentfield.ai/docs)** | **[Quick Start](https://agentfield.ai/docs/quick-start)** | **[Python SDK](https://agentfield.ai/api/python-sdk/overview)** | **[Go SDK](https://agentfield.ai/api/go-sdk/overview)** | **[TypeScript SDK](https://agentfield.ai/api/typescript-sdk/overview)** | **[REST API](https://agentfield.ai/api/rest-api/overview)**

</div>





## What is AgentField?

**AgentField is the backend infrastructure layer for autonomous AI.**

AI has outgrown frameworks and is moving from chatbots into backends—making decisions about refunds, coordinating supply chains, managing portfolios. These agents need infrastructure, not prompt wrappers.

AgentField is an open-source **control plane** that treats AI agents as first-class backend services and makes agents production-ready.

**Scale Infrastructure** *(think: Kubernetes)*
- **Routing & Discovery**: Agents find and call each other through standard REST APIs
- **Async Execution**: Fire-and-forget tasks that run for minutes, hours, or days
- **Durable State**: Built-in memory with vector search—no Redis or Pinecone required
- **Observability**: Automatic workflow DAGs, Prometheus metrics, structured logs

**Trust Infrastructure** *(think: Okta, rebuilt for agents)*
- **W3C DIDs**: Every agent gets a cryptographic identity—not a shared API key
- **Verifiable Credentials**: Tamper-proof audit trails for every action
- **Policy Enforcement**: Boundaries enforced by infrastructure, not prompts

Write [Python](https://agentfield.ai/api/python-sdk/overview), [Go](https://agentfield.ai/api/go-sdk/overview), [TypeScript](https://agentfield.ai/api/typescript-sdk/overview), or call via [REST](https://agentfield.ai/api/rest-api/overview). Get production infrastructure automatically.

---

## The AI Backend

Software keeps adding layers when complexity demands it. Frontend/backend separation. Data lakes and pipelines. Now: a **reasoning layer** that sits alongside your services, making decisions that used to be hardcoded.

We call this the AI Backend. Not a chatbot, not a copilot—infrastructure for software that can think.

**Guided autonomy:** Agents that reason freely within boundaries you define. Predictable enough to trust. Flexible enough to be useful.

📖 **[Read: The AI Backend](https://agentfield.ai/blog/posts/ai-backend/?utm_source=github-readme)** — Our thesis on why every serious backend will need a reasoning layer.

---

## See It In Action

<div align="center">
<img src="assets/UI.png" alt="AgentField Dashboard" width="100%" />
<br/>
<i>Real-time Observability • Execution Flow • Audit Trails</i>
</div>

---

## Build Agents in Any Language

<details open>
<summary><strong>Python</strong></summary>

```python
from agentfield import Agent, AIConfig

app = Agent(node_id="researcher", ai_config=AIConfig(model="gpt-4o"))

@app.skill()
def fetch_url(url: str) -> str:
    return requests.get(url).text

@app.reasoner()
async def summarize(url: str) -> dict:
    content = fetch_url(url)
    return await app.ai(f"Summarize: {content}")

app.run()  # → POST /api/v1/execute/researcher.summarize
```

[Full Python SDK Documentation →](https://agentfield.ai/api/python-sdk/overview)
</details>

<details>
<summary><strong>Go</strong></summary>

```go
agent, _ := agentfieldagent.New(agentfieldagent.Config{
    NodeID:        "researcher",
    AgentFieldURL: "http://localhost:8080",
})

agent.RegisterSkill("summarize", func(ctx context.Context, input map[string]any) (any, error) {
    url := input["url"].(string)
    // Your agent logic here
    return map[string]any{"summary": "..."}, nil
})

agent.Run(context.Background())
```

[Full Go SDK Documentation →](https://agentfield.ai/api/go-sdk/overview)
</details>

<details>
<summary><strong>TypeScript</strong></summary>

```typescript
import { Agent } from '@agentfield/sdk';

const agent = new Agent({
  nodeId: 'researcher',
  agentFieldUrl: 'http://localhost:8080',
});

agent.reasoner('summarize', async (ctx, input: { url: string }) => {
  const content = await fetch(input.url).then(r => r.text());
  return await ctx.ai(`Summarize: ${content}`);
});

agent.run();  // → POST /api/v1/execute/researcher.summarize
```

[Full TypeScript SDK Documentation →](https://agentfield.ai/api/typescript-sdk/overview)
</details>

<details>
<summary><strong>REST / Any Language</strong></summary>

```bash
# Call any agent from anywhere—no SDK required
curl -X POST http://localhost:8080/api/v1/execute/researcher.summarize \
  -H "Content-Type: application/json" \
  -d '{"input": {"url": "https://example.com"}}'
```

```javascript
// Frontend (React, Next.js, etc.)
const result = await fetch("http://localhost:8080/api/v1/execute/researcher.summarize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input: { url: "https://example.com" } }),
}).then(r => r.json());
```

[REST API Reference →](https://agentfield.ai/api/rest-api/overview)
</details>

---

## Quick Start

### 1. Install

```bash
curl -fsSL https://agentfield.ai/install.sh | bash
```

### 2. Create Your Agent

```bash
af init my-agent --defaults
cd my-agent && pip install -r requirements.txt
```

### 3. Start (Two Terminals Required)

AgentField uses a **control plane + agent node** architecture. You'll need two terminal windows:

**Terminal 1 – Start the Control Plane:**
```bash
af server
```
> Opens the dashboard at http://localhost:8080

**Terminal 2 – Start Your Agent:**
```bash
python main.py
```
> Agent auto-registers with the control plane

### 4. Test It

```bash
curl -X POST http://localhost:8080/api/v1/execute/my-agent.demo_echo \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'
```

<details>
<summary><strong>Other Languages / Options</strong></summary>

**Go:**
```bash
af init my-agent --defaults --language go
cd my-agent && go mod download
go run .
```

**TypeScript:**
```bash
af init my-agent --defaults --language typescript
cd my-agent && npm install
npm run dev
```

**Interactive mode** (choose language, set author info):
```bash
af init my-agent  # No --defaults flag
```
</details>

<details>
<summary><strong>Docker / Troubleshooting</strong></summary>

If running the **control plane in Docker** and your **agent node runs outside that container**, make sure the control plane can reach the agent at the URL it registers.

**Option A (agent on your host, control plane in Docker):**
```bash
docker run -p 8080:8080 agentfield/control-plane:latest

# Python agents (recommended)
export AGENTFIELD_URL="http://localhost:8080"
export AGENT_CALLBACK_URL="http://host.docker.internal:8001"
python main.py

# Go agents
export AGENTFIELD_URL="http://localhost:8080"
export AGENT_PUBLIC_URL="http://host.docker.internal:8001"
```

**Option B (agent + control plane both in Docker Compose / same network):**
- Set the agent callback/public URL to the agent container's service name, e.g. `http://my-agent:8001`.

**Linux note:** `host.docker.internal` may require `--add-host=host.docker.internal:host-gateway` or using a Compose setup where both containers share a network.
</details>

**Next Steps:** [Build Your First Agent](https://agentfield.ai/guides/getting-started/build-your-first-agent) | [Deploy to Production](https://agentfield.ai/guides/deployment/overview) | [Examples](https://agentfield.ai/examples)

---

## The Production Gap

Most frameworks stop at "make the LLM call." But production agents need:

[See the production-ready feature set →](https://agentfield.ai/docs/why-agentfield/production-ready-features)

### Scale & Reliability
Agents that run for hours or days. Webhooks with automatic retries. Backpressure handling when downstream services are slow.

```python
# Fire-and-forget: webhook called when done
result = await app.call(
    "research_agent.deep_dive",
    input={"topic": "quantum computing"},
    async_config=AsyncConfig(
        webhook_url="https://myapp.com/webhook",
        timeout_hours=6
    )
)
```

### Multi-Agent Coordination
Agents that discover and invoke each other through the control plane. Every call tracked. Every workflow visualized as a DAG.

```python
# Agent A calls Agent B—routed through control plane, fully traced
analysis = await app.call("analyst.evaluate", input={"data": dataset})
report = await app.call("writer.summarize", input={"analysis": analysis})
```

### Developer Experience
Standard REST APIs. No magic abstractions. Build agents the way you build microservices.

```bash
# Every agent is an API endpoint
curl -X POST http://localhost:8080/api/v1/execute/researcher.summarize \
  -H "Content-Type: application/json" \
  -d '{"input": {"url": "https://example.com"}}'
```

### Enterprise Ready
Cryptographic identity for every agent. Tamper-proof audit trails for every action. [Learn more about Identity & Trust](https://agentfield.ai/docs/core-concepts/identity-and-trust).

---

## A New Backend Paradigm

AgentField isn't a framework you extend. It's infrastructure you deploy on.

[See how AgentField compares to agent frameworks →](https://agentfield.ai/docs/why-agentfield/vs-agent-frameworks)

|                    | Agent Frameworks           | DAG/Workflow Engines    | AgentField                              |
| ------------------ | -------------------------- | ----------------------- | --------------------------------------- |
| **Architecture**   | Monolithic scripts         | Predetermined pipelines | Distributed microservices               |
| **Execution**      | Synchronous, blocking      | Scheduled, batch        | Async-native (webhooks, SSE, WebSocket) |
| **Coordination**   | Manual message passing     | Central scheduler       | Service mesh with discovery             |
| **Memory**         | External (Redis, Pinecone) | External                | Built-in + vector search                |
| **Multi-language** | SDK-locked                 | Config files            | Native REST APIs (any language)         |
| **Long-running**   | Timeouts, hacks            | Designed for batch      | Hours/days, durable execution           |
| **Audit**          | Logs (trust me)            | Logs                    | Cryptographic proofs (W3C DIDs/VCs)     |

**Not a DAG builder.** Agents decide what to do next—dynamically. The control plane tracks the execution graph automatically.

**Not tool attachment.** You don't just give an LLM a bag of MCP tools and hope. You define **Reasoners** (AI logic) and **Skills** (deterministic code) with explicit boundaries. [Learn more](https://agentfield.ai/docs/core-concepts/reasoners-and-skills).

---

## Key Features

### Scale Infrastructure
- **Control Plane**: Stateless Go service that routes, tracks, and orchestrates
- **Async by Default**: Fire-and-forget or wait. Webhooks with retries. SSE streaming.
- **Long-Running**: Tasks that run for hours or days with durable checkpointing
- **Backpressure**: Built-in queuing and circuit breakers

### Multi-Agent Native
- **Discovery**: Agents register capabilities. Others find them via API.
- **Cross-Agent Calls**: `app.call("other.reasoner", input={...})` routed through control plane
- **Workflow DAGs**: Every execution path visualized automatically
- **Shared Memory**: Scoped to global, agent, session, or run—with vector search

### Enterprise Ready
- **W3C DIDs**: Every agent gets a cryptographic identity
- **Verifiable Credentials**: Tamper-proof receipts for every action
- **Prometheus Metrics**: `/metrics` endpoint out of the box
- **Policy Enforcement**: "Only agents signed by 'Finance' can access this tool"

[Explore the full feature set →](https://agentfield.ai/docs/features)


## Identity & Trust

When agents move from answering questions to making decisions, approving refunds, coordinating supply chains, moving money, "check the logs" isn't enough.

AgentField gives every agent a [W3C Decentralized Identifier (DID)](https://www.w3.org/TR/did-core/)—a cryptographic identity. Every execution produces a Verifiable Credential: a tamper-proof receipt showing exactly what happened, who authorized it, and the full delegation chain.

```bash
# Export audit trail for any workflow
curl http://localhost:8080/api/ui/v1/workflows/{workflow_id}/vc-chain
```

For compliance teams: mathematical proof, not trust.

📖 **[Read: IAM for AI Backends](https://agentfield.ai/blog/posts/iam-ai-backends)** — Why OAuth can't secure autonomous software, and what replaces it.

[Full documentation →](https://agentfield.ai/docs/core-concepts/identity-and-trust)



## Architecture

<div align="center">
<img src="assets/arch.png" alt="AgentField Architecture Diagram" width="80%" />
</div>

[Learn more about the core architecture →](https://agentfield.ai/docs/why-agentfield/core-architecture)



## Is AgentField for you?

### Yes if:
- You're building an **AI backend** - agents that make decisions, not just answer questions
- You're building **multi-agent systems** that need to coordinate
- You need **production infrastructure**: async, retries, observability
- You want agents as **standard backend services** with REST APIs
- You need **audit trails** for compliance or debugging
- You have **multiple teams** deploying agents independently

### Not yet if:
- You're building a **single chatbot** (prompt orchestration frameworks like LangChain, CrewAI, LlamaIndex etc.. are great for that)
- You're **prototyping** and don't need production concerns yet

*When you're ready to ship agents to production, we'll be here.*

---

If you are **Backend Engineers** shipping AI into production who want standard APIs, not magic or **Platform Teams** who don't want to build another homegrown orchestrator or **Enterprise Teams** in regulated industries (Finance, Health) needing audit trails or **Frontend Developers** who just want to `fetch()` an agent without Python headaches, AgentField is built for you.

---

## Learn More

- 📖 **[The AI Backend](https://agentfield.ai/blog/posts/ai-backend)** — Why every backend needs a reasoning layer
- 📖 **[IAM for AI Backends](https://agentfield.ai/blog/posts/iam-ai-backends)** — Why agents need identity, not just API keys
- 📚 **[Documentation](https://agentfield.ai/docs)** — Full technical reference
- 🚀 **[Examples](https://agentfield.ai/examples)** — Production patterns and use cases

---

## Community

**Agents are becoming part of production backends. They need identity, governance, and infrastructure. That's why AgentField exists.**

- **[Documentation](https://agentfield.ai/docs)**
- **[GitHub Issues](https://github.com/Agent-Field/agentfield/issues)**
- **[Twitter/X](https://x.com/agentfield_ai)**
- **[Examples](https://agentfield.ai/examples)**

<p align="center">
  <strong>Built by developers who got tired of duct-taping agents together.</strong><br>
  <a href="https://agentfield.ai">agentfield.ai</a>
</p>
