<div align="center">

<img src="assets/github.png" alt="AgentField - Kubernetes, for AI Agents" width="100%" />

**A Kubernetes-style control plane that runs AI agents like microservices: REST/gRPC APIs, async webhooks, and cryptographic identity for every agent and execution.**

Write agents. Haxen deploys, scales, observes, and proves what happened.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.21+-00ADD8.svg)](https://go.dev/)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB.svg)](https://www.python.org/)
[![Deploy with Docker](https://img.shields.io/badge/deploy-docker-2496ED.svg)](https://docs.docker.com/)
[![Discord](https://img.shields.io/badge/discord-join-5865F2.svg)](https://discord.gg/your-discord)

**[📚 Docs](https://haxen.ai/docs)** • **[⚡ Quickstart](#-try-haxen-in-2-minutes)** • **[💬 Discord](https://discord.gg/your-discord)**

</div>

---

## TL;DR

- **Write agents in Python/Go** (or any language via REST/gRPC)
- **Deploy independently** like microservices—zero coordination between teams
- **Get production infrastructure automatically**: REST APIs, streaming, async queues, observability, cryptographic audit trails
- **Run anywhere**: local dev, Docker, Kubernetes, cloud

```bash
curl -fsSL https://haxen.ai/install.sh | bash && haxen init my-agents
```

---

## 🚀 Try Haxen in 2 Minutes

### Option 1: Local Install

```bash
# macOS/Linux - install CLI
curl -fsSL https://haxen.ai/install.sh | bash

# Start control plane + create your first agent
haxen dev
haxen init my-agents && cd my-agents
haxen run
```

### Option 2: Docker Compose

```bash
git clone https://github.com/agentfield/haxen
cd haxen && docker compose up
```

Your control plane is running at `http://localhost:8080`

**[📚 Full quickstart guide →](https://haxen.ai/docs/quick-start)** • **[💬 Need help? Discord](https://discord.gg/your-discord)**

---

## Hello, Agent (20 lines)

Write your first agent—automatically get a REST API:

```python
from haxen_sdk import Agent

# Create an agent
app = Agent("greeting-agent")

# Decorate a function—becomes a REST endpoint automatically
@app.reasoner()
async def say_hello(name: str) -> dict:
    message = await app.ai(f"Generate a personalized greeting for {name}")
    return {"greeting": message}
```

**Deploy:**
```bash
haxen run
```

**Call from anywhere** (REST API auto-generated):
```bash
curl -X POST http://localhost:8080/api/v1/execute/greeting-agent.say_hello \
  -H "Content-Type: application/json" \
  -d '{"input": {"name": "Alice"}}'
```

**You automatically get:**
- ✅ REST API at `/execute/greeting-agent.say_hello` (OpenAPI spec at `/openapi.yaml`)
- ✅ Async execution: `/execute/async/...` with webhook callbacks (HMAC-signed)
- ✅ Health checks: `/health/live` and `/health/ready` (Kubernetes-ready)
- ✅ Prometheus metrics: `/metrics` labeled by agent, version, and run_id
- ✅ Workflow DAG in UI (visual trace of execution)

**That's it.** One function = production-ready service.

**[📚 Docs](https://haxen.ai/docs)** • **[⚡ More examples](https://github.com/agentfield/haxen-examples)** • **[💬 Discord](https://discord.gg/your-discord)**

---

## Why Haxen?

Agent frameworks are great for **prototypes**. Haxen builds agents **and** runs them at production scale.

### What Hurts Today → What Haxen Does Automatically

| 🔴 **Without Haxen**                                                                                         | 🟢 **With Haxen**                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Monolithic deployments** — one team's change forces everyone to redeploy                                  | **Independent deployment** — teams ship agents on their own schedule, zero coordination                                                                    |
| **No native APIs** — your React app needs custom wrappers to call agents                                    | **REST & OpenAPI by default** (gRPC optional) — every function is an endpoint, auto-documented                                                             |
| **Manual orchestration** — you build queues, state management, agent coordination from scratch              | **Auto-orchestration** — agents call each other, workflows track automatically, shared memory (Global/Agent/Session/Run scopes) syncs state                |
| **No observability** — grep logs from 5 services to debug multi-agent flows                                 | **Built-in observability** — workflow DAGs (UI), execution traces, agent notes, Prometheus metrics (`/metrics` per agent, labeled by agent/version/run_id) |
| **DIY infrastructure** — you're building webhooks, SSE, async queues, health checks, retries yourself       | **Production infrastructure** — durable queues, webhook delivery (HMAC-signed), SSE streaming, graceful shutdown, K8s-ready health checks                  |
| **No identity or audit** — logs can be edited, screenshots faked, compliance teams need cryptographic proof | **Cryptographic identity** — W3C DIDs (`did:web` or `did:key`) for every agent, W3C Verifiable Credentials (JSON-LD) for tamper-proof audit trails         |

### The Analogy

```
Traditional Frameworks = Flask (single app)
Haxen = Kubernetes + Auth0 for AI (distributed infrastructure + identity)
```

> **Haxen isn't a framework you extend with infrastructure. It IS the infrastructure.**

Bring your own model/tooling; Haxen handles runtime, scale, and proof.

---

## ⚡ 30-Second Multi-Agent Demo

**The scenario:** Customer support system with 3 coordinating agents.

```python
from haxen_sdk import Agent

# Agent 1: Support orchestrator (Team: Customer Success)
support = Agent("support-agent")

@support.reasoner()
async def handle_ticket(ticket: dict) -> dict:
    # Call Agent 2 (different service, different team)
    sentiment = await support.call("sentiment-agent.analyze",
                                    text=ticket["message"])

    # Call Agent 3 (knowledge base, Data team)
    solutions = await support.call("kb-agent.search",
                                    query=ticket["issue"])

    # Conditional escalation
    if sentiment["urgency"] == "high":
        await support.call("escalation-agent.create_case", ticket=ticket)

    return {"solutions": solutions, "sentiment": sentiment}
```

**Deploy:**
```bash
haxen dev           # Start control plane
haxen run           # Deploy your agent
```

**You get automatically:**
- ✅ REST API: `POST /execute/support-agent.handle_ticket`
- ✅ Async execution: `POST /execute/async/...` with webhooks (HMAC-signed, 6 retry attempts with exponential backoff)
- ✅ Real-time streaming (SSE): Your frontend gets live updates as the workflow executes
- ✅ Workflow DAG: Visual graph showing which agent called which (auto-generated in UI)
- ✅ Shared memory: All 3 agents access the same state (Global/Agent/Session/Run scopes, automatic)
- ✅ Observability: Prometheus metrics (`/metrics`), execution traces, agent notes
- ✅ Identity: Cryptographic proof of every decision (W3C DIDs + Verifiable Credentials)
- ✅ Health checks, Docker/K8s ready, horizontal scaling

**From your React app:**
```javascript
// Call agents via REST API (no custom SDK needed)
const response = await fetch('http://haxen:8080/api/v1/execute/support-agent.handle_ticket', {
  method: 'POST',
  body: JSON.stringify({ input: { ticket: {...} } })
});

// Stream real-time updates
const eventSource = new EventSource(
  `http://haxen:8080/api/v1/workflows/runs/${runId}/events/stream`
);
eventSource.onmessage = (e) => {
  console.log('Agent update:', JSON.parse(e.data));
};
```

**🎨 UI SCREENSHOT #1 (HERO): Add here**
> **What to show:** Workflow DAG visualization from the Haxen UI showing the 3 agents (support-agent → sentiment-agent, support-agent → kb-agent, support-agent → escalation-agent) with execution times, status indicators (green checkmarks), and the visual graph. This is the "wow" moment that shows developers the automatic observability.
>
> **Recommended dimensions:** 1200x700px, annotate with arrows pointing to: "Auto-generated DAG", "Execution times", "Agent-to-agent calls"

**Scale individual agents:**
```bash
kubectl scale deployment sentiment-agent --replicas=10
# Other agents unaffected
```

### That's the difference.

Haxen **is** the infrastructure you'd otherwise spend 3 months building.

---

## Three Pillars: What You Get Out of the Box

### 🔧 Runtime & APIs

**Deploy AI agents like microservices—independent, discoverable, language-agnostic.**

| Feature                                       | What It Does                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **REST & OpenAPI by default** (gRPC optional) | Every agent function becomes an endpoint automatically                                                  |
| **Real-time streaming (SSE)**                 | Frontends get live updates as workflows execute; see [React example](#real-time-streaming)              |
| **Async execution**                           | Long-running tasks (5+ min) run in durable queues; webhook callbacks when done (HMAC-signed, 6 retries) |
| **Agent-to-agent calls**                      | `await agent.call("other-agent.function")` — control plane routes automatically                         |
| **Shared memory**                             | Zero-config state: Global → Agent → Session → Run scopes (automatic syncing)                            |
| **Language-agnostic**                         | Python SDK, Go SDK, or implement REST/gRPC protocol directly in any language                            |

...and many more !

**📚 [Runtime docs →](https://haxen.ai/docs/runtime)**

### 📊 Scale & Ops

**Production-grade observability, scaling, and reliability—no instrumentation required.**

| Feature                | What It Does                                                                                                                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workflow DAGs**      | Visual execution graphs (auto-generated in UI); see exactly which agent called which and when                                                                                                            |
| **Prometheus metrics** | Automatic `/metrics` endpoint per agent and control plane, labeled by `agent`, `version`, `run_id`; request rates, latencies, errors (injected via control plane proxy—agents need zero instrumentation) |
| **Durable queues**     | PostgreSQL `FOR UPDATE SKIP LOCKED` lease-based processing; survives crashes, fair scheduling, backpressure                                                                                              |
| **Health checks**      | `/health/live` and `/health/ready` for Kubernetes liveness/readiness probes                                                                                                                              |
| **Horizontal scaling** | Stateless control plane scales horizontally; scale individual agent nodes independently (`kubectl scale deployment my-agent --replicas=10`)                                                              |
| **Graceful shutdown**  | Completes in-flight work before exit; no dropped tasks                                                                                                                                                   |
| **Auto-retries**       | Failed executions retry with exponential backoff (configurable)                                                                                                                                          |

**How Prometheus metrics are injected:** The control plane acts as a reverse proxy for agent traffic. All agent-to-agent calls and executions flow through the control plane, which records latency, error rates, and throughput **without requiring agents to instrument their code**. Metrics are exposed at `/metrics` in Prometheus format.

**📚 [Scale & ops docs →](https://haxen.ai/docs/observability)**

### 🔒 Identity & Audit

**Cryptographic proof for compliance—tamper-proof, exportable, verifiable offline.**

| Feature                        | What It Does                                                                                                           |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **W3C DIDs**                   | Every agent gets a Decentralized Identifier (`did:web` or `did:key`); cryptographic identity for non-repudiation       |
| **W3C Verifiable Credentials** | Opt-in per agent; each execution generates a VC (JSON-LD format) with signed input/output hashes                       |
| **Tamper-proof audit trails**  | Export full VC chains for regulators; verify offline with `haxen verify audit.json` (no access to your systems needed) |
| **Non-repudiation**            | Agents cryptographically sign decisions; can't deny their actions                                                      |
| **Policy engine**              | Define rules for which executions require VCs (e.g., "all financial decisions > $10K")                                 |
| **Export formats**             | W3C VC JSON-LD (standard); import into compliance tools                                                                |

**Example: Enable for a specific agent**
```python
# Opt-in per agent (not all agents need VCs)
if app.vc_generator:
    app.vc_generator.set_enabled(True)

@app.reasoner()
async def approve_loan(application: dict) -> Decision:
    decision = await app.ai("Evaluate loan risk", ...)
    # VC generated automatically with DID signature + input/output hashes
    return decision
```

**For auditors/compliance:**
```bash
# Export cryptographic proof chain
curl http://haxen:8080/api/v1/did/workflow/wf_abc123/vc-chain > audit.json

# Verify offline (no access to your systems needed)
haxen verify audit.json
# ✓ All signatures valid (W3C VC spec)
# ✓ No tampering detected
# ✓ Complete provenance chain
```

**🎨 UI SCREENSHOT #2: Add here**
> **What to show:** Haxen UI showing the DID/VC verification interface. Display a workflow with DIDs for each agent, the VC chain visualization, and the verification status (green checkmarks showing "All signatures valid").
>
> **Caption:** "W3C DIDs and Verifiable Credentials—tamper-proof audit trails for compliance"

**📚 [Identity & audit docs →](https://haxen.ai/docs/identity)**

---

## 🏗️ Architecture

### Deploy AI Agents Like Kubernetes Deploys Containers

Haxen uses a **two-layer design**: a stateless **control plane** (like K8s control plane) and independent **agent nodes** (like pods):

```
┌───────────────────────────────────────────────────────────────────┐
│                   HAXEN CONTROL PLANE                             │
│           (Stateless Go Services - Scale Horizontally)            │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐ ┌──────────────────┐ ┌────────────────┐      │
│  │ RUNTIME & APIs  │ │ SCALE & OPS      │ │ IDENTITY &     │      │
│  │                 │ │                  │ │ AUDIT          │      │
│  ├─────────────────┤ ├──────────────────┤ ├────────────────┤      │
│  │ • REST/gRPC     │ │ • Workflow       │ │ • W3C DIDs     │      │
│  │   Gateway       │ │   Engine (DAGs)  │ │ • Verifiable   │      │
│  │ • Async Queue   │ │ • Prometheus     │ │   Credentials  │      │
│  │   (durable)     │ │   Metrics        │ │ • Audit Trails │      │
│  │ • SSE Streaming │ │ • Health Checks  │ │ • Policy       │      │
│  │ • Agent-to-     │ │ • Auto Retries   │ │   Engine       │      │
│  │   Agent Calls   │ │ • Graceful       │ │ • Non-         │      │
│  │ • Shared Memory │ │   Shutdown       │ │   Repudiation  │      │
│  │   (4 scopes)    │ │ • Backpressure   │ │                │      │
│  └─────────────────┘ └──────────────────┘ └────────────────┘      │
│                                                                   │
│  Deploy independently → Coordinate automatically → Trust built-in │
└───────────────────────────────────────────────────────────────────┘
       ▲                        ▲                      ▲
       │    REST API / gRPC / WebSocket / HTTP         │
       │    (Language-agnostic communication)          │
       ▼                        ▼                      ▼
┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐
│  Agent Node   │  │ Agent Node   │  │   Agent Node        │
│  (Container)  │  │ (Container)  │  │   (Container)       │
│               │  │              │  │                     │
│  Python SDK   │  │  Go SDK      │  │   Any Language      │
│  support-     │  │  analytics-  │  │   payment-          │
│  agent:v2     │  │  agent:v1    │  │   agent:v3          │
│               │  │              │  │   (REST/gRPC)       │
│  Team: CS     │  │  Team: Data  │  │   Team: Finance     │
│  Replicas: 5  │  │  Replicas: 2 │  │   Replicas: 3       │
└───────────────┘  └──────────────┘  └─────────────────────┘
```

### Think Kubernetes, But for AI

| Kubernetes             | Haxen                 | What It Means                                  |
| ---------------------- | --------------------- | ---------------------------------------------- |
| **Pods**               | **Agent Nodes**       | Your AI agent runs in a container              |
| **Services**           | **Agent Registry**    | Control plane discovers agents automatically   |
| **kubectl apply**      | **haxen run**         | Deploy your agent independently                |
| **Horizontal scaling** | **Scale agent nodes** | Add more replicas per agent                    |
| **Service mesh**       | **Control plane**     | Automatic routing, observability, state        |
| **Ingress**            | **API Gateway**       | Every agent function is a REST endpoint        |
| **Language-agnostic**  | **REST/gRPC API**     | Use Python, Go, Rust, JavaScript, any language |

### How It Works

1. **Write agents in any language** — Python SDK, Go SDK, or raw REST/gRPC
2. **Deploy as containers** — `docker build` + `haxen run` or `kubectl apply`
3. **Control plane orchestrates** — routing, state, workflows, identity, observability
4. **Agent nodes scale independently** — each team owns their nodes, deploys on their schedule
5. **Everything auto-coordinates** — agents call each other via control plane, memory syncs, workflows track

**Key Insight:** Each agent is a **microservice**. Teams deploy independently. The control plane makes them coordinate like a single system.

**Language Flexibility:** Use our Python/Go SDKs for convenience, or implement the REST/gRPC protocol directly in any language. The control plane is language-agnostic by design.

**[📚 Detailed architecture docs →](https://haxen.ai/docs/architecture)**

---

## When to Use Haxen (And When Not To)

### ✅ Use Haxen If:

- You're building **multi-agent systems** that need to coordinate
- You need **independent deployment**—multiple teams, different schedules
- You need **production infrastructure**: REST APIs, async queues, observability, health checks
- You need **compliance/audit trails** (finance, healthcare, legal)
- You want to **call agents from frontends** (React, mobile) without custom wrappers
- You're scaling to **multiple environments** (dev, staging, prod) and need consistency

### ❌ Start with a Framework If:

- You're **learning agent concepts** and want the simplest possible start (try LangChain or CrewAI first, then migrate to Haxen when you need production features)
- You're building a **single-agent chatbot** that will never scale beyond one service
- You don't need REST APIs, observability, or multi-agent coordination
- You're prototyping and don't plan to deploy to production

### The Bottom Line

**Frameworks = Build agents** (perfect for learning)
**Haxen = Build and run agents at any scale** (perfect from prototype to production)

You can start with Haxen and skip migration pain later. Or start with a framework and migrate when you hit the pain points above.

---

## 🐳 Deployment

**Local dev:**
```bash
haxen dev && haxen run
```

**Docker Compose:**
```yaml
services:
  haxen-server:
    image: haxen/server:latest
    ports: ["8080:8080"]

  my-agent:
    build: ./agents/my-agent
    environment:
      - HAXEN_SERVER=http://haxen-server:8080
```

**Kubernetes:**
```bash
kubectl apply -f haxen-control-plane.yaml
kubectl apply -f my-agent-deployment.yaml
kubectl scale deployment my-agent --replicas=10
```

**Cloud:** Works on Railway, Render, Fly.io, AWS, GCP, Azure

Each agent deploys independently. Control plane coordinates automatically.

**[📚 Full deployment guides →](https://haxen.ai/docs/deployment)**

---

## 🌍 Community & Contributing

We're building Haxen in the open. Join us:

- **[💬 Discord](https://discord.gg/your-discord)** — Get help, share projects, discuss architecture
- **[📚 Documentation](https://haxen.ai/docs)** — Guides, API reference, examples
- **[💡 GitHub Discussions](https://github.com/agentfield/haxen/discussions)** — Feature requests, Q&A
- **[🐦 Twitter/X](https://x.com/haxen_dev)** — Updates and announcements

### Contributing

Apache 2.0 licensed. Built by developers like you.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

---

## FAQ

<details>
<summary><strong>How does auth work? Do agents authenticate?</strong></summary>

Agents authenticate to the control plane via API keys (configurable per environment). The control plane handles all inter-agent routing—agents don't need to authenticate to each other.

For **end-user auth** (e.g., React app calling agents), you can integrate your existing auth system (JWT, OAuth) at the API gateway layer. Haxen respects your auth headers and passes them to agents via context.

**W3C DIDs** are for **identity** (proving which agent made a decision), not access control.

</details>

<details>
<summary><strong>What's the performance overhead?</strong></summary>

The control plane adds ~5-10ms latency per agent call (routing + state sync). For AI workloads (which take seconds to minutes), this is negligible.

**Prometheus metrics injection** is zero-cost—the control plane already proxies agent traffic, so recording metrics adds no extra network hops.

Benchmark: 10K requests/sec sustained on a single control plane instance (4 cores, 8GB RAM). Horizontal scaling tested to 100K+ req/sec.

</details>

<details>
<summary><strong>Can I use my own observability stack?</strong></summary>

Yes. Haxen exposes:
- **Prometheus metrics** at `/metrics` (scrape with your existing Prometheus)
- **Structured logs** (JSON) to stdout/stderr (ship to your log aggregator)
- **OpenTelemetry traces** (opt-in, export to Jaeger/Datadog/etc.)

The built-in workflow DAG UI is optional—you can disable it and use your own dashboards.

</details>

<details>
<summary><strong>Is this vendor-neutral? Can I switch models/providers?</strong></summary>

**100% vendor-neutral.** Haxen is infrastructure, not a model provider.

- Use **any LLM**: OpenAI, Anthropic, local Ollama, Hugging Face, etc.
- Use **any framework**: Call LangChain, CrewAI, raw model APIs—your choice
- Use **any language**: Python SDK, Go SDK, or raw REST/gRPC

Haxen handles deployment, orchestration, and observability. You control the AI logic.

</details>

---

## 📖 Resources

- **[📚 Documentation](https://haxen.ai/docs)** — Complete guides and API reference
- **[⚡ Quick Start Tutorial](https://haxen.ai/docs/quick-start)** — Build your first agent in 5 minutes
- **[🏗️ Architecture Deep Dive](https://haxen.ai/docs/architecture)** — How Haxen works under the hood
- **[📦 Examples Repository](https://github.com/agentfield/haxen-examples)** — Production-ready agent templates
- **[📝 Blog](https://haxen.ai/blog)** — Tutorials, case studies, best practices

---

<div align="center">

### ⭐ Star us to follow development

**Built by developers who got tired of duct-taping agents together**

**Join the future of autonomous software**

**[🌐 Website](https://haxen.ai) • [📚 Docs](https://haxen.ai/docs) • [💬 Discord](https://discord.gg/your-discord) • [🐦 Twitter](https://x.com/haxen_dev)**

**License:** [Apache 2.0](LICENSE)

---

*We believe autonomous software needs infrastructure that respects what makes it different—agents that reason, decide, and coordinate—while providing the same operational excellence that made traditional software successful.*

</div>
