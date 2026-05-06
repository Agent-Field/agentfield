<div align="center">

<img src="assets/github hero.png" alt="AgentField - The AI Backend" width="100%" />

# The AI Backend

### **Build and scale AI agents like APIs. Deploy, observe, and prove.**

*AI has outgrown chatbots and prompt orchestrators. Backend agents need backend infrastructure.*

[![Stars](https://img.shields.io/github/stars/Agent-Field/agentfield?style=flat&logo=github&logoColor=white&color=d4a24a&labelColor=0c0b09)](https://github.com/Agent-Field/agentfield/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-d4a24a.svg?style=flat&labelColor=0c0b09)](LICENSE)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fsantoshkumarradha%2Fd98e2ad73502b4075f6a5f0ae4f5cae5%2Fraw%2Fbadge.json&style=flat&logo=download&logoColor=white&labelColor=0c0b09&cacheSeconds=3600)](https://github.com/Agent-Field/agentfield)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fsantoshkumarradha%2F433fb09c2d54c3c2589125cfd3eb14a2%2Fraw%2Fbadge.json&style=flat&logo=github&logoColor=white&labelColor=0c0b09&cacheSeconds=3600)](https://github.com/Agent-Field/agentfield/actions/workflows/coverage.yml)
[![Last Commit](https://img.shields.io/github/last-commit/Agent-Field/agentfield?style=flat&logo=git&logoColor=white&color=d4a24a&labelColor=0c0b09)](https://github.com/Agent-Field/agentfield/commits/main)
[![Discord](https://img.shields.io/badge/discord-join%20us-d4a24a.svg?style=flat&labelColor=0c0b09&logo=discord&logoColor=white)](https://discord.gg/aBHaXMkpqh)

<!-- All outbound agentfield.ai links must use UTM parameters, not direct URLs. See assets/utm-links.csv for the full list — target URLs there are for verification and reference only, always use the UTM version in the README. Update both the README and the CSV when adding or changing links. -->
**[Docs](https://agentfield.ai/docs/learn?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-docs)** · **[Quick Start](https://agentfield.ai/docs/learn/quickstart?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-quickstart)** · **[Python SDK](https://agentfield.ai/docs/reference/sdks/python?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-python-sdk)** · **[Go SDK](https://agentfield.ai/docs/reference/sdks/go?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-go-sdk)** · **[TypeScript SDK](https://agentfield.ai/docs/reference/sdks/typescript?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-typescript-sdk)** · **[REST API](https://agentfield.ai/docs/reference/sdks/rest-api?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-rest-api)** · **[Examples](#built-with-agentfield)** · **[Discord](https://discord.gg/aBHaXMkpqh)**

</div>

---

AgentField is an open-source control plane that makes AI agents callable by any service in your stack - frontends, backends, other agents, cron jobs - just like any other API. You write agent logic in Python, Go, or TypeScript. AgentField turns it into production infrastructure: routing, coordination, memory, async execution, and cryptographic audit trails. Every function becomes a REST endpoint. Every agent gets a cryptographic identity. Every decision is traceable.

```python
from agentfield import Agent, AIConfig
from pydantic import BaseModel

app = Agent(
    node_id="claims-processor",
    version="2.1.0",# Canary deploys, A/B testing, blue-green rollouts
    ai_config=AIConfig(model="anthropic/claude-sonnet-4-20250514"),
)

class Decision(BaseModel):
    action: str# "approve", "deny", "escalate"
    confidence: float
    reasoning: str

@app.reasoner(tags=["insurance", "critical"])
async def evaluate_claim(claim: dict) -> dict:

    # Structured AI judgment - returns typed Pydantic output
    decision = await app.ai(
        system="Insurance claims adjuster. Evaluate and decide.",
        user=f"Claim #{claim['id']}: {claim['description']}",
        schema=Decision,
    )

    if decision.confidence < 0.85:
        # Human approval - suspends execution, notifies via webhook, resumes when approved
        await app.pause(
            approval_request_id=f"claim-{claim['id']}",
            approval_request_url=f"https://internal.acme.com/approvals/claim-{claim['id']}",
            expires_in_hours=48,
        )

    # Route to the next agent - traced through the control plane
    await app.call("notifier.send_decision", input={
        "claim_id": claim["id"],
        "decision": decision.model_dump(),
    })

    return decision.model_dump()

app.run()
# This single line exposes: POST /api/v1/execute/claims-processor.evaluate_claim
# The agent auto-registers with the control plane, gets a cryptographic identity, and every
# execution produces a verifiable, tamper-proof audit trail.
```

> **What you just saw:** `app.ai()` calls an LLM and returns structured output. `app.pause()` suspends for [human approval](https://agentfield.ai/docs/build/execution/human-in-the-loop?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-human-in-the-loop). `app.call()` routes to other agents through the control plane. `app.run()` auto-exposes everything as REST. [Read the full docs →](https://agentfield.ai/docs/learn?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-read-full-docs)

## Prompt to production (recommended)

**Describe the system in one line. Get a production-ready multi-agent backend.**

Works in Claude Code, Codex, Gemini CLI, OpenCode, Aider, Windsurf, and Cursor.

```bash
curl -fsSL https://agentfield.ai/install.sh | bash
```

In Claude Code, fire it with the shipped slash command:

```text
/agentfield a claims processor with risk scoring and human approval
```

Or paste any of these directly — the skill auto-matches, no slash command needed:

```text
Build a claims-processor agent with risk scoring, pattern detection,
and human approval for low-confidence decisions.

Build a research agent that spawns parallel investigators and recurses
into deeper sub-questions until the answer has citation-grade provenance.

Build a compliance reviewer for support transcripts — extract claims,
check each against policy, flag violations, emit a signed audit trail.
```

You get a Docker Compose stack already wired up — the agent, the control plane, and a local `curl` you paste into a terminal to try it.

[See it in action →](https://agentfield.ai/docs/learn/build-with-claude-code?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-prompt-to-production)

## Prefer to write it yourself?

```bash
af init my-agent --defaults                            # Scaffold agent
cd my-agent && pip install -r requirements.txt
af server          # Terminal 1 → Dashboard at http://localhost:8080
python main.py     # Terminal 2 → Agent auto-registers
```

```bash
# Call your agent
curl -X POST http://localhost:8080/api/v1/execute/my-agent.demo_echo \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'
```

<details>
<summary><b>Go / TypeScript / Docker</b></summary>

```bash
# Go
af init my-agent --defaults --language go && cd my-agent && go run .

# TypeScript
af init my-agent --defaults --language typescript && cd my-agent && npm install && npm run dev

# Docker (control plane only)
docker run -p 8080:8080 agentfield/control-plane:latest
```

[Deployment guide →](https://agentfield.ai/docs/reference/deploy?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-deploy) for Docker Compose, Kubernetes, and production setups.

</details>

## What You Get

**Build** - Python, Go, or TypeScript. Every function becomes a REST endpoint.

- **[Reasoners & Skills](https://agentfield.ai/docs/build/building-blocks/reasoners?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-reasoners)** - `@app.reasoner()` for AI judgment, `@app.skill()` for deterministic code
- **[Structured AI](https://agentfield.ai/docs/reference/sdks/python?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-structured-ai)** - `app.ai(schema=MyModel)` → typed Pydantic/Zod output from any LLM
- **[Harness](https://agentfield.ai/docs/build/intelligence/harness?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-harness)** - `app.harness("Fix the bug")` dispatches multi-turn tasks to Claude Code, Codex, Gemini CLI, or OpenCode
- **[Cross-Agent Calls](https://agentfield.ai/docs/build/coordination/cross-agent-calls?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-cross-agent-calls)** - `app.call("other-agent.func")` routes through the control plane with full tracing
- **[Discovery](https://agentfield.ai/docs/reference/sdks/python?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-discovery)** - `app.discover(tags=["ml*"])` finds agents and capabilities across the mesh. `tools="discover"` lets LLMs auto-invoke them.
- **[Memory](https://agentfield.ai/docs/build/coordination/shared-memory?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-memory)** - `app.memory.set()` / `.get()` / `.search()` - KV + vector search, four scopes, no Redis needed

**Run** - Production infrastructure for non-deterministic AI.

- **[Async Execution](https://agentfield.ai/docs/build/execution/async?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-async-execution)** - Fire-and-forget with webhooks, SSE streaming, retries. No timeout limits - agents run for hours or days.
- **[Human-in-the-Loop](https://agentfield.ai/docs/build/execution/human-in-the-loop?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-human-in-the-loop)** - `app.pause()` suspends execution for human approval. Crash-safe, durable, audited.
- **[Canary Deployments](https://agentfield.ai/docs/learn/features?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-canary-deployments)** - Traffic weight routing, A/B testing, blue-green deploys. Roll out agent versions at 5% → 50% → 100%.
- **[Observability](https://agentfield.ai/docs/learn/features?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-observability)** - Automatic workflow DAGs, Prometheus `/metrics`, structured logs, execution timeline.

**Govern** - IAM for AI agents. Identity, access control, and audit trails - built in.

- **[Cryptographic Identity](https://agentfield.ai/docs/build/governance/identity?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-crypto-identity)** - Every agent gets a W3C DID (decentralized identifier) - not a shared API key. Agents authenticate to each other the way services authenticate with mTLS, but with cryptographic signatures that travel with the agent.
- **[Verifiable Credentials](https://agentfield.ai/docs/build/governance/credentials?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-verifiable-credentials)** - Tamper-proof receipt for every execution. Offline-verifiable: `af vc verify audit.json`.
- **[Policy Enforcement](https://agentfield.ai/docs/build/governance/policy?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-policy-enforcement)** - Tag-based policy gates with cryptographic verification. "Only agents tagged 'finance' can call this" - enforced by infrastructure, not prompts.

[See the full production-ready feature set →](https://agentfield.ai/docs/learn/features?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-full-features)

<div align="center">
<img src="assets/features-strip.png" alt="90+ Production Features" width="100%" />
</div>

<details>
<summary><h4 align="center">▼ Click to expand full capabilities</h4></summary>

#### AI & LLM

| Feature | How |
|---|---|
| Structured output (Pydantic/Zod) | `app.ai(schema=MyModel)` |
| Multi-turn coding agents | `app.harness("task", provider="claude-code")` |
| LLM auto-discovers agents and tools | `app.ai(tools="discover")` |
| Multimodal (text, image, audio) | `app.ai("Describe", image_url="...")` |
| Streaming responses | `app.ai("...", stream=True)` |
| 100+ LLMs via LiteLLM | `AIConfig(model="anthropic/claude-sonnet-4-20250514")` |
| Temperature, max tokens, format | `app.ai(..., temperature=0.2)` |

#### Agent Mesh & Discovery

| Feature | How |
|---|---|
| Cross-agent calls with tracing | `app.call("agent.func", input={...})` |
| Discover agents by tag (wildcards) | `app.discover(tags=["ml*"])` |
| Discover by health status | `app.discover(health_status="active")` |
| Agent routers (namespacing) | `AgentRouter(prefix="billing")` |
| Auto context propagation | Workflow, session, actor IDs forwarded |
| Parallel agent execution | `asyncio.gather(app.call(...), ...)` |
| Auto-registration on startup | Service mesh with zero config |

#### Execution Engine

| Feature | How |
|---|---|
| Sync execution (REST) | `POST /api/v1/execute/{agent}.{func}` |
| Async (fire-and-forget) | `POST /api/v1/execute/async/{agent}.{func}` |
| Webhooks + HMAC-SHA256 signing | `AsyncConfig(webhook_url="...", secret="...")` |
| SSE streaming (real-time) | `/api/v1/execute/stream/{id}` |
| No timeout limits (hours/days) | Control plane allows unlimited duration |
| Execution polling | `GET /api/v1/executions/{id}` |
| Batch status checks | `POST /api/v1/executions/batch-status` |
| Progress updates mid-execution | Intermediate payloads during long tasks |
| Auto retries + exponential backoff | Transparent - control plane handles |
| Backpressure + queue depth limits | Fair scheduling, circuit breakers |
| Durable queue (PostgreSQL) | Atomic lease-based processing |

#### Memory (Distributed State)

| Feature | How |
|---|---|
| Key-value storage | `app.memory.set(key, value)` / `.get(key)` |
| Vector search (semantic) | `app.memory.search(embedding, top_k=5)` |
| Four scopes | Global, agent, session, run |
| Reactive memory events | `@app.memory.on_change("order_*")` |
| Metadata filtering | Filter stored values by metadata |
| Zero dependencies | Built into control plane - no Redis |

#### Human-in-the-Loop

| Feature | How |
|---|---|
| Durable pause/resume | `await app.pause(reason="...")` |
| Approval workflows with UI | `approval_request_url` for reviewers |
| Configurable timeouts | `expires_in_hours=24` + auto-escalation |
| Crash-safe state | Survives agent restarts |

#### Canary Deployments & Versioning

| Feature | How |
|---|---|
| Traffic weight routing | 5% → 50% → 100% rollouts |
| A/B testing | 50/50 splits with `X-Routed-Version` |
| Blue-green deployments | Instant weight switch, zero downtime |
| Per-version health tracking | Unhealthy versions auto-removed |
| Agent lifecycle states | pending → starting → ready → degraded → offline |

#### Identity & Governance

| Feature | How |
|---|---|
| Cryptographic identity per agent | Auto-generated W3C DID + Ed25519 keys |
| Verifiable Credentials | Tamper-proof receipt per execution |
| Offline VC verification | `af vc verify audit.json` |
| Tag-based access policies | ALLOW/DENY rules on caller → target tags |
| Cryptographically signed requests | Ed25519 signatures on cross-agent calls |
| VC hierarchy (3 tiers) | Platform → Node → Function control |
| Agent notes (audit log) | `app.note("Decision", tags=["critical"])` |
| Non-repudiation | Cryptographic proof of actions |
| Permission request workflows | Auto-created when access denied |

#### Observability & Fleet Management

| Feature | How |
|---|---|
| Automatic DAG visualization | Workflow graphs in dashboard |
| Prometheus metrics | `/metrics` out of the box |
| Structured JSON logging | Automatic from SDK |
| Execution timeline | Chronological decision trace |
| Health checks (K8s-ready) | `/health`, `/ready` endpoints |
| Correlation IDs | `X-Workflow-ID`, `X-Execution-ID` |
| Workflow DAG API | `GET /api/v1/workflows/{id}/dag` |
| Agent heartbeat monitoring | Auto health status transitions |

#### Harness (Multi-turn Coding Agents)

| Feature | How |
|---|---|
| 4 providers | Claude Code, Codex, Gemini CLI, OpenCode |
| Schema-constrained output | `schema=ResultModel` (Pydantic/Zod) |
| Cost capping | `max_budget_usd=3.0` |
| Turn limiting | `max_turns=100` |
| Tool access control | `tools=["Read", "Write", "Bash"]` |
| Environment injection | `env={"KEY": "value"}` |
| System prompt override | `system_prompt="..."` |
| Multi-layer output recovery | Cosmetic repair → retry → full retry |

#### Connector API (Fleet Management)

| Feature | How |
|---|---|
| Remote agent management | `/connector/reasoners` |
| Version traffic control | `/connector/.../weight` |
| Bearer token auth | `AGENTFIELD_CONNECTOR_TOKEN` |
| Air-gapped deployment | Outbound WebSocket only |

#### Developer Experience

| Feature | How |
|---|---|
| CLI scaffolding | `af init my-agent --defaults --language python\|go\|typescript` |
| Local dev with dashboard | `af server` → http://localhost:8080 |
| Hot reload | `af dev` auto-detects changes |
| Auto-REST from decorators | Every `@app.reasoner()` → `POST /api/v1/execute/...` |
| Python, Go, TypeScript SDKs | Native patterns per language |
| MCP server integration | `af add --mcp --url <server>` |
| Config storage API | `POST /api/v1/configs/:key` - database-backed |
| Docker + Kubernetes ready | Stateless control plane, horizontal scaling |

[Explore all features in detail →](https://agentfield.ai/docs/learn/features?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-explore-features)

</details>

## Built With AgentField

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://agentfield.ai/github/swe-af/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-swe-af-repo">
        <img src="assets/examples/autonomous-engineering-team.png" alt="Autonomous Engineering Team" />
      </a>
      <br/>
      <b>Autonomous Engineering Team</b>
      <br/>
      <sub>One API call spins up PM, architect, coders, QA, reviewers - hundreds of coordinated agents that plan, build, test, and ship.</sub>
      <br/><br/>
      <a href="https://agentfield.ai/github/swe-af/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-swe-af-repo">View project →</a>
    </td>
    <td align="center" width="50%">
      <a href="https://agentfield.ai/github/deepresearch/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-deepresearch-repo">
        <img src="assets/examples/deep-research-engine.png" alt="Deep Research Engine" />
      </a>
      <br/>
      <b>Deep Research Engine</b>
      <br/>
      <sub>Recursive research backend. Spawns parallel agents, evaluates quality, generates deeper agents, and recurses -10,000+ agents per query.</sub>
      <br/><br/>
      <a href="https://agentfield.ai/github/deepresearch/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-deepresearch-repo">View project →</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://agentfield.ai/github/mongodb/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-mongodb-repo">
        <img src="assets/examples/reactive-database-layer.png" alt="Reactive MongoDB Intelligence" />
      </a>
      <br/>
      <b>Reactive MongoDB Intelligence</b>
      <br/>
      <sub>Atlas Triggers + agent reasoning. Documents arrive raw and leave enriched - risk scores, pattern detection, evidence chains.</sub>
      <br/><br/>
      <a href="https://agentfield.ai/github/mongodb/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-mongodb-repo">View project →</a>
    </td>
    <td align="center" width="50%">
      <a href="https://agentfield.ai/github/sec-af/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-sec-af-repo">
        <img src="assets/examples/ai-security-auditor.png" alt="Autonomous Security Audit" />
      </a>
      <br/>
      <b>Autonomous Security Audit</b>
      <br/>
      <sub>250 coordinated agents trace every vulnerability source-to-sink and adversarially verify each finding. Confirmed exploits, not pattern flags.</sub>
      <br/><br/>
      <a href="https://agentfield.ai/github/sec-af/?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-sec-af-repo">View project →</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://agentfield.ai/github/cloudsecurity/?utm_source=github-readme&utm_campaign=github-readme&utm_content=cloudsec&utm_id=github-readme-cloudsec-repo">
        <img src="assets/examples/cloud-security.png" alt="CloudSecurity AF" />
      </a>
      <br/>
      <b>CloudSecurity AF</b>
      <br/>
      <sub>AI-native cloud infrastructure security scanner that performs shift-left attack path analysis directly from IaC, prioritizing the most dangerous risk chains before deployment.</sub>
      <br/><br/>
      <a href="https://agentfield.ai/github/cloudsecurity/?utm_source=github-readme&utm_campaign=github-readme&utm_content=cloudsec&utm_id=github-readme-cloudsec-repo">View project →</a>
    </td>
    <td align="center" width="50%">
    </td>
  </tr>
</table>

[See all examples →](https://agentfield.ai/docs/learn/examples?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-see-all-examples)

## See It In Action

<div align="center">
<img src="assets/UI.png" alt="AgentField Dashboard" width="100%" />
<br/>
<sub>Real-time workflow DAGs · Execution traces · Agent fleet management · Audit trails</sub>
</div>

## Architecture

<div align="center">
<img src="assets/arch.png" alt="AgentField Architecture" width="100%" />
</div>

The control plane is a stateless Go service. Agents connect from anywhere - your laptop, Docker, Kubernetes. They register capabilities, the control plane routes calls between them, tracks execution as DAGs, and enforces policies. [Full architecture docs →](https://agentfield.ai/docs/learn/architecture?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-architecture)

## Is AgentField for you?

**Yes if** you’re building beyond chatbots or small multi-agent workflows. If your agents are making decisions inside backend systems like approving refunds, processing claims, coordinating research, or running code, and you need routing, async execution, tracing, and audit trails.

**Not yet if** you’re still in the chatbot or early workflow stage, tools like LangChain or CrewAI are a great fit to explore and iterate. When you start pushing toward larger, production-grade agent systems, that’s where we come in.

## Learn More

- **[The AI Backend](https://agentfield.ai/blog/ai-backend?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-blog-ai-backend)** - Our thesis on why every backend needs a reasoning layer
- **[IAM for AI Backends](https://agentfield.ai/blog/iam-ai-backends?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-blog-iam)** - Why agents need identity, not API keys
- **[vs Agent Frameworks](https://agentfield.ai/docs/learn/vs-frameworks?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-vs-frameworks)** - How AgentField compares to LangChain, CrewAI, and workflow engines
- **[Full Documentation](https://agentfield.ai/docs/learn?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-full-docs)**

## Community

<div align="center">

[![Discord](https://img.shields.io/badge/Join%20Discord-d4a24a?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/aBHaXMkpqh)
[![Twitter](https://img.shields.io/badge/Follow%20on%20X-0c0b09?style=for-the-badge&logo=x&logoColor=white)](https://x.com/agentfield_ai)

**[GitHub Issues](https://github.com/Agent-Field/agentfield/issues)** · **[Documentation](https://agentfield.ai/docs/learn?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-community-docs)** · **[Examples](https://agentfield.ai/docs/learn/examples?utm_source=github-readme&utm_campaign=github-readme&utm_id=github-readme-community-examples)**

</div>

## License

[Apache 2.0](LICENSE)
