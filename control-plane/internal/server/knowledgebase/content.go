package knowledgebase

// LoadDefaultContent populates the KB with built-in articles and topic descriptions.
func LoadDefaultContent(kb *KB) {
	// Register topics
	kb.RegisterTopic("building", "How to create agents, reasoners, and multi-agent systems on AgentField")
	kb.RegisterTopic("patterns", "Architectural patterns for multi-agent systems")
	kb.RegisterTopic("observability", "Monitoring, metrics, webhooks, and debugging")
	kb.RegisterTopic("identity", "Decentralized identity (DID), verifiable credentials (VC), and authorization")
	kb.RegisterTopic("sdk", "SDK quickstarts and API references")
	kb.RegisterTopic("examples", "Reference implementations and example projects")

	// --- Building ---
	kb.Add(Article{
		ID: "building/reasoner-python", Topic: "building", Title: "Creating a Reasoner in Python",
		Summary:    "How to create a Python agent with @reasoner decorators that registers with the control plane",
		Difficulty: "beginner", SDK: "python",
		Tags:       []string{"reasoner", "python", "decorator", "quickstart"},
		Content: `# Creating a Reasoner in Python

A reasoner is a function decorated with @reasoner that becomes an HTTP endpoint managed by the control plane.

## Basic Setup

` + "```python" + `
from agentfield import Agent

app = Agent(name="my-agent")

@app.reasoner()
async def analyze(input: dict) -> dict:
    # Your reasoning logic here
    return {"result": "analysis complete"}

if __name__ == "__main__":
    app.run()
` + "```" + `

## Key Points
- Each @reasoner becomes a REST endpoint
- The control plane auto-discovers reasoners on registration
- Input/output are JSON objects
- Use type hints for automatic schema generation
`,
	})

	kb.Add(Article{
		ID: "building/ai-vs-harness", Topic: "building", Title: ".ai() vs .harness() — Choosing the Right Primitive",
		Summary:    "Decision guide for when to use .ai() (single-shot classification) vs .harness() (multi-turn agent)",
		Difficulty: "intermediate", SDK: "python",
		Tags:       []string{"ai", "harness", "primitives", "decision", "architecture"},
		Content: `# .ai() vs .harness()

## .ai() — Fast Structured Classification
Single-shot, no tools, no state. Input in, flat schema out.
- Flat Pydantic schema: 2-4 attributes max
- Input must fit in one context window (~500-2000 tokens)
- Best for: intake classification, routing gates, binary decisions

## .harness() — The Atomic Unit of Intelligence
Stateful, multi-turn, tool-using agent.
- Has tool access: reads files, calls APIs, invokes sub-agents
- Multi-turn: reads X, decides what to read next
- Can meta-prompt: use LLM reasoning to craft specific prompts

## Decision Tree
- Needs to read/navigate documents? → .harness()
- Process more than ~3,000 tokens? → .harness()
- Multi-turn decisions? → .harness()
- Fast classification (< 500 tokens)? → .ai()
- Simple routing decision (enum output)? → .ai()

When in doubt, use .harness(). Reserve .ai() for gates and classifiers.
`,
	})

	kb.Add(Article{
		ID: "building/memory-scopes", Topic: "building", Title: "Memory Scopes and Usage",
		Summary:    "Understanding global, agent, session, and run memory scopes",
		Difficulty: "beginner",
		Tags:       []string{"memory", "scopes", "state", "persistence"},
		Content: `# Memory Scopes

AgentField provides four memory scopes:

| Scope | Shared Across | Use Case |
|-------|---------------|----------|
| global | All agents, all sessions | Shared knowledge base |
| agent | One agent, all sessions | Agent-specific config |
| session | One session | Multi-turn conversation state |
| run | Single execution run | Workflow-scoped data |

## Usage (Python SDK)
` + "```python" + `
await agent.memory.set("session", session_id, "user_preference", "dark_mode")
value = await agent.memory.get("session", session_id, "user_preference")
` + "```" + `

## API Endpoints
- POST /api/v1/memory/set — Set a value
- POST /api/v1/memory/get — Get a value
- POST /api/v1/memory/delete — Delete a value
- GET /api/v1/memory/list — List all values in a scope
`,
	})

	kb.Add(Article{
		ID: "building/tool-calling", Topic: "building", Title: "Tool Calling in Harnesses",
		Summary:    "How to define and use tools within .harness() agents",
		Difficulty: "intermediate", SDK: "python",
		Tags:       []string{"tools", "harness", "function-calling"},
		Content: `# Tool Calling

Harnesses can use tools to interact with external systems.

## Defining Tools
` + "```python" + `
@app.tool()
async def search_database(query: str) -> list:
    """Search the database for matching records."""
    results = await db.search(query)
    return results

@app.reasoner()
async def researcher(input: dict) -> dict:
    result = await app.harness(
        prompt="Research the topic and find relevant data",
        input=input,
        tools=[search_database],
    )
    return result
` + "```" + `

## Key Points
- Tools are decorated functions with type hints
- The LLM decides when and how to call tools
- Tools should have clear docstrings (used as tool descriptions)
- Harnesses orchestrate tool usage across multiple turns
`,
	})

	kb.Add(Article{
		ID: "building/archei-rules", Topic: "building", Title: "Inter-Agent Data Flow (Archei Rules)",
		Summary:    "Rules for data format between agents: structured JSON vs string vs hybrid",
		Difficulty: "intermediate",
		Tags:       []string{"archei", "data-flow", "inter-agent", "json", "string"},
		Content: `# Archei Rules — Inter-Agent Data Flow

| Data Purpose | Format | Why |
|---|---|---|
| Drives programmatic routing | Structured JSON | Code consumes it |
| Context for another LLM | String | LLMs reason over natural language |
| Both | Hybrid | Scores as JSON, reasoning as string |

## Red Flags
- Parsing strings with regex → switch to structured JSON
- ` + "`if \"critical\" in output.text`" + ` → use enum field
- Passing JSON to LLM that reads it as text → use string
`,
	})

	kb.Add(Article{
		ID: "building/multi-agent-basics", Topic: "building", Title: "Building Multi-Agent Systems",
		Summary:    "How to compose multiple agents into a system using the control plane",
		Difficulty: "intermediate",
		Tags:       []string{"multi-agent", "composition", "orchestration", "workflow"},
		Content: `# Multi-Agent Systems

## Agent-to-Agent Communication
All communication goes through the control plane:
` + "```python" + `
result = await app.call("other-agent.reasoner_name", input={"query": "analyze this"})
` + "```" + `

## Workflow DAG
The control plane automatically tracks the call graph (DAG) of all agent interactions.
View it at: GET /api/ui/v1/workflows/:workflowId/dag

## Key Principles
1. Never direct agent-to-agent HTTP — always through control plane
2. The control plane handles routing, tracking, and metrics
3. Each agent registers its capabilities on startup
4. Use run_id to group related executions
`,
	})

	kb.Add(Article{
		ID: "building/budget-caps", Topic: "building", Title: "Budget Caps and Cost Control",
		Summary:    "Setting execution budgets to prevent runaway costs in adaptive agent systems",
		Difficulty: "intermediate",
		Tags:       []string{"budget", "cost", "limits", "adaptive"},
		Content: `# Budget Caps

Every adaptive mechanism needs hard limits:

| Loop | Budget | Example |
|------|--------|---------|
| Inner (per-agent) | Max N follows, 1 escalation | Max 5 reference follows |
| Middle (cross-agent) | Max N sub-agent spawns | Max 3 deep-dives |
| Outer (pipeline) | Max N iterations | Max 2 coverage passes |

## Implementation
Set budget caps in your agent configuration or at call time:
` + "```python" + `
result = await app.harness(
    prompt="Investigate thoroughly",
    input=data,
    max_turns=10,         # Inner loop cap
    max_sub_agents=3,     # Middle loop cap
)
` + "```" + `

Without caps, adaptive systems become unbounded cost sinks.
`,
	})

	kb.Add(Article{
		ID: "building/error-handling", Topic: "building", Title: "Error Handling and .ai() Fallbacks",
		Summary:    "Designing graceful .ai() to .harness() fallback chains",
		Difficulty: "intermediate", SDK: "python",
		Tags:       []string{"error-handling", "fallback", "ai", "harness", "resilience"},
		Content: `# Error Handling & Fallbacks

## The .ai() Fallback Pattern

Every .ai() call should have a fallback when input doesn't fit:

` + "```python" + `
class IntakeResult(BaseModel):
    contract_type: str
    confident: bool

intake = await app.ai(
    prompt="Classify this document",
    input={"text": first_pages},
    schema=IntakeResult,
)

if not intake.confident:
    intake = await app.call("my-project.intake_harness", input=full_doc)
` + "```" + `

## Strategies
1. Confidence flag in schema
2. Validation check after .ai()
3. Try/catch with .harness() fallback
`,
	})

	kb.Add(Article{
		ID: "building/reasoner-go", Topic: "building", Title: "Creating Agents in Go",
		Summary:    "How to build Go agents with skills using the Go SDK",
		Difficulty: "beginner", SDK: "go",
		Tags:       []string{"go", "skill", "sdk", "quickstart"},
		Content: `# Creating Agents in Go

` + "```go" + `
import agentfieldagent "github.com/Agent-Field/agentfield/sdk/go/agent"

agent, _ := agentfieldagent.New(agentfieldagent.Config{
    NodeID:        "my-agent",
    AgentFieldURL: "http://localhost:8080",
})

agent.RegisterSkill("greet", func(ctx context.Context, input map[string]any) (any, error) {
    return map[string]any{"message": "hello"}, nil
})

agent.Run(context.Background())
` + "```" + `

## Key Points
- Go SDK uses "skills" instead of "reasoners"
- Skills are functions registered with the agent builder
- The agent auto-registers with the control plane on startup
`,
	})

	kb.Add(Article{
		ID: "building/execution-lifecycle", Topic: "building", Title: "Execution Lifecycle",
		Summary:    "Understanding execution states: pending, running, completed, failed, cancelled, paused",
		Difficulty: "beginner",
		Tags:       []string{"execution", "lifecycle", "status", "state-machine"},
		Content: `# Execution Lifecycle

## States
- pending → running → completed
- pending → running → failed
- running → paused → running (resume)
- running → cancelled

## API Endpoints
- POST /api/v1/execute/:target — Sync execution
- POST /api/v1/execute/async/:target — Async execution
- GET /api/v1/executions/:id — Check status
- POST /api/v1/executions/:id/cancel — Cancel
- POST /api/v1/executions/:id/pause — Pause
- POST /api/v1/executions/:id/resume — Resume

## Workflow Tracking
Each execution gets a unique execution_id. Related executions share a run_id.
The DAG of executions is viewable at: GET /api/ui/v1/workflows/:workflowId/dag
`,
	})

	// --- Patterns ---
	kb.Add(Article{
		ID: "patterns/hunt-prove", Topic: "patterns", Title: "HUNT→PROVE Adversarial Pattern",
		Summary:    "Multiple hunters find issues, adversarial provers verify them to reduce false positives",
		Difficulty: "advanced",
		Tags:       []string{"hunt", "prove", "adversarial", "false-positives", "security"},
		Content: `# HUNT→PROVE

One set of agents finds potential issues. A separate set tries to disprove them.

## Architecture
Input → [Hunter A, B, C] → Findings Queue → [Prover A, B] → Verified Findings

## When to Use
Any problem where false positives are costly: security, legal, compliance, medical.

## Reference
See sec-af/ for a complete implementation.
`,
	})

	kb.Add(Article{
		ID: "patterns/fan-out-recurse", Topic: "patterns", Title: "Fan-Out → Filter → Gap-Find → Recurse",
		Summary:    "Breadth-first exploration followed by quality-gated recursion",
		Difficulty: "advanced",
		Tags:       []string{"fan-out", "recursion", "research", "comprehensive"},
		Content: `# Fan-Out → Recurse

Breadth-first exploration → quality filter → gap detection → recursive depth.

## When to Use
Research, due diligence, compliance audits — any problem requiring comprehensive coverage.

## Reference
See af-deep-research/ for implementation.
`,
	})

	kb.Add(Article{
		ID: "patterns/factory-loops", Topic: "patterns", Title: "Factory Control Loops (Inner/Middle/Outer)",
		Summary:    "Three nested control loops for long-running multi-step execution with adaptive replanning",
		Difficulty: "advanced",
		Tags:       []string{"factory", "loops", "control", "adaptive", "replanning"},
		Content: `# Factory Control Loops

Three nested loops for adaptive long-running execution:

| Loop | Scope | Budget |
|------|-------|--------|
| Inner | Per-agent self-adaptation | Max N follows |
| Middle | Cross-agent deep-dives | Max N spawns |
| Outer | Pipeline coverage | Max N iterations |

## Reference
See af-swe/ for implementation.
`,
	})

	kb.Add(Article{
		ID: "patterns/streaming-pipeline", Topic: "patterns", Title: "Streaming Pipeline",
		Summary:    "Downstream agents consume findings as they arrive via asyncio.Queue",
		Difficulty: "intermediate",
		Tags:       []string{"streaming", "pipeline", "async", "queue"},
		Content: `# Streaming Pipeline

Downstream agents consume findings as they arrive, overlapping work.

` + "```python" + `
findings_queue = asyncio.Queue()

async def analyst(sections):
    for finding in analyze(sections):
        await findings_queue.put(finding)

async def cross_ref_resolver():
    while True:
        finding = await findings_queue.get()
        check_combinations(finding, previous_findings)
` + "```" + `

## When to Use
Any pipeline where downstream agents can start on partial results.
`,
	})

	kb.Add(Article{
		ID: "patterns/meta-prompting", Topic: "patterns", Title: "Meta-Prompting (Harnesses Spawning Harnesses)",
		Summary:    "Parent harnesses craft investigation prompts for child harnesses at runtime",
		Difficulty: "advanced",
		Tags:       []string{"meta-prompting", "dynamic", "spawn", "harness"},
		Content: `# Meta-Prompting

Parent harnesses use LLM reasoning to craft child harness prompts at runtime.

## Key Properties
- Parent decides WHAT to investigate (intelligence, not script)
- Parent decides HOW to frame investigation
- Child has bounded autonomy
- Parent verifies child output

## When to Use
Investigation path depends on what's discovered during analysis.
`,
	})

	kb.Add(Article{
		ID: "patterns/parallel-hunters", Topic: "patterns", Title: "Parallel Hunters + Signal Cascade",
		Summary:    "Multiple specialized agents analyze different dimensions in parallel",
		Difficulty: "intermediate",
		Tags:       []string{"parallel", "hunters", "cascade", "concurrent"},
		Content: `# Parallel Hunters

Multiple specialized agents analyze different dimensions concurrently.

Input → [Hunter A, Hunter B, Hunter C] → Findings Queue → Downstream

## When to Use
Any problem with multiple independent analysis dimensions.
`,
	})

	kb.Add(Article{
		ID: "patterns/reactive-enrichment", Topic: "patterns", Title: "Reactive Document Enrichment",
		Summary:    "Event-driven trigger → enrichment pipeline → output",
		Difficulty: "intermediate",
		Tags:       []string{"reactive", "event-driven", "enrichment", "trigger"},
		Content: `# Reactive Enrichment

Event-driven: trigger → enrichment pipeline → output.
The engine is domain-agnostic; config defines the domain.

## When to Use
Problems triggered by data arriving: incidents, PRs, contracts, form submissions.

## Reference
See reactive-atlas/ for implementation.
`,
	})

	kb.Add(Article{
		ID: "patterns/three-loops", Topic: "patterns", Title: "Three Nested Control Loops",
		Summary:    "Adaptive depth at inner, middle, and outer levels with hard budget caps",
		Difficulty: "advanced",
		Tags:       []string{"loops", "adaptive", "budget", "control"},
		Content: `# Three Nested Control Loops

| Loop | Scope | Trigger | Budget |
|---|---|---|---|
| Inner | Per-agent | Found reference / critical finding | Max N follows |
| Middle | Cross-agent | Critical combination | Max N sub-agent spawns |
| Outer | Pipeline | Gap in analysis | Max N iterations |

Every loop needs a hard cap to prevent unbounded costs.
`,
	})

	kb.Add(Article{
		ID: "patterns/anti-patterns", Topic: "patterns", Title: "Anti-Patterns to Avoid",
		Summary:    "Common mistakes in multi-agent system design",
		Difficulty: "beginner",
		Tags:       []string{"anti-patterns", "mistakes", "best-practices"},
		Content: `# Anti-Patterns

1. Using .ai() for document analysis — use .harness()
2. Passing full documents as context — have harness extract relevant parts
3. Static dispatch where meta-prompting needed
4. Unbounded loops — always set budget caps
5. Structured JSON between LLMs — use strings for LLM context
6. Replicating programmatic work — use code for deterministic ops
7. Hard-failing .ai() calls — always design fallback paths
`,
	})

	// --- Observability ---
	kb.Add(Article{
		ID: "observability/metrics", Topic: "observability", Title: "Metrics and Prometheus Integration",
		Summary:    "Built-in Prometheus metrics for monitoring agent performance",
		Difficulty: "beginner",
		Tags:       []string{"metrics", "prometheus", "monitoring", "grafana"},
		Content: `# Metrics

AgentField exposes Prometheus metrics at GET /metrics.

## Available Metrics
- Execution counts by status
- Execution duration histograms
- Agent registration events
- Memory operation counts
- HTTP request latency

## Grafana Setup
Point your Grafana instance at the /metrics endpoint for dashboards.
`,
	})

	kb.Add(Article{
		ID: "observability/webhooks", Topic: "observability", Title: "Observability Webhooks",
		Summary:    "Configure webhooks for execution events and status changes",
		Difficulty: "intermediate",
		Tags:       []string{"webhooks", "events", "notifications"},
		Content: `# Observability Webhooks

## API Endpoints
- GET /api/v1/settings/webhooks — List webhooks
- POST /api/v1/settings/webhooks — Create webhook
- DELETE /api/v1/settings/webhooks/:id — Delete webhook

Webhooks fire on execution state changes (started, completed, failed).
`,
	})

	kb.Add(Article{
		ID: "observability/sse-events", Topic: "observability", Title: "Server-Sent Events (SSE) Streaming",
		Summary:    "Real-time streaming of node and execution events",
		Difficulty: "intermediate",
		Tags:       []string{"sse", "streaming", "events", "real-time"},
		Content: `# SSE Streaming

## Available SSE Endpoints
- GET /api/ui/v1/nodes/events — Node status changes
- GET /api/ui/v1/executions/events — Execution events
- GET /api/ui/v1/reasoners/events — Reasoner events
- GET /api/ui/v1/workflows/:id/notes/events — Workflow notes stream

Connect with EventSource or any SSE client for real-time updates.
`,
	})

	kb.Add(Article{
		ID: "observability/dag-visualization", Topic: "observability", Title: "Workflow DAG Visualization",
		Summary:    "Viewing the execution dependency graph for debugging workflows",
		Difficulty: "beginner",
		Tags:       []string{"dag", "workflow", "visualization", "debugging"},
		Content: `# DAG Visualization

## API
GET /api/ui/v1/workflows/:workflowId/dag

Returns the execution dependency graph showing:
- Parent-child execution relationships
- Status of each node
- Timing information
- Agent assignments

## UI
Visit /ui/workflows to see interactive DAG visualizations.
`,
	})

	kb.Add(Article{
		ID: "observability/execution-notes", Topic: "observability", Title: "Execution Notes (app.note())",
		Summary:    "Adding runtime annotations to executions for debugging and communication",
		Difficulty: "beginner",
		Tags:       []string{"notes", "annotations", "debugging", "app.note"},
		Content: `# Execution Notes

Agents can add notes during execution for debugging:

` + "```python" + `
await app.note("Starting phase 2 analysis")
await app.note("Found 3 critical issues", level="warning")
` + "```" + `

## API
- POST /api/v1/executions/note — Add a note
- GET /api/v1/executions/:id/notes — Get notes

Notes are visible in the UI and queryable via the API.
`,
	})

	kb.Add(Article{
		ID: "observability/dashboard", Topic: "observability", Title: "Dashboard API",
		Summary:    "Programmatic access to the AgentField dashboard data",
		Difficulty: "beginner",
		Tags:       []string{"dashboard", "ui", "summary", "stats"},
		Content: `# Dashboard API

## Endpoints
- GET /api/ui/v1/dashboard/summary — System overview
- GET /api/ui/v1/dashboard/enhanced — Detailed dashboard data
- GET /api/ui/v1/executions/stats — Execution statistics
- GET /api/ui/v1/executions/timeline — Hourly aggregated data
- GET /api/ui/v1/executions/recent — Recent activity feed
`,
	})

	// --- Identity ---
	kb.Add(Article{
		ID: "identity/did-setup", Topic: "identity", Title: "DID Setup and Configuration",
		Summary:    "Enabling decentralized identity for agents",
		Difficulty: "intermediate",
		Tags:       []string{"did", "identity", "setup", "configuration"},
		Content: `# DID Setup

## Enable in Config
Set Features.DID.Enabled = true in agentfield.yaml.

## Agent DID Generation
DIDs are auto-generated when agents register with DID features enabled.
Each agent gets a did:web identifier resolvable via standard W3C endpoints.

## Resolution Endpoints
- GET /.well-known/did.json — Server DID document
- GET /agents/:agentID/did.json — Agent DID document
`,
	})

	kb.Add(Article{
		ID: "identity/vc-audit", Topic: "identity", Title: "Verifiable Credentials and Audit Trails",
		Summary:    "Cryptographic audit trails using W3C Verifiable Credentials",
		Difficulty: "advanced",
		Tags:       []string{"vc", "audit", "verifiable-credentials", "cryptographic"},
		Content: `# Verifiable Credentials

## Enable VC Generation
` + "```python" + `
app.vc_generator.set_enabled(True)
` + "```" + `

## Audit Trail
GET /api/v1/did/workflow/:workflow_id/vc-chain

Returns the full chain of VCs for a workflow, verifiable offline.

## Offline Verification
` + "```bash" + `
af verify audit.json
` + "```" + `
`,
	})

	kb.Add(Article{
		ID: "identity/tag-authorization", Topic: "identity", Title: "Tag-Based Authorization",
		Summary:    "RBAC using tags and access policies for agent-to-agent communication",
		Difficulty: "advanced",
		Tags:       []string{"authorization", "tags", "rbac", "access-control"},
		Content: `# Tag-Based Authorization

## How It Works
1. Agents register with tags (e.g., "security-scanner", "production")
2. Admin defines access policies (which tags can call which)
3. Permission middleware enforces policies on execution endpoints

## Admin API
- GET /api/v1/admin/policies — List policies
- POST /api/v1/admin/policies — Create policy
- GET /api/v1/admin/tags/pending — Pending tag approvals
- POST /api/v1/admin/tags/approve — Approve tags
`,
	})

	kb.Add(Article{
		ID: "identity/did-auth", Topic: "identity", Title: "DID-Based Authentication",
		Summary:    "Authenticating API requests using DID signatures",
		Difficulty: "advanced",
		Tags:       []string{"did", "authentication", "signatures", "cryptographic"},
		Content: `# DID Authentication

When enabled, agents can sign requests with their DID keys.
The control plane verifies signatures using the agent's DID document.

## Configuration
Set Features.DID.Authorization.DIDAuthEnabled = true

## Headers
Agents include DID auth headers on requests.
The middleware validates signatures against the DID document.
`,
	})

	kb.Add(Article{
		ID: "identity/offline-verification", Topic: "identity", Title: "Offline VC Verification",
		Summary:    "Verifying execution audit trails without network access",
		Difficulty: "intermediate",
		Tags:       []string{"verification", "offline", "audit", "vc"},
		Content: `# Offline Verification

## Export
` + "```bash" + `
curl http://localhost:8080/api/ui/v1/did/export/vcs > audit.json
` + "```" + `

## Verify
` + "```bash" + `
af verify audit.json
` + "```" + `

The verification checks:
1. VC signature validity
2. Issuer DID resolution
3. Chain integrity (parent-child relationships)
4. Timestamp ordering
`,
	})

	// --- SDK ---
	kb.Add(Article{
		ID: "sdk/python-quickstart", Topic: "sdk", Title: "Python SDK Quickstart",
		Summary:    "Get started with the Python AgentField SDK in 5 minutes",
		Difficulty: "beginner", SDK: "python",
		Tags:       []string{"python", "quickstart", "getting-started", "install"},
		Content: `# Python SDK Quickstart

## Install
` + "```bash" + `
pip install agentfield
` + "```" + `

## Create Agent
` + "```python" + `
from agentfield import Agent

app = Agent(name="my-agent")

@app.reasoner()
async def hello(input: dict) -> dict:
    return {"message": f"Hello, {input.get('name', 'World')}!"}

app.run()
` + "```" + `

## Connect to Control Plane
` + "```bash" + `
export AGENTFIELD_SERVER=http://localhost:8080
python my_agent.py
` + "```" + `
`,
	})

	kb.Add(Article{
		ID: "sdk/go-quickstart", Topic: "sdk", Title: "Go SDK Quickstart",
		Summary:    "Get started with the Go AgentField SDK",
		Difficulty: "beginner", SDK: "go",
		Tags:       []string{"go", "quickstart", "getting-started"},
		Content: `# Go SDK Quickstart

## Install
` + "```bash" + `
go get github.com/Agent-Field/agentfield/sdk/go
` + "```" + `

## Create Agent
` + "```go" + `
agent, _ := agentfieldagent.New(agentfieldagent.Config{
    NodeID:        "my-agent",
    AgentFieldURL: "http://localhost:8080",
})
agent.RegisterSkill("hello", func(ctx context.Context, input map[string]any) (any, error) {
    name, _ := input["name"].(string)
    return map[string]any{"message": "Hello, " + name + "!"}, nil
})
agent.Run(context.Background())
` + "```" + `
`,
	})

	kb.Add(Article{
		ID: "sdk/python-decorators", Topic: "sdk", Title: "Python Decorator Reference",
		Summary:    "Complete reference for @reasoner, @tool, and other Python SDK decorators",
		Difficulty: "intermediate", SDK: "python",
		Tags:       []string{"python", "decorators", "reference", "reasoner", "tool"},
		Content: `# Python Decorator Reference

## @app.reasoner()
Registers a function as a reasoner endpoint.

## @app.tool()
Registers a function as a tool available to harnesses.

## app.ai()
Single-shot LLM call with structured output.

## app.harness()
Multi-turn agent execution with tool access.

## app.call()
Call another agent's reasoner through the control plane.

## app.note()
Add a runtime annotation to the current execution.
`,
	})

	kb.Add(Article{
		ID: "sdk/api-reference", Topic: "sdk", Title: "REST API Reference Summary",
		Summary:    "Overview of all control plane REST API endpoint groups",
		Difficulty: "beginner",
		Tags:       []string{"api", "reference", "rest", "endpoints"},
		Content: `# REST API Reference

## Endpoint Groups
| Group | Base Path | Description |
|-------|-----------|-------------|
| health | /health, /api/v1/health | Health checks |
| discovery | /api/v1/discovery/ | Agent capability discovery |
| nodes | /api/v1/nodes/ | Agent registration and lifecycle |
| execute | /api/v1/execute/ | Execution (sync and async) |
| executions | /api/v1/executions/ | Execution status and management |
| memory | /api/v1/memory/ | Memory CRUD operations |
| did | /api/v1/did/ | DID document resolution |
| agentic | /api/v1/agentic/ | Agent-optimized API layer |
| admin | /api/v1/admin/ | Admin operations |

Use GET /api/v1/agentic/discover?q=<keyword> to search endpoints programmatically.
`,
	})

	kb.Add(Article{
		ID: "sdk/harness-reference", Topic: "sdk", Title: "Harness API Reference",
		Summary:    "Detailed reference for the .harness() primitive",
		Difficulty: "intermediate", SDK: "python",
		Tags:       []string{"harness", "reference", "api", "multi-turn"},
		Content: `# Harness Reference

## Basic Usage
` + "```python" + `
result = await app.harness(
    prompt="Your instructions here",
    input={"data": your_data},
    tools=[tool1, tool2],
    max_turns=10,
)
` + "```" + `

## Parameters
- prompt: Instructions for the agent
- input: Input data (dict)
- tools: List of available tools
- max_turns: Maximum conversation turns
- model: Override the default model
`,
	})

	// --- Examples ---
	kb.Add(Article{
		ID: "examples/sec-af", Topic: "examples", Title: "sec-af — Security Auditor Agent",
		Summary:    "HUNT→PROVE adversarial security scanner with parallel hunters and streaming",
		Difficulty: "advanced",
		Tags:       []string{"security", "audit", "hunt", "prove", "parallel", "streaming"},
		Content: `# sec-af

A security auditor that uses parallel hunter agents to find vulnerabilities,
then adversarial prover agents to verify findings.

## Patterns Used
- HUNT→PROVE adversarial tension
- Parallel hunters + signal cascade
- Streaming pipeline (asyncio.Queue)
- Budget caps per agent

## Location
code/examples/sec-af/
`,
	})

	kb.Add(Article{
		ID: "examples/af-swe", Topic: "examples", Title: "af-swe — Software Engineering Agent",
		Summary:    "Factory control loops, topological DAG, parallel worktrees for autonomous coding",
		Difficulty: "advanced",
		Tags:       []string{"swe", "coding", "factory", "loops", "dag", "worktrees"},
		Content: `# af-swe

The canonical SWE agent implementation using AgentField.

## Patterns Used
- Factory control loops (inner/middle/outer)
- Topological DAG execution
- Parallel git worktrees
- Adaptive replanning

## Location
code/examples/af-swe/
`,
	})

	kb.Add(Article{
		ID: "examples/af-deep-research", Topic: "examples", Title: "af-deep-research — Autonomous Research",
		Summary:    "Fan-out → filter → gap-find → recurse pattern for comprehensive research",
		Difficulty: "advanced",
		Tags:       []string{"research", "deep-research", "fan-out", "recursive"},
		Content: `# af-deep-research

Autonomous research backend using recursive exploration.

## Patterns Used
- Fan-out → filter → gap-find → recurse
- Quality-gated recursion
- Source diversity

## Location
code/examples/af-deep-research/
`,
	})

	kb.Add(Article{
		ID: "examples/hello-world", Topic: "examples", Title: "Hello World Agent",
		Summary:    "Minimal agent example to verify your setup works",
		Difficulty: "beginner",
		Tags:       []string{"hello-world", "minimal", "quickstart", "beginner"},
		Content: `# Hello World Agent

## Create
` + "```bash" + `
af init hello-world
cd hello-world
` + "```" + `

## Run
` + "```bash" + `
af dev  # or: af run
` + "```" + `

## Test
` + "```bash" + `
curl -X POST http://localhost:8080/api/v1/execute/hello-world.hello \
  -H "Content-Type: application/json" \
  -d '{"input": {"name": "World"}}'
` + "```" + `
`,
	})

	kb.Add(Article{
		ID: "examples/tool-calling", Topic: "examples", Title: "Tool Calling Example",
		Summary:    "Example agent demonstrating tool definition and usage",
		Difficulty: "intermediate",
		Tags:       []string{"tools", "function-calling", "example"},
		Content: `# Tool Calling Example

Demonstrates defining tools and using them in harnesses.
See building/tool-calling article for the full pattern.
`,
	})

	kb.Add(Article{
		ID: "examples/agentic-rag", Topic: "examples", Title: "Agentic RAG Example",
		Summary:    "Retrieval-augmented generation with agent-controlled retrieval",
		Difficulty: "intermediate",
		Tags:       []string{"rag", "retrieval", "augmented-generation", "vector"},
		Content: `# Agentic RAG

Uses vector memory + harness for agent-controlled retrieval:

1. Store documents as vectors: POST /api/v1/memory/vector
2. Agent queries with: POST /api/v1/memory/vector/search
3. Agent reasons over retrieved context
4. Agent decides if more retrieval is needed (agentic loop)
`,
	})
}
