"""Shared product context used across documentation chatbot modules."""

PRODUCT_CONTEXT = """
## Product Overview
AgentField is a Kubernetes-style control plane with IAM for building next generation of autonomous software. It provides production infrastructure
for deploying, orchestrating, and observing multi-agent systems with cryptographic identity and audit trails.

**Architecture**: Distributed control plane + independent agent nodes. Think "Kubernetes for AI agents."

**Positioning**: AgentField is infrastructure, not an application framework. While agent frameworks help you build
single AI applications, AgentField provides the orchestration layer for
deploying and managing distributed multi-agent systems in production (like Kubernetes orchestrates containers).

## Design Philosophy

**Infrastructure-First Approach:**
- Control plane handles routing, identity, memory, and observability centrally
- Agents run as independent microservices, not embedded libraries
- Teams deploy agents independently without coordination
- Every agent function becomes a REST API automatically
- Stateless control plane enables horizontal scaling

**Production-Grade Guarantees:**
- Cryptographic identity for every agent and execution (W3C DID standard)
- Tamper-proof audit trails via Verifiable Credentials
- Zero-config distributed state management
- No timeout limits for async workflows (hours or days)
- Observable by default (workflow DAGs, execution traces, agent notes)

**Built for Multi-Team Scale:**
- Independent agent deployment (no monolithic coordination)
- Service discovery through control plane
- Shared memory fabric across distributed agents
- Cross-agent communication via REST APIs
- Works with any tech stack (Python, Go, React, mobile, .NET, etc.)

## Core Concepts & Terminology

**Agent Primitives:**
- **Reasoners**: AI-guided decision making functions (use LLMs for judgment)
- **Skills**: Deterministic functions (reliable execution, no AI)
- **Agent Nodes**: Independent services that register with the control plane
- **Control Plane**: Central orchestration server (handles routing, memory, identity)

**Identity & Trust:**
- **DIDs** (Decentralized Identifiers): Cryptographic identity for agents (W3C standard)
- **VCs** (Verifiable Credentials): Tamper-proof execution records (W3C standard)
- **Workflow DAGs**: Visual representation of agent execution chains
- **Security Model**: Every execution cryptographically signed and attributable
- **Audit Compliance**: Exportable proof chains for regulatory/compliance requirements
- **Zero-Trust Architecture**: Agents authenticate via DIDs, not shared secrets

**State Management:**
- **Memory Scopes**: Hierarchical state sharing (global, actor, session, workflow)
- **Zero-config memory**: Automatic state synchronization across distributed agents
- **Memory events**: Real-time reactive patterns (on_change listeners)

**Execution Patterns:**
- **Sync execution**: `/api/v1/execute/` (90 second timeout)
- **Async execution**: `/api/v1/execute/async/` (no timeout limits, hours/days)
- **Webhooks**: Callback URLs for async results
- **Cross-agent calls**: `app.call("agent.function")` for agent-to-agent communication

**Scalability & Production Architecture:**
- **Stateless Control Plane**: No session affinity, horizontal scaling to billions of requests
- **Independent Agent Scaling**: Each agent scales independently based on its load
- **Zero Coordination Overhead**: Agents don't need to know about each other to deploy
- **Deployment Flexibility**: Laptop → Docker → Kubernetes with same codebase, zero rewrites
- **Storage Tiers**: Local (SQLite/BoltDB) for dev, PostgreSQL for production/cloud
- **Failure Isolation**: Agent failures don't cascade; control plane handles routing around issues

**CLI Commands:**
- `af init`: Create new agent (Python, Go, or TypeScript)
- `af server`: Start control plane
- `af run`: Run agent locally
- `af dev`: Development mode with hot reload

**Key APIs:**
- `app.ai()`: LLM calls with structured output (Pydantic schemas)
- `app.memory`: State management (get/set/on_change)
- `app.call()`: Cross-agent communication
- `app.note()`: Observable execution notes

## Common Topics & Questions

**Getting Started:**
- Installation and setup (af init, af server)
- Creating first agent (Python, Go, or TypeScript choice)
- Understanding reasoners vs skills
- Basic agent structure and configuration

**Agent Development:**
- Registering reasoners and skills
- Using app.ai() for LLM integration
- Structured output with Pydantic/Go structs
- Router pattern for organizing code
- Agent notes and observability

**Multi-Agent Coordination:**
- Cross-agent communication patterns
- Shared memory and state management
- Memory scopes (when to use which)
- Event-driven workflows with memory.on_change

**Production Deployment:**
- Local development (embedded SQLite/BoltDB)
- Docker deployment
- Kubernetes deployment
- Environment variables and configuration

**Identity & Security:**
- DID generation and management
- Verifiable Credentials for audit trails
- Cryptographic proof of execution

**Advanced Features:**
- Async execution for long-running tasks
- Webhook integration
- Custom memory providers
- Performance optimization
- Testing strategies

## Documentation Structure

The documentation is organized by:
- **Getting Started**: Quick start, installation, first agent
- **Core Concepts**: Reasoners, skills, memory, identity, cross-agent communication
- **Guides**: Deployment, testing, multi-agent patterns, examples
- **API Reference**: Python SDK, Go SDK, TypeScript SDK, CLI commands, REST APIs
- **Examples**: Customer support, research assistant, terminal assistant

## Search Term Relationships

When users ask about:
- "Identity" or "authentication" or "security" → Look for: DIDs, Verifiable Credentials, cryptographic identity, audit trails, W3C standards
- "State" or "data sharing" → Look for: memory, scopes, cross-agent memory
- "Setup" or "getting started" → Look for: installation, af init, quick start
- "Deployment" or "production" → Look for: Docker, Kubernetes, local development, scaling
- "Agent communication" → Look for: app.call, cross-agent, workflows
- "Long-running tasks" → Look for: async execution, webhooks
- "Functions" or "endpoints" → Look for: reasoners, skills, API endpoints
- "Differences" or "comparison" or "vs" → Look for: infrastructure vs framework, control plane vs embedded library, multi-team vs single app, production features
- "Scale" or "scalability" → Look for: stateless control plane, independent scaling, billions of requests, horizontal scaling
- "Architecture" → Look for: distributed architecture, control plane, agent nodes, microservices, stateless design
- "SDK" or "language" or "languages" or "supported" → Look for: Python SDK, Go SDK, TypeScript SDK, npm install @agentfield/sdk, pip install agentfield
"""
