# AgentField Test Coverage Audit

**Date:** 2026-04-05
**Branch:** `feature/test-coverage-improvements`
**Audited by:** Parallel Gemini + manual scan

---

## Executive Summary

| Component | Source Files | Test Files | Coverage % | P0 Gaps | P1 Gaps |
|-----------|-------------|-----------|-----------|---------|---------|
| Control Plane (Go) | 181 | 68 tested | 37.6% | 14 | 22 |
| Python SDK | 53 | 45 tested | 84.9% | 2 | 4 |
| Go SDK | 36 | 21 tested | 58.3% | 1 | 4 |
| TypeScript SDK | 49 | 27 tested | 55.1% | 5 | 8 |
| Web UI (React) | 326 | 0 tested | 0% | N/A | N/A |
| **TOTAL** | **645** | **161 tested** | **25%** | **22** | **38** |

**Highest risk areas (most likely to cause production breakage):**
1. **Storage layer** — entire storage/ package untested (CRUD for all entities)
2. **Memory handlers** — memory read/write/events with no coverage
3. **MCP subsystem** — both control plane and SDK MCP code untested
4. **Execution state validation** — state machine transitions unchecked
5. **Verification/DID** — security-critical code with gaps

---

## P0 — CRITICAL (Will Break Users)

### Control Plane Go

| File | What It Does | Risk |
|------|-------------|------|
| `storage/storage.go` | Main storage interface + initialization | All data operations route through here |
| `storage/local.go` | SQLite/BoltDB backend for local mode | Every `af dev` user hits this |
| `storage/execution_records.go` | Execution CRUD (already has tests but storage.go doesn't) | Execution tracking breaks |
| `storage/execution_state_validation.go` | State machine: pending→running→done | Invalid state transitions corrupt data |
| `storage/events.go` | Event storage for SSE streaming | UI goes blind |
| `storage/models.go` | GORM model definitions | Schema drift = silent data loss |
| `handlers/memory.go` | Memory GET/SET/DELETE endpoints | Agent memory broken |
| `handlers/memory_access_control.go` | Memory permission enforcement | Security bypass |
| `handlers/memory_events.go` | Memory change notifications | Agent coordination broken |
| `handlers/nodes_rest.go` | Node registration/heartbeat REST | Agents can't connect |
| `handlers/reasoners.go` | Reasoner registration/listing | Skill discovery broken |
| `handlers/config_storage.go` | Config persistence endpoints | Settings lost |
| `events/execution_events.go` | Execution lifecycle event emission | No UI updates |
| `events/node_events.go` | Node status change events | Dashboard stale |

### Python SDK

| File | What It Does | Risk |
|------|-------------|------|
| `verification.py` | `LocalVerifier` — signature verification, policy evaluation | **Security**: unauthorized access or DOS |
| `agent.py` (partial) | `_PauseManager`, `handle_serverless`, DID initialization | Core lifecycle failures |

### Go SDK

| File | What It Does | Risk |
|------|-------------|------|
| `agent/harness.go` | Primary `Agent.Harness()` entry point | All harness-based skills break |

### TypeScript SDK

| File | What It Does | Risk |
|------|-------------|------|
| `client/AgentFieldClient.ts` | HTTP client to control plane | All TS agent communication |
| `agent/Agent.ts` | Agent lifecycle, registration, skill routing | Core agent functionality |
| `context/ExecutionContext.ts` | Execution context propagation | Context lost in workflows |
| `memory/MemoryClient.ts` | Memory operations | Agent state management |
| `workflow/WorkflowReporter.ts` | Workflow DAG status reporting | Workflow tracking broken |

---

## P1 — HIGH PRIORITY (Core Business Logic)

### Control Plane Go

| File | What It Does |
|------|-------------|
| `handlers/ui/dashboard.go` | Dashboard summary data |
| `handlers/ui/executions.go` (partial) | Execution listing — tests exist but thin |
| `handlers/ui/nodes.go` | Node management UI endpoints |
| `handlers/ui/reasoners.go` | Reasoner listing/detail |
| `handlers/ui/execution_logs.go` | Log streaming for UI |
| `handlers/ui/execution_timeline.go` | Timeline visualization data |
| `handlers/ui/node_logs.go` | Node log proxy/streaming |
| `handlers/ui/lifecycle.go` | Execution cancel/pause/resume UI |
| `handlers/ui/did.go` | DID/VC display in UI |
| `handlers/ui/identity.go` | Identity management |
| `handlers/connector/handlers.go` | Connector system handlers |
| `handlers/agentic/query.go` | Agentic query endpoint |
| `handlers/agentic/batch.go` | Batch execution endpoint |
| `handlers/agentic/discover.go` | Agent discovery endpoint |
| `handlers/agentic/status.go` | Agentic status endpoint |
| `services/ui_service.go` | UI business logic |
| `services/executions_ui_service.go` | Execution queries for UI |
| `services/did_web_service.go` | DID:web resolution |
| `services/tag_normalization.go` | Tag normalization logic |
| `mcp/manager.go` | MCP server lifecycle |
| `mcp/process.go` | MCP process management |
| `mcp/protocol_client.go` | MCP protocol communication |

### Python SDK

| File | What It Does |
|------|-------------|
| `mcp_manager.py` | MCP server lifecycle (subprocess management) |
| `agent_mcp.py` | MCP feature orchestration in agent |
| `mcp_stdio_bridge.py` | stdio↔HTTP bridge for MCP servers |
| `node_logs.py` | `ProcessLogRing`, `install_stdio_tee` — observability |

### Go SDK

| File | What It Does |
|------|-------------|
| `types/types.go` | Core serialization for control plane communication |
| `types/discovery.go` | Discovery message types |
| `types/status.go` | `NormalizeStatus` and terminal/active categorization |
| `did/types.go` | DID identity types |

### TypeScript SDK

| File | What It Does |
|------|-------------|
| `ai/AIClient.ts` | LLM completion API |
| `ai/ToolCalling.ts` | Tool use parsing and execution |
| `ai/RateLimiter.ts` | Rate limiting for API calls |
| `mcp/MCPClient.ts` | MCP client |
| `mcp/MCPToolRegistrar.ts` | MCP tool registration |
| `did/DidManager.ts` | DID identity management |
| `harness/runner.ts` | Harness execution loop |
| `router/AgentRouter.ts` | Request routing |

---

## P2 — MEDIUM PRIORITY (Utilities, Helpers, Infrastructure)

### Control Plane Go

- `cli/*` — 15 CLI command files with no tests (except init, root, vc, verify)
- `config/config.go` — Configuration loading
- `infrastructure/storage/*` — Filesystem config storage
- `mcp/capability_discovery.go`, `skill_generator.go`, `template.go`, `storage.go`
- `packages/*` — Package installer, git operations, runner
- `server/middleware/connector_capability.go`, `permission.go`
- `server/config_db.go`, `knowledgebase/*`
- `storage/migrations.go`, `sql_helpers.go`, `gorm_helpers.go`, `tx_utils.go`, `utils.go`
- `storage/vector_store*.go` — Vector memory backends
- `templates/templates.go` — Code generation templates
- `utils/*` — ID generator, path helpers
- `logger/helpers.go` — Logging utilities

### Python SDK

- `harness/_cli.py` — CLI helpers for harness
- `harness/providers/_factory.py` — Provider selection
- `logger.py` — Logging utilities

### Go SDK

- `ai/multimodal.go` — MIME detection
- `agent/process_logs.go` — Log streaming
- `agent/verification.go` — Local verification
- `harness/claudecode.go`, `codex.go`, `gemini.go`, `opencode.go` — Provider implementations
- `harness/factory.go`, `cli.go`, `result.go`, `provider.go` — Harness infrastructure

### TypeScript SDK

- `utils/*.ts` — HTTP agents, pattern matching, schema helpers
- `status/ExecutionStatus.ts` — Status tracking
- `types/*.ts` — Type definitions
- `harness/providers/*.ts` — Provider implementations

---

## Implementation Plan — Prioritized Work Packages

### WP1: Storage Layer Tests (P0, highest risk)
**Scope:** `control-plane/internal/storage/`
**Effort:** Large — 12+ test files needed
**Strategy:** Test against SQLite (local mode) for speed. Cover all CRUD operations.
**Files to test:**
- `storage.go` — Initialization, backend selection
- `local.go` — SQLite CRUD for nodes, reasoners, executions, workflows
- `execution_state_validation.go` — State machine transitions
- `events.go` — Event persistence
- `models.go` — GORM model validation
- `vector_store_sqlite.go` — Vector memory
- `sql_helpers.go`, `tx_utils.go` — Transaction helpers

### WP2: Memory & Event Handler Tests (P0)
**Scope:** `control-plane/internal/handlers/memory*.go`, `events/`
**Effort:** Medium — 5 test files
**Files:**
- `handlers/memory.go` — Memory CRUD endpoint tests
- `handlers/memory_access_control.go` — Permission checks
- `handlers/memory_events.go` — SSE event delivery
- `events/execution_events.go` — Event emission
- `events/node_events.go` — Node event lifecycle

### WP3: Core REST Handler Tests (P0)
**Scope:** `control-plane/internal/handlers/`
**Effort:** Medium — 4 test files
**Files:**
- `handlers/nodes_rest.go` — Node registration/heartbeat
- `handlers/reasoners.go` — Reasoner CRUD
- `handlers/config_storage.go` — Config persistence

### WP4: Python SDK Security & MCP Tests (P0+P1)
**Scope:** `sdk/python/agentfield/`
**Effort:** Medium — 6 test files
**Files:**
- `verification.py` → `test_verification.py`
- `agent.py` (serverless + pause) → `test_agent_serverless.py`
- `mcp_manager.py` → `test_mcp_manager.py`
- `agent_mcp.py` → `test_agent_mcp.py`
- `mcp_stdio_bridge.py` → `test_mcp_stdio_bridge.py`
- `node_logs.py` → `test_node_logs.py`

### WP5: Go SDK Type & Harness Tests (P0+P1)
**Scope:** `sdk/go/`
**Effort:** Small-Medium — 5 test files
**Files:**
- `agent/harness.go` → `agent/harness_test.go`
- `types/types.go` → `types/types_test.go`
- `types/status.go` → `types/status_test.go`
- `types/discovery.go` → `types/discovery_test.go`
- `did/types.go` → `did/types_test.go`

### WP6: TypeScript SDK Core Tests (P0+P1)
**Scope:** `sdk/typescript/`
**Effort:** Medium — 8 test files
**Files:**
- `client/AgentFieldClient.ts` → `agentfield_client.test.ts`
- `agent/Agent.ts` (expand existing) → `agent_lifecycle.test.ts`
- `context/ExecutionContext.ts` → `execution_context.test.ts`
- `memory/MemoryClient.ts` → `memory_client.test.ts`
- `workflow/WorkflowReporter.ts` → `workflow_reporter.test.ts`
- `ai/ToolCalling.ts` → `tool_calling_advanced.test.ts`
- `mcp/MCPClient.ts` → `mcp_client.test.ts`
- `router/AgentRouter.ts` → `agent_router.test.ts`

### WP7: UI Handler Tests (P1)
**Scope:** `control-plane/internal/handlers/ui/`
**Effort:** Large — 12+ test files
**Files:** All UI handlers (dashboard, executions, nodes, reasoners, logs, timeline, lifecycle, DID, identity, MCP, node_logs, node_log_settings, packages, recent_activity, workflow_runs, env, authorization, config)

### WP8: Agentic & Connector Handler Tests (P1)
**Scope:** `control-plane/internal/handlers/agentic/`, `connector/`
**Effort:** Medium — 7 test files

### WP9: MCP Subsystem Tests (P1)
**Scope:** `control-plane/internal/mcp/`
**Effort:** Medium — 6 test files
**Files:** manager, process, protocol_client, capability_discovery, skill_generator, storage

### WP10: Service Layer Tests (P1)
**Scope:** `control-plane/internal/services/`
**Effort:** Medium — 4 test files
**Files:** ui_service, executions_ui_service, did_web_service, tag_normalization

---

## Execution Strategy

**Phase 1 (This PR):** WP1-WP6 (all P0 + critical P1) — use git worktrees for parallel work
**Phase 2 (Follow-up):** WP7-WP10 (remaining P1 + P2)
**Phase 3 (Later):** Web UI tests (requires test framework setup first)

**Parallelization plan:**
- Worktree A: WP1 (Storage tests) — Codex CLI
- Worktree B: WP2+WP3 (Handler tests) — Codex CLI
- Worktree C: WP4 (Python SDK) — Gemini CLI
- Worktree D: WP5 (Go SDK) — Gemini CLI
- Worktree E: WP6 (TS SDK) — Gemini CLI
- Orchestrator merges worktree branches sequentially

**Test patterns to follow:**
- Go: Table-driven tests, `httptest.NewServer` for HTTP mocks, testify assertions
- Python: pytest fixtures, `unittest.mock`, async test support
- TypeScript: vitest, mock fetch, spy patterns
