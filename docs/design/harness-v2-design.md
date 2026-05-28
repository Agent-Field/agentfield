# `.harness()` — First-Class Coding Agent Integration for AgentField

> **Status**: Design Proposal  
> **Author**: Architecture brainstorm  
> **Scope**: Python SDK + TypeScript SDK (Go SDK tracked separately)  
> **Date**: 2026-03-02

---

## 1. Overview

Add `.harness()` as a first-class method on the Agent class — matching the DX of `.ai()` — enabling developers to dispatch multi-turn coding tasks to external coding agents (Claude Code, Codex, Gemini CLI, OpenCode).

**Key principle**: `.ai()` is for single-turn LLM calls. `.harness()` is for multi-turn agentic coding tasks that browse files, edit code, run tests, and iterate.

---

## 2. Developer Experience

### 2.1 Agent Construction

```python
from agentfield import Agent, AIConfig, HarnessConfig

app = Agent(
    node_id="my-agent",
    ai_config=AIConfig(model="openai/gpt-4o"),
    harness_config=HarnessConfig(
        provider="claude-code",   # Required — no implicit default
        model="sonnet",
    ),
)
```

```typescript
import { Agent } from '@agentfield/sdk';

const agent = new Agent({
    nodeId: 'my-agent',
    harnessConfig: {
        provider: 'claude-code',  // Required
        model: 'sonnet',
    },
});
```

### 2.2 Simple Call

```python
result = await app.harness("Fix the auth bug in src/auth.py")
print(result.text)  # "Fixed the auth bug by..."
```

```typescript
const result = await agent.harness('Fix the auth bug in src/auth.py');
console.log(result.text);
```

### 2.3 Schema-Constrained Output

```python
from pydantic import BaseModel

class BugFix(BaseModel):
    files_changed: list[str]
    summary: str
    tests_added: bool

fix = await app.harness(
    "Fix the auth bug and add tests",
    schema=BugFix,
    cwd="/my/project",
)
print(fix.files_changed)  # ["src/auth.py", "tests/test_auth.py"]
print(fix.tests_added)    # True
```

```typescript
import { z } from 'zod';

const BugFix = z.object({
    files_changed: z.array(z.string()),
    summary: z.string(),
    tests_added: z.boolean(),
});

const fix = await agent.harness('Fix the auth bug and add tests', {
    schema: BugFix,
    cwd: '/my/project',
});
```

### 2.4 Per-Call Overrides

```python
# Override provider, model, tools per-call (like .ai() overrides)
fix = await app.harness(
    "Refactor to async/await",
    provider="codex",      # Override default provider
    model="o3",            # Override default model
    max_turns=100,
    tools=["Read", "Write", "Edit", "Bash"],
    max_budget_usd=5.0,
)
```

### 2.5 Without Constructor Config

```python
# No harness_config on Agent — provide everything per-call
app = Agent(node_id="minimal-agent")

result = await app.harness(
    "Fix the bug",
    provider="gemini",     # Required when no harness_config
    model="flash",
    cwd="/my/project",
)
```

### 2.6 Inside a Reasoner (production pattern)

```python
@app.reasoner()
async def fix_issue(issue: dict) -> dict:
    result = await app.harness(
        f"Fix: {issue['title']}\n\n{issue['description']}",
        schema=IssueFixResult,
        cwd=issue['repo_path'],
        max_turns=150,
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    )
    return result.model_dump()
```

---

## 3. Architecture

### 3.1 System Diagram

```
Agent
├── .ai()      → AIConfig      → LiteLLM     → LLM APIs (100+ providers)
└── .harness() → HarnessConfig → HarnessRunner → Provider → {Claude Code, Codex, Gemini, OpenCode}
```

### 3.2 Component Stack

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent.harness()                            │
│          HarnessConfig (constructor param, optional)          │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                   AgentHarness (_handler.py)                  │
│  - Config resolution: harness_config defaults + per-call      │
│  - Provider validation                                        │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                   HarnessRunner (_runner.py)                  │
│  - Retry with exponential backoff + jitter                   │
│  - Schema output orchestration (file-write + 4-layer recovery)│
│  - Temp file lifecycle management                            │
│  - JSONL event logging                                       │
│  - Cost tracking                                             │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│              Provider Interface (_base.py)                    │
│      execute(prompt, options) → RawResult                    │
└──┬──────────┬──────────┬───────────┬─────────────────────────┘
   │          │          │           │
┌──▼────┐ ┌──▼────┐ ┌───▼───┐ ┌────▼─────┐
│Claude │ │ Codex │ │Gemini │ │OpenCode  │
│ Code  │ │       │ │  CLI  │ │          │
└───────┘ └───────┘ └───────┘ └──────────┘
```

### 3.3 Internal Wiring (mirrors ai_handler pattern)

```python
class Agent(FastAPI):
    def __init__(self, ..., harness_config: Optional[HarnessConfig] = None):
        ...
        self.harness_config = harness_config  # None is valid (per-call config)
        self._harness_handler: Optional[AgentHarness] = None  # Lazy

    @property
    def harness_handler(self) -> AgentHarness:
        """Lazy-loaded harness handler — only initialized when harness features are used."""
        if self._harness_handler is None:
            from agentfield.harness import AgentHarness
            self._harness_handler = AgentHarness(self)
        return self._harness_handler

    async def harness(self, prompt: str, *, schema=None, provider=None, ...):
        """Run a prompt through a multi-turn coding agent."""
        return await self.harness_handler.run(prompt, schema=schema, provider=provider, ...)
```

---

## 4. Provider Integration Strategy

### 4.1 Integration Matrix

| Provider | Python | TypeScript | Go (future) | Schema Support |
|---|---|---|---|---|
| **claude-code** | `claude_agent_sdk` (native Python SDK) | `@anthropic-ai/claude-agent-sdk` (native TS SDK) | CLI subprocess | File-write (universal) |
| **codex** | CLI subprocess `codex exec --json` | `@openai/codex-sdk` (native TS SDK) | CLI subprocess | File-write (universal) |
| **gemini** | CLI subprocess `gemini --output-format stream-json` | CLI subprocess | CLI subprocess | File-write (universal) |
| **opencode** | CLI subprocess | CLI subprocess | CLI subprocess | File-write (universal) |

### 4.2 Why SDK-First Where Available

- **In-process execution** — no binary dependency, no subprocess overhead
- **Same engine as CLI** — `claude_agent_sdk` and `@openai/codex-sdk` run the identical agent loop
- **Richer API** — hooks, sessions, structured output natively supported
- **Better error handling** — typed exceptions vs. parsing stderr

### 4.3 Why CLI Subprocess for Others

- No native SDK wrapping the CLI agent loop exists for Gemini/OpenCode
- CLI subprocess is lowest-maintenance integration (binary does all the work)
- All CLIs output JSONL event streams for consistent parsing
- Go SDK will use CLI subprocess for ALL providers (no Go SDKs exist)

### 4.4 Provider Interface

```python
# Python
class HarnessProvider(Protocol):
    """Each provider only implements execute(). Retry, schema, logging handled by Runner."""
    async def execute(self, prompt: str, options: HarnessOptions) -> RawResult: ...
```

```typescript
// TypeScript
interface HarnessProvider {
    execute(prompt: string, options: HarnessOptions): Promise<RawResult>;
}
```

```go
// Go (future)
type Provider interface {
    Execute(ctx context.Context, prompt string, opts HarnessOptions) (*RawResult, error)
}
```

---

## 5. Schema Handling

### 5.1 Universal File-Write Strategy

**Core principle**: Always instruct the coding agent to write JSON output to a file using its Write tool. Never rely on native `--json-schema` / `--output-schema` flags as the primary mechanism.

**Why file-write over native flags:**
- Native flags constrain the model's *text response* — a wrong abstraction for multi-turn coding agents whose core competency is writing files
- Large schemas can produce JSON that exceeds response token limits or gets truncated
- File-write works identically across ALL 4 providers (one code path, half the tests)
- Writing a JSON file is trivially easy for agents that refactor entire codebases

```
Developer writes:  await app.harness("...", schema=MyModel)

Runner (ALL providers):
  1. Convert schema → JSON Schema string
  2. Append OUTPUT REQUIREMENTS suffix to user prompt
  3. Execute agent (provider-agnostic)
  4. Read output file → parse → validate → return typed instance
```

### 5.2 Output File Convention

```
{cwd}/.agentfield_output.json
```

- **cwd, not `/tmp`**: Sandboxed agents (Docker, firejail) are *guaranteed* write access to cwd — they're already editing files there. `/tmp` may not resolve inside sandboxes.
- **Dotfile**: Hidden by default, won't clutter `ls` output. Harness deletes after reading.
- **Deterministic name**: No UUID — harness always knows where to look. Concurrent runs should use different cwds.

### 5.3 Prompt Suffix

Appended to the **end** of the user prompt (recency bias — models weight the end of input most heavily). Not in system prompt — CLI wrappers for Gemini/OpenCode may not expose system prompt control.

```
{user's actual task prompt}

---
OUTPUT REQUIREMENTS:
Write your final answer as valid JSON to the file: {cwd}/.agentfield_output.json
The JSON must conform to this schema:
{schema_json}
Do not include any text outside the JSON in that file. Do not wrap in markdown fences.
```

**Large schemas (>4K tokens)**: Write the schema to a file (`{cwd}/.agentfield_schema.json`) and instruct: `"Read the schema at {schema_path} and write conforming JSON to {output_path}."`

### 5.4 Four-Layer Recovery Strategy

On schema validation failure, recover cheapest-first:

```
Layer 1: Parse file → validate against schema           (happy path, no cost)
    ↓ fail
Layer 2: Cosmetic repair → re-validate                  (zero cost)
    - Strip markdown fences (```json ... ```)
    - Remove trailing commas
    - Fix truncated closing brackets/braces
    ↓ fail
Layer 3: Follow-up prompt in same session                (~1% cost of full retry)
    - "The JSON at {path} failed validation: {specific_error}. Rewrite the corrected file."
    - Agent has full context, trivial fix
    - Cap at 2 follow-up attempts
    ↓ fail
Layer 4: Full retry                                      (expensive, last resort)
    - Only if session is unrecoverable (agent crashed, file never written)
    - Limit to 1 full retry
    - This should be extremely rare
```

**No separate schema budget**: File-write is a trivially cheap operation (~few hundred tokens). Use the overall `max_budget_usd` — don't split it.

### 5.5 Schema Validation

- Python: `schema.model_validate(json.loads(file_content))`
- TypeScript: `schema.parse(JSON.parse(fileContent))` (Zod)
- Go: `json.Unmarshal(fileBytes, &schemaInstance)` (struct tags)

### 5.6 Watch-outs

- **Wait for completion**: Read the output file *after* confirming the agent session completed (exit code 0 or equivalent). Don't race.
- **Markdown fences**: The #1 cosmetic failure mode. Layer 2 must handle `` ```json ... ``` `` stripping before any retry logic fires.
- **Schema too large for prompt context**: If schema exceeds ~4K tokens, use the file-based schema approach (write schema to file, instruct agent to read it).
- **Cleanup**: Always delete `.agentfield_output.json` and `.agentfield_schema.json` after reading, even on failure.

---

## 6. Configuration

### 6.1 HarnessConfig (Python)

```python
class HarnessConfig(BaseModel):
    """Configuration for coding agent harness calls.
    
    Provider is required — there is no implicit default.
    All other fields have sensible defaults that can be overridden per-call.
    """
    # Provider selection (required)
    provider: str               # "claude-code" | "codex" | "gemini" | "opencode"
    model: str = "sonnet"
    
    # Execution limits
    max_turns: int = 30
    max_budget_usd: Optional[float] = None
    
    # Retry behavior
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    
    # Tools & permissions
    tools: List[str] = Field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep"
    ])
    permission_mode: Optional[str] = None  # "plan" | "auto" | None
    system_prompt: Optional[str] = None
    
    # Environment
    env: Dict[str, str] = Field(default_factory=dict)
    
    # Binary paths (for CLI-based providers)
    codex_bin: str = "codex"
    gemini_bin: str = "gemini"
    opencode_bin: str = "opencode"
```

### 6.2 HarnessConfig (TypeScript)

```typescript
interface HarnessConfig {
    /** Required — no default. "claude-code" | "codex" | "gemini" | "opencode" */
    provider: string;
    /** Default model identifier. Default: "sonnet" */
    model?: string;
    /** Maximum agent iterations. Default: 30 */
    maxTurns?: number;
    /** Cost cap in USD. */
    maxBudgetUsd?: number;
    /** Maximum retry attempts. Default: 3 */
    maxRetries?: number;
    /** Initial retry delay in seconds. Default: 1.0 */
    initialDelay?: number;
    /** Maximum retry delay in seconds. Default: 30.0 */
    maxDelay?: number;
    /** Retry backoff multiplier. Default: 2.0 */
    backoffFactor?: number;
    /** Default allowed tools. */
    tools?: string[];
    /** Permission mode. */
    permissionMode?: string;
    /** Default system prompt. */
    systemPrompt?: string;
    /** Default environment variables. */
    env?: Record<string, string>;
    /** Path to codex binary. */
    codexBin?: string;
    /** Path to gemini binary. */
    geminiBin?: string;
    /** Path to opencode binary. */
    opencodeBin?: string;
}
```

### 6.3 Config Resolution (hierarchical, matches .ai pattern)

```
1. HarnessConfig defaults (set at agent construction)
2. Per-call overrides (passed to .harness() method)
   → Per-call values win over HarnessConfig defaults
   → If no HarnessConfig AND no per-call provider → raise error
```

---

## 7. Result Types

### 7.1 HarnessResult (Python)

```python
@dataclass
class HarnessResult:
    """Final result from a harness invocation."""
    
    # Core output
    result: Optional[str]           # Raw text result
    parsed: Any                     # Validated schema instance (T | None)
    is_error: bool
    
    # Metrics
    cost_usd: Optional[float]      # Total execution cost
    num_turns: int                  # Number of agent turns
    duration_ms: int                # Wall clock time in milliseconds
    session_id: str                 # For potential session resume
    
    # Message history
    messages: List[Message]         # Full conversation history
    
    # Convenience properties
    @property
    def text(self) -> str:
        """Last text content from conversation, or result."""
        if self.result:
            return self.result
        for msg in reversed(self.messages):
            for block in reversed(msg.content):
                if isinstance(block, TextContent):
                    return block.text
        return ""
```

### 7.2 RawResult (internal, from providers)

```python
@dataclass
class RawResult:
    """Raw result from a single provider execution."""
    result: Optional[str]
    messages: List[Message]
    metrics: Metrics
    is_error: bool
```

### 7.3 Metrics

```python
@dataclass
class Metrics:
    """Execution metrics from a harness run."""
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    total_cost_usd: Optional[float]
    usage: Optional[Dict[str, Any]]
    session_id: str
```

---

## 8. File Structure

### 8.1 Python SDK

```
sdk/python/agentfield/harness/
├── __init__.py              # Public API: AgentHarness, HarnessConfig, HarnessResult
├── _handler.py              # AgentHarness class (attached to Agent, like AgentAI)
├── _runner.py               # HarnessRunner: retry, schema orchestration, logging
├── _result.py               # HarnessResult, RawResult, Metrics, Message types
├── _schema.py               # Schema handling: native flags + file-write fallback
├── _cli.py                  # Shared async subprocess utilities for CLI providers
├── providers/
│   ├── __init__.py
│   ├── _base.py             # HarnessProvider protocol
│   ├── _factory.py          # build_provider() router
│   ├── claude.py            # Claude Code provider (SDK-based: claude_agent_sdk)
│   ├── codex.py             # Codex provider (CLI-based: codex exec)
│   ├── gemini.py            # Gemini provider (CLI-based: gemini)
│   └── opencode.py          # OpenCode provider (CLI-based: opencode)
```

### 8.2 TypeScript SDK

```
sdk/typescript/src/harness/
├── index.ts                 # Public API exports
├── handler.ts               # Agent integration (like AIClient)
├── runner.ts                # HarnessRunner
├── types.ts                 # All type definitions
├── schema.ts                # Schema handling
├── cli.ts                   # Shared subprocess utilities
├── providers/
│   ├── index.ts
│   ├── base.ts              # HarnessProvider interface
│   ├── factory.ts           # buildProvider() router
│   ├── claude.ts            # Claude Code provider (SDK-based)
│   ├── codex.ts             # Codex provider (SDK-based: @openai/codex-sdk)
│   ├── gemini.ts            # Gemini provider (CLI-based)
│   └── opencode.ts          # OpenCode provider (CLI-based)
```

### 8.3 Go SDK (future)

```
sdk/go/harness/
├── config.go                # Config struct
├── runner.go                # Runner (retry, schema, logging)
├── result.go                # Result types
├── schema.go                # Schema handling
├── provider.go              # Provider interface
├── factory.go               # BuildProvider() router
├── cli.go                   # Shared subprocess utilities
├── claude.go                # Claude Code provider (CLI)
├── codex.go                 # Codex provider (CLI)
├── gemini.go                # Gemini provider (CLI)
└── opencode.go              # OpenCode provider (CLI)
```

---

## 9. Retry & Error Handling

### 9.1 Retry Strategy

```python
# Exponential backoff with jitter
for attempt in range(max_retries + 1):
    try:
        result = await provider.execute(prompt, options)
        if not result.is_error:
            return result
        if is_transient(result):
            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
            delay += random.uniform(-delay * 0.25, delay * 0.25)  # jitter
            await asyncio.sleep(delay)
            continue
        return result  # Non-transient error, return immediately
    except Exception as e:
        if is_transient(str(e)) and attempt < max_retries:
            await asyncio.sleep(delay)
            continue
        raise
```

### 9.2 Transient Error Patterns

```python
TRANSIENT_PATTERNS = {
    "rate limit", "rate_limit", "overloaded", "timeout", "timed out",
    "connection reset", "connection refused", "temporarily unavailable",
    "service unavailable", "503", "502", "504", "internal server error", "500",
}
```

### 9.3 Error Categories

```python
class ErrorKind(str, Enum):
    AUTH = "authentication_failed"
    BILLING = "billing_error"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    SERVER = "server_error"
    UNKNOWN = "unknown"
```

---

## 10. Provider Implementation Details

### 10.1 Claude Code Provider (Python — SDK-based)

```python
class ClaudeCodeProvider:
    async def execute(self, prompt: str, options: HarnessOptions) -> RawResult:
        from claude_agent_sdk import query, ClaudeAgentOptions
        
        opts = ClaudeAgentOptions(
            model=options.model,
            cwd=options.cwd,
            max_turns=options.max_turns,
            allowed_tools=options.allowed_tools,
            system_prompt=options.system_prompt,
            max_budget_usd=options.max_budget_usd,
            permission_mode=options.permission_mode,
            env=options.env,
        )
        
        messages = []
        result_text = None
        metrics = {}
        
        async for msg in query(prompt=prompt, options=opts):
            # Collect messages and extract result
            ...
        
        return RawResult(result=result_text, messages=messages, metrics=metrics, is_error=False)
```

### 10.2 Codex Provider (Python — CLI-based)

```python
class CodexProvider:
    def __init__(self, bin_path: str = "codex"):
        self.bin = bin_path
    
    async def execute(self, prompt: str, options: HarnessOptions) -> RawResult:
        cmd = [self.bin, "exec", "--json"]
        
        if options.cwd:
            cmd.extend(["-C", options.cwd])
        if options.permission_mode == "auto":
            cmd.append("--full-auto")
        
        cmd.append(prompt)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **options.env},
        )
        stdout, stderr = await proc.communicate()
        
        # Parse JSONL events from stdout
        return self._parse_jsonl_output(stdout.decode())
```

### 10.3 Gemini Provider (Python — CLI-based)

```python
class GeminiProvider:
    def __init__(self, bin_path: str = "gemini"):
        self.bin = bin_path
    
    async def execute(self, prompt: str, options: HarnessOptions) -> RawResult:
        cmd = [self.bin, "--output-format", "json"]
        
        if options.model:
            cmd.extend(["--model", options.model])
        if options.permission_mode == "auto":
            cmd.extend(["--approval-mode", "yolo"])
        
        cmd.append(prompt)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **options.env},
            cwd=options.cwd,
        )
        stdout, stderr = await proc.communicate()
        
        return self._parse_output(stdout.decode(), proc.returncode)
```

---

## 11. Implementation Phases

### Phase 1: Core + Claude Code + Codex (Python + TypeScript)

- [ ] `HarnessConfig` type definitions
- [ ] `HarnessResult`, `RawResult`, `Metrics` types
- [ ] `HarnessProvider` protocol/interface
- [ ] `AgentHarness` handler class
- [ ] `HarnessRunner` with retry + schema orchestration
- [ ] Schema handling (universal file-write + 4-layer recovery)
- [ ] Claude Code provider (Python: SDK, TypeScript: SDK)
- [ ] Codex provider (Python: CLI, TypeScript: SDK)
- [ ] `build_provider()` factory
- [ ] Wire into Agent class (`harness_config` param + `.harness()` method)
- [ ] Export from `__init__.py` / `index.ts`
- [ ] Unit tests for runner, schema, providers
- [ ] Integration tests with mock providers
- [ ] Documentation in docstrings

### Phase 2: Gemini + OpenCode Providers

- [ ] Gemini CLI provider (Python + TypeScript)
- [ ] OpenCode CLI provider (Python + TypeScript)
- [ ] Shared CLI subprocess utilities (`_cli.py`)
- [ ] Add to factory
- [ ] Tests for new providers

### Phase 3: Go SDK Port

- [ ] Port all types to Go structs
- [ ] Runner implementation
- [ ] All 4 providers (CLI-based)
- [ ] Wire into Go Agent struct
- [ ] Tests

### Stretch: Aider Provider

- [ ] Research Aider headless mode and output formats
- [ ] Implement Aider provider (CLI-based, all SDKs)
- [ ] Tests

---

## 12. Testing Strategy

### Unit Tests
- Runner: retry behavior, config resolution, schema orchestration
- Schema: native flag generation, file-write suffix, JSON parsing, validation
- Factory: provider routing for each provider name
- Each provider: command construction, output parsing

### Integration Tests (mock subprocess)
- Full flow: prompt → provider → parse → schema validate → return
- Retry on transient errors
- Schema failure → `is_error=True`
- Cost/metrics extraction

### E2E Tests (optional, CI-gated)
- Actual Claude Code call with schema
- Actual Codex call with schema
- Require API keys (skip in CI by default)

---

## 13. Dependencies

### Python SDK
- `claude-agent-sdk` (optional — lazy import, only when claude-code provider used)
- No new required dependencies (subprocess is stdlib)

### TypeScript SDK
- `@anthropic-ai/claude-agent-sdk` (optional peer dependency)
- `@openai/codex-sdk` (optional peer dependency)
- `zod-to-json-schema` (already a dependency)

### Go SDK
- No new dependencies (os/exec is stdlib)

---

## 14. Open Questions

1. **Session resume**: Should `.harness()` support resuming a previous session (by `session_id`)? Claude Code and Codex both support this. Could be a follow-up feature.

2. **MCP integration**: Should harness providers be discoverable as MCP tools? Codex already has `codex mcp-server` mode. Could be a follow-up.

3. **Cost aggregation**: Should harness costs roll up into the Agent's workflow metrics automatically? Requires control plane integration.

4. **Streaming**: Currently final-result-only. If streaming is needed later, add `async def harness_stream()` method.
