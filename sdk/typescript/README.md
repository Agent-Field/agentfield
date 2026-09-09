# AgentField TypeScript SDK

## Graceful shutdown

`serve()` installs SIGTERM and SIGINT handlers that notify the control plane and drain detached executions. Host processes that own signal handling can use `serve({ handleSignals: false })` and call the idempotent `shutdown()` method themselves. `AGENTFIELD_SHUTDOWN_TIMEOUT` accepts bare seconds (`30`) or durations (`30s`, `5m`) and defaults to 30 seconds. In Kubernetes, set `terminationGracePeriodSeconds` higher than this timeout.

The TypeScript SDK provides an idiomatic Node.js interface for building and running AgentField agents. It mirrors the Python SDK APIs, including AI, memory, discovery, and MCP tooling.

## Installing
```bash
npm install @agentfield/sdk
```

## Memory event subscriptions

Pass filters when starting a memory event client to apply them on the server before events are sent over the WebSocket:

```ts
import { MemoryEventClient } from '@agentfield/sdk';

const events = new MemoryEventClient('http://localhost:8080');
events.onEvent((event) => console.log(event.key));
events.start({
  patterns: ['user_*', 'session.*'],
  scope: 'session',
  scopeId: 'session-123'
});
```

`patterns` is a list of memory-key globs and is sent as a comma-separated query parameter. `scope` accepts `workflow`, `session`, `actor`, or `global`, while `scopeId` maps to the server's `scope_id` parameter. All filters are optional, and automatic reconnects reuse the original filters. Omitting them keeps the existing behavior of receiving all events.

Server-side filtering reduces WebSocket traffic and client-side processing for high-volume event streams.

## Rate limiting
AI calls are wrapped with a stateless rate limiter that matches the Python SDK: exponential backoff, container-scoped jitter, Retry-After support, and a circuit breaker.

Configure per-agent via `aiConfig`:
```ts
import { Agent } from '@agentfield/sdk';

const agent = new Agent({
  nodeId: 'demo',
  aiConfig: {
    model: 'gpt-4o',
    enableRateLimitRetry: true,           // default: true
    rateLimitMaxRetries: 20,              // max retry attempts
    rateLimitBaseDelay: 1.0,              // seconds
    rateLimitMaxDelay: 300.0,             // seconds cap
    rateLimitJitterFactor: 0.25,          // ±25% jitter
    rateLimitCircuitBreakerThreshold: 10, // consecutive failures before opening
    rateLimitCircuitBreakerTimeout: 300   // seconds before closing breaker
  }
});
```

To disable retries, set `enableRateLimitRetry: false`.

You can also use the limiter directly:
```ts
import { StatelessRateLimiter } from '@agentfield/sdk';

const limiter = new StatelessRateLimiter({ maxRetries: 3, baseDelay: 0.5 });
const result = await limiter.executeWithRetry(() => makeAiCall());
```

## AI Tool Calling

Let LLMs automatically discover and invoke agent capabilities:

```ts
import { Agent } from '@agentfield/sdk';
import type { ToolCallConfig } from '@agentfield/sdk';

const agent = new Agent({
  nodeId: 'orchestrator',
  agentFieldUrl: 'http://localhost:8080',
  aiConfig: { provider: 'openrouter', model: 'openai/gpt-4o-mini' },
});

agent.reasoner('ask', async (ctx) => {
  // Auto-discover all tools and let the LLM use them
  const { text, trace } = await ctx.aiWithTools(ctx.input.question, {
    tools: 'discover',
    system: 'You are a helpful assistant.',
  });

  console.log(`Tool calls: ${trace.totalToolCalls}, Turns: ${trace.totalTurns}`);
  return { answer: text };
});

// Filter by tags, set guardrails
agent.reasoner('weather', async (ctx) => {
  const { text } = await ctx.aiWithTools(ctx.input.cities, {
    tools: { tags: ['weather'] } satisfies ToolCallConfig,
    maxTurns: 3,
    maxToolCalls: 5,
  });
  return { report: text };
});
```

**Key features:**
- `tools: 'discover'` — Auto-discover all capabilities from the control plane
- `ToolCallConfig` — Filter by tags, agent IDs; lazy/eager schema hydration
- **Guardrails** — `maxTurns` and `maxToolCalls` prevent runaway loops
- **Observability** — `trace` tracks every tool call with latency

See `examples/ts_agent_nodes/tool_calling/` for a complete orchestrator + worker example.

## Execution Notes

Log execution progress with `ctx.note(message: string, tags?: string[])` for fire-and-forget debugging in the AgentField UI.

```ts
agent.reasoner('process', async (ctx) => {
  ctx.note('Starting processing', ['debug']);
  const result = await processData(ctx.input);
  ctx.note(`Completed: ${result.length} items`, ['info']);
  return result;
});
```

**Use `note()` for AgentField UI tracking, `console.log()` for local debugging.**

## Logging

- `AGENTFIELD_LOG_STDOUT` controls the on-by-default structured JSON mirror; `0`, `false`, `no`, or `off` disables it without disabling execution-scoped control-plane dispatch.
- `AGENTFIELD_LOGS_ENABLED` controls stdout/stderr capture and the node logs endpoint, not control-plane dispatch.
- `AGENTFIELD_LOG_MAX_LINE_BYTES` defaults to 16384 bytes. Values below 256 use that default; parsing accepts an integer prefix, so `512abc` yields 512.

See the [environment-variable reference](../../docs/ENVIRONMENT_VARIABLES.md) for details.

## Human-in-the-Loop Approvals

Use the `ApprovalClient` to pause agent execution for human review:

```ts
import { Agent, ApprovalClient } from '@agentfield/sdk';

const agent = new Agent({ nodeId: 'reviewer', agentFieldUrl: 'http://localhost:8080' });
const approvalClient = new ApprovalClient({
  baseURL: 'http://localhost:8080',
  nodeId: 'reviewer',
});

agent.reasoner<{ task: string }, { status: string }>('deploy', async (ctx) => {
  const plan = await ctx.ai(`Create deployment plan for: ${ctx.input.task}`);
  const approvalRequestId = `req-${ctx.executionId}`;

  // Create the human-facing approval request in your approval service first,
  // then pass its ID/URL to AgentField so the execution transitions to "waiting".
  await approvalClient.requestApproval(ctx.executionId, {
    approvalRequestId,
    approvalRequestUrl: `https://approvals.example.com/review/${approvalRequestId}`,
    expiresInHours: 24,
  });

  // Wait for human decision (polls with exponential backoff)
  const result = await approvalClient.waitForApproval(ctx.executionId, {
    pollIntervalMs: 5_000,
    timeoutMs: 3_600_000,
  });

  return { status: result.status };
});
```

**Methods:** `requestApproval()`, `getApprovalStatus()`, `waitForApproval()`

See `examples/ts-node-examples/waiting-state/` for a complete working example.
