# AgentField TypeScript SDK

The TypeScript SDK provides an idiomatic Node.js interface for building and running AgentField agents. It mirrors the Python SDK APIs, including AI, memory, discovery, and MCP tooling.

## Installing
```bash
npm install @agentfield/sdk
```

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

  // Request approval — transitions execution to "waiting"
  await approvalClient.requestApproval(ctx.executionId, {
    projectId: 'my-project',
    title: `Deploy: ${ctx.input.task}`,
    description: String(plan),
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
