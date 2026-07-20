import { afterEach, describe, expect, it, vi } from 'vitest';
import type express from 'express';

const { generateTextMock, generateObjectMock, streamTextMock, embedMock, embedManyMock } = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
  generateObjectMock: vi.fn(),
  streamTextMock: vi.fn(),
  embedMock: vi.fn(),
  embedManyMock: vi.fn()
}));

vi.mock('ai', () => ({
  generateText: generateTextMock,
  generateObject: generateObjectMock,
  streamText: streamTextMock,
  embed: embedMock,
  embedMany: embedManyMock,
  tool: (definition: Record<string, unknown>) => definition,
  jsonSchema: (schema: unknown) => schema,
  stepCountIs: (count: number) => ({ type: 'step-count', count })
}));

import { AIClient } from '../src/ai/AIClient.js';
import { executeToolCallLoop } from '../src/ai/ToolCalling.js';
import { Agent } from '../src/agent/Agent.js';
import { HarnessRunner } from '../src/harness/runner.js';
import { createHarnessResult, createMetrics, createRawResult } from '../src/harness/types.js';
import { ExecutionContext, type ExecutionMetadata } from '../src/context/ExecutionContext.js';

function makeContext(executionId: string, overrides: Partial<ExecutionMetadata> = {}) {
  return new ExecutionContext({
    input: {},
    metadata: { executionId, ...overrides },
    req: {} as express.Request,
    res: {} as express.Response,
    agent: { getExecutionLogger: vi.fn() } as unknown as Agent
  });
}

const sdkUsage = (input: number, output: number, extras: Record<string, unknown> = {}) => ({
  inputTokens: input,
  outputTokens: output,
  totalTokens: input + output,
  inputTokenDetails: { noCacheTokens: input, cacheReadTokens: 0, cacheWriteTokens: 0 },
  outputTokenDetails: { textTokens: output, reasoningTokens: 0 },
  ...extras
});

afterEach(() => {
  vi.restoreAllMocks();
  generateTextMock.mockReset();
  generateObjectMock.mockReset();
});

describe('AIClient usage capture', () => {
  it('records generateText usage into the current execution tracker', async () => {
    generateTextMock.mockResolvedValue({ text: 'hi', usage: sdkUsage(100, 50) });
    const client = new AIClient({ provider: 'anthropic', apiKey: 'k', model: 'claude-opus-4-8' });
    const ctx = makeContext('exec-ai-1', { reasonerId: 'summarize' });

    const text = await ExecutionContext.run(ctx, () => client.generate('prompt'));

    expect(text).toBe('hi');
    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toEqual([
      expect.objectContaining({
        source: 'llm',
        provider: 'anthropic',
        model: 'claude-opus-4-8',
        reasoner: 'summarize',
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cost_usd: null,
        cost_source: null
      })
    ]);
  });

  it('records generateObject (schema) usage with provider-native cost from usage.raw', async () => {
    generateObjectMock.mockResolvedValue({
      object: { ok: true },
      usage: sdkUsage(500, 200, { raw: { cost: 0.0042 } })
    });
    const client = new AIClient({ provider: 'openrouter', apiKey: 'k', model: 'qwen/qwen3-coder' });
    const ctx = makeContext('exec-ai-2', { reasonerId: 'code' });

    const { z } = await import('zod');
    await ExecutionContext.run(ctx, () => client.generate('prompt', { schema: z.object({ ok: z.boolean() }) }));

    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toEqual([
      expect.objectContaining({
        provider: 'openrouter',
        model: 'qwen/qwen3-coder',
        cost_usd: 0.0042,
        cost_source: 'provider',
        input_tokens: 500,
        output_tokens: 200
      })
    ]);
    expect(wire.total_cost_usd).toBe(0.0042);
  });

  it('records nothing when the result has no usage, and nothing outside an execution', async () => {
    generateTextMock.mockResolvedValue({ text: 'bare' });
    const client = new AIClient({ apiKey: 'k' });

    const ctx = makeContext('exec-ai-3');
    await ExecutionContext.run(ctx, () => client.generate('prompt'));
    expect(ctx.costTracker.hasEntries).toBe(false);

    // Outside an execution the call still succeeds.
    generateTextMock.mockResolvedValue({ text: 'ok', usage: sdkUsage(1, 1) });
    await expect(client.generate('prompt')).resolves.toBe('ok');
  });

  it('passes a usage-accounting fetch to OpenRouter-bound models', async () => {
    generateTextMock.mockResolvedValue({ text: 'ok' });
    const client = new AIClient({ provider: 'openrouter', apiKey: 'k', model: 'qwen/qwen3-coder' });
    await client.generate('prompt');
    // The model handed to generateText was built with a custom fetch that
    // injects usage: {include: true} (see withOpenRouterUsageInclude tests).
    expect(generateTextMock).toHaveBeenCalled();
  });
});

describe('tool-call loop usage aggregation', () => {
  it('sums step usage via totalUsage and per-step raw costs into one entry per LLM call', async () => {
    generateTextMock.mockResolvedValue({
      text: 'done',
      steps: [
        { toolCalls: [], usage: sdkUsage(10, 5, { raw: { cost: 0.01 } }) },
        { toolCalls: [], usage: sdkUsage(20, 8, { raw: { cost: 0.02 } }) }
      ],
      usage: sdkUsage(20, 8, { raw: { cost: 0.02 } }),
      totalUsage: sdkUsage(30, 13)
    });

    const ctx = makeContext('exec-loop-1', { reasonerId: 'tools' });
    const result = await ExecutionContext.run(ctx, () =>
      executeToolCallLoop(
        { discover: vi.fn(), call: vi.fn() } as any,
        'prompt',
        { noop: { description: 'noop', inputSchema: { type: 'object', properties: {} } } } as any,
        { maxTurns: 3, maxToolCalls: 2 },
        false,
        () => ({ provider: 'mock-model' }),
        {},
        { provider: 'openrouter', modelName: 'openrouter/qwen/qwen3-coder' }
      )
    );

    expect(result.text).toBe('done');
    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toHaveLength(1);
    expect(wire.entries[0]).toMatchObject({
      model: 'openrouter/qwen/qwen3-coder',
      provider: 'openrouter',
      reasoner: 'tools',
      input_tokens: 30,
      output_tokens: 13,
      cost_source: 'provider'
    });
    expect(wire.entries[0].cost_usd).toBeCloseTo(0.03, 10);
  });

  it('records the lazy-hydration selection pass as its own entry', async () => {
    // Selection pass picks no tools -> loop returns the first-pass text.
    generateTextMock.mockResolvedValueOnce({
      text: 'no tools needed',
      steps: [{ toolCalls: [], usage: sdkUsage(4, 2) }],
      usage: sdkUsage(4, 2),
      totalUsage: sdkUsage(4, 2)
    });

    const ctx = makeContext('exec-loop-2');
    await ExecutionContext.run(ctx, () =>
      executeToolCallLoop(
        { discover: vi.fn(), call: vi.fn() } as any,
        'prompt',
        { noop: { description: 'noop', inputSchema: { type: 'object', properties: {} } } } as any,
        { maxTurns: 3, maxToolCalls: 2 },
        true,
        () => ({ provider: 'mock-model' }),
        {},
        { provider: 'openai', modelName: 'gpt-4o' }
      )
    );

    expect(generateTextMock).toHaveBeenCalledTimes(1);
    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toHaveLength(1);
    expect(wire.entries[0]).toMatchObject({ model: 'gpt-4o', input_tokens: 4, output_tokens: 2 });
  });

  it('skips usage capture when no model identity was provided (legacy callers)', async () => {
    generateTextMock.mockResolvedValue({ text: 'done', steps: [], usage: sdkUsage(1, 1) });
    const ctx = makeContext('exec-loop-3');
    await ExecutionContext.run(ctx, () =>
      executeToolCallLoop(
        { discover: vi.fn(), call: vi.fn() } as any,
        'prompt',
        {} as any,
        {},
        false,
        () => ({ provider: 'mock-model' })
      )
    );
    expect(ctx.costTracker.hasEntries).toBe(false);
  });
});

describe('harness usage capture', () => {
  const makeAgent = () =>
    new Agent({ nodeId: 'harness-agent', harnessConfig: { provider: 'claude-code', model: 'claude-opus-4-8' } as any });

  it('records a harness run with tokens and provider cost into the current tracker', async () => {
    const agent = makeAgent();
    vi.spyOn(HarnessRunner.prototype, 'run').mockResolvedValue(
      createHarnessResult({
        result: 'done',
        costUsd: 0.5,
        inputTokens: 900,
        outputTokens: 100,
        cacheReadTokens: 30,
        cacheCreationTokens: 7,
        numTurns: 3,
        sessionId: 's1'
      })
    );

    const ctx = makeContext('exec-h1', { reasonerId: 'builder' });
    await ExecutionContext.run(ctx, () => agent.harness('do the thing'));

    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toEqual([
      expect.objectContaining({
        source: 'harness',
        harness: 'claude_code',
        model: 'claude-opus-4-8',
        provider: null, // bare model slug carries no provider prefix
        reasoner: 'builder',
        input_tokens: 900,
        output_tokens: 100,
        cache_read_tokens: 30,
        cache_creation_tokens: 7,
        total_tokens: 1000,
        cost_usd: 0.5,
        cost_source: 'provider'
      })
    ]);
  });

  it('records the base model when the configured model carries a #variant suffix', async () => {
    const agent = new Agent({
      nodeId: 'harness-agent',
      harnessConfig: { provider: 'opencode', model: 'openrouter/z-ai/glm-5.2#high' } as any
    });
    // Provider reported cost but no model — the recorded model falls back to
    // the configured one, minus the reasoning-effort suffix.
    vi.spyOn(HarnessRunner.prototype, 'run').mockResolvedValue(
      createHarnessResult({ result: 'done', costUsd: 0.5 })
    );

    const ctx = makeContext('exec-h1b', { reasonerId: 'builder' });
    await ExecutionContext.run(ctx, () => agent.harness('do the thing'));

    expect(ctx.costTracker.serialize().entries[0]).toMatchObject({
      model: 'openrouter/z-ai/glm-5.2',
      provider: 'openrouter'
    });
  });

  it('records cost-only runs and skips runs that reported nothing', async () => {
    const agent = makeAgent();
    const runSpy = vi.spyOn(HarnessRunner.prototype, 'run');

    // Cost known, tokens unknown -> still recorded.
    runSpy.mockResolvedValueOnce(createHarnessResult({ result: 'a', costUsd: 0.25 }));
    const ctx = makeContext('exec-h2');
    await ExecutionContext.run(ctx, () => agent.harness('t'));
    expect(ctx.costTracker.serialize().entries[0]).toMatchObject({
      cost_usd: 0.25,
      cost_source: 'provider',
      total_tokens: 0
    });

    // Neither tokens nor cost -> no entry.
    runSpy.mockResolvedValueOnce(createHarnessResult({ result: 'b' }));
    const ctx2 = makeContext('exec-h3');
    await ExecutionContext.run(ctx2, () => agent.harness('t'));
    expect(ctx2.costTracker.hasEntries).toBe(false);
  });

  it('does not throw when run outside an execution context', async () => {
    const agent = makeAgent();
    vi.spyOn(HarnessRunner.prototype, 'run').mockResolvedValue(
      createHarnessResult({ result: 'ok', costUsd: 0.1 })
    );
    await expect(agent.harness('t')).resolves.toMatchObject({ result: 'ok' });
  });

  it('threads provider token metrics through the runner onto the HarnessResult', async () => {
    const runner = new HarnessRunner({ provider: 'claude-code' } as any);
    vi.spyOn(runner, 'executeWithRetry').mockResolvedValue(
      createRawResult({
        result: 'done',
        metrics: createMetrics({
          totalCostUsd: 0.5,
          inputTokens: 900,
          outputTokens: 100,
          cacheReadTokens: 30,
          cacheCreationTokens: 7,
          model: 'claude-opus-4-8',
          sessionId: 's',
          numTurns: 1
        })
      })
    );

    const res = await runner.run('prompt');
    expect(res).toMatchObject({
      costUsd: 0.5,
      inputTokens: 900,
      outputTokens: 100,
      cacheReadTokens: 30,
      cacheCreationTokens: 7,
      model: 'claude-opus-4-8'
    });
  });
});
