import { describe, it, expect, vi } from 'vitest';
import type express from 'express';
import type { Agent } from '../src/agent/Agent.js';
import {
  CostTracker,
  USAGE_ENVELOPE_KEY,
  attachUsageToSyncResult,
  deriveProvider,
  usageSummaryOrNull
} from '../src/usage/costTracker.js';
import {
  extractProviderCostUsd,
  extractTokenUsage,
  recordAiSdkUsage
} from '../src/usage/aiUsage.js';
import { withOpenRouterUsageInclude } from '../src/ai/openrouterUsage.js';
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

describe('deriveProvider', () => {
  it('extracts the first slug segment, lowercased', () => {
    expect(deriveProvider('anthropic/claude-opus-4-8')).toBe('anthropic');
    expect(deriveProvider('OpenRouter/qwen/qwen3-coder')).toBe('openrouter');
  });

  it('returns null for bare model names and empty input', () => {
    expect(deriveProvider('gpt-4o')).toBeNull();
    expect(deriveProvider('')).toBeNull();
    expect(deriveProvider(undefined)).toBeNull();
    expect(deriveProvider('/leading-slash')).toBeNull();
  });
});

describe('CostTracker record/serialize (wire contract)', () => {
  it('serializes an unpriced entry with null cost and derived provider', () => {
    const t = new CostTracker();
    t.record({
      model: 'openrouter/qwen/qwen3-coder',
      reasonerName: 'code',
      inputTokens: 500,
      outputTokens: 200,
      totalTokens: 700
    });

    expect(t.serialize()).toEqual({
      total_cost_usd: null,
      total_input_tokens: 500,
      total_output_tokens: 200,
      total_tokens: 700,
      entries: [
        {
          source: 'llm',
          provider: 'openrouter',
          model: 'openrouter/qwen/qwen3-coder',
          harness: null,
          reasoner: 'code',
          input_tokens: 500,
          output_tokens: 200,
          cache_read_tokens: 0,
          cache_creation_tokens: 0,
          total_tokens: 700,
          cost_usd: null,
          cost_source: null
        }
      ]
    });
  });

  it('keeps an explicit provider over slug derivation and serializes unset strings as null', () => {
    const t = new CostTracker();
    t.record({ model: 'claude-opus-4-8', provider: 'anthropic', inputTokens: 1 });
    const wire = t.serialize().entries[0];
    expect(wire.provider).toBe('anthropic');
    expect(wire.reasoner).toBeNull();
    expect(wire.harness).toBeNull();
    expect(wire.cost_source).toBeNull();
  });

  it('falls back to input+output for a zero per-entry total, and totals costs to 6 decimals', () => {
    const t = new CostTracker();
    t.record({ model: 'm', inputTokens: 10, outputTokens: 5, costUsd: 0.1000004 });
    t.record({ model: 'm', inputTokens: 1, outputTokens: 1, costUsd: 0.2000004 });
    const wire = t.serialize();
    expect(wire.entries[0].total_tokens).toBe(15);
    expect(wire.total_tokens).toBe(17);
    expect(wire.total_cost_usd).toBe(0.300001);
  });

  it('records a costless-token and tokenless-cost entry without dropping either', () => {
    const t = new CostTracker();
    t.record({ model: 'a', inputTokens: 3 }); // tokens, no cost
    t.record({ model: 'b', costUsd: 0.5, costSource: 'provider' }); // cost, no tokens
    const wire = t.serialize();
    expect(wire.total_cost_usd).toBe(0.5);
    expect(wire.total_input_tokens).toBe(3);
    expect(wire.entries[1].cost_usd).toBe(0.5);
    expect(wire.entries[1].cost_source).toBe('provider');
  });

  it('exposes totals and reset', () => {
    const t = new CostTracker();
    expect(t.hasEntries).toBe(false);
    t.record({ model: 'm', totalTokens: 7, costUsd: 0.25 });
    expect(t.hasEntries).toBe(true);
    expect(t.callCount).toBe(1);
    expect(t.totalTokens).toBe(7);
    expect(t.totalCostUsd).toBe(0.25);
    t.reset();
    expect(t.hasEntries).toBe(false);
  });
});

describe('usageSummaryOrNull / attachUsageToSyncResult', () => {
  it('returns null for missing or empty trackers (no usage object emitted)', () => {
    expect(usageSummaryOrNull(undefined)).toBeNull();
    expect(usageSummaryOrNull(new CostTracker())).toBeNull();
  });

  it('merges usage under the reserved envelope key without touching user keys', () => {
    const t = new CostTracker();
    t.record({ model: 'm', inputTokens: 1 });
    const result = { answer: 42, usage: { user: 'data' } };
    const wrapped = attachUsageToSyncResult(result, t) as Record<string, unknown>;
    expect(wrapped.answer).toBe(42);
    expect(wrapped.usage).toEqual({ user: 'data' });
    expect(wrapped[USAGE_ENVELOPE_KEY]).toMatchObject({ total_input_tokens: 1 });
    // Original result object is not mutated.
    expect(USAGE_ENVELOPE_KEY in result).toBe(false);
  });

  it('passes non-plain-object results through unchanged', () => {
    const t = new CostTracker();
    t.record({ model: 'm', inputTokens: 1 });
    expect(attachUsageToSyncResult('scalar', t)).toBe('scalar');
    expect(attachUsageToSyncResult(7, t)).toBe(7);
    expect(attachUsageToSyncResult(null, t)).toBeNull();
    const arr = [1, 2];
    expect(attachUsageToSyncResult(arr, t)).toBe(arr);
    class Custom { x = 1; }
    const inst = new Custom();
    expect(attachUsageToSyncResult(inst, t)).toBe(inst);
  });

  it('leaves object results unchanged when no usage was recorded', () => {
    const result = { a: 1 };
    expect(attachUsageToSyncResult(result, new CostTracker())).toBe(result);
  });
});

describe('AI SDK usage extraction', () => {
  const sdkUsage = {
    inputTokens: 100,
    outputTokens: 50,
    totalTokens: 150,
    inputTokenDetails: { noCacheTokens: 90, cacheReadTokens: 2048, cacheWriteTokens: 64 },
    outputTokenDetails: { textTokens: 50, reasoningTokens: 0 },
    raw: { prompt_tokens: 100, completion_tokens: 50, cost: 0.0123 }
  };

  it('maps AI SDK v6 usage fields to contract token counts', () => {
    expect(extractTokenUsage(sdkUsage)).toEqual({
      inputTokens: 100,
      outputTokens: 50,
      totalTokens: 150,
      cacheReadTokens: 2048,
      cacheCreationTokens: 64
    });
  });

  it('falls back to the deprecated cachedInputTokens alias', () => {
    expect(extractTokenUsage({ inputTokens: 5, cachedInputTokens: 3 }).cacheReadTokens).toBe(3);
    expect(extractTokenUsage(undefined)).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      cacheReadTokens: 0,
      cacheCreationTokens: 0
    });
  });

  it('reads provider-native cost from usage.raw and sums per-step costs', () => {
    expect(extractProviderCostUsd({ usage: sdkUsage })).toBe(0.0123);
    expect(
      extractProviderCostUsd({
        steps: [{ usage: { raw: { cost: 0.01 } } }, { usage: { raw: { cost: 0.02 } } }, { usage: {} }]
      })
    ).toBeCloseTo(0.03, 10);
    expect(extractProviderCostUsd({ usage: { raw: {} } })).toBeNull();
    expect(extractProviderCostUsd({})).toBeNull();
    // Defensive: dedicated-provider metadata path.
    expect(
      extractProviderCostUsd({ providerMetadata: { openrouter: { usage: { cost: 0.4 } } } })
    ).toBe(0.4);
  });

  it('records into the current execution tracker with reasoner attribution', () => {
    const ctx = makeContext('exec-u1', { reasonerId: 'summarize' });
    ExecutionContext.run(ctx, () => {
      recordAiSdkUsage({ source: { usage: sdkUsage }, model: 'gpt-4o', provider: 'openai' });
    });
    const wire = ctx.costTracker.serialize();
    expect(wire.entries).toHaveLength(1);
    expect(wire.entries[0]).toMatchObject({
      model: 'gpt-4o',
      provider: 'openai',
      reasoner: 'summarize',
      input_tokens: 100,
      output_tokens: 50,
      cache_read_tokens: 2048,
      cache_creation_tokens: 64,
      cost_usd: 0.0123,
      cost_source: 'provider'
    });
  });

  it('skips empty results and calls made outside an execution', () => {
    const ctx = makeContext('exec-u2');
    ExecutionContext.run(ctx, () => {
      recordAiSdkUsage({ source: { usage: undefined }, model: 'gpt-4o' });
      recordAiSdkUsage({ source: {}, model: 'gpt-4o' });
    });
    expect(ctx.costTracker.hasEntries).toBe(false);
    // Outside any execution: must not throw.
    expect(() => recordAiSdkUsage({ source: { usage: sdkUsage }, model: 'gpt-4o' })).not.toThrow();
  });
});

describe('per-execution tracker isolation', () => {
  it('keeps concurrent executions from cross-contaminating usage', async () => {
    const ctxA = makeContext('exec-a', { reasonerId: 'a' });
    const ctxB = makeContext('exec-b', { reasonerId: 'b' });

    await Promise.all([
      ExecutionContext.run(ctxA, async () => {
        await new Promise((r) => setTimeout(r, 5));
        ExecutionContext.getCurrent()?.costTracker.record({ model: 'model-a', inputTokens: 1 });
        await new Promise((r) => setTimeout(r, 5));
        ExecutionContext.getCurrent()?.costTracker.record({ model: 'model-a', inputTokens: 1 });
      }),
      ExecutionContext.run(ctxB, async () => {
        await new Promise((r) => setTimeout(r, 7));
        ExecutionContext.getCurrent()?.costTracker.record({ model: 'model-b', outputTokens: 2 });
      })
    ]);

    expect(ctxA.costTracker.serialize().entries.map((e) => e.model)).toEqual(['model-a', 'model-a']);
    expect(ctxB.costTracker.serialize().entries.map((e) => e.model)).toEqual(['model-b']);
  });
});

describe('withOpenRouterUsageInclude', () => {
  it('injects usage accounting opt-in into JSON object bodies', async () => {
    const base = vi.fn().mockResolvedValue('ok');
    const wrapped = withOpenRouterUsageInclude(base);
    await wrapped('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ model: 'qwen/qwen3-coder', messages: [] })
    });
    const [, init] = base.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      model: 'qwen/qwen3-coder',
      messages: [],
      usage: { include: true }
    });
  });

  it('leaves an existing usage member and non-JSON bodies untouched', async () => {
    const base = vi.fn().mockResolvedValue('ok');
    const wrapped = withOpenRouterUsageInclude(base);

    const withUsage = JSON.stringify({ usage: { include: false } });
    await wrapped('u', { body: withUsage });
    expect(base.mock.calls[0][1].body).toBe(withUsage);

    await wrapped('u', { body: 'not json' });
    expect(base.mock.calls[1][1].body).toBe('not json');

    await wrapped('u');
    expect(base.mock.calls[2][1]).toBeUndefined();
  });
});
