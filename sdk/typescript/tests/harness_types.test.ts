import { describe, it, expect } from 'vitest';

import { buildProvider } from '../src/harness/providers/factory.js';
import { CodexProvider } from '../src/harness/providers/codex.js';
import {
  createHarnessResult,
  createMetrics,
  createRawResult,
  type HarnessConfig,
  type RawResult,
  type Metrics,
  type HarnessResult,
} from '../src/harness/index.js';

describe('harness types', () => {
  it('creates Metrics defaults', () => {
    const metrics: Metrics = createMetrics();

    expect(metrics.durationMs).toBe(0);
    expect(metrics.durationApiMs).toBe(0);
    expect(metrics.numTurns).toBe(0);
    expect(metrics.totalCostUsd).toBeUndefined();
    expect(metrics.usage).toBeUndefined();
    expect(metrics.sessionId).toBe('');
  });

  it('creates RawResult defaults', () => {
    const raw: RawResult = createRawResult();

    expect(raw.result).toBeUndefined();
    expect(raw.messages).toEqual([]);
    expect(raw.metrics.durationMs).toBe(0);
    expect(raw.isError).toBe(false);
    expect(raw.errorMessage).toBeUndefined();
  });

  it('creates HarnessResult defaults and text property', () => {
    const result: HarnessResult = createHarnessResult();

    expect(result.result).toBeUndefined();
    expect(result.parsed).toBeUndefined();
    expect(result.isError).toBe(false);
    expect(result.errorMessage).toBeUndefined();
    expect(result.costUsd).toBeUndefined();
    expect(result.numTurns).toBe(0);
    expect(result.durationMs).toBe(0);
    expect(result.sessionId).toBe('');
    expect(result.messages).toEqual([]);
    expect(result.text).toBe('');
  });

  it('reads text from result when present', () => {
    const result = createHarnessResult({ result: 'done', isError: true, errorMessage: 'boom' });

    expect(result.text).toBe('done');
    expect(result.isError).toBe(true);
    expect(result.errorMessage).toBe('boom');
  });

  it('buildProvider throws for unknown provider', async () => {
    const badConfig = { provider: 'unknown-provider' } as unknown as HarnessConfig;

    await expect(buildProvider(badConfig)).rejects.toThrow(/Unknown harness provider/);
  });

  it('buildProvider returns codex provider', async () => {
    const config: HarnessConfig = { provider: 'codex' };

    const provider = await buildProvider(config);
    expect(provider).toBeInstanceOf(CodexProvider);
  });
});
