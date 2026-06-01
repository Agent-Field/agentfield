import { afterEach, describe, expect, it, vi } from 'vitest';

import { HarnessRunner } from '../src/harness/runner.js';
import { createMetrics, createRawResult } from '../src/harness/types.js';
import type { HarnessProvider } from '../src/harness/providers/base.js';
import * as factory from '../src/harness/providers/factory.js';

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('HarnessRunner resilience behavior', () => {
  it('retries thrown transient errors with exponential backoff', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const provider: HarnessProvider = {
      execute: vi.fn()
        .mockRejectedValueOnce(new Error('503 service unavailable'))
        .mockRejectedValueOnce(new Error('timeout talking to upstream'))
        .mockResolvedValueOnce(createRawResult({ result: 'ok' }))
    };

    const runner = new HarnessRunner();
    const promise = runner.executeWithRetry(provider, 'prompt', {
      maxRetries: 2,
      initialDelay: 1,
      maxDelay: 10,
      backoffFactor: 3
    });

    await Promise.resolve();
    expect(provider.execute).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1000);
    expect(provider.execute).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(3000);
    await expect(promise).resolves.toMatchObject({ result: 'ok' });
    expect(provider.execute).toHaveBeenCalledTimes(3);
  });

  it('fails immediately on non-transient thrown errors', async () => {
    vi.useFakeTimers();

    const error = new Error('invalid api key');
    const provider: HarnessProvider = {
      execute: vi.fn().mockRejectedValue(error)
    };

    const runner = new HarnessRunner();

    await expect(
      runner.executeWithRetry(provider, 'prompt', {
        maxRetries: 3,
        initialDelay: 1,
        maxDelay: 10,
        backoffFactor: 2
      })
    ).rejects.toBe(error);
    expect(provider.execute).toHaveBeenCalledTimes(1);
  });

  it('computes backoff using the capped exponential base plus jitter', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const runner = new HarnessRunner();

    expect((runner as any).computeBackoffDelay(1, 2, 10, 0)).toBeCloseTo(1);
    expect((runner as any).computeBackoffDelay(1, 2, 10, 2)).toBeCloseTo(4);
    expect((runner as any).computeBackoffDelay(2, 3, 5, 3)).toBeCloseTo(5);
  });

  it('returns the final transient error result after retries are exhausted', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const provider: HarnessProvider = {
      execute: vi.fn().mockResolvedValue(
        createRawResult({
          isError: true,
          errorMessage: '504 gateway timeout',
          metrics: createMetrics({ totalCostUsd: 0.25, numTurns: 1, sessionId: 'attempt-1' })
        })
      )
    };

    const runner = new HarnessRunner();
    const promise = runner.executeWithRetry(provider, 'prompt', {
      maxRetries: 1,
      initialDelay: 1,
      maxDelay: 10,
      backoffFactor: 2
    });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(1000);

    const raw = await promise;

    expect(provider.execute).toHaveBeenCalledTimes(2);
    expect(raw.isError).toBe(true);
    expect(raw.errorMessage).toBe('504 gateway timeout');
  });

  it('surfaces only the final attempt metrics from run()', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const provider: HarnessProvider = {
      execute: vi.fn()
        .mockResolvedValueOnce(
          createRawResult({
            isError: true,
            errorMessage: 'rate limit exceeded',
            metrics: createMetrics({ totalCostUsd: 0.25, numTurns: 1, sessionId: 'attempt-1' })
          })
        )
        .mockResolvedValueOnce(
          createRawResult({
            result: 'ok',
            metrics: createMetrics({ totalCostUsd: 0.5, numTurns: 2, sessionId: 'attempt-2' })
          })
        )
    };
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const promise = runner.run('hello', {
      provider: 'codex',
      maxRetries: 1,
      initialDelay: 1,
      maxDelay: 10,
      backoffFactor: 2
    });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(1000);

    const result = await promise;

    expect(result.result).toBe('ok');
    expect(result.costUsd).toBe(0.5);
    expect(result.numTurns).toBe(2);
    expect(result.sessionId).toBe('attempt-2');
  });
});
