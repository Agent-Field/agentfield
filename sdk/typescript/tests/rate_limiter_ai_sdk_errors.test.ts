import { APICallError, generateText } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { StatelessRateLimiter } from '../src/ai/RateLimiter.js';

function providerError() {
  return new APICallError({
    message: 'Provider returned error',
    url: 'https://example.invalid/v1/chat',
    requestBodyValues: {},
    statusCode: 429,
    responseHeaders: { 'retry-after': '90' }
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('StatelessRateLimiter with AI SDK errors', () => {
  it.each([0, 1])('honors Retry-After when generateText runs with maxRetries=%i', async (maxRetries) => {
    vi.useFakeTimers();
    const limiter = new StatelessRateLimiter({ maxRetries: 1, maxDelay: 300, jitterFactor: 0 });
    const sleeps: number[] = [];
    vi.spyOn(limiter as any, '_sleep').mockImplementation(async (delay: any) => {
      sleeps.push(delay);
    });

    let calls = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        calls += 1;
        throw providerError();
      }
    });

    const outcome = limiter
      .executeWithRetry(() => generateText({ model, prompt: 'hi', maxRetries }))
      .catch((error: unknown) => error);
    await vi.runAllTimersAsync();

    await expect(outcome).resolves.toMatchObject({ name: 'RateLimitError', retryAfter: 90 });
    expect(sleeps).toEqual([90]);
    expect(calls).toBe(2 * (maxRetries + 1));
  });
});
