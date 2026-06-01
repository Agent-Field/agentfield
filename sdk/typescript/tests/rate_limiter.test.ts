import { afterEach, describe, expect, it, vi } from 'vitest';
import { RateLimitError, StatelessRateLimiter } from '../src/ai/RateLimiter.js';

class DummyHTTPError extends Error {
  statusCode = 429;
  retryAfter?: number;
  response: any;

  constructor(message = 'rate limited', retryAfter?: number) {
    super(message);
    this.retryAfter = retryAfter;
    this.response = {
      statusCode: this.statusCode,
      headers: { 'Retry-After': retryAfter }
    };
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('StatelessRateLimiter', () => {
  it('detects rate limit errors by status code', () => {
    const limiter = new StatelessRateLimiter();

    expect((limiter as any)._isRateLimitError(new DummyHTTPError())).toBe(true);
    expect((limiter as any)._isRateLimitError({ constructor: { name: 'RateLimitError' } })).toBe(true);
    expect((limiter as any)._isRateLimitError(new Error('other error'))).toBe(false);
  });

  it('prefers retry-after header when calculating backoff', () => {
    const limiter = new StatelessRateLimiter({ baseDelay: 1.0, jitterFactor: 0.0, maxDelay: 10.0 });

    expect((limiter as any)._calculateBackoffDelay(0, 2.5)).toBeCloseTo(2.5);
    expect((limiter as any)._calculateBackoffDelay(1, undefined)).toBeCloseTo(2.0);
  });

  it('retries on rate limits until success', async () => {
    const limiter = new StatelessRateLimiter({ maxRetries: 3, baseDelay: 0.01, jitterFactor: 0.0 });
    const sleeps: number[] = [];
    vi.spyOn(limiter as any, '_sleep').mockImplementation(async (delay: number) => {
      sleeps.push(delay);
    });

    let attempts = 0;
    const result = await limiter.executeWithRetry(async () => {
      attempts += 1;
      if (attempts < 3) {
        throw new DummyHTTPError('rate limited', 0.2);
      }
      return 'ok';
    });

    expect(result).toBe('ok');
    expect(attempts).toBe(3);
    expect(sleeps.length).toBe(2);
    expect(sleeps[0]).toBeCloseTo(0.2);
  });

  it('gives up after max retries', async () => {
    const limiter = new StatelessRateLimiter({ maxRetries: 1, baseDelay: 0.01, jitterFactor: 0.0 });
    vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

    await expect(
      limiter.executeWithRetry(async () => {
        throw new DummyHTTPError();
      })
    ).rejects.toBeInstanceOf(RateLimitError);
  });

  it('applies jitter and caps at max delay', () => {
    const limiter = new StatelessRateLimiter({ baseDelay: 0.5, jitterFactor: 0.3, maxDelay: 1.0 });
    (limiter as any)._containerSeed = 42;

    const attempt = 4;
    const expectedBase = limiter.maxDelay;
    const rng = (limiter as any)._createJitterRng((limiter as any)._containerSeed + attempt);
    const jitterRange = expectedBase * limiter.jitterFactor;
    const expectedDelay = Math.max(0.1, expectedBase + (rng() * 2 - 1) * jitterRange);

    const delay = (limiter as any)._calculateBackoffDelay(attempt);
    expect(delay).toBeCloseTo(expectedDelay);
  });

  it('extracts retry-after from attribute fallback', () => {
    const limiter = new StatelessRateLimiter();

    class Response {
      headers = { 'Retry-After': 'invalid' };
    }

    class ErrorWithRetry {
      response = new Response();
      retryAfter = '7';
      toString() {
        return 'rate limit';
      }
    }

    expect((limiter as any)._extractRetryAfter(new ErrorWithRetry())).toBeCloseTo(7);

    class ErrorWithoutRetry {
      response = new Response();
      retryAfter = {};
      toString() {
        return 'rate limit';
      }
    }

    expect((limiter as any)._extractRetryAfter(new ErrorWithoutRetry())).toBeUndefined();
  });

  it('detects rate limit keywords and status codes', () => {
    const limiter = new StatelessRateLimiter();

    class MsgError extends Error {
      statusCode = 503;
      override message = 'temporarily rate-limited by server';
    }

    expect((limiter as any)._isRateLimitError(new MsgError())).toBe(true);
    expect((limiter as any)._isRateLimitError(new Error('all good'))).toBe(false);
  });

  it('blocks while circuit breaker is open and then recovers', async () => {
    const limiter = new StatelessRateLimiter({
      maxRetries: 0,
      baseDelay: 0.01,
      jitterFactor: 0.0,
      circuitBreakerThreshold: 1,
      circuitBreakerTimeout: 5
    });
    vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

    let now = 100;
    vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

    await expect(
      limiter.executeWithRetry(async () => {
        throw new DummyHTTPError();
      })
    ).rejects.toBeInstanceOf(RateLimitError);

    expect((limiter as any)._circuitOpenTime).toBe(now);

    let shouldRun = false;
    await expect(
      limiter.executeWithRetry(async () => {
        shouldRun = true;
        return 'ok';
      })
    ).rejects.toBeInstanceOf(RateLimitError);
    expect(shouldRun).toBe(false);

    now += 10;
    const result = await limiter.executeWithRetry(async () => 'ok');

    expect(result).toBe('ok');
    expect((limiter as any)._consecutiveFailures).toBe(0);
    expect((limiter as any)._circuitOpenTime).toBeUndefined();
  });

  it('rethrows non rate limit errors immediately', async () => {
    const limiter = new StatelessRateLimiter();

    await expect(
      limiter.executeWithRetry(async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');
  });
});
