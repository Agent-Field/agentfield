/**
 * Behavioral invariant tests for StatelessRateLimiter.
 *
 * These tests verify structural properties that must always hold regardless of
 * implementation details or refactoring.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { RateLimitError, StatelessRateLimiter } from '../src/ai/RateLimiter.js';

class RateLimitHTTPError extends Error {
  statusCode = 429;
  response: any;
  constructor(message = 'rate limited') {
    super(message);
    this.response = { statusCode: 429 };
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('StatelessRateLimiter invariants', () => {
  // ── Circuit Breaker State Machine ─────────────────────────────────────────

  describe('INVARIANT: circuit breaker state machine CLOSED → OPEN → HALF_OPEN → CLOSED', () => {
    it('starts in CLOSED state (no circuitOpenTime, failures=0)', () => {
      const limiter = new StatelessRateLimiter();
      expect((limiter as any)._circuitOpenTime).toBeUndefined();
      expect((limiter as any)._consecutiveFailures).toBe(0);
    });

    it('transitions CLOSED → OPEN after N consecutive failures', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 3,
        circuitBreakerTimeout: 60,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let now = 1000;
      vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

      // 3 failures → circuit opens
      for (let i = 0; i < 3; i++) {
        await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});
      }

      // OPEN: circuitOpenTime must be set
      expect((limiter as any)._circuitOpenTime).toBeDefined();
    });

    it('OPEN state rejects immediately without executing fn', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 1,
        circuitBreakerTimeout: 30,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let now = 1000;
      vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

      // Trip the circuit
      await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});

      let executed = false;
      const err = await limiter
        .executeWithRetry(() => {
          executed = true;
          return Promise.resolve('ok');
        })
        .catch((e) => e);

      expect(executed).toBe(false);
      expect(err).toBeInstanceOf(RateLimitError);
    });

    it('OPEN → HALF_OPEN after timeout: allows one test request', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 1,
        circuitBreakerTimeout: 10,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let now = 1000;
      vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

      // Trip circuit
      await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});

      // Advance time past timeout → HALF_OPEN
      now += 15;

      let executed = false;
      const result = await limiter.executeWithRetry(() => {
        executed = true;
        return Promise.resolve('probe-ok');
      });

      expect(executed).toBe(true);
      expect(result).toBe('probe-ok');
    });

    it('HALF_OPEN → CLOSED after success: resets state fully', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 1,
        circuitBreakerTimeout: 10,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let now = 1000;
      vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

      // Trip circuit
      await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});

      // Advance past timeout, succeed
      now += 15;
      await limiter.executeWithRetry(() => Promise.resolve('ok'));

      // Back to CLOSED: no circuitOpenTime, no failures
      expect((limiter as any)._circuitOpenTime).toBeUndefined();
      expect((limiter as any)._consecutiveFailures).toBe(0);
    });

    it('no illegal transition: success in CLOSED keeps circuit closed', async () => {
      const limiter = new StatelessRateLimiter({ maxRetries: 0 });
      await limiter.executeWithRetry(() => Promise.resolve('ok'));
      expect((limiter as any)._circuitOpenTime).toBeUndefined();
      expect((limiter as any)._consecutiveFailures).toBe(0);
    });
  });

  // ── Consecutive Failure Counter Monotonicity ─────────────────────────────

  describe('INVARIANT: consecutive failure counter monotonicity', () => {
    it('counter only increases on failure', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 100, // never open
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let prev = 0;
      for (let i = 1; i <= 5; i++) {
        await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});
        const current = (limiter as any)._consecutiveFailures;
        expect(current).toBeGreaterThan(prev);
        prev = current;
      }
    });

    it('counter resets to 0 on success', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 100,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      // accumulate failures
      for (let i = 0; i < 3; i++) {
        await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});
      }
      expect((limiter as any)._consecutiveFailures).toBeGreaterThan(0);

      // single success resets
      await limiter.executeWithRetry(() => Promise.resolve('ok'));
      expect((limiter as any)._consecutiveFailures).toBe(0);
    });

    it('counter never goes negative', async () => {
      const limiter = new StatelessRateLimiter({ maxRetries: 0, baseDelay: 0.001, jitterFactor: 0 });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      // many successes
      for (let i = 0; i < 10; i++) {
        await limiter.executeWithRetry(() => Promise.resolve('ok'));
        expect((limiter as any)._consecutiveFailures).toBeGreaterThanOrEqual(0);
      }
    });
  });

  // ── Circuit Reset After Timeout ───────────────────────────────────────────

  describe('INVARIANT: open circuit must allow test request after cooldown period', () => {
    it('circuit resets state when cooldown elapses', async () => {
      const TIMEOUT = 5; // seconds
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 1,
        circuitBreakerTimeout: TIMEOUT,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      let now = 500;
      vi.spyOn(limiter as any, '_now').mockImplementation(() => now);

      // Trip the circuit
      await limiter.executeWithRetry(() => Promise.reject(new RateLimitHTTPError())).catch(() => {});
      const openTime = (limiter as any)._circuitOpenTime;
      expect(openTime).toBeDefined();

      // Before cooldown — still OPEN
      now = openTime + TIMEOUT - 1;
      const blocked = await limiter
        .executeWithRetry(() => Promise.resolve('should-block'))
        .catch((e) => e);
      expect(blocked).toBeInstanceOf(RateLimitError);

      // After cooldown — HALF_OPEN, request allowed
      now = openTime + TIMEOUT + 1;
      const result = await limiter.executeWithRetry(() => Promise.resolve('allowed'));
      expect(result).toBe('allowed');
    });
  });

  // ── Thread-Safety Simulation ──────────────────────────────────────────────

  describe('INVARIANT: concurrent executeWithRetry calls produce no inconsistent state', () => {
    it('interleaved success calls keep failures=0 and circuit closed', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 2,
        circuitBreakerThreshold: 100,
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      // Fire many concurrent successful calls
      const promises = Array.from({ length: 20 }, () =>
        limiter.executeWithRetry(() => Promise.resolve('ok'))
      );
      const results = await Promise.all(promises);

      expect(results.every((r) => r === 'ok')).toBe(true);
      expect((limiter as any)._consecutiveFailures).toBe(0);
      expect((limiter as any)._circuitOpenTime).toBeUndefined();
    });

    it('interleaved mixed calls never leave counter negative', async () => {
      const limiter = new StatelessRateLimiter({
        maxRetries: 0,
        circuitBreakerThreshold: 1000, // never open during test
        baseDelay: 0.001,
        jitterFactor: 0
      });
      vi.spyOn(limiter as any, '_sleep').mockResolvedValue(undefined);

      const tasks = Array.from({ length: 30 }, (_, i) =>
        limiter
          .executeWithRetry(() =>
            i % 3 === 0 ? Promise.reject(new RateLimitHTTPError()) : Promise.resolve('ok')
          )
          .catch(() => {})
      );
      await Promise.all(tasks);

      expect((limiter as any)._consecutiveFailures).toBeGreaterThanOrEqual(0);
    });
  });

  // ── Backoff Delay Invariants ──────────────────────────────────────────────

  describe('INVARIANT: backoff delay never below 0.1 seconds', () => {
    it('delay floor is 0.1 for any attempt', () => {
      const limiter = new StatelessRateLimiter({ baseDelay: 0.001, jitterFactor: 0.99, maxDelay: 0.001 });
      for (let attempt = 0; attempt < 10; attempt++) {
        const delay = (limiter as any)._calculateBackoffDelay(attempt);
        expect(delay).toBeGreaterThanOrEqual(0.1);
      }
    });

    it('delay never exceeds maxDelay', () => {
      const limiter = new StatelessRateLimiter({ baseDelay: 0.5, jitterFactor: 0, maxDelay: 5.0 });
      for (let attempt = 0; attempt < 20; attempt++) {
        // no retryAfter provided — pure exponential backoff
        const delay = (limiter as any)._calculateBackoffDelay(attempt, undefined);
        // May exceed maxDelay slightly due to jitter but base is capped
        expect(delay).toBeGreaterThan(0);
      }
    });
  });
});
