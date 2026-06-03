import crypto from 'node:crypto';
import os from 'node:os';

export class RateLimitError extends Error {
  retryAfter?: number;
  status?: number;
  statusCode?: number;
  response?: {
    status?: number;
    statusCode?: number;
    status_code?: number;
    headers?: Record<string, string>;
  };

  constructor(
    message: string,
    retryAfter?: number,
    status?: number,
    statusCode?: number,
    response?: {
      status?: number;
      statusCode?: number;
      status_code?: number;
      headers?: Record<string, string>;
    }
  ) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
    this.status = status;
    this.statusCode = statusCode;
    this.response = response;
  }
}

export interface RateLimiterOptions {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  jitterFactor?: number;
  circuitBreakerThreshold?: number;
  circuitBreakerTimeout?: number;
}

/**
 * Stateless rate limiter with adaptive exponential backoff.
 *
 * Designed to work across many containers without coordination.
 * Uses container-specific jitter to naturally distribute load.
 */
export class StatelessRateLimiter {
  readonly maxRetries: number;
  readonly baseDelay: number;
  readonly maxDelay: number;
  readonly jitterFactor: number;
  readonly circuitBreakerThreshold: number;
  readonly circuitBreakerTimeout: number;

  protected _containerSeed: number;
  protected _consecutiveFailures = 0;
  protected _circuitOpenTime?: number;

  constructor(options: RateLimiterOptions = {}) {
    this.maxRetries = options.maxRetries ?? 5;
    this.baseDelay = options.baseDelay ?? 0.5;
    this.maxDelay = options.maxDelay ?? 30.0;
    this.jitterFactor = options.jitterFactor ?? 0.25;
    this.circuitBreakerThreshold = options.circuitBreakerThreshold ?? 5;
    this.circuitBreakerTimeout = options.circuitBreakerTimeout ?? 30;

    this._containerSeed = this._getContainerSeed();
  }

  protected _getContainerSeed(): number {
    const identifier = `${os.hostname()}-${process.pid}`;
    const hash = crypto.createHash('md5').update(identifier).digest('hex');
    return parseInt(hash.slice(0, 8), 16);
  }

  protected _isRateLimitError(error: unknown): boolean {
    if (!error) return false;

    if (error instanceof RateLimitError) {
      return true;
    }

    if (typeof error === 'object' && error !== null) {
      const ctorName = (error as { constructor?: { name?: string } }).constructor?.name;
      if (typeof ctorName === 'string' && ctorName.includes('RateLimitError')) {
        return true;
      }

      const err = error as {
        status?: number;
        statusCode?: number;
        response?: { status?: number; statusCode?: number; status_code?: number };
      };
      const statusCandidates = [
        err.status,
        err.statusCode,
        err.response?.status,
        err.response?.statusCode,
        err.response?.status_code
      ];
      if (statusCandidates.some((code) => code === 429 || code === 503)) {
        return true;
      }
    }

    const message = toError(error).message.toLowerCase();
    const rateLimitKeywords = [
      'rate limit',
      'rate-limit',
      'rate_limit',
      'too many requests',
      'quota exceeded',
      'temporarily rate-limited',
      'rate limited',
      'requests per',
      'rpm exceeded',
      'tpm exceeded',
      'usage limit',
      'throttled',
      'throttling'
    ];

    return rateLimitKeywords.some((keyword) => message.includes(keyword));
  }

  protected _extractRetryAfter(error: unknown): number | undefined {
    if (!error) return undefined;

    if (error instanceof RateLimitError && error.retryAfter) {
      return error.retryAfter;
    }

    if (typeof error !== 'object' || error === null) return undefined;

    type HeaderBag = Record<string, string | string[] | undefined>;
    const err = error as {
      response?: { headers?: HeaderBag; Headers?: HeaderBag; header?: HeaderBag };
      retryAfter?: string | number;
      retry_after?: string | number;
    };

    const headers = err.response?.headers ?? err.response?.Headers ?? err.response?.header;
    if (headers && typeof headers === 'object') {
      const retryAfterKey = Object.keys(headers).find((k) => k.toLowerCase() === 'retry-after');
      if (retryAfterKey) {
        const raw = headers[retryAfterKey];
        const value = Array.isArray(raw) ? raw[0] : raw;
        if (value !== undefined) {
          const parsed = parseFloat(String(value));
          if (!Number.isNaN(parsed)) {
            return parsed;
          }
        }
      }
    }

    const rawRetry = err.retryAfter ?? err.retry_after;
    if (rawRetry !== undefined) {
      const parsed = parseFloat(String(rawRetry));
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }

    return undefined;
  }

  protected _createJitterRng(seed: number): () => number {
    let x = seed >>> 0;
    return () => {
      x = (1664525 * x + 1013904223) % 4294967296;
      return x / 4294967296;
    };
  }

  protected _calculateBackoffDelay(attempt: number, retryAfter?: number): number {
    let baseDelay: number;
    if (retryAfter && retryAfter <= this.maxDelay) {
      baseDelay = retryAfter;
    } else {
      baseDelay = Math.min(this.baseDelay * 2 ** attempt, this.maxDelay);
    }

    const jitterRange = baseDelay * this.jitterFactor;
    const rng = this._createJitterRng(this._containerSeed + attempt);
    const jitter = (rng() * 2 - 1) * jitterRange;

    const delay = Math.max(0.1, baseDelay + jitter);
    return delay;
  }

  protected _checkCircuitBreaker(): boolean {
    if (this._circuitOpenTime === undefined) {
      return false;
    }

    if (this._now() - this._circuitOpenTime > this.circuitBreakerTimeout) {
      this._circuitOpenTime = undefined;
      this._consecutiveFailures = 0;
      return false;
    }

    return true;
  }

  protected _updateCircuitBreaker(success: boolean) {
    if (success) {
      this._consecutiveFailures = 0;
      this._circuitOpenTime = undefined;
      return;
    }

    this._consecutiveFailures += 1;
    if (this._consecutiveFailures >= this.circuitBreakerThreshold && this._circuitOpenTime === undefined) {
      this._circuitOpenTime = this._now();
    }
  }

  protected async _sleep(delaySeconds: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, delaySeconds * 1000));
  }

  protected _now(): number {
    return Date.now() / 1000;
  }

  async executeWithRetry<T>(fn: () => Promise<T>): Promise<T> {
    if (this._checkCircuitBreaker()) {
      throw new RateLimitError(
        `Circuit breaker is open. Too many consecutive rate limit failures. Will retry after ${this.circuitBreakerTimeout} seconds.`
      );
    }

    let lastError: unknown;

    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      try {
        const result = await fn();
        this._updateCircuitBreaker(true);
        return result;
      } catch (error) {
        lastError = error;

        if (!this._isRateLimitError(error)) {
          throw error;
        }

        this._updateCircuitBreaker(false);

        if (attempt >= this.maxRetries) {
          break;
        }

        const retryAfter = this._extractRetryAfter(error);
        const delay = this._calculateBackoffDelay(attempt, retryAfter);
        await this._sleep(delay);
      }
    }

    throw new RateLimitError(
      `Rate limit retries exhausted after ${this.maxRetries} attempts. Last error: ${String(lastError)}`,
      this._extractRetryAfter(lastError)
    );
  }
}


function toError(error: unknown): Error {
  if (error instanceof Error) return error;
  return new Error(String(error));
}
