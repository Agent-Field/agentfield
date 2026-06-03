import { buildPromptSuffix, cleanupTempFiles, getOutputPath, parseAndValidate } from './schema.js';
import { buildProvider } from './providers/factory.js';
import type { HarnessProvider } from './providers/base.js';
import { createHarnessResult, createRawResult, type HarnessConfig, type HarnessOptions, type RawResult } from './types.js';

const TRANSIENT_PATTERNS = [
  'rate limit',
  'rate_limit',
  'overloaded',
  'timeout',
  'timed out',
  'connection reset',
  'connection refused',
  'temporarily unavailable',
  'service unavailable',
  '503',
  '502',
  '504',
  'internal server error',
  '500',
];

type RunnerOptions = Omit<HarnessOptions, 'schema'> & {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffFactor?: number;
  codexBin?: string;
  geminiBin?: string;
  opencodeBin?: string;
};

export class HarnessRunner {
  public constructor(private readonly config?: HarnessConfig) {}

  public async run(prompt: string, options: HarnessOptions = {}) {
    const { schema, ...rest } = options;
    const resolved = this.resolveOptions(this.config, rest);

    if (!resolved.provider) {
      throw new Error("No harness provider specified. Set 'provider' in HarnessConfig or pass it to .harness() call.");
    }

    const cwd = resolved.cwd ?? '.';
    const provider = await this.buildProvider(resolved.provider, resolved);
    const effectivePrompt = schema === undefined ? prompt : `${prompt}${buildPromptSuffix(schema, cwd)}`;
    const startTime = Date.now();

    try {
      const raw = await this.executeWithRetry(provider, effectivePrompt, resolved);

      if (schema !== undefined) {
        return this.handleSchemaOutput(raw, schema, cwd, startTime);
      }

      return createHarnessResult({
        result: raw.result,
        isError: raw.isError,
        errorMessage: raw.errorMessage,
        costUsd: raw.metrics.totalCostUsd,
        numTurns: raw.metrics.numTurns,
        durationMs: Date.now() - startTime,
        sessionId: raw.metrics.sessionId,
        messages: raw.messages,
      });
    } finally {
      if (schema !== undefined) {
        cleanupTempFiles(cwd);
      }
    }
  }

  public resolveOptions(config: Partial<HarnessConfig> | undefined, overrides: RunnerOptions): RunnerOptions {
    const out: RunnerOptions = {};
    if (config) {
      for (const key of [
        'provider',
        'model',
        'maxTurns',
        'maxBudgetUsd',
        'maxRetries',
        'initialDelay',
        'maxDelay',
        'backoffFactor',
        'tools',
        'permissionMode',
        'systemPrompt',
        'env',
        'cwd',
        'codexBin',
        'geminiBin',
        'opencodeBin',
      ] as const) {
        const value = config[key];
        if (value !== undefined && value !== null) {
          (out as Record<string, unknown>)[key] = value;
        }
      }
    }

    for (const [key, value] of Object.entries(overrides)) {
      if (value !== undefined && value !== null) {
        out[key as keyof RunnerOptions] = value as never;
      }
    }

    return out;
  }

  public isTransient(errorStr: string): boolean {
    const lower = errorStr.toLowerCase();
    return TRANSIENT_PATTERNS.some((pattern) => lower.includes(pattern));
  }

  public async executeWithRetry(provider: HarnessProvider, prompt: string, options: RunnerOptions): Promise<RawResult> {
    const maxRetries = options.maxRetries ?? 3;
    const initialDelay = options.initialDelay ?? 1.0;
    const maxDelay = options.maxDelay ?? 30.0;
    const backoffFactor = options.backoffFactor ?? 2.0;

    let lastError: unknown;

    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      try {
        const result = await provider.execute(prompt, options as Record<string, unknown>);
        if (!result.isError) {
          return result;
        }

        const message = result.errorMessage ?? '';
        if (this.isTransient(message) && attempt < maxRetries) {
          const delay = this.computeBackoffDelay(initialDelay, backoffFactor, maxDelay, attempt);
          await this.sleep(delay);
          continue;
        }
        return result;
      } catch (error: unknown) {
        lastError = error;
        const message = error instanceof Error ? error.message : String(error);
        if (this.isTransient(message) && attempt < maxRetries) {
          const delay = this.computeBackoffDelay(initialDelay, backoffFactor, maxDelay, attempt);
          await this.sleep(delay);
          continue;
        }
        throw error;
      }
    }

    if (lastError !== undefined) {
      throw lastError;
    }
    return createRawResult({ isError: true, errorMessage: 'Max retries exceeded' });
  }

  public handleSchemaOutput(raw: RawResult, schema: unknown, cwd: string, startTime: number) {
    const outputPath = getOutputPath(cwd);
    const parsed = parseAndValidate(outputPath, schema);

    if (parsed !== null) {
      return createHarnessResult({
        result: raw.result,
        parsed,
        isError: false,
        costUsd: raw.metrics.totalCostUsd,
        numTurns: raw.metrics.numTurns,
        durationMs: Date.now() - startTime,
        sessionId: raw.metrics.sessionId,
        messages: raw.messages,
      });
    }

    return createHarnessResult({
      result: raw.result,
      isError: true,
      errorMessage: 'Schema validation failed after parse and cosmetic repair attempts.',
      costUsd: raw.metrics.totalCostUsd,
      numTurns: raw.metrics.numTurns,
      durationMs: Date.now() - startTime,
      sessionId: raw.metrics.sessionId,
      messages: raw.messages,
    });
  }

  private async buildProvider(providerName: string, options: RunnerOptions): Promise<HarnessProvider> {
    const { provider: _, ...rest } = options;
    return buildProvider({ provider: providerName as HarnessConfig['provider'], ...rest });
  }

  private computeBackoffDelay(
    initialDelay: number,
    backoffFactor: number,
    maxDelay: number,
    attempt: number
  ): number {
    const base = Math.min(initialDelay * backoffFactor ** attempt, maxDelay);
    const jitter = (Math.random() * (base * 0.5)) - (base * 0.25);
    return base + jitter;
  }

  private sleep(delaySeconds: number): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, Math.max(0, delaySeconds) * 1000);
    });
  }
}
