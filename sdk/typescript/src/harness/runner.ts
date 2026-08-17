import fs from 'node:fs';
import path from 'node:path';
import { buildPromptSuffix, cleanupTempFiles, getOutputPath, parseAndValidate } from './schema.js';
import { buildProvider, resolveProviderName } from './providers/factory.js';
import type { HarnessProvider } from './providers/base.js';
import {
  createHarnessResult,
  createRawResult,
  type HarnessConfig,
  type HarnessOptions,
  type HarnessResult,
  type RawResult,
} from './types.js';

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

/** Copy provider-reported token/model metrics onto the harness result. */
function tokenMetrics(raw: RawResult): Pick<
  HarnessResult,
  'inputTokens' | 'outputTokens' | 'cacheReadTokens' | 'cacheCreationTokens' | 'totalTokens' | 'model'
> {
  const { inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens, totalTokens, model } = raw.metrics;
  return { inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens, totalTokens, model };
}

type RunnerOptions = Omit<HarnessOptions, 'schema'> & {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffFactor?: number;
  projectDir?: string;
  aforgeBin?: string;
  codexBin?: string;
  geminiBin?: string;
  opencodeBin?: string;
};

export class HarnessRunner {
  public constructor(private readonly config?: HarnessConfig) {}

  public async run(prompt: string, options: HarnessOptions = {}) {
    const { schema, ...rest } = options;
    const resolved = this.resolveOptions(this.config, rest);
    resolved.provider = resolveProviderName(resolved.provider);

    const provider = await this.buildProvider(resolved.provider, resolved);
    const cwd = resolved.cwd ?? '.';
    const outputRoot = resolved.projectDir ?? cwd;
    let outputDir: string | undefined;
    if (schema !== undefined) {
      fs.mkdirSync(outputRoot, { recursive: true });
      outputDir = fs.mkdtempSync(path.join(outputRoot, '.agentfield-out-'));
    }
    const effectivePrompt = schema === undefined ? prompt : `${prompt}${buildPromptSuffix(schema, outputDir!)}`;
    const startTime = Date.now();

    try {
      const raw = await this.executeWithRetry(provider, effectivePrompt, resolved);

      if (schema !== undefined) {
        return this.handleSchemaOutput(raw, schema, outputDir!, startTime);
      }

      return createHarnessResult({
        result: raw.result,
        isError: raw.isError,
        errorMessage: raw.errorMessage,
        failureType: raw.failureType,
        returnCode: raw.returnCode,
        costUsd: raw.metrics.totalCostUsd,
        numTurns: raw.metrics.numTurns,
        durationMs: Date.now() - startTime,
        sessionId: raw.metrics.sessionId,
        messages: raw.messages,
        ...tokenMetrics(raw),
      });
    } finally {
      if (schema !== undefined) {
        cleanupTempFiles(outputDir!);
        fs.rmSync(outputDir!, { recursive: true, force: true });
      }
    }
  }

  public resolveOptions(config: Partial<HarnessConfig> | undefined, overrides: RunnerOptions): RunnerOptions {
    const out: RunnerOptions = {};
    if (config) {
      for (const key of [
        'provider',
        'model',
        'variant',
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
        'projectDir',
        'aforgeBin',
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
        failureType: raw.failureType,
        returnCode: raw.returnCode,
        costUsd: raw.metrics.totalCostUsd,
        numTurns: raw.metrics.numTurns,
        durationMs: Date.now() - startTime,
        sessionId: raw.metrics.sessionId,
        messages: raw.messages,
        ...tokenMetrics(raw),
      });
    }

    return createHarnessResult({
      result: raw.result,
      isError: true,
      errorMessage: 'Schema validation failed after parse and cosmetic repair attempts.',
      failureType: 'schema',
      returnCode: raw.returnCode,
      costUsd: raw.metrics.totalCostUsd,
      numTurns: raw.metrics.numTurns,
      durationMs: Date.now() - startTime,
      sessionId: raw.metrics.sessionId,
      messages: raw.messages,
      ...tokenMetrics(raw),
    });
  }

  private async buildProvider(providerName: string, options: RunnerOptions): Promise<HarnessProvider> {
    const { provider: _, ...rest } = options;
    return buildProvider({ provider: providerName as NonNullable<HarnessConfig['provider']>, ...rest });
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
