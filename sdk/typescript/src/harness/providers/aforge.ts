import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createMetrics, createRawResult } from '../types.js';
import { runCli } from '../cli.js';
import { resolveModelAndVariant } from '../modelVariant.js';

const REASONING_VARIANTS = new Set(['off', 'low', 'medium', 'high']);
const DEFAULT_TIMEOUT_SECONDS = 1800;
const LANDING_WINDOW_SECONDS = 5;
const DEFAULT_MAX_CONCURRENT = 8;
const ANSI_PATTERN = /\x1B\[[0-?]*[ -/]*[@-~]/g;

class Semaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  public constructor(private readonly limit: number) {}

  public async use<T>(operation: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await operation();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    if (this.active < this.limit) {
      this.active += 1;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.waiters.push(() => {
        this.active += 1;
        resolve();
      });
    });
  }

  private release(): void {
    this.active -= 1;
    this.waiters.shift()?.();
  }
}

function resolveMaxConcurrent(): number {
  const parsed = Number.parseInt(process.env.AFORGE_MAX_CONCURRENT ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_CONCURRENT;
}

const aforgeSemaphore = new Semaphore(resolveMaxConcurrent());

function stripOpenRouterPrefix(model: string): string {
  return model.startsWith('openrouter/') ? model.slice('openrouter/'.length) : model;
}

function parseEnvelope(stdout: string): Record<string, unknown> | undefined {
  const lines = stdout.split('\n').map((line) => line.trim()).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const value: unknown = JSON.parse(lines[index]);
      if (typeof value === 'object' && value !== null && !Array.isArray(value) && 'deliverable' in value) {
        return value as Record<string, unknown>;
      }
    } catch {
      // Tolerate stray stdout from a wrapper and keep looking for the envelope.
    }
  }
  return undefined;
}

function numeric(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function timeoutSeconds(): number {
  const parsed = Number.parseInt(process.env.AGENTFIELD_HARNESS_TIMEOUT_SECONDS ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_SECONDS;
}

function innerTimeout(outer: number): number {
  return outer > LANDING_WINDOW_SECONDS ? outer - LANDING_WINDOW_SECONDS : 1;
}

function taskInput(prompt: string, systemPrompt: unknown): string {
  return typeof systemPrompt === 'string' && systemPrompt.trim()
    ? `${systemPrompt.trim()}\n\nTask:\n${prompt}`
    : prompt;
}

function crashMessage(exitCode: number, blockedOn: string, deliverable: string | undefined, stderr: string): string {
  const cleanStderr = stderr.trim().replace(ANSI_PATTERN, '');
  const exitContext = `aforge exit code ${exitCode}`;
  let message = exitCode < 0 ? `Process killed by signal ${-exitCode}. ${exitContext}` : exitContext;
  if (cleanStderr) {
    message += `. stderr: ${cleanStderr.slice(0, 1000)}`;
  } else if (blockedOn) {
    message += `. blocked_on: ${blockedOn.slice(0, 1000)}`;
  } else if (deliverable) {
    message += `. partial: ${deliverable.slice(0, 1000)}`;
  }
  return message;
}

function stringOptions(value: unknown): Record<string, string> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return {};
  }
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === 'string') {
      result[key] = item;
    }
  }
  return result;
}

/** Aforge CLI provider using canonical `aforge do --json` one-shot mode. */
export class AforgeProvider implements HarnessProvider {
  private readonly bin: string;

  public constructor(bin = 'aforge') {
    this.bin = bin === 'aforge' ? (process.env.AFORGE_BIN?.trim() || bin) : bin;
  }

  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    return aforgeSemaphore.use(() => this.executeImpl(prompt, options));
  }

  private async executeImpl(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const projectDir = typeof options.projectDir === 'string'
      ? options.projectDir
      : typeof options.project_dir === 'string'
        ? options.project_dir
        : undefined;
    const cwd = typeof options.cwd === 'string' ? options.cwd : undefined;
    const root = projectDir ?? cwd ?? '.';
    const outerTimeout = timeoutSeconds();
    const cmd = [
      this.bin,
      'do',
      '--json',
      '--yes-spend',
      '-w',
      root,
      '--timeout',
      String(innerTimeout(outerTimeout)),
    ];

    const { model, variant } = resolveModelAndVariant(options);
    const env: Record<string, string> = {};
    if (model) {
      env.AFORGE_MODEL = stripOpenRouterPrefix(model);
    }
    if (variant) {
      const normalized = variant.trim().toLowerCase();
      if (REASONING_VARIANTS.has(normalized)) {
        env.AFORGE_EXEC_REASONING = normalized;
      }
    }
    Object.assign(env, stringOptions(options.env));

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, {
        env,
        cwd: undefined,
        timeout: outerTimeout * 1000,
        idleSeconds: 0,
        inputText: taskInput(prompt, options.systemPrompt ?? options.system_prompt),
      });
      const envelope = parseEnvelope(stdout);
      const resultText = typeof envelope?.deliverable === 'string' && envelope.deliverable.trim()
        ? envelope.deliverable.trim()
        : undefined;
      const blockedOn = typeof envelope?.blocked_on === 'string' ? envelope.blocked_on.trim() : '';
      const usage = typeof envelope?.usage === 'object' && envelope.usage !== null && !Array.isArray(envelope.usage)
        ? envelope.usage as Record<string, unknown>
        : {};

      const isError = exitCode !== 0 || resultText === undefined || blockedOn !== '';
      const inputTokens = Math.trunc(numeric(usage.prompt_tokens) ?? 0);
      const outputTokens = Math.trunc(numeric(usage.completion_tokens) ?? 0);
      const cacheReadTokens = Math.trunc(numeric(usage.cached_tokens) ?? 0);
      const calls = Math.trunc(numeric(usage.calls) ?? 0);
      const providerCost = numeric(usage.cost);

      return createRawResult({
        result: resultText,
        messages: envelope ? [envelope] : [],
        metrics: createMetrics({
          durationApiMs: Date.now() - startApi,
          numTurns: calls,
          totalCostUsd: providerCost !== undefined && providerCost > 0 ? providerCost : undefined,
          usage,
          sessionId: '',
          inputTokens,
          outputTokens,
          cacheReadTokens,
          cacheCreationTokens: 0,
          totalTokens: inputTokens + outputTokens,
          model,
        }),
        isError,
        errorMessage: isError ? crashMessage(exitCode, blockedOn, resultText, stderr) : undefined,
        failureType: isError ? (exitCode === 2 ? 'timeout' : 'crash') : 'none',
        returnCode: exitCode,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('ENOENT')) {
        return createRawResult({
          isError: true,
          errorMessage: `Aforge binary not found at '${this.bin}'. Build it from https://github.com/Agent-Field/aforge-v2`,
          failureType: 'crash',
          metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
        });
      }
      const timedOut = /timed out|deadline exceeded|no progress/i.test(message);
      return createRawResult({
        isError: true,
        errorMessage: message,
        failureType: timedOut ? 'timeout' : 'crash',
        metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
      });
    }
  }
}
