import type { HarnessProvider } from './base.js';
import { resolveRoot } from './base.js';
import type { RawResult } from '../types.js';
import { createMetrics, createRawResult } from '../types.js';
import { runCli } from '../cli.js';
import { resolveModelAndVariant } from '../modelVariant.js';

const REASONING_VARIANTS = new Set(['off', 'low', 'medium', 'high']);
const DEFAULT_TIMEOUT_SECONDS = 1800;
const LANDING_WINDOW_SECONDS = 5;
const DEFAULT_MAX_CONCURRENT = 8;
const ANSI_PATTERN = /\x1B\[[0-?]*[ -/]*[@-~]/g;

// From aforge's DefaultModel; keep in step with aforge's built-in default.
export const AFORGE_DEFAULT_MODEL = '~deepseek/deepseek-v4-flash-latest';

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
  // Both canonical `do` and `exec` print one JSON object. Parse that shape
  // before the wrapper-compatible line scan.
  try {
    const value: unknown = JSON.parse(stdout.trim());
    if (typeof value === 'object' && value !== null && !Array.isArray(value)
      && ('deliverable' in value || 'text' in value)) {
      return value as Record<string, unknown>;
    }
  } catch {
    // Fall through to the wrapper-compatible line scan.
  }

  const lines = stdout.split('\n').map((line) => line.trim()).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const value: unknown = JSON.parse(lines[index]);
      if (typeof value === 'object' && value !== null && !Array.isArray(value)
        && ('deliverable' in value || 'text' in value)) {
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

/**
 * Aforge CLI provider. `exec` is the default direct one-shot entry point; set
 * `AGENTFIELD_AFORGE_COMMAND=do` to opt into Aforge's routed workflow.
 */
export class AforgeProvider implements HarnessProvider {
  private readonly bin: string;

  public constructor(bin = 'aforge') {
    this.bin = bin === 'aforge' ? (process.env.AFORGE_BIN?.trim() || bin) : bin;
  }

  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    return aforgeSemaphore.use(() => this.executeImpl(prompt, options));
  }

  private async executeImpl(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const root = resolveRoot(options) ?? '.';
    const outerTimeout = timeoutSeconds();
    const command = (process.env.AGENTFIELD_AFORGE_COMMAND ?? 'exec').trim().toLowerCase();
    if (command !== 'do' && command !== 'exec') {
      return createRawResult({
        isError: true,
        errorMessage: `AGENTFIELD_AFORGE_COMMAND must be 'do' or 'exec', got ${JSON.stringify(command)}`,
        failureType: 'crash',
        metrics: createMetrics(),
      });
    }
    const systemPrompt = options.systemPrompt ?? options.system_prompt;
    const cmd = command === 'exec'
      ? [
          this.bin,
          'exec',
          '--json',
          '-w',
          root,
          '--timeout',
          String(innerTimeout(outerTimeout)),
        ]
      : [
          this.bin,
          'do',
          '--json',
          '--yes-spend',
          '-w',
          root,
          '--timeout',
          String(innerTimeout(outerTimeout)),
        ];
    // --turns exists only on exec, not do. Aforge's --budget is a token budget,
    // not a USD cap, so maxBudgetUsd has no honest mapping.
    if (command === 'exec' && typeof options.maxTurns === 'number'
      && Number.isFinite(options.maxTurns) && options.maxTurns > 0) {
      cmd.push('--turns', String(Math.trunc(options.maxTurns)));
    }
    if (command === 'exec') {
      cmd.push('--context-fill', '60', '--completion-reserve', '65536');
    }
    if (command === 'exec' && typeof systemPrompt === 'string' && systemPrompt.trim()) {
      cmd.push('--system', systemPrompt.trim());
    }

    const { model, variant } = resolveModelAndVariant(options);
    const env: Record<string, string> = command === 'exec' ? { AFORGE_MODELS: '' } : {};
    if (model) {
      const slug = stripOpenRouterPrefix(model);
      env.AFORGE_MODEL = slug;
      if (command === 'exec') {
        cmd.push('--model', slug, '--plan-model', slug);
      }
    }
    if (variant) {
      const normalized = variant.trim().toLowerCase();
      if (REASONING_VARIANTS.has(normalized)) {
        env.AFORGE_EXEC_REASONING = normalized;
      }
    }
    Object.assign(env, stringOptions(options.env));

    const effectiveModel = model || env.AFORGE_MODEL || AFORGE_DEFAULT_MODEL;

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, {
        env,
        cwd: undefined,
        timeout: outerTimeout * 1000,
        idleSeconds: 0,
        inputText: command === 'exec' ? prompt : taskInput(prompt, systemPrompt),
      });
      const envelope = parseEnvelope(stdout);
      const outputValue = command === 'exec' ? envelope?.text : envelope?.deliverable;
      const resultText = typeof outputValue === 'string' && outputValue.trim()
        ? outputValue.trim()
        : undefined;
      const blockedOn = typeof envelope?.blocked_on === 'string' ? envelope.blocked_on.trim() : '';
      const stop = typeof envelope?.stop === 'string' ? envelope.stop.trim() : '';
      const usage = typeof envelope?.usage === 'object' && envelope.usage !== null && !Array.isArray(envelope.usage)
        ? envelope.usage as Record<string, unknown>
        : {};

      const isError = command === 'exec'
        ? exitCode < 0 || resultText === undefined || ![0, 2, 3].includes(exitCode)
        : exitCode !== 0 || resultText === undefined || blockedOn !== '';
      const inputTokens = Math.trunc(numeric(usage.prompt_tokens) ?? 0);
      const outputTokens = Math.trunc(numeric(usage.completion_tokens) ?? 0);
      const cacheReadTokens = Math.trunc(numeric(usage.cached_tokens) ?? 0);
      const calls = Math.trunc(numeric(command === 'exec' ? envelope?.turns : usage.calls) ?? 0);
      const nativeSpend = numeric(envelope?.spend);
      const legacyCost = numeric(usage.cost);
      const providerCost = nativeSpend !== undefined && nativeSpend > 0 ? nativeSpend : legacyCost;

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
          model: effectiveModel,
        }),
        isError,
        errorMessage: isError ? crashMessage(exitCode, blockedOn || stop, resultText, stderr) : undefined,
        failureType: isError
          ? ((command === 'do' && exitCode === 2) || (command === 'exec' && exitCode === 4)
              ? 'timeout'
              : 'crash')
          : 'none',
        returnCode: exitCode,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('ENOENT')) {
        return createRawResult({
          isError: true,
          errorMessage: `AForge binary not found at '${this.bin}'. Install it with \`af aforge ensure\`, or set AFORGE_BIN to its path.`,
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
