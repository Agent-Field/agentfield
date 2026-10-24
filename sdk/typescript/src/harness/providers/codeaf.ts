import path from 'node:path';
import { performance } from 'node:perf_hooks';
import type { HarnessProvider } from './base.js';
import { resolveRoot } from './base.js';
import { AforgeProvider } from './aforge.js';
import type { FailureType, RawResult } from '../types.js';
import { createMetrics, createRawResult } from '../types.js';
import { markProviderCommand, runCli } from '../cli.js';
import { findExecutable, HarnessProviderUnavailable, providerUnavailable } from '../availability.js';
import { agentfieldHome, codeafStateHome, ensureCodeafInstalled, findManagedCodeaf, isTruthy } from '../codeafInstall.js';
import { resolveModelAndVariant } from '../modelVariant.js';

export const CODEAF_DEFAULT_MODEL = '~deepseek/deepseek-v4-flash-latest';
const REASONING_VARIANTS = new Set(['off', 'low', 'medium', 'high']);
const ANSI_PATTERN = /\x1B\[[0-?]*[ -/]*[@-~]/g;

export function resolveCodeafMaxConcurrent(): number {
  const value = process.env.CODEAF_MAX_CONCURRENT ?? process.env.AFORGE_MAX_CONCURRENT;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 8;
}

class Semaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  public constructor(private readonly limit: number) {}

  public async use<T>(operation: () => Promise<T>): Promise<T> {
    if (this.active >= this.limit) {
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    } else {
      this.active += 1;
    }
    try {
      return await operation();
    } finally {
      const next = this.waiters.shift();
      if (next) {
        next();
      } else {
        this.active -= 1;
      }
    }
  }
}

const codeafSemaphore = new Semaphore(resolveCodeafMaxConcurrent());

interface ResolvedBinary {
  provider: 'codeaf' | 'aforge';
  binary: string;
}

let resolution: Promise<ResolvedBinary> | undefined;
let downloadError: string | undefined;
let aliasLogged = false;
let invalidCommandLogged = false;

async function resolveAutomaticBinary(): Promise<ResolvedBinary> {
  const managed = findManagedCodeaf();
  if (managed) {
    return { provider: 'codeaf', binary: managed };
  }
  try {
    const installed = await ensureCodeafInstalled();
    if (installed) {
      return { provider: 'codeaf', binary: installed };
    }
  } catch (error) {
    downloadError = error instanceof Error ? error.message : String(error);
  }
  const onPath = findExecutable('codeaf');
  if (onPath) {
    return { provider: 'codeaf', binary: onPath };
  }
  const legacyName = process.platform === 'win32' ? 'aforge.exe' : 'aforge';
  const legacy = findExecutable(path.join(agentfieldHome(), 'bin', legacyName)) ?? findExecutable('aforge');
  if (legacy) {
    console.warn(`CodeAF is not installed; falling back to the legacy aforge binary at ${legacy}. Run \`af codeaf ensure\`.`);
    return { provider: 'aforge', binary: legacy };
  }
  const unavailable = providerUnavailable('codeaf', 'codeaf');
  unavailable.message += ` (or set CODEAF_BIN)${downloadError ? `. Download failed: ${downloadError}` : ''}`;
  throw unavailable;
}

function resolveBinary(explicit?: string): Promise<ResolvedBinary> {
  const override = explicit?.trim() || process.env.CODEAF_BIN?.trim();
  if (override) {
    return Promise.resolve({ provider: 'codeaf', binary: override });
  }
  resolution ??= resolveAutomaticBinary().catch((error: unknown) => {
    resolution = undefined;
    throw error;
  });
  return resolution;
}

async function withinDeadline<T>(operation: Promise<T>, deadline: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), Math.max(0, deadline - performance.now()));
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function parseEnvelope(stdout: string): Record<string, unknown> | undefined {
  const parse = (text: string): Record<string, unknown> | undefined => {
    try {
      const value: unknown = JSON.parse(text);
      if (typeof value === 'object' && value !== null && !Array.isArray(value)
        && 'ok' in value && 'stop' in value) {
        return value as Record<string, unknown>;
      }
    } catch {
      // Wrappers may print diagnostics before the canonical envelope.
    }
    return undefined;
  };
  const whole = parse(stdout.trim());
  if (whole) {
    return whole;
  }
  const lines = stdout.split('\n');
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const envelope = parse(lines[index].trim());
    if (envelope) {
      return envelope;
    }
  }
  return undefined;
}

function numeric(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function failure(envelope: Record<string, unknown> | undefined, exitCode: number, stdout: string): FailureType {
  if (exitCode < 0) return 'crash';
  if (!envelope) return exitCode === 0 && !stdout.trim() ? 'no_output' : 'crash';
  const stop = text(envelope.stop);
  const answer = text(envelope.answer);
  if (stop === 'deadline') return 'timeout';
  if (stop === 'done') return envelope.ok === true ? 'none' : 'crash';
  if (stop === 'unchecked') return answer ? 'none' : 'crash';
  if (stop === 'budget' || stop === 'turn-cap') return answer ? 'none' : 'no_output';
  if (['question', 'incomplete', 'error', 'price'].includes(stop)) return 'crash';
  return exitCode === 0 && answer ? 'none' : 'crash';
}

function errorMessage(exitCode: number, envelope: Record<string, unknown> | undefined, stderr: string): string {
  const stop = text(envelope?.stop);
  const detail = text(envelope?.error) || text(envelope?.answer) || stderr.trim().replace(ANSI_PATTERN, '').slice(-1000);
  let message = `codeaf exit code ${exitCode} (stop=${stop}): ${detail.slice(0, 1000)}`;
  if (stop === 'question' && text(envelope?.blocked_on)) {
    message += `. blocked_on: ${text(envelope?.blocked_on).slice(0, 1000)}`;
  }
  return message;
}

function childEnvironment(options: Record<string, unknown>, variant: string | undefined): NodeJS.ProcessEnv {
  const overlay: Record<string, string> = {};
  if (typeof options.env === 'object' && options.env !== null && !Array.isArray(options.env)) {
    for (const [key, value] of Object.entries(options.env)) {
      if (typeof value === 'string') overlay[key] = value;
    }
  }
  const inherited = { ...process.env, ...overlay };
  const env: NodeJS.ProcessEnv = { CODEAF_MODELS: undefined, AFORGE_MODELS: undefined };
  if (inherited.CODEAF_HOME === undefined) {
    env.CODEAF_HOME = codeafStateHome(inherited);
  }
  if (inherited.CODEAF_NO_UPDATE_CHECK === undefined) env.CODEAF_NO_UPDATE_CHECK = '1';
  if (inherited.CODEAF_TELEMETRY === undefined && (isTruthy(inherited.CI) || isTruthy(inherited.DO_NOT_TRACK)
    || /^(0|false|no|off)$/i.test(inherited.AGENTFIELD_TELEMETRY_ENABLED?.trim() ?? ''))) {
    env.CODEAF_TELEMETRY = 'off';
  }
  const normalized = variant?.trim().toLowerCase();
  if (normalized && REASONING_VARIANTS.has(normalized)) env.CODEAF_EXEC_REASONING = normalized;
  return { ...env, ...overlay };
}

/** Canonical CodeAF headless adapter, with lazy managed installation and legacy fallback. */
export class CodeafProvider implements HarnessProvider {
  public constructor(private readonly bin?: string, private readonly alias = false) {}

  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const legacyPin = this.alias ? process.env.AFORGE_BIN?.trim() : undefined;
    if (legacyPin) {
      return { ...await new AforgeProvider(legacyPin).execute(prompt, options), provider: 'aforge' };
    }
    const started = performance.now();
    const parsedTimeout = Number.parseInt(process.env.AGENTFIELD_HARNESS_TIMEOUT_SECONDS ?? '', 10);
    const timeout = Number.isFinite(parsedTimeout) && parsedTimeout > 0 ? parsedTimeout : 1800;
    const deadline = started + timeout * 1000;
    let resolved: ResolvedBinary;
    try {
      resolved = await withinDeadline(resolveBinary(this.bin), deadline, 'Timed out while provisioning CodeAF');
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('Timed out while provisioning')) throw error;
      return createRawResult({ provider: 'codeaf', isError: true, failureType: 'timeout',
        errorMessage: error.message, metrics: createMetrics({ durationApiMs: performance.now() - started }) });
    }
    if (resolved.provider === 'aforge') {
      try {
        return { ...await withinDeadline(new AforgeProvider(resolved.binary, deadline).execute(prompt, options), deadline,
          'CodeAF fallback timed out'), provider: 'aforge' };
      } catch (error) {
        if (!(error instanceof Error) || !error.message.includes('timed out')) throw error;
        return createRawResult({ provider: 'aforge', isError: true, failureType: 'timeout', errorMessage: error.message });
      }
    }
    if (this.alias && !aliasLogged) {
      aliasLogged = true;
      console.info("provider 'aforge' now runs CodeAF; set AFORGE_BIN to pin the legacy binary");
    }
    try {
      return await withinDeadline(codeafSemaphore.use(() => this.executeImpl(resolved.binary, prompt, options, deadline)),
        deadline, 'CodeAF timed out');
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('timed out')) throw error;
      return createRawResult({ provider: 'codeaf', isError: true, failureType: 'timeout', errorMessage: error.message });
    }
  }

  private async executeImpl(binary: string, prompt: string, options: Record<string, unknown>, deadline: number): Promise<RawResult> {
    if (performance.now() >= deadline) {
      return createRawResult({ provider: 'codeaf', isError: true, failureType: 'timeout', errorMessage: 'CodeAF timed out' });
    }
    const root = resolveRoot(options) ?? '.';
    const outerTimeout = Math.max(0, deadline - performance.now());
    const requestedCommand = process.env.AGENTFIELD_CODEAF_COMMAND?.trim().toLowerCase()
      || process.env.AGENTFIELD_AFORGE_COMMAND?.trim().toLowerCase() || 'exec';
    const command = requestedCommand === 'do' ? 'do' : 'exec';
    if (!['exec', 'do'].includes(requestedCommand) && !invalidCommandLogged) {
      invalidCommandLogged = true;
      console.warn(`Invalid AGENTFIELD_CODEAF_COMMAND ${JSON.stringify(requestedCommand)}; using exec`);
    }
    const { model, variant } = resolveModelAndVariant(options);
    const overlay = typeof options.env === 'object' && options.env !== null ? options.env : {};
    const environment = { ...process.env, ...overlay } as NodeJS.ProcessEnv;
    const effectiveModel = (model || environment.CODEAF_MODEL?.trim() || environment.AFORGE_MODEL?.trim()
      || CODEAF_DEFAULT_MODEL).replace(/^openrouter\//, '');
    const systemPrompt = text(options.systemPrompt ?? options.system_prompt);
    const cmd = [binary, command, '--json', '--dir', root];
    if (command === 'do') cmd.push('--yes-spend');
    cmd.push('--model', effectiveModel);
    if (command === 'do') cmd.push('--plan-model', effectiveModel, '--check-model', effectiveModel);
    cmd.push('--timeout', `${Number(Math.max(1, outerTimeout / 1000 - 5).toFixed(3))}s`);
    if (command === 'exec') {
      if (typeof options.maxTurns === 'number' && Number.isFinite(options.maxTurns) && options.maxTurns > 0) {
        cmd.push('--max-turns', String(Math.trunc(options.maxTurns)));
      }
      if (systemPrompt) cmd.push('--system', systemPrompt);
    } else {
      cmd.push('-');
    }

    const startApi = Date.now();
    try {
      const env = childEnvironment(options, variant);
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw new Error('CodeAF timed out');
      const { stdout, stderr, exitCode } = await runCli(markProviderCommand('codeaf', cmd), {
        env,
        cwd: undefined,
        timeout: remaining,
        idleSeconds: 0,
        inputText: command === 'do' && systemPrompt ? `${systemPrompt}\n\nTask:\n${prompt}` : prompt,
      });
      const envelope = parseEnvelope(stdout);
      const failureType = failure(envelope, exitCode, stdout);
      const stop = text(envelope?.stop);
      if (failureType === 'none' && ['unchecked', 'budget', 'turn-cap'].includes(stop)) {
        console.warn(`CodeAF returned ${stop === 'unchecked' ? 'an unchecked answer' : `a partial answer (${stop})`}`);
      }
      const tokens = typeof envelope?.tokens === 'object' && envelope.tokens !== null && !Array.isArray(envelope.tokens)
        ? envelope.tokens as Record<string, unknown> : {};
      const inputTokens = Math.trunc(numeric(tokens.in) ?? 0);
      const outputTokens = Math.trunc(numeric(tokens.out) ?? 0);
      return createRawResult({
        provider: 'codeaf',
        result: failureType === 'none' ? text(envelope?.answer) || undefined : undefined,
        messages: envelope ? [envelope] : [],
        metrics: createMetrics({
          durationApiMs: Date.now() - startApi,
          numTurns: Math.trunc(numeric(envelope?.steps) ?? 0),
          totalCostUsd: numeric(envelope?.spend_usd),
          inputTokens,
          outputTokens,
          totalTokens: inputTokens + outputTokens,
          cacheReadTokens: 0,
          cacheCreationTokens: 0,
          model: text(envelope?.model) || effectiveModel,
          usage: tokens,
        }),
        isError: failureType !== 'none',
        errorMessage: failureType !== 'none' ? errorMessage(exitCode, envelope, stderr) : undefined,
        failureType,
        returnCode: exitCode,
      });
    } catch (error) {
      if (error instanceof HarnessProviderUnavailable) throw error;
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('ENOENT')) throw providerUnavailable('codeaf', binary);
      return createRawResult({
        provider: 'codeaf',
        isError: true,
        errorMessage: message,
        failureType: /timed out|deadline exceeded|no progress/i.test(message) ? 'timeout' : 'crash',
        metrics: createMetrics({ durationApiMs: Date.now() - startApi, model: effectiveModel }),
      });
    }
  }
}
