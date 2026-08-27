import type { HarnessProvider } from './base.js';
import { resolveRoot } from './base.js';
import type { RawResult } from '../types.js';
import { createMetrics, createRawResult } from '../types.js';
import { parseJsonl, runCli } from '../cli.js';
import { resolveModelAndVariant } from '../modelVariant.js';

type PiFlavor = 'pi' | 'omp';

const READ_ONLY_TOOLS = new Set(['read', 'grep', 'find', 'glob', 'ls', 'lsp']);
const ANSI_PATTERN = /\x1B\[[0-?]*[ -/]*[@-~]/g;

function normalizeTools(tools: unknown[], flavor: PiFlavor): string[] {
  const normalized: string[] = [];
  for (const tool of tools) {
    let name = String(tool).trim().toLowerCase();
    if (!name) {
      continue;
    }
    if (name === 'glob') {
      name = flavor === 'omp' ? 'glob' : 'find';
    }
    if (!normalized.includes(name)) {
      normalized.push(name);
    }
  }
  return normalized;
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function textContent(message: Record<string, unknown>): string | undefined {
  if (typeof message.content === 'string') {
    return message.content || undefined;
  }
  if (!Array.isArray(message.content)) {
    return undefined;
  }
  const text = message.content
    .filter((part): part is Record<string, unknown> => typeof part === 'object' && part !== null)
    .filter((part) => part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text as string)
    .join('');
  return text || undefined;
}

function parsePiEvents(events: Array<Record<string, unknown>>, configuredModel?: string) {
  let result: string | undefined;
  let sessionId = '';
  let numTurns = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let cacheCreationTokens = 0;
  let totalCostUsd: number | undefined;
  let reportedModel: string | undefined;
  let providerError: string | undefined;

  for (const event of events) {
    if (event.type === 'session' && typeof event.id === 'string') {
      sessionId = event.id;
    }
    if (event.type === 'turn_end') {
      numTurns += 1;
    }
    if (event.type !== 'message_end' || typeof event.message !== 'object' || event.message === null) {
      continue;
    }
    const message = event.message as Record<string, unknown>;
    if (message.role !== 'assistant') {
      continue;
    }

    result = textContent(message) ?? result;
    if (typeof message.model === 'string') {
      reportedModel = message.model;
    }

    if (typeof message.usage === 'object' && message.usage !== null) {
      const usage = message.usage as Record<string, unknown>;
      inputTokens += numberValue(usage.input);
      outputTokens += numberValue(usage.output);
      cacheReadTokens += numberValue(usage.cacheRead);
      cacheCreationTokens += numberValue(usage.cacheWrite);
      if (typeof usage.cost === 'object' && usage.cost !== null) {
        const cost = (usage.cost as Record<string, unknown>).total;
        if (typeof cost === 'number' && Number.isFinite(cost)) {
          totalCostUsd = (totalCostUsd ?? 0) + cost;
        }
      }
    }

    if (message.stopReason === 'error' || message.stopReason === 'aborted') {
      providerError = String(
        message.errorMessage ?? message.error ?? `Pi stopped with reason ${String(message.stopReason)}.`
      );
    } else {
      providerError = undefined;
    }
  }

  if (numTurns === 0 && result) {
    numTurns = 1;
  }

  return {
    result,
    providerError,
    metrics: createMetrics({
      numTurns,
      totalCostUsd,
      sessionId,
      inputTokens,
      outputTokens,
      cacheReadTokens,
      cacheCreationTokens,
      totalTokens: inputTokens + outputTokens,
      model: configuredModel ?? reportedModel,
    }),
  };
}

class PiFamilyProvider implements HarnessProvider {
  public constructor(
    private readonly flavor: PiFlavor,
    private readonly bin: string,
  ) {}

  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const cmd = [this.bin, '--print', '--mode', 'json'];
    const root = resolveRoot(options);

    if (this.flavor === 'omp' && root) {
      cmd.push('--cwd', root);
    }

    const { model, variant } = resolveModelAndVariant(options);
    if (model) {
      cmd.push('--model', model);
    }
    if (variant) {
      cmd.push('--thinking', variant);
    }

    if (typeof options.systemPrompt === 'string' && options.systemPrompt.trim()) {
      cmd.push('--system-prompt', options.systemPrompt.trim());
    }

    if (typeof options.resumeSessionId === 'string' && options.resumeSessionId) {
      cmd.push(this.flavor === 'omp' ? '--resume' : '--session', options.resumeSessionId);
    }

    // --tools is the enforced, vendor-documented read-only allowlist. Pi has no
    // approval flag (unknown options fail); OMP read-only tiers are auto-approved
    // even under always-ask.
    if (options.permissionMode === 'auto') {
      if (this.flavor === 'omp') {
        cmd.push('--auto-approve');
      }
    }

    const explicitTools = Array.isArray(options.tools);
    let tools = explicitTools ? normalizeTools(options.tools as unknown[], this.flavor) : [];
    if (options.permissionMode === 'plan') {
      tools = tools.filter((tool) => READ_ONLY_TOOLS.has(tool));
      if (tools.length === 0) {
        tools = ['read', 'grep', this.flavor === 'omp' ? 'glob' : 'find'];
      }
    }
    if (explicitTools || options.permissionMode === 'plan') {
      if (tools.length > 0) {
        cmd.push('--tools', tools.join(','));
      } else {
        cmd.push('--no-tools');
      }
    }

    const env = { ...options.env as Record<string, string> | undefined };
    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, {
        env,
        cwd: root,
        inputText: prompt,
        timeout: typeof options.timeout === 'number' ? options.timeout * 1000 : undefined,
      });
      const events = parseJsonl(stdout);
      const parsed = parsePiEvents(events, model);
      parsed.metrics.durationApiMs = Date.now() - startApi;

      const cleanStderr = stderr.trim().replace(ANSI_PATTERN, '').slice(0, 1000);
      let errorMessage: string | undefined;
      let failureType: NonNullable<RawResult['failureType']>;
      if (exitCode < 0) {
        errorMessage = `Process killed by signal ${-exitCode}.`;
        failureType = 'crash';
      } else if (exitCode !== 0) {
        errorMessage = cleanStderr || parsed.providerError || `Process exited with code ${exitCode}.`;
        failureType = 'crash';
      } else if (parsed.providerError) {
        errorMessage = parsed.providerError;
        failureType = 'api_error';
      } else if (!parsed.result) {
        errorMessage = cleanStderr || `${this.flavor} exited successfully without an assistant response.`;
        failureType = 'no_output';
      } else {
        failureType = 'none';
      }

      return createRawResult({
        result: parsed.result,
        messages: events,
        metrics: parsed.metrics,
        isError: failureType !== 'none',
        errorMessage,
        failureType,
        returnCode: exitCode,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const binaryMissing = message.includes('ENOENT');
      return createRawResult({
        isError: true,
        errorMessage: binaryMissing
          ? `${this.flavor === 'omp' ? 'OMP' : 'Pi'} binary not found at '${this.bin}'. ${
              this.flavor === 'omp'
                ? 'Install: curl -fsSL https://omp.sh/install | sh'
                : 'Install: npm install -g --ignore-scripts @earendil-works/pi-coding-agent'
            }`
          : message,
        failureType: /timed out|deadline exceeded|no progress/i.test(message)
          ? 'timeout'
          : 'crash',
        metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
      });
    }
  }
}

export class PiProvider extends PiFamilyProvider {
  public constructor(binPath = 'pi') {
    super('pi', binPath);
  }
}

export class OMPProvider extends PiFamilyProvider {
  public constructor(binPath = 'omp') {
    super('omp', binPath);
  }
}
