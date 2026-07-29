import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createRawResult, createMetrics } from '../types.js';
import { runCli, parseJsonl, extractFinalText } from '../cli.js';
import { resolveModelAndVariant } from '../modelVariant.js';

export class CodexProvider implements HarnessProvider {
  private readonly bin: string;

  constructor(binPath = 'codex') {
    this.bin = binPath;
  }

  async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const cmd = [this.bin, 'exec', '--json'];

    if (options.cwd) {
      cmd.push('-C', String(options.cwd));
    }
    if (options.permissionMode === 'auto') {
      cmd.push('--full-auto');
    }

    // Model via -m; reasoning effort has no dedicated flag — it's the
    // model_reasoning_effort config key. The effort comes from a "#variant"
    // suffix on the model (or an explicit options.variant), e.g.
    // "gpt-5.3-codex#high".
    const { model: modelValue, variant: variantValue } = resolveModelAndVariant(options);
    if (modelValue) {
      cmd.push('-m', modelValue);
    }
    if (variantValue) {
      cmd.push('-c', `model_reasoning_effort=${variantValue}`);
    }

    cmd.push(prompt);

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, {
        env: options.env as Record<string, string> | undefined,
        cwd: options.cwd as string | undefined,
      });

      const events = parseJsonl(stdout);
      const resultText = extractFinalText(events);

      let numTurns = 0;
      let sessionId = '';
      for (const event of events) {
        if (event.type === 'turn.completed') {
          numTurns += 1;
        }
        if (event.type === 'thread.started') {
          const threadId = event.thread_id;
          if (typeof threadId === 'string') {
            sessionId = threadId;
          }
        }
      }

      const isError = exitCode !== 0 && !resultText;

      return createRawResult({
        result: resultText,
        messages: events,
        metrics: createMetrics({ durationApiMs: Date.now() - startApi, numTurns, sessionId, model: modelValue }),
        isError,
        errorMessage: isError ? stderr.trim() : undefined,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('ENOENT')) {
        return createRawResult({
          isError: true,
          errorMessage: `Codex binary not found at '${this.bin}'. Install: https://github.com/openai/codex`,
          metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
        });
      }
      return createRawResult({
        isError: true,
        errorMessage: msg,
        metrics: createMetrics({ durationApiMs: Date.now() - startApi }),
      });
    }
  }
}
