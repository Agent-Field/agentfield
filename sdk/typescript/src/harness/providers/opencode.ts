import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createRawResult, createMetrics } from '../types.js';
import { runCli } from '../cli.js';

export class OpenCodeProvider implements HarnessProvider {
  private readonly bin: string;

  constructor(binPath = 'opencode') {
    this.bin = binPath;
  }

  async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const cmd = [this.bin];

    // Use -c for cwd (project directory)
    if (options.cwd && typeof options.cwd === 'string') {
      cmd.push('-c', options.cwd);
    } else if (options.project_dir && typeof options.project_dir === 'string') {
      cmd.push('-c', options.project_dir);
    }

    // Model is set via environment variable, not CLI flag
    const env: Record<string, string> = { ...(options.env as Record<string, string>) };
    if (options.model) {
      env['MODEL'] = String(options.model);
    }

    // Handle system prompt - prepend to user prompt since OpenCode
    // has no native --system-prompt flag
    let effectivePrompt = prompt;
    if (options.system_prompt && typeof options.system_prompt === 'string' && options.system_prompt.trim()) {
      effectivePrompt = `SYSTEM INSTRUCTIONS:\n${options.system_prompt.trim()}\n\n---\n\nUSER REQUEST:\n${prompt}`;
    }

    // Use -p for single prompt mode (non-interactive)
    cmd.push('-p', effectivePrompt);

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, { env });

      const resultText = stdout.trim() || undefined;
      const isError = exitCode !== 0 && !resultText;

      return createRawResult({
        result: resultText,
        messages: [],
        metrics: createMetrics({
          durationApiMs: Date.now() - startApi,
          numTurns: resultText ? 1 : 0,
          sessionId: '',
        }),
        isError,
        errorMessage: isError ? stderr.trim() : undefined,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('ENOENT')) {
        return createRawResult({
          isError: true,
          errorMessage: `OpenCode binary not found at '${this.bin}'. Install: https://github.com/opencode-ai/opencode`,
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
