import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createRawResult, createMetrics } from '../types.js';
import { runCli } from '../cli.js';

export class GeminiProvider implements HarnessProvider {
  private readonly bin: string;

  constructor(binPath = 'gemini') {
    this.bin = binPath;
  }

  async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const cmd = [this.bin];

    if (options.cwd) {
      cmd.push('-C', String(options.cwd));
    }
    if (options.permissionMode === 'auto') {
      cmd.push('--sandbox');
    }
    if (options.model) {
      cmd.push('-m', String(options.model));
    }
    cmd.push('-p', prompt);

    const startApi = Date.now();
    try {
      const { stdout, stderr, exitCode } = await runCli(cmd, {
        env: options.env as Record<string, string> | undefined,
        cwd: options.cwd as string | undefined,
      });

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
          errorMessage: `Gemini binary not found at '${this.bin}'. Install: https://github.com/google-gemini/gemini-cli`,
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
