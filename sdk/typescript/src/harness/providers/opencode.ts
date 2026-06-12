import type { HarnessProvider } from './base.js';
import type { RawResult } from '../types.js';
import { createRawResult, createMetrics } from '../types.js';
import { runCli } from '../cli.js';
import {
  isOpenRouterRequest,
  openRouterAttributionHeaders,
} from '../../ai/openrouterAttribution.js';

export class OpenCodeProvider implements HarnessProvider {
  private readonly bin: string;

  constructor(binPath = 'opencode') {
    this.bin = binPath;
  }

  async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    // opencode v1.4+ uses the `run` subcommand. Prior `-c <dir> -p <prompt>`
    // syntax is broken on v1.14: `-c` now means `--continue` (a boolean) and
    // there is no top-level `-p` flag, so opencode prints help to stdout and
    // exits 0 — the SDK then captures the help screen as the LLM response.
    // See agentfield#582.
    const cmd = [this.bin, 'run'];

    // Use --dir for project directory.
    if (options.cwd && typeof options.cwd === 'string') {
      cmd.push('--dir', options.cwd);
    } else if (options.project_dir && typeof options.project_dir === 'string') {
      cmd.push('--dir', options.project_dir);
    }

    const env: Record<string, string> = { ...(options.env as Record<string, string>) };

    // Pass model via -m flag on the run subcommand (not env var).
    if (options.model) {
      cmd.push('-m', String(options.model));
    }

    // Handle system prompt - prepend to user prompt since OpenCode
    // has no native --system-prompt flag.
    let effectivePrompt = prompt;
    if (options.system_prompt && typeof options.system_prompt === 'string' && options.system_prompt.trim()) {
      effectivePrompt = `SYSTEM INSTRUCTIONS:\n${options.system_prompt.trim()}\n\n---\n\nUSER REQUEST:\n${prompt}`;
    }

    // Prompt is the positional `message` arg to `opencode run`.
    cmd.push(effectivePrompt);

    const explicitModel = typeof options.model === 'string' ? options.model : undefined;
    if (
      explicitModel &&
      isOpenRouterRequest({ model: explicitModel }) &&
      !env.OPENCODE_CONFIG_CONTENT &&
      !process.env.OPENCODE_CONFIG_CONTENT
    ) {
      const modelSlug = explicitModel.slice('openrouter/'.length);
      const headers = openRouterAttributionHeaders({ env: { ...process.env, ...env } });
      if (modelSlug && Object.keys(headers).length > 0) {
        env.OPENCODE_CONFIG_CONTENT = JSON.stringify({
          provider: {
            openrouter: {
              models: {
                [modelSlug]: { headers },
              },
            },
          },
        });
      }
    }

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
