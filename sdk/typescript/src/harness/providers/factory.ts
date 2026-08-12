import type { HarnessProvider } from './base.js';
import type { HarnessConfig } from '../types.js';

export const SUPPORTED_PROVIDERS = new Set(['claude-code', 'codex', 'gemini', 'omp', 'opencode', 'pi']);
export const DEFAULT_HARNESS_PROVIDER = 'omp' as const;

export async function buildProvider(config: HarnessConfig): Promise<HarnessProvider> {
  const provider = config.provider ?? DEFAULT_HARNESS_PROVIDER;
  if (!SUPPORTED_PROVIDERS.has(provider)) {
    throw new Error(
      `Unknown harness provider: "${provider}". Supported: ${[...SUPPORTED_PROVIDERS].sort().join(', ')}`
    );
  }
  if (provider === 'claude-code') {
    const { ClaudeCodeProvider } = await import('./claude.js');
    return new ClaudeCodeProvider();
  }
  if (provider === 'codex') {
    const { CodexProvider } = await import('./codex.js');
    return new CodexProvider(config.codexBin ?? 'codex');
  }
  if (provider === 'gemini') {
    const { GeminiProvider } = await import('./gemini.js');
    return new GeminiProvider(config.geminiBin ?? 'gemini');
  }
  if (provider === 'opencode') {
    const { OpenCodeProvider } = await import('./opencode.js');
    return new OpenCodeProvider(config.opencodeBin ?? 'opencode');
  }
  if (provider === 'pi') {
    const { PiProvider } = await import('./pi.js');
    return new PiProvider(config.piBin ?? 'pi');
  }
  if (provider === 'omp') {
    const { OMPProvider } = await import('./pi.js');
    return new OMPProvider(config.ompBin ?? 'omp');
  }
  throw new Error(`Provider "${provider}" is not yet implemented.`);
}
