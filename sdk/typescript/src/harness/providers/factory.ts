import type { HarnessProvider } from './base.js';
import type { HarnessConfig } from '../types.js';

export const SUPPORTED_PROVIDERS = new Set(['claude-code', 'codex', 'gemini', 'omp', 'opencode', 'pi']);

export async function buildProvider(config: HarnessConfig): Promise<HarnessProvider> {
  if (!SUPPORTED_PROVIDERS.has(config.provider)) {
    throw new Error(
      `Unknown harness provider: "${config.provider}". Supported: ${[...SUPPORTED_PROVIDERS].sort().join(', ')}`
    );
  }
  if (config.provider === 'claude-code') {
    const { ClaudeCodeProvider } = await import('./claude.js');
    return new ClaudeCodeProvider();
  }
  if (config.provider === 'codex') {
    const { CodexProvider } = await import('./codex.js');
    return new CodexProvider(config.codexBin ?? 'codex');
  }
  if (config.provider === 'gemini') {
    const { GeminiProvider } = await import('./gemini.js');
    return new GeminiProvider(config.geminiBin ?? 'gemini');
  }
  if (config.provider === 'opencode') {
    const { OpenCodeProvider } = await import('./opencode.js');
    return new OpenCodeProvider(config.opencodeBin ?? 'opencode');
  }
  if (config.provider === 'pi') {
    const { PiProvider } = await import('./pi.js');
    return new PiProvider(config.piBin ?? 'pi');
  }
  if (config.provider === 'omp') {
    const { OmpProvider } = await import('./pi.js');
    return new OmpProvider(config.ompBin ?? 'omp');
  }
  throw new Error(`Provider "${config.provider}" is not yet implemented.`);
}
