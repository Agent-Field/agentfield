import type { HarnessProvider } from './base.js';
import type { HarnessConfig } from '../types.js';
import { SUPPORTED_PROVIDER_NAMES } from '../availability.js';

export const SUPPORTED_PROVIDERS: ReadonlySet<string> = new Set(SUPPORTED_PROVIDER_NAMES);
export const DEFAULT_HARNESS_PROVIDER = 'codeaf';
export const HARNESS_PROVIDER_ENV_VAR = 'AGENTFIELD_HARNESS_PROVIDER';

/**
 * Applies harness provider precedence: an explicit name wins, then
 * AGENTFIELD_HARNESS_PROVIDER, then DEFAULT_HARNESS_PROVIDER ("codeaf").
 * Blank / whitespace-only values are treated as unset.
 */
export function resolveProviderName(explicit?: string): string {
  const explicitName = explicit?.trim();
  if (explicitName) {
    return explicitName;
  }

  const envName = process.env[HARNESS_PROVIDER_ENV_VAR]?.trim();
  return envName || DEFAULT_HARNESS_PROVIDER;
}

let legacyPinLogged = false;

export function resolveExecutionProvider(config: { provider?: string; aforgeBin?: string }): string {
  if (!config.provider?.trim() && !process.env[HARNESS_PROVIDER_ENV_VAR]?.trim()
    && (config.aforgeBin !== undefined || process.env.AFORGE_BIN?.trim())) {
    if (!legacyPinLogged) {
      legacyPinLogged = true;
      console.info('Legacy aforge pin in effect; unset it or select provider "codeaf" to use CodeAF');
    }
    return 'aforge';
  }
  return resolveProviderName(config.provider);
}

export async function buildProvider(config: HarnessConfig): Promise<HarnessProvider> {
  const provider = resolveExecutionProvider(config);
  if (!SUPPORTED_PROVIDERS.has(provider)) {
    throw new Error(
      `Unknown harness provider: "${provider}". Supported: ${[...SUPPORTED_PROVIDERS].sort().join(', ')}`
    );
  }
  if (provider === 'aforge' && (config.aforgeBin !== undefined || process.env.AFORGE_BIN?.trim())) {
    const { AforgeProvider } = await import('./aforge.js');
    return new AforgeProvider(config.aforgeBin);
  }
  if (provider === 'codeaf' || provider === 'aforge') {
    const { CodeafProvider } = await import('./codeaf.js');
    return new CodeafProvider(config.codeafBin, provider === 'aforge');
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
