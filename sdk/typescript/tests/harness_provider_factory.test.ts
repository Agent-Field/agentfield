import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { AforgeProvider } from '../src/harness/providers/aforge.js';
import {
  buildProvider,
  HARNESS_PROVIDER_ENV_VAR,
  resolveProviderName,
} from '../src/harness/providers/factory.js';
import type { HarnessConfig } from '../src/harness/types.js';

let originalProviderEnv: string | undefined;

beforeEach(() => {
  originalProviderEnv = process.env[HARNESS_PROVIDER_ENV_VAR];
  delete process.env[HARNESS_PROVIDER_ENV_VAR];
});

afterEach(() => {
  if (originalProviderEnv === undefined) {
    delete process.env[HARNESS_PROVIDER_ENV_VAR];
  } else {
    process.env[HARNESS_PROVIDER_ENV_VAR] = originalProviderEnv;
  }
});

describe('harness provider factory', () => {
  it('defaults to aforge', () => {
    expect(resolveProviderName(undefined)).toBe('aforge');
  });

  it('honours the environment fallback', () => {
    process.env[HARNESS_PROVIDER_ENV_VAR] = 'codex';
    expect(resolveProviderName(undefined)).toBe('codex');
  });

  it('prefers an explicit provider over the environment', () => {
    process.env[HARNESS_PROVIDER_ENV_VAR] = 'codex';
    expect(resolveProviderName('gemini')).toBe('gemini');
  });

  it('treats blank explicit and environment values as unset', () => {
    expect(resolveProviderName('   ')).toBe('aforge');

    process.env[HARNESS_PROVIDER_ENV_VAR] = '  ';
    expect(resolveProviderName(undefined)).toBe('aforge');
  });

  it('builds the aforge provider when no provider is configured', async () => {
    await expect(buildProvider({})).resolves.toBeInstanceOf(AforgeProvider);
  });

  it('builds the additional pi and omp providers when named explicitly', async () => {
    const { PiProvider, OMPProvider } = await import('../src/harness/providers/pi.js');
    await expect(buildProvider({ provider: 'pi' })).resolves.toBeInstanceOf(PiProvider);
    await expect(buildProvider({ provider: 'omp' })).resolves.toBeInstanceOf(OMPProvider);
  });

  it('rejects genuinely unknown providers with the supported list', async () => {
    const config = { provider: 'nope' } as unknown as HarnessConfig;
    await expect(buildProvider(config)).rejects.toThrow(
      'Unknown harness provider: "nope". Supported: aforge, claude-code, codex, gemini, omp, opencode, pi'
    );
  });
});
