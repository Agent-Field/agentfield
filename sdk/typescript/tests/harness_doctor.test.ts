import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  HarnessProviderUnavailable,
  ensureCliAvailable,
  harnessDoctor,
} from '../src/harness/availability.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import type { HarnessConfig } from '../src/harness/types.js';

const { execFileMock } = vi.hoisted(() => ({
  execFileMock: vi.fn(),
}));

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>();
  return { ...actual, execFile: execFileMock };
});

function mockExecFileOutput(stdout: string): void {
  execFileMock.mockImplementation((...args: unknown[]) => {
    const callback = args[args.length - 1] as (
      error: Error | null,
      stdout: string,
      stderr: string
    ) => void;
    callback(null, stdout, '');
  });
}

describe('harness provider availability', () => {
  beforeEach(() => {
    execFileMock.mockReset();
  });

  it('reports an installed CLI provider with version and auth details', async () => {
    const versionProbe = vi.fn().mockResolvedValue('codex-cli 1.2.3\nextra');

    const [health] = await harnessDoctor(['codex'], {
      env: { OPENAI_API_KEY: 'configured' },
      resolveBinary: (binary) => binary === 'codex' ? '/usr/local/bin/codex' : undefined,
      versionProbe,
    });

    expect(versionProbe).toHaveBeenCalledWith(['/usr/local/bin/codex', '--version']);
    expect(health).toEqual({
      provider: 'codex',
      binary: '/usr/local/bin/codex',
      installed: true,
      version: 'codex-cli 1.2.3',
      auth: 'configured',
      usable: true,
      installCommand: 'npm install -g @openai/codex',
      authEnvVars: ['OPENAI_API_KEY'],
      issues: [],
    });
  });

  it('reports a missing binary without trying a version probe', async () => {
    const versionProbe = vi.fn();

    const [health] = await harnessDoctor(['opencode'], {
      env: {},
      resolveBinary: () => undefined,
      versionProbe,
    });

    expect(versionProbe).not.toHaveBeenCalled();
    expect(health).toMatchObject({
      provider: 'opencode',
      binary: null,
      installed: false,
      version: null,
      auth: 'unknown',
      usable: false,
      issues: ['binary_not_found'],
    });
  });

  it('marks a broken version probe as unusable', async () => {
    const [health] = await harnessDoctor(['gemini'], {
      env: {},
      resolveBinary: () => '/usr/local/bin/gemini',
      versionProbe: async () => { throw new Error('broken install'); },
    });

    expect(health).toMatchObject({
      installed: true,
      version: null,
      usable: false,
      issues: ['version_probe_failed'],
    });
  });

  it('runs the default version probe directly against the resolved binary path', async () => {
    mockExecFileOutput('opencode 1.0.0\n');

    const [health] = await harnessDoctor(['opencode'], {
      env: {},
      resolveBinary: () => '/usr/local/bin/opencode',
    });

    expect(execFileMock).toHaveBeenCalledWith(
      '/usr/local/bin/opencode',
      ['--version'],
      expect.objectContaining({ windowsHide: true, windowsVerbatimArguments: false }),
      expect.any(Function)
    );
    expect(health).toMatchObject({ version: 'opencode 1.0.0', usable: true, issues: [] });
  });

  it('routes Windows batch shims through cmd.exe for the default version probe', async () => {
    const platform = Object.getOwnPropertyDescriptor(process, 'platform');
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    mockExecFileOutput('codex-cli 9.9.9\n');

    try {
      const [health] = await harnessDoctor(['codex'], {
        env: {},
        resolveBinary: () => 'C:\\Program Files\\npm\\codex.cmd',
      });

      expect(execFileMock).toHaveBeenCalledWith(
        process.env.ComSpec ?? 'cmd.exe',
        ['/d', '/s', '/c', '""C:\\Program Files\\npm\\codex.cmd" "--version""'],
        expect.objectContaining({ windowsHide: true, windowsVerbatimArguments: true }),
        expect.any(Function)
      );
      expect(health).toMatchObject({ version: 'codex-cli 9.9.9', usable: true, issues: [] });
    } finally {
      if (platform) {
        Object.defineProperty(process, 'platform', platform);
      }
    }
  });

  it('checks the optional Claude wrapper without launching a provider run', async () => {
    const [health] = await harnessDoctor(['claude-code'], {
      env: { ANTHROPIC_API_KEY: 'configured' },
      wrapperProbe: async () => false,
    });

    expect(health).toEqual({
      provider: 'claude-code',
      binary: null,
      installed: false,
      version: null,
      auth: 'configured',
      usable: false,
      installCommand: 'npm install @anthropic-ai/claude-agent-sdk',
      authEnvVars: ['ANTHROPIC_API_KEY'],
      issues: ['wrapper_not_installed'],
    });
  });

  it('rejects unknown providers with the supported list', async () => {
    await expect(harnessDoctor(['not-a-provider'])).rejects.toThrow(
      'Unknown harness provider: "not-a-provider". Supported providers: aforge, claude-code, codex, gemini, omp, opencode, pi'
    );
  });

  it('raises a structured actionable error for a missing CLI', () => {
    expect(() => ensureCliAvailable('codex', 'codex-missing', () => undefined)).toThrow(
      HarnessProviderUnavailable
    );

    try {
      ensureCliAvailable('codex', 'codex-missing', () => undefined);
    } catch (error) {
      expect(error).toMatchObject({
        name: 'HarnessProviderUnavailable',
        provider: 'codex',
        binary: 'codex-missing',
        installCommand: 'npm install -g @openai/codex',
        missingAuthEnv: [],
      });
      expect(String(error)).toContain("binary 'codex-missing' was not found");
    }
  });

  it.each([
    ['aforge', 'aforgeBin'],
    ['codex', 'codexBin'],
    ['gemini', 'geminiBin'],
    ['opencode', 'opencodeBin'],
    ['pi', 'piBin'],
    ['omp', 'ompBin'],
  ] as const)('propagates the typed preflight error through the %s provider', async (provider, binKey) => {
    const config = {
      provider,
      [binKey]: `agentfield-definitely-missing-${provider}`,
    } as HarnessConfig;
    const instance = await buildProvider(config);

    await expect(instance.execute('do not launch', {})).rejects.toMatchObject({
      name: 'HarnessProviderUnavailable',
      provider,
      binary: `agentfield-definitely-missing-${provider}`,
    });
  });
});
