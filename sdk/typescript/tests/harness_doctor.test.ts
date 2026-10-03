import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  HarnessProviderUnavailable,
  ensureCliAvailable,
  harnessDoctor,
} from '../src/harness/availability.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import type { HarnessConfig } from '../src/harness/types.js';

const { execFileMock, realExecFile } = vi.hoisted(() => ({
  execFileMock: vi.fn(),
  realExecFile: {} as { execFile?: typeof import('node:child_process').execFile },
}));

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>();
  realExecFile.execFile = actual.execFile;
  return { ...actual, execFile: execFileMock };
});

/**
 * Point the mock at the real node:child_process execFile so one test can
 * genuinely spawn processes without loosening the mock for the whole file.
 */
function passThroughToRealExecFile(): void {
  execFileMock.mockImplementation((...args: unknown[]) => {
    const actual = realExecFile.execFile as ((...cbArgs: unknown[]) => unknown) | undefined;
    if (!actual) {
      throw new Error('real node:child_process execFile was not captured');
    }
    return actual(...args);
  });
}

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

  it('passes options.env to the binary resolver', async () => {
    const env = { PATH: '/custom/doctor/path', OPENAI_API_KEY: 'configured' };
    const resolveBinary = vi.fn().mockReturnValue('/custom/doctor/path/codex');

    const [health] = await harnessDoctor(['codex'], {
      env,
      resolveBinary,
      versionProbe: async () => 'codex-cli 0.0.0-test',
    });

    expect(resolveBinary).toHaveBeenCalledWith('codex', env);
    expect(resolveBinary.mock.calls[0]?.[1]).toBe(env);
    expect(health).toMatchObject({
      binary: '/custom/doctor/path/codex',
      installed: true,
      auth: 'configured',
      usable: true,
    });
  });

  it('discovers binaries on the options.env PATH instead of process.env', async () => {
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'agentfield-doctor-'));
    const binaryPath = path.join(directory, process.platform === 'win32' ? 'codex.exe' : 'codex');
    await fs.promises.writeFile(binaryPath, '');
    if (process.platform !== 'win32') {
      await fs.promises.chmod(binaryPath, 0o755);
    }
    const empty = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'agentfield-doctor-empty-'));

    try {
      const [found] = await harnessDoctor(['codex'], {
        env: { PATH: directory },
        versionProbe: async () => 'codex-cli 0.0.0-test',
      });
      expect(found).toMatchObject({
        binary: path.resolve(binaryPath),
        installed: true,
        usable: true,
        issues: [],
      });

      const [missing] = await harnessDoctor(['codex'], {
        env: { PATH: empty },
        versionProbe: async () => 'codex-cli 0.0.0-test',
      });
      expect(missing).toMatchObject({
        binary: null,
        installed: false,
        usable: false,
        issues: ['binary_not_found'],
      });
    } finally {
      await fs.promises.rm(directory, { recursive: true, force: true });
      await fs.promises.rm(empty, { recursive: true, force: true });
    }
  });

  it('reads PATHEXT from options.env during Windows executable resolution', async () => {
    const platform = Object.getOwnPropertyDescriptor(process, 'platform');
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'agentfield-doctor-pathext-'));
    const batchPath = path.join(directory, 'codex.cmd');
    await fs.promises.writeFile(batchPath, '@echo off\r\n');

    try {
      const [matched] = await harnessDoctor(['codex'], {
        env: { PATH: directory, PATHEXT: '.CMD' },
        versionProbe: async () => 'codex-cli 0.0.0-test',
      });
      expect(matched).toMatchObject({
        binary: path.resolve(batchPath),
        installed: true,
        usable: true,
        issues: [],
      });

      const [missed] = await harnessDoctor(['codex'], {
        env: { PATH: directory, PATHEXT: '.EXE' },
        versionProbe: async () => 'codex-cli 0.0.0-test',
      });
      expect(missed).toMatchObject({
        binary: null,
        installed: false,
        usable: false,
        issues: ['binary_not_found'],
      });
    } finally {
      if (platform) {
        Object.defineProperty(process, 'platform', platform);
      }
      await fs.promises.rm(directory, { recursive: true, force: true });
    }
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

  it('keeps a Windows batch shim path out of the cmd.exe command line', async () => {
    // Locks the invocation contract only. The mock does not execute cmd.exe.
    const platform = Object.getOwnPropertyDescriptor(process, 'platform');
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    mockExecFileOutput('codex-cli 9.9.9\n');
    const shim = 'C:\\Program Files (x86)\\R&D ^%x%!\\npm\\codex.cmd';

    try {
      const [health] = await harnessDoctor(['codex'], {
        env: {},
        resolveBinary: () => shim,
      });

      expect(execFileMock).toHaveBeenCalledWith(
        process.env.ComSpec ?? 'cmd.exe',
        [
          '/d',
          '/v:off',
          '/s',
          '/c',
          '""%AGENTFIELD_PROBE_ARG_0%" "%AGENTFIELD_PROBE_ARG_1%""',
        ],
        expect.objectContaining({
          windowsHide: true,
          windowsVerbatimArguments: true,
          env: expect.objectContaining({
            AGENTFIELD_PROBE_ARG_0: shim,
            AGENTFIELD_PROBE_ARG_1: '--version',
          }),
        }),
        expect.any(Function)
      );
      expect(health).toMatchObject({ version: 'codex-cli 9.9.9', usable: true, issues: [] });
    } finally {
      if (platform) {
        Object.defineProperty(process, 'platform', platform);
      }
    }
  });

  // Real end-to-end case: spawns cmd.exe and the batch shim for real, so it
  // only runs where a Windows cmd.exe exists (e.g. the windows-latest CI job).
  it.runIf(process.platform === 'win32')(
    'probes a real batch shim under a spaced temp directory through cmd.exe',
    async () => {
      const marker = `af-batch-probe-marker-${process.pid}-${Date.now()}`;
      // The mkdtemp prefix contains a space, so the resolved shim path can only
      // survive cmd.exe if it reaches it through the fixed %AGENTFIELD_PROBE_ARG_*%
      // template as one properly quoted expansion.
      const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'agentfield-doctor win-'));
      const probePath = path.join(directory, 'probe.cmd');
      // %~f0 is the probe's own full path; %* is the version argument list.
      await fs.promises.writeFile(probePath, `@echo off\r\necho ${marker} %~f0 %*\r\n`);
      const batchPath = path.join(directory, 'codex.cmd');
      await fs.promises.writeFile(batchPath, `@echo off\r\ncall "${probePath}" %*\r\n`);

      passThroughToRealExecFile();

      try {
        const [health] = await harnessDoctor(['codex'], {
          env: { ...process.env, PATH: directory, PATHEXT: '.CMD' },
        });

        expect(health).toMatchObject({
          binary: path.resolve(batchPath),
          installed: true,
          usable: true,
          issues: [],
        });
        expect(health.version).toContain(marker);
        expect(health.version?.toLowerCase()).toContain(path.resolve(probePath).toLowerCase());
        expect(health.version).toContain('--version');
      } finally {
        execFileMock.mockReset();
        await fs.promises.rm(directory, { recursive: true, force: true });
      }
    }
  );

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
