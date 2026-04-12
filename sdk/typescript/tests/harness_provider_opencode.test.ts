import { describe, it, expect, vi, afterEach } from 'vitest';

import { OpenCodeProvider } from '../src/harness/providers/opencode.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import * as cli from '../src/harness/cli.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('opencode provider', () => {
  it('constructs command and maps result', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'final text\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider('/usr/local/bin/opencode');
    const result = await provider.execute('hello', {
      cwd: '/tmp/work',
      env: { A: '1' },
    });

    expect(cli.runCli).toHaveBeenCalledWith(['/usr/local/bin/opencode', '-c', '/tmp/work', '-p', 'hello'], {
      env: { A: '1' },
    });
    expect(result.isError).toBe(false);
    expect(result.result).toBe('final text');
    expect(result.metrics.numTurns).toBe(1);
    expect(result.metrics.sessionId).toBe('');
    expect(result.messages).toEqual([]);
  });

  it('returns helpful message when binary is not found', async () => {
    vi.spyOn(cli, 'runCli').mockRejectedValue(new Error('spawn opencode ENOENT'));

    const provider = new OpenCodeProvider('opencode-missing');
    const result = await provider.execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toContain("OpenCode binary not found at 'opencode-missing'");
  });

  it('returns stderr when non-zero exit has no result', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '',
      stderr: 'boom',
      exitCode: 2,
    });

    const provider = new OpenCodeProvider('opencode');
    const result = await provider.execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.result).toBeUndefined();
    expect(result.errorMessage).toBe('boom');
  });

  it('passes model flag', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    const result = await provider.execute('hello', { model: 'openai/gpt-5' });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['opencode', '-p', 'hello'],
      {
        env: { MODEL: 'openai/gpt-5' },
      }
    );
    expect(result.isError).toBe(false);
  });
});

describe('provider factory', () => {
  it('routes opencode to OpenCodeProvider and passes opencodeBin', async () => {
    const provider = await buildProvider({ provider: 'opencode', opencodeBin: '/opt/opencode' });

    expect(provider).toBeInstanceOf(OpenCodeProvider);
  });
});
