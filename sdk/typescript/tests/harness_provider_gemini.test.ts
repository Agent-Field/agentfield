import { describe, it, expect, vi, afterEach } from 'vitest';

import { GeminiProvider } from '../src/harness/providers/gemini.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import * as cli from '../src/harness/cli.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('gemini provider', () => {
  it('constructs command and maps result', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'final text\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new GeminiProvider('/usr/local/bin/gemini');
    const result = await provider.execute('hello', {
      cwd: '/tmp/work',
      permissionMode: 'auto',
      env: { A: '1' },
    });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['/usr/local/bin/gemini', '-C', '/tmp/work', '--sandbox', '-p', 'hello'],
      { cwd: '/tmp/work', env: { A: '1' } }
    );
    expect(result.isError).toBe(false);
    expect(result.result).toBe('final text');
    expect(result.metrics.numTurns).toBe(1);
    expect(result.metrics.sessionId).toBe('');
    expect(result.messages).toEqual([]);
  });

  it('returns helpful message when binary is not found', async () => {
    vi.spyOn(cli, 'runCli').mockRejectedValue(new Error('spawn gemini ENOENT'));

    const provider = new GeminiProvider('gemini-missing');
    const result = await provider.execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toContain("Gemini binary not found at 'gemini-missing'");
  });

  it('returns stderr when non-zero exit has no result', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '',
      stderr: 'boom',
      exitCode: 2,
    });

    const provider = new GeminiProvider('gemini');
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

    const provider = new GeminiProvider();
    const result = await provider.execute('hello', { model: 'gemini-2.5-pro' });

    expect(cli.runCli).toHaveBeenCalledWith(['gemini', '-m', 'gemini-2.5-pro', '-p', 'hello'], {
      cwd: undefined,
      env: undefined,
    });
    expect(result.isError).toBe(false);
  });
});

describe('provider factory', () => {
  it('routes gemini to GeminiProvider and passes geminiBin', async () => {
    const provider = await buildProvider({ provider: 'gemini', geminiBin: '/opt/gemini' });

    expect(provider).toBeInstanceOf(GeminiProvider);
  });
});
