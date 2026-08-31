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

    expect(cli.runCli).toHaveBeenCalledWith(
      ['/usr/local/bin/opencode', 'run', '--dir', '/tmp/work', 'hello'],
      { env: { A: '1' } },
    );
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
      ['opencode', 'run', '-m', 'openai/gpt-5', 'hello'],
      { env: {} },
    );
    expect(result.isError).toBe(false);
  });

  it('maps a #variant model suffix to the --variant flag', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    const result = await provider.execute('hello', { model: 'openrouter/z-ai/glm-5.2#high' });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['opencode', 'run', '-m', 'openrouter/z-ai/glm-5.2', '--variant', 'high', 'hello'],
      // OpenRouter model: env also carries the attribution overlay (asserted
      // separately below, keyed off the base model).
      { env: expect.objectContaining({}) },
    );
    expect(result.isError).toBe(false);
    expect(result.metrics.model).toBe('openrouter/z-ai/glm-5.2');
  });

  it('lets an explicit variant option win over the model suffix', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'openai/gpt-5#low', variant: 'max' });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['opencode', 'run', '-m', 'openai/gpt-5', '--variant', 'max', 'hello'],
      { env: {} },
    );
  });

  it('passes no --variant flag for a bare model', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'deepseek/deepseek-v4-flash' });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['opencode', 'run', '-m', 'deepseek/deepseek-v4-flash', 'hello'],
      { env: {} },
    );
  });

  it('keys the OpenRouter overlay off the base model when a variant is present', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'openrouter/openai/gpt-4o#high' });

    const call = vi.mocked(cli.runCli).mock.calls[0];
    const env = call[1]?.env ?? {};
    const overlay = JSON.parse(env.OPENCODE_CONFIG_CONTENT);

    expect(Object.keys(overlay.provider.openrouter.models)).toEqual(['openai/gpt-4o']);
  });

  it('adds OpenCode header overlay for explicit OpenRouter model', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'openrouter/openai/gpt-4o' });

    const call = vi.mocked(cli.runCli).mock.calls[0];
    const env = call[1]?.env ?? {};
    const overlay = JSON.parse(env.OPENCODE_CONFIG_CONTENT);

    expect(overlay.provider.openrouter.models['openai/gpt-4o'].headers).toEqual({
      'HTTP-Referer': 'https://agentfield.ai',
      'X-OpenRouter-Title': 'AgentField AI',
      'X-Title': 'AgentField AI',
    });
  });

  it('does not add OpenCode overlay for non-OpenRouter model', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'openai/gpt-5' });

    expect(vi.mocked(cli.runCli).mock.calls[0][1]?.env).toEqual({});
  });
});

describe('provider factory', () => {
  it('routes opencode to OpenCodeProvider and passes opencodeBin', async () => {
    const provider = await buildProvider({ provider: 'opencode', opencodeBin: '/opt/opencode' });

    expect(provider).toBeInstanceOf(OpenCodeProvider);
  });
});
