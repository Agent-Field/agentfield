import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AFORGE_DEFAULT_MODEL, AforgeProvider } from '../src/harness/providers/aforge.js';
import { buildProvider, SUPPORTED_PROVIDERS } from '../src/harness/providers/factory.js';
import * as cli from '../src/harness/cli.js';

function envelope(
  deliverable = 'done',
  options: { settled?: boolean; blockedOn?: string; usage?: Record<string, unknown> } = {}
): string {
  return JSON.stringify({
    settled: options.settled ?? true,
    deliverable,
    blocked_on: options.blockedOn ?? '',
    spend_usd: 0.0123,
    elapsed_ms: 12,
    usage: options.usage ?? {},
  });
}

function execEnvelope(
  text = 'done',
  options: { stop?: string; usage?: Record<string, unknown>; turns?: number } = {}
): string {
  return JSON.stringify({
    text,
    stop: options.stop ?? 'done',
    usage: options.usage ?? {},
    artifacts: [],
    turns: options.turns ?? 1,
    elapsed_ms: 12,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  delete process.env.AGENTFIELD_HARNESS_TIMEOUT_SECONDS;
  delete process.env.AFORGE_BIN;
  delete process.env.AGENTFIELD_AFORGE_COMMAND;
});

beforeEach(() => {
  process.env.AGENTFIELD_AFORGE_COMMAND = 'do';
});

describe('aforge provider', () => {
  it('honors AFORGE_BIN unless a binary is explicit', async () => {
    process.env.AFORGE_BIN = '/opt/aforge-env';
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });

    await new AforgeProvider().execute('hello', {});
    await new AforgeProvider('/explicit/aforge').execute('hello', {});

    expect(vi.mocked(cli.runCli).mock.calls[0][0][0]).toBe('/opt/aforge-env');
    expect(vi.mocked(cli.runCli).mock.calls[1][0][0]).toBe('/explicit/aforge');
  });

  it('maps the do command, stdin prompt, JSON envelope, and metrics', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: envelope(' final answer ', {
        usage: {
          calls: 3,
          prompt_tokens: 100,
          completion_tokens: 50,
          cached_tokens: 20,
          cost: 0.0123,
        },
      }),
      stderr: '',
      exitCode: 0,
    });

    const result = await new AforgeProvider('/opt/aforge').execute('prompt that stays off argv', {
      projectDir: '/project',
      cwd: '/project/nested',
      systemPrompt: '  be precise  ',
      model: 'openrouter/z-ai/glm-5.2#high',
    });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['/opt/aforge', 'do', '--json', '--yes-spend', '-w', '/project', '--timeout', '1795'],
      {
        env: { AFORGE_MODEL: 'z-ai/glm-5.2', AFORGE_EXEC_REASONING: 'high' },
        cwd: undefined,
        timeout: 1_800_000,
        idleSeconds: 0,
        inputText: 'be precise\n\nTask:\nprompt that stays off argv',
      }
    );
    expect(result.result).toBe('final answer');
    expect(result.isError).toBe(false);
    expect(result.failureType).toBe('none');
    expect(result.returnCode).toBe(0);
    expect(result.metrics).toMatchObject({
      numTurns: 3,
      totalCostUsd: 0.0123,
      inputTokens: 100,
      outputTokens: 50,
      cacheReadTokens: 20,
      cacheCreationTokens: 0,
      totalTokens: 150,
      model: 'openrouter/z-ai/glm-5.2',
    });
    expect(result.messages[0].deliverable).toBe(' final answer ');
  });

  it('maps the opt-in exec command and original envelope contract', async () => {
    delete process.env.AGENTFIELD_AFORGE_COMMAND;
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: execEnvelope(' linear answer ', {
        turns: 4,
        usage: {
          calls: 3,
          prompt_tokens: 100,
          completion_tokens: 50,
          cached_tokens: 20,
          cost: 0.0123,
        },
      }),
      stderr: '',
      exitCode: 0,
    });

    const result = await new AforgeProvider('/opt/aforge').execute('prompt that stays off argv', {
      projectDir: '/project',
      systemPrompt: '  be precise  ',
      model: 'openrouter/deepseek/deepseek-v4-flash-0731#high',
    });

    expect(cli.runCli).toHaveBeenCalledWith(
      [
        '/opt/aforge', 'exec', '--json', '-w', '/project', '--timeout', '1795',
        '--context-fill', '60', '--completion-reserve', '65536',
        '--system', 'be precise',
        '--model', 'deepseek/deepseek-v4-flash-0731',
        '--plan-model', 'deepseek/deepseek-v4-flash-0731',
      ],
      {
        env: {
          AFORGE_MODELS: '',
          AFORGE_MODEL: 'deepseek/deepseek-v4-flash-0731',
          AFORGE_EXEC_REASONING: 'high',
        },
        cwd: undefined,
        timeout: 1_800_000,
        idleSeconds: 0,
        inputText: 'prompt that stays off argv',
      }
    );
    expect(result.result).toBe('linear answer');
    expect(result.isError).toBe(false);
    expect(result.metrics.numTurns).toBe(4);
    expect(result.metrics.inputTokens).toBe(100);
    expect(result.metrics.totalCostUsd).toBe(0.0123);
    expect(result.metrics.model).toBe('openrouter/deepseek/deepseek-v4-flash-0731');
  });

  it.each([
    [{ maxTurns: 2 }, '2'],
    [{}, undefined],
    [{ maxTurns: 0 }, undefined],
    [{ maxTurns: -2 }, undefined],
    [{ maxTurns: Number.NaN }, undefined],
    [{ maxTurns: Number.POSITIVE_INFINITY }, undefined],
    [{ maxBudgetUsd: 1.5 }, undefined],
  ] as const)('maps exec turn options %o to %s', async (options, expectedTurns) => {
    process.env.AGENTFIELD_AFORGE_COMMAND = 'exec';
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: execEnvelope(), stderr: '', exitCode: 0,
    });

    await new AforgeProvider().execute('hello', options);

    const cmd = vi.mocked(cli.runCli).mock.calls[0][0];
    if (expectedTurns === undefined) {
      expect(cmd).not.toContain('--turns');
    } else {
      const timeoutIndex = cmd.indexOf('--timeout');
      expect(cmd.slice(timeoutIndex + 2, timeoutIndex + 4)).toEqual(['--turns', expectedTurns]);
    }
    expect(cmd).not.toContain('--budget');
  });

  it('does not pass a positive turn cap to do', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });

    await new AforgeProvider().execute('hello', { maxTurns: 2 });

    expect(vi.mocked(cli.runCli).mock.calls[0][0]).not.toContain('--turns');
  });

  it.each([
    { options: { model: 'some/model#high' }, expected: 'some/model' },
    { options: { env: { AFORGE_MODEL: 'env/model' } }, expected: 'env/model' },
    { options: {}, expected: AFORGE_DEFAULT_MODEL },
  ])('reports the effective model for options $options', async ({ options, expected }) => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });

    const result = await new AforgeProvider().execute('hello', options);

    expect(result.metrics.model).toBe(expected);
  });

  it('accepts an exec budget partial with usable text', async () => {
    process.env.AGENTFIELD_AFORGE_COMMAND = 'exec';
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: execEnvelope('usable', { stop: 'budget', turns: 2 }),
      stderr: '',
      exitCode: 2,
    });

    const result = await new AforgeProvider().execute('hello', {});

    expect(result.result).toBe('usable');
    expect(result.isError).toBe(false);
    expect(result.failureType).toBe('none');
  });

  it('uses cwd as the root and gives aforge a timeout landing window', async () => {
    process.env.AGENTFIELD_HARNESS_TIMEOUT_SECONDS = '2400';
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });

    await new AforgeProvider().execute('hello', { cwd: '/cwd-only' });

    expect(cli.runCli).toHaveBeenCalledWith(
      ['aforge', 'do', '--json', '--yes-spend', '-w', '/cwd-only', '--timeout', '2395'],
      expect.objectContaining({ timeout: 2_400_000, idleSeconds: 0, inputText: 'hello' })
    );
  });

  it('ignores unknown variants and lets caller env override derived env', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });
    const provider = new AforgeProvider();

    await provider.execute('hello', { model: 'openrouter/x/y#turbo' });
    await provider.execute('hello', {
      model: 'openrouter/x/y#low',
      variant: 'HIGH',
      env: { AFORGE_MODEL: 'override/model', AFORGE_EXEC_REASONING: 'off', EXTRA: '1' },
    });

    expect(vi.mocked(cli.runCli).mock.calls[0][1]?.env).toEqual({ AFORGE_MODEL: 'x/y' });
    expect(vi.mocked(cli.runCli).mock.calls[1][1]?.env).toEqual({
      AFORGE_MODEL: 'override/model',
      AFORGE_EXEC_REASONING: 'off',
      EXTRA: '1',
    });
  });

  it.each([
    ['success', 0, 'done', '', 'none', false],
    ['timeout with partial', 2, 'usable', '', 'timeout', true],
    ['blocked', 1, '', 'Which repository?', 'crash', true],
    ['zero without deliverable', 0, '', '', 'crash', true],
    ['signal', -9, '', '', 'crash', true],
  ])('applies %s exit semantics', async (_name, exitCode, deliverable, blockedOn, failureType, wantError) => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: envelope(deliverable as string, { settled: exitCode === 0, blockedOn: blockedOn as string }),
      stderr: '',
      exitCode: exitCode as number,
    });

    const result = await new AforgeProvider().execute('hello', {});

    expect(result.isError).toBe(wantError);
    expect(result.failureType).toBe(failureType);
  });

  it('parses the last envelope and leaves zero cost unknown', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: `stray diagnostic\n{"type":"event"}\n${envelope('real result', { usage: { calls: 1, cost: 0 } })}`,
      stderr: '',
      exitCode: 0,
    });

    const result = await new AforgeProvider().execute('hello', {});

    expect(result.result).toBe('real result');
    expect(result.metrics.totalCostUsd).toBeUndefined();
  });

  it('parses the canonical pretty-printed envelope', async () => {
    const pretty = JSON.stringify(JSON.parse(envelope('pretty result')), null, 2);
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: pretty, stderr: '', exitCode: 0 });

    const result = await new AforgeProvider().execute('hello', {});

    expect(result.result).toBe('pretty result');
    expect(result.isError).toBe(false);
  });

  it('classifies missing binaries and timeouts', async () => {
    vi.spyOn(cli, 'runCli').mockRejectedValueOnce(new Error('spawn aforge ENOENT'));
    const missing = await new AforgeProvider('aforge-missing').execute('hello', {});
    expect(missing.isError).toBe(true);
    expect(missing.failureType).toBe('crash');
    expect(missing.errorMessage).toContain('aforge-missing');

    vi.spyOn(cli, 'runCli').mockRejectedValueOnce(new Error('CLI timed out after 1ms'));
    const timeout = await new AforgeProvider().execute('hello', {});
    expect(timeout.isError).toBe(true);
    expect(timeout.failureType).toBe('timeout');
  });
});

describe('aforge provider factory', () => {
  it('registers aforge and threads aforgeBin', async () => {
    expect(SUPPORTED_PROVIDERS.has('aforge')).toBe(true);
    const provider = await buildProvider({ provider: 'aforge', aforgeBin: '/opt/aforge' });
    expect(provider).toBeInstanceOf(AforgeProvider);
  });
});
