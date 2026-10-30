import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CODEAF_DEFAULT_MODEL, CodeafProvider, resolveCodeafMaxConcurrent } from '../src/harness/providers/codeaf.js';
import * as cli from '../src/harness/cli.js';
import { providerUnavailable } from '../src/harness/availability.js';

function envelope(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    ok: true, stop: 'done', answer: ' answer ', files: ['result.txt'], error: '',
    spend_usd: 0, tokens: { in: 123, out: 45 }, seconds: 999, model: 'returned/model',
    steps: 4, calls: 8, ...overrides,
  });
}

let directory: string;

beforeEach(() => {
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-codeaf-provider-'));
  for (const name of [
    'AGENTFIELD_CODEAF_COMMAND', 'AGENTFIELD_HARNESS_TIMEOUT_SECONDS', 'CODEAF_HOME',
    'CODEAF_NO_UPDATE_CHECK', 'CODEAF_TELEMETRY', 'CODEAF_EXEC_REASONING', 'CODEAF_MODELS',
    'AFORGE_MODELS', 'CI', 'DO_NOT_TRACK', 'AGENTFIELD_TELEMETRY_ENABLED', 'CODEAF_MAX_CONCURRENT',
    'AFORGE_MAX_CONCURRENT', 'AFORGE_MODEL', 'CODEAF_MODEL', 'AGENTFIELD_AFORGE_COMMAND',
  ]) vi.stubEnv(name, undefined);
  vi.stubEnv('AGENTFIELD_HOME', directory);
  vi.stubEnv('HOME', path.join(directory, 'user-home'));
  vi.stubEnv('OPENROUTER_API_KEY', 'test-codeaf-key');
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: envelope(), stderr: '', exitCode: 0 });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  fs.rmSync(directory, { recursive: true, force: true });
});

describe('V7 CodeAF invocation', () => {
  it('passes exact exec flags, stdin, current option names, model and reasoning variant', async () => {
    const result = await new CodeafProvider('/explicit/codeaf').execute('private task', {
      projectDir: '/project', cwd: '/project/nested', model: 'openrouter/vendor/model#low',
      variant: 'HIGH', systemPrompt: '  be precise  ', maxTurns: 7.9, maxBudgetUsd: 10,
    });
    expect(cli.runCli).toHaveBeenCalledWith([
      '/explicit/codeaf', 'exec', '--json', '--dir', '/project', '--model', 'vendor/model',
      '--timeout', expect.any(String), '--max-turns', '7', '--system', 'be precise',
    ], {
      env: { CODEAF_HOME: path.join(directory, 'codeaf'), CODEAF_NO_UPDATE_CHECK: '1',
        CODEAF_MODELS: undefined, AFORGE_MODELS: undefined, CODEAF_EXEC_REASONING: 'high' },
      cwd: undefined, timeout: expect.any(Number), idleSeconds: 0, inputText: 'private task',
    });
    expect(result.provider).toBe('codeaf');
    expect(fs.statSync(path.join(directory, 'codeaf')).isDirectory()).toBe(true);
  });

  it('passes exact do flags, pins all three models, and combines system/task on stdin', async () => {
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', 'do');
    await new CodeafProvider('/explicit/codeaf').execute('private task', {
      cwd: '/cwd', model: 'openrouter/openrouter/vendor/model', system_prompt: ' system ', maxTurns: 10,
    });
    expect(cli.runCli).toHaveBeenCalledWith([
      '/explicit/codeaf', 'do', '--json', '--dir', '/cwd', '--yes-spend',
      '--model', 'openrouter/vendor/model', '--plan-model', 'openrouter/vendor/model',
      '--check-model', 'openrouter/vendor/model', '--timeout', expect.any(String), '-',
    ], expect.objectContaining({ inputText: 'system\n\nTask:\nprivate task', idleSeconds: 0 }));
  });

  it.each(['exec', 'do'])('always supplies the predictable default model for %s', async (command) => {
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', command);
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({ model: '' }), stderr: '', exitCode: 0 });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', {});
    const cmd = vi.mocked(cli.runCli).mock.calls[0][0];
    expect(cmd.slice(cmd.indexOf('--model'), cmd.indexOf('--model') + 2)).toEqual(['--model', CODEAF_DEFAULT_MODEL]);
    expect(result.metrics.model).toBe('~deepseek/deepseek-v4-flash-latest');
    expect(vi.mocked(cli.runCli).mock.calls[0][1]?.inputText).toBe('task');
  });

  it('warns once and uses exec for an invalid command', async () => {
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', 'run');
    const provider = new CodeafProvider('/explicit/codeaf');
    await provider.execute('task', {});
    await provider.execute('task', {});
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', 'RUN');
    await provider.execute('task', {});
    expect(vi.mocked(cli.runCli).mock.calls[0][0][1]).toBe('exec');
    expect(vi.mocked(cli.runCli).mock.calls[2][0][1]).toBe('exec');
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['2400', '2395s', 2_400_000], ['5', '1s', 5000], ['1', '1s', 1000],
    ['0', '1795s', 1_800_000], ['-1', '1795s', 1_800_000], ['garbage', '1795s', 1_800_000],
  ])('keeps the landing window for outer timeout %s', async (value, inner, timeout) => {
    vi.stubEnv('AGENTFIELD_HARNESS_TIMEOUT_SECONDS', value);
    await new CodeafProvider('/explicit/codeaf').execute('task', {});
    const [cmd, options] = vi.mocked(cli.runCli).mock.calls[0];
    expect(Number.parseFloat(cmd[cmd.indexOf('--timeout') + 1])).toBeCloseTo(Number.parseFloat(inner), 1);
    expect(options!.timeout).toBeGreaterThan(timeout - 100);
    expect(options!.timeout).toBeLessThanOrEqual(timeout);
  });

  it.each([0, -1, NaN, Infinity])('omits invalid maxTurns %s', async (maxTurns) => {
    await new CodeafProvider('/explicit/codeaf').execute('task', { maxTurns });
    expect(vi.mocked(cli.runCli).mock.calls[0][0]).not.toContain('--max-turns');
  });
});

describe('V8 CodeAF child environment', () => {
  it.each([
    ['CI', 'TRUE'], ['DO_NOT_TRACK', 'yes'], ['AGENTFIELD_TELEMETRY_ENABLED', '0'],
    ['AGENTFIELD_TELEMETRY_ENABLED', 'false'], ['AGENTFIELD_TELEMETRY_ENABLED', 'no'],
    ['AGENTFIELD_TELEMETRY_ENABLED', 'OFF'],
  ])('disables telemetry for %s=%s', async (name, value) => {
    vi.stubEnv(name, value);
    await new CodeafProvider('/explicit/codeaf').execute('task', {});
    expect(vi.mocked(cli.runCli).mock.calls[0][1]?.env?.CODEAF_TELEMETRY).toBe('off');
  });

  it('leaves telemetry unset without an opt-out and ignores unknown variants', async () => {
    await new CodeafProvider('/explicit/codeaf').execute('task', { model: 'some/model#turbo', env: ['invalid'] });
    const env = vi.mocked(cli.runCli).mock.calls[0][1]?.env!;
    expect(env).not.toHaveProperty('CODEAF_TELEMETRY');
    expect(env).not.toHaveProperty('CODEAF_EXEC_REASONING');
    expect(Object.keys(env).filter((name) => name.startsWith('AFORGE_'))).toEqual(['AFORGE_MODELS']);
    expect(env.AFORGE_MODELS).toBeUndefined();
  });

  it('preserves inherited settings, removes inherited panels, and lets caller overlay win', async () => {
    vi.stubEnv('CODEAF_HOME', '/inherited/home');
    vi.stubEnv('CODEAF_NO_UPDATE_CHECK', '0');
    vi.stubEnv('CODEAF_TELEMETRY', 'on');
    vi.stubEnv('CODEAF_MODELS', 'inherited-panel');
    vi.stubEnv('AFORGE_MODELS', 'old-inherited-panel');
    vi.stubEnv('CI', '1');
    const provider = new CodeafProvider('/explicit/codeaf');
    await provider.execute('task', { variant: 'low' });
    const first = vi.mocked(cli.runCli).mock.calls[0][1]?.env!;
    expect(first).not.toHaveProperty('CODEAF_HOME');
    expect(first).not.toHaveProperty('CODEAF_NO_UPDATE_CHECK');
    expect(first).not.toHaveProperty('CODEAF_TELEMETRY');
    expect(first.CODEAF_MODELS).toBeUndefined();
    expect(first.AFORGE_MODELS).toBeUndefined();

    await provider.execute('task', { variant: 'high', env: {
      CODEAF_HOME: '/overlay/home', CODEAF_NO_UPDATE_CHECK: '', CODEAF_TELEMETRY: 'on',
      CODEAF_EXEC_REASONING: 'off', CODEAF_MODELS: 'panel', AFORGE_MODELS: 'old-panel', EXTRA: '1', INVALID: 3,
    } });
    expect(vi.mocked(cli.runCli).mock.calls[1][1]?.env).toEqual({
      CODEAF_HOME: '/overlay/home', CODEAF_NO_UPDATE_CHECK: '', CODEAF_TELEMETRY: 'on',
      CODEAF_EXEC_REASONING: 'off', CODEAF_MODELS: 'panel', AFORGE_MODELS: 'old-panel', EXTRA: '1',
    });
  });

  it('evaluates telemetry triggers after the per-call overlay', async () => {
    vi.stubEnv('CI', '1');
    await new CodeafProvider('/explicit/codeaf').execute('task', { env: { CI: '0', DO_NOT_TRACK: '0' } });
    expect(vi.mocked(cli.runCli).mock.calls[0][1]?.env).not.toHaveProperty('CODEAF_TELEMETRY');
  });
});

describe('V9–V10 canonical CodeAF results', () => {
  it('maps canonical metrics, keeps measured zero spend, and ignores every legacy alias', async () => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({
      text: 'wrong', deliverable: 'wrong', turns: 100, elapsed_ms: 999999,
      usage: { prompt_tokens: 999, completion_tokens: 999, cost: 123 }, artifacts: ['wrong'],
    }), stderr: '', exitCode: 0 });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', { model: 'requested/model' });
    expect(result).toMatchObject({ result: 'answer', provider: 'codeaf', isError: false, failureType: 'none', returnCode: 0 });
    expect(result.metrics).toMatchObject({
      numTurns: 4, inputTokens: 123, outputTokens: 45, totalTokens: 168, totalCostUsd: 0,
      model: 'returned/model', cacheReadTokens: 0, cacheCreationTokens: 0,
    });
    expect(result.metrics.durationApiMs).toBeLessThan(999000);
    expect(result.messages[0].files).toEqual(['result.txt']);
  });

  it.each([
    ['whole pretty JSON', JSON.stringify(JSON.parse(envelope()), null, 2)],
    ['diagnostics plus the last canonical object', `log\n${envelope({ answer: 'first' })}\n${envelope()}\n{"event":"tail"}\ngarbage`],
  ])('parses %s', async (_name, stdout) => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout, stderr: '', exitCode: 0 });
    expect((await new CodeafProvider('/explicit/codeaf').execute('task', {})).result).toBe('answer');
  });

  it.each([
    ['done', true, 'answer', 0, 'none'],
    ['done', false, 'answer', 0, 'crash'],
    ['deadline', false, 'partial', 3, 'timeout'],
    ['deadline', false, 'partial', 124, 'timeout'],
    ['unchecked', false, 'answer', 2, 'none'],
    ['unchecked', false, '', 2, 'crash'],
    ['budget', false, 'partial', 3, 'none'],
    ['turn-cap', false, 'partial', 3, 'none'],
    ['budget', false, '', 3, 'no_output'],
    ['turn-cap', false, ' ', 3, 'no_output'],
    ['question', false, 'partial', 4, 'crash'],
    ['incomplete', false, 'partial', 2, 'crash'],
    ['error', false, 'partial', 1, 'crash'],
    ['price', false, 'partial', 3, 'crash'],
    ['future-stop', true, 'answer', 0, 'none'],
    ['future-stop', true, 'answer', 1, 'crash'],
    ['future-stop', true, '', 0, 'crash'],
    ['done', true, 'answer', -9, 'crash'],
  ])('maps stop=%s ok=%s answer=%s exit=%s to %s', async (stop, ok, answer, exitCode, failureType) => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({ stop, ok, answer }), stderr: '', exitCode });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', {});
    expect(result.failureType).toBe(failureType);
    expect(result.isError).toBe(failureType !== 'none');
    expect(result.result).toBe(failureType === 'none' ? answer : undefined);
    if (['unchecked', 'budget', 'turn-cap'].includes(stop) && failureType === 'none') {
      expect(console.warn).toHaveBeenCalled();
      if (stop !== 'unchecked') expect(vi.mocked(console.warn).mock.calls[0][0]).toContain('partial');
    }
  });

  it.each([
    ['', 0, 'no_output'], ['  \n ', 0, 'no_output'], ['', 1, 'crash'], ['', -15, 'crash'],
    ['diagnostic', 0, 'crash'], ['{"text":"legacy","deliverable":"legacy"}', 0, 'crash'],
    ['[]', 0, 'crash'], ['{"ok":true,"answer":"missing stop"}', 0, 'crash'],
  ])('maps missing canonical envelope %s exit=%s to %s', async (stdout, exitCode, failureType) => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout, stderr: '', exitCode });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', {});
    expect(result).toMatchObject({ isError: true, failureType, messages: [] });
    expect(result.result).toBeUndefined();
  });

  it('includes blocked_on for questions and prefers canonical error over answer/stderr', async () => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({
      ok: false, stop: 'question', answer: 'partial answer', error: 'explicit error', blocked_on: 'Which project?',
    }), stderr: 'stderr detail', exitCode: 4 });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', {});
    expect(result.errorMessage).toBe('codeaf exit code 4 (stop=question): explicit error. blocked_on: Which project?');
  });

  it('uses canonical answer or a bounded stripped stderr tail for errors', async () => {
    const provider = new CodeafProvider('/explicit/codeaf');
    vi.mocked(cli.runCli).mockResolvedValueOnce({ stdout: envelope({ stop: 'error' }), stderr: 'stderr', exitCode: 1 });
    expect((await provider.execute('task', {})).errorMessage).toBe('codeaf exit code 1 (stop=error): answer');
    vi.mocked(cli.runCli).mockResolvedValueOnce({ stdout: '', stderr: `head${'x'.repeat(1500)}\x1b[31mTAIL\x1b[0m`, exitCode: 1 });
    const result = await provider.execute('task', {});
    expect(result.errorMessage).toBe(`codeaf exit code 1 (stop=): ${'x'.repeat(996)}TAIL`);
  });

  it('ignores nonnumeric metrics and falls back to the requested model', async () => {
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({
      model: ' ', steps: 'wrong', spend_usd: '0.5', tokens: [],
    }), stderr: '', exitCode: 0 });
    const result = await new CodeafProvider('/explicit/codeaf').execute('task', { model: 'openrouter/vendor/model#medium' });
    expect(result.metrics).toMatchObject({ numTurns: 0, inputTokens: 0, outputTokens: 0, model: 'vendor/model' });
    expect(result.metrics.totalCostUsd).toBeUndefined();
  });

  it('classifies outer timeout and crashes, and preserves typed missing-binary errors', async () => {
    const provider = new CodeafProvider('/explicit/codeaf');
    vi.mocked(cli.runCli).mockRejectedValueOnce(new Error('CLI timed out after 1ms'));
    expect(await provider.execute('task', {})).toMatchObject({ isError: true, failureType: 'timeout', provider: 'codeaf' });
    vi.mocked(cli.runCli).mockRejectedValueOnce('unexpected crash');
    expect(await provider.execute('task', {})).toMatchObject({ isError: true, failureType: 'crash' });
    vi.mocked(cli.runCli).mockRejectedValueOnce(new Error('spawn codeaf ENOENT'));
    await expect(provider.execute('task', {})).rejects.toMatchObject({ provider: 'codeaf', installCommand: 'af codeaf ensure' });
    const missing = providerUnavailable('codeaf', '/explicit/codeaf');
    vi.mocked(cli.runCli).mockRejectedValueOnce(missing);
    await expect(provider.execute('task', {})).rejects.toBe(missing);
  });
});

describe('V12 CodeAF concurrency parsing', () => {
  it.each(['bad', '', '0', '-1', '1.5', '2garbage', 'Infinity', 'NaN'])('invalid %s falls back to eight', (value) => {
    vi.stubEnv('CODEAF_MAX_CONCURRENT', value);
    expect(resolveCodeafMaxConcurrent()).toBe(8);
  });

  it('uses the new setting, then the legacy setting only when the new one is unset', () => {
    expect(resolveCodeafMaxConcurrent()).toBe(8);
    vi.stubEnv('AFORGE_MAX_CONCURRENT', '3');
    expect(resolveCodeafMaxConcurrent()).toBe(3);
    vi.stubEnv('CODEAF_MAX_CONCURRENT', '2');
    expect(resolveCodeafMaxConcurrent()).toBe(2);
    vi.stubEnv('CODEAF_MAX_CONCURRENT', 'bad');
    expect(resolveCodeafMaxConcurrent()).toBe(8);
  });
});

describe('F2 legacy command and model fallbacks', () => {
  it.each([
    ['R3a', '', ' DO ', 'do'],
    ['R3b', 'Exec', 'do', 'exec'],
  ])('%s normalizes command casing while retaining precedence', async (_contract, current, legacy, want) => {
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', current);
    vi.stubEnv('AGENTFIELD_AFORGE_COMMAND', legacy);
    await new CodeafProvider('/fixture').execute('p', {});
    expect(vi.mocked(cli.runCli).mock.calls[0][0][1]).toBe(want);
    expect(console.warn).not.toHaveBeenCalled();
  });

  it.each([
    ['', 'do', 'do'], ['exec', 'do', 'exec'], [' ', 'do', 'do'], ['bad', 'do', 'exec'],
  ])('F2a/b command precedence %s / %s', async (current, legacy, want) => {
    vi.stubEnv('AGENTFIELD_CODEAF_COMMAND', current);
    vi.stubEnv('AGENTFIELD_AFORGE_COMMAND', legacy);
    await new CodeafProvider('/fixture').execute('p', {});
    expect(vi.mocked(cli.runCli).mock.calls[0][0][1]).toBe(want);
  });
  it.each([
    [undefined, '', 'openrouter/old', {}, 'old'],
    [undefined, 'new', 'old', {}, 'new'],
    ['openrouter/explicit#high', 'new', 'old', {}, 'explicit'],
    [undefined, 'ambient', 'old', { CODEAF_MODEL: ' ', AFORGE_MODEL: 'openrouter/overlay' }, 'overlay'],
    [undefined, '', 'old', { AFORGE_MODEL: 'overlay' }, 'overlay'],
    [undefined, '', '', {}, CODEAF_DEFAULT_MODEL],
  ])('F2c-f model precedence', async (model, current, legacy, env, want) => {
    vi.stubEnv('CODEAF_MODEL', current);
    vi.stubEnv('AFORGE_MODEL', legacy);
    vi.stubEnv('AGENTFIELD_AFORGE_COMMAND', 'do');
    vi.mocked(cli.runCli).mockResolvedValue({ stdout: envelope({ model: '' }), stderr: '', exitCode: 0 });
    const result = await new CodeafProvider('/fixture').execute('p', { model, env });
    const cmd = vi.mocked(cli.runCli).mock.calls[0][0];
    for (const flag of ['--model', '--plan-model', '--check-model']) expect(cmd[cmd.indexOf(flag) + 1]).toBe(want);
    expect(result.metrics.model).toBe(want);
  });
});
