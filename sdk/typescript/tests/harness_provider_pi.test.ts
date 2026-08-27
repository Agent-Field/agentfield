import { afterEach, describe, expect, it, vi } from 'vitest';

import * as cli from '../src/harness/cli.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import { OMPProvider, PiProvider } from '../src/harness/providers/pi.js';

afterEach(() => {
  vi.restoreAllMocks();
});

function eventStream(text: string): string {
  return [
    { type: 'session', id: 'session-123' },
    { type: 'turn_start' },
    {
      type: 'message_end',
      message: {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'internal' },
          { type: 'text', text },
        ],
        model: 'google/gemini-2.5-flash',
        usage: {
          input: 120,
          output: 30,
          cacheRead: 10,
          cacheWrite: 4,
          cost: { total: 0.0025 },
        },
        stopReason: 'stop',
      },
    },
    { type: 'turn_end' },
    { type: 'agent_end' },
  ].map((event) => JSON.stringify(event)).join('\n');
}

const providers = [
  { name: 'pi', provider: new PiProvider() },
  { name: 'omp', provider: new OMPProvider() },
];

it.each(providers)('$name classifies a successful assistant response', async ({ provider }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({
    stdout: eventStream('done'),
    stderr: '',
    exitCode: 0,
  });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({ isError: false, failureType: 'none', returnCode: 0 });
});

it.each(providers)('$name classifies a signal death as a crash', async ({ provider }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr: '', exitCode: -9 });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({
    isError: true,
    failureType: 'crash',
    returnCode: -9,
    errorMessage: 'Process killed by signal 9.',
  });
});

it.each(providers)('$name uses cleaned stderr for a non-zero exit', async ({ provider }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({
    stdout: '',
    stderr: '  \u001b[31mprovider failed\u001b[0m  ',
    exitCode: 2,
  });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({
    isError: true,
    failureType: 'crash',
    returnCode: 2,
    errorMessage: 'provider failed',
  });
});

it.each(providers)('$name falls back to the exit code for an empty error', async ({ provider }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr: '', exitCode: 2 });

  const result = await provider.execute('hello', {});

  expect(result.errorMessage).toBe('Process exited with code 2.');
});

it.each(providers)('$name classifies a provider event error', async ({ provider }) => {
  const stdout = JSON.stringify({
    type: 'message_end',
    message: {
      role: 'assistant',
      content: [],
      stopReason: 'error',
      errorMessage: 'provider detail',
    },
  });
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout, stderr: '', exitCode: 0 });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({
    isError: true,
    failureType: 'api_error',
    returnCode: 0,
    errorMessage: 'provider detail',
  });
});

it.each(providers)('$name clears a recovered provider event error', async ({ provider }) => {
  const stdout = [
    {
      type: 'message_end',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'partial' }],
        stopReason: 'error',
        errorMessage: 'upstream 503',
      },
    },
    { type: 'turn_end' },
    {
      type: 'message_end',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'FINAL ANSWER' }],
        stopReason: 'stop',
      },
    },
    { type: 'turn_end' },
  ].map((event) => JSON.stringify(event)).join('\n');
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout, stderr: '', exitCode: 0 });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({
    result: 'FINAL ANSWER',
    isError: false,
    failureType: 'none',
    returnCode: 0,
  });
  expect(result.errorMessage).toBeUndefined();
});

it.each(providers)('$name classifies a successful exit without output', async ({ provider }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr: '', exitCode: 0 });

  const result = await provider.execute('hello', {});

  expect(result).toMatchObject({ isError: true, failureType: 'no_output', returnCode: 0 });
});

it.each([
  { message: 'spawn ENOENT', failureType: 'crash' },
  { message: 'CLI timed out after 1000ms', failureType: 'timeout' },
])('classifies a rejected CLI as $failureType', async ({ message, failureType }) => {
  vi.spyOn(cli, 'runCli').mockRejectedValue(new Error(message));

  const result = await new PiProvider().execute('hello', {});

  expect(result).toMatchObject({ isError: true, failureType });
});

describe.each([
  {
    name: 'pi',
    provider: new PiProvider('/opt/pi'),
    prefix: ['/opt/pi', '--print', '--mode', 'json'],
    permissionFlag: undefined,
    globTool: 'find',
  },
  {
    name: 'omp',
    provider: new OMPProvider('/opt/omp'),
    prefix: ['/opt/omp', '--print', '--mode', 'json', '--cwd', '/tmp/project'],
    permissionFlag: '--auto-approve',
    globTool: 'glob',
  },
])('$name provider', ({ provider, prefix, permissionFlag, globTool }) => {
  it('maps common harness options and native metrics', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: eventStream('done'),
      stderr: '',
      exitCode: 0,
    });

    const result = await provider.execute('implement this', {
      projectDir: '/tmp/project',
      model: 'openrouter/google/gemini-2.5-flash#high',
      permissionMode: 'auto',
      systemPrompt: 'Be precise.',
      tools: ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep'],
      env: { EXTRA: '1' },
    });

    const [cmd, options] = vi.mocked(cli.runCli).mock.calls[0];
    expect(cmd.slice(0, prefix.length)).toEqual(prefix);
    if (permissionFlag) {
      expect(cmd).toContain(permissionFlag);
    }
    expectApprovalFlags(cmd, permissionFlag);
    expect(cmd.slice(cmd.indexOf('--model'), cmd.indexOf('--model') + 2)).toEqual([
      '--model',
      'openrouter/google/gemini-2.5-flash',
    ]);
    expect(cmd.slice(cmd.indexOf('--thinking'), cmd.indexOf('--thinking') + 2)).toEqual([
      '--thinking',
      'high',
    ]);
    expect(cmd[cmd.indexOf('--tools') + 1]).toBe(`read,write,edit,bash,${globTool},grep`);
    expect(options).toEqual({
      env: { EXTRA: '1' },
      cwd: '/tmp/project',
      inputText: 'implement this',
      timeout: undefined,
    });

    expect(result.isError).toBe(false);
    expect(result.result).toBe('done');
    expect(result.metrics).toMatchObject({
      sessionId: 'session-123',
      numTurns: 1,
      totalCostUsd: 0.0025,
      inputTokens: 120,
      outputTokens: 30,
      cacheReadTokens: 10,
      cacheCreationTokens: 4,
      totalTokens: 150,
      model: 'openrouter/google/gemini-2.5-flash',
    });
  });
});

it.each([
  { provider: new PiProvider(), resumeFlag: '--session', tools: 'read,grep,find' },
  { provider: new OMPProvider(), resumeFlag: '--resume', tools: 'read,grep,glob' },
])('keeps $resumeFlag retries read-only', async ({ provider, resumeFlag, tools }) => {
  vi.spyOn(cli, 'runCli').mockResolvedValue({
    stdout: eventStream('plan'),
    stderr: '',
    exitCode: 0,
  });

  await provider.execute('plan this', {
    permissionMode: 'plan',
    tools: ['Read', 'Write', 'Bash', 'Grep', 'Glob'],
    resumeSessionId: 'abc123',
  });

  const cmd = vi.mocked(cli.runCli).mock.calls[0][0];
  expect(cmd[cmd.indexOf('--tools') + 1]).toBe(tools);
  expect(cmd[cmd.indexOf(resumeFlag) + 1]).toBe('abc123');
  expectApprovalFlags(cmd);
});

function expectApprovalFlags(cmd: string[], allowed?: string): void {
  const approvalFlags = [
    '--approve',
    '--auto-approve',
    '--yolo',
    '-y',
    '--approval-mode',
    '--permission-mode',
  ];
  expect(approvalFlags.filter((flag) => cmd.includes(flag))).toEqual(allowed ? [allowed] : []);
}

it.each([
  { provider: new PiProvider('pi-missing'), installHint: '@earendil-works/pi-coding-agent' },
  { provider: new OMPProvider('omp-missing'), installHint: 'omp.sh/install' },
])('returns actionable install guidance for a missing binary', async ({ provider, installHint }) => {
  vi.spyOn(cli, 'runCli').mockRejectedValue(new Error('spawn ENOENT'));

  const result = await provider.execute('hello', {});

  expect(result.isError).toBe(true);
  expect(result.errorMessage).toContain(installHint);
});

describe('provider factory', () => {
  it('routes pi and omp and passes binary overrides', async () => {
    const pi = await buildProvider({ provider: 'pi', piBin: '/opt/pi' });
    const omp = await buildProvider({ provider: 'omp', ompBin: '/opt/omp' });

    expect(pi).toBeInstanceOf(PiProvider);
    expect(omp).toBeInstanceOf(OMPProvider);
    // Pi and OMP are additional providers: they must be named explicitly.
    // A provider-less config still resolves to the default, aforge.
  });
});
