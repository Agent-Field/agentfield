import { afterEach, describe, expect, it, vi } from 'vitest';

import * as cli from '../src/harness/cli.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import { OmpProvider, PiProvider } from '../src/harness/providers/pi.js';

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

describe.each([
  {
    name: 'pi',
    provider: new PiProvider('/opt/pi'),
    prefix: ['/opt/pi', '--print', '--mode', 'json'],
    permissionFlag: '--approve',
    globTool: 'find',
  },
  {
    name: 'omp',
    provider: new OmpProvider('/opt/omp'),
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
    expect(cmd).toContain(permissionFlag);
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
  { provider: new OmpProvider(), resumeFlag: '--resume', tools: 'read,grep,glob' },
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
});

describe('provider factory', () => {
  it('routes pi and omp and passes binary overrides', async () => {
    const pi = await buildProvider({ provider: 'pi', piBin: '/opt/pi' });
    const omp = await buildProvider({ provider: 'omp', ompBin: '/opt/omp' });

    expect(pi).toBeInstanceOf(PiProvider);
    expect(omp).toBeInstanceOf(OmpProvider);
  });
});
