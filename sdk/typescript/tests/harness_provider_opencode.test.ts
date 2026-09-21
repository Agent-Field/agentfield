import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

import { OpenCodeProvider } from '../src/harness/providers/opencode.js';
import { buildProvider } from '../src/harness/providers/factory.js';
import * as cli from '../src/harness/cli.js';

const workerInstruction =
  'You are an AgentField-launched worker. Complete the assigned prompt directly. ' +
  'Do not invoke AgentField orchestration, the `af` CLI, `swe-planner.plan`, ' +
  'or delegate work back to AgentField.';

function latestInvocation(): { cmd: string[]; env: Record<string, string> } {
  const calls = vi.mocked(cli.runCli).mock.calls;
  const call = calls[calls.length - 1];
  if (!call) {
    throw new Error('runCli was not called');
  }
  return { cmd: call[0], env: call[1]?.env ?? {} };
}

function latestConfig(): Record<string, any> {
  return JSON.parse(latestInvocation().env.OPENCODE_CONFIG_CONTENT) as Record<string, any>;
}

beforeEach(() => {
  vi.stubEnv('OPENCODE_CONFIG_CONTENT', '');
  vi.stubEnv('AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT', '');
  vi.stubEnv('AGENTFIELD_OPENCODE_STEPS', '');
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
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

    const { cmd, env } = latestInvocation();
    expect(cmd).toEqual([
      '/usr/local/bin/opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '--dir',
      '/tmp/work',
      'hello',
    ]);
    expect(env.A).toBe('1');
    const overlay = JSON.parse(env.OPENCODE_CONFIG_CONTENT);
    expect(overlay.$schema).toBe('https://opencode.ai/config.json');
    expect(overlay.default_agent).toBe('agentfield-harness');
    expect(overlay.agent['agentfield-harness']).toEqual({
      mode: 'primary',
      steps: 500,
      permission: {
        '*': 'allow',
        skill: { 'agentfield*': 'deny' },
        question: 'deny',
        task: 'deny',
      },
      prompt: workerInstruction,
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
    expect(result.failureType).toBe('crash');
  });

  it('returns a fallback when non-zero exit has no output', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr: '', exitCode: 1 });

    const result = await new OpenCodeProvider().execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toBe('Process exited with code 1 and produced no output.');
    expect(result.failureType).toBe('crash');
  });

  it.each(['Model not found: foo/bar', 'Unauthorized'])(
    'treats an empty successful result with known stderr failure %j as an error',
    async (stderr) => {
      vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: '', stderr, exitCode: 0 });

      const result = await new OpenCodeProvider().execute('hello', {});

      expect(result.isError).toBe(true);
      expect(result.errorMessage).toContain(stderr);
      expect(result.failureType).toBe('crash');
    },
  );

  it('strips ANSI before matching and surfaces the matching stderr window', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '',
      stderr: `migration prelude\n\u001b[31mModel not found: foo/bar\u001b[0m\ncontext`,
      exitCode: 0,
    });

    const result = await new OpenCodeProvider().execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toBe('migration prelude\nModel not found: foo/bar\ncontext');
  });

  it('lets a non-empty result win over stderr failure noise', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'answer',
      stderr: 'Unauthorized',
      exitCode: 0,
    });

    const result = await new OpenCodeProvider().execute('hello', {});

    expect(result.isError).toBe(false);
    expect(result.result).toBe('answer');
    expect(result.failureType).toBe('none');
  });

  it('does not treat unknown stderr with an empty successful result as an error', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '',
      stderr: 'one-time migration complete',
      exitCode: 0,
    });

    const result = await new OpenCodeProvider().execute('hello', {});

    expect(result.isError).toBe(false);
    expect(result.failureType).toBe('none');
  });

  it('reports a negative exit code as a signal crash', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: '',
      stderr: 'last diagnostic',
      exitCode: -9,
    });

    const result = await new OpenCodeProvider().execute('hello', {});

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toBe('Process killed by signal 9. stderr: last diagnostic');
    expect(result.failureType).toBe('crash');
  });

  it('passes model flag', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    const result = await provider.execute('hello', { model: 'openai/gpt-5' });

    expect(latestInvocation().cmd).toEqual([
      'opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '-m',
      'openai/gpt-5',
      'hello',
    ]);
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

    expect(latestInvocation().cmd).toEqual([
      'opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '-m',
      'openrouter/z-ai/glm-5.2',
      '--variant',
      'high',
      'hello',
    ]);
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

    expect(latestInvocation().cmd).toEqual([
      'opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '-m',
      'openai/gpt-5',
      '--variant',
      'max',
      'hello',
    ]);
  });

  it('passes no --variant flag for a bare model', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'deepseek/deepseek-v4-flash' });

    expect(latestInvocation().cmd).toEqual([
      'opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '-m',
      'deepseek/deepseek-v4-flash',
      'hello',
    ]);
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
      'X-OpenRouter-Categories': 'cli-agent,programming-app',
    });
    expect(overlay.agent['agentfield-harness']).toBeDefined();
  });

  it('does not add an OpenRouter attribution section for a non-OpenRouter model', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({
      stdout: 'ok\n',
      stderr: '',
      exitCode: 0,
    });

    const provider = new OpenCodeProvider();
    await provider.execute('hello', { model: 'openai/gpt-5' });

    const overlay = latestConfig();
    expect(overlay.provider).toBeUndefined();
    expect(overlay.agent['agentfield-harness']).toBeDefined();
  });

  it('configures the harness agent while keeping the task as the only positional prompt', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });

    const task = 'inspect the repository byte-for-byte';
    await new OpenCodeProvider().execute(task, {
      model: 'openai/gpt-5#low',
      variant: 'max',
      system_prompt: '  Work autonomously.  ',
      tools: ['Read', 'Write', 'Bash'],
      permission_mode: 'plan',
      max_turns: 3,
      env: { CUSTOM: 'kept' },
    });

    const { cmd, env } = latestInvocation();
    expect(cmd).toEqual([
      'opencode',
      'run',
      '--agent',
      'agentfield-harness',
      '-m',
      'openai/gpt-5',
      '--variant',
      'max',
      task,
    ]);
    expect(cmd.filter((part) => part === '--variant')).toHaveLength(1);
    expect(cmd).not.toContain(expect.stringContaining('SYSTEM INSTRUCTIONS:'));
    expect(env.CUSTOM).toBe('kept');

    const overlay = latestConfig();
    expect(overlay.$schema).toBe('https://opencode.ai/config.json');
    expect(overlay.default_agent).toBe('agentfield-harness');
    const agent = overlay.agent['agentfield-harness'];
    expect(agent).toEqual({
      mode: 'primary',
      steps: 500,
      permission: {
        '*': 'allow',
        skill: { 'agentfield*': 'deny' },
        question: 'deny',
        task: 'deny',
      },
      prompt: `Work autonomously.\n\n${workerInstruction}`,
      model: 'openai/gpt-5',
      reasoningEffort: 'max',
    });
    expect(agent).not.toHaveProperty('max_turns');
    expect(Object.keys(agent.permission)).toEqual(['*', 'skill', 'question', 'task']);
  });

  it('accepts the camelCase systemPrompt key that HarnessRunner passes', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });

    // HarnessRunner forwards HarnessOptions verbatim, so a system prompt set
    // through the public API arrives as `systemPrompt`, not `system_prompt`.
    await new OpenCodeProvider().execute('task', { systemPrompt: '  Work autonomously.  ' });

    expect(latestConfig().agent['agentfield-harness'].prompt).toBe(
      `Work autonomously.\n\n${workerInstruction}`,
    );
  });

  it('deep-merges per-call config, which wins over ambient config', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
    vi.stubEnv(
      'OPENCODE_CONFIG_CONTENT',
      JSON.stringify({ provider: { ambient: { model: 'ambient' } }, plugin: ['ambient'] }),
    );
    const callerConfig = JSON.stringify({
      provider: { 'keep-me': { npm: 'custom/provider' } },
      plugin: ['caller'],
      mcp: { local: { type: 'local', command: ['tool'] } },
      agent: {
        'custom-agent': { prompt: 'preserve this agent' },
        'agentfield-harness': {
          mode: 'secondary',
          temperature: 0.2,
          permission: { read: 'deny', skill: { 'other*': 'allow' } },
        },
      },
    });

    await new OpenCodeProvider().execute('hello', {
      model: 'openai/gpt-5',
      env: { OPENCODE_CONFIG_CONTENT: callerConfig },
    });

    const merged = latestConfig();
    expect(merged.provider.ambient).toBeUndefined();
    expect(merged.provider['keep-me']).toEqual({ npm: 'custom/provider' });
    expect(merged.plugin).toEqual(['caller']);
    expect(merged.mcp.local.command).toEqual(['tool']);
    expect(merged.agent['custom-agent']).toEqual({ prompt: 'preserve this agent' });
    const generatedAgent = merged.agent['agentfield-harness'];
    expect(generatedAgent.mode).toBe('primary');
    expect(generatedAgent.temperature).toBe(0.2);
    expect(generatedAgent.model).toBe('openai/gpt-5');
    expect(generatedAgent.permission).toEqual({
      '*': 'allow',
      read: 'deny',
      skill: { 'other*': 'allow', 'agentfield*': 'deny' },
      question: 'deny',
      task: 'deny',
    });
    expect(Object.keys(generatedAgent.permission)).toEqual([
      '*',
      'read',
      'skill',
      'question',
      'task',
    ]);
    expect(Object.keys(generatedAgent.permission.skill)).toEqual(['other*', 'agentfield*']);
  });

  it('deep-merges ambient config when no per-call config is supplied', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
    vi.stubEnv(
      'OPENCODE_CONFIG_CONTENT',
      JSON.stringify({ provider: { custom: { options: { baseURL: 'http://local' } } } }),
    );

    await new OpenCodeProvider().execute('hello', {});

    const merged = latestConfig();
    expect(merged.provider.custom.options.baseURL).toBe('http://local');
    expect(merged.default_agent).toBe('agentfield-harness');
  });

  it.each(['1', ' TrUe ', 'YES', 'on'])(
    'restores the inline prompt path for per-call opt-out value %j',
    async (optOut) => {
      vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
      const callerConfig = JSON.stringify({
        provider: { custom: { model: 'keep' } },
        agent: { 'agentfield-harness': { prompt: 'remove me', temperature: 0.2 } },
      });

      await new OpenCodeProvider().execute('complete the task', {
        model: 'openai/gpt-5#low',
        variant: 'max',
        system_prompt: '  caller instructions  ',
        tools: ['Read'],
        permission_mode: 'auto',
        env: {
          AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT: optOut,
          OPENCODE_CONFIG_CONTENT: callerConfig,
        },
      });

      const { cmd } = latestInvocation();
      expect(cmd).toEqual([
        'opencode',
        'run',
        '--agent',
        'agentfield-harness',
        '-m',
        'openai/gpt-5',
        '--variant',
        'max',
        `SYSTEM INSTRUCTIONS:\ncaller instructions\n\n${workerInstruction}\n\n---\n\nUSER REQUEST:\ncomplete the task`,
      ]);
      const merged = latestConfig();
      const agent = merged.agent['agentfield-harness'];
      expect(agent.prompt).toBeUndefined();
      expect(agent.temperature).toBe(0.2);
      expect(agent.permission).toEqual({
        '*': 'allow',
        skill: { 'agentfield*': 'deny' },
        question: 'deny',
        task: 'deny',
      });
    },
  );

  it('reads the inline opt-out from the ambient environment', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
    vi.stubEnv('AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT', ' yes ');

    await new OpenCodeProvider().execute('ambient task', {});

    expect(latestInvocation().cmd.at(-1)).toBe(
      `SYSTEM INSTRUCTIONS:\n${workerInstruction}\n\n---\n\nUSER REQUEST:\nambient task`,
    );
    expect(latestConfig().agent['agentfield-harness'].prompt).toBeUndefined();
  });

  it.each([
    ['37', 37],
    ['0', 500],
    ['-2', 500],
    ['not-a-number', 500],
  ])('validates the per-call steps override %j', async (value, expected) => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
    vi.stubEnv('AGENTFIELD_OPENCODE_STEPS', '91');

    await new OpenCodeProvider().execute('hello', {
      max_turns: 3,
      env: { AGENTFIELD_OPENCODE_STEPS: value },
    });

    expect(latestConfig().agent['agentfield-harness'].steps).toBe(expected);
  });

  it('reads the steps override from the ambient environment', async () => {
    vi.spyOn(cli, 'runCli').mockResolvedValue({ stdout: 'ok\n', stderr: '', exitCode: 0 });
    vi.stubEnv('AGENTFIELD_OPENCODE_STEPS', ' 73 ');

    await new OpenCodeProvider().execute('hello', { max_turns: 3 });

    expect(latestConfig().agent['agentfield-harness'].steps).toBe(73);
  });

  it.each(['{not-json', '[]', 'null', '"string"'])(
    'rejects malformed caller config %j before launching the child',
    async (content) => {
      const runCli = vi.spyOn(cli, 'runCli').mockResolvedValue({
        stdout: 'should not run',
        stderr: '',
        exitCode: 0,
      });

      await expect(
        new OpenCodeProvider().execute('hello', {
          env: { OPENCODE_CONFIG_CONTENT: content },
        }),
      ).rejects.toThrow(/OPENCODE_CONFIG_CONTENT/);
      expect(runCli).not.toHaveBeenCalled();
    },
  );
});

describe('provider factory', () => {
  it('routes opencode to OpenCodeProvider and passes opencodeBin', async () => {
    const provider = await buildProvider({ provider: 'opencode', opencodeBin: '/opt/opencode' });

    expect(provider).toBeInstanceOf(OpenCodeProvider);
  });
});
