import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import type { HarnessConfig, RawResult } from '../src/harness/types.js';
import type { HarnessProvider } from '../src/harness/providers/base.js';
import { createMetrics, createRawResult } from '../src/harness/types.js';
import { getOutputPath } from '../src/harness/schema.js';
import { HarnessRunner } from '../src/harness/runner.js';
import * as factory from '../src/harness/providers/factory.js';

const tempDirs: string[] = [];

function makeTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentfield-runner-'));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  for (const dir of tempDirs.splice(0, tempDirs.length)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

class MockProvider implements HarnessProvider {
  public callCount = 0;
  public lastPrompt: string | undefined;
  public lastOptions: Record<string, unknown> | undefined;

  public constructor(private readonly results: RawResult[] = [createRawResult({ result: 'default result' })]) {}

  public async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    this.callCount += 1;
    this.lastPrompt = prompt;
    this.lastOptions = options;
    if (this.callCount <= this.results.length) {
      return this.results[this.callCount - 1];
    }
    return createRawResult({ result: 'default result' });
  }
}

class FileWritingProvider extends MockProvider {
  public readonly outputPaths: string[] = [];

  public constructor(private readonly payload: string, result?: RawResult) {
    super([result ?? createRawResult({ result: 'ok' })]);
  }

  public override async execute(prompt: string, options: Record<string, unknown>): Promise<RawResult> {
    const match = prompt.match(/(\S*\.agentfield_output\.json)/);
    if (!match) {
      throw new Error('schema prompt did not contain an output path');
    }
    this.outputPaths.push(match[1]);
    fs.writeFileSync(match[1], this.payload, 'utf8');
    return super.execute(prompt, options);
  }
}

describe('harness runner', () => {
  it('resolveOptions merges config with per-call overrides', () => {
    const cfg: HarnessConfig = {
      provider: 'codex',
      model: 'sonnet',
      maxTurns: 30,
      maxBudgetUsd: 2,
      tools: ['Read'],
      permissionMode: 'plan',
      systemPrompt: 'base',
      env: { A: '1' },
      cwd: '/tmp/base',
      projectDir: '/tmp/project',
      aforgeBin: 'aforge',
      codexBin: 'codex',
      geminiBin: 'gemini',
      opencodeBin: 'opencode',
    };

    const runner = new HarnessRunner(cfg);
    const options = runner.resolveOptions(cfg, {
      model: 'gpt-4.1',
      maxTurns: 10,
      env: { B: '2' },
      cwd: '/tmp/override',
      maxBudgetUsd: undefined,
    });

    expect(options.provider).toBe('codex');
    expect(options.model).toBe('gpt-4.1');
    expect(options.maxTurns).toBe(10);
    expect(options.maxBudgetUsd).toBe(2);
    expect(options.env).toEqual({ B: '2' });
    expect(options.cwd).toBe('/tmp/override');
    expect(options.projectDir).toBe('/tmp/project');
    expect(options.aforgeBin).toBe('aforge');
  });

  it('isTransient matches transient errors and rejects non-transient', () => {
    const runner = new HarnessRunner();
    expect(runner.isTransient('HTTP 503 service unavailable')).toBe(true);
    expect(runner.isTransient('rate limit reached')).toBe(true);
    expect(runner.isTransient('connection reset by peer')).toBe(true);
    expect(runner.isTransient('validation failed')).toBe(false);
    expect(runner.isTransient('permission denied')).toBe(false);
  });

  it('resolveOptions threads variant from config with per-call override winning', () => {
    const cfg: HarnessConfig = {
      provider: 'opencode',
      model: 'openai/gpt-5',
      variant: 'low',
    };

    const runner = new HarnessRunner(cfg);
    expect(runner.resolveOptions(cfg, {}).variant).toBe('low');
    expect(runner.resolveOptions(cfg, { variant: 'max' }).variant).toBe('max');
  });

  it('run threads variant through to the provider options', async () => {
    const cwd = makeTempDir();
    const provider = new MockProvider();
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    await runner.run('hello', { provider: 'opencode', model: 'openai/gpt-5#low', variant: 'max', cwd });

    expect(provider.lastOptions?.model).toBe('openai/gpt-5#low');
    expect(provider.lastOptions?.variant).toBe('max');
  });

  it('run without schema returns plain HarnessResult', async () => {
    const cwd = makeTempDir();
    const provider = new MockProvider([
      createRawResult({
        result: 'done',
        failureType: 'none',
        returnCode: 0,
        metrics: createMetrics({ numTurns: 2, totalCostUsd: 0.42, sessionId: 'sess-1' }),
      }),
    ]);
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const result = await runner.run('hello', { provider: 'codex', cwd });

    expect(result.isError).toBe(false);
    expect(result.result).toBe('done');
    expect(result.parsed).toBeUndefined();
    expect(result.costUsd).toBe(0.42);
    expect(result.numTurns).toBe(2);
    expect(result.sessionId).toBe('sess-1');
    expect(result.failureType).toBe('none');
    expect(result.returnCode).toBe(0);
  });

  it('run with schema injects suffix and parses output', async () => {
    const cwd = makeTempDir();
    const schema = z.object({ name: z.string(), count: z.number() });
    const provider = new FileWritingProvider(JSON.stringify({ name: 'ok', count: 1 }));
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const result = await runner.run('produce json', { provider: 'codex', schema, cwd });

    expect(provider.lastPrompt).toContain('OUTPUT REQUIREMENTS');
    expect(provider.outputPaths).toHaveLength(1);
    expect(provider.lastPrompt).toContain(provider.outputPaths[0]);
    expect(provider.outputPaths[0]).toMatch(/\.agentfield-out-[^/]+\/\.agentfield_output\.json$/);
    expect(result.isError).toBe(false);
    expect(result.parsed).toEqual({ name: 'ok', count: 1 });
    expect(fs.existsSync(provider.outputPaths[0])).toBe(false);
  });

  it('isolates concurrent schema runs that share one cwd', async () => {
    const cwd = makeTempDir();
    const schema = z.object({ name: z.string(), count: z.number() });
    const outputPaths: string[] = [];
    const provider: HarnessProvider = {
      async execute(prompt: string): Promise<RawResult> {
        const match = prompt.match(/(\S*\.agentfield_output\.json)/);
        if (!match) {
          throw new Error('schema prompt did not contain an output path');
        }
        outputPaths.push(match[1]);
        const first = prompt.startsWith('first');
        await new Promise((resolve) => setTimeout(resolve, 10));
        fs.writeFileSync(match[1], JSON.stringify({ name: first ? 'first' : 'second', count: first ? 1 : 2 }));
        return createRawResult({ result: 'done' });
      },
    };
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const [first, second] = await Promise.all([
      runner.run('first', { provider: 'aforge', schema, cwd }),
      runner.run('second', { provider: 'aforge', schema, cwd }),
    ]);

    expect(first.parsed).toEqual({ name: 'first', count: 1 });
    expect(second.parsed).toEqual({ name: 'second', count: 2 });
    expect(new Set(outputPaths).size).toBe(2);
    expect(fs.readdirSync(cwd).filter((name) => name.startsWith('.agentfield-out-'))).toEqual([]);
    expect(fs.existsSync(getOutputPath(cwd))).toBe(false);
  });

  it('run defaults to aforge when no provider is configured', async () => {
    vi.stubEnv('AGENTFIELD_HARNESS_PROVIDER', '');
    const provider = new MockProvider();
    const buildProviderSpy = vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);
    const runner = new HarnessRunner();
    await runner.run('hello', {});

    expect(buildProviderSpy).toHaveBeenCalledWith(expect.objectContaining({ provider: 'aforge' }));
    expect(provider.lastOptions?.provider).toBe('aforge');
  });

  it('per-call provider wins over the runner default and environment', async () => {
    vi.stubEnv('AGENTFIELD_HARNESS_PROVIDER', 'gemini');
    const provider = new MockProvider();
    const buildProviderSpy = vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);
    const runner = new HarnessRunner({ provider: 'codex' });

    await runner.run('hello', { provider: 'opencode' });

    expect(buildProviderSpy).toHaveBeenCalledWith(expect.objectContaining({ provider: 'opencode' }));
    expect(provider.lastOptions?.provider).toBe('opencode');
  });

  it('retries on transient error then succeeds', async () => {
    const cwd = makeTempDir();
    const provider = new MockProvider([
      createRawResult({ isError: true, errorMessage: 'rate limit exceeded' }),
      createRawResult({ result: 'ok', metrics: createMetrics({ numTurns: 2 }) }),
    ]);
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const sleepSpy = vi.spyOn(globalThis, 'setTimeout');
    const runner = new HarnessRunner();
    const result = await runner.run('hello', {
      provider: 'codex',
      cwd,
      maxRetries: 3,
      initialDelay: 0.01,
      maxDelay: 1,
      backoffFactor: 2,
    });

    expect(result.isError).toBe(false);
    expect(result.result).toBe('ok');
    expect(provider.callCount).toBe(2);
    expect(sleepSpy).toHaveBeenCalled();
  });

  it('does not retry for non-transient errors', async () => {
    const cwd = makeTempDir();
    const provider = new MockProvider([
      createRawResult({ isError: true, errorMessage: 'validation failed' }),
      createRawResult({ result: 'should not happen' }),
    ]);
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const result = await runner.run('hello', { provider: 'codex', cwd, maxRetries: 3 });

    expect(result.isError).toBe(true);
    expect(result.errorMessage).toBe('validation failed');
    expect(provider.callCount).toBe(1);
  });

  it('schema validation failure returns isError=true', async () => {
    const cwd = makeTempDir();
    const schema = z.object({ name: z.string(), count: z.number() });
    const provider = new FileWritingProvider(JSON.stringify({ name: 'ok' }));
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    const result = await runner.run('produce bad json', { provider: 'codex', schema, cwd });

    expect(result.isError).toBe(true);
    expect(result.parsed).toBeUndefined();
    expect(result.errorMessage).toContain('Schema validation failed');
  });

  it('always cleans temp files even when provider execution fails', async () => {
    const cwd = makeTempDir();
    const largeSchema = {
      type: 'object',
      properties: {
        payload: {
          type: 'string',
          description: 'x'.repeat(20000),
        },
      },
    };

    const provider: HarnessProvider = {
      execute: async () => {
        throw new Error('boom');
      },
    };
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner();
    await expect(runner.run('trigger failure', { provider: 'codex', schema: largeSchema, cwd })).rejects.toThrow(
      /boom/
    );

    expect(fs.existsSync(path.join(cwd, '.agentfield_output.json'))).toBe(false);
    expect(fs.existsSync(path.join(cwd, '.agentfield_schema.json'))).toBe(false);
  });

  it('applies harness config defaults and per-call overrides in run', async () => {
    const cwd = makeTempDir();
    const cfg: HarnessConfig = {
      provider: 'codex',
      model: 'default-model',
      maxTurns: 30,
      maxBudgetUsd: 1.5,
      tools: ['Read', 'Write'],
      permissionMode: 'plan',
      systemPrompt: 'base system',
      env: { BASE: '1' },
      cwd,
    };
    const provider = new MockProvider([createRawResult({ result: 'ok' })]);
    vi.spyOn(factory, 'buildProvider').mockResolvedValue(provider);

    const runner = new HarnessRunner(cfg);
    await runner.run('hello', {
      model: 'override-model',
      maxTurns: 5,
      env: { OVERRIDE: '1' },
      permissionMode: 'auto',
    });

    expect(provider.lastOptions).toMatchObject({
      provider: 'codex',
      model: 'override-model',
      maxTurns: 5,
      maxBudgetUsd: 1.5,
      permissionMode: 'auto',
      systemPrompt: 'base system',
      env: { OVERRIDE: '1' },
    });
  });
});
