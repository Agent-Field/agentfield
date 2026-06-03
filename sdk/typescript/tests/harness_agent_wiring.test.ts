import { describe, it, expect, vi } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { HarnessRunner } from '../src/harness/runner.js';

describe('Agent harness wiring', () => {
  const makeAgent = (harnessConfig?: Record<string, unknown>) =>
    new Agent({
      nodeId: 'test-agent',
      harnessConfig: harnessConfig as any,
    });

  it('accepts harnessConfig in constructor', () => {
    const agent = makeAgent({ provider: 'claude-code' });
    expect(agent.config.harnessConfig).toEqual({ provider: 'claude-code' });
  });

  it('harnessConfig defaults to undefined', () => {
    const agent = makeAgent();
    expect(agent.config.harnessConfig).toBeUndefined();
  });

  it('getHarnessRunner returns lazy-initialized HarnessRunner', async () => {
    const agent = makeAgent({ provider: 'codex' });
    const runner = await agent.getHarnessRunner();
    expect(runner).toBeInstanceOf(HarnessRunner);
    expect(await agent.getHarnessRunner()).toBe(runner);
  });

  it('getHarnessRunner works without harnessConfig', async () => {
    const agent = makeAgent();
    const runner = await agent.getHarnessRunner();
    expect(runner).toBeInstanceOf(HarnessRunner);
  });

  it('harness() delegates to runner.run()', async () => {
    const agent = makeAgent({ provider: 'claude-code' });
    const mockResult = {
      result: 'done',
      parsed: undefined,
      isError: false,
      numTurns: 1,
      durationMs: 50,
      sessionId: 'sess-1',
      messages: [],
      get text() { return this.result ?? ''; },
    };

    const runSpy = vi.spyOn(HarnessRunner.prototype, 'run').mockResolvedValue(mockResult as any);

    const result = await agent.harness('Do something', { provider: 'claude-code' });
    expect(result).toBe(mockResult);
    expect(runSpy).toHaveBeenCalledWith('Do something', { provider: 'claude-code' });

    runSpy.mockRestore();
  });

  it('harness() works with no options', async () => {
    const agent = makeAgent({ provider: 'claude-code' });
    const mockResult = {
      result: 'ok',
      isError: false,
      numTurns: 0,
      durationMs: 0,
      sessionId: '',
      messages: [],
      get text() { return this.result ?? ''; },
    };

    const runSpy = vi.spyOn(HarnessRunner.prototype, 'run').mockResolvedValue(mockResult as any);

    const result = await agent.harness('task');
    expect(result).toBe(mockResult);
    expect(runSpy).toHaveBeenCalledWith('task', {});

    runSpy.mockRestore();
  });
});
