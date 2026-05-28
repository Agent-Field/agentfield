import type express from 'express';
import { describe, expect, it, vi, type Mock } from 'vitest';

const { buildToolConfigMock, executeToolCallLoopMock } = vi.hoisted(() => ({
  buildToolConfigMock: vi.fn(),
  executeToolCallLoopMock: vi.fn()
}));

vi.mock('../src/ai/ToolCalling.js', async () => {
  const actual = await vi.importActual('../src/ai/ToolCalling.js');
  return {
    ...actual,
    buildToolConfig: buildToolConfigMock,
    executeToolCallLoop: executeToolCallLoopMock
  };
});

import { ExecutionContext, type ExecutionMetadata } from '../src/context/ExecutionContext.js';
import { ReasonerContext, getCurrentContext } from '../src/context/ReasonerContext.js';
import { SkillContext, getCurrentSkillContext } from '../src/context/SkillContext.js';

type AgentLike = {
  call: Mock;
  discover: Mock;
  note: Mock;
  getExecutionLogger: Mock;
  getAIClient: Mock;
  getMemoryInterface: Mock;
  getWorkflowReporter: Mock;
  getDidInterface: Mock;
};

function makeAgent() {
  const logger = { name: 'logger' };
  const aiClient = {
    generate: vi.fn(),
    stream: vi.fn(),
    getModel: vi.fn(() => 'model-1')
  };
  const memory = { name: 'memory' };
  const workflow = { name: 'workflow' };
  const did = { name: 'did' };

  const agent: AgentLike = {
    call: vi.fn(),
    discover: vi.fn(),
    note: vi.fn(),
    getExecutionLogger: vi.fn(() => logger),
    getAIClient: vi.fn(() => aiClient),
    getMemoryInterface: vi.fn(() => memory),
    getWorkflowReporter: vi.fn(() => workflow),
    getDidInterface: vi.fn(() => did)
  };

  return { agent, logger, aiClient, memory, workflow, did };
}

function makeExecutionContext(agent: AgentLike, metadata: ExecutionMetadata = { executionId: 'exec-1' }) {
  return new ExecutionContext({
    input: { prompt: 'hi' },
    metadata,
    req: { id: 'req-1' } as unknown as express.Request,
    res: { id: 'res-1' } as unknown as express.Response,
    agent: agent as never
  });
}

describe('ReasonerContext additional coverage', () => {
  it('delegates ai(), aiStream(), call(), discover(), and note() to the underlying services', async () => {
    const { agent, aiClient, logger, memory, workflow, did } = makeAgent();
    aiClient.generate.mockResolvedValue('generated');
    aiClient.stream.mockResolvedValue({ stream: true });
    agent.call.mockResolvedValue({ ok: true });
    agent.discover.mockResolvedValue({ format: 'json' });
    const ctx = new ReasonerContext({
      input: { prompt: 'hi' },
      executionId: 'exec-1',
      runId: 'run-1',
      sessionId: 'session-1',
      actorId: 'actor-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      reasonerId: 'planner',
      callerDid: 'did:key:caller',
      targetDid: 'did:key:target',
      agentNodeDid: 'did:key:node',
      req: {} as express.Request,
      res: {} as express.Response,
      agent: agent as never,
      logger: logger as never,
      aiClient: aiClient as never,
      memory: memory as never,
      workflow: workflow as never,
      did: did as never
    });

    await expect(ctx.ai('prompt')).resolves.toBe('generated');
    await expect(ctx.aiStream('prompt')).resolves.toEqual({ stream: true });
    await expect(ctx.call('node.plan', { x: 1 })).resolves.toEqual({ ok: true });
    await expect(ctx.discover({ tags: ['a'] })).resolves.toEqual({ format: 'json' });

    ctx.note('remember this', ['tag-1']);

    expect(aiClient.generate).toHaveBeenCalledWith('prompt', undefined);
    expect(aiClient.stream).toHaveBeenCalledWith('prompt', undefined);
    expect(agent.call).toHaveBeenCalledWith('node.plan', { x: 1 });
    expect(agent.discover).toHaveBeenCalledWith({ tags: ['a'] });
    expect(agent.note).toHaveBeenCalledWith('remember this', ['tag-1'], {
      executionId: 'exec-1',
      runId: 'run-1',
      sessionId: 'session-1',
      actorId: 'actor-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      reasonerId: 'planner',
      callerDid: 'did:key:caller',
      targetDid: 'did:key:target',
      agentNodeDid: 'did:key:node'
    });
  });

  it('aiWithTools() builds tool config, merges limits, and passes the model getter through', async () => {
    const { agent, aiClient, logger, memory, workflow, did } = makeAgent();
    buildToolConfigMock.mockResolvedValue({
      tools: { a: { description: 'tool' } },
      config: { maxTurns: 3, maxToolCalls: 4, schemaHydration: 'lazy' },
      needsLazyHydration: true
    });
    executeToolCallLoopMock.mockResolvedValue({ text: 'done', trace: { calls: [], totalTurns: 1, totalToolCalls: 0 } });
    const ctx = new ReasonerContext({
      input: {},
      executionId: 'exec-1',
      req: {} as express.Request,
      res: {} as express.Response,
      agent: agent as never,
      logger: logger as never,
      aiClient: aiClient as never,
      memory: memory as never,
      workflow: workflow as never,
      did: did as never
    });

    const result = await ctx.ai('use tools', { tools: 'discover', maxTurns: 8, maxToolCalls: 9, temperature: 0.2 });

    expect(result).toEqual({ text: 'done', trace: { calls: [], totalTurns: 1, totalToolCalls: 0 } });
    expect(buildToolConfigMock).toHaveBeenCalledWith('discover', agent);
    expect(executeToolCallLoopMock).toHaveBeenCalledTimes(1);
    const args = executeToolCallLoopMock.mock.calls[0] as unknown[];
    expect(args[0]).toBe(agent);
    expect(args[1]).toBe('use tools');
    expect(args[2]).toEqual({ a: { description: 'tool' } });
    expect(args[3]).toEqual({ maxTurns: 8, maxToolCalls: 9, schemaHydration: 'lazy' });
    expect(args[4]).toBe(true);
    expect((args[5] as () => string)()).toBe('model-1');
    expect(args[6]).toEqual({ tools: 'discover', maxTurns: 8, maxToolCalls: 9, temperature: 0.2 });
    expect(aiClient.getModel).toHaveBeenCalledWith({ tools: 'discover', maxTurns: 8, maxToolCalls: 9, temperature: 0.2 });
  });

  it('getCurrentContext() reflects the active execution lifecycle and hydrates dependencies from the agent', async () => {
    const { agent, logger, aiClient, memory, workflow, did } = makeAgent();
    const metadata: ExecutionMetadata = {
      executionId: 'exec-ctx',
      runId: 'run-1',
      sessionId: 'session-1',
      actorId: 'actor-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      reasonerId: 'planner',
      callerDid: 'did:key:caller',
      targetDid: 'did:key:target',
      agentNodeDid: 'did:key:node'
    };
    const execution = makeExecutionContext(agent, metadata);

    expect(getCurrentContext()).toBeUndefined();

    await ExecutionContext.run(execution, async () => {
      const current = getCurrentContext<{ prompt: string }>();
      expect(current).toBeInstanceOf(ReasonerContext);
      expect(current?.input).toEqual({ prompt: 'hi' });
      expect(current?.executionId).toBe('exec-ctx');
      expect(current?.runId).toBe('run-1');
      expect(current?.sessionId).toBe('session-1');
      expect(current?.actorId).toBe('actor-1');
      expect(current?.workflowId).toBe('wf-1');
      expect(current?.rootWorkflowId).toBe('root-1');
      expect(current?.parentExecutionId).toBe('parent-1');
      expect(current?.reasonerId).toBe('planner');
      expect(current?.callerDid).toBe('did:key:caller');
      expect(current?.targetDid).toBe('did:key:target');
      expect(current?.agentNodeDid).toBe('did:key:node');
      expect(current?.req).toBe(execution.req);
      expect(current?.res).toBe(execution.res);
      expect(current?.agent).toBe(agent);
      expect(current?.logger).toBe(logger);
      expect(current?.aiClient).toBe(aiClient);
      expect(current?.memory).toBe(memory);
      expect(current?.workflow).toBe(workflow);
      expect(current?.did).toBe(did);
    });

    expect(agent.getExecutionLogger).toHaveBeenCalled();
    expect(agent.getAIClient).toHaveBeenCalled();
    expect(agent.getMemoryInterface).toHaveBeenCalledWith(metadata);
    expect(agent.getWorkflowReporter).toHaveBeenCalledWith(metadata);
    expect(agent.getDidInterface).toHaveBeenCalledWith(metadata, { prompt: 'hi' });
    expect(getCurrentContext()).toBeUndefined();
  });
});

describe('SkillContext additional coverage', () => {
  it('delegates discover() and resolves the current skill context from the execution store', async () => {
    const { agent, logger, memory, workflow, did } = makeAgent();
    agent.discover.mockResolvedValue({ format: 'compact' });
    const execution = makeExecutionContext(agent, {
      executionId: 'exec-skill',
      sessionId: 'session-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      reasonerId: 'skill-router',
      callerDid: 'did:key:caller',
      agentNodeDid: 'did:key:node'
    });

    const direct = new SkillContext({
      input: { prompt: 'hi' },
      executionId: 'exec-skill',
      sessionId: 'session-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      reasonerId: 'skill-router',
      callerDid: 'did:key:caller',
      agentNodeDid: 'did:key:node',
      req: {} as express.Request,
      res: {} as express.Response,
      agent: agent as never,
      logger: logger as never,
      memory: memory as never,
      workflow: workflow as never,
      did: did as never
    });

    await expect(direct.discover({ tags: ['skill'] })).resolves.toEqual({ format: 'compact' });
    expect(getCurrentSkillContext()).toBeUndefined();

    await ExecutionContext.run(execution, async () => {
      const current = getCurrentSkillContext<{ prompt: string }>();
      expect(current).toBeInstanceOf(SkillContext);
      expect(current?.input).toEqual({ prompt: 'hi' });
      expect(current?.executionId).toBe('exec-skill');
      expect(current?.sessionId).toBe('session-1');
      expect(current?.workflowId).toBe('wf-1');
      expect(current?.rootWorkflowId).toBe('root-1');
      expect(current?.reasonerId).toBe('skill-router');
      expect(current?.callerDid).toBe('did:key:caller');
      expect(current?.agentNodeDid).toBe('did:key:node');
      expect(current?.logger).toBe(logger);
      expect(current?.memory).toBe(memory);
      expect(current?.workflow).toBe(workflow);
      expect(current?.did).toBe(did);
    });

    expect(getCurrentSkillContext()).toBeUndefined();
  });
});
