import type express from 'express';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExecutionContext } from '../src/context/ExecutionContext.js';
import type { ExecutionMetadata } from '../src/context/ExecutionContext.js';
import { Agent } from '../src/agent/Agent.js';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { DidManager } from '../src/did/DidManager.js';
import type { MemoryChangeEvent } from '../src/memory/MemoryInterface.js';
import { MemoryEventClient } from '../src/memory/MemoryEventClient.js';
import type { AgentRouter } from '../src/router/AgentRouter.js';

vi.mock('ws', () => {
  class MockWebSocket {
    readonly on = vi.fn();
    readonly removeAllListeners = vi.fn();
    readonly terminate = vi.fn();

    constructor(_url: string, _options?: unknown) {}
  }

  return { default: MockWebSocket };
});

type MockResponse = express.Response & {
  body?: unknown;
  statusCode: number;
};

type AgentInternals = {
  dispatchMemoryEvent(event: MemoryChangeEvent): void;
  registerWithControlPlane(): Promise<void>;
  waitForApproval(): Promise<void>;
  executeReasoner(req: express.Request, res: express.Response, name: string): Promise<void>;
  executeSkill(req: express.Request, res: express.Response, name: string): Promise<void>;
  executeServerlessHttp(req: express.Request, res: express.Response, explicitName?: string): Promise<void>;
};

function createAgent(config?: ConstructorParameters<typeof Agent>[0]) {
  return new Agent({
    nodeId: 'agent-test',
    agentFieldUrl: 'http://control-plane.local',
    didEnabled: false,
    devMode: true,
    ...config
  });
}

function createRequest(overrides?: Partial<express.Request>): express.Request {
  return {
    body: {},
    headers: {},
    params: {},
    path: '/execute',
    query: {},
    ip: '127.0.0.1',
    ...overrides
  } as express.Request;
}

function createResponse(): MockResponse {
  const response = {
    body: undefined,
    headersSent: false,
    statusCode: 200
  } as MockResponse;

  response.status = vi.fn((code: number) => {
    response.statusCode = code;
    return response;
  }) as unknown as express.Response['status'];

  response.json = vi.fn((body: unknown) => {
    response.body = body;
    response.headersSent = true;
    return response;
  }) as unknown as express.Response['json'];

  response.setHeader = vi.fn() as unknown as express.Response['setHeader'];

  return response;
}

describe('Agent runtime paths', () => {
  beforeEach(() => {
    vi.spyOn(AgentFieldClient.prototype, 'publishExecutionLogs').mockImplementation(() => {});
    vi.spyOn(AgentFieldClient.prototype, 'publishWorkflowEvent').mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('applies defaults, includes router registrations, and tracks realtime validation targets', () => {
    const agent = createAgent({ didEnabled: undefined, deploymentType: undefined });
    const router = {
      reasoners: [{ name: 'router/reasoner', handler: async () => ({ ok: true }) }],
      skills: [{ name: 'router-skill', handler: async () => ({ ok: true }) }]
    } as unknown as AgentRouter;

    agent.reasoner('live-reasoner', async () => ({ ok: true }), { requireRealtimeValidation: true });
    agent.skill('live-skill', async () => ({ ok: true }), { requireRealtimeValidation: true });
    agent.includeRouter(router);

    expect(agent.config.port).toBe(8001);
    expect(agent.config.host).toBe('0.0.0.0');
    expect(agent.config.didEnabled).toBe(true);
    expect(agent.config.deploymentType).toBe('long_running');
    expect(agent.reasoners.get('router/reasoner')).toBeDefined();
    expect(agent.skills.get('router-skill')).toBeDefined();
    expect((agent as unknown as { realtimeValidationFunctions: Set<string> }).realtimeValidationFunctions).toEqual(
      new Set(['live-reasoner', 'live-skill'])
    );
  });

  it('starts memory watching and dispatches matching scoped events only', () => {
    const agent = createAgent();
    const start = vi.spyOn(MemoryEventClient.prototype, 'start').mockImplementation(() => {});
    const watched: MemoryChangeEvent[] = [];

    agent.watchMemory(['session.*', 'workflow.done'], (event) => {
      watched.push(event);
    }, { scope: 'session', scopeId: 'session-1' });

    const internals = agent as unknown as AgentInternals;
    internals.dispatchMemoryEvent({
      key: 'session.user',
      operation: 'set',
      value: { ok: true },
      scope: 'session',
      scopeId: 'session-1'
    });
    internals.dispatchMemoryEvent({
      key: 'workflow.done',
      operation: 'set',
      value: { ok: true },
      scope: 'session',
      scopeId: 'session-1'
    });
    internals.dispatchMemoryEvent({
      key: 'session.user',
      operation: 'set',
      value: { ok: false },
      scope: 'workflow',
      scopeId: 'workflow-1'
    });

    expect(start).toHaveBeenCalledTimes(1);
    expect(watched).toHaveLength(2);
    expect(watched.map((event) => event.key)).toEqual(['session.user', 'workflow.done']);
  });

  it('registers with the control plane, waits for approval, and wires DID credentials', async () => {
    vi.useFakeTimers();

    const agent = createAgent({ didEnabled: true, tags: ['reviewed'] });
    const register = vi.spyOn(AgentFieldClient.prototype, 'register').mockResolvedValue({
      status: 'pending_approval',
      pending_tags: ['reviewed']
    });
    const getNode = vi.spyOn(AgentFieldClient.prototype, 'getNode')
      .mockResolvedValueOnce({ lifecycle_status: 'pending_approval' })
      .mockResolvedValueOnce({ lifecycle_status: 'approved' });
    const didRegister = vi.spyOn(DidManager.prototype, 'registerAgent').mockResolvedValue(true);
    vi.spyOn(DidManager.prototype, 'getIdentitySummary').mockReturnValue({
      agentDid: 'did:agent:123',
      reasonerCount: 1,
      skillCount: 0
    });
    vi.spyOn(DidManager.prototype, 'getIdentityPackage').mockReturnValue({
      agentDid: {
        did: 'did:agent:123',
        privateKeyJwk: 'private-jwk'
      },
      reasonerDids: {},
      skillDids: {},
      agentfieldServerId: 'srv-1'
    });
    const setDIDCredentials = vi.spyOn(AgentFieldClient.prototype, 'setDIDCredentials').mockImplementation(() => {});

    agent.reasoner('plan', async () => ({ ok: true }));

    const registering = (agent as unknown as AgentInternals).registerWithControlPlane();
    await vi.advanceTimersByTimeAsync(10_000);
    await registering;

    expect(register).toHaveBeenCalledWith(expect.objectContaining({
      id: 'agent-test',
      deployment_type: 'long_running',
      tags: ['reviewed'],
      proposed_tags: ['reviewed'],
      reasoners: [expect.objectContaining({ id: 'plan' })]
    }));
    expect(getNode).toHaveBeenCalledTimes(2);
    expect(didRegister).toHaveBeenCalledTimes(1);
    expect(setDIDCredentials).toHaveBeenCalledWith('did:agent:123', 'private-jwk');
  });

  it('waitForApproval times out after repeated pending statuses', async () => {
    vi.useFakeTimers();

    const agent = createAgent();
    vi.spyOn(AgentFieldClient.prototype, 'getNode').mockResolvedValue({ lifecycle_status: 'pending_approval' });

    const pending = expect((agent as unknown as AgentInternals).waitForApproval()).rejects.toThrow('approval timed out');
    await vi.advanceTimersByTimeAsync(5 * 60 * 1000);
    await pending;
  });

  it('handler() returns discovery payload for discover actions', async () => {
    const agent = createAgent({ version: '1.0.0', deploymentType: 'serverless' });

    const result = await agent.handler()({
      path: '/discover'
    });

    expect(result).toEqual({
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: {
        node_id: 'agent-test',
        version: '1.0.0',
        deployment_type: 'serverless',
        reasoners: [],
        skills: []
      }
    });
  });

  it('handler() normalizes serverless input, merges execution context, and executes reasoners and skills', async () => {
    const agent = createAgent();
    const reasonerCalls: ExecutionMetadata[] = [];
    const skillCalls: ExecutionMetadata[] = [];

    vi.spyOn(DidManager.prototype, 'getAgentDid').mockReturnValue('did:agent:self');
    vi.spyOn(DidManager.prototype, 'getFunctionDid').mockImplementation((name) => `did:function:${name}`);

    agent.reasoner('plan/work', async (ctx) => {
      reasonerCalls.push({
        executionId: ctx.executionId,
        runId: ctx.runId,
        workflowId: ctx.workflowId,
        rootWorkflowId: ctx.rootWorkflowId,
        parentExecutionId: ctx.parentExecutionId,
        sessionId: ctx.sessionId,
        actorId: ctx.actorId,
        reasonerId: ctx.reasonerId,
        callerDid: ctx.did.metadata.callerDid,
        targetDid: ctx.did.metadata.targetDid,
        agentNodeDid: ctx.did.metadata.agentNodeDid
      });
      return { seen: ctx.input, did: ctx.did.metadata.targetDid };
    });
    agent.skill('format', async (ctx) => {
      skillCalls.push({
        executionId: ctx.executionId,
        reasonerId: ctx.reasonerId,
        workflowId: ctx.workflowId,
        rootWorkflowId: ctx.rootWorkflowId
      });
      return { formatted: ctx.input.text.toUpperCase() };
    });

    const reasonerResult = await agent.handler()({
      path: '/execute',
      target: 'plan/work',
      body: JSON.stringify({ input: { text: 'hello' } }),
      headers: { 'x-session-id': 'session-1' },
      execution_context: {
        execution_id: 'exec-1',
        run_id: 'run-1',
        workflow_id: 'wf-1',
        root_workflow_id: 'root-1',
        parent_execution_id: 'parent-1',
        actor_id: 'actor-1',
        caller_did: 'did:caller:1'
      }
    });
    const skillResult = await agent.handler()({
      path: '/execute',
      body: JSON.stringify({ skill: 'format', data: { text: 'hello' } }),
      headers: {}
    });

    expect(reasonerResult).toEqual({
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: { seen: { text: 'hello' }, did: 'did:function:plan/work' }
    });
    expect(skillResult).toEqual({
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: { formatted: 'HELLO' }
    });
    expect(reasonerCalls).toEqual([{
      executionId: 'exec-1',
      runId: 'run-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      sessionId: 'session-1',
      actorId: 'actor-1',
      reasonerId: 'plan/work',
      callerDid: 'did:caller:1',
      targetDid: 'did:function:plan/work',
      agentNodeDid: 'did:agent:self'
    }]);
    expect(skillCalls[0]?.reasonerId).toBe('format');
    expect(skillCalls[0]?.workflowId).toBeTruthy();
    expect(skillCalls[0]?.rootWorkflowId).toBe(skillCalls[0]?.workflowId);
  });

  it('handler() returns 400 for missing targets, 404 for missing handlers, and 500 for thrown errors', async () => {
    const agent = createAgent();
    agent.reasoner('boom', async () => {
      throw new Error('exploded');
    });

    await expect(agent.handler()({ path: '/execute', body: '{}' })).resolves.toEqual({
      statusCode: 400,
      headers: { 'content-type': 'application/json' },
      body: { error: "Missing 'target' or 'reasoner' in request" }
    });

    await expect(agent.handler()({ path: '/execute/missing', body: '{}' })).resolves.toEqual({
      statusCode: 404,
      headers: { 'content-type': 'application/json' },
      body: { error: 'Reasoner not found: missing' }
    });

    await expect(agent.handler()({ path: '/execute/boom', body: '{}' })).resolves.toEqual({
      statusCode: 500,
      headers: { 'content-type': 'application/json' },
      body: { error: 'exploded' }
    });
  });

  it('executeReasoner, executeSkill, and executeServerlessHttp handle inbound request success and error responses', async () => {
    const agent = createAgent();

    agent.reasoner('plan', async (ctx) => ({ ok: ctx.input.value }));
    agent.skill('format', async (ctx) => ({ text: ctx.input.text.toUpperCase() }));
    agent.reasoner('forbidden', async () => {
      const error = new Error('not allowed') as Error & { status: number; responseData: { code: string } };
      error.status = 403;
      error.responseData = { code: 'policy_denied' };
      throw error;
    });

    const reasonerRes = createResponse();
    await (agent as unknown as AgentInternals).executeReasoner(
      createRequest({ body: { value: 7 }, headers: { 'x-run-id': 'run-2' } }),
      reasonerRes,
      'plan'
    );

    const skillRes = createResponse();
    await (agent as unknown as AgentInternals).executeSkill(
      createRequest({ body: { text: 'hi' }, path: '/skills/format' }),
      skillRes,
      'format'
    );

    const executeRes = createResponse();
    await (agent as unknown as AgentInternals).executeServerlessHttp(
      createRequest({ body: { reasoner: 'forbidden' }, path: '/execute' }),
      executeRes
    );

    const missingRes = createResponse();
    await (agent as unknown as AgentInternals).executeServerlessHttp(
      createRequest({ body: {}, path: '/execute' }),
      missingRes
    );

    expect(reasonerRes.body).toEqual({ ok: 7 });
    expect(skillRes.body).toEqual({ text: 'HI' });
    expect(executeRes.statusCode).toBe(403);
    expect(executeRes.body).toEqual({
      error: 'not allowed',
      error_details: { code: 'policy_denied' }
    });
    expect(missingRes.statusCode).toBe(400);
    expect(missingRes.body).toEqual({ error: "Missing 'target' or 'reasoner' in request" });
  });

  it('call() executes local reasoners, publishes workflow states, and propagates failures', async () => {
    const agent = createAgent({ nodeId: 'agent-1' });
    const publishWorkflowEvent = vi.spyOn(AgentFieldClient.prototype, 'publishWorkflowEvent').mockResolvedValue(undefined);

    agent.reasoner('nested/path', async (ctx) => ({
      input: ctx.input,
      executionId: ctx.executionId,
      reasonerId: ctx.reasonerId
    }));
    agent.reasoner('explode', async () => {
      throw new Error('call failed');
    });

    const result = await ExecutionContext.run(new ExecutionContext({
      input: { parent: true },
      metadata: {
        executionId: 'parent-exec',
        runId: 'run-9',
        workflowId: 'wf-9',
        rootWorkflowId: 'root-9',
        sessionId: 'session-9',
        actorId: 'actor-9'
      },
      req: {} as express.Request,
      res: {} as express.Response,
      agent
    }), async () => agent.call('agent-1.nested:path', { ok: true }));

    await expect(agent.call('agent-1.explode', {})).rejects.toThrow('call failed');

    expect(result).toMatchObject({
      input: { ok: true },
      reasonerId: 'nested/path'
    });
    expect(publishWorkflowEvent).toHaveBeenCalledWith(expect.objectContaining({
      reasonerId: 'nested/path',
      agentNodeId: 'agent-1',
      status: 'running'
    }));
    expect(publishWorkflowEvent).toHaveBeenCalledWith(expect.objectContaining({
      reasonerId: 'nested/path',
      agentNodeId: 'agent-1',
      status: 'succeeded',
      result: expect.objectContaining({ input: { ok: true } })
    }));
    expect(publishWorkflowEvent).toHaveBeenCalledWith(expect.objectContaining({
      reasonerId: 'explode',
      agentNodeId: 'agent-1',
      status: 'failed',
      error: 'call failed'
    }));
  });

  it('call() delegates remote targets and note() emits only when execution metadata exists', async () => {
    const agent = createAgent({ nodeId: 'agent-1', agentFieldUrl: 'http://control-plane.local/api/v1' });
    const execute = vi.spyOn(AgentFieldClient.prototype, 'execute').mockResolvedValue({ remote: true });
    const sendNote = vi.spyOn(AgentFieldClient.prototype, 'sendNote').mockImplementation(() => {});

    const result = await ExecutionContext.run(new ExecutionContext({
      input: {},
      metadata: {
        executionId: 'exec-remote',
        runId: 'run-remote',
        workflowId: 'wf-remote',
        rootWorkflowId: 'root-remote',
        callerDid: 'did:caller:remote',
        targetDid: 'did:target:remote',
        agentNodeDid: 'did:agent:remote'
      },
      req: {} as express.Request,
      res: {} as express.Response,
      agent
    }), async () => {
      agent.note('during-execution', ['debug']);
      return agent.call('other-agent.plan', { hello: 'world' });
    });

    agent.note('outside-execution', ['ignored']);

    expect(result).toEqual({ remote: true });
    expect(execute).toHaveBeenCalledWith('other-agent.plan', { hello: 'world' }, expect.objectContaining({
      runId: 'run-remote',
      workflowId: 'wf-remote',
      rootWorkflowId: 'root-remote',
      reasonerId: 'plan',
      callerDid: 'did:caller:remote',
      targetDid: 'did:target:remote',
      agentNodeDid: 'did:agent:remote',
      agentNodeId: 'agent-1'
    }));
    expect(sendNote).toHaveBeenCalledTimes(1);
    expect(sendNote).toHaveBeenCalledWith(
      'during-execution',
      ['debug'],
      'agent-1',
      expect.objectContaining({ executionId: 'exec-remote' }),
      'http://control-plane.local/api/ui/v1',
      true
    );
  });
});
