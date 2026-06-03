import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

type AxiosMockInstance = {
  post: Mock;
  get: Mock;
};

type AxiosResponseError = Error & {
  response?: {
    status: number;
    data?: unknown;
  };
};

const { createMock, createdInstances } = vi.hoisted(() => {
  const instances: AxiosMockInstance[] = [];
  const create: Mock = vi.fn(() => {
    const instance: AxiosMockInstance = {
      post: vi.fn(),
      get: vi.fn()
    };
    instances.push(instance);
    return instance;
  });

  return {
    createMock: create,
    createdInstances: instances
  };
});

vi.mock('axios', () => ({
  default: { create: createMock },
  create: createMock
}));

import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import {
  HEADER_CALLER_DID,
  HEADER_DID_NONCE,
  HEADER_DID_SIGNATURE,
  HEADER_DID_TIMESTAMP
} from '../src/client/DIDAuthenticator.js';

const TEST_DID = 'did:key:z6MkiH8o2J7v6h8o2J7v6h8o2J7v6h8o2J7v6h8o2J7v6h8o';
const TEST_JWK = JSON.stringify({
  kty: 'OKP',
  crv: 'Ed25519',
  d: Buffer.alloc(32, 7).toString('base64url')
});

function getHttp(): AxiosMockInstance {
  const http = createdInstances.at(-1);
  if (!http) {
    throw new Error('Expected axios.create() to have produced an instance');
  }
  return http;
}

function makeResponseError(status: number, data?: Record<string, unknown>): AxiosResponseError {
  const err = new Error(`Request failed with status ${status}`) as AxiosResponseError;
  err.response = { status, data };
  return err;
}

describe('AgentFieldClient', () => {
  beforeEach(() => {
    createMock.mockClear();
    createdInstances.length = 0;
  });

  it('creates an axios client with the trimmed base URL and default timeout', () => {
    new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local/',
      defaultHeaders: { 'X-Tenant-ID': 'tenant-1' }
    });

    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: 'http://control-plane.local',
        timeout: 30000
      })
    );
  });

  it('register() POSTs the JSON payload with a content-type header', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: { ok: true } });

    const payload = {
      id: 'node-1',
      version: '1.0.0',
      skills: [],
      reasoners: []
    };

    await expect(client.register(payload)).resolves.toEqual({ ok: true });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/nodes/register',
      JSON.stringify(payload),
      {
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      }
    );
  });

  it('heartbeat() POSTs the node status payload to the node heartbeat path', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      version: '1.2.3',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: { status: 'degraded' } });

    await client.heartbeat('degraded');

    expect(http.post).toHaveBeenCalledTimes(1);
    const [path, body, config] = http.post.mock.calls[0];
    expect(path).toBe('/api/v1/nodes/node-1/heartbeat');
    expect(config).toEqual({
      headers: expect.objectContaining({
        'Content-Type': 'application/json'
      })
    });
    expect(JSON.parse(body as string)).toMatchObject({
      status: 'degraded',
      version: '1.2.3'
    });
    expect(typeof JSON.parse(body as string).timestamp).toBe('string');
  });

  it('execute() POSTs to the target path and forwards execution metadata as headers', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      defaultHeaders: { Authorization: 'Bearer tenant-token' }
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: { result: { ok: true } } });

    const result = await client.execute('agent.name:plan', { prompt: 'hi' }, {
      runId: 'run-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      parentExecutionId: 'parent-1',
      reasonerId: 'plan',
      sessionId: 'session-1',
      actorId: 'actor-1',
      callerDid: 'did:key:caller',
      targetDid: 'did:key:target',
      agentNodeDid: 'did:key:node',
      agentNodeId: 'node-1'
    });

    expect(result).toEqual({ ok: true });
    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/execute/agent.name:plan',
      JSON.stringify({ input: { prompt: 'hi' } }),
      {
        headers: expect.objectContaining({
          Authorization: 'Bearer tenant-token',
          'Content-Type': 'application/json',
          'X-Run-ID': 'run-1',
          'X-Workflow-ID': 'wf-1',
          'X-Root-Workflow-ID': 'root-1',
          'X-Parent-Execution-ID': 'parent-1',
          'X-Reasoner-ID': 'plan',
          'X-Session-ID': 'session-1',
          'X-Actor-ID': 'actor-1',
          'X-Caller-DID': 'did:key:caller',
          'X-Target-DID': 'did:key:target',
          'X-Agent-Node-DID': 'did:key:node',
          'X-Agent-Node-ID': 'node-1'
        })
      }
    );
  });

  it.each([
    [{ message: 'permission denied' }, 'permission denied'],
    [{ error: 'bad target' }, 'bad target']
  ])('execute() surfaces structured response errors from the control plane', async (body, expectedMessage) => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockRejectedValue(makeResponseError(403, body));

    await expect(client.execute('remote.plan', { foo: 'bar' })).rejects.toThrow(
      `execute remote.plan failed (403): ${expectedMessage}`
    );
  });

  it('execute() includes the 5xx status in structured server errors', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockRejectedValue(makeResponseError(500, { error: 'control plane unavailable' }));

    await expect(client.execute('remote.plan', { foo: 'bar' })).rejects.toThrow(
      'execute remote.plan failed (500): control plane unavailable'
    );
  });

  it('execute() re-throws transport failures without a control-plane response body', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    const networkError = new Error('socket hang up');
    http.post.mockRejectedValue(networkError);

    await expect(client.execute('remote.plan', { foo: 'bar' })).rejects.toBe(networkError);
  });

  it('attaches DID signing headers to register, heartbeat, and execute requests when credentials are configured', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      did: TEST_DID,
      privateKeyJwk: TEST_JWK
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: { ok: true, result: { ok: true } } });

    await client.register({ id: 'node-1' });
    await client.heartbeat('ready');
    await client.execute('remote.plan', { foo: 'bar' });

    expect(http.post).toHaveBeenCalledTimes(3);
    for (const [, , config] of http.post.mock.calls) {
      const headers = (config as { headers: Record<string, string> }).headers;
      expect(headers[HEADER_CALLER_DID]).toBe(TEST_DID);
      expect(headers[HEADER_DID_SIGNATURE]).toEqual(expect.any(String));
      expect(headers[HEADER_DID_TIMESTAMP]).toEqual(expect.any(String));
      expect(headers[HEADER_DID_NONCE]).toEqual(expect.any(String));
    }
  });

  it('uses shorter dev-mode timeouts for workflow events and execution logs', async () => {
    const devClient = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      devMode: true
    });
    const devHttp = getHttp();
    devHttp.post.mockResolvedValue({ data: { ok: true } });

    await devClient.publishWorkflowEvent({
      executionId: 'exec-1',
      runId: 'run-1',
      reasonerId: 'plan',
      agentNodeId: 'node-1',
      status: 'running'
    });
    devClient.publishExecutionLogs({
      v: 1,
      ts: '2026-04-07T00:00:00.000Z',
      execution_id: 'exec-1',
      level: 'info',
      source: 'sdk.test',
      message: 'hello'
    });

    expect(devHttp.post).toHaveBeenNthCalledWith(
      1,
      '/api/v1/workflow/executions/events',
      expect.any(String),
      expect.objectContaining({ timeout: 1000 })
    );
    expect(devHttp.post).toHaveBeenNthCalledWith(
      2,
      '/api/v1/executions/exec-1/logs',
      expect.any(String),
      expect.objectContaining({ timeout: 1000 })
    );

    const prodClient = new AgentFieldClient({
      nodeId: 'node-2',
      agentFieldUrl: 'http://control-plane.local'
    });
    const prodHttp = getHttp();
    prodHttp.post.mockResolvedValue({ data: { ok: true } });

    prodClient.publishExecutionLogs({
      v: 1,
      ts: '2026-04-07T00:00:00.000Z',
      execution_id: 'exec-2',
      level: 'info',
      source: 'sdk.test',
      message: 'hello'
    });

    expect(prodHttp.post).toHaveBeenCalledWith(
      '/api/v1/executions/exec-2/logs',
      expect.any(String),
      expect.objectContaining({ timeout: 5000 })
    );
  });
});
