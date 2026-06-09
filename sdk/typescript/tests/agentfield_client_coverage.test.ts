import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import {
  HEADER_CALLER_DID,
  HEADER_DID_NONCE,
  HEADER_DID_SIGNATURE,
  HEADER_DID_TIMESTAMP
} from '../src/client/DIDAuthenticator.js';

type AxiosMockInstance = {
  post: Mock;
  get: Mock;
};

const { createMock, postMock, getMock, createdInstances } = vi.hoisted(() => {
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
    postMock: vi.fn(),
    getMock: vi.fn(),
    createdInstances: instances
  };
});

vi.mock('axios', () => ({
  default: {
    create: createMock,
    post: postMock,
    get: getMock
  },
  create: createMock,
  post: postMock,
  get: getMock
}));

import { AgentFieldClient } from '../src/client/AgentFieldClient.js';

const TEST_DID = 'did:key:z6MkiH8o2J7v6h8o2J7v6h8o2J7v6h8o2J7v6h8o2J7v6h8o';
const TEST_JWK = JSON.stringify({
  kty: 'OKP',
  crv: 'Ed25519',
  d: Buffer.alloc(32, 9).toString('base64url')
});

function getHttp(): AxiosMockInstance {
  const http = createdInstances.at(-1);
  if (!http) {
    throw new Error('Expected axios.create() to return an instance');
  }
  return http;
}

describe('AgentFieldClient additional coverage', () => {
  beforeEach(() => {
    createMock.mockClear();
    postMock.mockReset();
    getMock.mockReset();
    createdInstances.length = 0;
  });

  it('getNode() GETs the encoded path and merges default headers with the API key', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      apiKey: 'api-key-1',
      defaultHeaders: {
        'X-Tenant-ID': 'tenant-1',
        'X-Number': 42 as unknown as string,
        'X-Ignore': undefined
      }
    });
    const http = getHttp();
    http.get.mockResolvedValue({ data: { id: 'node/1' } });

    await expect(client.getNode('node/1')).resolves.toEqual({ id: 'node/1' });

    expect(http.get).toHaveBeenCalledWith('/api/v1/nodes/node%2F1', {
      headers: {
        'X-Tenant-ID': 'tenant-1',
        'X-Number': '42',
        'X-API-Key': 'api-key-1'
      }
    });
  });

  it('discoverCapabilities() dedupes filters and maps JSON discovery responses', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      defaultHeaders: { Authorization: 'Bearer token' }
    });
    const http = getHttp();
    http.get.mockResolvedValue({
      data: {
        discovered_at: '2026-04-09T00:00:00Z',
        total_agents: 1,
        total_reasoners: 1,
        total_skills: 1,
        pagination: {
          limit: 10,
          offset: 5,
          has_more: true
        },
        capabilities: [
          {
            agent_id: 'node-1',
            base_url: 'http://control-plane.local',
            version: '1.0.0',
            health_status: 'healthy',
            deployment_type: 'local',
            last_heartbeat: '2026-04-09T00:00:00Z',
            reasoners: [
              {
                id: 'planner',
                description: 'Plans tasks',
                tags: ['planning'],
                input_schema: { type: 'object' },
                output_schema: { type: 'object' },
                examples: [{ input: 'x' }],
                invocation_target: 'node-1:planner'
              }
            ],
            skills: [
              {
                id: 'search',
                description: 'Search docs',
                tags: ['search'],
                input_schema: { type: 'object' },
                invocation_target: 'node-1:skill:search'
              }
            ]
          }
        ]
      }
    });

    const result = await client.discoverCapabilities({
      format: 'json',
      agent: 'node-1',
      nodeId: 'node-1',
      agentIds: ['node-1', 'node-2'],
      nodeIds: ['node-2', 'node-1'],
      tags: ['search', 'search', 'planning'],
      includeInputSchema: true,
      includeOutputSchema: false,
      includeDescriptions: true,
      includeExamples: false,
      healthStatus: 'HEALTHY',
      limit: 10,
      offset: 5,
      headers: { 'X-Trace-ID': 'trace-1' }
    });

    expect(http.get).toHaveBeenCalledWith('/api/v1/discovery/capabilities', {
      params: {
        format: 'json',
        agent_ids: 'node-1,node-2',
        tags: 'search,planning',
        include_input_schema: 'true',
        include_output_schema: 'false',
        include_descriptions: 'true',
        include_examples: 'false',
        health_status: 'healthy',
        limit: '10',
        offset: '5'
      },
      headers: {
        Authorization: 'Bearer token',
        'X-Trace-ID': 'trace-1',
        Accept: 'application/json'
      },
      responseType: 'json',
      transformResponse: expect.any(Function)
    });
    expect(result).toEqual({
      format: 'json',
      raw: expect.stringContaining('"discovered_at":"2026-04-09T00:00:00Z"'),
      json: {
        discoveredAt: '2026-04-09T00:00:00Z',
        totalAgents: 1,
        totalReasoners: 1,
        totalSkills: 1,
        pagination: {
          limit: 10,
          offset: 5,
          hasMore: true
        },
        capabilities: [
          {
            agentId: 'node-1',
            baseUrl: 'http://control-plane.local',
            version: '1.0.0',
            healthStatus: 'healthy',
            deploymentType: 'local',
            lastHeartbeat: '2026-04-09T00:00:00Z',
            reasoners: [
              {
                id: 'planner',
                description: 'Plans tasks',
                tags: ['planning'],
                inputSchema: { type: 'object' },
                outputSchema: { type: 'object' },
                examples: [{ input: 'x' }],
                invocationTarget: 'node-1:planner'
              }
            ],
            skills: [
              {
                id: 'search',
                description: 'Search docs',
                tags: ['search'],
                inputSchema: { type: 'object' },
                invocationTarget: 'node-1:skill:search'
              }
            ]
          }
        ]
      }
    });
    expect(result.raw).toContain('"total_agents":1');
  });

  it('discoverCapabilities() supports compact and xml formats', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.get
      .mockResolvedValueOnce({
        data: JSON.stringify({
          discovered_at: '2026-04-09T00:00:00Z',
          reasoners: [{ id: 'r-1', agent_id: 'node-1', target: 'node-1:plan', tags: ['plan'] }],
          skills: [{ id: 's-1', agent_id: 'node-1', target: 'node-1:skill:search', tags: ['search'] }]
        })
      })
      .mockResolvedValueOnce({
        data: '<capabilities><agent id="node-1" /></capabilities>'
      });

    const compact = await client.discoverCapabilities({ format: 'compact' });
    const xml = await client.discoverCapabilities({ format: 'xml' });

    expect(compact).toEqual({
      format: 'compact',
      raw: expect.stringContaining('"reasoners"'),
      compact: {
        discoveredAt: '2026-04-09T00:00:00Z',
        reasoners: [{ id: 'r-1', agentId: 'node-1', target: 'node-1:plan', tags: ['plan'] }],
        skills: [{ id: 's-1', agentId: 'node-1', target: 'node-1:skill:search', tags: ['search'] }]
      }
    });
    expect(xml).toEqual({
      format: 'xml',
      raw: '<capabilities><agent id="node-1" /></capabilities>',
      xml: '<capabilities><agent id="node-1" /></capabilities>'
    });
  });

  it('publishWorkflowEvent() maps payload fields and uses dev timeout', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local',
      devMode: true
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    client.publishWorkflowEvent({
      executionId: 'exec-1',
      runId: 'run-1',
      workflowId: 'wf-1',
      rootWorkflowId: 'root-1',
      reasonerId: 'planner',
      agentNodeId: 'node-1',
      status: 'running',
      parentExecutionId: 'parent-1',
      parentWorkflowId: 'parent-wf-1',
      statusReason: 'started',
      inputData: { prompt: 'hi' },
      result: { ok: true },
      error: 'none',
      durationMs: 123
    });
    await Promise.resolve();

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/workflow/executions/events',
      JSON.stringify({
        execution_id: 'exec-1',
        workflow_id: 'wf-1',
        run_id: 'run-1',
        root_workflow_id: 'root-1',
        reasoner_id: 'planner',
        type: 'planner',
        agent_node_id: 'node-1',
        status: 'running',
        status_reason: 'started',
        parent_execution_id: 'parent-1',
        parent_workflow_id: 'parent-wf-1',
        input_data: { prompt: 'hi' },
        result: { ok: true },
        error: 'none',
        duration_ms: 123
      }),
      {
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        timeout: 1000
      }
    );
  });

  it('publishExecutionLogs() skips missing execution ids and posts both payload shapes', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    client.publishExecutionLogs({ entries: [] });
    client.publishExecutionLogs({
      v: 1,
      ts: '2026-04-09T00:00:00Z',
      execution_id: 'exec-single',
      level: 'info',
      source: 'sdk.logger',
      message: 'one'
    } as never);
    client.publishExecutionLogs({
      entries: [{ execution_id: 'exec-batch', level: 'info', message: 'two' }]
    } as never);
    await Promise.resolve();

    expect(http.post).toHaveBeenNthCalledWith(
      1,
      '/api/v1/executions/exec-single/logs',
      JSON.stringify({
        v: 1,
        ts: '2026-04-09T00:00:00Z',
        execution_id: 'exec-single',
        level: 'info',
        source: 'sdk.logger',
        message: 'one'
      }),
      {
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        timeout: 5000
      }
    );
    expect(http.post).toHaveBeenNthCalledWith(
      2,
      '/api/v1/executions/exec-batch/logs',
      JSON.stringify({
        entries: [{ execution_id: 'exec-batch', level: 'info', message: 'two' }]
      }),
      {
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        timeout: 5000
      }
    );
  });

  it('updateExecutionStatus() validates executionId and rounds progress', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    await expect(client.updateExecutionStatus('', {})).rejects.toThrow(
      'executionId is required to update workflow status'
    );

    await client.updateExecutionStatus('exec-1', {
      result: { ok: true },
      durationMs: 50,
      progress: 7.6,
      statusReason: 'halfway'
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/executions/exec-1/status',
      JSON.stringify({
        status: 'running',
        result: { ok: true },
        error: undefined,
        duration_ms: 50,
        progress: 8,
        status_reason: 'halfway'
      }),
      {
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      }
    );
  });

  it('supports credential mutation helpers and sendNote() posts lowercase execution headers', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://control-plane.local'
    });

    expect(client.didAuthConfigured).toBe(false);
    expect(client.getDID()).toBeUndefined();

    client.setDIDCredentials(TEST_DID, TEST_JWK);

    expect(client.didAuthConfigured).toBe(true);
    expect(client.getDID()).toBe(TEST_DID);

    postMock.mockResolvedValue({ data: {} });

    client.sendNote(
      'hello',
      ['tag-1'],
      'node-1',
      {
        runId: 'run-1',
        executionId: 'exec-1',
        sessionId: 'session-1',
        actorId: 'actor-1',
        workflowId: 'wf-1',
        rootWorkflowId: 'root-1',
        parentExecutionId: 'parent-1',
        reasonerId: 'planner',
        callerDid: 'did:key:caller',
        targetDid: 'did:key:target',
        agentNodeDid: 'did:key:node'
      },
      'http://ui.local',
      true
    );
    await Promise.resolve();

    const [url, body, config] = postMock.mock.calls[0] as [string, string, { headers: Record<string, string>; timeout: number }];
    expect(url).toBe('http://ui.local/executions/note');
    expect(JSON.parse(body)).toMatchObject({
      message: 'hello',
      tags: ['tag-1'],
      agent_node_id: 'node-1'
    });
    expect(config.timeout).toBe(5000);
    expect(config.headers).toEqual(
      expect.objectContaining({
        'Content-Type': 'application/json',
        'x-run-id': 'run-1',
        'x-execution-id': 'exec-1',
        'x-session-id': 'session-1',
        'x-actor-id': 'actor-1',
        'x-workflow-id': 'wf-1',
        'x-root-workflow-id': 'root-1',
        'x-parent-execution-id': 'parent-1',
        'x-reasoner-id': 'planner',
        'x-caller-did': 'did:key:caller',
        'x-target-did': 'did:key:target',
        'x-agent-node-did': 'did:key:node',
        'x-agent-node-id': 'node-1',
        [HEADER_CALLER_DID]: TEST_DID,
        [HEADER_DID_SIGNATURE]: expect.any(String),
        [HEADER_DID_TIMESTAMP]: expect.any(String),
        [HEADER_DID_NONCE]: expect.any(String)
      })
    );
  });
});
