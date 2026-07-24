import { describe, it, expect, beforeEach, vi } from 'vitest';
import axios from 'axios';
import { MemoryClient } from '../src/memory/MemoryClient.js';
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { WorkflowReporter } from '../src/workflow/WorkflowReporter.js';
import { MemoryEventClient } from '../src/memory/MemoryEventClient.js';

vi.mock('axios', () => {
  const create = vi.fn(() => ({
    post: vi.fn(),
    get: vi.fn()
  }));

  const isAxiosError = (err: any) => Boolean(err?.isAxiosError);

  return {
    default: { create, isAxiosError },
    create,
    isAxiosError
  };
});

const getCreatedClient = () => {
  const mockCreate = (axios as any).create as ReturnType<typeof vi.fn>;
  const last = mockCreate.mock.results.at(-1);
  return last?.value;
};

describe('MemoryClient vector operations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('sends embedding payloads with scoped headers for setVector', async () => {
    const client = new MemoryClient('http://localhost:8080');
    const http = getCreatedClient();
    const post = vi.fn().mockResolvedValue({ data: {} });
    http.post = post;

    await client.setVector(
      'chunk_1',
      [0.1, 0.2],
      { source: 'doc' },
      { scope: 'workflow', metadata: { workflowId: 'wf-123', runId: 'wf-123' } }
    );

    expect(post).toHaveBeenCalledWith(
      '/api/v1/memory/vector/set',
      {
        key: 'chunk_1',
        embedding: [0.1, 0.2],
        metadata: { source: 'doc' },
        scope: 'workflow'
      },
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Workflow-ID': 'wf-123'
        })
      })
    );
  });

  it('searches vectors with query_embedding/top_k and returns results', async () => {
    const client = new MemoryClient('http://localhost:8080');
    const http = getCreatedClient();
    const results = [{ key: 'chunk_1', score: 0.9 }];
    http.post = vi.fn().mockResolvedValue({ data: results });

    const res = await client.searchVector([0.5, 0.2], {
      topK: 5,
      filters: { tag: 'a' },
      scope: 'session',
      scopeId: 's1',
      metadata: { sessionId: 's1' }
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/vector/search',
      {
        query_embedding: [0.5, 0.2],
        top_k: 5,
        filters: { tag: 'a' },
        scope: 'session'
      },
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Session-ID': 's1'
        })
      })
    );
    expect(res).toEqual(results);
  });

  it('deletes vectors with scoped headers', async () => {
    const client = new MemoryClient('http://localhost:8080');
    const http = getCreatedClient();
    const post = vi.fn().mockResolvedValue({ data: {} });
    http.post = post;

    await client.deleteVector('chunk_2', {
      scope: 'session',
      scopeId: 's1',
      metadata: { sessionId: 's1' }
    });

    expect(post).toHaveBeenCalledWith(
      '/api/v1/memory/vector/delete',
      {
        key: 'chunk_2',
        scope: 'session'
      },
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Session-ID': 's1'
        })
      })
    );
  });

  it('deletes scalar memory with scoped headers', async () => {
    const client = new MemoryClient('http://localhost:8080');
    const http = getCreatedClient();
    const post = vi.fn().mockResolvedValue({ data: {} });
    http.post = post;

    await client.delete('foo', { scope: 'session', scopeId: 's1', metadata: { sessionId: 's1' } });

    expect(post).toHaveBeenCalledWith(
      '/api/v1/memory/delete',
      { key: 'foo', scope: 'session' },
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Session-ID': 's1'
        })
      })
    );
  });

  it('lists keys using scope param and scoped headers', async () => {
    const client = new MemoryClient('http://localhost:8080');
    const http = getCreatedClient();
    http.get = vi.fn().mockResolvedValue({ data: [{ key: 'a' }, { key: 'b' }] });

    const keys = await client.listKeys('workflow', {
      scopeId: 'wf-1',
      metadata: { workflowId: 'wf-1' }
    });

    expect(http.get).toHaveBeenCalledWith(
      '/api/v1/memory/list',
      expect.objectContaining({
        params: { scope: 'workflow' },
        headers: expect.objectContaining({
          'X-Workflow-ID': 'wf-1'
        })
      })
    );
    expect(keys).toEqual(['a', 'b']);
  });
});

describe('AgentFieldClient discovery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('maps discovery response to typed shape', async () => {
    const payload = {
      discovered_at: '2025-01-01T00:00:00Z',
      total_agents: 1,
      total_reasoners: 1,
      total_skills: 1,
      pagination: { limit: 10, offset: 0, has_more: false },
      capabilities: [
        {
          agent_id: 'agent-1',
          base_url: 'http://agent',
          version: '1.0.0',
          health_status: 'active',
          deployment_type: 'long_running',
          last_heartbeat: 'now',
          reasoners: [
            {
              id: 'r1',
              invocation_target: 'agent-1:r1',
              tags: ['ml'],
              description: 'reasoner',
              input_schema: { type: 'object' },
              output_schema: { type: 'object' },
              examples: [{ name: 'ex' }]
            }
          ],
          skills: [
            {
              id: 's1',
              invocation_target: 'agent-1:s1',
              tags: ['tag'],
              description: 'skill',
              input_schema: { type: 'object' }
            }
          ]
        }
      ]
    };

    const client = new AgentFieldClient({ nodeId: 'tester' });
    const http = getCreatedClient();
    http.get = vi.fn().mockResolvedValue({ data: payload });

    const result = await client.discoverCapabilities({
      agent: 'agent-1',
      includeInputSchema: true,
      includeOutputSchema: true
    });

    expect(http.get).toHaveBeenCalledWith(
      '/api/v1/discovery/capabilities',
      expect.objectContaining({
        params: expect.objectContaining({
          agent: 'agent-1',
          include_input_schema: 'true',
          include_output_schema: 'true',
          format: 'json'
        }),
        headers: expect.objectContaining({
          Accept: 'application/json'
        })
      })
    );

    expect(result.format).toBe('json');
    expect(result.json?.capabilities[0].reasoners[0].invocationTarget).toBe('agent-1:r1');
    expect(result.json?.capabilities[0].skills[0].id).toBe('s1');
  });
});

describe('WorkflowReporter', () => {
  it('forwards progress updates to AgentFieldClient', async () => {
    const client = {
      updateExecutionStatus: vi.fn().mockResolvedValue({})
    } as unknown as AgentFieldClient;

    const reporter = new WorkflowReporter(client, { executionId: 'exec-1', runId: 'run-1' });
    await reporter.progress(42, { result: { ok: true } });

    expect(client.updateExecutionStatus).toHaveBeenCalledWith('exec-1', {
      status: 'running',
      progress: 42,
      result: { ok: true },
      error: undefined,
      durationMs: undefined
    });
  });
});

describe('MemoryInterface helpers', () => {
  it('performs hierarchical get fallback', async () => {
    const calls: any[] = [];
    const stubClient = {
      get: vi.fn().mockImplementation((_key: string, options: any) => {
        calls.push(options.scope);
        if (options.scope === 'session') return 'session-value';
        return undefined;
      })
    } as any;

    const memory = new MemoryInterface({
      client: stubClient,
      defaultScope: 'workflow',
      metadata: { workflowId: 'wf-1', sessionId: 's-1' }
    });

    const result = await memory.get('k1');
    expect(result).toBe('session-value');
    expect(calls).toEqual(['workflow', 'session']);
  });

  it('scoped helpers set the correct scope and scopeId', async () => {
    const set = vi.fn();
    const memory = new MemoryInterface({
      client: { set } as any,
      defaultScope: 'workflow',
      metadata: { workflowId: 'wf-1' }
    });

    await memory.session('s1').set('k', 'v');
    expect(set).toHaveBeenCalledWith(
      'k',
      'v',
      expect.objectContaining({
        scope: 'session',
        scopeId: 's1'
      })
    );
  });
});

describe('MemoryEventClient history operations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('gets MemoryEvent history', async () => {
    const client = new MemoryEventClient('http://localhost:8080');
    const http = getCreatedClient();
    const get = vi.fn().mockResolvedValue({
      data: [
        {
          scope: "session",
          scope_id: "s1",
          key: "cart.total",
          action: "set",
          data: 10,
        },
      ]
    });
    http.get = get;

    const result = await client.history({ patterns: ["cart.*"], limit: 1, since: new Date(Date.parse("2050-11-22T05:00:00.000Z")) });

    expect(get).toHaveBeenCalledWith(
      '/api/v1/memory/events/history',
      expect.objectContaining({
        params: expect.objectContaining({
          limit: 1,
          since: "2050-11-22T05:00:00.000Z",
          patterns: "cart.*"
        })
      })
    );

    expect(result).not.null;
    expect(result.length).toEqual(1);
    expect(result[0].key).toEqual("cart.total");
  });
});
