import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';

type AxiosMockInstance = { post: Mock; get: Mock };

const { createMock, createdInstances } = vi.hoisted(() => {
  const instances: AxiosMockInstance[] = [];
  const mock: Mock = vi.fn(() => {
    const instance: AxiosMockInstance = {
      post: vi.fn(),
      get: vi.fn()
    };
    instances.push(instance);
    return instance;
  });
  return { createMock: mock, createdInstances: instances };
});

vi.mock('axios', () => ({
  default: { create: createMock },
  create: createMock
}));

import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { MemoryClient } from '../src/memory/MemoryClient.js';

describe('header forwarding', () => {
  beforeEach(() => {
    createMock.mockClear();
    createdInstances.length = 0;
  });

  it('merges default headers with execution metadata', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://example.com',
      defaultHeaders: { Authorization: 'Bearer tenant-token' }
    });
    const axiosInstance = createdInstances.at(-1)!;
    axiosInstance.post.mockResolvedValue({ data: { result: { ok: true } } });

    await client.execute('test-target', { foo: 'bar' }, { runId: 'run-123' });

    expect(axiosInstance.post).toHaveBeenCalledWith(
      '/api/v1/execute/test-target',
      JSON.stringify({ input: { foo: 'bar' } }),
      {
        headers: expect.objectContaining({
          Authorization: 'Bearer tenant-token',
          'Content-Type': 'application/json',
          'X-Run-ID': 'run-123'
        })
      }
    );
  });

  it('applies default headers to discovery with per-call headers', async () => {
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: 'http://example.com',
      defaultHeaders: { Authorization: 'Bearer tenant-token' }
    });
    const axiosInstance = createdInstances.at(-1)!;
    axiosInstance.get.mockResolvedValue({
      data: {
        discovered_at: '',
        total_agents: 0,
        total_reasoners: 0,
        total_skills: 0,
        pagination: { limit: 0, offset: 0, has_more: false },
        capabilities: []
      }
    });

    await client.discoverCapabilities({ headers: { 'X-Tenant-ID': 'tenant-a' } });

    expect(axiosInstance.get).toHaveBeenCalledWith('/api/v1/discovery/capabilities', {
      params: { format: 'json' },
      headers: {
        Authorization: 'Bearer tenant-token',
        'X-Tenant-ID': 'tenant-a',
        Accept: 'application/json'
      },
      responseType: 'json',
      transformResponse: expect.any(Function)
    });
  });

  it('includes default headers in memory requests', async () => {
    const memoryClient = new MemoryClient('http://example.com', {
      Authorization: 'Bearer tenant-token'
    });
    const axiosInstance = createdInstances.at(-1)!;
    axiosInstance.post.mockResolvedValue({ data: {} });

    await memoryClient.set('key', { data: true }, { metadata: { workflowId: 'wf-1' } });

    expect(axiosInstance.post).toHaveBeenCalledWith(
      '/api/v1/memory/set',
      { key: 'key', data: { data: true } },
      {
        headers: {
          Authorization: 'Bearer tenant-token',
          'X-Workflow-ID': 'wf-1'
        }
      }
    );
  });
});
