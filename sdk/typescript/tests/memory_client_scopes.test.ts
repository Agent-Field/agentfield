import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

type AxiosMockInstance = {
  post: Mock;
  get: Mock;
};

type AxiosLikeError = Error & {
  isAxiosError: boolean;
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

vi.mock('axios', () => {
  const isAxiosError = (err: unknown) =>
    typeof err === 'object' && err !== null && 'isAxiosError' in err && Boolean(err.isAxiosError);

  return {
    default: { create: createMock, isAxiosError },
    create: createMock,
    isAxiosError
  };
});

import { MemoryClient } from '../src/memory/MemoryClient.js';

function getHttp(): AxiosMockInstance {
  const http = createdInstances.at(-1);
  if (!http) {
    throw new Error('Expected axios.create() to have produced an instance');
  }
  return http;
}

function makeAxiosError(status: number, data?: Record<string, unknown>): AxiosLikeError {
  const err = new Error(`Request failed with status ${status}`) as AxiosLikeError;
  err.isAxiosError = true;
  err.response = { status, data };
  return err;
}

describe('MemoryClient scoped headers', () => {
  beforeEach(() => {
    createMock.mockClear();
    createdInstances.length = 0;
  });

  it('set() uses X-Workflow-ID for workflow scope and forwards metadata headers', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    await client.set('order.1', { total: 42 }, {
      scope: 'workflow',
      metadata: {
        workflowId: 'wf-1',
        runId: 'run-1',
        parentExecutionId: 'parent-1',
        callerDid: 'did:key:caller'
      }
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/set',
      {
        key: 'order.1',
        data: { total: 42 },
        scope: 'workflow'
      },
      {
        headers: expect.objectContaining({
          'X-Workflow-ID': 'wf-1',
          'X-Run-ID': 'run-1',
          'X-Parent-Execution-ID': 'parent-1',
          'X-Caller-DID': 'did:key:caller'
        })
      }
    );
  });

  it('set() uses an explicit session scopeId over metadata.sessionId', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    await client.set('order.2', { total: 99 }, {
      scope: 'session',
      scopeId: 'session-explicit',
      metadata: {
        sessionId: 'session-from-metadata',
        runId: 'run-2'
      }
    });

    const [, , config] = http.post.mock.calls[0];
    expect((config as { headers: Record<string, string> }).headers).toEqual(
      expect.objectContaining({
        'X-Session-ID': 'session-explicit',
        'X-Workflow-ID': 'run-2',
        'X-Run-ID': 'run-2'
      })
    );
  });

  it('set() sends no scope-specific header for global scope', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    await client.set('global.key', { enabled: true }, { scope: 'global' });

    const [, , config] = http.post.mock.calls[0];
    const headers = (config as { headers: Record<string, string> }).headers;
    expect(headers['X-Workflow-ID']).toBeUndefined();
    expect(headers['X-Session-ID']).toBeUndefined();
    expect(headers['X-Actor-ID']).toBeUndefined();
  });

  it('get() builds scoped headers and returns undefined for 404 responses', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockRejectedValue(makeAxiosError(404, { error: 'not found' }));

    await expect(
      client.get('order.3', {
        scope: 'actor',
        metadata: {
          actorId: 'actor-1',
          callerDid: 'did:key:caller'
        }
      })
    ).resolves.toBeUndefined();

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/get',
      {
        key: 'order.3',
        scope: 'actor'
      },
      {
        headers: expect.objectContaining({
          'X-Actor-ID': 'actor-1',
          'X-Caller-DID': 'did:key:caller'
        })
      }
    );
  });

  it('delete() sends the delete payload with scoped headers', async () => {
    const client = new MemoryClient('http://control-plane.local', {
      Authorization: 'Bearer tenant-token'
    });
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });

    await client.delete('order.4', {
      scope: 'session',
      scopeId: 'session-4',
      metadata: {
        runId: 'run-4',
        parentExecutionId: 'parent-4'
      }
    });

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/delete',
      {
        key: 'order.4',
        scope: 'session'
      },
      {
        headers: expect.objectContaining({
          Authorization: 'Bearer tenant-token',
          'X-Session-ID': 'session-4',
          'X-Workflow-ID': 'run-4',
          'X-Run-ID': 'run-4',
          'X-Parent-Execution-ID': 'parent-4'
        })
      }
    );
  });

  it('listKeys() sends the workflow filter through params and scoped headers', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.get.mockResolvedValue({ data: [{ key: 'a' }, { key: 'b' }] });

    const keys = await client.listKeys('workflow', {
      metadata: {
        workflowId: 'wf-list',
        runId: 'run-list'
      }
    });

    expect(keys).toEqual(['a', 'b']);
    expect(http.get).toHaveBeenCalledWith(
      '/api/v1/memory/list',
      {
        params: { scope: 'workflow' },
        headers: expect.objectContaining({
          'X-Workflow-ID': 'wf-list',
          'X-Run-ID': 'run-list'
        })
      }
    );
  });

  it('re-throws 500 responses from get()', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    const serverError = makeAxiosError(500, { error: 'boom' });
    http.post.mockRejectedValue(serverError);

    await expect(client.get('order.5')).rejects.toBe(serverError);
  });
});
