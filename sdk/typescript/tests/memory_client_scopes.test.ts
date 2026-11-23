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
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import type { AIClient } from '../src/ai/AIClient.js';
import type { MemoryScope } from '../src/types/agent.js';

const memoryOperations: Array<[
  string,
  (memory: MemoryInterface, scope: MemoryScope, scopeId?: string) => unknown
]> = [
  ['set', (memory, scope, id) => memory.set('order', 42, scope, id)],
  ['get', (memory, scope, id) => memory.get('order', scope, id)],
  ['delete', (memory, scope, id) => memory.delete('order', scope, id)],
  ['exists', (memory, scope, id) => memory.exists('order', scope, id)],
  ['listKeys', (memory, scope, id) => memory.listKeys(scope, id)],
  ['setVector', (memory, scope, id) => memory.setVector('order', [1, 2], undefined, scope, id)],
  ['deleteVector', (memory, scope, id) => memory.deleteVector('order', scope, id)],
  ['embedAndSet', (memory, scope, id) => memory.embedAndSet('order', 'text', undefined, scope, id)],
  ['deleteVectors', (memory, scope, id) => memory.deleteVectors(['order', 'other'], scope, id)]
];

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

  it('uses the destination actor id when a session handle overrides its scope', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });
    const memory = new MemoryInterface({
      client,
      metadata: { sessionId: 'session-context', actorId: 'actor-context' }
    }).session('session-handle');

    await memory.set('order', { total: 42 }, 'actor');

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/set',
      { key: 'order', data: { total: 42 }, scope: 'actor' },
      { headers: { 'X-Session-ID': 'session-context', 'X-Actor-ID': 'actor-context' } }
    );
  });

  describe.each([
    { label: 'another scope', scope: 'actor', scopeId: undefined, actorId: 'actor-context', sessionId: 'session-context' },
    { label: 'the handle scope', scope: 'session', scopeId: undefined, actorId: 'actor-context', sessionId: 'session-handle' },
    { label: 'an explicit id', scope: 'actor', scopeId: 'actor-explicit', actorId: 'actor-explicit', sessionId: 'session-context' }
  ] as const)('$label', ({ scope, scopeId, actorId, sessionId }) => {
    it.each(memoryOperations)('%s uses the requested scope id', async (_name, invoke) => {
      const client = new MemoryClient('http://control-plane.local');
      const http = getHttp();
      http.post.mockResolvedValue({ data: { data: 42 } });
      http.get.mockResolvedValue({ data: [] });
      const memory = new MemoryInterface({
        client,
        aiClient: { embed: vi.fn(async () => [1, 2]) } as unknown as AIClient,
        metadata: {
          workflowId: 'workflow-context',
          sessionId: 'session-context',
          actorId: 'actor-context'
        }
      }).session('session-handle');

      await invoke(memory, scope, scopeId);

      const requests = [
        ...http.post.mock.calls.map(([, body, config]) => ({ scope: body.scope, headers: config.headers })),
        ...http.get.mock.calls.map(([, config]) => ({ scope: config.params.scope, headers: config.headers }))
      ];
      expect(requests.length).toBeGreaterThan(0);
      for (const request of requests) {
        expect(request).toEqual({
          scope,
          headers: {
            'X-Workflow-ID': 'workflow-context',
            'X-Session-ID': sessionId,
            'X-Actor-ID': actorId
          }
        });
      }
    });
  });

  it('resolves session ids from metadata when switching from workflow, actor, and global handles', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });
    const memory = new MemoryInterface({ client, metadata: { sessionId: 'session-context' } });

    for (const handle of [memory.workflow('workflow-handle'), memory.actor('actor-handle'), memory.globalScope]) {
      await handle.set('order', 42, 'session');
    }

    expect(http.post).toHaveBeenCalledTimes(3);
    for (const [, body, config] of http.post.mock.calls) {
      expect(body.scope).toBe('session');
      expect(config.headers).toEqual({ 'X-Session-ID': 'session-context' });
    }
  });

  it('does not substitute the source id when destination metadata is missing', async () => {
    const client = new MemoryClient('http://control-plane.local');
    const http = getHttp();
    http.post.mockResolvedValue({ data: {} });
    const memory = new MemoryInterface({ client }).session('session-handle');

    await memory.set('order', 42, 'actor');

    expect(http.post).toHaveBeenCalledWith(
      '/api/v1/memory/set', { key: 'order', data: 42, scope: 'actor' }, { headers: {} }
    );
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
