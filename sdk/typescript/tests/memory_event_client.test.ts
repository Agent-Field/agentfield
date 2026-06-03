import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import axios from 'axios';
import { MemoryEventClient } from '../src/memory/MemoryEventClient.js';

const { MockWebSocket } = vi.hoisted(() => {
  class HoistedMockWebSocket {
    static instances: HoistedMockWebSocket[] = [];
    private listeners = new Map<string, Array<(...args: unknown[]) => unknown>>();

    readonly url: string;
    readonly options?: { headers?: Record<string, string> };
    terminate = vi.fn();

    constructor(url: string, options?: { headers?: Record<string, string> }) {
      this.url = url;
      this.options = options;
      HoistedMockWebSocket.instances.push(this);
    }

    on(event: string, listener: (...args: unknown[]) => unknown) {
      const current = this.listeners.get(event) ?? [];
      current.push(listener);
      this.listeners.set(event, current);
      return this;
    }

    emit(event: string, ...args: unknown[]) {
      for (const listener of this.listeners.get(event) ?? []) {
        listener(...args);
      }
      return true;
    }

    removeAllListeners() {
      this.listeners.clear();
      return this;
    }
  }

  return { MockWebSocket: HoistedMockWebSocket };
});

vi.mock('axios', () => {
  const create = vi.fn(() => ({
    get: vi.fn()
  }));
  return {
    default: { create },
    create,
    isAxiosError: vi.fn(() => false)
  };
});

vi.mock('ws', () => ({
  default: MockWebSocket
}));

type MockHttpClient = {
  get: ReturnType<typeof vi.fn>;
};

const getHttpClient = (): MockHttpClient => {
  const mockCreate = (axios as unknown as { create: ReturnType<typeof vi.fn> }).create;
  const client = mockCreate.mock.results.at(-1)?.value as MockHttpClient | undefined;
  if (!client) {
    throw new Error('expected axios.create to have been called');
  }
  return client;
};

describe('MemoryEventClient exported methods', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    MockWebSocket.instances = [];
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('starts a websocket client once, sanitizes headers, and dispatches parsed events', async () => {
    const client = new MemoryEventClient('http://localhost:8080', {
      Authorization: 'Bearer token',
      Cookie: 'sid=1',
      'X-Trace-Id': 123,
      Accept: 'application/json'
    });
    const handler = vi.fn(async () => {});

    client.onEvent(handler);
    client.start();
    client.start();

    expect(MockWebSocket.instances).toHaveLength(1);
    const socket = MockWebSocket.instances[0];
    expect(socket.url).toBe('ws://localhost:8080/api/v1/memory/events/ws');
    expect(socket.options).toEqual({
      headers: {
        Authorization: 'Bearer token',
        Cookie: 'sid=1',
        'X-Trace-Id': '123'
      }
    });

    socket.emit('open');
    await socket.emit('message', Buffer.from(JSON.stringify({
      key: 'memo',
      data: { value: 1 },
      scope: 'workflow',
      scopeId: 'wf-1',
      timestamp: '2025-01-01T00:00:00Z',
      agentId: 'agent-1'
    })));

    expect(handler).toHaveBeenCalledWith({
      key: 'memo',
      data: { value: 1 },
      scope: 'workflow',
      scopeId: 'wf-1',
      timestamp: '2025-01-01T00:00:00Z',
      agentId: 'agent-1'
    });
  });

  it('swallows malformed websocket messages and supports reconnect scheduling and stop cleanup', async () => {
    vi.useFakeTimers();
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const client = new MemoryEventClient('http://localhost:8080');

    client.start();
    const firstSocket = MockWebSocket.instances[0];

    await firstSocket.emit('message', Buffer.from('{bad-json'));
    expect(errorSpy).toHaveBeenCalledWith('Failed to handle memory event', expect.any(SyntaxError));

    firstSocket.emit('error', new Error('disconnect'));
    firstSocket.emit('close');
    expect(MockWebSocket.instances).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(1000);
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(firstSocket.terminate).toHaveBeenCalledTimes(1);

    const secondSocket = MockWebSocket.instances[1];
    client.stop();
    expect(secondSocket.terminate).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(2000);
    expect(MockWebSocket.instances).toHaveLength(2);
  });

  it('requests history with normalized params and returns an empty list on transport errors', async () => {
    const client = new MemoryEventClient('http://localhost:8080', { 'X-Trace-Id': 'trace' }, 'api-key');
    const http = getHttpClient();
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    http.get.mockResolvedValueOnce({
      data: [
        {
          key: 'memo',
          data: { ok: true },
          scope: 'session',
          scopeId: 'sess-1',
          timestamp: '2025-01-01T00:00:00Z',
          agentId: 'agent-1'
        }
      ]
    });
    http.get.mockRejectedValueOnce(new Error('boom'));

    await expect(
      client.history({
        patterns: ['memo:*', 'task:*'],
        since: new Date('2025-01-01T00:00:00Z'),
        limit: 5,
        scope: 'session',
        scopeId: 'sess-1',
        metadata: {
          workflowId: 'wf-1',
          sessionId: 'sess-1'
        }
      })
    ).resolves.toEqual([
      {
        key: 'memo',
        data: { ok: true },
        scope: 'session',
        scopeId: 'sess-1',
        timestamp: '2025-01-01T00:00:00Z',
        agentId: 'agent-1'
      }
    ]);

    expect(http.get).toHaveBeenNthCalledWith(1, '/api/v1/memory/events/history', {
      params: {
        limit: 5,
        patterns: 'memo:*,task:*',
        since: '2025-01-01T00:00:00.000Z',
        scope: 'session',
        scope_id: 'sess-1'
      },
      headers: {
        'X-Trace-Id': 'trace',
        'X-Workflow-ID': 'wf-1',
        'X-Session-ID': 'sess-1',
        'X-API-Key': 'api-key'
      }
    });

    await expect(client.history()).resolves.toEqual([]);
    expect(errorSpy).toHaveBeenCalledWith('Failed to get event history: Error: boom');
  });
});
