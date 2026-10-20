import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import type { MemoryClient } from '../src/memory/MemoryClient.js';
import type { MemoryEventClient } from '../src/memory/MemoryEventClient.js';

const { requests, sockets, MockWebSocket } = vi.hoisted(() => {
  const requests = {
    get: vi.fn().mockResolvedValue({ data: [] }),
    post: vi.fn().mockResolvedValue({ data: {} })
  };
  const sockets: Array<{ url: string; headers: Record<string, string> }> = [];
  class MockWebSocket {
    constructor(url: string, options: { headers: Record<string, string> }) {
      sockets.push({ url, headers: options.headers });
    }
    on() { return this; }
    removeAllListeners() { return this; }
    terminate() {}
  }
  return { requests, sockets, MockWebSocket };
});

vi.mock('axios', () => ({
  default: { create: vi.fn(() => requests) },
  isAxiosError: vi.fn(() => false)
}));
vi.mock('ws', () => ({ default: MockWebSocket }));

// Use the real clients created by Agent and fake only their transports: this
// catches missing constructor headers for both HTTP and the unfiltered socket.
describe('Agent memory authentication headers', () => {
  let events: MemoryEventClient | undefined;
  beforeEach(() => {
    vi.clearAllMocks();
    sockets.length = 0;
  });
  afterEach(() => events?.stop());

  it.each([
    { defaults: undefined, key: 'configured-key', node: 'agent-beta' },
    {
      defaults: { 'X-API-Key': 'explicit-key', 'X-Agent-Node-ID': 'explicit-node' },
      key: 'explicit-key', node: 'explicit-node'
    },
    {
      defaults: { 'x-api-key': 'lower-key', 'x-agent-node-id': 'lower-node' },
      key: 'lower-key', node: 'lower-node'
    }
  ])('sends identity and API key, preserving defaults: $defaults', async ({ defaults, key, node }) => {
    const originalDefaults = defaults ? { ...defaults } : undefined;
    const agent = new Agent({
      nodeId: 'agent-beta',
      agentFieldUrl: 'http://control-plane.test:8080',
      apiKey: 'configured-key',
      defaultHeaders: defaults
    });
    const clients = agent as unknown as { memoryClient: MemoryClient; memoryEventClient: MemoryEventClient };
    events = clients.memoryEventClient;
    events.start();
    await events.history();
    // No metadata override: verify the defaults the Agent gives MemoryClient.
    await clients.memoryClient.set('key', 'value', { scope: 'global' });
    await agent.getMemoryInterface({ actorId: 'user-123' }).set('key', 'value', 'actor');

    const normalized = (headers: Record<string, string>) =>
      Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
    expect(sockets).toHaveLength(1);
    expect(sockets[0].url).toBe('ws://control-plane.test:8080/api/v1/memory/events/ws');
    expect(normalized(sockets[0].headers)).toMatchObject({ 'x-api-key': key, 'x-agent-node-id': node });
    // No second header differing only by case may override an explicit default.
    expect(Object.keys(sockets[0].headers).filter(k => k.toLowerCase() === 'x-api-key')).toHaveLength(1);
    expect(Object.keys(sockets[0].headers).filter(k => k.toLowerCase() === 'x-agent-node-id')).toHaveLength(1);
    expect(requests.get).toHaveBeenCalledWith('/api/v1/memory/events/history', expect.objectContaining({
      headers: expect.any(Object)
    }));
    expect(normalized(requests.get.mock.calls[0][1].headers)).toMatchObject({ 'x-api-key': key, 'x-agent-node-id': node });
    expect(requests.post).toHaveBeenCalledTimes(2);
    for (const [path, , options] of requests.post.mock.calls) {
      expect(path).toBe('/api/v1/memory/set');
      expect(normalized(options.headers)['x-api-key']).toBe(key);
    }
    expect(defaults).toEqual(originalDefaults);
  });
});
