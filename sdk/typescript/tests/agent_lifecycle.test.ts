import type http from 'node:http';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { Agent } from '../src/agent/Agent.js';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { MemoryEventClient } from '../src/memory/MemoryEventClient.js';

type RegisterPayload = {
  id: string;
  version: string;
  base_url: string;
  public_url: string;
  deployment_type: string;
  reasoners: Array<Record<string, unknown>>;
  skills: Array<Record<string, unknown>>;
  proposed_tags: string[];
  tags: string[];
};

type FakeServer = {
  close: ReturnType<typeof vi.fn>;
  on: ReturnType<typeof vi.fn>;
};

function createFakeServer(): FakeServer {
  const server: FakeServer = {
    close: vi.fn((callback?: (err?: Error) => void) => {
      callback?.();
      return server;
    }),
    on: vi.fn((_event: string, _handler: (...args: unknown[]) => void) => server)
  };

  return server;
}

function attachFakeListener(agent: Agent, server: FakeServer) {
  const listen = vi.fn((port: number, host: string, callback?: () => void) => {
    expect(port).toBe(4123);
    expect(host).toBe('0.0.0.0');
    callback?.();
    return server as unknown as http.Server;
  });

  (agent.app as unknown as { listen: typeof listen }).listen = listen;
  return listen;
}

describe('Agent lifecycle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('serve() registers the agent with reasoner and skill definitions, then starts heartbeats', async () => {
    const register = vi.spyOn(AgentFieldClient.prototype, 'register').mockResolvedValue({});
    const heartbeat = vi.spyOn(AgentFieldClient.prototype, 'heartbeat').mockResolvedValue({
      status: 'running',
      node_id: 'agent-1'
    });
    const memoryStart = vi.spyOn(MemoryEventClient.prototype, 'start').mockImplementation(() => {});
    const memoryStop = vi.spyOn(MemoryEventClient.prototype, 'stop').mockImplementation(() => {});

    const agent = new Agent({
      nodeId: 'agent-1',
      version: '1.2.3',
      agentFieldUrl: 'http://control-plane.local',
      didEnabled: false,
      port: 4123,
      host: '0.0.0.0',
      heartbeatIntervalMs: 1000
    });
    agent.reasoner(
      'plan',
      async () => ({ ok: true }),
      {
        tags: ['core', 'planner'],
        inputSchema: z.object({ prompt: z.string() }),
        outputSchema: z.object({ ok: z.boolean() })
      }
    );
    agent.skill(
      'format',
      () => ({ ok: true }),
      {
        tags: ['text'],
        inputSchema: z.object({ value: z.string() })
      }
    );

    const fakeServer = createFakeServer();
    const listen = attachFakeListener(agent, fakeServer);

    await agent.serve();

    expect(register).toHaveBeenCalledTimes(1);
    const payload = register.mock.calls[0][0] as RegisterPayload;
    expect(payload).toMatchObject({
      id: 'agent-1',
      version: '1.2.3',
      base_url: 'http://127.0.0.1:4123',
      public_url: 'http://127.0.0.1:4123',
      deployment_type: 'long_running'
    });
    expect(payload.reasoners).toEqual([
      expect.objectContaining({
        id: 'plan',
        tags: ['core', 'planner'],
        proposed_tags: ['core', 'planner'],
        input_schema: expect.objectContaining({ type: 'object' }),
        output_schema: expect.objectContaining({ type: 'object' })
      })
    ]);
    expect(payload.skills).toEqual([
      expect.objectContaining({
        id: 'format',
        tags: ['text'],
        proposed_tags: ['text'],
        input_schema: expect.objectContaining({ type: 'object' })
      })
    ]);
    expect(listen).toHaveBeenCalledTimes(1);
    expect(memoryStart).toHaveBeenCalledTimes(1);
    expect(heartbeat).toHaveBeenNthCalledWith(1, 'starting');
    expect(heartbeat).toHaveBeenNthCalledWith(2, 'ready');

    await vi.advanceTimersByTimeAsync(1000);
    expect(heartbeat).toHaveBeenNthCalledWith(3, 'ready');

    await agent.shutdown();
    expect(memoryStop).toHaveBeenCalledTimes(1);
    expect(fakeServer.close).toHaveBeenCalledTimes(1);
  });

  it('shutdown() stops the heartbeat interval so no more heartbeats fire afterwards', async () => {
    const heartbeat = vi.spyOn(AgentFieldClient.prototype, 'heartbeat').mockResolvedValue({
      status: 'running',
      node_id: 'agent-1'
    });
    vi.spyOn(AgentFieldClient.prototype, 'register').mockResolvedValue({});
    vi.spyOn(MemoryEventClient.prototype, 'start').mockImplementation(() => {});
    vi.spyOn(MemoryEventClient.prototype, 'stop').mockImplementation(() => {});

    const agent = new Agent({
      nodeId: 'agent-1',
      agentFieldUrl: 'http://control-plane.local',
      didEnabled: false,
      port: 4123,
      host: '0.0.0.0',
      heartbeatIntervalMs: 1000
    });

    attachFakeListener(agent, createFakeServer());

    await agent.serve();
    expect(heartbeat).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(1000);
    expect(heartbeat).toHaveBeenCalledTimes(3);

    await agent.shutdown();
    await vi.advanceTimersByTimeAsync(5000);

    expect(heartbeat).toHaveBeenCalledTimes(3);
  });

  it('serve() surfaces control-plane registration failures when devMode is disabled', async () => {
    const registerError = new Error('registration failed');
    const register = vi.spyOn(AgentFieldClient.prototype, 'register').mockRejectedValue(registerError);
    const heartbeat = vi.spyOn(AgentFieldClient.prototype, 'heartbeat').mockResolvedValue({
      status: 'running',
      node_id: 'agent-1'
    });

    const agent = new Agent({
      nodeId: 'agent-1',
      agentFieldUrl: 'http://control-plane.local',
      didEnabled: false,
      devMode: false,
      port: 4123,
      host: '0.0.0.0',
      heartbeatIntervalMs: 1000
    });
    const listen = attachFakeListener(agent, createFakeServer());

    await expect(agent.serve()).rejects.toBe(registerError);

    expect(register).toHaveBeenCalledTimes(1);
    expect(listen).not.toHaveBeenCalled();
    expect(heartbeat).not.toHaveBeenCalled();
  });
});
