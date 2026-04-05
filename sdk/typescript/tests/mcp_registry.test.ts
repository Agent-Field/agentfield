import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MCPClientRegistry } from '../src/mcp/MCPClientRegistry.js';
import { MCPToolRegistrar } from '../src/mcp/MCPToolRegistrar.js';
import { MCPClient } from '../src/mcp/MCPClient.js';
import type { Agent } from '../src/agent/Agent.js';

// ---------------------------------------------------------------------------
// Mock MCPClient so no real axios instances are created
// ---------------------------------------------------------------------------

vi.mock('../src/mcp/MCPClient.js', () => {
  return {
    MCPClient: vi.fn().mockImplementation((config: any, devMode?: boolean) => ({
      alias: config.alias,
      baseUrl: config.url ?? `http://localhost:${config.port}`,
      transport: config.transport ?? 'http',
      healthCheck: vi.fn().mockResolvedValue(true),
      listTools: vi.fn().mockResolvedValue([]),
      callTool: vi.fn().mockResolvedValue({}),
      lastHealthStatus: false
    }))
  };
});

// ---------------------------------------------------------------------------
// MCPClientRegistry
// ---------------------------------------------------------------------------

describe('MCPClientRegistry', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('registers and retrieves clients by alias', () => {
    const registry = new MCPClientRegistry();
    const client = registry.register({ alias: 'search', url: 'http://mcp:9000' });

    expect(client.alias).toBe('search');
    expect(registry.get('search')).toBe(client);
  });

  it('returns undefined for unregistered alias', () => {
    const registry = new MCPClientRegistry();
    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('lists all registered clients', () => {
    const registry = new MCPClientRegistry();
    registry.register({ alias: 'a', url: 'http://a:9000' });
    registry.register({ alias: 'b', url: 'http://b:9000' });

    const clients = registry.list();
    expect(clients).toHaveLength(2);
    expect(clients.map((c) => c.alias)).toEqual(['a', 'b']);
  });

  it('clears all clients', () => {
    const registry = new MCPClientRegistry();
    registry.register({ alias: 'x', url: 'http://x:9000' });
    registry.clear();
    expect(registry.list()).toHaveLength(0);
    expect(registry.get('x')).toBeUndefined();
  });

  // -----------------------------------------------------------------------
  // healthSummary
  // -----------------------------------------------------------------------
  describe('healthSummary()', () => {
    it('returns disabled status when no servers are registered', async () => {
      const registry = new MCPClientRegistry();
      const summary = await registry.healthSummary();

      expect(summary.status).toBe('disabled');
      expect(summary.totalServers).toBe(0);
      expect(summary.healthyServers).toBe(0);
      expect(summary.servers).toEqual([]);
    });

    it('returns ok when all servers are healthy', async () => {
      const registry = new MCPClientRegistry();
      registry.register({ alias: 'a', url: 'http://a:9000' });
      registry.register({ alias: 'b', url: 'http://b:9000' });

      const summary = await registry.healthSummary();
      expect(summary.status).toBe('ok');
      expect(summary.totalServers).toBe(2);
      expect(summary.healthyServers).toBe(2);
      expect(summary.servers).toHaveLength(2);
    });

    it('returns degraded when some servers are unhealthy', async () => {
      const registry = new MCPClientRegistry();
      const healthy = registry.register({ alias: 'a', url: 'http://a:9000' });
      const unhealthy = registry.register({ alias: 'b', url: 'http://b:9000' });
      (unhealthy.healthCheck as ReturnType<typeof vi.fn>).mockResolvedValue(false);

      const summary = await registry.healthSummary();
      expect(summary.status).toBe('degraded');
      expect(summary.healthyServers).toBe(1);
    });

    it('returns degraded when all servers are unhealthy', async () => {
      const registry = new MCPClientRegistry();
      const c = registry.register({ alias: 'a', url: 'http://a:9000' });
      (c.healthCheck as ReturnType<typeof vi.fn>).mockResolvedValue(false);

      const summary = await registry.healthSummary();
      expect(summary.status).toBe('degraded');
      expect(summary.healthyServers).toBe(0);
    });
  });
});

// ---------------------------------------------------------------------------
// MCPToolRegistrar
// ---------------------------------------------------------------------------

describe('MCPToolRegistrar', () => {
  /** Create a minimal mock Agent with a skill() method and skills map */
  function createMockAgent(): Agent {
    const skillsMap = new Map<string, any>();
    return {
      skill: vi.fn((name: string, handler: any, opts: any) => {
        skillsMap.set(name, { name, handler, ...opts });
      }),
      skills: {
        get: (name: string) => skillsMap.get(name),
        all: () => Array.from(skillsMap.values())
      }
    } as unknown as Agent;
  }

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('registers MCP servers into the registry', () => {
    const agent = createMockAgent();
    const registry = new MCPClientRegistry();
    const registrar = new MCPToolRegistrar(agent, registry);

    registrar.registerServers([
      { alias: 'a', url: 'http://a:9000' },
      { alias: 'b', url: 'http://b:9000' }
    ]);

    expect(registry.list()).toHaveLength(2);
  });

  describe('registerAll()', () => {
    it('registers tools from healthy servers as agent skills', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'search', description: 'Search things', inputSchema: { type: 'object' } },
        { name: 'read', description: 'Read file' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered).toHaveLength(2);
      expect(result.registered[0].skillName).toBe('srv_search');
      expect(result.registered[1].skillName).toBe('srv_read');
      expect(agent.skill).toHaveBeenCalledTimes(2);
    });

    it('skips unhealthy servers', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'down', url: 'http://down:9000' });
      (client.healthCheck as ReturnType<typeof vi.fn>).mockResolvedValue(false);
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'A tool' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered).toHaveLength(0);
      expect(agent.skill).not.toHaveBeenCalled();
    });

    it('skips tools with no name', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: '', description: 'empty name' },
        { name: null, description: 'null name' },
        { description: 'missing name' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered).toHaveLength(0);
    });

    it('does not register duplicate skills', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'A tool' }
      ]);

      await registrar.registerAll();
      await registrar.registerAll();

      expect(agent.skill).toHaveBeenCalledTimes(1);
    });

    it('applies namespace to skill names', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry, { namespace: 'mcp' });

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'search', description: 'Search' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered[0].skillName).toBe('mcp_srv_search');
    });

    it('sanitizes special characters in skill names', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'my-srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'read-file!', description: 'Read' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered[0].skillName).toBe('my_srv_read_file');
    });

    it('prefixes with mcp_ when name starts with digit', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: '123srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'Tool' }
      ]);

      const result = await registrar.registerAll();
      expect(result.registered[0].skillName).toBe('mcp_123srv_tool');
    });

    it('includes custom tags and mcp/alias tags', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry, { tags: ['custom'] });

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'Tool' }
      ]);

      await registrar.registerAll();
      const [, , opts] = (agent.skill as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(opts.tags).toContain('mcp');
      expect(opts.tags).toContain('srv');
      expect(opts.tags).toContain('custom');
    });

    it('logs in devMode when skipping and registering', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});

      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry, { devMode: true });

      const unhealthy = registry.register({ alias: 'down', url: 'http://down:9000' });
      (unhealthy.healthCheck as ReturnType<typeof vi.fn>).mockResolvedValue(false);

      const healthy = registry.register({ alias: 'up', url: 'http://up:9000' });
      (healthy.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'Tool' }
      ]);

      await registrar.registerAll();
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Skipping MCP server down'));
      expect(infoSpy).toHaveBeenCalledWith(expect.stringContaining('Registered MCP skill'));

      warnSpy.mockRestore();
      infoSpy.mockRestore();
    });

    it('registered skill handler calls client.callTool', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'greet', description: 'Greet' }
      ]);
      (client.callTool as ReturnType<typeof vi.fn>).mockResolvedValue({ message: 'hi' });

      await registrar.registerAll();

      // Get the handler that was passed to agent.skill
      const [, handler] = (agent.skill as ReturnType<typeof vi.fn>).mock.calls[0];
      const result = await handler({ input: { name: 'Alice' } });

      expect(client.callTool).toHaveBeenCalledWith('greet', { name: 'Alice' });
      expect(result).toEqual({
        status: 'success',
        result: { message: 'hi' },
        server: 'srv',
        tool: 'greet'
      });
    });

    it('registered skill handler handles non-object input', async () => {
      const agent = createMockAgent();
      const registry = new MCPClientRegistry();
      const registrar = new MCPToolRegistrar(agent, registry);

      const client = registry.register({ alias: 'srv', url: 'http://srv:9000' });
      (client.listTools as ReturnType<typeof vi.fn>).mockResolvedValue([
        { name: 'tool', description: 'Tool' }
      ]);
      (client.callTool as ReturnType<typeof vi.fn>).mockResolvedValue('ok');

      await registrar.registerAll();

      const [, handler] = (agent.skill as ReturnType<typeof vi.fn>).mock.calls[0];
      // Pass non-object input
      const result = await handler({ input: null });

      expect(client.callTool).toHaveBeenCalledWith('tool', {});
      expect(result.status).toBe('success');
    });
  });
});
