import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { MCPClient } from '../src/mcp/MCPClient.js';

// ---------------------------------------------------------------------------
// Module-level axios mock
// ---------------------------------------------------------------------------

vi.mock('axios', () => {
  const create = vi.fn(() => ({
    post: vi.fn(),
    get: vi.fn()
  }));

  return {
    default: { create },
    create
  };
});

function getHttpMock() {
  const mockCreate = (axios as any).create as ReturnType<typeof vi.fn>;
  const last = mockCreate.mock.results.at(-1);
  return last?.value as { post: ReturnType<typeof vi.fn>; get: ReturnType<typeof vi.fn> };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MCPClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  describe('constructor', () => {
    it('creates a client with url config', () => {
      const client = new MCPClient({ alias: 'test', url: 'http://mcp:9000' });
      expect(client.alias).toBe('test');
      expect(client.baseUrl).toBe('http://mcp:9000');
      expect(client.transport).toBe('http');
    });

    it('creates a client with port config', () => {
      const client = new MCPClient({ alias: 'test', port: 3001 });
      expect(client.baseUrl).toBe('http://localhost:3001');
    });

    it('strips trailing slash from url', () => {
      const client = new MCPClient({ alias: 'test', url: 'http://mcp:9000/' });
      expect(client.baseUrl).toBe('http://mcp:9000');
    });

    it('uses specified transport', () => {
      const client = new MCPClient({ alias: 'test', url: 'http://mcp:9000', transport: 'bridge' });
      expect(client.transport).toBe('bridge');
    });

    it('throws when alias is missing', () => {
      expect(() => new MCPClient({ alias: '', url: 'http://mcp:9000' })).toThrow('alias is required');
    });

    it('throws when both url and port are missing', () => {
      expect(() => new MCPClient({ alias: 'test' })).toThrow('requires a url or port');
    });
  });

  // -------------------------------------------------------------------------
  // healthCheck
  // -------------------------------------------------------------------------
  describe('healthCheck()', () => {
    it('returns true on successful /health GET', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.get.mockResolvedValue({ data: {} });

      const result = await client.healthCheck();
      expect(result).toBe(true);
      expect(http.get).toHaveBeenCalledWith('/health');
      expect(client.lastHealthStatus).toBe(true);
    });

    it('returns false when /health fails', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.get.mockRejectedValue(new Error('ECONNREFUSED'));

      const result = await client.healthCheck();
      expect(result).toBe(false);
      expect(client.lastHealthStatus).toBe(false);
    });

    it('logs warning in devMode when health check fails', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' }, true);
      const http = getHttpMock();
      http.get.mockRejectedValue(new Error('down'));

      await client.healthCheck();
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('MCP health check failed'), 'down');
      warnSpy.mockRestore();
    });

    it('does not log in non-devMode when health check fails', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' }, false);
      const http = getHttpMock();
      http.get.mockRejectedValue(new Error('down'));

      await client.healthCheck();
      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    });
  });

  // -------------------------------------------------------------------------
  // listTools
  // -------------------------------------------------------------------------
  describe('listTools()', () => {
    it('uses JSON-RPC for http transport', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: {
          result: {
            tools: [
              { name: 'search', description: 'Search tool', inputSchema: { type: 'object' } }
            ]
          }
        }
      });

      const tools = await client.listTools();
      expect(tools).toHaveLength(1);
      expect(tools[0].name).toBe('search');
      expect(tools[0].description).toBe('Search tool');
      expect(http.post).toHaveBeenCalledWith('/mcp/v1', expect.objectContaining({
        jsonrpc: '2.0',
        method: 'tools/list'
      }));
    });

    it('uses bridge endpoint for bridge transport', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { tools: [{ name: 'read', description: 'Read file' }] }
      });

      const tools = await client.listTools();
      expect(tools).toHaveLength(1);
      expect(tools[0].name).toBe('read');
      expect(http.post).toHaveBeenCalledWith('/mcp/tools/list');
    });

    it('returns empty array on error', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockRejectedValue(new Error('timeout'));

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });

    it('normalizes tools with missing fields', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { result: { tools: [{ name: null }, {}] } }
      });

      const tools = await client.listTools();
      expect(tools[0].name).toBe('unknown');
      expect(tools[1].name).toBe('unknown');
    });

    it('handles missing tools array in response', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: {} } });

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });

    it('normalizes input_schema to inputSchema', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      const schema = { type: 'object', properties: {} };
      http.post.mockResolvedValue({
        data: { result: { tools: [{ name: 'tool', input_schema: schema }] } }
      });

      const tools = await client.listTools();
      expect(tools[0].inputSchema).toEqual(schema);
      expect(tools[0].input_schema).toEqual(schema);
    });
  });

  // -------------------------------------------------------------------------
  // callTool
  // -------------------------------------------------------------------------
  describe('callTool()', () => {
    it('throws when toolName is empty', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      await expect(client.callTool('')).rejects.toThrow('toolName is required');
    });

    it('calls via JSON-RPC for http transport', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: { output: 'hello' } } });

      const result = await client.callTool('greet', { name: 'Alice' });
      expect(result).toEqual({ output: 'hello' });
      expect(http.post).toHaveBeenCalledWith('/mcp/v1', expect.objectContaining({
        jsonrpc: '2.0',
        method: 'tools/call',
        params: { name: 'greet', arguments: { name: 'Alice' } }
      }));
    });

    it('calls via bridge endpoint for bridge transport', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: 'bridge-result' } });

      const result = await client.callTool('read', { path: '/tmp' });
      expect(result).toBe('bridge-result');
      expect(http.post).toHaveBeenCalledWith('/mcp/tools/call', {
        tool_name: 'read',
        arguments: { path: '/tmp' }
      });
    });

    it('returns res.data when bridge result is undefined', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { raw: 'value' } });

      const result = await client.callTool('tool', {});
      expect(result).toEqual({ raw: 'value' });
    });

    it('throws on JSON-RPC error response', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { error: { message: 'tool not found' } }
      });

      await expect(client.callTool('missing')).rejects.toThrow('tool not found');
    });

    it('throws on JSON-RPC error without message', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { error: 'some string error' }
      });

      await expect(client.callTool('broken')).rejects.toThrow('some string error');
    });

    it('returns res.data when result is undefined in JSON-RPC', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { other: 'field' } });

      const result = await client.callTool('tool');
      expect(result).toEqual({ other: 'field' });
    });

    it('defaults arguments to empty object', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: 'ok' } });

      await client.callTool('tool');
      const [, body] = http.post.mock.calls[0];
      expect(body.params.arguments).toEqual({});
    });

    it('re-throws HTTP errors', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      const http = getHttpMock();
      http.post.mockRejectedValue(new Error('network down'));

      await expect(client.callTool('tool')).rejects.toThrow('network down');
    });

    it('logs warning in devMode on error', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' }, true);
      const http = getHttpMock();
      http.post.mockRejectedValue(new Error('fail'));

      await expect(client.callTool('tool')).rejects.toThrow('fail');
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('MCP callTool failed'), 'fail');
      warnSpy.mockRestore();
    });
  });

  // -------------------------------------------------------------------------
  // lastHealthStatus
  // -------------------------------------------------------------------------
  describe('lastHealthStatus', () => {
    it('defaults to false', () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://mcp:9000' });
      expect(client.lastHealthStatus).toBe(false);
    });
  });
});
