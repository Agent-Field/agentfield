import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { MCPClient } from '../src/mcp/MCPClient.js';
import type { MCPServerConfig } from '../src/types/agent.js';

// ---------------------------------------------------------------------------
// Module-level axios mock (mirrors the pattern used in memory_and_discovery.test.ts)
// ---------------------------------------------------------------------------

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

/** Returns the most-recently created axios instance */
function getHttpMock() {
  const mockCreate = (axios as any).create as ReturnType<typeof vi.fn>;
  const last = mockCreate.mock.results.at(-1);
  return last?.value as { post: ReturnType<typeof vi.fn>; get: ReturnType<typeof vi.fn> };
}

/** Network-style error (no response) */
function networkError(message = 'connect ECONNREFUSED') {
  const err: any = new Error(message);
  err.isAxiosError = true;
  err.code = 'ECONNREFUSED';
  return err;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MCPClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Constructor validation
  // -------------------------------------------------------------------------
  describe('constructor', () => {
    it('creates an instance with a url', () => {
      const client = new MCPClient({ alias: 'my-server', url: 'http://localhost:9000' });
      expect(client.alias).toBe('my-server');
      expect(client.baseUrl).toBe('http://localhost:9000');
    });

    it('creates an instance with a port', () => {
      const client = new MCPClient({ alias: 'srv', port: 9001 });
      expect(client.baseUrl).toBe('http://localhost:9001');
    });

    it('strips trailing slash from url', () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://example.com/mcp/' });
      expect(client.baseUrl).toBe('http://example.com/mcp');
    });

    it('throws when alias is missing', () => {
      expect(() => new MCPClient({ alias: '', url: 'http://localhost:9000' })).toThrow(
        'MCP server alias is required'
      );
    });

    it('throws when neither url nor port is provided', () => {
      expect(() => new MCPClient({ alias: 'srv' } as MCPServerConfig)).toThrow(
        'MCP server "srv" requires a url or port'
      );
    });

    it('defaults transport to "http"', () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      expect(client.transport).toBe('http');
    });

    it('stores transport from config', () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000', transport: 'bridge' });
      expect(client.transport).toBe('bridge');
    });
  });

  // -------------------------------------------------------------------------
  // healthCheck()
  // -------------------------------------------------------------------------
  describe('healthCheck()', () => {
    it('returns true when GET /health succeeds', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.get.mockResolvedValue({ data: { status: 'ok' } });

      const result = await client.healthCheck();

      expect(result).toBe(true);
      expect(http.get).toHaveBeenCalledWith('/health');
    });

    it('returns false when GET /health throws', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.get.mockRejectedValue(networkError());

      const result = await client.healthCheck();
      expect(result).toBe(false);
    });

    it('updates lastHealthStatus to true on success', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.get.mockResolvedValue({});

      await client.healthCheck();
      expect(client.lastHealthStatus).toBe(true);
    });

    it('updates lastHealthStatus to false on failure', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.get.mockRejectedValue(networkError());

      await client.healthCheck();
      expect(client.lastHealthStatus).toBe(false);
    });

    it('does not throw in devMode on error', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' }, true);
      const http = getHttpMock();
      http.get.mockRejectedValue(networkError());

      await expect(client.healthCheck()).resolves.toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // listTools() – http transport
  // -------------------------------------------------------------------------
  describe('listTools() – http transport', () => {
    it('POSTs jsonrpc tools/list and returns normalized tools', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: {
          jsonrpc: '2.0',
          id: 1,
          result: {
            tools: [{ name: 'echo', description: 'Echo back', inputSchema: { type: 'object' } }]
          }
        }
      });

      const tools = await client.listTools();

      expect(tools).toHaveLength(1);
      expect(tools[0]?.name).toBe('echo');
      expect(tools[0]?.description).toBe('Echo back');
      expect(tools[0]?.inputSchema).toEqual({ type: 'object' });
    });

    it('returns empty array when result.tools is absent', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { jsonrpc: '2.0', id: 1, result: {} } });

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });

    it('returns empty array on network error', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockRejectedValue(networkError());

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });

    it('normalizes tool with input_schema (snake_case)', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: {
          result: {
            tools: [{ name: 'tool1', input_schema: { type: 'object', properties: {} } }]
          }
        }
      });

      const tools = await client.listTools();
      expect(tools[0]?.inputSchema).toEqual({ type: 'object', properties: {} });
    });

    it('falls back to "unknown" when tool name is missing', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { result: { tools: [{ description: 'nameless' }] } }
      });

      const tools = await client.listTools();
      expect(tools[0]?.name).toBe('unknown');
    });
  });

  // -------------------------------------------------------------------------
  // listTools() – bridge transport
  // -------------------------------------------------------------------------
  describe('listTools() – bridge transport', () => {
    it('POSTs to /mcp/tools/list and returns tools', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { tools: [{ name: 'search', description: 'Search' }] }
      });

      const tools = await client.listTools();

      expect(http.post).toHaveBeenCalledWith('/mcp/tools/list');
      expect(tools[0]?.name).toBe('search');
    });

    it('returns empty array when tools key is absent', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: {} });

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });
  });

  // -------------------------------------------------------------------------
  // callTool() – http transport
  // -------------------------------------------------------------------------
  describe('callTool() – http transport', () => {
    it('throws when toolName is empty', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      await expect(client.callTool('')).rejects.toThrow('toolName is required');
    });

    it('POSTs jsonrpc tools/call and returns result', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { jsonrpc: '2.0', id: 1, result: { value: 'hello' } }
      });

      const result = await client.callTool('echo', { msg: 'hi' });
      expect(result).toEqual({ value: 'hello' });
    });

    it('sends correct jsonrpc payload', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: {} } });

      await client.callTool('my_tool', { x: 1 });

      const [url, body] = http.post.mock.calls[0];
      expect(url).toBe('/mcp/v1');
      expect(body.jsonrpc).toBe('2.0');
      expect(body.method).toBe('tools/call');
      expect(body.params).toEqual({ name: 'my_tool', arguments: { x: 1 } });
    });

    it('throws when response contains an error field', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({
        data: { jsonrpc: '2.0', id: 1, error: { message: 'tool not found' } }
      });

      await expect(client.callTool('missing')).rejects.toThrow('tool not found');
    });

    it('re-throws network errors', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockRejectedValue(networkError('connection refused'));

      await expect(client.callTool('any')).rejects.toThrow('connection refused');
    });

    it('returns raw data when result is undefined', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { ok: true } });

      const result = await client.callTool('raw');
      expect(result).toEqual({ ok: true });
    });

    it('uses empty object as default for arguments', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: 'ok' } });

      await client.callTool('no-args');

      const [, body] = http.post.mock.calls[0];
      expect(body.params.arguments).toEqual({});
    });
  });

  // -------------------------------------------------------------------------
  // callTool() – bridge transport
  // -------------------------------------------------------------------------
  describe('callTool() – bridge transport', () => {
    it('POSTs to /mcp/tools/call with tool_name and arguments', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { result: { echoed: 'hello' } } });

      const result = await client.callTool('echo', { text: 'hello' });

      expect(http.post).toHaveBeenCalledWith('/mcp/tools/call', {
        tool_name: 'echo',
        arguments: { text: 'hello' }
      });
      expect(result).toEqual({ echoed: 'hello' });
    });

    it('falls back to full response data when result key is absent', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000', transport: 'bridge' });
      const http = getHttpMock();
      http.post.mockResolvedValue({ data: { status: 'done' } });

      const result = await client.callTool('tool');
      expect(result).toEqual({ status: 'done' });
    });
  });

  // -------------------------------------------------------------------------
  // devMode behaviour
  // -------------------------------------------------------------------------
  describe('devMode', () => {
    it('listTools returns [] without throwing in devMode on error', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' }, true);
      const http = getHttpMock();
      http.post.mockRejectedValue(networkError());

      const tools = await client.listTools();
      expect(tools).toEqual([]);
    });

    it('callTool still throws in devMode (error is re-thrown after logging)', async () => {
      const client = new MCPClient({ alias: 'srv', url: 'http://localhost:9000' }, true);
      const http = getHttpMock();
      http.post.mockRejectedValue(networkError('network fail'));

      await expect(client.callTool('tool')).rejects.toThrow('network fail');
    });
  });
});
