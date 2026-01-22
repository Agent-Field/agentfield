import axios, { type AxiosInstance } from 'axios';
import type { MCPServerConfig } from '../types/agent.js';
import type { MCPTool } from '../types/mcp.js';
import { httpAgent, httpsAgent } from '../utils/httpAgents.js';

export class MCPClient {
  readonly alias: string;
  readonly baseUrl: string;
  readonly transport: 'http' | 'bridge';
  private readonly http: AxiosInstance;
  private readonly devMode: boolean;
  private lastHealthy = false;

  constructor(config: MCPServerConfig, devMode?: boolean) {
    if (!config.alias) {
      throw new Error('MCP server alias is required');
    }
    if (!config.url && !config.port) {
      throw new Error(`MCP server "${config.alias}" requires a url or port`);
    }

    this.alias = config.alias;
    this.transport = config.transport ?? 'http';
    this.baseUrl = (config.url ?? `http://localhost:${config.port}`).replace(/\/$/, '');
    this.http = axios.create({
      baseURL: this.baseUrl,
      headers: config.headers,
      timeout: 30000,
      httpAgent,
      httpsAgent
    });
    this.devMode = Boolean(devMode);
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.http.get('/health');
      this.lastHealthy = true;
      return true;
    } catch (err) {
      this.lastHealthy = false;
      if (this.devMode) {
        console.warn(`MCP health check failed for ${this.alias}:`, err instanceof Error ? err.message : err);
      }
      return false;
    }
  }

  async listTools(): Promise<MCPTool[]> {
    try {
      if (this.transport === 'bridge') {
        const res = await this.http.post('/mcp/tools/list');
        const tools = res.data?.tools ?? [];
        return this.normalizeTools(tools);
      }

      const res = await this.http.post('/mcp/v1', {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/list',
        params: {}
      });
      const tools = res.data?.result?.tools ?? [];
      return this.normalizeTools(tools);
    } catch (err) {
      if (this.devMode) {
        console.warn(`MCP listTools failed for ${this.alias}:`, err instanceof Error ? err.message : err);
      }
      return [];
    }
  }

  async callTool(toolName: string, arguments_: Record<string, any> = {}): Promise<any> {
    if (!toolName) {
      throw new Error('toolName is required');
    }

    try {
      if (this.transport === 'bridge') {
        const res = await this.http.post('/mcp/tools/call', {
          tool_name: toolName,
          arguments: arguments_
        });
        return res.data?.result ?? res.data;
      }

      const res = await this.http.post('/mcp/v1', {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: { name: toolName, arguments: arguments_ }
      });

      if (res.data?.error) {
        throw new Error(String(res.data.error?.message ?? res.data.error));
      }

      if (res.data?.result !== undefined) {
        return res.data.result;
      }

      return res.data;
    } catch (err) {
      if (this.devMode) {
        console.warn(`MCP callTool failed for ${this.alias}.${toolName}:`, err instanceof Error ? err.message : err);
      }
      throw err;
    }
  }

  get lastHealthStatus() {
    return this.lastHealthy;
  }

  private normalizeTools(tools: any[]): MCPTool[] {
    return (tools ?? []).map((tool) => ({
      name: tool?.name ?? 'unknown',
      description: tool?.description,
      inputSchema: tool?.inputSchema ?? tool?.input_schema,
      input_schema: tool?.input_schema
    }));
  }
}
