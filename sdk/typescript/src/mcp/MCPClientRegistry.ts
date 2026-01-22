import type { MCPServerConfig } from '../types/agent.js';
import type { MCPHealthSummary } from '../types/mcp.js';
import { MCPClient } from './MCPClient.js';

export class MCPClientRegistry {
  private readonly clients = new Map<string, MCPClient>();
  private readonly devMode: boolean;

  constructor(devMode?: boolean) {
    this.devMode = Boolean(devMode);
  }

  register(config: MCPServerConfig): MCPClient {
    const client = new MCPClient(config, this.devMode);
    this.clients.set(config.alias, client);
    return client;
  }

  get(alias: string) {
    return this.clients.get(alias);
  }

  list(): MCPClient[] {
    return Array.from(this.clients.values());
  }

  clear(): void {
    this.clients.clear();
  }

  async healthSummary(): Promise<MCPHealthSummary> {
    if (!this.clients.size) {
      return {
        status: 'disabled',
        totalServers: 0,
        healthyServers: 0,
        servers: []
      };
    }

    const results = await Promise.all(
      Array.from(this.clients.values()).map(async (client) => {
        const healthy = await client.healthCheck();
        return {
          alias: client.alias,
          baseUrl: client.baseUrl,
          transport: client.transport,
          healthy
        };
      })
    );

    const healthyCount = results.filter((r) => r.healthy).length;
    const status: MCPHealthSummary['status'] =
      healthyCount === 0 ? 'degraded' : healthyCount === results.length ? 'ok' : 'degraded';

    return {
      status,
      totalServers: results.length,
      healthyServers: healthyCount,
      servers: results
    };
  }
}
