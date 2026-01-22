import http from 'node:http';
import https from 'node:https';

/**
 * Shared HTTP agents with connection pooling to prevent socket exhaustion.
 *
 * These agents are shared across all SDK HTTP clients (AgentFieldClient,
 * MemoryClient, DidClient, MCPClient) to ensure consistent connection
 * pooling behavior and prevent socket leaks.
 *
 * Configuration:
 * - keepAlive: true - Reuse connections instead of creating new ones
 * - maxSockets: 10 - Max connections per host (IPv4/IPv6 counted separately)
 * - maxTotalSockets: 50 - Total connections across all hosts (prevents exhaustion)
 * - maxFreeSockets: 5 - Idle sockets to keep for reuse
 */
export const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 10,
  maxTotalSockets: 50,
  maxFreeSockets: 5
});

export const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 10,
  maxTotalSockets: 50,
  maxFreeSockets: 5
});
