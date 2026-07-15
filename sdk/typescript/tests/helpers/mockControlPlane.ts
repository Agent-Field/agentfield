import http from 'node:http';
import express from 'express';
import type { AddressInfo } from 'node:net';
import type { Agent } from '../../src/agent/Agent.js';

/**
 * A configurable mock control plane for exercising the SDK's client/agent HTTP
 * contracts. Records every request and lets a test canned per-route responses.
 * The async dispatch path is detached, so assertions poll via {@link waitFor}
 * rather than assuming synchronous ordering.
 */
export interface RecordedCall {
  method: string;
  path: string;
  body: any;
  headers: http.IncomingHttpHeaders;
}

export class MockControlPlane {
  server!: http.Server;
  calls: RecordedCall[] = [];
  private handlers: Array<{
    method: string;
    test: (path: string) => boolean;
    respond: (call: RecordedCall) => { status: number; body: any };
  }> = [];

  on(
    method: string,
    match: (path: string) => boolean,
    respond: (call: RecordedCall) => { status: number; body: any }
  ): this {
    this.handlers.push({ method, test: match, respond });
    return this;
  }

  async start(): Promise<string> {
    const app = express();
    app.use(express.json());
    app.use((req, res) => {
      const call: RecordedCall = {
        method: req.method,
        path: req.path,
        body: req.body,
        headers: req.headers
      };
      this.calls.push(call);
      const handler = this.handlers.find((h) => h.method === req.method && h.test(req.path));
      if (handler) {
        const { status, body } = handler.respond(call);
        res.status(status).json(body);
      } else {
        // Default OK so best-effort calls (heartbeat, logs, events, register,
        // notes) don't fail the agent in devMode.
        res.status(200).json({ ok: true });
      }
    });
    return new Promise((resolve) => {
      this.server = app.listen(0, '127.0.0.1', () => {
        const addr = this.server.address() as AddressInfo;
        resolve(`http://127.0.0.1:${addr.port}`);
      });
    });
  }

  find(method: string, pathIncludes: string): RecordedCall | undefined {
    return this.calls.find((c) => c.method === method && c.path.includes(pathIncludes));
  }

  findAll(method: string, pathIncludes: string): RecordedCall[] {
    return this.calls.filter((c) => c.method === method && c.path.includes(pathIncludes));
  }

  async waitFor(method: string, pathIncludes: string, timeoutMs = 3000): Promise<RecordedCall> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const hit = this.find(method, pathIncludes);
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 10));
    }
    throw new Error(`timed out waiting for ${method} ${pathIncludes}`);
  }

  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server?.listening) {
        this.server.closeAllConnections();
        this.server.close(() => resolve());
      } else {
        resolve();
      }
    });
  }
}

/** Listen an agent's Express app on an ephemeral port; returns its base URL. */
export async function listenAgent(agent: Agent): Promise<string> {
  const server = await new Promise<http.Server>((resolve) => {
    const s = agent.app.listen(0, '127.0.0.1', () => resolve(s));
  });
  (agent as any).__testServer = server;
  const addr = server.address() as AddressInfo;
  return `http://127.0.0.1:${addr.port}`;
}

export async function closeAgent(agent: Agent): Promise<void> {
  const server: http.Server | undefined = (agent as any).__testServer;
  if (server?.listening) {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}
