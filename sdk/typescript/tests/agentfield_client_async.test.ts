import { describe, it, expect, afterEach } from 'vitest';
import crypto from 'node:crypto';
import http from 'node:http';
import express from 'express';
import type { AddressInfo } from 'node:net';
import { AgentFieldClient } from '../src/client/AgentFieldClient.js';
import { PauseClock } from '../src/agent/pause.js';

let server: http.Server;

afterEach(() => {
  return new Promise<void>((resolve) => {
    if (server?.listening) {
      server.closeAllConnections();
      server.close(() => resolve());
    } else {
      resolve();
    }
  });
});

interface Rec {
  method: string;
  path: string;
  body: any;
}

async function startServer(
  route: (req: express.Request, res: express.Response, recs: Rec[]) => void
): Promise<{ base: string; recs: Rec[] }> {
  const recs: Rec[] = [];
  const app = express();
  app.use(express.json());
  app.use((req, res) => {
    recs.push({ method: req.method, path: req.path, body: req.body });
    route(req, res, recs);
  });
  const base = await new Promise<string>((resolve) => {
    server = app.listen(0, '127.0.0.1', () => {
      const addr = server.address() as AddressInfo;
      resolve(`http://127.0.0.1:${addr.port}`);
    });
  });
  return { base, recs };
}

function makeClient(base: string): AgentFieldClient {
  return new AgentFieldClient({ nodeId: 'node-1', agentFieldUrl: base, didEnabled: false });
}

describe('AgentFieldClient.executeAsync', () => {
  it('returns the execution_id from the 202 response body', async () => {
    const { base, recs } = await startServer((req, res) => {
      res.status(202).json({ execution_id: 'exec-xyz' });
    });
    const id = await makeClient(base).executeAsync('node.reasoner', { a: 1 }, { runId: 'run-1' });
    expect(id).toBe('exec-xyz');
    expect(recs[0].path).toBe('/api/v1/execute/async/node.reasoner');
    expect(recs[0].body).toEqual({ input: { a: 1 } });
  });

  it('falls back to the X-Execution-ID response header', async () => {
    const { base } = await startServer((req, res) => {
      res.setHeader('X-Execution-ID', 'exec-from-header');
      res.status(202).json({ ok: true });
    });
    const id = await makeClient(base).executeAsync('node.reasoner', {});
    expect(id).toBe('exec-from-header');
  });

  it('throws a structured error on a 4xx', async () => {
    const { base } = await startServer((req, res) => {
      res.status(403).json({ error: 'permission_denied', message: 'nope' });
    });
    await expect(makeClient(base).executeAsync('node.reasoner', {})).rejects.toThrow(/403/);
  });
});

describe('AgentFieldClient.getExecutionStatus', () => {
  it('maps the snake_case status snapshot', async () => {
    const { base } = await startServer((req, res) => {
      res.status(200).json({
        execution_id: 'exec-1',
        status: 'succeeded',
        status_reason: 'done',
        result: { k: 'v' },
        duration_ms: 1234
      });
    });
    const snap = await makeClient(base).getExecutionStatus('exec-1');
    expect(snap).toMatchObject({
      executionId: 'exec-1',
      status: 'succeeded',
      statusReason: 'done',
      result: { k: 'v' },
      durationMs: 1234
    });
  });
});

describe('AgentFieldClient.waitForExecutionResult', () => {
  it('polls until succeeded and unwraps the result', async () => {
    const statuses = ['running', 'running', 'succeeded'];
    let i = 0;
    const { base } = await startServer((req, res) => {
      const status = statuses[Math.min(i++, statuses.length - 1)];
      const body: any = { execution_id: 'exec-1', status };
      if (status === 'succeeded') body.result = { answer: 42 };
      res.status(200).json(body);
    });
    const result = await makeClient(base).waitForExecutionResult('exec-1', { pollIntervalMs: 10, maxIntervalMs: 20 });
    expect(result).toEqual({ answer: 42 });
  });

  it('throws when the execution fails', async () => {
    const { base } = await startServer((req, res) => {
      res.status(200).json({ execution_id: 'exec-1', status: 'failed', error: 'blew up' });
    });
    await expect(
      makeClient(base).waitForExecutionResult('exec-1', { pollIntervalMs: 10 })
    ).rejects.toThrow(/failed: blew up/);
  });

  it('fires onChildWaiting/onChildRunning and pauses the clock across a WAITING window', async () => {
    const statuses = ['waiting', 'waiting', 'running', 'succeeded'];
    let i = 0;
    const { base } = await startServer((req, res) => {
      const status = statuses[Math.min(i++, statuses.length - 1)];
      const body: any = { execution_id: 'exec-1', status };
      if (status === 'succeeded') body.result = { ok: true };
      res.status(200).json(body);
    });

    const clock = new PauseClock();
    const events: string[] = [];
    const result = await makeClient(base).waitForExecutionResult('exec-1', {
      pollIntervalMs: 15,
      maxIntervalMs: 15,
      pauseClock: clock,
      onChildWaiting: () => { events.push('waiting'); },
      onChildRunning: () => { events.push('running'); }
    });

    expect(result).toEqual({ ok: true });
    // Exactly one waiting edge and one running edge (transitions, not per-poll).
    expect(events).toEqual(['waiting', 'running']);
    // The clock recorded some paused time from the WAITING window.
    expect(clock.totalPaused()).toBeGreaterThan(0);
  });
});

describe('AgentFieldClient.notifyAwaiterStatus', () => {
  it('posts { status, reason } to the awaiter-status endpoint', async () => {
    const { base, recs } = await startServer((req, res) => {
      res.status(200).json({ status: req.body.status, applied: true });
    });
    await makeClient(base).notifyAwaiterStatus('exec-9', 'waiting', 'awaiting child c1');
    expect(recs[0].path).toBe('/api/v1/agents/node-1/executions/exec-9/awaiter-status');
    expect(recs[0].body).toEqual({ status: 'waiting', reason: 'awaiting child c1' });
  });

  it('rejects an invalid status without hitting the network', async () => {
    const { base, recs } = await startServer((req, res) => res.status(200).json({}));
    await expect(
      makeClient(base).notifyAwaiterStatus('exec-9', 'bogus' as any)
    ).rejects.toThrow(/must be 'waiting' or 'running'/);
    expect(recs.length).toBe(0);
  });
});

describe('AgentFieldClient.reportExecutionResult', () => {
  it('wraps a non-object result and posts the terminal status', async () => {
    const { base, recs } = await startServer((req, res) => res.status(200).json({ ok: true }));
    const ok = await makeClient(base).reportExecutionResult('exec-1', {
      status: 'succeeded',
      result: 'scalar-value',
      durationMs: 5
    });
    expect(ok).toBe(true);
    expect(recs[0].path).toBe('/api/v1/executions/exec-1/status');
    expect(recs[0].body.status).toBe('succeeded');
    expect(recs[0].body.result).toEqual({ result: 'scalar-value' });
  });

  it('passes an object result through unchanged', async () => {
    const { base, recs } = await startServer((req, res) => res.status(200).json({ ok: true }));
    await makeClient(base).reportExecutionResult('exec-1', {
      status: 'succeeded',
      result: { already: 'object' }
    });
    expect(recs[0].body.result).toEqual({ already: 'object' });
  });

  it('serializes a cancellation status reason for the control plane', async () => {
    const { base, recs } = await startServer((req, res) => res.status(200).json({ ok: true }));
    await makeClient(base).reportExecutionResult('exec-1', {
      status: 'cancelled',
      error: 'cancelled during graceful shutdown',
      statusReason: 'shutdown timeout exceeded'
    });
    expect(recs[0].body).toMatchObject({
      status: 'cancelled',
      error: 'cancelled during graceful shutdown',
      status_reason: 'shutdown timeout exceeded'
    });
  });

  it('retries on failure and returns false after exhausting retries', async () => {
    const { base, recs } = await startServer((req, res) => res.status(500).json({ error: 'down' }));
    const ok = await makeClient(base).reportExecutionResult('exec-1', { status: 'failed', error: 'x' }, 2);
    expect(ok).toBe(false);
    expect(recs.length).toBe(2);
  });

  it.each([400, 401, 403, 404, 409, 422])(
    'does not retry a non-retryable %i response',
    async (code) => {
      const { base, recs } = await startServer((req, res) => res.status(code).json({ error: 'nope' }));
      const ok = await makeClient(base).reportExecutionResult('exec-1', { status: 'succeeded' });
      expect(ok).toBe(false);
      expect(recs.length).toBe(1);
    }
  );

  it.each([408, 429, 503])('retries a transient %i response', async (code) => {
    const { base, recs } = await startServer((req, res, all) =>
      all.length === 1 ? res.status(code).json({ error: 'busy' }) : res.status(200).json({ ok: true })
    );
    const ok = await makeClient(base).reportExecutionResult('exec-1', { status: 'succeeded' }, 2);
    expect(ok).toBe(true);
    expect(recs.length).toBe(2);
  });

  it('retries transport errors', async () => {
    const { base } = await startServer((req, res) => res.status(200).json({ ok: true }));
    const port = new URL(base).port;
    await new Promise<void>((resolve) => server.close(() => resolve()));
    const ok = await makeClient(`http://127.0.0.1:${port}`).reportExecutionResult(
      'exec-1',
      { status: 'succeeded' },
      2
    );
    expect(ok).toBe(false);
  });

  it('signs each attempt with a fresh DID nonce', async () => {
    const nonces: string[] = [];
    const { base } = await startServer((req, res, all) => {
      nonces.push(String(req.header('X-DID-Nonce')));
      if (all.length === 1) res.status(503).json({ error: 'busy' });
      else res.status(200).json({ ok: true });
    });
    const { privateKey } = crypto.generateKeyPairSync('ed25519');
    const client = new AgentFieldClient({
      nodeId: 'node-1',
      agentFieldUrl: base,
      did: 'did:key:test',
      privateKeyJwk: JSON.stringify(privateKey.export({ format: 'jwk' }))
    });
    const ok = await client.reportExecutionResult('exec-1', { status: 'succeeded' }, 2);
    expect(ok).toBe(true);
    expect(nonces).toHaveLength(2);
    expect(nonces[0]).not.toBe('undefined');
    expect(nonces[0]).not.toBe(nonces[1]);
  });
});

describe('AgentFieldClient.updateExecutionStatus', () => {
  it('retries a transient 503 progress update', async () => {
    const { base, recs } = await startServer((req, res, all) =>
      all.length === 1 ? res.status(503).json({ error: 'busy' }) : res.status(200).json({ ok: true })
    );
    await makeClient(base).updateExecutionStatus('exec-1', { status: 'running', progress: 40 }, 2);
    expect(recs.length).toBe(2);
    expect(recs[1].body).toMatchObject({ status: 'running', progress: 40 });
  });

  it('throws without retrying a 409 conflict', async () => {
    const { base, recs } = await startServer((req, res) => res.status(409).json({ error: 'terminal' }));
    await expect(
      makeClient(base).updateExecutionStatus('exec-1', { status: 'running', progress: 40 })
    ).rejects.toThrow(/409/);
    expect(recs.length).toBe(1);
  });
});
