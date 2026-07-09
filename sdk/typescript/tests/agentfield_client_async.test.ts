import { describe, it, expect, afterEach } from 'vitest';
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

  it('retries on failure and returns false after exhausting retries', async () => {
    const { base, recs } = await startServer((req, res) => res.status(500).json({ error: 'down' }));
    const ok = await makeClient(base).reportExecutionResult('exec-1', { status: 'failed', error: 'x' }, 2);
    expect(ok).toBe(false);
    expect(recs.length).toBe(2);
  });
});
