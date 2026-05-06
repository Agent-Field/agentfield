import { describe, it, expect } from 'vitest';
import express from 'express';
import http from 'node:http';

import { Agent } from '../src/agent/Agent.js';
import { CancelRegistry, installCancelRoute } from '../src/agent/cancel.js';

/**
 * Lightweight stand-in for supertest. Boots an Express app on an ephemeral
 * port, fires a single request, returns parsed JSON body + status.
 */
async function postJson(
  app: express.Express,
  path: string,
  body: unknown = {},
  headers: Record<string, string> = {}
): Promise<{ status: number; body: any }> {
  const server = http.createServer(app);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
  const port = (server.address() as { port: number }).port;
  try {
    const payload = JSON.stringify(body);
    const resp = await new Promise<{ status: number; body: any }>((resolve, reject) => {
      const req = http.request(
        {
          host: '127.0.0.1',
          port,
          path,
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'content-length': Buffer.byteLength(payload).toString(),
            ...headers
          }
        },
        (res) => {
          let data = '';
          res.setEncoding('utf8');
          res.on('data', (chunk) => (data += chunk));
          res.on('end', () => {
            const status = res.statusCode ?? 0;
            try {
              resolve({ status, body: data ? JSON.parse(data) : {} });
            } catch {
              resolve({ status, body: data });
            }
          });
        }
      );
      req.on('error', reject);
      req.write(payload);
      req.end();
    });
    return resp;
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

describe('CancelRegistry', () => {
  it('register returns a fresh AbortController each time', () => {
    const reg = new CancelRegistry();
    const a = reg.register('exec-1');
    const b = reg.register('exec-2');
    expect(a.controller).not.toBe(b.controller);
    expect(a.controller.signal.aborted).toBe(false);
    expect(reg.size()).toBe(2);
  });

  it('cancel aborts the registered controller and removes it', () => {
    const reg = new CancelRegistry();
    const { controller } = reg.register('exec-1');
    expect(reg.cancel('exec-1')).toBe(true);
    expect(controller.signal.aborted).toBe(true);
    expect(reg.size()).toBe(0);
  });

  it('cancel returns false for unknown id', () => {
    const reg = new CancelRegistry();
    expect(reg.cancel('missing')).toBe(false);
    expect(reg.cancel('')).toBe(false);
  });

  it('release deregisters without aborting', () => {
    const reg = new CancelRegistry();
    const { controller, release } = reg.register('exec-1');
    release();
    expect(reg.size()).toBe(0);
    expect(controller.signal.aborted).toBe(false);
    // Subsequent cancel is a no-op.
    expect(reg.cancel('exec-1')).toBe(false);
  });

  it('release is idempotent', () => {
    const reg = new CancelRegistry();
    const { release } = reg.register('exec-1');
    release();
    release();
    expect(reg.size()).toBe(0);
  });

  it('register with empty id produces a usable but unregistered controller', () => {
    const reg = new CancelRegistry();
    const { controller } = reg.register('');
    expect(reg.size()).toBe(0);
    expect(controller.signal.aborted).toBe(false);
  });

  it('replacing the same id evicts the prior registration on release', () => {
    const reg = new CancelRegistry();
    const first = reg.register('exec-1');
    const second = reg.register('exec-1');
    // Releasing the FIRST entry must not delete the SECOND registration.
    first.release();
    expect(reg.size()).toBe(1);
    expect(reg.cancel('exec-1')).toBe(true);
    expect(second.controller.signal.aborted).toBe(true);
  });

  it('double cancel is idempotent', () => {
    const reg = new CancelRegistry();
    reg.register('exec-1');
    expect(reg.cancel('exec-1')).toBe(true);
    // Second cancel on already-cancelled (and now removed) id returns false.
    expect(reg.cancel('exec-1')).toBe(false);
  });
});

describe('installCancelRoute', () => {
  function makeApp() {
    const app = express();
    app.use(express.json());
    const reg = new CancelRegistry();
    installCancelRoute(app, reg, { info: () => {} });
    return { app, reg };
  }

  it('returns 200 with cancelled:true for an active execution', async () => {
    const { app, reg } = makeApp();
    const { controller } = reg.register('exec-active');
    const resp = await postJson(app, '/_internal/executions/exec-active/cancel');
    expect(resp.status).toBe(200);
    expect(resp.body).toEqual({ cancelled: true, execution_id: 'exec-active' });
    expect(controller.signal.aborted).toBe(true);
  });

  it('returns 200 with cancelled:false + reason for unknown execution', async () => {
    const { app } = makeApp();
    const resp = await postJson(app, '/_internal/executions/exec-missing/cancel');
    expect(resp.status).toBe(200);
    expect(resp.body).toEqual({
      cancelled: false,
      execution_id: 'exec-missing',
      reason: 'execution_not_active'
    });
  });

  it('forwards the cancel reason to AbortController.abort', async () => {
    const { app, reg } = makeApp();
    const { controller } = reg.register('exec-reason');
    await postJson(
      app,
      '/_internal/executions/exec-reason/cancel',
      { reason: 'user clicked cancel' }
    );
    expect(controller.signal.aborted).toBe(true);
    expect(controller.signal.reason).toBe('user clicked cancel');
  });

  it('logs an info line with the X-AgentField-Source header on success', async () => {
    const app = express();
    app.use(express.json());
    const reg = new CancelRegistry();
    const logs: Array<{ message: string; meta?: any }> = [];
    installCancelRoute(app, reg, {
      info: (message, meta) => logs.push({ message, meta })
    });
    reg.register('exec-logged');
    await postJson(
      app,
      '/_internal/executions/exec-logged/cancel',
      {},
      { 'x-agentfield-source': 'cancel-dispatcher' }
    );
    expect(logs).toHaveLength(1);
    expect(logs[0].message).toBe('cancel-callback fired');
    expect(logs[0].meta).toMatchObject({
      executionId: 'exec-logged',
      source: 'cancel-dispatcher'
    });
  });
});

describe('Agent integration', () => {
  it('passes a non-aborted signal to reasoner handlers by default', async () => {
    const agent = new Agent({ nodeId: 'test-cancel-agent', devMode: true });
    let capturedSignal: AbortSignal | undefined;
    agent.reasoner('inspect-signal', async (ctx) => {
      capturedSignal = ctx.signal;
      return { ok: true };
    });

    // Drive the reasoner through Agent.execute() which goes through
    // runReasoner — same path the HTTP route uses.
    const result = await (agent as any).executeInvocation({
      targetName: 'inspect-signal',
      targetType: 'reasoner',
      input: {},
      metadata: { executionId: 'exec-it-1' },
      respond: false
    });
    expect(result).toEqual({ ok: true });
    expect(capturedSignal).toBeDefined();
    expect(capturedSignal!.aborted).toBe(false);
  });

  it('aborts ctx.signal mid-flight when the cancel route fires', async () => {
    const agent = new Agent({ nodeId: 'test-cancel-agent', devMode: true });
    let capturedSignal: AbortSignal | undefined;
    let started = false;
    let aborted = false;

    agent.reasoner('long-running', async (ctx) => {
      capturedSignal = ctx.signal;
      started = true;
      // Wait for the abort to propagate via the signal listener.
      await new Promise<void>((resolve) => {
        ctx.signal.addEventListener('abort', () => {
          aborted = true;
          resolve();
        });
      });
      return { aborted: true };
    });

    // Start the reasoner in the background — runReasoner registers the
    // controller against execution_id "exec-it-2" before invoking the
    // handler. The inflight promise resolves when the abort fires.
    const inflight = (agent as any).executeInvocation({
      targetName: 'long-running',
      targetType: 'reasoner',
      input: {},
      metadata: { executionId: 'exec-it-2' },
      respond: false
    });

    // Wait for the handler to actually start (and register).
    while (!started) {
      await new Promise((r) => setTimeout(r, 5));
    }

    const resp = await postJson(agent.app, '/_internal/executions/exec-it-2/cancel');
    expect(resp.status).toBe(200);
    expect(resp.body.cancelled).toBe(true);

    const result = await inflight;
    expect(result).toEqual({ aborted: true });
    expect(aborted).toBe(true);
    expect(capturedSignal!.aborted).toBe(true);
  });
});
