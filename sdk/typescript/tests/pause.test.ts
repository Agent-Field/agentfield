import { describe, it, expect, vi, afterEach } from 'vitest';
import express from 'express';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import {
  PauseClock,
  PauseManager,
  ApprovalResult,
  installApprovalWebhookRoute
} from '../src/agent/pause.js';

// ---------------------------------------------------------------------------
// ApprovalResult
// ---------------------------------------------------------------------------
describe('ApprovalResult', () => {
  it('exposes approved/changesRequested convenience getters', () => {
    expect(new ApprovalResult({ decision: 'approved' }).approved).toBe(true);
    expect(new ApprovalResult({ decision: 'approved' }).changesRequested).toBe(false);
    expect(new ApprovalResult({ decision: 'request_changes' }).changesRequested).toBe(true);
    expect(new ApprovalResult({ decision: 'rejected' }).approved).toBe(false);
    expect(new ApprovalResult({ decision: 'expired' }).approved).toBe(false);
  });

  it('defaults optional fields', () => {
    const r = new ApprovalResult({ decision: 'approved' });
    expect(r.feedback).toBe('');
    expect(r.executionId).toBe('');
    expect(r.approvalRequestId).toBe('');
    expect(r.rawResponse).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// PauseClock
// ---------------------------------------------------------------------------
describe('PauseClock', () => {
  it('accumulates paused time across intervals', async () => {
    const clock = new PauseClock();
    expect(clock.totalPaused()).toBe(0);
    clock.startPause();
    await new Promise((r) => setTimeout(r, 30));
    // Mid-pause, totalPaused reflects the in-progress interval.
    expect(clock.totalPaused()).toBeGreaterThanOrEqual(20);
    clock.endPause();
    const afterFirst = clock.totalPaused();
    expect(afterFirst).toBeGreaterThanOrEqual(20);
    // When not paused, totalPaused is stable.
    await new Promise((r) => setTimeout(r, 20));
    expect(clock.totalPaused()).toBe(afterFirst);
    // A second pause adds to the cumulative total.
    clock.startPause();
    await new Promise((r) => setTimeout(r, 20));
    clock.endPause();
    expect(clock.totalPaused()).toBeGreaterThan(afterFirst);
  });

  it('is idempotent on double start / double end', () => {
    const clock = new PauseClock();
    clock.startPause();
    clock.startPause(); // no-op — does not reset the start time
    clock.endPause();
    clock.endPause(); // no-op — does not double-count
    expect(clock.totalPaused()).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// PauseManager
// ---------------------------------------------------------------------------
describe('PauseManager', () => {
  it('resolves a pending pause by approvalRequestId', async () => {
    const mgr = new PauseManager();
    const p = mgr.register('req-1', 'exec-1');
    expect(mgr.pendingCount()).toBe(1);

    const ok = mgr.resolve('req-1', new ApprovalResult({ decision: 'approved', feedback: 'lgtm' }));
    expect(ok).toBe(true);
    const result = await p;
    expect(result.approved).toBe(true);
    expect(result.feedback).toBe('lgtm');
    expect(mgr.pendingCount()).toBe(0);
  });

  it('resolves by executionId as a fallback', async () => {
    const mgr = new PauseManager();
    const p = mgr.register('req-2', 'exec-2');
    const ok = mgr.resolveByExecutionId('exec-2', new ApprovalResult({ decision: 'rejected' }));
    expect(ok).toBe(true);
    expect((await p).decision).toBe('rejected');
  });

  it('is idempotent: a second register returns the same promise', () => {
    const mgr = new PauseManager();
    const a = mgr.register('req-3', 'exec-3');
    const b = mgr.register('req-3', 'exec-3');
    expect(a).toBe(b);
    expect(mgr.pendingCount()).toBe(1);
  });

  it('returns false when resolving an unknown request', () => {
    const mgr = new PauseManager();
    expect(mgr.resolve('nope', new ApprovalResult({ decision: 'approved' }))).toBe(false);
    expect(mgr.resolveByExecutionId('nope', new ApprovalResult({ decision: 'approved' }))).toBe(false);
  });

  it('cancelAll resolves every pending pause with a cancelled result', async () => {
    const mgr = new PauseManager();
    const p1 = mgr.register('req-a', 'exec-a');
    const p2 = mgr.register('req-b', 'exec-b');
    mgr.cancelAll();
    expect(mgr.pendingCount()).toBe(0);
    expect((await p1).decision).toBe('cancelled');
    expect((await p2).decision).toBe('cancelled');
  });
});

// ---------------------------------------------------------------------------
// installApprovalWebhookRoute
// ---------------------------------------------------------------------------
describe('installApprovalWebhookRoute', () => {
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

  function startAgentApp(manager: PauseManager): Promise<string> {
    const app = express();
    app.use(express.json());
    installApprovalWebhookRoute(app, manager);
    return new Promise((resolve) => {
      server = app.listen(0, '127.0.0.1', () => {
        const addr = server.address() as AddressInfo;
        resolve(`http://127.0.0.1:${addr.port}`);
      });
    });
  }

  it('resolves the matching pause and replies { status, resolved }', async () => {
    const manager = new PauseManager();
    const pending = manager.register('req-webhook', 'exec-webhook');
    const base = await startAgentApp(manager);

    const res = await fetch(`${base}/webhooks/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        execution_id: 'exec-webhook',
        approval_request_id: 'req-webhook',
        decision: 'approved',
        feedback: 'ship it',
        response: JSON.stringify({ reviewer: 'alice' })
      })
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: 'received', resolved: true });

    const result = await pending;
    expect(result.approved).toBe(true);
    expect(result.feedback).toBe('ship it');
    expect(result.rawResponse).toEqual({ reviewer: 'alice' });
  });

  it('falls back to execution_id when approval_request_id is absent', async () => {
    const manager = new PauseManager();
    const pending = manager.register('req-fallback', 'exec-fallback');
    const base = await startAgentApp(manager);

    const res = await fetch(`${base}/webhooks/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ execution_id: 'exec-fallback', decision: 'request_changes' })
    });

    expect((await res.json())).toEqual({ status: 'received', resolved: true });
    expect((await pending).changesRequested).toBe(true);
  });

  it('replies resolved:false when there is no matching pending pause', async () => {
    const manager = new PauseManager();
    const base = await startAgentApp(manager);

    const res = await fetch(`${base}/webhooks/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ execution_id: 'unknown', approval_request_id: 'unknown', decision: 'approved' })
    });

    expect((await res.json())).toEqual({ status: 'received', resolved: false });
  });

  it('accepts an object response payload directly', async () => {
    const manager = new PauseManager();
    const pending = manager.register('req-obj', 'exec-obj');
    const base = await startAgentApp(manager);

    await fetch(`${base}/webhooks/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        approval_request_id: 'req-obj',
        decision: 'approved',
        response: { note: 'inline object' }
      })
    });

    expect((await pending).rawResponse).toEqual({ note: 'inline object' });
  });
});
