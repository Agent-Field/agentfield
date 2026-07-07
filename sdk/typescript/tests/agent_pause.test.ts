import { describe, it, expect, afterEach } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { ApprovalResult } from '../src/agent/pause.js';
import { MockControlPlane, listenAgent, closeAgent, type RecordedCall } from './helpers/mockControlPlane.js';

describe('ctx.pause() / Agent.pause() end-to-end', () => {
  let cp: MockControlPlane;
  let agent: Agent;

  afterEach(async () => {
    await closeAgent(agent);
    await cp?.stop();
  });

  it('pauses to WAITING, then resumes when the approval webhook arrives', async () => {
    cp = new MockControlPlane();
    // request-approval → CP acknowledges pending.
    cp.on('POST', (p) => p.includes('/request-approval'), () => ({
      status: 200,
      body: { approval_request_id: 'req-pause-1', status: 'pending' }
    }));
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'pauser', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('review', async (ctx) => {
      const result = await ctx.pause({ approvalRequestId: 'req-pause-1', expiresInHours: 1 });
      return { decision: result.decision, feedback: result.feedback, approved: result.approved };
    });
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-pause-1', 'X-Run-ID': 'run-exec-pause-1' },
      body: JSON.stringify({})
    });
    expect(res.status).toBe(202);

    // The reasoner should transition the execution to waiting via request-approval.
    const approvalCall = await cp.waitFor('POST', '/request-approval');
    expect(approvalCall.body.approval_request_id).toBe('req-pause-1');
    expect(approvalCall.body.callback_url).toContain('/webhooks/approval');

    // No terminal status yet — the reasoner is parked.
    expect(cp.find('POST', '/status')).toBeUndefined();

    // Simulate the control plane delivering the human decision to the agent.
    const webhookRes = await fetch(`${agentUrl}/webhooks/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        execution_id: 'exec-pause-1',
        approval_request_id: 'req-pause-1',
        decision: 'approved',
        feedback: 'go ahead'
      })
    });
    expect((await webhookRes.json())).toEqual({ status: 'received', resolved: true });

    // The reasoner resumes and reports its terminal result.
    const statusCall = await cp.waitFor('POST', '/executions/exec-pause-1/status');
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ decision: 'approved', feedback: 'go ahead', approved: true });
  });

  it('returns an expired result when the pause times out', async () => {
    cp = new MockControlPlane();
    cp.on('POST', (p) => p.includes('/request-approval'), () => ({
      status: 200,
      body: { approval_request_id: 'req-expire', status: 'pending' }
    }));
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'pauser', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('review', async (ctx) => {
      const result = await ctx.pause({ approvalRequestId: 'req-expire', timeoutMs: 100 });
      return { decision: result.decision };
    });
    const agentUrl = await listenAgent(agent);

    await fetch(`${agentUrl}/reasoners/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-expire', 'X-Run-ID': 'run-exec-expire' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-expire/status', 3000);
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ decision: 'expired' });
  });

  it('Agent.pause() throws when it cannot reach the control plane', async () => {
    cp = new MockControlPlane();
    cp.on('POST', (p) => p.includes('/request-approval'), () => ({
      status: 500,
      body: { error: 'boom' }
    }));
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'pauser', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('review', async (ctx) => {
      await ctx.pause({ approvalRequestId: 'req-err', timeoutMs: 500 });
      return { ok: true };
    });
    const agentUrl = await listenAgent(agent);

    await fetch(`${agentUrl}/reasoners/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-err', 'X-Run-ID': 'run-exec-err' },
      body: JSON.stringify({})
    });

    // request-approval failed → reasoner throws → reported as failed.
    const statusCall = await cp.waitFor('POST', '/executions/exec-err/status', 3000);
    expect(statusCall.body.status).toBe('failed');
  });
});

describe('multi-hop pause propagation (awaiter-status cascade)', () => {
  let cp: MockControlPlane;
  let agent: Agent;

  afterEach(async () => {
    await closeAgent(agent);
    await cp?.stop();
  });

  it('pushes the parent to WAITING while an awaited child is waiting, then back to running', async () => {
    // Child execution status timeline: waiting → waiting → running (succeeded).
    const statuses = ['waiting', 'waiting', 'succeeded'];
    let idx = 0;

    cp = new MockControlPlane();
    cp.on('POST', (p) => p.includes('/execute/async/'), () => ({
      status: 202,
      body: { execution_id: 'child-exec-1' }
    }));
    cp.on('GET', (p) => p.includes('/executions/child-exec-1'), () => {
      const status = statuses[Math.min(idx++, statuses.length - 1)];
      const body: Record<string, unknown> = { execution_id: 'child-exec-1', status };
      if (status === 'succeeded') body.result = { child: 'done' };
      return { status: 200, body };
    });
    let awaiterCalls: RecordedCall[] = [];
    cp.on('POST', (p) => p.includes('/awaiter-status'), (call) => {
      awaiterCalls.push(call);
      return { status: 200, body: { status: call.body.status, applied: true } };
    });
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'parent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    // Parent reasoner calls a remote child. The child is emulated purely by the
    // mock control plane's status timeline above.
    agent.reasoner('orchestrate', async (ctx) => {
      const childResult = await ctx.call('worker.doWork', { task: 'x' });
      return { childResult };
    });
    const agentUrl = await listenAgent(agent);

    await fetch(`${agentUrl}/reasoners/orchestrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'parent-exec-1', 'X-Run-ID': 'run-parent-exec-1' },
      body: JSON.stringify({})
    });

    // Parent completes once the child reaches succeeded.
    const statusCall = await cp.waitFor('POST', '/executions/parent-exec-1/status', 5000);
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ childResult: { child: 'done' } });

    // The parent must have cascaded: at least one waiting, then a running, both
    // scoped to the parent's own execution id.
    const waiting = awaiterCalls.filter((c) => c.body.status === 'waiting');
    const running = awaiterCalls.filter((c) => c.body.status === 'running');
    expect(waiting.length).toBeGreaterThanOrEqual(1);
    expect(running.length).toBeGreaterThanOrEqual(1);
    expect(waiting[0].path).toContain('/executions/parent-exec-1/awaiter-status');
  });
});
