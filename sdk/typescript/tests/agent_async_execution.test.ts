import { describe, it, expect, afterEach } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { MockControlPlane, listenAgent, closeAgent } from './helpers/mockControlPlane.js';

describe('async-execution dispatch', () => {
  let cp: MockControlPlane;
  let agent: Agent;

  afterEach(async () => {
    await closeAgent(agent);
    await cp?.stop();
  });

  it('202-acks a control-plane dispatch and reports succeeded out-of-band', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({
      nodeId: 'async-agent',
      agentFieldUrl: cpUrl,
      didEnabled: false,
      devMode: true
    });
    agent.reasoner('echo', async (ctx) => ({ echoed: (ctx.input as any).value }));
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/echo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-async-1', 'X-Run-ID': 'run-exec-async-1' },
      body: JSON.stringify({ value: 42 })
    });

    // Fast-ack: 202 with the execution id, connection not held for the handler.
    expect(res.status).toBe(202);
    expect(await res.json()).toEqual({ status: 'processing', execution_id: 'exec-async-1' });

    // Terminal result delivered out-of-band via /status.
    const statusCall = await cp.waitFor('POST', '/executions/exec-async-1/status');
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ echoed: 42 });
    expect(statusCall.body.reasoner).toBe('echo');
    expect(typeof statusCall.body.duration_ms).toBe('number');
  });

  it('reports failed when a detached reasoner throws', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'async-agent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('boom', async () => {
      throw new Error('kaboom');
    });
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/boom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-async-2', 'X-Run-ID': 'run-exec-async-2' },
      body: JSON.stringify({})
    });
    expect(res.status).toBe(202);

    const statusCall = await cp.waitFor('POST', '/executions/exec-async-2/status');
    expect(statusCall.body.status).toBe('failed');
    expect(statusCall.body.error).toContain('kaboom');
  });

  it('wraps a non-object result so the control plane accepts it', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'async-agent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('scalar', async () => 'just-a-string' as any);
    const agentUrl = await listenAgent(agent);

    await fetch(`${agentUrl}/reasoners/scalar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-async-3', 'X-Run-ID': 'run-exec-async-3' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-async-3/status');
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ result: 'just-a-string' });
  });

  it('runs synchronously (no 202) when there is no X-Execution-ID header', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'async-agent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('echo', async (ctx) => ({ echoed: (ctx.input as any).value }));
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/echo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: 7 })
    });

    // Synchronous path: 200 with the result inline, no /status callback.
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ echoed: 7 });
    expect(cp.find('POST', '/status')).toBeUndefined();
  });

  it('runs synchronously (inline result) when X-Execution-ID is present but X-Run-ID is not', async () => {
    // Reproduces the legacy synchronous invoke endpoint
    // (POST /api/v1/reasoners/{node}.{reasoner}), which sets X-Execution-ID but
    // NOT X-Run-ID for long-running agents and forwards the agent's response
    // verbatim. The agent must return the result inline (200), not a 202 marker.
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({ nodeId: 'async-agent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    agent.reasoner('echo', async (ctx) => ({ echoed: (ctx.input as any).value }));
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/echo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-legacy-1' },
      body: JSON.stringify({ value: 5 })
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ echoed: 5 });
    expect(cp.find('POST', '/status')).toBeUndefined();
  });

  it('runs synchronously when asyncExecution is disabled, even with the header', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({
      nodeId: 'async-agent',
      agentFieldUrl: cpUrl,
      didEnabled: false,
      devMode: true,
      asyncExecution: false
    });
    agent.reasoner('echo', async (ctx) => ({ echoed: (ctx.input as any).value }));
    const agentUrl = await listenAgent(agent);

    const res = await fetch(`${agentUrl}/reasoners/echo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-sync-1', 'X-Run-ID': 'run-exec-sync-1' },
      body: JSON.stringify({ value: 9 })
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ echoed: 9 });
  });

  it('aborts and reports reasoner_timeout when the active budget is exceeded', async () => {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();

    agent = new Agent({
      nodeId: 'async-agent',
      agentFieldUrl: cpUrl,
      didEnabled: false,
      devMode: true,
      executionBudgetMs: 150 // tiny budget so the watchdog fires quickly
    });
    agent.reasoner('slow', async (ctx) => {
      // Cooperative: respect the abort signal so the watchdog can stop us.
      await new Promise((resolve, reject) => {
        const t = setTimeout(resolve, 5000);
        ctx.signal.addEventListener('abort', () => {
          clearTimeout(t);
          reject(new Error('aborted'));
        });
      });
      return { done: true };
    });
    const agentUrl = await listenAgent(agent);

    await fetch(`${agentUrl}/reasoners/slow`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-timeout-1', 'X-Run-ID': 'run-exec-timeout-1' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-timeout-1/status', 4000);
    expect(statusCall.body.status).toBe('failed');
    expect(statusCall.body.error_details).toEqual({ reason: 'reasoner_timeout' });
  });
});
