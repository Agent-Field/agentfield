import { describe, it, expect, afterEach } from 'vitest';
import { Agent } from '../src/agent/Agent.js';
import { USAGE_ENVELOPE_KEY } from '../src/usage/costTracker.js';
import { MockControlPlane, listenAgent, closeAgent } from './helpers/mockControlPlane.js';

/**
 * Usage transport contract:
 * - async dispatch: the terminal /status callback carries a top-level "usage"
 *   object (all terminal states), omitted entirely when nothing was recorded;
 * - sync dispatch: plain-object 200 bodies carry the summary under the
 *   reserved "__agentfield_usage__" sibling key, non-object bodies and a
 *   user's own "usage" key are never touched.
 */
describe('usage transport', () => {
  let cp: MockControlPlane;
  let agent: Agent;

  afterEach(async () => {
    await closeAgent(agent);
    await cp?.stop();
  });

  async function startAgent(register: (agent: Agent) => void): Promise<string> {
    cp = new MockControlPlane();
    const cpUrl = await cp.start();
    agent = new Agent({ nodeId: 'usage-agent', agentFieldUrl: cpUrl, didEnabled: false, devMode: true });
    register(agent);
    return listenAgent(agent);
  }

  it('attaches usage to the async succeeded status callback', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('priced', async (ctx) => {
        ctx.costTracker.record({
          model: 'anthropic/claude-opus-4-8',
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
          costUsd: 0.0123,
          costSource: 'provider',
          reasonerName: ctx.reasonerId
        });
        return { ok: true };
      });
    });

    await fetch(`${agentUrl}/reasoners/priced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-usage-1', 'X-Run-ID': 'run-usage-1' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-usage-1/status');
    expect(statusCall.body.status).toBe('succeeded');
    expect(statusCall.body.result).toEqual({ ok: true });
    expect(statusCall.body.usage).toEqual({
      total_cost_usd: 0.0123,
      total_input_tokens: 100,
      total_output_tokens: 50,
      total_tokens: 150,
      entries: [
        {
          source: 'llm',
          provider: 'anthropic',
          model: 'anthropic/claude-opus-4-8',
          harness: null,
          reasoner: 'priced',
          input_tokens: 100,
          output_tokens: 50,
          cache_read_tokens: 0,
          cache_creation_tokens: 0,
          total_tokens: 150,
          cost_usd: 0.0123,
          cost_source: 'provider'
        }
      ]
    });
  });

  it('attaches usage to a failed status callback (tokens consumed before the throw)', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('doomed', async (ctx) => {
        ctx.costTracker.record({ model: 'gpt-4o', inputTokens: 10, outputTokens: 2 });
        throw new Error('kaboom');
      });
    });

    await fetch(`${agentUrl}/reasoners/doomed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-usage-2', 'X-Run-ID': 'run-usage-2' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-usage-2/status');
    expect(statusCall.body.status).toBe('failed');
    expect(statusCall.body.error).toContain('kaboom');
    expect(statusCall.body.usage.total_input_tokens).toBe(10);
    expect(statusCall.body.usage.entries).toHaveLength(1);
  });

  it('omits the usage key entirely when nothing was recorded', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('plain', async () => ({ ok: true }));
    });

    await fetch(`${agentUrl}/reasoners/plain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-usage-3', 'X-Run-ID': 'run-usage-3' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-usage-3/status');
    expect(statusCall.body.status).toBe('succeeded');
    expect('usage' in statusCall.body).toBe(false);
  });

  it('rolls a nested local agent.call() usage into the parent execution report', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('child', async (ctx) => {
        // Capture sites read the ambient execution's reasoner id — the child
        // execution's, not the parent's, even though the tracker is shared.
        ctx.costTracker.record({ model: 'gpt-4o-mini', inputTokens: 7, reasonerName: ctx.reasonerId });
        return 'child-done';
      });
      a.reasoner('parent', async (ctx) => {
        ctx.costTracker.record({ model: 'gpt-4o', inputTokens: 3, reasonerName: ctx.reasonerId });
        await ctx.call('usage-agent.child', {});
        return { ok: true };
      });
    });

    await fetch(`${agentUrl}/reasoners/parent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Execution-ID': 'exec-usage-4', 'X-Run-ID': 'run-usage-4' },
      body: JSON.stringify({})
    });

    const statusCall = await cp.waitFor('POST', '/executions/exec-usage-4/status');
    expect(statusCall.body.status).toBe('succeeded');
    const entries = statusCall.body.usage.entries;
    expect(entries).toHaveLength(2);
    expect(entries.map((e: any) => [e.model, e.reasoner])).toEqual([
      ['gpt-4o', 'parent'],
      ['gpt-4o-mini', 'child']
    ]);
    expect(statusCall.body.usage.total_input_tokens).toBe(10);
  });

  it('merges usage into a sync 200 object body under the reserved envelope key', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('sync-priced', async (ctx) => {
        ctx.costTracker.record({ model: 'gpt-4o', inputTokens: 20, outputTokens: 4 });
        // A user result may carry its own "usage" key — that is payload.
        return { answer: 42, usage: { user: 'data' } };
      });
    });

    const res = await fetch(`${agentUrl}/reasoners/sync-priced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.answer).toBe(42);
    expect(body.usage).toEqual({ user: 'data' });
    expect(body[USAGE_ENVELOPE_KEY].total_input_tokens).toBe(20);
    expect(body[USAGE_ENVELOPE_KEY].entries).toHaveLength(1);
    // Sync path responds inline — no /status callback.
    expect(cp.find('POST', '/status')).toBeUndefined();
  });

  it('leaves non-object sync results unchanged even when usage was recorded', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('sync-scalar', async (ctx) => {
        ctx.costTracker.record({ model: 'gpt-4o', inputTokens: 5 });
        return 'just-a-string' as any;
      });
    });

    const res = await fetch(`${agentUrl}/reasoners/sync-scalar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toBe('just-a-string');
  });

  it('leaves the sync 200 body untouched when nothing was recorded', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('sync-plain', async () => ({ ok: true }));
    });

    const res = await fetch(`${agentUrl}/reasoners/sync-plain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    expect(await res.json()).toEqual({ ok: true });
  });

  it('carries the envelope on the serverless /execute path too', async () => {
    const agentUrl = await startAgent((a) => {
      a.reasoner('serverless', async (ctx) => {
        ctx.costTracker.record({ model: 'gpt-4o', totalTokens: 30, costUsd: 0.002, costSource: 'provider' });
        return { done: true };
      });
    });

    const res = await fetch(`${agentUrl}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target: 'serverless', input: {} })
    });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.done).toBe(true);
    expect(body[USAGE_ENVELOPE_KEY].total_cost_usd).toBe(0.002);
  });
});
