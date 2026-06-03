import { afterEach, describe, expect, it, vi } from 'vitest';

import { Agent } from '../src/agent/Agent.js';
import { AgentRouter } from '../src/router/AgentRouter.js';
import { ReasonerContext } from '../src/context/ReasonerContext.js';
import { SkillContext } from '../src/context/SkillContext.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('AgentRouter dispatch integration', () => {
  it('dispatches included router reasoners through agent.call()', async () => {
    const router = new AgentRouter({ prefix: 'ops' });
    const reasoner = vi.fn((ctx: ReasonerContext<{ message: string }>) => ({
      echoed: ctx.input.message,
      hasAi: typeof ctx.ai === 'function'
    }));
    router.reasoner('echo', reasoner);

    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });
    agent.includeRouter(router);

    const result = await agent.call('local.ops_echo', { message: 'hello' });

    expect(result).toEqual({ echoed: 'hello', hasAi: true });
    expect(reasoner).toHaveBeenCalledTimes(1);
    expect(reasoner.mock.calls[0]?.[0]).toBeInstanceOf(ReasonerContext);
    expect(reasoner.mock.calls[0]?.[0]).not.toBeInstanceOf(SkillContext);
  });

  it('dispatches included router skills through the serverless execute handler', async () => {
    const router = new AgentRouter({ prefix: 'ops' });
    const skill = vi.fn((ctx: SkillContext<{ text: string }>) => ({
      upper: ctx.input.text.toUpperCase(),
      hasAi: 'ai' in (ctx as object)
    }));
    router.skill('format', skill);

    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });
    agent.includeRouter(router);

    const response = await agent.handler()({
      path: '/execute',
      body: { skill: 'ops_format', input: { text: 'hello' } }
    } as any);

    expect(response).toMatchObject({
      statusCode: 200,
      body: { upper: 'HELLO', hasAi: false }
    });
    expect(skill).toHaveBeenCalledTimes(1);
    expect(skill.mock.calls[0]?.[0]).toBeInstanceOf(SkillContext);
    expect(skill.mock.calls[0]?.[0]).not.toBeInstanceOf(ReasonerContext);
  });

  it('treats colon-only targets as literal local reasoner names', async () => {
    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });
    agent.reasoner('reasoner:foo', () => ({ ok: true }));

    await expect(agent.call('reasoner:foo', {})).resolves.toEqual({ ok: true });
  });

  it('prefers a reasoner over a skill for /execute targets unless type=skill is explicit', async () => {
    const reasoner = vi.fn(() => ({ kind: 'reasoner' }));
    const skill = vi.fn(() => ({ kind: 'skill' }));

    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });
    agent.reasoner('shared', reasoner);
    agent.skill('shared', skill);

    const defaultResponse = await agent.handler()({
      path: '/execute',
      body: { target: 'shared', input: {} }
    } as any);
    const explicitSkillResponse = await agent.handler()({
      path: '/execute',
      body: { skill: 'shared', type: 'skill', input: {} }
    } as any);

    expect(defaultResponse).toMatchObject({
      statusCode: 200,
      body: { kind: 'reasoner' }
    });
    expect(explicitSkillResponse).toMatchObject({
      statusCode: 200,
      body: { kind: 'skill' }
    });
    expect(reasoner).toHaveBeenCalledTimes(1);
    expect(skill).toHaveBeenCalledTimes(1);
  });

  it('does not enforce input schemas at dispatch time in the current implementation', async () => {
    const skill = vi.fn((ctx: SkillContext<Record<string, unknown>>) => ctx.input);
    const router = new AgentRouter();
    router.skill('needs-input', skill, {
      inputSchema: {
        type: 'object',
        required: ['x'],
        properties: {
          x: { type: 'string' }
        }
      }
    });

    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });
    agent.includeRouter(router);

    const response = await agent.handler()({
      path: '/execute',
      body: { skill: 'needs-input', input: {} }
    } as any);

    expect(response).toMatchObject({
      statusCode: 200,
      body: {}
    });
    expect(skill).toHaveBeenCalledTimes(1);
  });

  it('returns a plain 404 payload for unknown targets', async () => {
    const agent = new Agent({ nodeId: 'local', devMode: true, didEnabled: false });

    const response = await agent.handler()({
      path: '/execute',
      body: { target: 'missing', input: {} }
    } as any);

    expect(response).toEqual({
      statusCode: 404,
      headers: { 'content-type': 'application/json' },
      body: { error: 'Reasoner not found: missing' }
    });
  });
});
