import { describe, it, expect } from 'vitest';
import { Agent } from '../src/agent/Agent.js';

describe('Agent serverless handler', () => {
  it('returns discovery metadata via serverless event', async () => {
    const agent = new Agent({ nodeId: 'svless-discovery', version: '1.2.3', deploymentType: 'serverless', devMode: true });
    agent.reasoner('ping', async () => ({ ok: true }));

    const handler = agent.handler();
    const response = await handler({ path: '/discover' });

    expect(response.statusCode).toBe(200);
    expect(response.body.deployment_type).toBe('serverless');
    expect(response.body.reasoners.map((r: any) => r.id)).toContain('ping');
  });

  it('executes a reasoner through serverless event payload', async () => {
    const agent = new Agent({ nodeId: 'svless-exec', deploymentType: 'serverless', devMode: true });
    agent.reasoner('echo', async (ctx) => ({ echoed: ctx.input.msg, executionId: ctx.executionId }));

    const handler = agent.handler();
    const response = await handler({
      path: '/execute',
      httpMethod: 'POST',
      reasoner: 'echo',
      input: { msg: 'hi' },
      headers: { 'x-execution-id': 'exec-123' }
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toEqual({ echoed: 'hi', executionId: 'exec-123' });
  });

  it('executes a skill when target type is skill', async () => {
    const agent = new Agent({ nodeId: 'svless-skill', deploymentType: 'serverless', devMode: true });
    agent.skill('upper', (ctx) => ({ value: String(ctx.input.text ?? '').toUpperCase() }));

    const handler = agent.handler();
    const response = await handler({
      path: '/execute',
      type: 'skill',
      target: 'upper',
      input: { text: 'serverless' }
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toEqual({ value: 'SERVERLESS' });
  });
});
