import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModelV3 } from 'ai/test';
import { AIClient } from '../src/ai/AIClient.js';
import { ReasonerContext } from '../src/context/ReasonerContext.js';

const answer = (content: any[]) => ({
  content,
  finishReason: { unified: content[0].type === 'tool-call' ? 'tool-calls' : 'stop', raw: 'stop' },
  usage: { inputTokens: { total: 10 }, outputTokens: { total: 2 } },
  warnings: [],
});

describe.each(['eager', 'lazy'] as const)('tool generation defaults (%s)', (schemaHydration) => {
  it.each([
    { config: { temperature: 0.7, maxTokens: 512 }, overrides: {}, expected: { temperature: 0.7, maxOutputTokens: 512 } },
    { config: { temperature: 0.7, maxTokens: 512 }, overrides: { temperature: 0, maxTokens: 128 }, expected: { temperature: 0, maxOutputTokens: 128 } },
    { config: {}, overrides: {}, expected: { temperature: undefined, maxOutputTokens: undefined } },
  ])('passes $expected to every model turn', async ({ config, overrides, expected }) => {
    const model = new MockLanguageModelV3({ doGenerate: [
      answer([{ type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum', input: '{}' }]),
      answer([{ type: 'text', text: 'done' }]),
    ] as any });
    const aiClient = new AIClient(config);
    vi.spyOn(aiClient, 'getModel').mockReturnValue(model);
    const agent = {
      call: vi.fn().mockResolvedValue({ total: 3 }),
      discover: vi.fn().mockResolvedValue({ json: { capabilities: [{
        agentId: 'node', reasoners: [{ id: 'sum', invocationTarget: 'node:sum', inputSchema: { type: 'object', properties: {} } }], skills: [],
      }] } }),
    };
    const context = new ReasonerContext({
      input: {}, executionId: 'defaults-test', agent, aiClient,
      req: {}, res: {}, logger: {}, memory: {}, workflow: {}, did: {},
    } as any);

    const result = await context.ai('go', { tools: { schemaHydration }, ...overrides });

    expect(result).toMatchObject({ text: 'done' });
    expect(model.doGenerateCalls).toHaveLength(2);
    for (const call of model.doGenerateCalls) {
      expect(call.temperature).toBe(expected.temperature);
      expect(call.maxOutputTokens).toBe(expected.maxOutputTokens);
    }
  });
});
