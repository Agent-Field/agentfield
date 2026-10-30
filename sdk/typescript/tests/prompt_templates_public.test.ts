import { expect, it, vi } from 'vitest';
import { tool } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { z } from 'zod';
import { AIClient } from '../src/ai/AIClient.js';
import { ReasonerContext } from '../src/context/ReasonerContext.js';

it('preserves default payloads through aiWithTools and the real AI SDK', async () => {
  const answer = (content: any[]) => ({
    content,
    finishReason: { unified: content[0].type === 'tool-call' ? 'tool-calls' : 'stop', raw: 'stop' },
    usage: { inputTokens: { total: 10 }, outputTokens: { total: 2 } },
    warnings: [],
  });
  const model = new MockLanguageModelV3({ doGenerate: [
    answer([{ type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum', input: '{"v":1}' }]),
    answer([{ type: 'text', text: 'done' }]),
  ] as any });
  const aiClient = new AIClient();
  vi.spyOn(aiClient, 'getModel').mockReturnValue(model);
  const agent = { call: vi.fn().mockResolvedValue({ text: 'say "hi"\n雪 {0} %s ${x}' }) };
  const context = new ReasonerContext({
    input: {}, executionId: 'payload-test', agent, aiClient,
    req: {}, res: {}, logger: {}, memory: {}, workflow: {}, did: {},
  } as any);
  const result = await context.aiWithTools('go', {
    tools: { node__sum: tool({ inputSchema: z.object({ v: z.number() }) }) },
    system: 'CALLER', maxTurns: 3,
  });
  expect(result.text).toBe('done');
  expect(agent.call).toHaveBeenCalledExactlyOnceWith('node.sum', { v: 1 });
  expect(model.doGenerateCalls).toHaveLength(2);
  expect(model.doGenerateCalls[0].prompt).toEqual([
    { role: 'system', content: 'CALLER' },
    { role: 'user', content: [{ type: 'text', text: 'go' }] },
  ]);
  expect(model.doGenerateCalls[1].prompt).toEqual([
    ...model.doGenerateCalls[0].prompt,
    { role: 'assistant', content: [
      { type: 'tool-call', toolCallId: 'c1', toolName: 'node__sum', input: { v: 1 } },
    ] },
    { role: 'tool', content: [
      { type: 'tool-result', toolCallId: 'c1', toolName: 'node__sum',
        output: { type: 'json', value: { text: 'say "hi"\n雪 {0} %s ${x}' } } },
    ] },
  ]);
  expect(result.trace.messages?.map(m => m.source)).toEqual([
    'user', 'user', 'assistant', 'sdk.tool_result', 'assistant',
  ]);
});
