import { beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

const {
  generateTextMock,
  generateObjectMock,
  streamTextMock,
  toolMock,
  jsonSchemaMock,
  stepCountIsMock
} = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
  generateObjectMock: vi.fn(),
  streamTextMock: vi.fn(),
  toolMock: vi.fn((definition: Record<string, unknown>) => definition),
  jsonSchemaMock: vi.fn((schema: unknown) => schema),
  stepCountIsMock: vi.fn((count: number) => ({ type: 'step-count', count }))
}));

vi.mock('ai', () => ({
  generateText: generateTextMock,
  generateObject: generateObjectMock,
  streamText: streamTextMock,
  embed: vi.fn(),
  embedMany: vi.fn(),
  tool: toolMock,
  jsonSchema: jsonSchemaMock,
  stepCountIs: stepCountIsMock
}));

vi.mock('@ai-sdk/openai', () => ({
  createOpenAI: () => {
    const fn = Object.assign((modelId: string) => ({ id: `openai:${modelId}` }), {
      chat: (modelId: string) => ({ id: `openai:chat:${modelId}` }),
      embedding: (modelId: string) => ({ id: `openai:embedding:${modelId}` })
    });
    return fn;
  }
}));

vi.mock('@ai-sdk/anthropic', () => ({
  createAnthropic: () => (modelId: string) => ({ id: `anthropic:${modelId}` })
}));

vi.mock('@ai-sdk/google', () => ({
  createGoogleGenerativeAI: () => Object.assign((modelId: string) => ({ id: modelId }), {
    textEmbeddingModel: (modelId: string) => ({ id: modelId })
  })
}));

vi.mock('@ai-sdk/mistral', () => ({
  createMistral: () => Object.assign((modelId: string) => ({ id: modelId }), {
    textEmbeddingModel: (modelId: string) => ({ id: modelId })
  })
}));

vi.mock('@ai-sdk/groq', () => ({ createGroq: () => (modelId: string) => ({ id: modelId }) }));
vi.mock('@ai-sdk/xai', () => ({ createXai: () => (modelId: string) => ({ id: modelId }) }));
vi.mock('@ai-sdk/deepseek', () => ({ createDeepSeek: () => (modelId: string) => ({ id: modelId }) }));
vi.mock('@ai-sdk/cohere', () => ({
  createCohere: () => Object.assign((modelId: string) => ({ id: modelId }), {
    textEmbeddingModel: (modelId: string) => ({ id: modelId })
  })
}));

import { AIClient } from '../src/ai/AIClient.js';
import { rejectsTemperature, temperatureForModel } from '../src/ai/modelParams.js';
import { executeToolCallLoop } from '../src/ai/ToolCalling.js';

describe('rejectsTemperature', () => {
  it.each([
    ['gpt-5', true],
    ['gpt-5-2025-08-07', true],
    ['gpt-5-mini', true],
    ['gpt-5-nano', true],
    ['gpt-5-pro', true],
    ['gpt-5.2-pro', true],
    ['gpt-5.4-pro', true],
    ['gpt-5.5', true],
    ['gpt-5.6-sol', true],
    ['o1', true],
    ['o3-mini', true],
    ['o4-mini', true],
    ['openai/o3-mini', true],
    [' AZURE/O3-MINI ', true],
    ['openai/gpt-5-mini', true],
    ['azure/deployments/GPT-5', true],
    ['gpt-5-codex', true],
    ['openai/gpt-5.1-codex-mini', true],
    ['gpt-5.2-codex', true],
    ['gpt-5.3-codex', true],
    ['claude-sonnet-5-5', true],
    ['claude-opus-4-7', true],
    ['claude-opus-4-8', true],
    ['anthropic/claude-sonnet-5', true],
    ['claude-fable-5-1', true],
    ['claude-haiku-5-5', true],
    ['claude-opus-5-5', true],
    ['gpt-4o', false],
    ['gpt-4.1', false],
    ['openai/gpt-4o', false],
    ['gpt-5.1', false],
    ['gpt-5.2', false],
    ['gpt-5.4', false],
    ['gpt-5-chat-latest', false],
    ['claude-sonnet-4-6', false],
    ['anthropic/claude-opus-4-1', false],
    ['llama-3', false],
    ['gemini-3.8-flash', false],
    ['google/gemini-2.5-flash', false],
    ['omni-1', false],
    ['o', false],
    ['gpt-50', false],
    ['', false]
  ] as const)('%j -> %s', (model, rejected) => {
    expect(rejectsTemperature(model)).toBe(rejected);
    expect(temperatureForModel(model, undefined)).toBeUndefined();
    expect(temperatureForModel(model, 0)).toBe(rejected ? undefined : 0);
    expect(temperatureForModel(model, 0.7)).toBe(rejected ? undefined : 0.7);
  });

  it('treats a missing model as accepting temperature', () => {
    expect(rejectsTemperature(undefined)).toBe(false);
    expect(rejectsTemperature(null)).toBe(false);
    expect(temperatureForModel(undefined, 0.2)).toBe(0.2);
  });
});

const schema = z.object({ value: z.string() });

function sentTemperature(mock: { mock: { calls: unknown[][] } }): unknown {
  const call = mock.mock.calls.at(-1)?.[0] as { temperature?: unknown } | undefined;
  return call?.temperature;
}

describe('AIClient temperature compatibility', () => {
  beforeEach(() => {
    generateTextMock.mockReset();
    generateObjectMock.mockReset();
    streamTextMock.mockReset();
    generateTextMock.mockResolvedValue({ text: 'ok' });
    generateObjectMock.mockResolvedValue({ object: { value: 'ok' } });
    streamTextMock.mockReturnValue({ textStream: (async function* () { yield 'ok'; })() });
  });

  it.each([
    'gpt-5',
    'openai/gpt-5-mini',
    'o3-mini',
    'openai/o4-mini',
    'gpt-5.5',
    'anthropic/claude-opus-5',
    'claude-sonnet-5-20260101'
  ])('omits temperature for %s on generate, structured output, and stream', async (model) => {
    const client = new AIClient({ apiKey: 'test', model, temperature: 0.3, provider: 'openai' });

    await client.generate('hello');
    expect(sentTemperature(generateTextMock)).toBeUndefined();

    await client.generate('hello', { schema, temperature: 0.9 });
    expect(sentTemperature(generateObjectMock)).toBeUndefined();

    await client.stream('hello', { temperature: 0 });
    expect(sentTemperature(streamTextMock)).toBeUndefined();
  });

  it.each([
    'gpt-4o',
    'openai/gpt-5.2',
    'gpt-5.4',
    'omni-moderation',
    'claude-sonnet-4-5'
  ])('keeps a configured temperature for %s', async (model) => {
    const client = new AIClient({ apiKey: 'test', model, temperature: 0.3 });

    await client.generate('hello');
    expect(sentTemperature(generateTextMock)).toBe(0.3);

    await client.generate('hello', { schema, temperature: 0.9 });
    expect(sentTemperature(generateObjectMock)).toBe(0.9);

    await client.stream('hello', { temperature: 0 });
    expect(sentTemperature(streamTextMock)).toBe(0);
  });

  it('uses the per-request model when deciding whether to send temperature', async () => {
    const client = new AIClient({ apiKey: 'test', model: 'gpt-4o', temperature: 0.2 });

    await client.generate('hello', { model: 'gpt-5' });
    expect(sentTemperature(generateTextMock)).toBeUndefined();

    await client.generate('hello', { model: 'gpt-4o-mini', temperature: 0.5 });
    expect(sentTemperature(generateTextMock)).toBe(0.5);
  });
});

describe('tool-call loop temperature compatibility', () => {
  beforeEach(() => {
    generateTextMock.mockReset();
    generateTextMock.mockResolvedValue({
      text: 'done',
      steps: [],
      response: { messages: [] }
    });
  });

  it('omits temperature on the selection pass and the execution pass', async () => {
    generateTextMock
      .mockResolvedValueOnce({
        text: '',
        steps: [{ toolCalls: [{ toolName: 'worker__sum' }] }],
        response: { messages: [] }
      })
      .mockResolvedValueOnce({
        text: 'done',
        steps: [],
        response: { messages: [] }
      });

    await executeToolCallLoop(
      { discover: vi.fn().mockResolvedValue({}), call: vi.fn() } as never,
      'prompt',
      { worker__sum: { description: 'sum', inputSchema: {} } } as never,
      {},
      true,
      () => ({ id: 'model' }),
      { temperature: 0.4 },
      { modelName: 'openai/gpt-5', provider: 'openai' }
    );

    expect(generateTextMock).toHaveBeenCalledTimes(2);
    for (const call of generateTextMock.mock.calls) {
      expect((call[0] as { temperature?: unknown }).temperature).toBeUndefined();
    }
  });

  it('keeps temperature, including zero, for models that accept it', async () => {
    await executeToolCallLoop(
      { call: vi.fn() } as never,
      'prompt',
      { worker__sum: { description: 'sum', inputSchema: {} } } as never,
      {},
      false,
      () => ({ id: 'model' }),
      { temperature: 0 },
      { modelName: 'gpt-4o', provider: 'openai' }
    );

    expect(generateTextMock).toHaveBeenCalledTimes(1);
    expect(sentTemperature(generateTextMock)).toBe(0);
  });
});
