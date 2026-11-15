import { beforeEach, describe, expect, it, vi } from 'vitest';

const { streamTextMock } = vi.hoisted(() => ({ streamTextMock: vi.fn() }));

vi.mock('ai', () => ({
  generateText: vi.fn(),
  generateObject: vi.fn(),
  streamText: streamTextMock,
  embed: vi.fn(),
  embedMany: vi.fn(),
  tool: (definition: Record<string, unknown>) => definition,
  jsonSchema: (schema: unknown) => schema,
  stepCountIs: (count: number) => ({ type: 'step-count', count })
}));

vi.mock('@ai-sdk/openai', () => ({
  createOpenAI: () =>
    Object.assign((modelId: string) => ({ id: `openai:${modelId}` }), {
      chat: (modelId: string) => ({ id: `openai:chat:${modelId}` }),
      embedding: (modelId: string) => ({ id: `openai:embedding:${modelId}` })
    })
}));

import { AIClient } from '../src/ai/AIClient.js';

function rateLimitError() {
  return Object.assign(new Error('Too Many Requests'), { statusCode: 429 });
}

function failingStream(error: unknown) {
  return {
    fullStream: (async function* () {
      yield { type: 'error', error };
    })()
  };
}

function okStream(chunks: string[]) {
  return {
    fullStream: (async function* () {
      for (const chunk of chunks) yield { type: 'text-delta', text: chunk };
    })()
  };
}

async function collect(stream: AsyncIterable<string>) {
  const out: string[] = [];
  for await (const chunk of stream) out.push(chunk);
  return out;
}

const config = {
  apiKey: 'test',
  provider: 'openai' as const,
  model: 'gpt-4o',
  rateLimitBaseDelay: 0.001,
  rateLimitMaxDelay: 0.002
};

describe('AIClient.stream rate limiting', () => {
  beforeEach(() => streamTextMock.mockReset());

  it('retries a rate-limited stream before the first chunk', async () => {
    streamTextMock
      .mockReturnValueOnce(failingStream(rateLimitError()))
      .mockReturnValueOnce(failingStream(rateLimitError()))
      .mockReturnValueOnce(okStream(['a', 'b']));
    const client = new AIClient({ ...config, rateLimitMaxRetries: 3 });

    expect(await collect(await client.stream('hi'))).toEqual(['a', 'b']);
    expect(streamTextMock).toHaveBeenCalledTimes(3);
  });

  it('honors rateLimitMaxRetries and then surfaces the error', async () => {
    streamTextMock.mockImplementation(() => failingStream(rateLimitError()));
    const client = new AIClient({ ...config, rateLimitMaxRetries: 2 });

    await expect(collect(await client.stream('hi'))).rejects.toThrow('Too Many Requests');
    expect(streamTextMock).toHaveBeenCalledTimes(3);
  });

  it('does not retry when retries are disabled', async () => {
    streamTextMock.mockImplementation(() => failingStream(rateLimitError()));
    const client = new AIClient({ ...config, enableRateLimitRetry: false });

    await expect(collect(await client.stream('hi'))).rejects.toThrow('Too Many Requests');
    expect(streamTextMock).toHaveBeenCalledTimes(1);
  });

  it('never retries after a chunk was emitted', async () => {
    streamTextMock.mockReturnValue({
      fullStream: (async function* () {
        yield { type: 'text-delta', text: 'a' };
        yield { type: 'error', error: rateLimitError() };
      })()
    });
    const client = new AIClient({ ...config, rateLimitMaxRetries: 3 });
    const seen: string[] = [];

    await expect(
      (async () => {
        for await (const chunk of await client.stream('hi')) seen.push(chunk);
      })()
    ).rejects.toThrow('Too Many Requests');
    expect(seen).toEqual(['a']);
    expect(streamTextMock).toHaveBeenCalledTimes(1);
  });

  it('does not retry non-rate-limit errors', async () => {
    streamTextMock.mockReturnValue(failingStream(new Error('boom')));
    const client = new AIClient({ ...config, rateLimitMaxRetries: 3 });

    await expect(collect(await client.stream('hi'))).rejects.toThrow('boom');
    expect(streamTextMock).toHaveBeenCalledTimes(1);
  });
});
