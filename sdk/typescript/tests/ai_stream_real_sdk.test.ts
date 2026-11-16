import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModelV3 } from 'ai/test';

const modelHolder = vi.hoisted(() => ({ factory: undefined as undefined | (() => unknown) }));

vi.mock('@ai-sdk/openai', () => {
  const make = () => modelHolder.factory!();
  return {
    createOpenAI: () =>
      Object.assign(make, { chat: make, embedding: make })
  };
});

import { AIClient } from '../src/ai/AIClient.js';

function rateLimit() {
  return Object.assign(new Error('Too Many Requests'), { statusCode: 429 });
}

function usage() {
  return {
    inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 1, text: 1, reasoning: 0 }
  };
}

const config = {
  apiKey: 'test',
  provider: 'openai' as const,
  model: 'gpt-4o',
  rateLimitBaseDelay: 0.001,
  rateLimitMaxDelay: 0.002,
  rateLimitMaxRetries: 3
};

async function collect(stream: AsyncIterable<string>) {
  let out = '';
  for await (const chunk of stream) out += chunk;
  return out;
}

describe('AIClient.stream with the real streamText', () => {
  it('retries a 429 raised by doStream before the first chunk', async () => {
    let calls = 0;
    modelHolder.factory = () =>
      new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          if (calls < 3) throw rateLimit();
          return {
            stream: new ReadableStream({
              start(c) {
                c.enqueue({ type: 'stream-start', warnings: [] });
                c.enqueue({ type: 'text-start', id: '1' });
                c.enqueue({ type: 'text-delta', id: '1', delta: 'hi' });
                c.enqueue({ type: 'text-end', id: '1' });
                c.enqueue({ type: 'finish', finishReason: { unified: 'stop', raw: 'stop' }, usage: usage() });
                c.close();
              }
            })
          };
        }
      });
    const client = new AIClient(config);
    expect(await collect(await client.stream('hi'))).toBe('hi');
    expect(calls).toBe(3);
  });

  it('surfaces the error instead of an empty stream when retries run out', async () => {
    let calls = 0;
    modelHolder.factory = () =>
      new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          throw rateLimit();
        }
      });
    const client = new AIClient({ ...config, rateLimitMaxRetries: 2 });
    await expect(collect(await client.stream('hi'))).rejects.toThrow('Too Many Requests');
    expect(calls).toBe(3);
  });

  it('does not start a request while the circuit breaker is open', async () => {
    let calls = 0;
    modelHolder.factory = () =>
      new MockLanguageModelV3({
        doStream: async () => {
          calls++;
          throw rateLimit();
        }
      });
    const client = new AIClient({
      ...config,
      rateLimitMaxRetries: 0,
      rateLimitCircuitBreakerThreshold: 1,
      rateLimitCircuitBreakerTimeout: 60
    });
    await expect(collect(await client.stream('hi'))).rejects.toThrow('Too Many Requests');
    expect(calls).toBe(1);
    await expect(collect(await client.stream('hi'))).rejects.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toBe(1);
  });
});
