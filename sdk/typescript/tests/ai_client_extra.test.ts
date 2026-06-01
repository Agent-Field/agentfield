import { beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

const {
  generateTextMock,
  generateObjectMock,
  streamTextMock,
  embedMock,
  embedManyMock,
  executeWithRetryMock,
  rateLimiterCtorMock
} = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
  generateObjectMock: vi.fn(),
  streamTextMock: vi.fn(),
  embedMock: vi.fn(),
  embedManyMock: vi.fn(),
  executeWithRetryMock: vi.fn(async <T>(fn: () => Promise<T>) => fn()),
  rateLimiterCtorMock: vi.fn()
}));

const {
  createOpenAIMock,
  createAnthropicMock,
  createGoogleMock,
  createMistralMock,
  createGroqMock,
  createXaiMock,
  createDeepSeekMock,
  createCohereMock
} = vi.hoisted(() => {
  const makeProvider = (name: string) => {
    const fn = vi.fn((modelId: string) => ({ id: `${name}:${modelId}` })) as {
      (modelId: string): { id: string };
      chat: ReturnType<typeof vi.fn>;
      embedding: ReturnType<typeof vi.fn>;
      textEmbeddingModel: ReturnType<typeof vi.fn>;
    };
    fn.chat = vi.fn((modelId: string) => ({ id: `${name}:chat:${modelId}` }));
    fn.embedding = vi.fn((modelId: string) => ({ id: `${name}:embedding:${modelId}` }));
    fn.textEmbeddingModel = vi.fn((modelId: string) => ({ id: `${name}:textEmbedding:${modelId}` }));
    return { factory: vi.fn(() => fn), fn };
  };

  return {
    createOpenAIMock: makeProvider('openai'),
    createAnthropicMock: makeProvider('anthropic'),
    createGoogleMock: makeProvider('google'),
    createMistralMock: makeProvider('mistral'),
    createGroqMock: makeProvider('groq'),
    createXaiMock: makeProvider('xai'),
    createDeepSeekMock: makeProvider('deepseek'),
    createCohereMock: makeProvider('cohere')
  };
});

vi.mock('ai', () => ({
  generateText: generateTextMock,
  generateObject: generateObjectMock,
  streamText: streamTextMock,
  embed: embedMock,
  embedMany: embedManyMock
}));

vi.mock('../src/ai/RateLimiter.js', () => ({
  StatelessRateLimiter: vi.fn().mockImplementation((config: object) => {
    rateLimiterCtorMock(config);
    return {
      executeWithRetry: executeWithRetryMock
    };
  })
}));

vi.mock('@ai-sdk/openai', () => ({
  createOpenAI: createOpenAIMock.factory
}));

vi.mock('@ai-sdk/anthropic', () => ({
  createAnthropic: createAnthropicMock.factory
}));

vi.mock('@ai-sdk/google', () => ({
  createGoogleGenerativeAI: createGoogleMock.factory
}));

vi.mock('@ai-sdk/mistral', () => ({
  createMistral: createMistralMock.factory
}));

vi.mock('@ai-sdk/groq', () => ({
  createGroq: createGroqMock.factory
}));

vi.mock('@ai-sdk/xai', () => ({
  createXai: createXaiMock.factory
}));

vi.mock('@ai-sdk/deepseek', () => ({
  createDeepSeek: createDeepSeekMock.factory
}));

vi.mock('@ai-sdk/cohere', () => ({
  createCohere: createCohereMock.factory
}));

import { AIClient } from '../src/ai/AIClient.js';

describe('AIClient extra coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    generateTextMock.mockReset();
    generateObjectMock.mockReset();
    streamTextMock.mockReset();
    embedMock.mockReset();
    embedManyMock.mockReset();
    executeWithRetryMock.mockReset();
    rateLimiterCtorMock.mockReset();

    executeWithRetryMock.mockImplementation(async <T>(fn: () => Promise<T>) => fn());
    generateTextMock.mockResolvedValue({ text: 'plain text' });
    generateObjectMock.mockResolvedValue({ object: { ok: true } });
    streamTextMock.mockReturnValue({
      textStream: (async function* () {
        yield 'one';
        yield 'two';
      })()
    });
    embedMock.mockResolvedValue({ embedding: [0.1, 0.2] });
    embedManyMock.mockResolvedValue({ embeddings: [[1], [2]] });
  });

  it('passes configured defaults through text generation and streaming', async () => {
    const client = new AIClient({
      apiKey: 'test-key',
      provider: 'openai',
      model: 'gpt-4.1-mini',
      temperature: 0.4,
      maxTokens: 321
    });

    await expect(client.generate('hello')).resolves.toBe('plain text');
    const stream = await client.stream('stream me', { system: 'sys', temperature: 0.1, maxTokens: 22 });
    expect(Array.isArray([])).toBe(true);
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(generateTextMock).toHaveBeenCalledWith(expect.objectContaining({
      prompt: 'hello',
      temperature: 0.4,
      maxOutputTokens: 321,
      model: { id: 'openai:gpt-4.1-mini' }
    }));
    expect(streamTextMock).toHaveBeenCalledWith(expect.objectContaining({
      prompt: 'stream me',
      system: 'sys',
      temperature: 0.1,
      maxOutputTokens: 22
    }));
    expect(chunks).toEqual(['one', 'two']);
  });

  it('uses generateObject with JSON repair for structured output', async () => {
    const client = new AIClient({ apiKey: 'test-key' });
    const schema = z.object({ value: z.number() });

    await expect(client.generate('json please', { schema, system: 'sys' })).resolves.toEqual({ ok: true });

    const repair = generateObjectMock.mock.calls[0]?.[0]?.experimental_repairText as
      | ((input: { text: string }) => Promise<string | null>)
      | undefined;
    expect(repair).toBeTypeOf('function');
    await expect(repair?.({ text: '```json\n{"value": 1,}\n```' })).resolves.toBe('{"value": 1}');
    await expect(repair?.({ text: 'prefix {"value": 2} suffix' })).resolves.toBe('{"value": 2}');
    await expect(repair?.({ text: 'not valid json' })).resolves.toBeNull();
  });

  it('supports provider-specific model builders and request-level overrides', async () => {
    const client = new AIClient({
      provider: 'openai',
      apiKey: 'base-key',
      baseUrl: 'https://base.example'
    });

    await client.generate('anthropic', { provider: 'anthropic', model: 'claude-3-7-sonnet' });
    await client.generate('google', { provider: 'google', model: 'gemini-2.5-flash' });
    await client.generate('mistral', { provider: 'mistral', model: 'mistral-large' });
    await client.generate('groq', { provider: 'groq', model: 'llama-3.3-70b' });
    await client.generate('xai', { provider: 'xai', model: 'grok-3' });
    await client.generate('deepseek', { provider: 'deepseek', model: 'deepseek-chat' });
    await client.generate('cohere', { provider: 'cohere', model: 'command-r' });
    await client.generate('openrouter', { provider: 'openrouter', model: 'openai/gpt-4.1-mini' });
    await client.generate('ollama', { provider: 'ollama', model: 'llama3.2', maxTokens: 12 });

    expect(createAnthropicMock.fn).toHaveBeenCalledWith('claude-3-7-sonnet');
    expect(createGoogleMock.fn).toHaveBeenCalledWith('gemini-2.5-flash');
    expect(createMistralMock.fn).toHaveBeenCalledWith('mistral-large');
    expect(createGroqMock.fn).toHaveBeenCalledWith('llama-3.3-70b');
    expect(createXaiMock.fn).toHaveBeenCalledWith('grok-3');
    expect(createDeepSeekMock.fn).toHaveBeenCalledWith('deepseek-chat');
    expect(createCohereMock.fn).toHaveBeenCalledWith('command-r');
    expect(createOpenAIMock.fn.chat).toHaveBeenCalledWith('openai/gpt-4.1-mini');
    expect(createOpenAIMock.fn.chat).toHaveBeenCalledWith('llama3.2');
    expect(createOpenAIMock.factory).toHaveBeenCalledWith(expect.objectContaining({
      apiKey: 'base-key',
      baseURL: 'https://base.example'
    }));
  });

  it('builds embedding models across supported providers and rejects unsupported ones', async () => {
    const openaiClient = new AIClient({ provider: 'openai', apiKey: 'key' });
    const googleClient = new AIClient({ provider: 'google', apiKey: 'key', embeddingModel: 'embed-custom' });
    const mistralClient = new AIClient({ provider: 'mistral', apiKey: 'key' });
    const cohereClient = new AIClient({ provider: 'cohere', apiKey: 'key' });
    const openrouterClient = new AIClient({ provider: 'openrouter', apiKey: 'key' });
    const ollamaClient = new AIClient({ provider: 'ollama' });

    await expect(openaiClient.embed('text')).resolves.toEqual([0.1, 0.2]);
    await expect(googleClient.embedMany(['a', 'b'])).resolves.toEqual([[1], [2]]);
    await mistralClient.embed('text');
    await cohereClient.embed('text');
    await openrouterClient.embed('text');
    await ollamaClient.embed('text');

    expect(createOpenAIMock.fn.embedding).toHaveBeenCalledWith('text-embedding-3-small');
    expect(createGoogleMock.fn.textEmbeddingModel).toHaveBeenCalledWith('embed-custom');
    expect(createMistralMock.fn.textEmbeddingModel).toHaveBeenCalledWith('text-embedding-3-small');
    expect(createCohereMock.fn.textEmbeddingModel).toHaveBeenCalledWith('text-embedding-3-small');
    expect(embedMock).toHaveBeenCalledTimes(5);
    expect(embedManyMock).toHaveBeenCalledTimes(1);

    await expect(new AIClient({ provider: 'anthropic' }).embed('x')).rejects.toThrow(
      'Embedding generation is not supported for anthropic provider'
    );
  });

  it('creates and reuses the rate limiter when retries are enabled', async () => {
    const client = new AIClient({
      provider: 'openai',
      apiKey: 'key',
      rateLimitMaxRetries: 4,
      rateLimitBaseDelay: 2,
      rateLimitMaxDelay: 8,
      rateLimitJitterFactor: 0.1,
      rateLimitCircuitBreakerThreshold: 5,
      rateLimitCircuitBreakerTimeout: 6
    });

    await client.generate('first');
    await client.embed('second');

    expect(rateLimiterCtorMock).toHaveBeenCalledTimes(1);
    expect(rateLimiterCtorMock).toHaveBeenCalledWith({
      maxRetries: 4,
      baseDelay: 2,
      maxDelay: 8,
      jitterFactor: 0.1,
      circuitBreakerThreshold: 5,
      circuitBreakerTimeout: 6
    });
    expect(executeWithRetryMock).toHaveBeenCalledTimes(2);
  });

  it('skips rate limiter retries when disabled and exposes getModel', async () => {
    const client = new AIClient({
      provider: 'ollama',
      enableRateLimitRetry: false
    });

    const model = client.getModel({ model: 'llama3.1' });
    await client.generate('no retry');

    expect(model).toEqual({ id: 'openai:chat:llama3.1' });
    expect(executeWithRetryMock).not.toHaveBeenCalled();
    expect(generateTextMock).toHaveBeenCalledWith(expect.objectContaining({
      model: { id: 'openai:chat:gpt-4o' }
    }));
  });
});
