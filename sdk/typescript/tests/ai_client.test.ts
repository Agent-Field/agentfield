import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AIClient } from '../src/ai/AIClient.js';

// Hoist all mocks
const {
  generateTextMock,
  generateObjectMock,
  streamTextMock,
  embedMock,
  embedManyMock
} = vi.hoisted(() => ({
  generateTextMock: vi.fn(),
  generateObjectMock: vi.fn(),
  streamTextMock: vi.fn(),
  embedMock: vi.fn(),
  embedManyMock: vi.fn()
}));

// Provider factory mocks
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
  const makeMockProvider = (name: string) => {
    const embeddingFn = vi.fn((modelId: string) => ({ id: `${name}-embed-${modelId}` }));
    const textEmbeddingFn = vi.fn((modelId: string) => ({ id: `${name}-text-embed-${modelId}` }));
    const modelFn = vi.fn((modelId: string) => ({ id: `${name}-${modelId}` })) as any;
    modelFn.embedding = embeddingFn;
    modelFn.textEmbeddingModel = textEmbeddingFn;
    return { factory: vi.fn(() => modelFn), modelFn, embeddingFn, textEmbeddingFn };
  };

  return {
    createOpenAIMock: makeMockProvider('openai'),
    createAnthropicMock: makeMockProvider('anthropic'),
    createGoogleMock: makeMockProvider('google'),
    createMistralMock: makeMockProvider('mistral'),
    createGroqMock: makeMockProvider('groq'),
    createXaiMock: makeMockProvider('xai'),
    createDeepSeekMock: makeMockProvider('deepseek'),
    createCohereMock: makeMockProvider('cohere')
  };
});

// Mock all AI SDK modules
vi.mock('ai', () => ({
  generateText: generateTextMock,
  generateObject: generateObjectMock,
  streamText: streamTextMock,
  embed: embedMock,
  embedMany: embedManyMock
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

describe('AIClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    generateTextMock.mockResolvedValue({ text: 'mocked response' });
    generateObjectMock.mockResolvedValue({ object: { key: 'value' } });
    embedMock.mockResolvedValue({ embedding: [0.1, 0.2, 0.3] });
    embedManyMock.mockResolvedValue({ embeddings: [[0.1], [0.2]] });
  });

  describe('provider selection for text generation', () => {
    it('creates OpenAI provider by default', async () => {
      const client = new AIClient({ apiKey: 'test-key' });
      await client.generate('test prompt');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createOpenAIMock.modelFn).toHaveBeenCalledWith('gpt-4o');
    });

    it('creates Anthropic provider when specified', async () => {
      const client = new AIClient({ provider: 'anthropic', apiKey: 'test-key', model: 'claude-3-opus' });
      await client.generate('test prompt');

      expect(createAnthropicMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createAnthropicMock.modelFn).toHaveBeenCalledWith('claude-3-opus');
    });

    it('creates Google provider when specified', async () => {
      const client = new AIClient({ provider: 'google', apiKey: 'test-key', model: 'gemini-2.0-flash' });
      await client.generate('test prompt');

      expect(createGoogleMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createGoogleMock.modelFn).toHaveBeenCalledWith('gemini-2.0-flash');
    });

    it('creates Mistral provider when specified', async () => {
      const client = new AIClient({ provider: 'mistral', apiKey: 'test-key', model: 'mistral-large' });
      await client.generate('test prompt');

      expect(createMistralMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createMistralMock.modelFn).toHaveBeenCalledWith('mistral-large');
    });

    it('creates Groq provider when specified', async () => {
      const client = new AIClient({ provider: 'groq', apiKey: 'test-key', model: 'llama-3.1-70b' });
      await client.generate('test prompt');

      expect(createGroqMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createGroqMock.modelFn).toHaveBeenCalledWith('llama-3.1-70b');
    });

    it('creates xAI provider when specified', async () => {
      const client = new AIClient({ provider: 'xai', apiKey: 'test-key', model: 'grok-2' });
      await client.generate('test prompt');

      expect(createXaiMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createXaiMock.modelFn).toHaveBeenCalledWith('grok-2');
    });

    it('creates DeepSeek provider when specified', async () => {
      const client = new AIClient({ provider: 'deepseek', apiKey: 'test-key', model: 'deepseek-chat' });
      await client.generate('test prompt');

      expect(createDeepSeekMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createDeepSeekMock.modelFn).toHaveBeenCalledWith('deepseek-chat');
    });

    it('creates Cohere provider when specified', async () => {
      const client = new AIClient({ provider: 'cohere', apiKey: 'test-key', model: 'command-r-plus' });
      await client.generate('test prompt');

      expect(createCohereMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: 'test-key' })
      );
      expect(createCohereMock.modelFn).toHaveBeenCalledWith('command-r-plus');
    });

    it('creates OpenRouter with correct baseURL', async () => {
      const client = new AIClient({ provider: 'openrouter', apiKey: 'test-key', model: 'anthropic/claude-3' });
      await client.generate('test prompt');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'test-key',
          baseURL: 'https://openrouter.ai/api/v1'
        })
      );
    });

    it('creates Ollama with correct baseURL and dummy key', async () => {
      const client = new AIClient({ provider: 'ollama', model: 'llama3' });
      await client.generate('test prompt');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'ollama',
          baseURL: 'http://localhost:11434/v1'
        })
      );
    });

    it('allows custom baseUrl to override defaults', async () => {
      const client = new AIClient({
        provider: 'openrouter',
        apiKey: 'test-key',
        baseUrl: 'https://custom.openrouter.ai/api/v1'
      });
      await client.generate('test prompt');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://custom.openrouter.ai/api/v1'
        })
      );
    });
  });

  describe('embedding support', () => {
    const noEmbeddingProviders = ['anthropic', 'xai', 'deepseek', 'groq'] as const;

    it.each(noEmbeddingProviders)('throws for %s embeddings', async (provider) => {
      const client = new AIClient({ provider, apiKey: 'test' });
      await expect(client.embed('test')).rejects.toThrow(`Embedding generation is not supported for ${provider} provider`);
    });

    it('uses OpenAI embedding model for openai provider', async () => {
      const client = new AIClient({ provider: 'openai', apiKey: 'test' });
      await client.embed('test text');

      expect(createOpenAIMock.embeddingFn).toHaveBeenCalledWith('text-embedding-3-small');
      expect(embedMock).toHaveBeenCalled();
    });

    it('uses Google textEmbeddingModel for google provider', async () => {
      const client = new AIClient({ provider: 'google', apiKey: 'test' });
      await client.embed('test text');

      expect(createGoogleMock.textEmbeddingFn).toHaveBeenCalled();
      expect(embedMock).toHaveBeenCalled();
    });

    it('uses Mistral textEmbeddingModel for mistral provider', async () => {
      const client = new AIClient({ provider: 'mistral', apiKey: 'test' });
      await client.embed('test text');

      expect(createMistralMock.textEmbeddingFn).toHaveBeenCalled();
      expect(embedMock).toHaveBeenCalled();
    });

    it('uses Cohere textEmbeddingModel for cohere provider', async () => {
      const client = new AIClient({ provider: 'cohere', apiKey: 'test' });
      await client.embed('test text');

      expect(createCohereMock.textEmbeddingFn).toHaveBeenCalled();
      expect(embedMock).toHaveBeenCalled();
    });

    it('uses OpenAI embedding for OpenRouter', async () => {
      const client = new AIClient({ provider: 'openrouter', apiKey: 'test' });
      await client.embed('test text');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://openrouter.ai/api/v1'
        })
      );
      expect(createOpenAIMock.embeddingFn).toHaveBeenCalled();
    });

    it('uses OpenAI embedding for Ollama', async () => {
      const client = new AIClient({ provider: 'ollama' });
      await client.embed('test text');

      expect(createOpenAIMock.factory).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: 'ollama',
          baseURL: 'http://localhost:11434/v1'
        })
      );
      expect(createOpenAIMock.embeddingFn).toHaveBeenCalled();
    });

    it('supports embedMany for supported providers', async () => {
      const client = new AIClient({ provider: 'openai', apiKey: 'test' });
      const result = await client.embedMany(['text1', 'text2']);

      expect(embedManyMock).toHaveBeenCalledWith(
        expect.objectContaining({
          values: ['text1', 'text2']
        })
      );
      expect(result).toEqual([[0.1], [0.2]]);
    });
  });

  describe('per-request provider override', () => {
    it('allows overriding provider at request time', async () => {
      const client = new AIClient({ provider: 'openai', apiKey: 'test-key' });
      await client.generate('test prompt', { provider: 'anthropic' });

      expect(createAnthropicMock.factory).toHaveBeenCalled();
      expect(createOpenAIMock.factory).not.toHaveBeenCalled();
    });

    it('allows overriding model at request time', async () => {
      const client = new AIClient({ provider: 'openai', apiKey: 'test-key', model: 'gpt-4o' });
      await client.generate('test prompt', { model: 'gpt-4o-mini' });

      expect(createOpenAIMock.modelFn).toHaveBeenCalledWith('gpt-4o-mini');
    });
  });

  describe('configuration passthrough', () => {
    it('passes temperature to generateText', async () => {
      const client = new AIClient({ apiKey: 'test', temperature: 0.7 });
      await client.generate('test prompt');

      expect(generateTextMock).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.7 })
      );
    });

    it('passes maxTokens to generateText', async () => {
      const client = new AIClient({ apiKey: 'test', maxTokens: 1000 });
      await client.generate('test prompt');

      expect(generateTextMock).toHaveBeenCalledWith(
        expect.objectContaining({ maxOutputTokens: 1000 })
      );
    });

    it('allows per-request temperature override', async () => {
      const client = new AIClient({ apiKey: 'test', temperature: 0.7 });
      await client.generate('test prompt', { temperature: 0.9 });

      expect(generateTextMock).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.9 })
      );
    });
  });
});
