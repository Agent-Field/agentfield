import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AIClient } from '../src/ai/AIClient.js';
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import type { MemoryClient } from '../src/memory/MemoryClient.js';

const { embedMock, embedManyMock, generateTextMock, streamTextMock } = vi.hoisted(() => ({
  embedMock: vi.fn(),
  embedManyMock: vi.fn(),
  generateTextMock: vi.fn(),
  streamTextMock: vi.fn()
}));

const { openAIEmbeddingMock, openAIProvider, createOpenAIMock } = vi.hoisted(() => {
  const embedding = vi.fn((modelId: string) => ({ id: `embed-${modelId}` }));
  const provider = vi.fn((modelId: string) => ({ id: modelId })) as any;
  (provider as any).embedding = embedding;
  return {
    openAIEmbeddingMock: embedding,
    openAIProvider: provider,
    createOpenAIMock: vi.fn(() => provider)
  };
});

vi.mock('ai', () => ({
  generateText: generateTextMock,
  streamText: streamTextMock,
  embed: embedMock,
  embedMany: embedManyMock
}));

vi.mock('@ai-sdk/openai', () => ({
  createOpenAI: createOpenAIMock
}));

vi.mock('@ai-sdk/anthropic', () => ({
  createAnthropic: vi.fn(() => vi.fn())
}));

describe('AIClient embeddings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    embedMock.mockResolvedValue({ embedding: [0.1, 0.2] });
    embedManyMock.mockResolvedValue({ embeddings: [[0.1], [0.2]] });
  });

  it('embeds text using default embedding model', async () => {
    const client = new AIClient({ apiKey: 'test' });
    const result = await client.embed('hello');

    expect(openAIEmbeddingMock).toHaveBeenCalledWith('text-embedding-3-small');
    expect(embedMock).toHaveBeenCalledWith(
      expect.objectContaining({
        value: 'hello',
        model: expect.any(Object)
      })
    );
    expect(result).toEqual([0.1, 0.2]);
  });

  it('embeds many texts with an override model', async () => {
    const client = new AIClient({ apiKey: 'test', embeddingModel: 'text-embedding-ada-002' });
    const result = await client.embedMany(['a', 'b'], { model: 'text-embedding-3-large' });

    expect(openAIEmbeddingMock).toHaveBeenCalledWith('text-embedding-3-large');
    expect(embedManyMock).toHaveBeenCalledWith(
      expect.objectContaining({
        values: ['a', 'b'],
        model: expect.any(Object)
      })
    );
    expect(result).toEqual([[0.1], [0.2]]);
  });

  it('throws when provider does not support embeddings', async () => {
    const client = new AIClient({ provider: 'anthropic' });
    await expect(client.embed('hi')).rejects.toThrow(/Embedding generation is not supported/);
  });
});

describe('MemoryInterface embedding helpers', () => {
  it('embeds text then stores vector with scope defaults', async () => {
    const setVector = vi.fn();
    const memoryClient = { setVector } as unknown as MemoryClient;
    const aiClient = { embed: vi.fn().mockResolvedValue([0.9]), embedMany: vi.fn() } as any;

    const memory = new MemoryInterface({
      client: memoryClient,
      aiClient,
      defaultScope: 'workflow',
      defaultScopeId: 'wf-1',
      metadata: { workflowId: 'wf-1' }
    });

    const embedding = await memory.embedAndSet('key-1', 'text to embed', { tag: 'demo' });

    expect(aiClient.embed).toHaveBeenCalledWith('text to embed', undefined);
    expect(setVector).toHaveBeenCalledWith(
      'key-1',
      [0.9],
      { tag: 'demo' },
      expect.objectContaining({
        scope: 'workflow',
        scopeId: 'wf-1',
        metadata: { workflowId: 'wf-1' }
      })
    );
    expect(embedding).toEqual([0.9]);
  });

  it('deletes multiple vectors with a helper', async () => {
    const deleteVector = vi.fn();
    const memory = new MemoryInterface({
      client: { deleteVector } as unknown as MemoryClient,
      defaultScope: 'session',
      defaultScopeId: 's-1'
    });

    await memory.deleteVectors(['a', 'b'], 'session', 's-1');

    expect(deleteVector).toHaveBeenNthCalledWith(
      1,
      'a',
      expect.objectContaining({
        scope: 'session',
        scopeId: 's-1'
      })
    );
    expect(deleteVector).toHaveBeenNthCalledWith(
      2,
      'b',
      expect.objectContaining({
        scope: 'session',
        scopeId: 's-1'
      })
    );
  });
});
