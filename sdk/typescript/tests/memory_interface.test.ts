import { describe, expect, it, vi } from 'vitest';
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import type {
  MemoryClient,
  MemoryRequestMetadata,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import type { MemoryEventClient } from '../src/memory/MemoryEventClient.js';
import type { AIClient, AIEmbeddingOptions } from '../src/ai/AIClient.js';
import type { MemoryScope } from '../src/types/agent.js';

type MockMemoryClient = Pick<
  MemoryClient,
  'set' | 'get' | 'setVector' | 'deleteVector' | 'searchVector' | 'delete' | 'exists' | 'listKeys'
>;

type MockAIClient = Pick<AIClient, 'embed' | 'embedMany'>;
type MockMemoryEventClient = Pick<MemoryEventClient, 'onEvent'>;

const createMemoryClient = (): MockMemoryClient => ({
  set: vi.fn(async () => {}),
  get: vi.fn(async () => undefined),
  setVector: vi.fn(async () => {}),
  deleteVector: vi.fn(async () => {}),
  searchVector: vi.fn(async () => [] as VectorSearchResult[]),
  delete: vi.fn(async () => {}),
  exists: vi.fn(async () => false),
  listKeys: vi.fn(async () => [] as string[])
});

describe('MemoryInterface exported methods', () => {
  it('routes CRUD operations through the configured client and fallback scope order', async () => {
    const client = createMemoryClient();
    const metadata: MemoryRequestMetadata = {
      workflowId: 'wf-1',
      sessionId: 'sess-1',
      actorId: 'actor-1'
    };
    client.get = vi
      .fn<MockMemoryClient['get']>()
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce('session-value');

    const memory = new MemoryInterface({
      client: client as MemoryClient,
      defaultScope: 'workflow',
      metadata
    });

    await memory.set('greeting', { ok: true });
    await expect(memory.get<string>('greeting')).resolves.toBe('session-value');
    await memory.get('greeting', 'actor', 'explicit-actor');
    await memory.delete('greeting');
    await expect(memory.exists('greeting')).resolves.toBe(false);
    await expect(memory.listKeys()).resolves.toEqual([]);

    expect(client.set).toHaveBeenCalledWith('greeting', { ok: true }, {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
    expect(client.get).toHaveBeenNthCalledWith(1, 'greeting', {
      scope: 'workflow',
      scopeId: 'wf-1',
      metadata
    });
    expect(client.get).toHaveBeenNthCalledWith(2, 'greeting', {
      scope: 'session',
      scopeId: 'sess-1',
      metadata
    });
    expect(client.get).toHaveBeenNthCalledWith(3, 'greeting', {
      scope: 'actor',
      scopeId: 'explicit-actor',
      metadata
    });
    expect(client.delete).toHaveBeenCalledWith('greeting', {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
    expect(client.exists).toHaveBeenCalledWith('greeting', {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
    expect(client.listKeys).toHaveBeenCalledWith('workflow', {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
  });

  it('supports vector operations, embeddings, scoped clones, and event subscriptions', async () => {
    const client = createMemoryClient();
    const eventClient: MockMemoryEventClient = {
      onEvent: vi.fn()
    };
    const aiClient: MockAIClient = {
      embed: vi.fn(async (_text: string, _options?: AIEmbeddingOptions) => [0.1, 0.2]),
      embedMany: vi.fn(async (_texts: string[], _options?: AIEmbeddingOptions) => [[0.1], [0.2]])
    };
    const metadata: MemoryRequestMetadata = {
      workflowId: 'wf-1',
      sessionId: 'sess-1',
      actorId: 'actor-1',
      runId: 'run-1'
    };
    const searchResults: VectorSearchResult[] = [
      { key: 'doc-1', scope: 'workflow', scopeId: 'wf-1', score: 0.9 }
    ];
    client.searchVector = vi.fn(async () => searchResults);

    const memory = new MemoryInterface({
      client: client as MemoryClient,
      eventClient: eventClient as MemoryEventClient,
      aiClient: aiClient as AIClient,
      metadata
    });

    await memory.setVector('doc-1', [1, 2], { tag: 'note' });
    await memory.deleteVector('doc-1');
    await expect(
      memory.searchVector([3, 4], { topK: 5, filters: { kind: 'note' }, scope: 'session' })
    ).resolves.toEqual(searchResults);
    await expect(memory.embedText('hello')).resolves.toEqual([0.1, 0.2]);
    await expect(memory.embedTexts(['a', 'b'])).resolves.toEqual([[0.1], [0.2]]);
    await expect(memory.embedAndSet('doc-2', 'text', { category: 'memo' }, 'actor', 'actor-1')).resolves.toEqual([
      0.1,
      0.2
    ]);
    await memory.deleteVectors(['doc-1', 'doc-2'], 'session', 'sess-1');

    const workflowMemory = memory.workflow('wf-override');
    const sessionMemory = memory.session('sess-override');
    const actorMemory = memory.actor('actor-override');
    const globalMemory = memory.globalScope;
    const handler = vi.fn();
    memory.onEvent(handler);

    await workflowMemory.set('wf-key', 1);
    await sessionMemory.set('session-key', 2);
    await actorMemory.set('actor-key', 3);
    await globalMemory.set('global-key', 4);

    expect(client.setVector).toHaveBeenCalledWith('doc-1', [1, 2], { tag: 'note' }, {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
    expect(client.deleteVector).toHaveBeenCalledWith('doc-1', {
      scope: 'workflow',
      scopeId: undefined,
      metadata
    });
    expect(client.searchVector).toHaveBeenCalledWith([3, 4], {
      topK: 5,
      filters: { kind: 'note' },
      scope: 'session',
      metadata
    } satisfies VectorSearchOptions);
    expect(aiClient.embed).toHaveBeenCalledWith('hello', undefined);
    expect(aiClient.embedMany).toHaveBeenCalledWith(['a', 'b'], undefined);
    expect(client.setVector).toHaveBeenCalledWith('doc-2', [0.1, 0.2], { category: 'memo' }, {
      scope: 'actor',
      scopeId: 'actor-1',
      metadata
    });
    expect(client.deleteVector).toHaveBeenNthCalledWith(2, 'doc-1', {
      scope: 'session',
      scopeId: 'sess-1',
      metadata
    });
    expect(client.deleteVector).toHaveBeenNthCalledWith(3, 'doc-2', {
      scope: 'session',
      scopeId: 'sess-1',
      metadata
    });
    expect(eventClient.onEvent).toHaveBeenCalledWith(handler);
    expect(client.set).toHaveBeenNthCalledWith(1, 'wf-key', 1, {
      scope: 'workflow',
      scopeId: 'wf-override',
      metadata
    });
    expect(client.set).toHaveBeenNthCalledWith(2, 'session-key', 2, {
      scope: 'session',
      scopeId: 'sess-override',
      metadata
    });
    expect(client.set).toHaveBeenNthCalledWith(3, 'actor-key', 3, {
      scope: 'actor',
      scopeId: 'actor-override',
      metadata
    });
    expect(client.set).toHaveBeenNthCalledWith(4, 'global-key', 4, {
      scope: 'global',
      scopeId: 'global',
      metadata
    });
  });

  it('rejects embedding methods when no AI client is configured and resolves implicit scope ids', async () => {
    const client = createMemoryClient();
    const metadata: MemoryRequestMetadata = {
      runId: 'run-1',
      sessionId: 'sess-1',
      actorId: 'actor-1'
    };
    client.get = vi
      .fn<MockMemoryClient['get']>()
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce('actor-value');

    const memory = new MemoryInterface({
      client: client as MemoryClient,
      defaultScope: 'workflow',
      metadata
    });

    await expect(memory.embedText('hello')).rejects.toThrow('AI client not configured for embeddings');
    await expect(memory.embedTexts(['hello'])).rejects.toThrow('AI client not configured for embeddings');
    await expect(memory.getWithFallback<string>('shared')).resolves.toBe('actor-value');

    expect(client.get).toHaveBeenNthCalledWith(1, 'shared', {
      scope: 'workflow',
      scopeId: 'run-1',
      metadata
    });
    expect(client.get).toHaveBeenNthCalledWith(2, 'shared', {
      scope: 'session',
      scopeId: 'sess-1',
      metadata
    });
    expect(client.get).toHaveBeenNthCalledWith(3, 'shared', {
      scope: 'actor',
      scopeId: 'actor-1',
      metadata
    });
  });
});
