import type { MemoryScope } from '../types/agent.js';
import type { MemoryClient, MemoryRequestMetadata, VectorSearchOptions } from './MemoryClient.js';
import type { MemoryEventClient } from './MemoryEventClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export interface MemoryChangeEvent {
  key: string;
  data: any;
  scope: MemoryScope;
  scopeId: string;
  timestamp: string | Date;
  agentId: string;
}

export type MemoryWatchHandler = (event: MemoryChangeEvent) => Promise<void> | void;

export class MemoryInterface {
  private readonly client: MemoryClient;
  private readonly eventClient?: MemoryEventClient;
  private readonly aiClient?: AIClient;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId?: string;
  private readonly metadata?: MemoryRequestMetadata;

  constructor(params: {
    client: MemoryClient;
    eventClient?: MemoryEventClient;
    aiClient?: AIClient;
    defaultScope?: MemoryScope;
    defaultScopeId?: string;
    metadata?: MemoryRequestMetadata;
  }) {
    this.client = params.client;
    this.eventClient = params.eventClient;
    this.aiClient = params.aiClient;
    this.defaultScope = params.defaultScope ?? 'workflow';
    this.defaultScopeId = params.defaultScopeId;
    this.metadata = params.metadata;
  }

  async set(key: string, data: any, scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    await this.client.set(key, data, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  get<T = any>(key: string, scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    // If caller uses defaults, perform hierarchical fallback lookup; otherwise fetch the explicit scope only.
    if (scope === this.defaultScope && scopeId === this.defaultScopeId) {
      return this.getWithFallback<T>(key);
    }

    return this.client.get<T>(key, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  async getWithFallback<T = any>(key: string) {
    for (const candidate of this.getScopeOrder()) {
      const value = await this.client.get<T>(key, {
        scope: candidate.scope,
        scopeId: candidate.scopeId,
        metadata: this.metadata
      });
      if (value !== undefined) return value;
    }
    return undefined;
  }

  async setVector(
    key: string,
    embedding: number[],
    metadata?: any,
    scope: MemoryScope = this.defaultScope,
    scopeId = this.defaultScopeId
  ) {
    await this.client.setVector(key, embedding, metadata, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  async deleteVector(key: string, scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    await this.client.deleteVector(key, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  searchVector(queryEmbedding: number[], options: Omit<VectorSearchOptions, 'metadata'> = {}) {
    return this.client.searchVector(queryEmbedding, {
      ...options,
      metadata: this.metadata
    });
  }

  delete(key: string, scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    return this.client.delete(key, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  exists(key: string, scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    return this.client.exists(key, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  listKeys(scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    return this.client.listKeys(scope, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  async embedText(text: string, options?: AIEmbeddingOptions) {
    if (!this.aiClient) {
      throw new Error('AI client not configured for embeddings');
    }
    return this.aiClient.embed(text, options);
  }

  async embedTexts(texts: string[], options?: AIEmbeddingOptions) {
    if (!this.aiClient) {
      throw new Error('AI client not configured for embeddings');
    }
    return this.aiClient.embedMany(texts, options);
  }

  async embedAndSet(
    key: string,
    text: string,
    metadata?: any,
    scope: MemoryScope = this.defaultScope,
    scopeId = this.defaultScopeId,
    embeddingOptions?: AIEmbeddingOptions
  ) {
    const embedding = await this.embedText(text, embeddingOptions);
    await this.setVector(key, embedding, metadata, scope, scopeId);
    return embedding;
  }

  async deleteVectors(keys: string[], scope: MemoryScope = this.defaultScope, scopeId = this.defaultScopeId) {
    for (const key of keys) {
      await this.deleteVector(key, scope, scopeId);
    }
  }

  workflow(scopeId: string) {
    return this.cloneWithScope('workflow', scopeId);
  }

  session(scopeId: string) {
    return this.cloneWithScope('session', scopeId);
  }

  actor(scopeId: string) {
    return this.cloneWithScope('actor', scopeId);
  }

  get globalScope() {
    return this.cloneWithScope('global', 'global');
  }

  onEvent(handler: MemoryWatchHandler) {
    this.eventClient?.onEvent(handler);
  }

  private cloneWithScope(scope: MemoryScope, scopeId?: string) {
    return new MemoryInterface({
      client: this.client,
      eventClient: this.eventClient,
      aiClient: this.aiClient,
      defaultScope: scope,
      defaultScopeId: scopeId ?? this.resolveScopeId(scope, this.metadata),
      metadata: this.metadata
    });
  }

  private getScopeOrder(): Array<{ scope: MemoryScope; scopeId?: string }> {
    const metadata = this.metadata ?? {};
    const order: Array<{ scope: MemoryScope; scopeId?: string }> = [];

    const pushUnique = (scope: MemoryScope, scopeId?: string) => {
      const key = `${scope}:${scopeId ?? ''}`;
      if (!order.some((c) => `${c.scope}:${c.scopeId ?? ''}` === key)) {
        order.push({ scope, scopeId });
      }
    };

    pushUnique(this.defaultScope, this.defaultScopeId ?? this.resolveScopeId(this.defaultScope, metadata));

    const defaultSequence: MemoryScope[] = ['workflow', 'session', 'actor', 'global'];
    defaultSequence.forEach((scope) => {
      pushUnique(scope, this.resolveScopeId(scope, metadata));
    });

    return order;
  }

  private resolveScopeId(scope: MemoryScope, metadata?: MemoryRequestMetadata) {
    switch (scope) {
      case 'workflow':
        return metadata?.workflowId ?? metadata?.runId;
      case 'session':
        return metadata?.sessionId;
      case 'actor':
        return metadata?.actorId;
      case 'global':
        return 'global';
      default:
        return undefined;
    }
  }
}
