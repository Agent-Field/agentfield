import axios, { AxiosInstance, isAxiosError } from 'axios';
import type { MemoryScope } from '../types/agent.js';
import { httpAgent, httpsAgent } from '../utils/httpAgents.js';

export interface MemoryRequestMetadata {
  workflowId?: string;
  sessionId?: string;
  actorId?: string;
  runId?: string;
  executionId?: string;
  parentExecutionId?: string;
  callerDid?: string;
  targetDid?: string;
  agentNodeDid?: string;
  agentNodeId?: string;
}

export interface MemoryRequestOptions {
  scope?: MemoryScope;
  scopeId?: string;
  metadata?: MemoryRequestMetadata;
  headers?: Record<string, string | number | boolean | undefined>;
}

export interface VectorSearchOptions extends MemoryRequestOptions {
  topK?: number;
  filters?: Record<string, any>;
}

export interface VectorSearchResult {
  key: string;
  scope: string;
  scopeId: string;
  score: number;
  metadata?: Record<string, any>;
}

export class MemoryClient {
  private readonly http: AxiosInstance;
  private readonly defaultHeaders: Record<string, string>;

  constructor(baseUrl: string, defaultHeaders?: Record<string, string | number | boolean | undefined>) {
    this.http = axios.create({
      baseURL: baseUrl.replace(/\/$/, ''),
      timeout: 30000,
      httpAgent,
      httpsAgent
    });
    this.defaultHeaders = this.sanitizeHeaders(defaultHeaders ?? {});
  }

  async set(key: string, data: any, options: MemoryRequestOptions = {}) {
    const payload: any = { key, data };
    if (options.scope) payload.scope = options.scope;

    await this.http.post('/api/v1/memory/set', payload, {
      headers: this.buildHeaders(options)
    });
  }

  async get<T = any>(key: string, options: MemoryRequestOptions = {}): Promise<T | undefined> {
    try {
      const payload: any = { key };
      if (options.scope) payload.scope = options.scope;

      const res = await this.http.post('/api/v1/memory/get', payload, {
        headers: this.buildHeaders(options)
      });
      return res.data?.data as T;
    } catch (err) {
      if (isAxiosError(err) && err.response?.status === 404) {
        return undefined;
      }
      throw err;
    }
  }

  async delete(key: string, options: MemoryRequestOptions = {}) {
    const payload: any = { key };
    if (options.scope) payload.scope = options.scope;

    await this.http.post('/api/v1/memory/delete', payload, {
      headers: this.buildHeaders(options)
    });
  }

  async listKeys(scope: MemoryScope, options: MemoryRequestOptions = {}) {
    const res = await this.http.get('/api/v1/memory/list', {
      params: { scope },
      headers: this.buildHeaders({ ...options, scope })
    });
    return (res.data ?? []).map((item: any) => item?.key).filter(Boolean) as string[];
  }

  async exists(key: string, options: MemoryRequestOptions = {}) {
    const value = await this.get(key, options);
    return value !== undefined;
  }

  async setVector(key: string, embedding: number[], metadata?: any, options: MemoryRequestOptions = {}) {
    const payload: any = {
      key,
      embedding
    };
    if (metadata !== undefined) payload.metadata = metadata;
    if (options.scope) payload.scope = options.scope;

    await this.http.post('/api/v1/memory/vector/set', payload, {
      headers: this.buildHeaders(options)
    });
  }

  async deleteVector(key: string, options: MemoryRequestOptions = {}) {
    const payload: any = { key };
    if (options.scope) payload.scope = options.scope;

    await this.http.post('/api/v1/memory/vector/delete', payload, {
      headers: this.buildHeaders(options)
    });
  }

  async searchVector(queryEmbedding: number[], options: VectorSearchOptions = {}): Promise<VectorSearchResult[]> {
    const payload: any = {
      query_embedding: queryEmbedding,
      top_k: options.topK ?? 10
    };
    if (options.filters) payload.filters = options.filters;
    if (options.scope) payload.scope = options.scope;

    const res = await this.http.post('/api/v1/memory/vector/search', payload, {
      headers: this.buildHeaders(options)
    });
    return res.data ?? [];
  }

  private buildHeaders(options: MemoryRequestOptions = {}) {
    const { scope, scopeId, metadata } = options;
    const headers: Record<string, string> = { ...this.defaultHeaders };

    const workflowId = metadata?.workflowId ?? metadata?.runId;
    if (workflowId) headers['X-Workflow-ID'] = workflowId;
    if (metadata?.sessionId) headers['X-Session-ID'] = metadata.sessionId;
    if (metadata?.actorId) headers['X-Actor-ID'] = metadata.actorId;
    if (metadata?.runId) headers['X-Run-ID'] = metadata.runId;
    if (metadata?.executionId) headers['X-Execution-ID'] = metadata.executionId;
    if (metadata?.parentExecutionId) headers['X-Parent-Execution-ID'] = metadata.parentExecutionId;
    if (metadata?.callerDid) headers['X-Caller-DID'] = metadata.callerDid;
    if (metadata?.targetDid) headers['X-Target-DID'] = metadata.targetDid;
    if (metadata?.agentNodeDid) headers['X-Agent-Node-DID'] = metadata.agentNodeDid;
    if (metadata?.agentNodeId) headers['X-Agent-Node-ID'] = metadata.agentNodeId;

    const headerName = this.scopeToHeader(scope);
    const resolvedScopeId = this.resolveScopeId(scope, scopeId, metadata);
    if (headerName && resolvedScopeId) {
      headers[headerName] = resolvedScopeId;
    }

    return { ...headers, ...this.sanitizeHeaders(options.headers ?? {}) };
  }

  private scopeToHeader(scope?: MemoryScope) {
    switch (scope) {
      case 'workflow':
        return 'X-Workflow-ID';
      case 'session':
        return 'X-Session-ID';
      case 'actor':
        return 'X-Actor-ID';
      default:
        return undefined;
    }
  }

  private resolveScopeId(scope?: MemoryScope, scopeId?: string, metadata?: MemoryRequestMetadata) {
    if (scopeId) return scopeId;
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

  private sanitizeHeaders(headers: Record<string, any>): Record<string, string> {
    const sanitized: Record<string, string> = {};
    Object.entries(headers).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      sanitized[key] = typeof value === 'string' ? value : String(value);
    });
    return sanitized;
  }
}
