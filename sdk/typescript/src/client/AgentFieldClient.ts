import axios, { AxiosInstance } from 'axios';
import type {
  AgentConfig,
  DiscoveryOptions,
  DiscoveryFormat,
  DiscoveryResult,
  DiscoveryResponse,
  CompactDiscoveryResponse,
  HealthStatus
} from '../types/agent.js';
import {
  isExecutionLogBatchPayload,
  type ExecutionLogTransportPayload
} from '../observability/ExecutionLogger.js';
import { httpAgent, httpsAgent } from '../utils/httpAgents.js';
import { DIDAuthenticator } from './DIDAuthenticator.js';

export interface ExecutionStatusUpdate {
  status?: string;
  result?: Record<string, any>;
  error?: string;
  durationMs?: number;
  progress?: number;
  statusReason?: string;
}

// Raw discovery payload from API (snake_case)
interface RawDiscoveryPayload {
  discovered_at?: string;
  total_agents?: number;
  total_reasoners?: number;
  total_skills?: number;
  pagination?: {
    limit?: number;
    offset?: number;
    has_more?: boolean;
  };
  capabilities?: RawCapability[];
}

interface RawCapability {
  agent_id?: string;
  base_url?: string;
  version?: string;
  health_status?: string;
  deployment_type?: string;
  last_heartbeat?: string;
  reasoners?: RawReasoner[];
  skills?: RawSkill[];
}

interface RawReasoner {
  id?: string;
  description?: string;
  tags?: string[];
  input_schema?: any;
  output_schema?: any;
  examples?: any;
  invocation_target?: string;
}

interface RawSkill {
  id?: string;
  description?: string;
  tags?: string[];
  input_schema?: any;
  invocation_target?: string;
}

// Compact format
interface RawCompactDiscoveryPayload {
  discovered_at?: string;
  reasoners?: RawCompactCapability[];
  skills?: RawCompactCapability[];
}

interface RawCompactCapability {
  id?: string;
  agent_id?: string;
  target?: string;
  tags?: string[];
}

interface ExecutionError extends Error {
  status: number;
  responseData: unknown;
}
export class AgentFieldClient {
  private readonly http: AxiosInstance;
  private readonly config: AgentConfig;
  private readonly defaultHeaders: Record<string, string>;
  private didAuthenticator: DIDAuthenticator;

  constructor(config: AgentConfig) {
    const baseURL = (config.agentFieldUrl ?? 'http://localhost:8080').replace(/\/$/, '');
this.http = axios.create({
      baseURL,
      timeout: 30000,
      httpAgent,
      httpsAgent
    });
    this.config = config;

    const mergedHeaders = { ...(config.defaultHeaders ?? {}) };
    if (config.apiKey) {
      mergedHeaders['X-API-Key'] = config.apiKey;
    }
    this.defaultHeaders = this.sanitizeHeaders(mergedHeaders);
    this.didAuthenticator = new DIDAuthenticator(config.did, config.privateKeyJwk);
  }

  async register(payload: any): Promise<any> {
    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const res = await this.http.post('/api/v1/nodes/register', bodyStr, {
      headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders })
    });
    return res.data;
  }

  async getNode(nodeId: string): Promise<any> {
    const res = await this.http.get(`/api/v1/nodes/${encodeURIComponent(nodeId)}`, {
      headers: this.mergeHeaders({})
    });
    return res.data;
  }

  async heartbeat(status: 'starting' | 'ready' | 'degraded' | 'offline' = 'ready'): Promise<HealthStatus> {
    const nodeId = this.config.nodeId;
    const bodyStr = JSON.stringify({ status, version: this.config.version ?? '', timestamp: new Date().toISOString() });
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const res = await this.http.post(
      `/api/v1/nodes/${nodeId}/heartbeat`,
      bodyStr,
      { headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }) }
    );
    return res.data as HealthStatus;
  }

  async execute<T = any>(
    target: string,
    input: any,
    metadata?: {
      runId?: string;
      workflowId?: string;
      rootWorkflowId?: string;
      parentExecutionId?: string;
      reasonerId?: string;
      sessionId?: string;
      actorId?: string;
      callerDid?: string;
      targetDid?: string;
      agentNodeDid?: string;
      agentNodeId?: string;
    }
  ): Promise<T> {
    const headers: Record<string, string> = {};
    if (metadata?.runId) headers['X-Run-ID'] = metadata.runId;
    if (metadata?.workflowId) headers['X-Workflow-ID'] = metadata.workflowId;
    if (metadata?.rootWorkflowId) headers['X-Root-Workflow-ID'] = metadata.rootWorkflowId;
    if (metadata?.parentExecutionId) headers['X-Parent-Execution-ID'] = metadata.parentExecutionId;
    if (metadata?.reasonerId) headers['X-Reasoner-ID'] = metadata.reasonerId;
    if (metadata?.sessionId) headers['X-Session-ID'] = metadata.sessionId;
    if (metadata?.actorId) headers['X-Actor-ID'] = metadata.actorId;
    if (metadata?.callerDid) headers['X-Caller-DID'] = metadata.callerDid;
    if (metadata?.targetDid) headers['X-Target-DID'] = metadata.targetDid;
    if (metadata?.agentNodeDid) headers['X-Agent-Node-DID'] = metadata.agentNodeDid;
    if (metadata?.agentNodeId) headers['X-Agent-Node-ID'] = metadata.agentNodeId;

    const bodyStr = JSON.stringify({ input });
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    try {
      const res = await this.http.post(
        `/api/v1/execute/${target}`,
        bodyStr,
        { headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...headers, ...authHeaders }) }
      );
      return (res.data?.result as T) ?? res.data;
    } catch (err: any) {
      // Extract structured error from control plane response (e.g., 403 permission_denied).
      const respData = err?.response?.data;
      if (respData) {
        const status = err.response.status;
        const msg = respData.message || respData.error || JSON.stringify(respData);
        const enriched: ExecutionError = Object.assign(
          new Error(`execute ${target} failed (${status}): ${msg}`),
          {
            status,
            responseData: respData
          }
        );
        throw enriched;
      }
      throw err;
    }
  }

  async publishWorkflowEvent(event: {
    executionId: string;
    runId: string;
    workflowId?: string;
    rootWorkflowId?: string;
    reasonerId: string;
    agentNodeId: string;
    status: 'waiting' | 'running' | 'succeeded' | 'failed';
    parentExecutionId?: string;
    parentWorkflowId?: string;
    statusReason?: string;
    inputData?: Record<string, any>;
    result?: any;
    error?: string;
    durationMs?: number;
  }) {
    const payload = {
      execution_id: event.executionId,
      workflow_id: event.workflowId ?? event.runId,
      run_id: event.runId,
      root_workflow_id: event.rootWorkflowId ?? event.workflowId ?? event.runId,
      reasoner_id: event.reasonerId,
      type: event.reasonerId,
      agent_node_id: event.agentNodeId,
      status: event.status,
      status_reason: event.statusReason,
      parent_execution_id: event.parentExecutionId,
      parent_workflow_id: event.parentWorkflowId ?? event.workflowId ?? event.runId,
      input_data: event.inputData ?? {},
      result: event.result,
      error: event.error,
      duration_ms: event.durationMs
    };

    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const request = this.http
      .post('/api/v1/workflow/executions/events', bodyStr, {
        headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }),
        timeout: this.config.devMode ? 1000 : undefined
      })
      .catch(() => {
        // Best-effort; avoid throwing to keep agent execution resilient
      });

    // Fire and forget to avoid blocking local executions in tests/dev mode.
    void request;
  }

  publishExecutionLogs(payload: ExecutionLogTransportPayload): void {
    const executionId = isExecutionLogBatchPayload(payload)
      ? payload.entries[0]?.execution_id
      : payload.execution_id;

    if (!executionId) {
      return;
    }

    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const request = this.http
      .post(`/api/v1/executions/${encodeURIComponent(executionId)}/logs`, bodyStr, {
        headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }),
        timeout: this.config.devMode ? 1000 : 5000
      })
      .catch(() => {
        // Best-effort; execution logs must never break the agent runtime.
      });

    void request;
  }

  async updateExecutionStatus(executionId: string, update: ExecutionStatusUpdate) {
    if (!executionId) {
      throw new Error('executionId is required to update workflow status');
    }

    const payload = {
      status: update.status ?? 'running',
      result: update.result,
      error: update.error,
      duration_ms: update.durationMs,
      progress: update.progress !== undefined ? Math.round(update.progress) : undefined,
      status_reason: update.statusReason
    };

    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    await this.http.post(`/api/v1/executions/${executionId}/status`, bodyStr, {
      headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders })
    });
  }

  async discoverCapabilities(options: DiscoveryOptions = {}): Promise<DiscoveryResult> {
    const format = (options.format ?? 'json').toLowerCase() as DiscoveryFormat;
    const params: Record<string, string> = { format };
    const dedupe = (values?: string[]) =>
      Array.from(new Set((values ?? []).filter(Boolean))).map((v) => v!);

    const combinedAgents = dedupe([
      ...(options.agent ? [options.agent] : []),
      ...(options.nodeId ? [options.nodeId] : []),
      ...(options.agentIds ?? []),
      ...(options.nodeIds ?? [])
    ]);

    if (combinedAgents.length === 1) {
      params.agent = combinedAgents[0];
    } else if (combinedAgents.length > 1) {
      params.agent_ids = combinedAgents.join(',');
    }

    if (options.reasoner) params.reasoner = options.reasoner;
    if (options.skill) params.skill = options.skill;
    if (options.tags?.length) params.tags = dedupe(options.tags).join(',');

    if (options.includeInputSchema !== undefined) {
      params.include_input_schema = String(Boolean(options.includeInputSchema));
    }
    if (options.includeOutputSchema !== undefined) {
      params.include_output_schema = String(Boolean(options.includeOutputSchema));
    }
    if (options.includeDescriptions !== undefined) {
      params.include_descriptions = String(Boolean(options.includeDescriptions));
    }
    if (options.includeExamples !== undefined) {
      params.include_examples = String(Boolean(options.includeExamples));
    }
    if (options.healthStatus) params.health_status = options.healthStatus.toLowerCase();
    if (options.limit !== undefined) params.limit = String(options.limit);
    if (options.offset !== undefined) params.offset = String(options.offset);

    const res = await this.http.get('/api/v1/discovery/capabilities', {
      params,
      headers: this.mergeHeaders({
        ...(options.headers ?? {}),
        Accept: format === 'xml' ? 'application/xml' : 'application/json'
      }),
      responseType: format === 'xml' ? 'text' : 'json',
      transformResponse: (data) => data // preserve raw body for xml
    });

    const raw = typeof res.data === 'string' ? res.data : JSON.stringify(res.data);
    if (format === 'xml') {
      return { format: 'xml', raw, xml: raw };
    }

    const parsed: RawDiscoveryPayload | RawCompactDiscoveryPayload = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
    if (format === 'compact') {
      return {
        format: 'compact',
        raw,
        compact: this.mapCompactDiscovery(parsed as RawCompactDiscoveryPayload)
      };
    }

    return {
      format: 'json',
      raw,
      json: this.mapDiscoveryResponse(parsed as RawDiscoveryPayload)
    };
  }

  private mapDiscoveryResponse(payload: RawDiscoveryPayload): DiscoveryResponse {
    return {
      discoveredAt: String(payload.discovered_at ?? ''),
      totalAgents: Number(payload.total_agents ?? 0),
      totalReasoners: Number(payload.total_reasoners ?? 0),
      totalSkills: Number(payload.total_skills ?? 0),
      pagination: {
        limit: Number(payload.pagination?.limit ?? 0),
        offset: Number(payload.pagination?.offset ?? 0),
        hasMore: Boolean(payload.pagination?.has_more)
      },
      capabilities: (payload.capabilities ?? []).map((cap) => ({
        agentId: cap.agent_id ?? '',
        baseUrl: cap.base_url ?? '',
        version: cap.version ?? '',
        healthStatus: cap.health_status ?? '',
        deploymentType: cap.deployment_type,
        lastHeartbeat: cap.last_heartbeat,
        reasoners: (cap.reasoners ?? []).map((r) => ({
          id: r.id ?? '',
          description: r.description,
          tags: r.tags ?? [],
          inputSchema: r.input_schema,
          outputSchema: r.output_schema,
          examples: r.examples,
          invocationTarget: r.invocation_target ?? ''
        })),
        skills: (cap.skills ?? []).map((s) => ({
          id: s.id ?? '',
          description: s.description,
          tags: s.tags ?? [],
          inputSchema: s.input_schema,
          invocationTarget: s.invocation_target ?? ''
        }))
      }))
    };
  }

  private mapCompactDiscovery(payload: RawCompactDiscoveryPayload): CompactDiscoveryResponse {
    const toCap = (cap: RawCompactCapability) => ({
      id: cap.id ?? '',
      agentId: cap.agent_id ?? '',
      target: cap.target ?? '',
      tags: cap.tags ?? []
    });

    return {
      discoveredAt: String(payload.discovered_at ?? ''),
      reasoners: (payload.reasoners ?? []).map(toCap),
      skills: (payload.skills ?? []).map(toCap)
    };
  }

  private sanitizeHeaders(headers: Record<string, any>): Record<string, string> {
    const sanitized: Record<string, string> = {};
    Object.entries(headers).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      sanitized[key] = typeof value === 'string' ? value : String(value);
    });
    return sanitized;
  }

  private mergeHeaders(headers?: Record<string, any>): Record<string, string> {
    return {
      ...this.defaultHeaders,
      ...this.sanitizeHeaders(headers ?? {})
    };
  }

  private buildExecutionHeaders(metadata: {
    runId?: string;
    executionId?: string;
    sessionId?: string;
    actorId?: string;
    workflowId?: string;
    rootWorkflowId?: string;
    parentExecutionId?: string;
    reasonerId?: string;
    callerDid?: string;
    targetDid?: string;
    agentNodeDid?: string;
    agentNodeId?: string;
  }): Record<string, string> {
    const headers: Record<string, string> = {};
    if (metadata.runId) headers['x-run-id'] = metadata.runId;
    if (metadata.executionId) headers['x-execution-id'] = metadata.executionId;
    if (metadata.sessionId) headers['x-session-id'] = metadata.sessionId;
    if (metadata.actorId) headers['x-actor-id'] = metadata.actorId;
    if (metadata.workflowId) headers['x-workflow-id'] = metadata.workflowId;
    if (metadata.rootWorkflowId) headers['x-root-workflow-id'] = metadata.rootWorkflowId;
    if (metadata.parentExecutionId) headers['x-parent-execution-id'] = metadata.parentExecutionId;
    if (metadata.reasonerId) headers['x-reasoner-id'] = metadata.reasonerId;
    if (metadata.callerDid) headers['x-caller-did'] = metadata.callerDid;
    if (metadata.targetDid) headers['x-target-did'] = metadata.targetDid;
    if (metadata.agentNodeDid) headers['x-agent-node-did'] = metadata.agentNodeDid;
    if (metadata.agentNodeId) headers['x-agent-node-id'] = metadata.agentNodeId;
    return headers;
  }

  setDIDCredentials(did: string, privateKeyJwk: string): void {
    this.didAuthenticator.setCredentials(did, privateKeyJwk);
  }

  get didAuthConfigured(): boolean {
    return this.didAuthenticator.isConfigured;
  }

  getDID(): string | undefined {
    return this.didAuthenticator.did;
  }

  sendNote(message: string, tags: string[], agentNodeId: string, metadata: {
    runId?: string;
    executionId?: string;
    sessionId?: string;
    actorId?: string;
    workflowId?: string;
    rootWorkflowId?: string;
    parentExecutionId?: string;
    reasonerId?: string;
    callerDid?: string;
    targetDid?: string;
    agentNodeDid?: string;
  }, uiApiBaseUrl: string, devMode?: boolean): void {
    const payload = {
      message,
      tags: tags ?? [],
      timestamp: Date.now() / 1000,
      agent_node_id: agentNodeId
    };

    const executionHeaders = this.buildExecutionHeaders({ ...metadata, agentNodeId });
    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const headers = this.mergeHeaders({
      'Content-Type': 'application/json',
      ...executionHeaders,
      ...authHeaders
    });

    const request = axios
      .post(`${uiApiBaseUrl}/executions/note`, bodyStr, {
        headers,
        timeout: devMode ? 5000 : 10000,
        httpAgent,
        httpsAgent
      })
      .catch(() => {});
    void request;
  }
}
