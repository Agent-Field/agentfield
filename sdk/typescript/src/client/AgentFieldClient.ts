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

export interface ExecutionStatusUpdate {
  status?: string;
  result?: Record<string, any>;
  error?: string;
  durationMs?: number;
  progress?: number;
}

export class AgentFieldClient {
  private readonly http: AxiosInstance;
  private readonly config: AgentConfig;
  private readonly defaultHeaders: Record<string, string>;

  constructor(config: AgentConfig) {
    const baseURL = (config.agentFieldUrl ?? 'http://localhost:8080').replace(/\/$/, '');
    this.http = axios.create({ baseURL });
    this.config = config;

    const mergedHeaders = { ...(config.defaultHeaders ?? {}) };
    if (config.apiKey) {
      mergedHeaders['X-API-Key'] = config.apiKey;
    }
    this.defaultHeaders = this.sanitizeHeaders(mergedHeaders);
  }

  async register(payload: any) {
    await this.http.post('/api/v1/nodes/register', payload, { headers: this.mergeHeaders() });
  }

  async heartbeat(status: 'starting' | 'ready' | 'degraded' | 'offline' = 'ready'): Promise<HealthStatus> {
    const nodeId = this.config.nodeId;
    const res = await this.http.post(
      `/api/v1/nodes/${nodeId}/heartbeat`,
      {
        status,
        timestamp: new Date().toISOString()
      },
      { headers: this.mergeHeaders() }
    );
    return res.data as HealthStatus;
  }

  async execute<T = any>(
    target: string,
    input: any,
    metadata?: {
      runId?: string;
      workflowId?: string;
      parentExecutionId?: string;
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
    if (metadata?.parentExecutionId) headers['X-Parent-Execution-ID'] = metadata.parentExecutionId;
    if (metadata?.sessionId) headers['X-Session-ID'] = metadata.sessionId;
    if (metadata?.actorId) headers['X-Actor-ID'] = metadata.actorId;
    if (metadata?.callerDid) headers['X-Caller-DID'] = metadata.callerDid;
    if (metadata?.targetDid) headers['X-Target-DID'] = metadata.targetDid;
    if (metadata?.agentNodeDid) headers['X-Agent-Node-DID'] = metadata.agentNodeDid;
    if (metadata?.agentNodeId) headers['X-Agent-Node-ID'] = metadata.agentNodeId;

    const res = await this.http.post(
      `/api/v1/execute/${target}`,
      {
        input
      },
      { headers: this.mergeHeaders(headers) }
    );
    return (res.data?.result as T) ?? res.data;
  }

  async publishWorkflowEvent(event: {
    executionId: string;
    runId: string;
    workflowId?: string;
    reasonerId: string;
    agentNodeId: string;
    status: 'running' | 'succeeded' | 'failed';
    parentExecutionId?: string;
    parentWorkflowId?: string;
    inputData?: Record<string, any>;
    result?: any;
    error?: string;
    durationMs?: number;
  }) {
    const payload = {
      execution_id: event.executionId,
      workflow_id: event.workflowId ?? event.runId,
      run_id: event.runId,
      reasoner_id: event.reasonerId,
      type: event.reasonerId,
      agent_node_id: event.agentNodeId,
      status: event.status,
      parent_execution_id: event.parentExecutionId,
      parent_workflow_id: event.parentWorkflowId ?? event.workflowId ?? event.runId,
      input_data: event.inputData ?? {},
      result: event.result,
      error: event.error,
      duration_ms: event.durationMs
    };

    const request = this.http
      .post('/api/v1/workflow/executions/events', payload, {
        headers: this.mergeHeaders(),
        timeout: this.config.devMode ? 1000 : undefined
      })
      .catch(() => {
        // Best-effort; avoid throwing to keep agent execution resilient
      });

    // Fire and forget to avoid blocking local executions in tests/dev mode.
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
      progress: update.progress !== undefined ? Math.round(update.progress) : undefined
    };

    await this.http.post(`/api/v1/executions/${executionId}/status`, payload, { headers: this.mergeHeaders() });
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

    const parsed = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
    if (format === 'compact') {
      return {
        format: 'compact',
        raw,
        compact: this.mapCompactDiscovery(parsed as any)
      };
    }

    return {
      format: 'json',
      raw,
      json: this.mapDiscoveryResponse(parsed as any)
    };
  }

  private mapDiscoveryResponse(payload: any): DiscoveryResponse {
    return {
      discoveredAt: String(payload?.discovered_at ?? ''),
      totalAgents: Number(payload?.total_agents ?? 0),
      totalReasoners: Number(payload?.total_reasoners ?? 0),
      totalSkills: Number(payload?.total_skills ?? 0),
      pagination: {
        limit: Number(payload?.pagination?.limit ?? 0),
        offset: Number(payload?.pagination?.offset ?? 0),
        hasMore: Boolean(payload?.pagination?.has_more)
      },
      capabilities: (payload?.capabilities ?? []).map((cap: any) => ({
        agentId: cap?.agent_id ?? '',
        baseUrl: cap?.base_url ?? '',
        version: cap?.version ?? '',
        healthStatus: cap?.health_status ?? '',
        deploymentType: cap?.deployment_type,
        lastHeartbeat: cap?.last_heartbeat,
        reasoners: (cap?.reasoners ?? []).map((r: any) => ({
          id: r?.id ?? '',
          description: r?.description,
          tags: r?.tags ?? [],
          inputSchema: r?.input_schema,
          outputSchema: r?.output_schema,
          examples: r?.examples,
          invocationTarget: r?.invocation_target ?? ''
        })),
        skills: (cap?.skills ?? []).map((s: any) => ({
          id: s?.id ?? '',
          description: s?.description,
          tags: s?.tags ?? [],
          inputSchema: s?.input_schema,
          invocationTarget: s?.invocation_target ?? ''
        }))
      }))
    };
  }

  private mapCompactDiscovery(payload: any): CompactDiscoveryResponse {
    const toCap = (cap: any) => ({
      id: cap?.id ?? '',
      agentId: cap?.agent_id ?? '',
      target: cap?.target ?? '',
      tags: cap?.tags ?? []
    });

    return {
      discoveredAt: String(payload?.discovered_at ?? ''),
      reasoners: (payload?.reasoners ?? []).map(toCap),
      skills: (payload?.skills ?? []).map(toCap)
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
}
