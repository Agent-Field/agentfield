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
import { normalizeStatus as normalizeExecutionStatus, isTerminal as isTerminalExecutionStatus } from '../status/ExecutionStatus.js';

export interface ExecutionStatusUpdate {
  status?: string;
  result?: Record<string, any>;
  error?: string;
  durationMs?: number;
  progress?: number;
  statusReason?: string;
}

/** Metadata forwarded as `X-*` headers on execute / executeAsync dispatch. */
export interface ExecuteMetadata {
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
  replaySourceRunId?: string;
  replayBeforeExecutionId?: string;
  replayMode?: string;
}

/** Terminal (or in-flight) status snapshot from `GET /executions/{id}`. */
export interface ExecutionStatusSnapshot {
  executionId: string;
  status: string;
  statusReason?: string;
  result?: any;
  error?: string;
  errorDetails?: unknown;
  durationMs?: number;
}

/** Options for {@link AgentFieldClient.waitForExecutionResult}. */
export interface WaitForExecutionOptions {
  /** Total timeout in milliseconds. Omit for no wall-clock cap. */
  timeoutMs?: number;
  /** Initial poll interval in milliseconds (default: 1000). */
  pollIntervalMs?: number;
  /** Maximum poll interval in milliseconds (default: 5000). */
  maxIntervalMs?: number;
  /**
   * Pause-clock whose paused time is excluded from the wall-clock timeout, so
   * a parent awaiting a paused descendant does not spuriously time out.
   */
  pauseClock?: import('../agent/pause.js').PauseClock;
  /** Fired once when the awaited child transitions into `waiting`. */
  onChildWaiting?: () => void | Promise<void>;
  /** Fired once when the awaited child leaves `waiting` (back to running). */
  onChildRunning?: () => void | Promise<void>;
}

export interface RestartExecutionOptions {
  scope?: 'workflow' | 'execution';
  reuse?: 'succeeded-before' | 'all-succeeded' | 'none';
  fork?: boolean;
  reason?: string;
  input?: Record<string, unknown>;
  context?: Record<string, unknown>;
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
      replaySourceRunId?: string;
      replayBeforeExecutionId?: string;
      replayMode?: string;
    }
  ): Promise<T> {
    const headers = this.buildExecuteHeaders(metadata);

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

  async restartExecution(executionId: string, options: RestartExecutionOptions = {}): Promise<any> {
    if (!executionId) {
      throw new Error('executionId is required');
    }
    const body = {
      scope: options.scope ?? 'workflow',
      reuse: options.reuse ?? 'succeeded-before',
      fork: options.fork,
      reason: options.reason,
      input: options.input,
      context: options.context
    };
    const bodyStr = JSON.stringify(body);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    try {
      const res = await this.http.post(
        `/api/v1/executions/${encodeURIComponent(executionId)}/restart`,
        bodyStr,
        { headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }) }
      );
      return res.data;
    } catch (err: any) {
      const respData = err?.response?.data;
      if (respData) {
        const status = err.response.status;
        const msg = respData.message || respData.error || JSON.stringify(respData);
        const enriched: ExecutionError = Object.assign(
          new Error(`restart execution ${executionId} failed (${status}): ${msg}`),
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

  /** Build the `X-*` dispatch headers shared by execute / executeAsync. */
  private buildExecuteHeaders(metadata?: ExecuteMetadata): Record<string, string> {
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
    if (metadata?.replaySourceRunId) headers['X-AgentField-Replay-Source-Run-ID'] = metadata.replaySourceRunId;
    if (metadata?.replayBeforeExecutionId) headers['X-AgentField-Replay-Before-Execution-ID'] = metadata.replayBeforeExecutionId;
    if (metadata?.replayMode) headers['X-AgentField-Replay-Mode'] = metadata.replayMode;
    return headers;
  }

  /**
   * Submit an async execution and return its `execution_id`.
   *
   * POSTs to `/api/v1/execute/async/{target}`, which enqueues the execution
   * and responds `202 Accepted` immediately (the control plane runs it and
   * tracks status out-of-band). Use with {@link waitForExecutionResult} to
   * poll for the terminal result without holding a synchronous connection —
   * this is what lets a parent await a descendant that legitimately pauses
   * (WAITING) for a long time without hitting the dispatch ceiling.
   */
  async executeAsync(
    target: string,
    input: any,
    metadata?: ExecuteMetadata
  ): Promise<string> {
    const headers = this.buildExecuteHeaders(metadata);
    const bodyStr = JSON.stringify({ input });
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    try {
      const res = await this.http.post(
        `/api/v1/execute/async/${target}`,
        bodyStr,
        { headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...headers, ...authHeaders }) }
      );
      const executionId =
        res.data?.execution_id ??
        res.data?.executionId ??
        (typeof res.headers?.['x-execution-id'] === 'string' ? res.headers['x-execution-id'] : undefined);
      if (!executionId) {
        throw new Error(`execute async ${target} returned no execution_id`);
      }
      return executionId;
    } catch (err: any) {
      const respData = err?.response?.data;
      if (respData) {
        const status = err.response.status;
        const msg = respData.message || respData.error || JSON.stringify(respData);
        const enriched: ExecutionError = Object.assign(
          new Error(`execute async ${target} failed (${status}): ${msg}`),
          { status, responseData: respData }
        );
        throw enriched;
      }
      throw err;
    }
  }

  /** Fetch the current status snapshot for an execution. */
  async getExecutionStatus(executionId: string): Promise<ExecutionStatusSnapshot> {
    const res = await this.http.get(
      `/api/v1/executions/${encodeURIComponent(executionId)}`,
      { headers: this.mergeHeaders({}) }
    );
    const data = res.data ?? {};
    return {
      executionId: data.execution_id ?? executionId,
      status: data.status ?? 'unknown',
      statusReason: data.status_reason ?? undefined,
      result: data.result,
      error: data.error ?? undefined,
      errorDetails: data.error_details,
      durationMs: data.duration_ms ?? undefined
    };
  }

  /**
   * Poll an async execution until it reaches a terminal status, returning its
   * result (or throwing on failure/cancellation/timeout).
   *
   * While polling, this observes the child's status transitions: when the
   * child enters `waiting` it starts the supplied pause-clock and fires
   * `onChildWaiting`; when the child leaves `waiting` it stops the clock and
   * fires `onChildRunning`. Paused time is excluded from `timeoutMs`, so a
   * parent awaiting a paused descendant does not time out while the descendant
   * legitimately waits. Mirrors the Python SDK's `wait_for_execution_result`.
   */
  async waitForExecutionResult<T = any>(
    executionId: string,
    opts: WaitForExecutionOptions = {}
  ): Promise<T> {
    const pollInterval = opts.pollIntervalMs ?? 1000;
    const maxInterval = opts.maxIntervalMs ?? 5000;
    const pauseClock = opts.pauseClock;
    const start = Date.now();
    let interval = pollInterval;
    let childWaiting = false;

    const activeElapsed = () => (Date.now() - start) - (pauseClock?.totalPaused() ?? 0);

    try {
      while (true) {
        let snapshot: ExecutionStatusSnapshot;
        try {
          snapshot = await this.getExecutionStatus(executionId);
        } catch {
          // Transient poll failure — back off and retry rather than aborting
          // the whole wait on a single blip.
          if (opts.timeoutMs != null && activeElapsed() >= opts.timeoutMs) {
            throw new Error(`waitForExecutionResult(${executionId}) timed out`);
          }
          await sleep(interval);
          interval = Math.min(interval * 2, maxInterval);
          continue;
        }

        const status = normalizeExecutionStatus(snapshot.status);

        // Observe WAITING transitions so the awaiter can pause its own clock
        // and cascade its status upward (multi-hop propagation).
        if (status === 'waiting' && !childWaiting) {
          childWaiting = true;
          pauseClock?.startPause();
          await safePauseCallback(opts.onChildWaiting);
        } else if (status !== 'waiting' && childWaiting) {
          childWaiting = false;
          pauseClock?.endPause();
          await safePauseCallback(opts.onChildRunning);
        }

        if (isTerminalExecutionStatus(status)) {
          if (status === 'succeeded') {
            return (snapshot.result?.result as T) ?? (snapshot.result as T);
          }
          const detail = snapshot.error || snapshot.statusReason || status;
          const failure: ExecutionError = Object.assign(
            new Error(`execution ${executionId} ${status}: ${detail}`),
            { status: 0, responseData: snapshot.errorDetails ?? snapshot.error }
          );
          throw failure;
        }

        if (opts.timeoutMs != null && activeElapsed() >= opts.timeoutMs) {
          throw new Error(`waitForExecutionResult(${executionId}) timed out`);
        }

        await sleep(interval);
        interval = Math.min(interval * 2, maxInterval);
      }
    } finally {
      // Guarantee the clock is unpaused if we exit while the child was waiting.
      if (childWaiting) {
        pauseClock?.endPause();
      }
    }
  }

  /**
   * Notify the control plane that THIS execution is now `waiting` or `running`
   * because of its awaited child's state — the multi-hop pause propagation
   * hook. Distinct from request-approval: no approval id, no webhook, no human.
   * It exists purely so ancestors watching this execution see WAITING
   * transitively while a descendant is paused, and stop counting wall-clock.
   *
   * POSTs to `/api/v1/agents/{node}/executions/{id}/awaiter-status`.
   */
  async notifyAwaiterStatus(
    executionId: string,
    status: 'waiting' | 'running',
    reason = ''
  ): Promise<void> {
    if (status !== 'waiting' && status !== 'running') {
      throw new Error(`notifyAwaiterStatus: status must be 'waiting' or 'running', got '${status}'`);
    }
    const nodeId = this.config.nodeId ?? '';
    const payload: Record<string, unknown> = { status };
    if (reason) payload.reason = reason;
    const bodyStr = JSON.stringify(payload);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));
    const res = await this.http.post(
      `/api/v1/agents/${encodeURIComponent(nodeId)}/executions/${encodeURIComponent(executionId)}/awaiter-status`,
      bodyStr,
      {
        headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }),
        timeout: 10000
      }
    );
    if (res.status >= 400) {
      throw new Error(`awaiter-status update failed (${res.status})`);
    }
  }

  /**
   * Deliver a terminal execution result to the control plane, with retries.
   *
   * This is the out-of-band completion callback used by async-execution
   * dispatch: after a reasoner has been 202-acked and run detached, its final
   * status (`succeeded` / `failed` / `cancelled`) is POSTed to
   * `/api/v1/executions/{id}/status`. Retries with exponential backoff because
   * a dropped terminal callback would leave the execution stuck forever.
   */
  async reportExecutionResult(
    executionId: string,
    payload: {
      status: string;
      result?: any;
      error?: string;
      errorDetails?: unknown;
      durationMs?: number;
      completedAt?: string;
      reasoner?: string;
    },
    maxRetries = 5
  ): Promise<boolean> {
    const body: Record<string, unknown> = {
      status: payload.status,
      execution_id: executionId,
      duration_ms: payload.durationMs,
      completed_at: payload.completedAt,
      reasoner: payload.reasoner
    };
    if (payload.result !== undefined) body.result = wrapResult(payload.result);
    if (payload.error !== undefined) body.error = payload.error;
    if (payload.errorDetails !== undefined) body.error_details = payload.errorDetails;

    const bodyStr = JSON.stringify(body);
    const authHeaders = this.didAuthenticator.signRequest(Buffer.from(bodyStr));

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const res = await this.http.post(
          `/api/v1/executions/${encodeURIComponent(executionId)}/status`,
          bodyStr,
          {
            headers: this.mergeHeaders({ 'Content-Type': 'application/json', ...authHeaders }),
            timeout: 30000
          }
        );
        if (res.status >= 200 && res.status < 300) {
          return true;
        }
      } catch {
        // fall through to backoff/retry
      }
      if (attempt < maxRetries - 1) {
        await sleep(2 ** attempt * 1000);
      }
    }
    return false;
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
    replaySourceRunId?: string;
    replayBeforeExecutionId?: string;
    replayMode?: string;
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
    if (metadata.replaySourceRunId) headers['x-agentfield-replay-source-run-id'] = metadata.replaySourceRunId;
    if (metadata.replayBeforeExecutionId) headers['x-agentfield-replay-before-execution-id'] = metadata.replayBeforeExecutionId;
    if (metadata.replayMode) headers['x-agentfield-replay-mode'] = metadata.replayMode;
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run an awaiter-status callback without letting a slow/failed control plane
 * stall the poll loop. Bounded to 2s and swallows errors — mirrors the Python
 * SDK's `_safe_pause_callback` so a transient blip can't break the call graph.
 */
async function safePauseCallback(cb?: () => void | Promise<void>): Promise<void> {
  if (!cb) return;
  try {
    await Promise.race([
      Promise.resolve().then(cb),
      sleep(2000)
    ]);
  } catch {
    // best-effort; propagation is advisory
  }
}

/**
 * The control plane's status-update `result` field is a JSON object. Wrap
 * non-object results so a reasoner returning a scalar/array still delivers a
 * terminal status instead of being rejected by the control plane. Objects pass
 * through unchanged so the common case matches sync-dispatch behaviour.
 */
function wrapResult(result: unknown): Record<string, any> {
  if (result !== null && typeof result === 'object' && !Array.isArray(result)) {
    return result as Record<string, any>;
  }
  return { result };
}
