import express from 'express';
import type http from 'node:http';
import { randomUUID } from 'node:crypto';
import type {
  AgentConfig,
  AgentHandler,
  DeploymentType,
  HealthStatus,
  ServerlessEvent,
  ServerlessResponse
} from '../types/agent.js';
import { ReasonerRegistry } from './ReasonerRegistry.js';
import { SkillRegistry } from './SkillRegistry.js';
import { AgentRouter } from '../router/AgentRouter.js';
import type { ReasonerHandler, ReasonerOptions } from '../types/reasoner.js';
import type { SkillHandler, SkillOptions } from '../types/skill.js';
import { ExecutionContext, type ExecutionMetadata } from '../context/ExecutionContext.js';
import { ReasonerContext } from '../context/ReasonerContext.js';
import { SkillContext } from '../context/SkillContext.js';
import { AIClient } from '../ai/AIClient.js';
import { AgentFieldClient } from '../client/AgentFieldClient.js';
import { MemoryClient } from '../memory/MemoryClient.js';
import { MemoryEventClient } from '../memory/MemoryEventClient.js';
import {
  MemoryInterface,
  type MemoryChangeEvent,
  type MemoryWatchHandler
} from '../memory/MemoryInterface.js';
import { DidClient } from '../did/DidClient.js';
import { DidInterface } from '../did/DidInterface.js';
import { DidManager } from '../did/DidManager.js';
import { matchesPattern } from '../utils/pattern.js';
import { WorkflowReporter } from '../workflow/WorkflowReporter.js';
import type { DiscoveryOptions } from '../types/agent.js';
import type { MCPToolRegistration } from '../types/mcp.js';
import { MCPClientRegistry } from '../mcp/MCPClientRegistry.js';
import { MCPToolRegistrar } from '../mcp/MCPToolRegistrar.js';

class TargetNotFoundError extends Error {}

export class Agent {
  readonly config: AgentConfig;
  readonly app: express.Express;
  readonly reasoners = new ReasonerRegistry();
  readonly skills = new SkillRegistry();
  private server?: http.Server;
  private heartbeatTimer?: NodeJS.Timeout;
  private readonly aiClient: AIClient;
  private readonly agentFieldClient: AgentFieldClient;
  private readonly memoryClient: MemoryClient;
  private readonly memoryEventClient: MemoryEventClient;
  private readonly didClient: DidClient;
  private readonly didManager: DidManager;
  private readonly memoryWatchers: Array<{ pattern: string; handler: MemoryWatchHandler; scope?: string; scopeId?: string }> = [];
  private readonly mcpClientRegistry?: MCPClientRegistry;
  private readonly mcpToolRegistrar?: MCPToolRegistrar;

  constructor(config: AgentConfig) {
    const mcp = config.mcp
      ? {
          autoRegisterTools: config.mcp.autoRegisterTools ?? true,
          ...config.mcp
        }
      : undefined;

    this.config = {
      port: 8001,
      agentFieldUrl: 'http://localhost:8080',
      host: '0.0.0.0',
      ...config,
      didEnabled: config.didEnabled ?? true,
      deploymentType: config.deploymentType ?? 'long_running',
      mcp
    };

    this.app = express();
    this.app.use(express.json());

    this.aiClient = new AIClient(this.config.aiConfig);
    this.agentFieldClient = new AgentFieldClient(this.config);
    this.memoryClient = new MemoryClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.memoryEventClient = new MemoryEventClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.didClient = new DidClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.didManager = new DidManager(this.didClient, this.config.nodeId);
    this.memoryEventClient.onEvent((event) => this.dispatchMemoryEvent(event));

    if (this.config.mcp?.servers?.length) {
      this.mcpClientRegistry = new MCPClientRegistry(this.config.devMode);
      this.mcpToolRegistrar = new MCPToolRegistrar(this, this.mcpClientRegistry, {
        namespace: this.config.mcp.namespace,
        tags: this.config.mcp.tags,
        devMode: this.config.devMode
      });
      this.mcpToolRegistrar.registerServers(this.config.mcp.servers);
    }

    this.registerDefaultRoutes();
  }

  reasoner<TInput = any, TOutput = any>(
    name: string,
    handler: ReasonerHandler<TInput, TOutput>,
    options?: ReasonerOptions
  ) {
    this.reasoners.register(name, handler, options);
    return this;
  }

  skill<TInput = any, TOutput = any>(
    name: string,
    handler: SkillHandler<TInput, TOutput>,
    options?: SkillOptions
  ) {
    this.skills.register(name, handler, options);
    return this;
  }

  includeRouter(router: AgentRouter) {
    this.reasoners.includeRouter(router);
    this.skills.includeRouter(router);
  }

  handler(adapter?: (event: any, context?: any) => ServerlessEvent): AgentHandler {
    return async (event: any, res?: any): Promise<ServerlessResponse | void> => {
      // If a response object is provided, treat this as a standard HTTP request (e.g., Vercel/Netlify)
      if (res && typeof res === 'object' && typeof (res as any).setHeader === 'function') {
        return this.handleHttpRequest(event as http.IncomingMessage, res as http.ServerResponse);
      }

      // Fallback to a generic serverless event contract (AWS Lambda, Cloud Functions, etc.)
      const normalized = adapter ? adapter(event) : (event as ServerlessEvent);
      return this.handleServerlessEvent(normalized);
    };
  }

  watchMemory(pattern: string | string[], handler: MemoryWatchHandler, options?: { scope?: string; scopeId?: string }) {
    const patterns = Array.isArray(pattern) ? pattern : [pattern];
    patterns.forEach((p) =>
      this.memoryWatchers.push({ pattern: p, handler, scope: options?.scope, scopeId: options?.scopeId })
    );
    this.memoryEventClient.start();
  }

  discover(options?: DiscoveryOptions) {
    return this.agentFieldClient.discoverCapabilities(options);
  }

  async registerMcpTools(): Promise<{ registered: MCPToolRegistration[] }> {
    if (!this.mcpToolRegistrar) return { registered: [] };
    return this.mcpToolRegistrar.registerAll();
  }

  getAIClient() {
    return this.aiClient;
  }

  getMemoryInterface(metadata?: ExecutionMetadata) {
    const defaultScope = this.config.memoryConfig?.defaultScope ?? 'workflow';
    const defaultScopeId =
      defaultScope === 'session'
        ? metadata?.sessionId
        : defaultScope === 'actor'
          ? metadata?.actorId
          : metadata?.workflowId ?? metadata?.runId ?? metadata?.sessionId ?? metadata?.actorId;
    return new MemoryInterface({
      client: this.memoryClient,
      eventClient: this.memoryEventClient,
      aiClient: this.aiClient,
      defaultScope,
      defaultScopeId,
      metadata: {
        workflowId: metadata?.workflowId ?? metadata?.runId,
        sessionId: metadata?.sessionId,
        actorId: metadata?.actorId,
        runId: metadata?.runId,
        executionId: metadata?.executionId,
        parentExecutionId: metadata?.parentExecutionId,
        callerDid: metadata?.callerDid,
        targetDid: metadata?.targetDid,
        agentNodeDid: metadata?.agentNodeDid,
        agentNodeId: this.config.nodeId
      }
    });
  }

  getWorkflowReporter(metadata: ExecutionMetadata) {
    return new WorkflowReporter(this.agentFieldClient, {
      executionId: metadata.executionId,
      runId: metadata.runId,
      workflowId: metadata.workflowId,
      agentNodeId: this.config.nodeId
    });
  }

  getDidInterface(metadata: ExecutionMetadata, defaultInput?: any, targetName?: string) {
    // Resolve DIDs from the identity package if available
    const agentNodeDid = metadata.agentNodeDid
      ?? this.didManager.getAgentDid()
      ?? this.config.defaultHeaders?.['X-Agent-Node-DID']?.toString();

    // For caller DID: use provided value, or fall back to agent DID
    const callerDid = metadata.callerDid ?? this.didManager.getAgentDid();

    // For target DID: use provided value, or resolve from function name
    const targetDid = metadata.targetDid
      ?? (targetName ? this.didManager.getFunctionDid(targetName) : undefined)
      ?? this.didManager.getAgentDid();

    return new DidInterface({
      client: this.didClient,
      metadata: {
        ...metadata,
        agentNodeDid,
        callerDid,
        targetDid
      },
      enabled: Boolean(this.config.didEnabled),
      defaultInput
    });
  }

  async serve(): Promise<void> {
    if (this.config.mcp?.autoRegisterTools !== false) {
      try {
        await this.registerMcpTools();
      } catch (err) {
        if (this.config.devMode) {
          console.warn('MCP tool registration failed', err);
        }
      }
    }

    await this.registerWithControlPlane();
    const port = this.config.port ?? 8001;
    const host = this.config.host ?? '0.0.0.0';
    // First heartbeat marks the node as starting; subsequent interval sets ready.
    await this.agentFieldClient.heartbeat('starting');
    await new Promise<void>((resolve, reject) => {
      this.server = this.app
        .listen(port, host, () => resolve())
        .on('error', reject);
    });
    this.memoryEventClient.start();
    this.startHeartbeat();
  }

  async shutdown(): Promise<void> {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }
    await new Promise<void>((resolve, reject) => {
      this.server?.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    this.memoryEventClient.stop();
  }

  async call(target: string, input: any) {
    const { agentId, name } = this.parseTarget(target);
    if (!agentId || agentId === this.config.nodeId) {
      const local = this.reasoners.get(name);
      if (!local) throw new Error(`Reasoner not found: ${name}`);
      const parentMetadata = ExecutionContext.getCurrent()?.metadata;
      const runId = parentMetadata?.runId ?? parentMetadata?.executionId ?? randomUUID();
      const metadata = {
        ...parentMetadata,
        executionId: randomUUID(),
        parentExecutionId: parentMetadata?.executionId,
        runId,
        workflowId: parentMetadata?.workflowId ?? runId
      };
      const dummyReq = {} as express.Request;
      const dummyRes = {} as express.Response;
      const execCtx = new ExecutionContext({
        input,
        metadata: {
          ...metadata,
          executionId: metadata.executionId ?? randomUUID()
        },
        req: dummyReq,
        res: dummyRes,
        agent: this
      });
      const startTime = Date.now();

      const emitEvent = async (status: 'running' | 'succeeded' | 'failed', payload: any) => {
        await this.agentFieldClient.publishWorkflowEvent({
          executionId: execCtx.metadata.executionId,
          runId: execCtx.metadata.runId ?? execCtx.metadata.executionId,
          workflowId: execCtx.metadata.workflowId,
          reasonerId: name,
          agentNodeId: this.config.nodeId,
          status,
          parentExecutionId: execCtx.metadata.parentExecutionId,
          parentWorkflowId: execCtx.metadata.workflowId,
          inputData: status === 'running' ? input : undefined,
          result: status === 'succeeded' ? payload : undefined,
          error: status === 'failed' ? (payload?.message ?? String(payload)) : undefined,
          durationMs: status === 'running' ? undefined : Date.now() - startTime
        });
      };

      await emitEvent('running', null);

      return ExecutionContext.run(execCtx, async () => {
        try {
          const result = await local.handler(
            new ReasonerContext({
              input,
              executionId: execCtx.metadata.executionId,
              runId: execCtx.metadata.runId,
              sessionId: execCtx.metadata.sessionId,
              actorId: execCtx.metadata.actorId,
              workflowId: execCtx.metadata.workflowId,
              parentExecutionId: execCtx.metadata.parentExecutionId,
              callerDid: execCtx.metadata.callerDid,
              targetDid: execCtx.metadata.targetDid,
              agentNodeDid: execCtx.metadata.agentNodeDid,
              req: dummyReq,
              res: dummyRes,
              agent: this,
              aiClient: this.aiClient,
              memory: this.getMemoryInterface(execCtx.metadata),
              workflow: this.getWorkflowReporter(execCtx.metadata),
              did: this.getDidInterface(execCtx.metadata, input, name)
            })
          );
          await emitEvent('succeeded', result);
          return result;
        } catch (err) {
          await emitEvent('failed', err);
          throw err;
        }
      });
    }

    const metadata = ExecutionContext.getCurrent()?.metadata;
    return this.agentFieldClient.execute(target, input, {
      runId: metadata?.runId ?? metadata?.executionId,
      workflowId: metadata?.workflowId ?? metadata?.runId,
      parentExecutionId: metadata?.executionId,
      sessionId: metadata?.sessionId,
      actorId: metadata?.actorId,
      callerDid: metadata?.callerDid,
      targetDid: metadata?.targetDid,
      agentNodeDid: metadata?.agentNodeDid,
      agentNodeId: this.config.nodeId
    });
  }

  private registerDefaultRoutes() {
    this.app.get('/health', (_req, res) => {
      res.json(this.health());
    });

    // Discovery endpoint used for serverless registration (mirrors Python behaviour)
    this.app.get('/discover', (_req, res) => {
      res.json(this.discoveryPayload(this.config.deploymentType ?? 'long_running'));
    });

    // MCP health probe expected by control-plane UI
    this.app.get('/health/mcp', async (_req, res) => {
      if (!this.mcpClientRegistry) {
        res.json({ status: 'disabled', totalServers: 0, healthyServers: 0, servers: [] });
        return;
      }

      try {
        const summary = await this.mcpClientRegistry.healthSummary();
        res.json(summary);
      } catch (err: any) {
        res.status(500).json({ status: 'error', error: err?.message ?? 'MCP health check failed' });
      }
    });

    this.app.get('/mcp/status', (_req, res) => {
      res.json(this.mcpStatus());
    });

    this.app.get('/status', (_req, res) => {
      res.json({
        ...this.health(),
        reasoners: this.reasoners.all().map((r) => r.name),
        skills: this.skills.all().map((s) => s.name)
      });
    });

    this.app.get('/reasoners', (_req, res) => {
      res.json(this.reasoners.all().map((r) => r.name));
    });

    this.app.get('/skills', (_req, res) => {
      res.json(this.skills.all().map((s) => s.name));
    });

    this.app.post('/api/v1/reasoners/*', (req, res) => this.executeReasoner(req, res, (req.params as any)[0]));
    this.app.post('/reasoners/:name', (req, res) => this.executeReasoner(req, res, req.params.name));

    this.app.post('/api/v1/skills/*', (req, res) => this.executeSkill(req, res, (req.params as any)[0]));
    this.app.post('/skills/:name', (req, res) => this.executeSkill(req, res, req.params.name));

    // Serverless-friendly execute endpoint that accepts { target, input } or { reasoner, input }
    this.app.post('/execute', (req, res) => this.executeServerlessHttp(req, res));
    this.app.post('/execute/:name', (req, res) => this.executeServerlessHttp(req, res, req.params.name));
  }

  private async executeReasoner(req: express.Request, res: express.Response, name: string) {
    try {
      await this.executeInvocation({
        targetName: name,
        targetType: 'reasoner',
        input: req.body,
        metadata: this.buildMetadata(req),
        req,
        res,
        respond: true
      });
    } catch (err: any) {
      if (err instanceof TargetNotFoundError) {
        res.status(404).json({ error: err.message });
      } else {
        res.status(500).json({ error: err?.message ?? 'Execution failed' });
      }
    }
  }

  private async executeSkill(req: express.Request, res: express.Response, name: string) {
    try {
      await this.executeInvocation({
        targetName: name,
        targetType: 'skill',
        input: req.body,
        metadata: this.buildMetadata(req),
        req,
        res,
        respond: true
      });
    } catch (err: any) {
      if (err instanceof TargetNotFoundError) {
        res.status(404).json({ error: err.message });
      } else {
        res.status(500).json({ error: err?.message ?? 'Execution failed' });
      }
    }
  }

  private buildMetadata(req: express.Request) {
    return this.buildMetadataFromHeaders(req.headers);
  }

  private async executeServerlessHttp(req: express.Request, res: express.Response, explicitName?: string) {
    const invocation = this.extractInvocationDetails({
      path: req.path,
      explicitTarget: explicitName,
      query: req.query as Record<string, any>,
      body: req.body
    });

    if (!invocation.name) {
      res.status(400).json({ error: "Missing 'target' or 'reasoner' in request" });
      return;
    }

    try {
      const result = await this.executeInvocation({
        targetName: invocation.name,
        targetType: invocation.targetType,
        input: invocation.input,
        metadata: this.buildMetadata(req),
        req,
        res,
        respond: true
      });

      if (result !== undefined && !res.headersSent) {
        res.json(result);
      }
    } catch (err: any) {
      if (err instanceof TargetNotFoundError) {
        res.status(404).json({ error: err.message });
      } else {
        res.status(500).json({ error: err?.message ?? 'Execution failed' });
      }
    }
  }

  private buildMetadataFromHeaders(
    headers: Record<string, string | string[] | undefined>,
    overrides?: Partial<ExecutionMetadata>
  ): ExecutionMetadata {
    const normalized: Record<string, string | undefined> = {};
    Object.entries(headers ?? {}).forEach(([key, value]) => {
      normalized[key.toLowerCase()] = Array.isArray(value) ? value[0] : value;
    });

    const executionId = overrides?.executionId ?? normalized['x-execution-id'] ?? randomUUID();
    const runId = overrides?.runId ?? normalized['x-run-id'] ?? executionId;
    const workflowId = overrides?.workflowId ?? normalized['x-workflow-id'] ?? runId;

    return {
      executionId,
      runId,
      workflowId,
      sessionId: overrides?.sessionId ?? normalized['x-session-id'],
      actorId: overrides?.actorId ?? normalized['x-actor-id'],
      parentExecutionId: overrides?.parentExecutionId ?? normalized['x-parent-execution-id'],
      callerDid: overrides?.callerDid ?? normalized['x-caller-did'],
      targetDid: overrides?.targetDid ?? normalized['x-target-did'],
      agentNodeDid:
        overrides?.agentNodeDid ?? normalized['x-agent-node-did'] ?? normalized['x-agent-did']
    };
  }

  private handleHttpRequest(req: http.IncomingMessage | express.Request, res: http.ServerResponse | express.Response) {
    const handler = this.app as unknown as (req: http.IncomingMessage, res: http.ServerResponse) => void;
    return handler(req as any, res as any);
  }

  private async handleServerlessEvent(event: ServerlessEvent): Promise<ServerlessResponse> {
    const path = event?.path ?? event?.rawPath ?? '';
    const action = event?.action ?? '';

    if (path === '/discover' || action === 'discover') {
      return {
        statusCode: 200,
        headers: { 'content-type': 'application/json' },
        body: this.discoveryPayload(this.config.deploymentType ?? 'serverless')
      };
    }

    const body = this.normalizeEventBody(event);
    const invocation = this.extractInvocationDetails({
      path,
      query: event?.queryStringParameters,
      body,
      reasoner: event?.reasoner,
      target: event?.target,
      skill: event?.skill,
      type: event?.type
    });

    if (!invocation.name) {
      return {
        statusCode: 400,
        headers: { 'content-type': 'application/json' },
        body: { error: "Missing 'target' or 'reasoner' in request" }
      };
    }

    const metadata = this.buildMetadataFromHeaders(event?.headers ?? {}, this.mergeExecutionContext(event));

    try {
      const result = await this.executeInvocation({
        targetName: invocation.name,
        targetType: invocation.targetType,
        input: invocation.input,
        metadata
      });

      return { statusCode: 200, headers: { 'content-type': 'application/json' }, body: result };
    } catch (err: any) {
      if (err instanceof TargetNotFoundError) {
        return {
          statusCode: 404,
          headers: { 'content-type': 'application/json' },
          body: { error: err.message }
        };
      }

      return {
        statusCode: 500,
        headers: { 'content-type': 'application/json' },
        body: { error: err?.message ?? 'Execution failed' }
      };
    }
  }

  private normalizeEventBody(event: ServerlessEvent) {
    const parsed = this.parseBody((event as any)?.body);
    if (parsed && typeof parsed === 'object' && event?.input !== undefined && (parsed as any).input === undefined) {
      return { ...(parsed as Record<string, any>), input: event.input };
    }
    if ((parsed === undefined || parsed === null) && event?.input !== undefined) {
      return { input: event.input };
    }
    return parsed;
  }

  private mergeExecutionContext(event: ServerlessEvent): Partial<ExecutionMetadata> {
    const ctx = (event?.executionContext ?? (event as any)?.execution_context) as Partial<
      ExecutionMetadata & {
        execution_id?: string;
        run_id?: string;
        workflow_id?: string;
        parent_execution_id?: string;
        session_id?: string;
        actor_id?: string;
        caller_did?: string;
        target_did?: string;
        agent_node_did?: string;
      }
    >;

    if (!ctx) return {};

    return {
      executionId: (ctx as any).executionId ?? ctx.execution_id ?? ctx.executionId,
      runId: ctx.runId ?? (ctx as any).run_id,
      workflowId: ctx.workflowId ?? (ctx as any).workflow_id,
      parentExecutionId: ctx.parentExecutionId ?? (ctx as any).parent_execution_id,
      sessionId: ctx.sessionId ?? (ctx as any).session_id,
      actorId: ctx.actorId ?? (ctx as any).actor_id,
      callerDid: (ctx as any).callerDid ?? (ctx as any).caller_did,
      targetDid: (ctx as any).targetDid ?? (ctx as any).target_did,
      agentNodeDid: (ctx as any).agentNodeDid ?? (ctx as any).agent_node_did
    };
  }

  private extractInvocationDetails(params: {
    path?: string;
    explicitTarget?: string;
    query?: Record<string, any>;
    body?: any;
    reasoner?: string;
    target?: string;
    skill?: string;
    type?: string;
  }): { name?: string; targetType?: 'reasoner' | 'skill'; input: any } {
    const pathTarget = this.parsePathTarget(params.path);
    const name =
      this.firstDefined<string>(
        params.explicitTarget,
        pathTarget.name,
        params.query?.target,
        params.query?.reasoner,
        params.query?.skill,
        params.target,
        params.reasoner,
        params.skill,
        params.body?.target,
        params.body?.reasoner,
        params.body?.skill
      ) ?? pathTarget.name;

    const typeValue = (this.firstDefined<string>(
      pathTarget.targetType,
      params.type,
      params.query?.type,
      params.query?.targetType,
      params.body?.type,
      params.body?.targetType
    ) ?? undefined) as 'reasoner' | 'skill' | undefined;

    const input = this.normalizeInputPayload(params.body);

    return { name: name ?? undefined, targetType: typeValue, input };
  }

  private parsePathTarget(
    path?: string
  ): { name?: string; targetType?: 'reasoner' | 'skill' } {
    if (!path) return {};

    const normalized = path.split('?')[0];
    const reasonerMatch = normalized.match(/\/reasoners\/([^/]+)/);
    if (reasonerMatch?.[1]) {
      return { name: reasonerMatch[1], targetType: 'reasoner' };
    }

    const skillMatch = normalized.match(/\/skills\/([^/]+)/);
    if (skillMatch?.[1]) {
      return { name: skillMatch[1], targetType: 'skill' };
    }

    const executeMatch = normalized.match(/\/execute\/([^/]+)/);
    if (executeMatch?.[1]) {
      return { name: executeMatch[1] };
    }

    return {};
  }

  private parseBody(body: any) {
    if (body === undefined || body === null) return body;
    if (typeof body === 'string') {
      try {
        return JSON.parse(body);
      } catch {
        return body;
      }
    }
    return body;
  }

  private normalizeInputPayload(body: any) {
    if (body === undefined || body === null) return {};
    const parsed = this.parseBody(body);

    if (parsed && typeof parsed === 'object') {
      const { target, reasoner, skill, type, targetType, ...rest } = parsed as Record<string, any>;
      if ((parsed as any).input !== undefined) {
        return (parsed as any).input;
      }
      if ((parsed as any).data !== undefined) {
        return (parsed as any).data;
      }
      if (Object.keys(rest).length === 0) {
        return {};
      }
      return rest;
    }

    return parsed;
  }

  private firstDefined<T>(...values: Array<T | undefined | null>): T | undefined {
    for (const value of values) {
      if (value !== undefined && value !== null) {
        return value as T;
      }
    }
    return undefined;
  }

  private reasonerDefinitions() {
    return this.reasoners.all().map((r) => ({
      id: r.name,
      input_schema: r.options?.inputSchema ?? {},
      output_schema: r.options?.outputSchema ?? {},
      memory_config: r.options?.memoryConfig ?? {
        auto_inject: [] as string[],
        memory_retention: '',
        cache_results: false
      },
      tags: r.options?.tags ?? []
    }));
  }

  private skillDefinitions() {
    return this.skills.all().map((s) => ({
      id: s.name,
      input_schema: s.options?.inputSchema ?? {},
      tags: s.options?.tags ?? []
    }));
  }

  private discoveryPayload(deploymentType: DeploymentType) {
    return {
      node_id: this.config.nodeId,
      version: this.config.version,
      deployment_type: deploymentType,
      reasoners: this.reasonerDefinitions(),
      skills: this.skillDefinitions()
    };
  }

  private async executeInvocation(params: {
    targetName: string;
    targetType?: 'reasoner' | 'skill';
    input: any;
    metadata: ExecutionMetadata;
    req?: express.Request;
    res?: express.Response;
    respond?: boolean;
  }) {
    const targetType = params.targetType;

    if (targetType === 'skill') {
      const skill = this.skills.get(params.targetName);
      if (!skill) {
        throw new TargetNotFoundError(`Skill not found: ${params.targetName}`);
      }
      return this.runSkill(skill, params);
    }

    const reasoner = this.reasoners.get(params.targetName);
    if (reasoner) {
      return this.runReasoner(reasoner, params);
    }

    const fallbackSkill = this.skills.get(params.targetName);
    if (fallbackSkill) {
      return this.runSkill(fallbackSkill, params);
    }

    throw new TargetNotFoundError(`Reasoner not found: ${params.targetName}`);
  }

  private async runReasoner(
    reasoner: { handler: ReasonerHandler<any, any> },
    params: {
      targetName: string;
      input: any;
      metadata: ExecutionMetadata;
      req?: express.Request;
      res?: express.Response;
      respond?: boolean;
    }
  ) {
    const req = params.req ?? ({} as express.Request);
    const res = params.res ?? ({} as express.Response);
    const execCtx = new ExecutionContext({
      input: params.input,
      metadata: params.metadata,
      req,
      res,
      agent: this
    });

    return ExecutionContext.run(execCtx, async () => {
      try {
        const ctx = new ReasonerContext({
          input: params.input,
          executionId: params.metadata.executionId,
          runId: params.metadata.runId,
          sessionId: params.metadata.sessionId,
          actorId: params.metadata.actorId,
          workflowId: params.metadata.workflowId,
          parentExecutionId: params.metadata.parentExecutionId,
          callerDid: params.metadata.callerDid,
          targetDid: params.metadata.targetDid,
          agentNodeDid: params.metadata.agentNodeDid,
          req,
          res,
          agent: this,
          aiClient: this.aiClient,
          memory: this.getMemoryInterface(params.metadata),
          workflow: this.getWorkflowReporter(params.metadata),
          did: this.getDidInterface(params.metadata, params.input, params.targetName)
        });

        const result = await reasoner.handler(ctx);
        if (params.respond && params.res) {
          params.res.json(result);
          return;
        }
        return result;
      } catch (err: any) {
        if (params.respond && params.res) {
          params.res.status(500).json({ error: err?.message ?? 'Execution failed' });
          return;
        }
        throw err;
      }
    });
  }

  private async runSkill(
    skill: { handler: SkillHandler<any, any> },
    params: {
      targetName: string;
      input: any;
      metadata: ExecutionMetadata;
      req?: express.Request;
      res?: express.Response;
      respond?: boolean;
    }
  ) {
    const req = params.req ?? ({} as express.Request);
    const res = params.res ?? ({} as express.Response);
    const execCtx = new ExecutionContext({
      input: params.input,
      metadata: params.metadata,
      req,
      res,
      agent: this
    });

    return ExecutionContext.run(execCtx, async () => {
      try {
        const ctx = new SkillContext({
          input: params.input,
          executionId: params.metadata.executionId,
          sessionId: params.metadata.sessionId,
          workflowId: params.metadata.workflowId,
          req,
          res,
          agent: this,
          memory: this.getMemoryInterface(params.metadata),
          workflow: this.getWorkflowReporter(params.metadata),
          did: this.getDidInterface(params.metadata, params.input, params.targetName)
        });

        const result = await skill.handler(ctx);
        if (params.respond && params.res) {
          params.res.json(result);
          return;
        }
        return result;
      } catch (err: any) {
        if (params.respond && params.res) {
          params.res.status(500).json({ error: err?.message ?? 'Execution failed' });
          return;
        }
        throw err;
      }
    });
  }

  private async registerWithControlPlane() {
    try {
      const reasoners = this.reasonerDefinitions();
      const skills = this.skillDefinitions();

      const port = this.config.port ?? 8001;
      const hostForUrl = this.config.publicUrl
        ? undefined
        : (this.config.host && this.config.host !== '0.0.0.0' ? this.config.host : '127.0.0.1');
      const publicUrl =
        this.config.publicUrl ?? `http://${hostForUrl ?? '127.0.0.1'}:${port}`;

      await this.agentFieldClient.register({
        id: this.config.nodeId,
        version: this.config.version,
        base_url: publicUrl,
        public_url: publicUrl,
        deployment_type: this.config.deploymentType ?? 'long_running',
        reasoners,
        skills
      });

      // Register with DID system if enabled
      if (this.config.didEnabled) {
        try {
          const didRegistered = await this.didManager.registerAgent(reasoners, skills);
          if (didRegistered) {
            const summary = this.didManager.getIdentitySummary();
            console.log(`[DID] Agent registered with DID: ${summary.agentDid}`);
            console.log(`[DID] Reasoner DIDs: ${summary.reasonerCount}, Skill DIDs: ${summary.skillCount}`);
          }
        } catch (didErr) {
          if (!this.config.devMode) {
            console.warn('[DID] DID registration failed:', didErr);
          }
          // DID registration failure is non-fatal - agent can still operate without VCs
        }
      }
    } catch (err) {
      if (!this.config.devMode) {
        throw err;
      }
      console.warn('Control plane registration failed (devMode=true), continuing locally', err);
    }
  }

  private startHeartbeat() {
    const interval = this.config.heartbeatIntervalMs ?? 30_000;
    if (interval <= 0) return;

    const tick = async () => {
      try {
        await this.agentFieldClient.heartbeat('ready');
      } catch (err) {
        if (!this.config.devMode) {
          console.warn('Heartbeat failed', err);
        }
      }
    };

    this.heartbeatTimer = setInterval(tick, interval);
    tick();
  }

  private health(): HealthStatus {
    return {
      status: 'running',
      node_id: this.config.nodeId,
      version: this.config.version
    };
  }

  private mcpStatus() {
    const servers = this.mcpClientRegistry
      ? this.mcpClientRegistry.list().map((client) => ({
          alias: client.alias,
          baseUrl: client.baseUrl,
          transport: client.transport
        }))
      : [];

    const skills = this.skills
      .all()
      .filter((skill) => skill.options?.tags?.includes('mcp'))
      .map((skill) => skill.name);

    return {
      status: servers.length ? 'configured' : 'disabled',
      servers,
      skills
    };
  }

  private dispatchMemoryEvent(event: MemoryChangeEvent) {
    this.memoryWatchers.forEach(({ pattern, handler, scope, scopeId }) => {
      const scopeMatch = (!scope || scope === event.scope) && (!scopeId || scopeId === event.scopeId);
      if (scopeMatch && matchesPattern(pattern, event.key)) {
        handler(event);
      }
    });
  }

  private parseTarget(target: string): { agentId?: string; name: string } {
    if (!target.includes('.')) {
      return { name: target };
    }
    const [agentId, remainder] = target.split('.', 2);
    const name = remainder.replace(':', '/');
    return { agentId, name };
  }
}
