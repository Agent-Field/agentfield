import express from 'express';
import rateLimit from 'express-rate-limit';
import type http from 'node:http';
import { randomUUID } from 'node:crypto';
import axios, { AxiosInstance } from 'axios';
import type {
  AgentConfig,
  AgentHandler,
  DeploymentType,
  HealthStatus,
  ServerlessEvent,
  ServerlessResponse,
  RawExecutionContext
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
import type { HarnessRunner } from '../harness/runner.js';
import type { HarnessOptions, HarnessResult } from '../harness/types.js';
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
import { toJsonSchema } from '../utils/schema.js';
import { WorkflowReporter } from '../workflow/WorkflowReporter.js';
import type { DiscoveryOptions } from '../types/agent.js';
import {
  createExecutionLogger,
  type ExecutionLogContext,
  type ExecutionLogger
} from '../observability/ExecutionLogger.js';
import { LocalVerifier } from '../verification/LocalVerifier.js';
import type { Request, Response } from 'express';
import type { ParamsDictionary } from 'express-serve-static-core';
import {
  installStdioLogCapture,
  ProcessLogRing,
  registerAgentfieldLogsRoute
} from './processLogs.js';

interface WildcardParams extends ParamsDictionary {
  0: string;
}
class TargetNotFoundError extends Error {}

const harnessRunners = new WeakMap<object, HarnessRunner>();

function normalizeExecutionContext(
  ctx: RawExecutionContext
): Partial<ExecutionMetadata> {
  return {
    executionId: ctx.executionId ?? ctx.execution_id,
    runId: ctx.runId ?? ctx.run_id,
    workflowId: ctx.workflowId ?? ctx.workflow_id,
    rootWorkflowId: ctx.rootWorkflowId ?? ctx.root_workflow_id,
    parentExecutionId: ctx.parentExecutionId ?? ctx.parent_execution_id,
    reasonerId: ctx.reasonerId ?? ctx.reasoner_id,
    sessionId: ctx.sessionId ?? ctx.session_id,
    actorId: ctx.actorId ?? ctx.actor_id,
    callerDid: ctx.callerDid ?? ctx.caller_did,
    targetDid: ctx.targetDid ?? ctx.target_did,
    agentNodeDid: ctx.agentNodeDid ?? ctx.agent_node_did
  };
}

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
  private readonly localVerifier?: LocalVerifier;
  private readonly realtimeValidationFunctions = new Set<string>();
  private readonly processLogRing = new ProcessLogRing();
  private readonly executionLogger: ExecutionLogger;

  constructor(config: AgentConfig) {
    this.config = {
      port: 8001,
      agentFieldUrl: 'http://localhost:8080',
      host: '0.0.0.0',
      ...config,
      didEnabled: config.didEnabled ?? true,
      deploymentType: config.deploymentType ?? 'long_running'
    };

    this.app = express();
    this.app.use(express.json());

    this.aiClient = new AIClient(this.config.aiConfig);
    this.agentFieldClient = new AgentFieldClient(this.config);
    this.memoryClient = new MemoryClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.memoryEventClient = new MemoryEventClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.didClient = new DidClient(this.config.agentFieldUrl!, this.config.defaultHeaders);
    this.didManager = new DidManager(this.didClient, this.config.nodeId);
    this.executionLogger = createExecutionLogger({
      contextProvider: () => this.buildExecutionLogContext(),
      transport: {
        emit: (payload) => this.agentFieldClient.publishExecutionLogs(payload)
      }
    });
    this.memoryEventClient.onEvent((event) => this.dispatchMemoryEvent(event));


    // Initialize local verifier for decentralized verification
    if (this.config.localVerification && this.config.agentFieldUrl) {
      this.localVerifier = new LocalVerifier(
        this.config.agentFieldUrl,
        this.config.verificationRefreshInterval ?? 300,
        300,
        this.config.apiKey,
      );
    }

    this.registerDefaultRoutes();
    installStdioLogCapture(this.processLogRing);
    registerAgentfieldLogsRoute(this.app, this.processLogRing);
  }

  reasoner<TInput = any, TOutput = any>(
    name: string,
    handler: ReasonerHandler<TInput, TOutput>,
    options?: ReasonerOptions
  ) {
    this.reasoners.register(name, handler, options);
    if (options?.requireRealtimeValidation) {
      this.realtimeValidationFunctions.add(name);
    }
    return this;
  }

  skill<TInput = any, TOutput = any>(
    name: string,
    handler: SkillHandler<TInput, TOutput>,
    options?: SkillOptions
  ) {
    this.skills.register(name, handler, options);
    if (options?.requireRealtimeValidation) {
      this.realtimeValidationFunctions.add(name);
    }
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

  getAIClient() {
    return this.aiClient;
  }

  getExecutionLogger() {
    return this.executionLogger;
  }

  async getHarnessRunner(): Promise<HarnessRunner> {
    const cached = harnessRunners.get(this);
    if (cached) return cached;
    const { HarnessRunner: RunnerClass } = await import('../harness/runner.js');
    const runner = new RunnerClass(this.config.harnessConfig);
    harnessRunners.set(this, runner);
    return runner;
  }

  async harness(prompt: string, options?: HarnessOptions): Promise<HarnessResult> {
    const runner = await this.getHarnessRunner();
    return runner.run(prompt, options ?? {});
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
      agentNodeId: this.config.nodeId,
      reasonerId: metadata.reasonerId
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

  note(message: string, tags: string[] = [], metadata?: ExecutionMetadata): void {
    const execCtx = ExecutionContext.getCurrent();
    const execMetadata = metadata ?? execCtx?.metadata;
    if (!execMetadata) return;

    const baseUrl = (this.config.agentFieldUrl ?? 'http://localhost:8080').replace(/\/$/, '');
    let uiApiUrl = baseUrl.replace(/\/api\/v1$/, '/api/ui/v1');
    if (!uiApiUrl.includes('/api/ui/v1')) {
      uiApiUrl = `${baseUrl}/api/ui/v1`;
    }

    this.agentFieldClient.sendNote(message, tags, this.config.nodeId, execMetadata, uiApiUrl, this.config.devMode);
  }

  private buildExecutionLogContext(metadata?: ExecutionMetadata): ExecutionLogContext | undefined {
    const current = metadata ?? ExecutionContext.getCurrent()?.metadata;
    if (!current) return undefined;

    return {
      executionId: current.executionId,
      runId: current.runId,
      workflowId: current.workflowId,
      rootWorkflowId: current.rootWorkflowId ?? current.workflowId ?? current.runId ?? current.executionId,
      parentExecutionId: current.parentExecutionId,
      sessionId: current.sessionId,
      actorId: current.actorId,
      agentNodeId: this.config.nodeId,
      reasonerId: current.reasonerId,
      callerDid: current.callerDid,
      targetDid: current.targetDid,
      agentNodeDid: current.agentNodeDid
    };
  }

  async serve(): Promise<void> {
    await this.registerWithControlPlane();

    // Perform a blocking initial refresh for local verification before accepting requests
    if (this.localVerifier) {
      try {
        const ok = await this.localVerifier.refresh();
        if (!ok) {
          console.warn('[LocalVerifier] Initial refresh partially failed — some verification data may be stale');
        }
      } catch (err) {
        console.warn('[LocalVerifier] Initial refresh failed:', err);
      }
    }

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
    const parentMetadata = ExecutionContext.getCurrent()?.metadata;
    if (!agentId || agentId === this.config.nodeId) {
      const local = this.reasoners.get(name);
      if (!local) throw new Error(`Reasoner not found: ${name}`);
      const runId = parentMetadata?.runId ?? parentMetadata?.executionId ?? randomUUID();
      const rootWorkflowId = parentMetadata?.rootWorkflowId ?? parentMetadata?.workflowId ?? runId;
      const metadata = {
        ...parentMetadata,
        executionId: randomUUID(),
        parentExecutionId: parentMetadata?.executionId,
        runId,
        workflowId: parentMetadata?.workflowId ?? runId,
        rootWorkflowId,
        reasonerId: name
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
      this.executionLogger.system('agent.call.started', 'Local agent call started', {
        target,
        reasonerId: name,
        executionId: metadata.executionId,
        parentExecutionId: metadata.parentExecutionId,
        runId: metadata.runId,
        workflowId: metadata.workflowId,
        rootWorkflowId: metadata.rootWorkflowId
      });

      const emitEvent = async (status: 'running' | 'succeeded' | 'failed', payload: any) => {
        await this.agentFieldClient.publishWorkflowEvent({
          executionId: execCtx.metadata.executionId,
          runId: execCtx.metadata.runId ?? execCtx.metadata.executionId,
          workflowId: execCtx.metadata.workflowId,
          rootWorkflowId: execCtx.metadata.rootWorkflowId,
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
        this.executionLogger.system('execution.started', 'Execution started', {
          target,
          reasonerId: name,
          executionId: execCtx.metadata.executionId,
          parentExecutionId: execCtx.metadata.parentExecutionId,
          runId: execCtx.metadata.runId,
          workflowId: execCtx.metadata.workflowId,
          rootWorkflowId: execCtx.metadata.rootWorkflowId
        });
        this.executionLogger.system('reasoner.started', 'Reasoner execution started', {
          target: name,
          executionId: execCtx.metadata.executionId,
          runId: execCtx.metadata.runId,
          workflowId: execCtx.metadata.workflowId,
          rootWorkflowId: execCtx.metadata.rootWorkflowId
        });
        try {
          const result = await local.handler(
            new ReasonerContext({
              input,
              executionId: execCtx.metadata.executionId,
              runId: execCtx.metadata.runId,
              sessionId: execCtx.metadata.sessionId,
              actorId: execCtx.metadata.actorId,
              workflowId: execCtx.metadata.workflowId,
              rootWorkflowId: execCtx.metadata.rootWorkflowId,
              parentExecutionId: execCtx.metadata.parentExecutionId,
              reasonerId: name,
              callerDid: execCtx.metadata.callerDid,
              targetDid: execCtx.metadata.targetDid,
              agentNodeDid: execCtx.metadata.agentNodeDid,
              req: dummyReq,
              res: dummyRes,
              agent: this,
              logger: this.executionLogger,
              aiClient: this.aiClient,
              memory: this.getMemoryInterface(execCtx.metadata),
              workflow: this.getWorkflowReporter(execCtx.metadata),
              did: this.getDidInterface(execCtx.metadata, input, name)
            })
          );
          this.executionLogger.system('reasoner.completed', 'Reasoner execution completed', {
            target: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime
          });
          this.executionLogger.system('execution.completed', 'Execution completed', {
            target,
            reasonerId: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime
          });
          this.executionLogger.system('agent.call.completed', 'Local agent call completed', {
            target,
            reasonerId: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime
          });
          await emitEvent('succeeded', result);
          return result;
        } catch (err) {
          this.executionLogger.error('Reasoner execution failed', {
            target: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime,
            error: err instanceof Error ? err.message : String(err)
          }, {
            eventType: 'reasoner.failed',
            source: 'sdk.runtime',
            systemGenerated: true
          });
          this.executionLogger.error('Execution failed', {
            target,
            reasonerId: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime,
            error: err instanceof Error ? err.message : String(err)
          }, {
            eventType: 'execution.failed',
            source: 'sdk.runtime',
            systemGenerated: true
          });
          this.executionLogger.error('Local agent call failed', {
            target,
            reasonerId: name,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            workflowId: execCtx.metadata.workflowId,
            rootWorkflowId: execCtx.metadata.rootWorkflowId,
            durationMs: Date.now() - startTime,
            error: err instanceof Error ? err.message : String(err)
          }, {
            eventType: 'agent.call.failed',
            source: 'sdk.runtime',
            systemGenerated: true
          });
          await emitEvent('failed', err);
          throw err;
        }
      });
    }

    const executionId = parentMetadata?.executionId ?? randomUUID();
    const runId = parentMetadata?.runId ?? parentMetadata?.executionId ?? executionId;
    const workflowId = parentMetadata?.workflowId ?? runId;
    const rootWorkflowId = parentMetadata?.rootWorkflowId ?? workflowId;
    this.executionLogger.system('agent.call.started', 'Remote agent call started', {
      target,
      agentNodeId: agentId,
      executionId,
      parentExecutionId: parentMetadata?.executionId,
      runId,
      workflowId,
      rootWorkflowId,
      reasonerId: name
    });

    try {
      const result = await this.agentFieldClient.execute(target, input, {
        runId,
        workflowId,
        rootWorkflowId,
        parentExecutionId: parentMetadata?.executionId,
        reasonerId: name,
        sessionId: parentMetadata?.sessionId,
        actorId: parentMetadata?.actorId,
        callerDid: parentMetadata?.callerDid,
        targetDid: parentMetadata?.targetDid,
        agentNodeDid: parentMetadata?.agentNodeDid,
        agentNodeId: this.config.nodeId
      });
      this.executionLogger.system('agent.call.completed', 'Remote agent call completed', {
        target,
        agentNodeId: agentId,
        executionId,
        parentExecutionId: parentMetadata?.executionId,
        runId,
        workflowId,
        rootWorkflowId,
        reasonerId: name
      });
      return result;
    } catch (err) {
      this.executionLogger.error('Remote agent call failed', {
        target,
        agentNodeId: agentId,
        executionId,
        parentExecutionId: parentMetadata?.executionId,
        runId,
        workflowId,
        rootWorkflowId,
        reasonerId: name,
        error: err instanceof Error ? err.message : String(err)
      }, {
        eventType: 'agent.call.failed',
        source: 'sdk.runtime',
        systemGenerated: true
      });
      throw err;
    }
  }

  private registerDefaultRoutes() {
    this.app.get('/health', (_req, res) => {
      res.json(this.health());
    });

    // Discovery endpoint used for serverless registration (mirrors Python behaviour)
    this.app.get('/discover', (_req, res) => {
      res.json(this.discoveryPayload(this.config.deploymentType ?? 'long_running'));
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

    // Local verification middleware for execution endpoints
    if (this.localVerifier) {
      const verifier = this.localVerifier;
      const realtimeFunctions = this.realtimeValidationFunctions;

      // Rate limiter for auth endpoints: max 30 attempts per identity per 60s window.
      // Uses X-Caller-DID when present so agents behind shared NAT/gateway don't
      // exhaust each other's quota. Falls back to IP when no DID is claimed.
      const authRateLimiter = rateLimit({
        windowMs: 60_000,
        max: 30,
        standardHeaders: true,
        legacyHeaders: false,
        keyGenerator: (req) => {
          const callerDID = req.headers['x-caller-did'];
          if (typeof callerDID === 'string' && callerDID.length > 0) {
            return callerDID;
          }
          return req.ip ?? 'unknown';
        },
        message: { error: 'rate_limit_exceeded', message: 'Too many authentication attempts. Try again later.' },
        skip: (req) => {
          const path = req.path;
          if (!path.startsWith('/reasoners/') && !path.startsWith('/skills/') &&
              !path.startsWith('/execute') && !path.startsWith('/api/v1/reasoners/') &&
              !path.startsWith('/api/v1/skills/')) {
            return true;
          }
          const parts = path.replace(/^\/+/, '').split('/');
          const funcName = parts[parts.length - 1] ?? '';
          return realtimeFunctions.has(funcName);
        },
      });
      this.app.use(authRateLimiter);

      this.app.use(async (req, res, next) => {
        const path = req.path;

        // Only verify execution endpoints
        if (!path.startsWith('/reasoners/') && !path.startsWith('/skills/') &&
            !path.startsWith('/execute') && !path.startsWith('/api/v1/reasoners/') &&
            !path.startsWith('/api/v1/skills/')) {
          return next();
        }

        // Extract function name
        const parts = path.replace(/^\/+/, '').split('/');
        const funcName = parts[parts.length - 1] ?? '';

        // Skip for realtime-validated functions
        if (realtimeFunctions.has(funcName)) {
          return next();
        }

        // Refresh cache if stale
        if (verifier.needsRefresh) {
          try {
            await verifier.refresh();
          } catch (err) {
            console.warn('[LocalVerifier] Cache refresh failed:', err);
          }
        }

        // Extract DID auth headers
        const callerDid = req.headers['x-caller-did'] as string | undefined;
        const signature = req.headers['x-did-signature'] as string | undefined;
        const timestamp = req.headers['x-did-timestamp'] as string | undefined;
        const nonce = req.headers['x-did-nonce'] as string | undefined;

        // C4: Require DID authentication — fail closed when callerDid is missing
        if (!callerDid) {
          return res.status(401).json({
            error: 'did_auth_required',
            message: 'DID authentication required',
          });
        }

        // Check revocation
        if (verifier.checkRevocation(callerDid)) {
          return res.status(403).json({
            error: 'did_revoked',
            message: `Caller DID ${callerDid} has been revoked`,
          });
        }

        // Check registration — reject DIDs not registered with the control plane
        if (!verifier.checkRegistration(callerDid)) {
          return res.status(403).json({
            error: 'did_not_registered',
            message: `Caller DID ${callerDid} is not registered with the control plane`,
          });
        }

        // C5: Require signature when callerDid is present
        if (!signature) {
          return res.status(401).json({
            error: 'signature_required',
            message: 'DID signature required',
          });
        }

        // Verify signature
        if (timestamp) {
          const body = Buffer.isBuffer(req.body) ? req.body : Buffer.from(JSON.stringify(req.body));
          const valid = await verifier.verifySignature(callerDid, signature, timestamp, body, nonce);
          if (!valid) {
            return res.status(401).json({
              error: 'signature_invalid',
              message: 'DID signature verification failed',
            });
          }
        } else {
          // Timestamp is required for signature verification
          return res.status(401).json({
            error: 'signature_invalid',
            message: 'DID signature verification failed: missing timestamp',
          });
        }

        // C6: Evaluate access policy after successful signature verification
        // Caller tags cannot be resolved at agent-side middleware level (would require
        // a control plane lookup). Pass empty array — policies that require specific
        // caller tags will not match, which is correct fail-open behavior for
        // agent-side verification. The control plane remains the primary policy
        // enforcement point with full caller context.
        const agentTags = this.config.tags ?? [];
        const allowed = verifier.evaluatePolicy(
          [],        // caller tags (not resolvable without control plane)
          agentTags, // target tags (this agent's own tags)
          funcName,
          typeof req.body === 'object' && req.body !== null ? req.body : {},
        );
        if (!allowed) {
          return res.status(403).json({
            error: 'policy_denied',
            message: 'Access denied by policy',
          });
        }

        next();
      });
    }

    this.app.post('/api/v1/reasoners/*', (req: Request<WildcardParams>, res: Response) => this.executeReasoner(req, res, req.params[0]));
    this.app.post('/reasoners/:name', (req, res) => this.executeReasoner(req, res, req.params.name));

    this.app.post('/api/v1/skills/*', (req: Request<WildcardParams>, res: Response) => this.executeSkill(req, res, req.params[0]));
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
        const body: Record<string, any> = { error: err?.message ?? 'Execution failed' };
        if (err?.responseData) body.error_details = err.responseData;
        // Propagate upstream HTTP status (e.g. 403 from permission middleware)
        const statusCode = (err?.status >= 400) ? err.status : 500;
        res.status(statusCode).json(body);
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
        const body: Record<string, any> = { error: err?.message ?? 'Execution failed' };
        if (err?.responseData) body.error_details = err.responseData;
        // Propagate upstream HTTP status (e.g. 403 from permission middleware)
        const statusCode = (err?.status >= 400) ? err.status : 500;
        res.status(statusCode).json(body);
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
        const body: Record<string, any> = { error: err?.message ?? 'Execution failed' };
        if (err?.responseData) body.error_details = err.responseData;
        // Propagate upstream HTTP status (e.g. 403 from permission middleware)
        const statusCode = (err?.status >= 400) ? err.status : 500;
        res.status(statusCode).json(body);
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
    const rootWorkflowId =
      overrides?.rootWorkflowId ?? normalized['x-root-workflow-id'] ?? workflowId;

    return {
      executionId,
      runId,
      workflowId,
      rootWorkflowId,
      sessionId: overrides?.sessionId ?? normalized['x-session-id'],
      actorId: overrides?.actorId ?? normalized['x-actor-id'],
      parentExecutionId: overrides?.parentExecutionId ?? normalized['x-parent-execution-id'],
      reasonerId: overrides?.reasonerId ?? normalized['x-reasoner-id'],
      callerDid: overrides?.callerDid ?? normalized['x-caller-did'],
      targetDid: overrides?.targetDid ?? normalized['x-target-did'],
      agentNodeDid:
        overrides?.agentNodeDid ?? normalized['x-agent-node-did'] ?? normalized['x-agent-did']
    };
  }

  private handleHttpRequest(req: http.IncomingMessage | express.Request, res: http.ServerResponse | express.Response) {
    const handler = this.app as unknown as (req: http.IncomingMessage, res: http.ServerResponse) => void;
    return handler(req as http.IncomingMessage, res as http.ServerResponse);
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
    interface ParsedBody {
      input?: unknown;
      [key: string]: unknown;
    }

    const parsed = this.parseBody(event?.body) as ParsedBody | null | undefined;

    if (
      parsed &&
      typeof parsed === 'object' &&
      event?.input !== undefined &&
      parsed.input === undefined
    ) {
      return { ...parsed, input: event.input };
    }
    if ((parsed === undefined || parsed === null) && event?.input !== undefined) {
      return { input: event.input };
    }
    return parsed;
  }

  private mergeExecutionContext(event: ServerlessEvent): Partial<ExecutionMetadata> {
    const rawCtx = event?.executionContext ?? event?.execution_context;
    return rawCtx ? normalizeExecutionContext(rawCtx) : {};
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
      interface ParsedBody {
        input?: any;
        data?: any;
        [key: string]: any;
      }

      const parsedBody = parsed as ParsedBody;
      if (parsedBody.input !== undefined) return parsedBody.input;
      if (parsedBody.data !== undefined) return parsedBody.data;
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
    return this.reasoners.all().map((r) => {
      const tags = r.options?.tags ?? [];
      return {
        id: r.name,
        input_schema: toJsonSchema(r.options?.inputSchema),
        output_schema: toJsonSchema(r.options?.outputSchema),
        memory_config: r.options?.memoryConfig ?? {
          auto_inject: [] as string[],
          memory_retention: '',
          cache_results: false
        },
        tags,
        proposed_tags: tags
      };
    });
  }

  private skillDefinitions() {
    return this.skills.all().map((s) => {
      const tags = s.options?.tags ?? [];
      return {
        id: s.name,
        input_schema: toJsonSchema(s.options?.inputSchema),
        tags,
        proposed_tags: tags
      };
    });
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
    const executionMetadata: ExecutionMetadata = {
      ...params.metadata,
      rootWorkflowId:
        params.metadata.rootWorkflowId ?? params.metadata.workflowId ?? params.metadata.runId ?? params.metadata.executionId,
      reasonerId: params.metadata.reasonerId ?? params.targetName
    };
    const execCtx = new ExecutionContext({
      input: params.input,
      metadata: executionMetadata,
      req,
      res,
      agent: this
    });

    return ExecutionContext.run(execCtx, async () => {
      this.executionLogger.system('execution.started', 'Execution started', {
        target: params.targetName,
        reasonerId: executionMetadata.reasonerId,
        executionId: executionMetadata.executionId,
        parentExecutionId: executionMetadata.parentExecutionId,
        runId: executionMetadata.runId,
        workflowId: executionMetadata.workflowId,
        rootWorkflowId: executionMetadata.rootWorkflowId
      });
      this.executionLogger.system('reasoner.started', 'Reasoner execution started', {
        target: params.targetName,
        executionId: executionMetadata.executionId,
        runId: executionMetadata.runId,
        workflowId: executionMetadata.workflowId,
        rootWorkflowId: executionMetadata.rootWorkflowId
      });
      try {
        const ctx = new ReasonerContext({
          input: params.input,
          executionId: executionMetadata.executionId,
          runId: executionMetadata.runId,
          sessionId: executionMetadata.sessionId,
          actorId: executionMetadata.actorId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          parentExecutionId: executionMetadata.parentExecutionId,
          reasonerId: executionMetadata.reasonerId,
          callerDid: executionMetadata.callerDid,
          targetDid: executionMetadata.targetDid,
          agentNodeDid: executionMetadata.agentNodeDid,
          req,
          res,
          agent: this,
          logger: this.executionLogger,
          aiClient: this.aiClient,
          memory: this.getMemoryInterface(executionMetadata),
          workflow: this.getWorkflowReporter(executionMetadata),
          did: this.getDidInterface(executionMetadata, params.input, params.targetName)
        });

        const result = await reasoner.handler(ctx);
        this.executionLogger.system('reasoner.completed', 'Reasoner execution completed', {
          target: params.targetName,
          executionId: executionMetadata.executionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId
        });
        this.executionLogger.system('execution.completed', 'Execution completed', {
          target: params.targetName,
          reasonerId: executionMetadata.reasonerId,
          executionId: executionMetadata.executionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId
        });
        if (params.respond && params.res) {
          params.res.json(result);
          return;
        }
        return result;
      } catch (err: any) {
        this.executionLogger.error('Reasoner execution failed', {
          target: params.targetName,
          executionId: executionMetadata.executionId,
          parentExecutionId: executionMetadata.parentExecutionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          error: err?.message ?? 'Execution failed'
        }, {
          eventType: 'reasoner.failed',
          source: 'sdk.runtime',
          systemGenerated: true
        });
        this.executionLogger.error('Execution failed', {
          target: params.targetName,
          reasonerId: executionMetadata.reasonerId,
          executionId: executionMetadata.executionId,
          parentExecutionId: executionMetadata.parentExecutionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          error: err?.message ?? 'Execution failed'
        }, {
          eventType: 'execution.failed',
          source: 'sdk.runtime',
          systemGenerated: true
        });
        if (params.respond && params.res) {
          const body: Record<string, any> = { error: err?.message ?? 'Execution failed' };
          if (err?.responseData) body.error_details = err.responseData;
          const statusCode = (err?.status >= 400)
            ? err.status
            : ((err?.statusCode >= 400) ? err.statusCode : 500);
          params.res.status(statusCode).json(body);
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
    const executionMetadata: ExecutionMetadata = {
      ...params.metadata,
      rootWorkflowId:
        params.metadata.rootWorkflowId ?? params.metadata.workflowId ?? params.metadata.runId ?? params.metadata.executionId,
      reasonerId: params.metadata.reasonerId ?? params.targetName
    };
    const execCtx = new ExecutionContext({
      input: params.input,
      metadata: executionMetadata,
      req,
      res,
      agent: this
    });

    return ExecutionContext.run(execCtx, async () => {
      this.executionLogger.system('execution.started', 'Execution started', {
        target: params.targetName,
        reasonerId: executionMetadata.reasonerId,
        executionId: executionMetadata.executionId,
        parentExecutionId: executionMetadata.parentExecutionId,
        runId: executionMetadata.runId,
        workflowId: executionMetadata.workflowId,
        rootWorkflowId: executionMetadata.rootWorkflowId
      });
      this.executionLogger.system('skill.started', 'Skill execution started', {
        target: params.targetName,
        executionId: executionMetadata.executionId,
        runId: executionMetadata.runId,
        workflowId: executionMetadata.workflowId,
        rootWorkflowId: executionMetadata.rootWorkflowId
      });
      try {
        const ctx = new SkillContext({
          input: params.input,
          executionId: executionMetadata.executionId,
          sessionId: executionMetadata.sessionId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          reasonerId: executionMetadata.reasonerId,
          req,
          res,
          agent: this,
          logger: this.executionLogger,
          memory: this.getMemoryInterface(executionMetadata),
          workflow: this.getWorkflowReporter(executionMetadata),
          did: this.getDidInterface(executionMetadata, params.input, params.targetName)
        });

        const result = await skill.handler(ctx);
        this.executionLogger.system('skill.completed', 'Skill execution completed', {
          target: params.targetName,
          executionId: executionMetadata.executionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId
        });
        this.executionLogger.system('execution.completed', 'Execution completed', {
          target: params.targetName,
          reasonerId: executionMetadata.reasonerId,
          executionId: executionMetadata.executionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId
        });
        if (params.respond && params.res) {
          params.res.json(result);
          return;
        }
        return result;
      } catch (err: any) {
        this.executionLogger.error('Skill execution failed', {
          target: params.targetName,
          executionId: executionMetadata.executionId,
          parentExecutionId: executionMetadata.parentExecutionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          error: err?.message ?? 'Execution failed'
        }, {
          eventType: 'skill.failed',
          source: 'sdk.runtime',
          systemGenerated: true
        });
        this.executionLogger.error('Execution failed', {
          target: params.targetName,
          reasonerId: executionMetadata.reasonerId,
          executionId: executionMetadata.executionId,
          parentExecutionId: executionMetadata.parentExecutionId,
          runId: executionMetadata.runId,
          workflowId: executionMetadata.workflowId,
          rootWorkflowId: executionMetadata.rootWorkflowId,
          error: err?.message ?? 'Execution failed'
        }, {
          eventType: 'execution.failed',
          source: 'sdk.runtime',
          systemGenerated: true
        });
        if (params.respond && params.res) {
          const body: Record<string, any> = { error: err?.message ?? 'Execution failed' };
          if (err?.responseData) body.error_details = err.responseData;
          const statusCode = (err?.status >= 400)
            ? err.status
            : ((err?.statusCode >= 400) ? err.statusCode : 500);
          params.res.status(statusCode).json(body);
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

      const agentTags = this.config.tags ?? [];
      const regResponse = await this.agentFieldClient.register({
        id: this.config.nodeId,
        version: this.config.version ?? '',
        base_url: publicUrl,
        public_url: publicUrl,
        deployment_type: this.config.deploymentType ?? 'long_running',
        reasoners,
        skills,
        proposed_tags: agentTags,
        tags: agentTags
      });

      // Handle pending approval state: poll until approved
      if (regResponse?.status === 'pending_approval') {
        const pendingTags = regResponse.pending_tags ?? [];
        console.log(`[AgentField] Node ${this.config.nodeId} registered but awaiting tag approval (pending tags: ${pendingTags.join(', ')})`);
        await this.waitForApproval();
        console.log(`[AgentField] Node ${this.config.nodeId} tag approval granted`);
      }

      // Register with DID system if enabled
      if (this.config.didEnabled) {
        try {
          const didRegistered = await this.didManager.registerAgent(reasoners, skills);
          if (didRegistered) {
            const summary = this.didManager.getIdentitySummary();
            console.log(`[DID] Agent registered with DID: ${summary.agentDid}`);
            console.log(`[DID] Reasoner DIDs: ${summary.reasonerCount}, Skill DIDs: ${summary.skillCount}`);

            // Wire DID credentials to the HTTP client for request signing
            const pkg = this.didManager.getIdentityPackage();
            if (pkg?.agentDid?.did && pkg?.agentDid?.privateKeyJwk) {
              this.agentFieldClient.setDIDCredentials(pkg.agentDid.did, pkg.agentDid.privateKeyJwk);
            }
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

  private async waitForApproval(): Promise<void> {
    const pollInterval = 5000; // 5 seconds
    const timeoutMs = 5 * 60 * 1000; // 5 minutes
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      try {
        const node = await this.agentFieldClient.getNode(this.config.nodeId);
        const status = node?.lifecycle_status;
        if (status && status !== 'pending_approval') {
          return;
        }
        console.log(`[AgentField] Node ${this.config.nodeId} still pending approval...`);
      } catch (err) {
        console.warn('[AgentField] Polling for approval status failed:', err);
      }
    }

    throw new Error(
      `[AgentField] Node ${this.config.nodeId} approval timed out after ${timeoutMs / 1000}s`
    );
  }

  private startHeartbeat() {
    const interval = this.config.heartbeatIntervalMs ?? 30_000;
    if (interval <= 0) return;

    const tick = async () => {
      try {
        await this.agentFieldClient.heartbeat('ready');
      } catch (err) {
        console.warn('Heartbeat failed', err);
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
