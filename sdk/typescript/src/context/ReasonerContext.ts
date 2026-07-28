import type express from 'express';
import { ExecutionContext } from './ExecutionContext.js';
import type { AIClient, AIRequestOptions, AIStream, ZodSchema } from '../ai/AIClient.js';
import type { MemoryInterface } from '../memory/MemoryInterface.js';
import type { Agent } from '../agent/Agent.js';
import type { WorkflowReporter } from '../workflow/WorkflowReporter.js';
import type { DiscoveryOptions } from '../types/agent.js';
import type { DidInterface } from '../did/DidInterface.js';
import type { AIToolRequestOptions, ToolCallTrace } from '../ai/ToolCalling.js';
import { buildToolConfig, executeToolCallLoop } from '../ai/ToolCalling.js';
import type { ExecutionLogger } from '../observability/ExecutionLogger.js';
import { CostTracker } from '../usage/costTracker.js';
import type { TriggerContext } from '../triggers/types.js';

export class ReasonerContext<TInput = any> {
  readonly input: TInput;
  readonly executionId: string;
  readonly runId?: string;
  readonly sessionId?: string;
  readonly actorId?: string;
  readonly workflowId?: string;
  readonly rootWorkflowId?: string;
  readonly parentExecutionId?: string;
  readonly reasonerId?: string;
  readonly callerDid?: string;
  readonly targetDid?: string;
  readonly agentNodeDid?: string;
  readonly req: express.Request;
  readonly res: express.Response;
  readonly agent: Agent;
  readonly logger: ExecutionLogger;
  readonly aiClient: AIClient;
  readonly memory: MemoryInterface;
  readonly workflow: WorkflowReporter;
  readonly did: DidInterface;
  /**
   * Per-execution token/cost usage accumulator. LLM calls made through
   * `ctx.ai()` / `ctx.aiWithTools()` and harness runs record into it
   * automatically; reasoner authors may also `record()` custom entries. Its
   * serialized summary is attached to the execution's terminal report.
   */
  readonly costTracker: CostTracker;
  /**
   * AbortSignal that fires when the control plane cancels this execution
   * (per-execution cancel, the bottom-up cancel-tree endpoint, or any
   * future source that flips the bus). Pass it through to `fetch`, the
   * @anthropic-ai/sdk, the openai SDK, or anywhere that accepts
   * `{ signal }` to short-circuit in-flight work mid-call. For pure-JS
   * CPU loops, check `ctx.signal.aborted` periodically and throw.
   */
  readonly signal: AbortSignal;
  /**
   * Trigger context populated when the reasoner was invoked by an inbound
   * webhook event or cron schedule. `undefined` for direct calls via
   * `app.call(...)` or HTTP POST without a dispatcher envelope.
   */
  readonly trigger?: TriggerContext;

  constructor(params: {
    input: TInput;
    executionId: string;
    runId?: string;
    sessionId?: string;
    actorId?: string;
    workflowId?: string;
    rootWorkflowId?: string;
    parentExecutionId?: string;
    reasonerId?: string;
    callerDid?: string;
    targetDid?: string;
    agentNodeDid?: string;
    req: express.Request;
    res: express.Response;
    agent: Agent;
    logger: ExecutionLogger;
    aiClient: AIClient;
    memory: MemoryInterface;
    workflow: WorkflowReporter;
    did: DidInterface;
    signal?: AbortSignal;
    costTracker?: CostTracker;
    trigger?: TriggerContext;
  }) {
    this.input = params.input;
    this.executionId = params.executionId;
    this.runId = params.runId;
    this.sessionId = params.sessionId;
    this.actorId = params.actorId;
    this.workflowId = params.workflowId;
    this.rootWorkflowId = params.rootWorkflowId;
    this.parentExecutionId = params.parentExecutionId;
    this.reasonerId = params.reasonerId;
    this.callerDid = params.callerDid;
    this.targetDid = params.targetDid;
    this.agentNodeDid = params.agentNodeDid;
    this.req = params.req;
    this.res = params.res;
    this.agent = params.agent;
    this.logger = params.logger;
    this.aiClient = params.aiClient;
    this.memory = params.memory;
    this.workflow = params.workflow;
    this.did = params.did;
    // Default to a never-aborted signal when none provided so existing
    // call sites (tests, manual invocations) continue to work.
    this.signal = params.signal ?? new AbortController().signal;
    // Fall back to the ambient execution's tracker (or a standalone one) so
    // manually constructed contexts keep working.
    this.costTracker =
      params.costTracker ?? ExecutionContext.getCurrent()?.costTracker ?? new CostTracker();
    this.trigger = params.trigger;
  }

  ai<T>(prompt: string, options: AIRequestOptions & { schema: ZodSchema<T> }): Promise<T>;
  ai(prompt: string, options?: AIToolRequestOptions): Promise<string>;
  ai(prompt: string, options?: AIToolRequestOptions): Promise<unknown> {
    if (options?.tools) {
      return this.aiWithTools(prompt, options);
    }
    return this.aiClient.generate(prompt, options);
  }

  /**
   * AI call with automatic tool calling via discover -> ai -> call loop.
   *
   * Discovers available capabilities, presents them as tools to the LLM,
   * dispatches tool calls via agent.call(), and iterates until a final response.
   *
   * @returns Object with `text` (final response) and `trace` (observability data).
   */
  async aiWithTools(
    prompt: string,
    options: AIToolRequestOptions = {}
  ): Promise<{ text: string; trace: ToolCallTrace }> {
    const toolsParam = options.tools ?? 'discover';
    const { tools, config, needsLazyHydration } = await buildToolConfig(toolsParam, this.agent);

    const mergedConfig = {
      ...config,
      maxTurns: options.maxTurns ?? config.maxTurns ?? 10,
      maxToolCalls: options.maxToolCalls ?? config.maxToolCalls ?? 25
    };

    // Resolve the provider/model pair once so the tool loop can attribute
    // token/cost usage to the actual model it calls. Optional so partial
    // AIClient doubles (tests, custom clients) keep working.
    const modelChoice =
      typeof this.aiClient.resolveModelChoice === 'function'
        ? this.aiClient.resolveModelChoice(options)
        : undefined;

    return executeToolCallLoop(
      this.agent,
      prompt,
      tools,
      mergedConfig,
      needsLazyHydration,
      () => this.aiClient.getModel(options),
      options,
      modelChoice
    );
  }

  aiStream(prompt: string, options?: AIRequestOptions): Promise<AIStream> {
    return this.aiClient.stream(prompt, options);
  }

  call(target: string, input: any) {
    return this.agent.call(target, input);
  }

  /**
   * Pause this execution for external approval / resumption.
   *
   * Transitions the execution to `waiting` on the control plane and blocks
   * until a decision arrives via the agent's approval webhook, or the timeout
   * elapses (returning `{ decision: 'expired' }`). The caller creates the
   * approval request on an external service first and passes its
   * `approvalRequestId`. Delegates to {@link Agent.pause}. See its docs for the
   * async-execution requirement that lets a pause outlive the dispatch ceiling.
   */
  pause(opts: {
    approvalRequestId: string;
    approvalRequestUrl?: string;
    expiresInHours?: number;
    timeoutMs?: number;
  }): Promise<import('../agent/pause.js').ApprovalResult> {
    return this.agent.pause({ ...opts, executionId: this.executionId });
  }

  discover(options?: DiscoveryOptions) {
    return this.agent.discover(options);
  }

  note(message: string, tags: string[] = []): void {
    this.agent.note(message, tags, {
      executionId: this.executionId,
      runId: this.runId,
      sessionId: this.sessionId,
      actorId: this.actorId,
      workflowId: this.workflowId,
      rootWorkflowId: this.rootWorkflowId,
      parentExecutionId: this.parentExecutionId,
      reasonerId: this.reasonerId,
      callerDid: this.callerDid,
      targetDid: this.targetDid,
      agentNodeDid: this.agentNodeDid
    });
  }
}

export function getCurrentContext<TInput = any>(): ReasonerContext<TInput> | undefined {
  const execution = ExecutionContext.getCurrent();
  if (!execution) return undefined;
  const { metadata, input, agent, req, res } = execution;
  return new ReasonerContext<TInput>({
    input,
    executionId: metadata.executionId,
    runId: metadata.runId,
    sessionId: metadata.sessionId,
    actorId: metadata.actorId,
    workflowId: metadata.workflowId,
    rootWorkflowId: metadata.rootWorkflowId,
    parentExecutionId: metadata.parentExecutionId,
    reasonerId: metadata.reasonerId,
    callerDid: metadata.callerDid,
    targetDid: metadata.targetDid,
    agentNodeDid: metadata.agentNodeDid,
    req,
    res,
    agent,
    logger: agent.getExecutionLogger(),
    aiClient: agent.getAIClient(),
    memory: agent.getMemoryInterface(metadata),
    workflow: agent.getWorkflowReporter(metadata),
    did: agent.getDidInterface(metadata, input),
    costTracker: execution.costTracker
  });
}
