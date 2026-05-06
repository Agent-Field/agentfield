/**
 * Tool calling support for AgentField agents.
 *
 * Converts discovered capabilities into Vercel AI SDK tool definitions and provides
 * an automatic tool-call execution loop that dispatches calls via agent.call().
 */

import { generateText, tool, jsonSchema, stepCountIs } from 'ai';
import type { ToolSet } from 'ai';
import type {
  AgentCapability,
  ReasonerCapability,
  SkillCapability,
  DiscoveryOptions,
  DiscoveryResult
} from '../types/agent.js';
import type { Agent } from '../agent/Agent.js';
import type { AIRequestOptions } from './AIClient.js';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface ToolCallConfig {
  /** Maximum number of LLM turns in the tool-call loop (default: 10). */
  maxTurns?: number;
  /** Maximum total tool calls allowed (default: 25). */
  maxToolCalls?: number;
  /** Maximum candidate tools to present to the LLM. */
  maxCandidateTools?: number;
  /** Maximum tools to hydrate with full schemas (lazy mode). */
  maxHydratedTools?: number;
  /** Schema hydration mode: "eager" includes full schemas, "lazy" sends metadata first. */
  schemaHydration?: 'eager' | 'lazy';
  /** Whether to broaden discovery if no tools match (default: false). */
  fallbackBroadening?: boolean;
  /** Filter by tags during discovery. */
  tags?: string[];
  /** Filter by agent IDs during discovery. */
  agentIds?: string[];
  /** Filter by health status (default: "healthy"). */
  healthStatus?: string;
}

const DEFAULT_MAX_TURNS = 10;
const DEFAULT_MAX_TOOL_CALLS = 25;
const DEFAULT_HEALTH_STATUS = undefined;

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

export interface ToolCallRecord {
  toolName: string;
  arguments: Record<string, any>;
  result?: any;
  error?: string;
  latencyMs: number;
  turn: number;
}

export interface ToolCallTrace {
  calls: ToolCallRecord[];
  totalTurns: number;
  totalToolCalls: number;
  finalResponse?: string;
}

// ---------------------------------------------------------------------------
// Tool options for ctx.ai()
// ---------------------------------------------------------------------------

/**
 * Options for tool-calling in ctx.ai().
 *
 * Accepts multiple forms:
 * - "discover": auto-discover all tools from control plane
 * - ToolCallConfig: discover with filtering/progressive options
 * - DiscoveryResult: use pre-fetched discovery results
 * - AgentCapability[]: convert capability list directly
 * - ToolSet: raw Vercel AI SDK tool map
 */
export type ToolsOption =
  | 'discover'
  | ToolCallConfig
  | DiscoveryResult
  | AgentCapability[]
  | (ReasonerCapability | SkillCapability)[]
  | ToolSet;

export interface AIToolRequestOptions extends AIRequestOptions {
  /** Tool definitions for LLM tool calling. */
  tools?: ToolsOption;
  /** Maximum LLM turns in the tool-call loop. */
  maxTurns?: number;
  /** Maximum total tool calls allowed. */
  maxToolCalls?: number;
}

type ToolConfig = {
  description:  string
  inputSchema:  any;
  execute?: (args: Record<string, unknown>) => Promise<any>;
}

// ---------------------------------------------------------------------------
// Capability -> Tool Definition Conversion
// ---------------------------------------------------------------------------

/**
 * Convert discovery invocation_target format to agent.call() target format.
 *
 * Discovery returns colon-separated targets:
 *   - "node_id:skill:function_name" for skills
 *   - "node_id:function_name" for reasoners
 *
 * agent.call() expects dot-separated: "node_id.function_name"
 */
function invocationTargetToCallTarget(invocationTarget: string): string {
  if (invocationTarget.includes(':skill:')) {
    const parts = invocationTarget.split(':skill:');
    return `${parts[0]}.${parts[1]}`;
  }
  if (invocationTarget.includes(':')) {
    const idx = invocationTarget.indexOf(':');
    return `${invocationTarget.substring(0, idx)}.${invocationTarget.substring(idx + 1)}`;
  }
  return invocationTarget;
}

function makeExecute(agent: Agent, invocationTarget: string) {
  const callTarget = invocationTargetToCallTarget(invocationTarget);
  return async (args: any) => agent.call(callTarget, args);
}

/**
 * Convert an invocation_target to an LLM-safe function name.
 * Many providers (e.g., Google) only allow alphanumeric, underscores, dashes, and dots.
 * Colons are replaced with double-underscores for a reversible mapping.
 */
function sanitizeToolName(invocationTarget: string): string {
  return invocationTarget.replace(/:/g, '__');
}

function unsanitizeToolName(sanitizedName: string): string {
  return sanitizedName.replace(/__/g, ':');
}

/**
 * Convert a single ReasonerCapability or SkillCapability to a Vercel AI SDK tool.
 */
export function capabilityToTool(
  cap: ReasonerCapability | SkillCapability,
  agent: Agent
): ToolSet[string] {
  const rawSchema = cap.inputSchema ?? { type: 'object', properties: {} };
  const schema = rawSchema.type ? rawSchema : { type: 'object', properties: rawSchema };

  return tool({
    description: cap.description ?? `Call ${cap.invocationTarget}`,
    inputSchema: jsonSchema(schema),
    execute: makeExecute(agent, cap.invocationTarget)
  });
}

/**
 * Convert a single capability to a metadata-only tool (no full schema).
 * Used for progressive/lazy discovery.
 */
export function capabilityToMetadataTool(
  cap: ReasonerCapability | SkillCapability,
  agent: Agent
): ToolSet[string] {
  return tool({
    description: cap.description ?? `Call ${cap.invocationTarget}`,
    inputSchema: jsonSchema({ type: 'object', properties: {} }),
    execute: makeExecute(agent, cap.invocationTarget)
  });
}

/**
 * Convert a list of capabilities into a Vercel AI SDK tool map.
 */
export function capabilitiesToTools(
  capabilities: (AgentCapability | ReasonerCapability | SkillCapability)[],
  agent: Agent,
  metadataOnly = false
): ToolSet {
  const tools: ToolSet = {};
  const convert = metadataOnly ? capabilityToMetadataTool : capabilityToTool;

  for (const cap of capabilities) {
    if ('reasoners' in cap && 'skills' in cap) {
      const agentCap = cap as AgentCapability;
      for (const r of agentCap.reasoners) {
        tools[sanitizeToolName(r.invocationTarget)] = convert(r, agent);
      }
      for (const s of agentCap.skills) {
        tools[sanitizeToolName(s.invocationTarget)] = convert(s, agent);
      }
    } else {
      const c = cap as ReasonerCapability | SkillCapability;
      tools[sanitizeToolName(c.invocationTarget)] = convert(c, agent);
    }
  }

  return tools;
}

// ---------------------------------------------------------------------------
// Discovery helpers
// ---------------------------------------------------------------------------

function limitToolSet(tools: ToolSet, max: number): ToolSet {
  const entries = Object.entries(tools);
  if (entries.length <= max) return tools;
  const limited: ToolSet = {};
  for (let i = 0; i < max; i++) {
    limited[entries[i][0]] = entries[i][1];
  }
  return limited;
}

async function discoverTools(
  agent: Agent,
  config: ToolCallConfig,
  hydrateSchemas = true
): Promise<{ tools: ToolSet; capabilities: AgentCapability[] }> {
  const discoveryOpts: DiscoveryOptions = {
    tags: config.tags,
    agentIds: config.agentIds,
    includeInputSchema: hydrateSchemas,
    includeOutputSchema: false,
    includeDescriptions: true,
    healthStatus: config.healthStatus ?? DEFAULT_HEALTH_STATUS
  };

  const result = await agent.discover(discoveryOpts);
  if (!result.json) return { tools: {}, capabilities: [] };

  const caps = result.json.capabilities;
  let tools = capabilitiesToTools(caps, agent, !hydrateSchemas);

  if (config.maxCandidateTools) {
    tools = limitToolSet(tools, config.maxCandidateTools);
  }

  return { tools, capabilities: caps };
}

async function hydrateSelectedTools(
  agent: Agent,
  config: ToolCallConfig,
  selectedNames: string[]
): Promise<ToolSet> {
  const discoveryOpts: DiscoveryOptions = {
    tags: config.tags,
    agentIds: config.agentIds,
    includeInputSchema: true,
    includeOutputSchema: false,
    includeDescriptions: true,
    healthStatus: config.healthStatus ?? DEFAULT_HEALTH_STATUS
  };

  const result = await agent.discover(discoveryOpts);
  if (!result.json) return {};

  // selectedNames are sanitized (from LLM), unsanitize for matching
  const selectedSet = new Set(selectedNames.map(unsanitizeToolName));
  const tools: ToolSet = {};

  for (const cap of result.json.capabilities) {
    for (const r of cap.reasoners) {
      if (selectedSet.has(r.invocationTarget)) {
        tools[sanitizeToolName(r.invocationTarget)] = capabilityToTool(r, agent);
      }
    }
    for (const s of cap.skills) {
      if (selectedSet.has(s.invocationTarget)) {
        tools[sanitizeToolName(s.invocationTarget)] = capabilityToTool(s, agent);
      }
    }
  }

  if (config.maxHydratedTools) {
    return limitToolSet(tools, config.maxHydratedTools);
  }

  return tools;
}

// ---------------------------------------------------------------------------
// Build tool config from various input forms
// ---------------------------------------------------------------------------

function isToolCallConfig(obj: any): obj is ToolCallConfig {
  const keys = ['maxTurns', 'maxToolCalls', 'tags', 'schemaHydration', 'agentIds',
                'healthStatus', 'fallbackBroadening', 'maxCandidateTools', 'maxHydratedTools'];
  return typeof obj === 'object' && !Array.isArray(obj) && keys.some(k => k in obj);
}

function isDiscoveryResult(obj: any): obj is DiscoveryResult {
  return typeof obj === 'object' && !Array.isArray(obj) && 'raw' in obj && 'format' in obj;
}

export async function buildToolConfig(
  toolsParam: ToolsOption,
  agent: Agent
): Promise<{
  tools: ToolSet;
  config: ToolCallConfig;
  needsLazyHydration: boolean;
}> {
  const baseConfig: ToolCallConfig = {
    maxTurns: DEFAULT_MAX_TURNS,
    maxToolCalls: DEFAULT_MAX_TOOL_CALLS,
    schemaHydration: 'eager',
    fallbackBroadening: false,
    healthStatus: DEFAULT_HEALTH_STATUS
  };

  // "discover" - simple auto-discovery
  if (toolsParam === 'discover') {
    const { tools } = await discoverTools(agent, baseConfig);
    return { tools, config: baseConfig, needsLazyHydration: false };
  }

  // ToolCallConfig object
  if (isToolCallConfig(toolsParam)) {
    const config = { ...baseConfig, ...toolsParam };
    const isLazy = config.schemaHydration === 'lazy';
    const { tools } = await discoverTools(agent, config, !isLazy);
    return { tools, config, needsLazyHydration: isLazy };
  }

  // DiscoveryResult
  if (isDiscoveryResult(toolsParam)) {
    if (toolsParam.json) {
      const tools = capabilitiesToTools(toolsParam.json.capabilities, agent);
      return { tools, config: baseConfig, needsLazyHydration: false };
    }
    return { tools: {}, config: baseConfig, needsLazyHydration: false };
  }

  // Array of capabilities
  if (Array.isArray(toolsParam)) {
    const tools = capabilitiesToTools(toolsParam, agent);
    return { tools, config: baseConfig, needsLazyHydration: false };
  }

  // Raw ToolSet map - pass through directly
  if (typeof toolsParam === 'object') {
    return { tools: toolsParam as ToolSet, config: baseConfig, needsLazyHydration: false };
  }

  throw new Error(
    `Invalid tools parameter: expected "discover", ToolCallConfig, DiscoveryResult, ` +
    `capability array, or tool map, got ${typeof toolsParam}`
  );
}

// ---------------------------------------------------------------------------
// Tool-call execution loop
// ---------------------------------------------------------------------------

function wrapToolsWithObservability(
  toolMap: ToolSet,
  agent: Agent,
  trace: ToolCallTrace,
  maxToolCalls: number,
  getCurrentTurn: () => number
): { tools: ToolSet; getTotalCalls: () => number } {
  let totalCalls = 0;
  const observableTools: ToolSet = {};

  for (const [name, t] of Object.entries(toolMap)) {
    const originalTool = t as ToolConfig;
    observableTools[name] = tool({
      description: originalTool.description ?? '',
      inputSchema: originalTool.inputSchema,
      execute: async (args: any) => {
        totalCalls++;
        trace.totalToolCalls = totalCalls;

        if (totalCalls > maxToolCalls) {
          const record: ToolCallRecord = {
            toolName: name,
            arguments: args,
            error: 'Tool call limit reached',
            latencyMs: 0,
            turn: getCurrentTurn()
          };
          trace.calls.push(record);
          return { error: 'Tool call limit reached. Please provide a final response.' };
        }

        const record: ToolCallRecord = {
          toolName: name,
          arguments: args,
          latencyMs: 0,
          turn: getCurrentTurn()
        };

        const invocationTarget = unsanitizeToolName(name);
        const callTarget = invocationTargetToCallTarget(invocationTarget);
        const start = Date.now();
        try {
          const result = await agent.call(callTarget, args);
          record.result = result;
          record.latencyMs = Date.now() - start;
          trace.calls.push(record);
          return result;
        } catch (err: any) {
          record.error = err.message ?? String(err);
          record.latencyMs = Date.now() - start;
          trace.calls.push(record);
          return { error: record.error, tool: name };
        }
      }
    });
  }

  return { tools: observableTools, getTotalCalls: () => totalCalls };
}

export async function executeToolCallLoop(
  agent: Agent,
  prompt: string,
  toolMap: ToolSet,
  config: ToolCallConfig,
  needsLazyHydration: boolean,
  buildModel: () => any,
  options: AIRequestOptions = {}
): Promise<{ text: string; trace: ToolCallTrace }> {
  const maxTurns = config.maxTurns ?? DEFAULT_MAX_TURNS;
  const maxToolCalls = config.maxToolCalls ?? DEFAULT_MAX_TOOL_CALLS;

  const trace: ToolCallTrace = {
    calls: [],
    totalTurns: 0,
    totalToolCalls: 0
  };

  let activeTools = toolMap;

  // Lazy hydration: first do a selection pass with non-executable tools to see
  // which tools the LLM picks, then hydrate full schemas for those tools.
  if (needsLazyHydration) {
    // Create non-executable tool stubs so the LLM selects but doesn't execute
    const selectionTools: ToolSet = {};
    for (const [name, t] of Object.entries(toolMap)) {
      const orig = t as ToolConfig;
      selectionTools[name] = tool({
        description: orig.description ?? '',
        inputSchema: orig.inputSchema,
        // No execute — AI SDK will stop after LLM selects tools
      });
    }

    const selectionResult = await generateText({
      model: buildModel(),
      prompt,
      system: options.system,
      temperature: options.temperature,
      maxOutputTokens: options.maxTokens,
      tools: selectionTools,
      stopWhen: stepCountIs(1)  // Stop after LLM's first response (tool selection)
    });

    // Extract which tools the LLM tried to call
    const selectedNames = new Set<string>();
    for (const step of selectionResult.steps) {
      for (const tc of step.toolCalls) {
        selectedNames.add(tc.toolName);
      }
    }

    if (selectedNames.size > 0) {
      // Hydrate full schemas for selected tools
      const hydratedTools = await hydrateSelectedTools(
        agent, config, Array.from(selectedNames)
      );
      if (Object.keys(hydratedTools).length > 0) {
        activeTools = hydratedTools;
      }
    } else {
      // LLM didn't select any tools in the selection pass — just return text
      trace.totalTurns = selectionResult.steps.length;
      trace.finalResponse = selectionResult.text;
      return { text: selectionResult.text, trace };
    }
  }

  let currentTurn = 0;
  const { tools: observableTools } = wrapToolsWithObservability(
    activeTools, agent, trace, maxToolCalls, () => currentTurn
  );

  const model = buildModel();
  const result = await generateText({
    model,
    prompt,
    system: options.system,
    temperature: options.temperature,
    maxOutputTokens: options.maxTokens,
    tools: observableTools,
    stopWhen: stepCountIs(maxTurns),
    onStepFinish: () => {
      currentTurn++;
      trace.totalTurns = currentTurn;
    }
  });

  trace.finalResponse = result.text;
  trace.totalTurns = result.steps.length;

  return { text: result.text, trace };
}
