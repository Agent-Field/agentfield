import type http from 'node:http';
import type { ReasonerDefinition } from './reasoner.js';
import type { SkillDefinition } from './skill.js';
import type { MemoryChangeEvent, MemoryWatchHandler } from '../memory/MemoryInterface.js';
import type { ExecutionMetadata } from '../context/ExecutionContext.js';
import type { HarnessConfig } from '../harness/types.js';

export type DeploymentType = 'long_running' | 'serverless';

export interface AgentConfig {
  nodeId: string;
  version?: string;
  teamId?: string;
  agentFieldUrl?: string;
  port?: number;
  host?: string;
  publicUrl?: string;
  aiConfig?: AIConfig;
  harnessConfig?: HarnessConfig;
  memoryConfig?: MemoryConfig;
  didEnabled?: boolean;
  devMode?: boolean;
  heartbeatIntervalMs?: number;
  defaultHeaders?: Record<string, string | number | boolean | undefined>;
  apiKey?: string;
  did?: string;
  privateKeyJwk?: string;
  deploymentType?: DeploymentType;
  /** Enable decentralized local verification of incoming DID signatures. */
  localVerification?: boolean;
  /** Cache refresh interval for local verification in seconds (default: 300). */
  verificationRefreshInterval?: number;
  /** Agent-level tags for tag-based authorization policies. */
  tags?: string[];
}

export interface AIConfig {
  provider?:
    | 'openai'
    | 'anthropic'
    | 'google'
    | 'mistral'
    | 'groq'
    | 'xai'
    | 'deepseek'
    | 'cohere'
    | 'openrouter'
    | 'ollama';
  model?: string;
  embeddingModel?: string;
  apiKey?: string;
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
  enableRateLimitRetry?: boolean;
  rateLimitMaxRetries?: number;
  rateLimitBaseDelay?: number;
  rateLimitMaxDelay?: number;
  rateLimitJitterFactor?: number;
  rateLimitCircuitBreakerThreshold?: number;
  rateLimitCircuitBreakerTimeout?: number;
}

export interface MemoryConfig {
  defaultScope?: MemoryScope;
  ttl?: number;
}

export type MemoryScope = 'workflow' | 'session' | 'actor' | 'global';

export interface AgentCapability {
  agentId: string;
  baseUrl: string;
  version: string;
  healthStatus: string;
  deploymentType?: string;
  lastHeartbeat?: string;
  reasoners: ReasonerCapability[];
  skills: SkillCapability[];
}

export interface ReasonerCapability {
  id: string;
  description?: string;
  tags: string[];
  inputSchema?: any;
  outputSchema?: any;
  examples?: any[];
  invocationTarget: string;
}

export interface SkillCapability {
  id: string;
  description?: string;
  tags: string[];
  inputSchema?: any;
  invocationTarget: string;
}

export interface DiscoveryResponse {
  discoveredAt: string;
  totalAgents: number;
  totalReasoners: number;
  totalSkills: number;
  pagination: DiscoveryPagination;
  capabilities: AgentCapability[];
}

export interface DiscoveryPagination {
  limit: number;
  offset: number;
  hasMore: boolean;
}

export interface CompactCapability {
  id: string;
  agentId: string;
  target: string;
  tags: string[];
}

export interface CompactDiscoveryResponse {
  discoveredAt: string;
  reasoners: CompactCapability[];
  skills: CompactCapability[];
}

export type DiscoveryFormat = 'json' | 'compact' | 'xml';

export interface DiscoveryResult {
  format: DiscoveryFormat;
  raw: string;
  json?: DiscoveryResponse;
  compact?: CompactDiscoveryResponse;
  xml?: string;
}

export interface DiscoveryOptions {
  agent?: string;
  nodeId?: string;
  agentIds?: string[];
  nodeIds?: string[];
  reasoner?: string;
  skill?: string;
  tags?: string[];
  includeInputSchema?: boolean;
  includeOutputSchema?: boolean;
  includeDescriptions?: boolean;
  includeExamples?: boolean;
  format?: DiscoveryFormat;
  healthStatus?: string;
  limit?: number;
  offset?: number;
  headers?: Record<string, string>;
}

export interface AgentState {
  reasoners: Map<string, ReasonerDefinition>;
  skills: Map<string, SkillDefinition>;
  memoryWatchers: Array<{ pattern: string; handler: MemoryWatchHandler; scope?: string; scopeId?: string }>;
}

// Health status returned by the agent `/status` endpoint.
export interface HealthStatus {
  status: 'ok' | 'running';
  node_id: string;
  version?: string;
}

export interface ServerlessEvent {
  path?: string;
  rawPath?: string;
  httpMethod?: string;
  method?: string;
  action?: string;
  headers?: Record<string, string | undefined>;
  queryStringParameters?: Record<string, string | undefined>;
  target?: string;
  reasoner?: string;
  skill?: string;
  type?: 'reasoner' | 'skill';
  body?: any;
  input?: any;
  executionContext?: RawExecutionContext;
  execution_context?: RawExecutionContext;
}

export interface ServerlessResponse {
  statusCode: number;
  headers?: Record<string, string>;
  body: any;
}

export type ServerlessAdapter = (event: any, context?: any) => ServerlessEvent;

export type AgentHandler = (
  event: ServerlessEvent | http.IncomingMessage,
  res?: http.ServerResponse
) => Promise<ServerlessResponse | void> | ServerlessResponse | void;

export type Awaitable<T> = T | Promise<T>;

export interface RawExecutionContext {
  executionId?: string;
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

  // snake_case variants
  execution_id?: string;
  run_id?: string;
  workflow_id?: string;
  root_workflow_id?: string;
  parent_execution_id?: string;
  reasoner_id?: string;
  session_id?: string;
  actor_id?: string;
  caller_did?: string;
  target_did?: string;
  agent_node_did?: string;
}