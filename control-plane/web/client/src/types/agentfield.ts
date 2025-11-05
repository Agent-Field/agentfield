export interface AgentNode {
  id: string;
  base_url: string;
  version: string;
  team_id?: string;
  health_status: HealthStatus;
  lifecycle_status?: LifecycleStatus;
  last_heartbeat?: string;
  registered_at?: string;
  deployment_type?: string; // "long_running" or "serverless"
  invocation_url?: string; // For serverless agents
  mcp_summary?: MCPSummaryForUI;
  mcp_servers?: MCPServerHealthForUI[];
  reasoners?: ReasonerDefinition[];
  skills?: SkillDefinition[];
}

export interface AgentNodeSummary {
  id: string;
  base_url: string;
  version: string;
  team_id: string;
  health_status: HealthStatus;
  lifecycle_status: LifecycleStatus;
  last_heartbeat?: string;
  deployment_type?: string; // "long_running" or "serverless"
  invocation_url?: string; // For serverless agents
  mcp_summary?: MCPSummaryForUI;
  reasoner_count: number;
  skill_count: number;
}

export interface AgentNodeDetailsForUI extends AgentNode {}

export interface AgentNodeDetailsForUIWithPackage extends AgentNode {
  package_info?: {
    package_id: string;
  };
}

export interface MCPHealthResponse {
  status: string;
  mcp_servers?: MCPServerHealthForUI[];
  mcp_summary?: MCPSummaryForUI;
}

export interface MCPServerActionResponse {
  status: string;
  success?: boolean;
  error_details?: MCPErrorDetails;
  server_alias?: string;
}

export interface MCPToolsResponse {
  tools: MCPTool[];
}

export interface MCPOverallStatusResponse {
  status: string;
}

export interface MCPToolTestRequest {
  node_id: string;
  server_alias: string;
  tool_name: string;
  parameters: any;
  timeout_ms?: number;
}

export interface MCPToolTestResponse {
  success?: boolean;
  error?: string;
  execution_time_ms?: number;
  result?: any;
}

export interface MCPServerMetricsResponse {
  metrics: MCPServerMetrics | MCPNodeMetrics;
  node_id?: string;
  server_alias?: string;
  timestamp: string;
}

export interface MCPHealthEventResponse {
  events: MCPHealthEvent[];
}

export interface MCPHealthResponseModeAware {
  status: string;
  mcp_servers?: MCPServerHealthForUI[];
  mcp_summary?: MCPSummaryForUI;
}

export interface MCPError extends Error {
  code: string;
  details?: any;
  isRetryable: boolean;
  retryAfterMs?: number;
}

export type AppMode = 'user' | 'admin' | 'developer';

export interface EnvResponse {
  agent_id: string;
  package_id: string;
  variables: Record<string, string>;
  masked_keys: string[];
  file_exists: boolean;
  last_modified?: string;
}

export interface SetEnvRequest {
  variables: Record<string, string>;
}

export interface ConfigSchemaResponse {
  schema: ConfigurationSchema;
  metadata?: {
    package_name?: string;
    package_version?: string;
    description?: string;
  };
}

export type AgentState = 'active' | 'inactive' | 'starting' | 'stopping' | 'error';

export interface AgentStatus {
  status: string;
  state?: AgentState;
  state_transition?: {
    from: AgentState;
    to: AgentState;
    reason?: string;
  };
  health_score?: number;
  last_seen?: string;
  health_status?: HealthStatus;
  lifecycle_status?: LifecycleStatus;
  mcp_status?: {
    running_servers: number;
    total_servers: number;
    service_status?: string;
  };
}

export interface AgentStatusUpdate {
  status: string;
  health_status?: string;
  lifecycle_status?: string;
  last_heartbeat?: string;
}

export type StatusSource = 'agent' | 'mcp' | 'system';

export type HealthStatus = 'starting' | 'ready' | 'degraded' | 'offline' | 'active' | 'inactive' | 'unknown';

export type LifecycleStatus =
  | 'starting'
  | 'ready'
  | 'degraded'
  | 'offline'
  | 'running'
  | 'stopped'
  | 'error'
  | 'unknown';

export type MCPServerAction = 'start' | 'stop' | 'restart';

export type MCPServerStatus = 'running' | 'stopped' | 'error' | 'starting' | 'unknown';

export interface MCPServerHealthForUI {
  alias: string;
  status: MCPServerStatus;
  tool_count: number;
  started_at?: string;
  last_health_check?: string;
  error_message?: string;
  port?: number;
  process_id?: number;
  success_rate?: number;
  avg_response_time_ms?: number;
  status_icon?: string;
  status_color?: string;
  uptime_formatted?: string;
}

export interface MCPSummaryForUI {
  service_status: string;
  running_servers: number;
  total_servers: number;
  total_tools: number;
  overall_health: number;
  has_issues: boolean;
  capabilities_available: boolean;
}

export interface MCPHealthEvent {
  timestamp: string;
  type: string;
  server_alias?: string;
  node_id?: string;
  message: string;
  details?: any;
  data?: any;
}

export interface MCPServerMetrics {
  alias: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_response_time_ms: number;
  peak_response_time_ms: number;
  requests_per_minute: number;
  uptime_seconds: number;
  error_rate_percent: number;
}

export interface MCPNodeMetrics {
  node_id: string;
  total_requests: number;
  avg_response_time: number;
  error_rate: number;
  timestamp: string;
  servers: MCPServerMetrics[];
  total_servers: number;
  active_servers: number;
  overall_health_score: number;
}

export interface MCPErrorDetails {
  message?: string;
  code?: string;
}

export type AgentConfigurationStatus = 'configured' | 'not_configured' | 'partially_configured' | 'unknown';

export interface AgentPackage {
  id: string;
  package_id?: string;
  name: string;
  version: string;
  description?: string;
  author?: string;
  tags?: string[];
  installed_at?: string;
  configuration_status?: AgentConfigurationStatus;
  configuration_schema?: ConfigurationSchema;
}

export type AgentLifecycleState = 'running' | 'stopped' | 'starting' | 'stopping' | 'error' | 'unknown';

export interface AgentLifecycleInfo {
  id: string;
  status: AgentLifecycleState;
  started_at?: string;
  last_updated?: string;
  error_message?: string;
}

export interface ReasonerDefinition {
  id: string;
  name: string;
  description?: string;
  input_schema?: any;
  tags?: string[];
  memory_config?: {
    memory_retention?: string;
    [key: string]: any;
  };
}

export interface SkillDefinition {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
}

export type AgentConfiguration = Record<string, any>;

export type ConfigFieldType = 'text' | 'secret' | 'number' | 'boolean' | 'select';

export interface ConfigFieldOption {
  value: string;
  label: string;
  description?: string;
}

export interface ConfigFieldValidation {
  min?: number;
  max?: number;
  pattern?: string;
}

export interface ConfigField {
  name: string;
  type: ConfigFieldType;
  label?: string;
  description?: string;
  required?: boolean;
  default?: any;
  options?: ConfigFieldOption[];
  validation?: ConfigFieldValidation;
}

export interface ConfigurationSchema {
  fields?: ConfigField[];
  user_environment?: {
    required?: ConfigField[];
    optional?: ConfigField[];
  };
  metadata?: Record<string, any>;
  version?: string;
}

export interface MCPTool {
  name: string;
  description?: string;
  input_schema?: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
  };
  inputSchema?: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
  };
}

export interface MCPHealthResponseUser {
  status: string;
}

export interface MCPHealthResponseDeveloper {
  mcp_servers: MCPServerHealthForUI[];
  mcp_summary: MCPSummaryForUI;
}
