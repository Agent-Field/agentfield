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
  reasoner_count: number;
  skill_count: number;
}

export interface AgentNodeDetailsForUI extends AgentNode {}

export interface AgentNodeDetailsForUIWithPackage extends AgentNode {
  package_info?: {
    package_id: string;
  };
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
}

export interface AgentStatusUpdate {
  status: string;
  health_status?: string;
  lifecycle_status?: string;
  last_heartbeat?: string;
}

export type StatusSource = 'agent' | 'system';

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

