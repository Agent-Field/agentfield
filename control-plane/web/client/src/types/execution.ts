export interface ExecutionRequest {
  input: any;
  context?: {
    workflow_id?: string;
    session_id?: string;
    user_id?: string;
  };
  memory_options?: {
    auto_inject?: string[];
    store_result?: boolean;
    result_ttl?: string;
  };
  webhook?: {
    url: string;
    secret?: string;
    headers?: Record<string, string>;
  };
}

import type { CanonicalStatus } from "../utils/status";

export interface ExecutionResponse {
  execution_id: string;
  result: any;
  duration_ms: number;
  cost?: number;
  status: CanonicalStatus;
  error_message?: string;
  memory_updates?: MemoryUpdate[];
  timestamp: string;
  node_id: string;
  type: string;
  target: string;
  workflow_id: string;
  run_id?: string;
}

export interface MemoryUpdate {
  scope: string;
  key: string;
  action: "created" | "updated" | "deleted";
}

export interface ExecutionHistory {
  executions: ExecutionHistoryItem[];
  total: number;
  page: number;
  limit: number;
}

export interface ExecutionHistoryItem {
  execution_id: string;
  input: any;
  result?: any;
  duration_ms: number;
  status: CanonicalStatus;
  error_message?: string;
  timestamp: string;
  cost?: number;
}

export interface PerformanceMetrics {
  avg_response_time_ms: number;
  success_rate: number;
  total_executions: number;
  executions_last_24h: number;
  error_rate: number;
  cost_last_24h?: number;
  recent_executions: RecentExecution[];
  performance_trend: PerformanceTrend[];
}

export interface RecentExecution {
  execution_id: string;
  duration_ms: number;
  status: CanonicalStatus;
  timestamp: string;
}

export interface PerformanceTrend {
  timestamp: string;
  avg_response_time: number;
  success_rate: number;
  execution_count: number;
}

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: any;
  examples?: any[];
  example?: any;
  enum?: any[];
  const?: any;
  format?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema | JsonSchema[];
  additionalItems?: JsonSchema | boolean;
  minItems?: number;
  maxItems?: number;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  oneOf?: JsonSchema[];
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  additionalProperties?: boolean | JsonSchema;
}

export interface FormField {
  name: string;
  label: string;
  type: "string" | "number" | "boolean" | "object" | "array" | "select";
  required: boolean;
  description?: string;
  placeholder?: string;
  options?: string[];
  enumValues?: any[];
  schema?: JsonSchema;
  defaultValue?: any;
  examples?: any[];
  format?: string;
  itemSchema?: JsonSchema | null;
  tupleSchemas?: JsonSchema[];
  minItems?: number;
  maxItems?: number;
  combinator?: "oneOf" | "anyOf" | "allOf";
  variantSchemas?: JsonSchema[];
  variantTitles?: string[];
}

export interface ExecutionTemplate {
  id: string;
  name: string;
  description?: string;
  input: any;
  created_at: string;
  last_used?: string;
}

export interface AsyncExecuteResponse {
  execution_id: string;
  workflow_id: string;
  run_id?: string;
  status: string;
  target: string;
  type: string;
  estimated_completion?: string;
  created_at: string;
  webhook_registered?: boolean;
}

export interface ExecutionStatusResponse {
  execution_id: string;
  workflow_id: string;
  run_id?: string;
  status: string;
  target: string;
  type: string;
  progress: number;
  result?: any;
  error?: string;
  started_at: string;
  completed_at?: string;
  duration?: number;
}
