/**
 * Payload of GET /api/ui/v1/dashboard/home. Every number is computed by the
 * control plane from recorded rows; optional sections are absent when the
 * feature they describe is not in use.
 */

export type HomeWindowPreset = "1h" | "6h" | "24h" | "7d";

export type HomeAgentStatus =
  | "running"
  | "degraded"
  | "offline"
  | "starting"
  | "pending_approval"
  | "unknown";

export interface HomeAgent {
  id: string;
  version?: string;
  status: HomeAgentStatus;
  health: string;
  lifecycle: string;
  last_heartbeat: string;
  reasoners: number;
  skills: number;
}

export interface HomeAgentCounts {
  registered: number;
  running: number;
  offline: number;
  degraded: number;
  starting: number;
  pending_approval: number;
  unknown: number;
}

export interface HomeEntryPoint {
  agent_node_id: string;
  reasoner_id: string;
  runs: number;
  failed_runs: number;
}

export interface HomeStoppedAgent {
  id: string;
  last_heartbeat: string;
  entry_points: HomeEntryPoint[];
}

export interface HomeStatusCounts {
  total: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  terminal: number;
}

export interface HomeTotals {
  counts: HomeStatusCounts;
  held: number;
  running_now: number;
  p50_ms?: number;
  p95_ms?: number;
}

export interface HomeBaseline {
  start: string;
  end: string;
  span_seconds: number;
  counts: HomeStatusCounts;
  p95_ms?: number;
  busy_ms: number;
}

export interface HomeBucket {
  start: string;
  calls: number;
  failed: number;
  terminal: number;
  p95_ms?: number;
}

export interface HomeProblem {
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  message: string;
  count: number;
  first_seen: string;
  last_seen: string;
  example_execution_id: string;
  example_run_id: string;
  buckets: number[];
}

export interface HomeEdge {
  caller: string;
  target: string;
  calls: number;
  failed: number;
  held: number;
}

export interface HomeEntryEdge {
  source: string;
  target: string;
  calls: number;
  failed: number;
  held: number;
}

export interface HomeEndpoint {
  agent_node_id: string;
  reasoner_id: string;
  calls: number;
  failed: number;
  terminal: number;
  p50_ms?: number;
  p95_ms?: number;
  caller_agents: string[];
  direct_calls: number;
  trigger_calls: number;
}

export interface HomeApproval {
  execution_id: string;
  run_id: string;
  agent_node_id: string;
  reasoner_id: string;
  request_url?: string;
  requested_at?: string;
  expires_at?: string;
}

export interface HomeSpend {
  window_cost_usd?: number;
  window_tokens: number;
  window_rows: number;
  baseline_cost_usd?: number;
  baseline_rows: number;
}

export interface HomeWebhookFailure {
  execution_id: string;
  url: string;
  attempt_count: number;
  last_error?: string;
  last_attempt_at?: string;
}

export interface HomeWebhooks {
  registered: number;
  delivered: number;
  failed: number;
  pending: number;
  recent_failures: HomeWebhookFailure[];
}

export interface HomeTriggers {
  configured: number;
  enabled: number;
  events: number;
  dispatched: number;
  failed: number;
  dead_letters: number;
}

export interface HomeForwarder {
  enabled: boolean;
  dead_letters: number;
}

export interface DeveloperHomeResponse {
  generated_at: string;
  window: {
    preset: HomeWindowPreset;
    start: string;
    end: string;
    bucket_seconds: number;
  };
  agents: HomeAgent[];
  agent_counts: HomeAgentCounts;
  stopped_agents: HomeStoppedAgent[];
  totals: HomeTotals;
  baseline?: HomeBaseline;
  series: HomeBucket[];
  problems: HomeProblem[];
  edges: HomeEdge[];
  entries: HomeEntryEdge[];
  endpoints: HomeEndpoint[];
  approvals: { total: number; items: HomeApproval[] };
  spend?: HomeSpend;
  webhooks?: HomeWebhooks;
  triggers?: HomeTriggers;
  forwarder?: HomeForwarder;
}
