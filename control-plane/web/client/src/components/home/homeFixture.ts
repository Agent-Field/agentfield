import type { DeveloperHomeResponse } from "@/types/developerHome";

export const FIXTURE_NOW = "2026-11-09T12:00:00Z";

/** A small mesh in its last hour: one agent stopped, failures on one edge. */
export function homeFixture(overrides: Partial<DeveloperHomeResponse> = {}): DeveloperHomeResponse {
  return {
    generated_at: FIXTURE_NOW,
    window: {
      preset: "1h",
      start: "2026-11-09T11:00:00Z",
      end: FIXTURE_NOW,
      bucket_seconds: 300,
    },
    agents: [
      { id: "billing", status: "running", health: "active", lifecycle: "ready", last_heartbeat: "2026-11-09T11:59:50Z", reasoners: 2, skills: 0 },
      { id: "orchestrator", status: "offline", health: "inactive", lifecycle: "offline", last_heartbeat: "2026-11-09T11:40:00Z", reasoners: 3, skills: 0 },
      { id: "payments", status: "pending_approval", health: "active", lifecycle: "pending_approval", last_heartbeat: "2026-11-09T11:59:55Z", reasoners: 1, skills: 0 },
      { id: "spare", status: "running", health: "active", lifecycle: "ready", last_heartbeat: "2026-11-09T11:59:58Z", reasoners: 1, skills: 0 },
    ],
    agent_counts: { registered: 4, running: 2, offline: 1, degraded: 0, starting: 0, pending_approval: 1, unknown: 0 },
    stopped_agents: [
      {
        id: "orchestrator",
        last_heartbeat: "2026-11-09T11:40:00Z",
        entry_points: [{ agent_node_id: "orchestrator", reasoner_id: "process_order", runs: 40, failed_runs: 30 }],
      },
    ],
    totals: {
      counts: { total: 200, succeeded: 150, failed: 40, cancelled: 0, terminal: 190 },
      held: 2,
      running_now: 8,
      p50_ms: 120,
      p95_ms: 2400,
    },
    baseline: {
      start: "2026-11-02T11:00:00Z",
      end: "2026-11-09T11:00:00Z",
      span_seconds: 7 * 24 * 3600,
      counts: { total: 16800, succeeded: 16632, failed: 168, cancelled: 0, terminal: 16800 },
      p95_ms: 900,
      busy_ms: 7 * 24 * 3600 * 1000 * 3,
    },
    series: [
      { start: "2026-11-09T11:00:00Z", calls: 20, failed: 0, terminal: 20, p95_ms: 800 },
      { start: "2026-11-09T11:30:00Z", calls: 100, failed: 20, terminal: 95, p95_ms: 2000 },
      { start: "2026-11-09T11:55:00Z", calls: 80, failed: 20, terminal: 75, p95_ms: 2600 },
    ],
    problems: [
      {
        agent_node_id: "billing",
        reasoner_id: "charge",
        status: "failed",
        message: "connection refused to <id> after N attempts",
        count: 40,
        first_seen: "2026-11-09T11:41:00Z",
        last_seen: "2026-11-09T11:59:00Z",
        example_execution_id: "exec_9",
        example_run_id: "run_9",
        buckets: [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10],
      },
    ],
    edges: [
      { caller: "orchestrator", target: "billing", calls: 60, failed: 40, held: 0 },
      { caller: "billing", target: "billing", calls: 5, failed: 0, held: 0 },
      { caller: "orchestrator", target: "payments", calls: 10, failed: 0, held: 2 },
    ],
    entries: [
      { source: "", target: "orchestrator", calls: 40, failed: 0, held: 0 },
      { source: "webhook", target: "billing", calls: 4, failed: 0, held: 0 },
    ],
    endpoints: [
      { agent_node_id: "billing", reasoner_id: "charge", calls: 64, failed: 40, terminal: 64, p50_ms: 100, p95_ms: 2400, caller_agents: ["orchestrator"], direct_calls: 0, trigger_calls: 4 },
      { agent_node_id: "orchestrator", reasoner_id: "process_order", calls: 40, failed: 0, terminal: 40, p50_ms: 900, p95_ms: 3000, caller_agents: [], direct_calls: 40, trigger_calls: 0 },
    ],
    approvals: {
      total: 1,
      items: [
        { execution_id: "exec_a", run_id: "run_a", agent_node_id: "payments", reasoner_id: "pay", requested_at: "2026-11-09T11:18:00Z" },
      ],
    },
    ...overrides,
  };
}
