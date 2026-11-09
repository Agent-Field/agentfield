/**
 * Pure logic for the developer Home page: the status line, the alert rules,
 * the figures compared with their usual level, and the call-map layout.
 * Everything here reads the /dashboard/home payload and never invents a
 * value: when a number is unknown the formatters return "—" and optional
 * cards are left out.
 */
import type {
  DeveloperHomeResponse,
  HomeWindowPreset,
} from "@/types/developerHome";
import type { LLMHealthResponse, QueueStatusResponse } from "@/hooks/queries";

export const HOME_WINDOWS: ReadonlyArray<{ value: HomeWindowPreset; label: string }> = [
  { value: "1h", label: "1h" },
  { value: "6h", label: "6h" },
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
];

const WINDOW_PHRASE: Record<HomeWindowPreset, string> = {
  "1h": "in the last hour",
  "6h": "in the last 6 hours",
  "24h": "in the last 24 hours",
  "7d": "in the last 7 days",
};

export function windowPhrase(preset: HomeWindowPreset): string {
  return WINDOW_PHRASE[preset] ?? "in this window";
}

/** Hours covered by the response window. */
export function windowHours(home: DeveloperHomeResponse): number {
  const ms = Date.parse(home.window.end) - Date.parse(home.window.start);
  return ms > 0 ? ms / 3_600_000 : 1;
}

// ─── formatting ─────────────────────────────────────────────────────────────

const countFormat = new Intl.NumberFormat("en-US");

export function formatCount(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return "—";
  return countFormat.format(Math.round(n));
}

/** A rate per hour; small rates keep one decimal so 0.4/h is not shown as 0. */
export function formatRate(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return "—";
  if (n > 0 && n < 10) return n.toFixed(1);
  return formatCount(n);
}

export function formatPct(p: number | undefined | null): string {
  if (p == null || Number.isNaN(p)) return "—";
  if (p === 0) return "0%";
  if (p < 0.1) return "<0.1%";
  if (p < 10) return `${p.toFixed(1)}%`;
  return `${Math.round(p)}%`;
}

export function formatMs(ms: number | undefined | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 3_600_000) return `${(ms / 60_000).toFixed(1)} min`;
  return `${(ms / 3_600_000).toFixed(1)} h`;
}

export function formatUsd(v: number | undefined | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (v > 0 && v < 0.01) return "<$0.01";
  return `$${v.toFixed(v < 10 ? 2 : 0)}`;
}

/** Compact age between two instants: "40 s", "12 min", "3 h", "2 d". */
export function formatAge(from: string | Date | undefined, now: Date): string {
  if (!from) return "—";
  const t = typeof from === "string" ? Date.parse(from) : from.getTime();
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, Math.round((now.getTime() - t) / 1000));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s < 86_400) return `${Math.round(s / 3600)} h`;
  return `${Math.round(s / 86_400)} d`;
}

export function failedPct(failed: number, terminal: number): number | undefined {
  return terminal > 0 ? (failed / terminal) * 100 : undefined;
}

// ─── fixed alert rules ──────────────────────────────────────────────────────

/** A failure-rate alert needs this many finished calls in the window... */
export const FAILURE_ALERT_MIN_FINISHED = 20;
/** ...and this many in the baseline, so a handful of calls never alarms. */
export const FAILURE_ALERT_MIN_BASELINE = 50;
/** The rate must be at least this multiple of the usual rate... */
export const FAILURE_ALERT_RATIO = 2;
/** ...and at least this many percentage points above it. */
export const FAILURE_ALERT_MIN_POINTS = 5;

export type AlertTone = "bad" | "warn";

export interface HomeAction {
  label: string;
  href: string;
}

export interface HomeAlert {
  id: string;
  tone: AlertTone;
  title: string;
  detail: string;
  rule: string;
  actions: HomeAction[];
}

export interface FailureRateCheck {
  current?: number;
  usual?: number;
  alert: boolean;
}

export function checkFailureRate(home: DeveloperHomeResponse): FailureRateCheck {
  const current = failedPct(home.totals.counts.failed, home.totals.counts.terminal);
  const b = home.baseline;
  const usual = b ? failedPct(b.counts.failed, b.counts.terminal) : undefined;
  const alert =
    current !== undefined &&
    usual !== undefined &&
    home.totals.counts.terminal >= FAILURE_ALERT_MIN_FINISHED &&
    (b?.counts.terminal ?? 0) >= FAILURE_ALERT_MIN_BASELINE &&
    current >= usual * FAILURE_ALERT_RATIO &&
    current - usual >= FAILURE_ALERT_MIN_POINTS;
  return { current, usual, alert };
}

export function runsFilterHref(params: { status?: string; search?: string }): string {
  const q = new URLSearchParams();
  if (params.status) q.set("status", params.status);
  if (params.search) q.set("search", params.search);
  const s = q.toString();
  return s ? `/runs?${s}` : "/runs";
}

export function buildAlerts(
  home: DeveloperHomeResponse,
  extras: { llm?: LLMHealthResponse; queue?: QueueStatusResponse },
  now: Date,
): HomeAlert[] {
  const alerts: HomeAlert[] = [];
  const phrase = windowPhrase(home.window.preset);
  const windowStart = Date.parse(home.window.start);

  for (const agent of home.stopped_agents) {
    const lastBeat = Date.parse(agent.last_heartbeat);
    const recent = !Number.isNaN(lastBeat) && lastBeat >= windowStart;
    if (!recent && agent.entry_points.length === 0) continue;

    const parts: string[] = [];
    if (agent.entry_points.length > 0) {
      const list = agent.entry_points
        .slice(0, 4)
        .map((e) => `${e.agent_node_id}.${e.reasoner_id} (${formatCount(e.runs)} run${e.runs === 1 ? "" : "s"})`)
        .join(", ");
      const more = agent.entry_points.length > 4 ? ` and ${agent.entry_points.length - 4} more` : "";
      parts.push(`Entry points whose runs reached it ${phrase}: ${list}${more}.`);
    }
    parts.push("New calls to it are refused, without a run being recorded, until it sends a heartbeat again.");
    alerts.push({
      id: `stopped:${agent.id}`,
      tone: "bad",
      title: `${agent.id} stopped sending heartbeats ${formatAge(agent.last_heartbeat, now)} ago`,
      detail: parts.join(" "),
      rule: "Shown for a registered agent the control plane marked offline when its heartbeats stopped.",
      actions: [
        { label: "Failed runs", href: runsFilterHref({ status: "failed", search: agent.id }) },
        { label: "Agents", href: "/agents" },
      ],
    });
  }

  const rate = checkFailureRate(home);
  if (rate.alert) {
    alerts.push({
      id: "failure-rate",
      tone: "bad",
      title: `Failed calls at ${formatPct(rate.current)}, usual ${formatPct(rate.usual)}`,
      detail: `${formatCount(home.totals.counts.failed)} of ${formatCount(home.totals.counts.terminal)} finished calls failed ${phrase}.`,
      rule: `Shown when the failure rate is at least ${FAILURE_ALERT_RATIO}× the usual rate and ${FAILURE_ALERT_MIN_POINTS} points above it, with ${FAILURE_ALERT_MIN_FINISHED}+ finished calls.`,
      actions: [{ label: "Failed runs", href: runsFilterHref({ status: "failed" }) }],
    });
  }

  if (home.triggers && home.triggers.dead_letters > 0) {
    const n = home.triggers.dead_letters;
    alerts.push({
      id: "trigger-dlq",
      tone: "bad",
      title: `${formatCount(n)} trigger event${n === 1 ? " is" : "s are"} in the dead-letter queue`,
      detail: "The dispatcher could not deliver them to their target reasoner.",
      rule: "Shown when the trigger dead-letter queue is not empty.",
      actions: [{ label: "Triggers", href: "/triggers" }],
    });
  }

  if (extras.llm?.enabled) {
    const open = extras.llm.endpoints.filter((e) => e.circuit_state === "open");
    if (open.length > 0) {
      alerts.push({
        id: "llm-circuit",
        tone: "bad",
        title: `LLM circuit open: ${open.map((e) => e.name).join(", ")}`,
        detail: open[0].last_error ? `Last error: ${open[0].last_error}` : "Calls to it are being refused until it recovers.",
        rule: "Shown when a configured LLM endpoint's circuit breaker is open.",
        actions: [],
      });
    }
  }

  if (extras.queue?.enabled) {
    const full = extras.queue.agents.filter((a) => a.max > 0 && a.running >= a.max);
    for (const a of full) {
      alerts.push({
        id: `queue:${a.agent_node_id}`,
        tone: "warn",
        title: `${a.agent_node_id} is at its concurrency limit (${a.running}/${a.max})`,
        detail: "New calls to it wait for a free slot.",
        rule: "Shown when an agent uses every slot the concurrency limiter allows.",
        actions: [{ label: "Runs", href: runsFilterHref({ search: a.agent_node_id }) }],
      });
    }
  }

  return alerts;
}

// ─── status line ────────────────────────────────────────────────────────────

export interface StatusLine {
  problems: number;
  healthy: number;
  registered: number;
  detail: string[];
}

export function buildStatusLine(home: DeveloperHomeResponse, alerts: HomeAlert[]): StatusLine {
  const detail: string[] = [];
  const calls = home.totals.counts.total;
  detail.push(`${formatCount(calls)} call${calls === 1 ? "" : "s"} ${windowPhrase(home.window.preset)}`);
  if (home.problems.length > 0) {
    detail.push(`${home.problems.length} failure cause${home.problems.length === 1 ? "" : "s"}`);
  }
  const waiting = home.approvals.total + home.agent_counts.pending_approval;
  if (waiting > 0) detail.push(`${formatCount(waiting)} waiting on people`);
  return {
    problems: alerts.length,
    healthy: home.agent_counts.running,
    registered: home.agent_counts.registered,
    detail,
  };
}

// ─── figures compared with their usual level ────────────────────────────────

export type FigureTone = "bad" | "warn" | undefined;

export interface Figure {
  key: string;
  label: string;
  value: string;
  usual?: string;
  /** False for a figure that has no usual level to compare with. */
  comparable?: boolean;
  tone?: FigureTone;
  note?: string;
  spark?: number[];
  href?: string;
}

export function buildFigures(home: DeveloperHomeResponse, now: Date): Figure[] {
  const hours = windowHours(home);
  const b = home.baseline;
  const bHours = b && b.span_seconds > 0 ? b.span_seconds / 3600 : undefined;
  const figures: Figure[] = [];

  figures.push({
    key: "calls",
    label: "Calls / hour",
    value: formatRate(home.totals.counts.total / hours),
    usual: b && bHours ? formatRate(b.counts.total / bHours) : undefined,
    spark: home.series.map((s) => s.calls),
    href: "/runs",
  });

  const rate = checkFailureRate(home);
  let failTone: FigureTone;
  if (rate.alert) failTone = "bad";
  else if (rate.current !== undefined && rate.usual !== undefined && rate.current > rate.usual * 1.5 && rate.current - rate.usual >= 1) failTone = "warn";
  figures.push({
    key: "failed",
    label: "Failed",
    value: formatPct(rate.current),
    usual: rate.usual !== undefined ? formatPct(rate.usual) : undefined,
    tone: failTone,
    note: home.totals.counts.terminal > 0
      ? `${formatCount(home.totals.counts.failed)} of ${formatCount(home.totals.counts.terminal)} finished`
      : undefined,
    spark: home.series.map((s) => (s.terminal > 0 ? (s.failed / s.terminal) * 100 : 0)),
    href: runsFilterHref({ status: "failed" }),
  });

  const p95 = home.totals.p95_ms;
  const usualP95 = b?.p95_ms;
  figures.push({
    key: "p95",
    label: "p95 duration",
    value: formatMs(p95),
    usual: usualP95 !== undefined ? formatMs(usualP95) : undefined,
    tone: p95 !== undefined && usualP95 !== undefined && usualP95 > 0 && p95 > usualP95 * 2 ? "warn" : undefined,
    note: home.totals.p50_ms !== undefined ? `p50 ${formatMs(home.totals.p50_ms)}` : undefined,
    spark: home.series.filter((s) => s.p95_ms !== undefined).map((s) => s.p95_ms as number),
  });

  figures.push({
    key: "running",
    label: "Running now",
    value: formatCount(home.totals.running_now),
    usual: b && b.span_seconds > 0 ? `${(b.busy_ms / (b.span_seconds * 1000)).toFixed(1)} avg` : undefined,
    href: runsFilterHref({ status: "running" }),
  });

  const pendingAgents = home.agent_counts.pending_approval;
  const waiting = home.approvals.total + pendingAgents;
  const oldest = home.approvals.items[0]?.requested_at;
  const notes: string[] = [];
  if (oldest) notes.push(`oldest ${formatAge(oldest, now)}`);
  if (pendingAgents > 0) notes.push(`${pendingAgents} agent${pendingAgents === 1 ? "" : "s"} awaiting tag approval`);
  figures.push({
    key: "waiting",
    label: "Waiting on people",
    value: formatCount(waiting),
    comparable: false,
    tone: waiting > 0 ? "warn" : undefined,
    note: notes.join(" · ") || undefined,
  });

  if (home.spend) {
    const s = home.spend;
    figures.push({
      key: "spend",
      label: "Spend / hour",
      value: s.window_cost_usd !== undefined ? formatUsd(s.window_cost_usd / hours) : "—",
      usual: s.baseline_cost_usd !== undefined && bHours ? formatUsd(s.baseline_cost_usd / bHours) : undefined,
      note: s.window_tokens > 0 ? `${formatCount(s.window_tokens)} tokens` : undefined,
    });
  }

  return figures;
}

// ─── call map layout ────────────────────────────────────────────────────────

export const DIRECT_SOURCE = "direct calls";

export interface MapNode {
  id: string;
  label: string;
  kind: "source" | "agent";
  layer: number;
  x: number;
  y: number;
  status?: string;
  selfCalls: number;
}

export interface MapLink {
  id: string;
  from: string;
  to: string;
  calls: number;
  failed: number;
  held: number;
  tone: "bad" | "warn" | "ok";
  width: number;
  path: string;
}

export interface CallMapLayout {
  nodes: MapNode[];
  links: MapLink[];
  width: number;
  height: number;
  layers: number;
  idle: string[];
}

export const MAP_NODE_W = 156;
export const MAP_NODE_H = 24;
const MAP_ROW = 40;
const MAP_PAD_X = 8;
const MAP_PAD_TOP = 8;
const MAP_PAD_BOTTOM = 12;
/** Extra room under the boxes when a back-call arc dips below them. */
const MAP_ARC_ROOM = 26;

function sourceId(source: string): string {
  return source ? `trigger:${source}` : DIRECT_SOURCE;
}

/**
 * Lays the agents out in columns: who calls in from outside on the left, then
 * each agent one column right of the furthest agent that calls it. Back edges
 * found by a depth-first walk are drawn as arcs and do not move columns, so a
 * cycle cannot push the layout off to the right.
 */
export function layoutCallMap(home: DeveloperHomeResponse, width = 760): CallMapLayout {
  const agentStatus = new Map(home.agents.map((a) => [a.id, a.status]));
  const selfCalls = new Map<string, number>();
  type RawLink = { from: string; to: string; calls: number; failed: number; held: number };
  const raw: RawLink[] = [];

  for (const e of home.entries) {
    raw.push({ from: sourceId(e.source), to: e.target, calls: e.calls, failed: e.failed, held: e.held });
  }
  for (const e of home.edges) {
    if (e.caller === e.target) {
      selfCalls.set(e.caller, (selfCalls.get(e.caller) ?? 0) + e.calls);
      continue;
    }
    raw.push({ from: e.caller, to: e.target, calls: e.calls, failed: e.failed, held: e.held });
  }

  const sources = [...new Set(home.entries.map((e) => sourceId(e.source)))].sort((a, b) =>
    a === DIRECT_SOURCE ? -1 : b === DIRECT_SOURCE ? 1 : a.localeCompare(b),
  );
  const agentIds = new Set<string>();
  for (const l of raw) {
    if (!sources.includes(l.from)) agentIds.add(l.from);
    agentIds.add(l.to);
  }
  for (const s of home.stopped_agents) agentIds.add(s.id);
  for (const id of selfCalls.keys()) agentIds.add(id);

  const out = new Map<string, string[]>();
  for (const l of raw) {
    if (!out.has(l.from)) out.set(l.from, []);
    out.get(l.from)!.push(l.to);
  }
  for (const list of out.values()) list.sort();

  // Depth-first walk from the sources marks back edges.
  const back = new Set<string>();
  const state = new Map<string, 1 | 2>();
  const visit = (n: string) => {
    state.set(n, 1);
    for (const m of out.get(n) ?? []) {
      const st = state.get(m);
      if (st === 1) back.add(`${n}>${m}`);
      else if (!st) visit(m);
    }
    state.set(n, 2);
  };
  for (const s of sources) if (!state.has(s)) visit(s);
  for (const a of [...agentIds].sort()) if (!state.has(a)) visit(a);

  // Longest path over the remaining edges.
  const layer = new Map<string, number>();
  for (const s of sources) layer.set(s, 0);
  const forward = raw.filter((l) => !back.has(`${l.from}>${l.to}`));
  const hasSources = sources.length > 0;
  for (const a of agentIds) layer.set(a, hasSources ? 1 : 0);
  for (let i = 0; i <= agentIds.size + 1; i++) {
    let changed = false;
    for (const l of forward) {
      const want = (layer.get(l.from) ?? 0) + 1;
      if (want > (layer.get(l.to) ?? 0)) {
        layer.set(l.to, want);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const layers = Math.max(1, ...[...layer.values()].map((v) => v + 1));
  const columns: string[][] = Array.from({ length: layers }, () => []);
  for (const s of sources) columns[0].push(s);
  for (const a of [...agentIds].sort()) columns[layer.get(a) ?? 0].push(a);

  // Order each column by the mean row of its callers to cut crossings.
  const row = new Map<string, number>();
  columns[0].forEach((id, i) => row.set(id, i));
  for (let c = 1; c < layers; c++) {
    const score = (id: string) => {
      const preds = forward.filter((l) => l.to === id && row.has(l.from));
      if (preds.length === 0) return Number.MAX_SAFE_INTEGER;
      return preds.reduce((s, l) => s + (row.get(l.from) ?? 0), 0) / preds.length;
    };
    columns[c].sort((a, b) => score(a) - score(b) || a.localeCompare(b));
    columns[c].forEach((id, i) => row.set(id, i));
  }

  const tallest = Math.max(1, ...columns.map((c) => c.length));
  const hasArcs = raw.some((l) => (layer.get(l.to) ?? 0) <= (layer.get(l.from) ?? 0));
  const height = MAP_PAD_TOP + tallest * MAP_ROW + MAP_PAD_BOTTOM + (hasArcs ? MAP_ARC_ROOM : 0);
  const colGap = layers > 1 ? (width - 2 * MAP_PAD_X - MAP_NODE_W) / (layers - 1) : 0;

  const nodes: MapNode[] = [];
  const pos = new Map<string, MapNode>();
  columns.forEach((col, c) => {
    const offset = ((tallest - col.length) * MAP_ROW) / 2;
    col.forEach((id, i) => {
      const isSource = sources.includes(id);
      const node: MapNode = {
        id,
        label: id,
        kind: isSource ? "source" : "agent",
        layer: c,
        x: MAP_PAD_X + c * colGap,
        y: MAP_PAD_TOP + offset + i * MAP_ROW,
        status: isSource ? undefined : agentStatus.get(id),
        selfCalls: selfCalls.get(id) ?? 0,
      };
      nodes.push(node);
      pos.set(id, node);
    });
  });

  const maxCalls = Math.max(1, ...raw.map((l) => l.calls));
  const links: MapLink[] = raw.map((l) => {
    const a = pos.get(l.from)!;
    const b = pos.get(l.to)!;
    let path: string;
    if (b.x > a.x) {
      const x1 = a.x + MAP_NODE_W;
      const y1 = a.y + MAP_NODE_H / 2;
      const x2 = b.x;
      const y2 = b.y + MAP_NODE_H / 2;
      const mid = (x2 - x1) / 2;
      path = `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`;
    } else {
      // Back or same-column call: an arc under both boxes.
      const x1 = a.x + MAP_NODE_W / 2;
      const y1 = a.y + MAP_NODE_H;
      const x2 = b.x + MAP_NODE_W / 2;
      const y2 = b.y + MAP_NODE_H;
      const dip = Math.max(y1, y2) + 22;
      path = `M ${x1} ${y1} C ${x1} ${dip}, ${x2} ${dip}, ${x2} ${y2}`;
    }
    const tone: MapLink["tone"] = l.failed > 0 ? "bad" : l.held > 0 ? "warn" : "ok";
    const width = 1 + 3 * (Math.log10(l.calls + 1) / Math.log10(maxCalls + 1));
    return { id: `${l.from}>${l.to}`, ...l, tone, width: Math.round(width * 10) / 10, path };
  });
  // Draw quiet links first so failing ones sit on top.
  const order = { ok: 0, warn: 1, bad: 2 } as const;
  links.sort((x, y) => order[x.tone] - order[y.tone]);

  const idle = home.agents
    .filter((a) => !agentIds.has(a.id) && a.status === "running")
    .map((a) => a.id)
    .sort();

  return { nodes, links, width, height, layers, idle };
}

/** True when an install has nothing to show yet: no agent ever registered and no calls. */
export function isEmptyInstall(home: DeveloperHomeResponse, runCount: number): boolean {
  return home.agent_counts.registered === 0 && home.totals.counts.total === 0 && runCount === 0;
}

// ─── first agent ────────────────────────────────────────────────────────────

export function connectSnippet(origin: string): string {
  return `from agentfield import Agent

app = Agent(
    node_id="my-first-agent",
    agentfield_server="${origin}",
    # api_key="...",  # needed when the control plane requires an API key
)

@app.reasoner()
async def hello(name: str) -> dict:
    return {"greeting": f"Hello, {name}"}

if __name__ == "__main__":
    app.serve(port=8001)`;
}

export function callSnippet(origin: string): string {
  return [
    `curl -X POST ${origin}/api/v1/execute/my-first-agent.hello \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"input": {"name": "world"}}'`,
  ].join("\n");
}
