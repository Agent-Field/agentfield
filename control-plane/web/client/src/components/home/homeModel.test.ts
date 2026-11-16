import { describe, expect, it } from "vitest";

import {
  DIRECT_SOURCE,
  buildAlerts,
  buildFigures,
  buildStatusLine,
  callSnippet,
  checkFailureRate,
  connectSnippet,
  formatAge,
  formatCount,
  formatMs,
  formatPct,
  formatRate,
  formatUsd,
  isEmptyInstall,
  layoutCallMap,
  runsFilterHref,
  windowHours,
  windowPhrase,
} from "./homeModel";
import { FIXTURE_NOW, homeFixture } from "./homeFixture";

const now = new Date(FIXTURE_NOW);

describe("formatters", () => {
  it("render unknown values as a dash, never as zero", () => {
    expect(formatCount(undefined)).toBe("—");
    expect(formatRate(null)).toBe("—");
    expect(formatPct(undefined)).toBe("—");
    expect(formatMs(undefined)).toBe("—");
    expect(formatUsd(undefined)).toBe("—");
    expect(formatAge(undefined, now)).toBe("—");
    expect(formatAge("not a date", now)).toBe("—");
  });

  it("format known values", () => {
    expect(formatCount(12345)).toBe("12,345");
    expect(formatRate(0.4)).toBe("0.4");
    expect(formatRate(1500)).toBe("1,500");
    expect(formatPct(0)).toBe("0%");
    expect(formatPct(0.04)).toBe("<0.1%");
    expect(formatPct(1.25)).toBe("1.3%");
    expect(formatPct(42.4)).toBe("42%");
    expect(formatMs(80)).toBe("80 ms");
    expect(formatMs(1234)).toBe("1.23 s");
    expect(formatMs(12_500)).toBe("12.5 s");
    expect(formatMs(90_000)).toBe("1.5 min");
    expect(formatMs(7_200_000)).toBe("2.0 h");
    expect(formatUsd(0.004)).toBe("<$0.01");
    expect(formatUsd(1.5)).toBe("$1.50");
    expect(formatUsd(42)).toBe("$42");
    expect(formatAge("2026-11-09T11:59:20Z", now)).toBe("40 s");
    expect(formatAge("2026-11-09T11:48:00Z", now)).toBe("12 min");
    expect(formatAge("2026-11-09T09:00:00Z", now)).toBe("3 h");
    expect(formatAge(new Date("2026-11-07T12:00:00Z"), now)).toBe("2 d");
  });

  it("build filter links for the runs page", () => {
    expect(runsFilterHref({})).toBe("/runs");
    expect(runsFilterHref({ status: "failed", search: "billing" })).toBe("/runs?status=failed&search=billing");
  });

  it("describe each window", () => {
    expect(windowPhrase("1h")).toBe("in the last hour");
    expect(windowPhrase("7d")).toBe("in the last 7 days");
    expect(windowHours(homeFixture())).toBe(1);
  });
});

describe("alerts", () => {
  it("report a stopped agent with the entry points that went through it", () => {
    const alerts = buildAlerts(homeFixture(), {}, now);
    const stopped = alerts.find((a) => a.id === "stopped:orchestrator");
    expect(stopped).toBeDefined();
    expect(stopped!.title).toBe("orchestrator stopped sending heartbeats 20 min ago");
    expect(stopped!.detail).toBe(
      "Entry points whose runs reached it in the last hour: orchestrator.process_order (40 runs). New calls to it are refused, without a run being recorded, until it sends a heartbeat again.",
    );
    expect(stopped!.actions[0].href).toBe("/runs?status=failed&search=orchestrator");
  });

  it("skip an agent that went quiet long ago and that nothing routes through", () => {
    const home = homeFixture({
      stopped_agents: [{ id: "old", last_heartbeat: "2026-10-01T00:00:00Z", entry_points: [] }],
    });
    expect(buildAlerts(home, {}, now).some((a) => a.id.startsWith("stopped:"))).toBe(false);
  });

  it("raise the failure-rate alert only when the fixed rule holds", () => {
    const home = homeFixture();
    const rate = checkFailureRate(home);
    expect(rate.current).toBeCloseTo((40 / 190) * 100);
    expect(rate.usual).toBeCloseTo(1);
    expect(rate.alert).toBe(true);
    expect(buildAlerts(home, {}, now).some((a) => a.id === "failure-rate")).toBe(true);

    // Without a baseline there is no "usual", so no alert.
    expect(checkFailureRate(homeFixture({ baseline: undefined })).alert).toBe(false);

    // Too few finished calls in the window.
    const quiet = homeFixture({
      totals: { counts: { total: 10, succeeded: 5, failed: 5, cancelled: 0, terminal: 10 }, held: 0, running_now: 0 },
    });
    expect(checkFailureRate(quiet).alert).toBe(false);

    // Twice the usual rate but under five points above it.
    const small = homeFixture({
      totals: { counts: { total: 1000, succeeded: 970, failed: 30, cancelled: 0, terminal: 1000 }, held: 0, running_now: 0 },
    });
    expect(checkFailureRate(small).alert).toBe(false);
  });

  it("show trigger dead letters, an open LLM circuit and a full queue only when configured", () => {
    const home = homeFixture({
      triggers: { configured: 2, enabled: 2, events: 10, dispatched: 8, failed: 2, dead_letters: 3 },
    });
    const ids = buildAlerts(
      home,
      {
        llm: {
          enabled: true,
          healthy: false,
          endpoints: [{ name: "openai", healthy: false, circuit_state: "open", consecutive_failures: 3, last_error: "timeout" }],
        },
        queue: { enabled: true, max_per_agent: 2, total_running: 2, agents: [{ agent_node_id: "billing", running: 2, max: 2, available: 0 }] },
      },
      now,
    ).map((a) => a.id);
    expect(ids).toEqual(expect.arrayContaining(["trigger-dlq", "llm-circuit", "queue:billing"]));

    const off = buildAlerts(
      homeFixture({ stopped_agents: [], baseline: undefined }),
      {
        llm: { enabled: false, healthy: false, endpoints: [{ name: "x", healthy: false, circuit_state: "open", consecutive_failures: 1 }] },
        queue: { enabled: false, max_per_agent: 0, total_running: 0, agents: [{ agent_node_id: "billing", running: 5, max: 1, available: 0 }] },
      },
      now,
    );
    expect(off).toEqual([]);
  });
});

describe("status line and figures", () => {
  it("count problems, healthy agents and what waits on people", () => {
    const home = homeFixture();
    const status = buildStatusLine(home, buildAlerts(home, {}, now));
    expect(status.problems).toBe(2);
    expect(status.healthy).toBe(2);
    expect(status.registered).toBe(4);
    expect(status.detail).toEqual(["200 calls in the last hour", "1 failure cause", "2 waiting on people"]);
  });

  it("compare each figure with its usual level", () => {
    const figures = buildFigures(homeFixture(), now);
    const byKey = Object.fromEntries(figures.map((f) => [f.key, f]));
    expect(byKey.calls.value).toBe("200");
    expect(byKey.calls.usual).toBe("100");
    expect(byKey.failed.value).toBe("21%");
    expect(byKey.failed.usual).toBe("1.0%");
    expect(byKey.failed.tone).toBe("bad");
    expect(byKey.p95.value).toBe("2.40 s");
    expect(byKey.p95.usual).toBe("900 ms");
    expect(byKey.p95.tone).toBe("warn");
    expect(byKey.running.value).toBe("8");
    expect(byKey.running.usual).toBe("3.0 avg");
    expect(byKey.waiting.value).toBe("2");
    expect(byKey.waiting.note).toBe("oldest 42 min · 1 agent awaiting tag approval");
    expect(byKey.spend).toBeUndefined();
  });

  it("leave usual levels unknown without a baseline and add spend only when recorded", () => {
    const figures = buildFigures(
      homeFixture({
        baseline: undefined,
        spend: { window_cost_usd: 3, window_tokens: 12000, window_rows: 4, baseline_rows: 0 },
      }),
      now,
    );
    const byKey = Object.fromEntries(figures.map((f) => [f.key, f]));
    expect(byKey.calls.usual).toBeUndefined();
    expect(byKey.failed.usual).toBeUndefined();
    expect(byKey.failed.tone).toBeUndefined();
    expect(byKey.spend.value).toBe("$3.00");
    expect(byKey.spend.usual).toBeUndefined();
    expect(byKey.spend.note).toBe("12,000 tokens");

    const unknownCost = buildFigures(
      homeFixture({ spend: { window_tokens: 0, window_rows: 2, baseline_rows: 0 } }),
      now,
    ).find((f) => f.key === "spend");
    expect(unknownCost?.value).toBe("—");
  });
});

describe("call map layout", () => {
  it("places callers left of the agents they call and colours links by outcome", () => {
    const layout = layoutCallMap(homeFixture());
    const node = (id: string) => layout.nodes.find((n) => n.id === id)!;
    expect(node(DIRECT_SOURCE).layer).toBe(0);
    expect(node("trigger:webhook").layer).toBe(0);
    expect(node("orchestrator").layer).toBe(1);
    expect(node("billing").layer).toBe(2);
    expect(node("payments").layer).toBe(2);
    expect(node("billing").selfCalls).toBe(5);
    expect(node("orchestrator").status).toBe("offline");

    const link = (id: string) => layout.links.find((l) => l.id === id)!;
    expect(link("orchestrator>billing").tone).toBe("bad");
    expect(link("orchestrator>payments").tone).toBe("warn");
    expect(link(`${DIRECT_SOURCE}>orchestrator`).tone).toBe("ok");
    expect(layout.links.some((l) => l.from === l.to)).toBe(false);
    // Failing links are drawn last so they sit on top.
    expect(layout.links[layout.links.length - 1].tone).toBe("bad");
    expect(layout.idle).toEqual(["spare"]);
  });

  it("keeps a cycle from pushing agents off to the right", () => {
    const layout = layoutCallMap(
      homeFixture({
        entries: [{ source: "", target: "a", calls: 1, failed: 0, held: 0 }],
        edges: [
          { caller: "a", target: "b", calls: 3, failed: 0, held: 0 },
          { caller: "b", target: "a", calls: 2, failed: 0, held: 0 },
        ],
        stopped_agents: [],
      }),
    );
    expect(layout.layers).toBe(3);
    const back = layout.links.find((l) => l.id === "b>a")!;
    expect(back.path.startsWith("M ")).toBe(true);
  });

  it("shows a stopped agent even when nothing called it", () => {
    const layout = layoutCallMap(
      homeFixture({
        entries: [],
        edges: [],
        stopped_agents: [{ id: "orchestrator", last_heartbeat: "2026-11-09T11:40:00Z", entry_points: [] }],
      }),
    );
    expect(layout.nodes.map((n) => n.id)).toEqual(["orchestrator"]);
    expect(layout.links).toEqual([]);
  });
});

describe("empty install", () => {
  it("is only an install with no agent and no calls", () => {
    const empty = homeFixture({
      agents: [],
      agent_counts: { registered: 0, running: 0, offline: 0, degraded: 0, starting: 0, pending_approval: 0, unknown: 0 },
      totals: { counts: { total: 0, succeeded: 0, failed: 0, cancelled: 0, terminal: 0 }, held: 0, running_now: 0 },
    });
    expect(isEmptyInstall(empty, 0)).toBe(true);
    expect(isEmptyInstall(empty, 3)).toBe(false);
    expect(isEmptyInstall(homeFixture(), 0)).toBe(false);
  });

  it("points the snippets at this control plane", () => {
    expect(connectSnippet("http://cp.test:8080")).toContain('agentfield_server="http://cp.test:8080"');
    expect(callSnippet("http://cp.test:8080")).toContain("http://cp.test:8080/api/v1/execute/my-first-agent.hello");
  });
});
