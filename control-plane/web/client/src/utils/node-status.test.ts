import { describe, expect, it } from "vitest";

import type { AgentNodeSummary, HealthStatus, LifecycleStatus } from "@/types/agentfield";
import { getNodeStatusPresentation, summarizeNodeStatuses } from "./node-status";

function createNode(
  lifecycle_status: LifecycleStatus,
  health_status: HealthStatus,
): AgentNodeSummary {
  return {
    id: `${lifecycle_status}-${health_status}`,
    base_url: "https://example.com",
    version: "1.0.0",
    team_id: "team-1",
    lifecycle_status,
    health_status,
    reasoner_count: 0,
    skill_count: 0,
  };
}

describe("getNodeStatusPresentation", () => {
  it("forces offline when node health is inactive", () => {
    const presentation = getNodeStatusPresentation("running", "inactive");

    expect(presentation.kind).toBe("offline");
    expect(presentation.label).toBe("Offline");
    expect(presentation.online).toBe(false);
    expect(presentation.shouldPulse).toBe(false);
    expect(presentation.theme.status).toBe("offline");
  });

  it("maps degraded health to error status", () => {
    const presentation = getNodeStatusPresentation("ready", "degraded");

    expect(presentation.kind).toBe("error");
    expect(presentation.label).toBe("Error");
    expect(presentation.online).toBe(false);
    expect(presentation.shouldPulse).toBe(false);
  });

  it("keeps explicitly degraded lifecycle states distinct", () => {
    const presentation = getNodeStatusPresentation("degraded", "ready");

    expect(presentation.kind).toBe("degraded");
    expect(presentation.label).toBe("Degraded");
    expect(presentation.online).toBe(true);
    expect(presentation.shouldPulse).toBe(true);
  });

  it("marks starting states as online and pulsing", () => {
    const presentation = getNodeStatusPresentation("starting", "starting");

    expect(presentation.kind).toBe("starting");
    expect(presentation.label).toBe("Starting");
    expect(presentation.online).toBe(true);
    expect(presentation.shouldPulse).toBe(true);
  });

  it("preserves running and ready lifecycle themes", () => {
    expect(getNodeStatusPresentation("running", "ready").kind).toBe("running");
    expect(getNodeStatusPresentation("running", "ready").online).toBe(true);
    expect(getNodeStatusPresentation("running", "ready").shouldPulse).toBe(true);

    expect(getNodeStatusPresentation("ready", "ready").kind).toBe("ready");
    expect(getNodeStatusPresentation("ready", "ready").online).toBe(true);
    expect(getNodeStatusPresentation("ready", "ready").shouldPulse).toBe(false);
  });

  it("returns unknown for missing statuses", () => {
    const presentation = getNodeStatusPresentation(null, null);

    expect(presentation.kind).toBe("unknown");
    expect(presentation.label).toBe("Unknown");
    expect(presentation.online).toBe(false);
    expect(presentation.shouldPulse).toBe(false);
  });

  it("treats stopped and offline lifecycles as offline", () => {
    expect(getNodeStatusPresentation("stopped", "ready").kind).toBe("offline");
    expect(getNodeStatusPresentation("offline", "ready").kind).toBe("offline");
  });

  it("treats lifecycle errors as errors", () => {
    const presentation = getNodeStatusPresentation("error", "ready");

    expect(presentation.kind).toBe("error");
    expect(presentation.label).toBe("Error");
  });
});

describe("summarizeNodeStatuses", () => {
  it("aggregates totals, online state, and readiness buckets", () => {
    const summary = summarizeNodeStatuses([
      createNode("ready", "ready"),
      createNode("running", "ready"),
      createNode("starting", "starting"),
      createNode("degraded", "ready"),
      createNode("ready", "degraded"),
      createNode("running", "inactive"),
      createNode("unknown", "unknown"),
    ]);

    expect(summary).toEqual({
      total: 7,
      online: 4,
      offline: 3,
      degraded: 1,
      starting: 1,
      ready: 2,
    });
  });

  it("returns zero counts for an empty list", () => {
    expect(summarizeNodeStatuses([])).toEqual({
      total: 0,
      online: 0,
      offline: 0,
      degraded: 0,
      starting: 0,
      ready: 0,
    });
  });
});
