import { describe, expect, it } from "vitest";
import { statusTone } from "@/lib/theme";
import type { LifecycleStatus } from "@/types/agentfield";
import {
  getLifecycleTheme,
  isLifecycleOnline,
  normalizeLifecycleStatus,
} from "./lifecycle-status";

const LIFECYCLE_STATUSES: LifecycleStatus[] = [
  "starting",
  "ready",
  "running",
  "degraded",
  "stopped",
  "offline",
  "error",
  "unknown",
];

describe("normalizeLifecycleStatus", () => {
  it("returns each canonical lifecycle status unchanged", () => {
    for (const status of LIFECYCLE_STATUSES) {
      expect(normalizeLifecycleStatus(status)).toBe(status);
    }
  });

  it("normalizes empty and unknown inputs", () => {
    expect(normalizeLifecycleStatus(null)).toBe("unknown");
    expect(normalizeLifecycleStatus(undefined)).toBe("unknown");
    expect(normalizeLifecycleStatus("")).toBe("unknown");
    expect(normalizeLifecycleStatus("UNKNOWN_STRING")).toBe("unknown");
  });

  it("trims and lowercases incoming values", () => {
    expect(normalizeLifecycleStatus("  Ready  ")).toBe("ready");
  });

  it("supports lifecycle aliases", () => {
    expect(normalizeLifecycleStatus("up")).toBe("ready");
    expect(normalizeLifecycleStatus("down")).toBe("offline");
    expect(normalizeLifecycleStatus("unhealthy")).toBe("degraded");
  });
});

describe("getLifecycleTheme", () => {
  const expectations: Record<
    LifecycleStatus,
    {
      tone: keyof typeof statusTone;
      motion: "none" | "pulse" | "live";
      online: boolean;
      label: string;
      indicatorClass: string;
    }
  > = {
    starting: {
      tone: "info",
      motion: "pulse",
      online: true,
      label: "Starting",
      indicatorClass: statusTone.info.solidBg,
    },
    ready: {
      tone: "success",
      motion: "none",
      online: true,
      label: "Ready",
      indicatorClass: statusTone.success.solidBg,
    },
    running: {
      tone: "success",
      motion: "live",
      online: true,
      label: "Running",
      indicatorClass: statusTone.success.solidBg,
    },
    degraded: {
      tone: "warning",
      motion: "pulse",
      online: true,
      label: "Degraded",
      indicatorClass: statusTone.warning.solidBg,
    },
    stopped: {
      tone: "neutral",
      motion: "none",
      online: false,
      label: "Stopped",
      indicatorClass: statusTone.neutral.solidBg,
    },
    offline: {
      tone: "error",
      motion: "none",
      online: false,
      label: "Offline",
      indicatorClass: statusTone.error.solidBg,
    },
    error: {
      tone: "error",
      motion: "none",
      online: false,
      label: "Error",
      indicatorClass: statusTone.error.solidBg,
    },
    unknown: {
      tone: "neutral",
      motion: "none",
      online: false,
      label: "Unknown",
      indicatorClass: statusTone.neutral.solidBg,
    },
  };

  it("returns the expected theme for every lifecycle status", () => {
    for (const status of LIFECYCLE_STATUSES) {
      const theme = getLifecycleTheme(status);
      expect(theme.status).toBe(status);
      expect(theme.tone).toBe(expectations[status].tone);
      expect(theme.motion).toBe(expectations[status].motion);
      expect(theme.online).toBe(expectations[status].online);
      expect(theme.label).toBe(expectations[status].label);
      expect(theme.indicatorClass).toBe(expectations[status].indicatorClass);
    }
  });
});

describe("isLifecycleOnline", () => {
  it("treats reachable lifecycle states as online", () => {
    expect(isLifecycleOnline("ready")).toBe(true);
    expect(isLifecycleOnline("running")).toBe(true);
    expect(isLifecycleOnline("starting")).toBe(true);
    expect(isLifecycleOnline("degraded")).toBe(true);
  });

  it("treats non-reachable lifecycle states as offline", () => {
    expect(isLifecycleOnline("stopped")).toBe(false);
    expect(isLifecycleOnline("error")).toBe(false);
    expect(isLifecycleOnline("offline")).toBe(false);
    expect(isLifecycleOnline("unknown")).toBe(false);
  });
});
