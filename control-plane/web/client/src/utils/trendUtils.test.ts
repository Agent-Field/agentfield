import { describe, expect, it } from "vitest";

import {
  calculateTrend,
  formatDeltaWithArrow,
  getTrendColorClass,
  normalizeForSparkline,
} from "./trendUtils";

describe("calculateTrend", () => {
  it("calculates upward trends and positive polarity colors", () => {
    expect(calculateTrend(120, 100, "up-is-good")).toEqual({
      direction: "up",
      percentage: 20,
      absoluteDelta: 20,
      displayText: "+20.0%",
      color: "success",
    });
  });

  it("calculates downward trends and inverts color for down-is-good", () => {
    expect(calculateTrend(60, 100, "down-is-good")).toEqual({
      direction: "down",
      percentage: -40,
      absoluteDelta: -40,
      displayText: "-40.0%",
      color: "success",
    });
  });

  it("treats tiny deltas as flat", () => {
    expect(calculateTrend(100.05, 100, "up-is-good")).toEqual({
      direction: "flat",
      percentage: 0.04999999999999716,
      absoluteDelta: 0.04999999999999716,
      displayText: "—",
      color: "muted",
    });
  });

  it("handles previous zero and neutral polarity", () => {
    expect(calculateTrend(5, 0)).toEqual({
      direction: "up",
      percentage: 100,
      absoluteDelta: 5,
      displayText: "+100.0%",
      color: "muted",
    });
  });

  it("returns flat when both values are zero", () => {
    expect(calculateTrend(0, 0, "down-is-good")).toEqual({
      direction: "flat",
      percentage: 0,
      absoluteDelta: 0,
      displayText: "—",
      color: "muted",
    });
  });

  it("marks opposite-direction changes as destructive", () => {
    expect(calculateTrend(80, 100, "up-is-good").color).toBe("destructive");
    expect(calculateTrend(120, 100, "down-is-good").color).toBe("destructive");
  });
});

describe("formatDeltaWithArrow", () => {
  it("formats arrows for non-flat deltas and hides flat ones", () => {
    expect(formatDeltaWithArrow(calculateTrend(120, 100))).toBe("↑ +20.0%");
    expect(formatDeltaWithArrow(calculateTrend(80, 100))).toBe("↓ -20.0%");
    expect(formatDeltaWithArrow(calculateTrend(0, 0))).toBe("—");
  });
});

describe("getTrendColorClass", () => {
  it("maps trend colors to utility classes", () => {
    expect(getTrendColorClass("success")).toBe("text-emerald-500");
    expect(getTrendColorClass("destructive")).toBe("text-destructive");
    expect(getTrendColorClass("muted")).toBe("text-muted-foreground");
  });
});

describe("normalizeForSparkline", () => {
  it("returns an empty array when given no data", () => {
    expect(normalizeForSparkline([])).toEqual([]);
  });

  it("normalizes a varied series to the 0-1 range", () => {
    expect(normalizeForSparkline([10, 20, 30])).toEqual([0, 0.5, 1]);
    expect(normalizeForSparkline([-5, 0, 5])).toEqual([0, 0.5, 1]);
  });

  it("returns a flat midline when every value is identical", () => {
    expect(normalizeForSparkline([7, 7, 7])).toEqual([0.5, 0.5, 0.5]);
  });
});
