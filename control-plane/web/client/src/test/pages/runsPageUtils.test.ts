import { describe, expect, it } from "vitest";

import {
  PREVIEW_JSON_MAX,
  formatAbsoluteStarted,
  formatDuration,
  formatPreviewJson,
  formatRelativeStarted,
  getPaginationPages,
  hasMeaningfulPayload,
  shortRunIdDisplay,
} from "@/pages/runsPageUtils";

describe("runsPageUtils", () => {
  it("formats run ids, timestamps, durations, and payload previews", () => {
    expect(shortRunIdDisplay("run-1234")).toBe("…1234");
    expect(shortRunIdDisplay("abc", 1)).toBe("abc");
    expect(shortRunIdDisplay("run-abcdefgh", 6)).toBe("…cdefgh");

    expect(formatAbsoluteStarted("not-a-date")).toBe("—");
    expect(formatAbsoluteStarted("2026-04-08T16:00:00Z")).toContain("2026");

    const started = Date.UTC(2026, 3, 8, 16, 0, 0);
    expect(formatRelativeStarted(started, started+5_000, true)).toBe("just now");
    expect(formatRelativeStarted(started, started+42_000, true)).toBe("42s ago");
    expect(formatRelativeStarted(started, started+125_000, true)).toBe("2m 5s ago");
    expect(formatRelativeStarted(started, started+3_660_000, true)).toBe("1h 1m ago");
    expect(formatRelativeStarted(started, started+9_000, false)).toBe("just now");
    expect(formatRelativeStarted(started, started+75_000, false)).toBe("1 minute ago");
    expect(formatRelativeStarted(started, started+3_600_000, false)).toBe("1 hour ago");
    expect(formatRelativeStarted(started, started+8*24*60*60*1000, false)).toBe("last week");

    expect(formatDuration(undefined)).toBe("—");
    expect(formatDuration(undefined, true)).toBe("—");
    expect(formatDuration(950, true)).toBe("950ms");
    expect(formatDuration(1_500, true)).toBe("1.5s");
    expect(formatDuration(125_000, true)).toBe("2m 5s");
    expect(formatDuration(3 * 60 * 60 * 1000, true)).toBe("3h");
    expect(formatDuration((26 * 60 * 60 * 1000), true)).toBe("1d 2h");

    expect(hasMeaningfulPayload(null)).toBe(false);
    expect(hasMeaningfulPayload(undefined)).toBe(false);
    expect(hasMeaningfulPayload("   ")).toBe(false);
    expect(hasMeaningfulPayload([])).toBe(false);
    expect(hasMeaningfulPayload({})).toBe(false);
    expect(hasMeaningfulPayload("text")).toBe(true);
    expect(hasMeaningfulPayload([1])).toBe(true);
    expect(hasMeaningfulPayload({ ok: true })).toBe(true);
    expect(hasMeaningfulPayload(0)).toBe(true);

    expect(PREVIEW_JSON_MAX).toBe(10_000);
    expect(formatPreviewJson({ alpha: 1, beta: [1, 2] })).toContain('"alpha": 1');
    expect(formatPreviewJson("x".repeat(12_000)).length).toBeLessThanOrEqual(PREVIEW_JSON_MAX + 64);
  });

  it("builds compact pagination ranges with ellipses", () => {
    expect(getPaginationPages(1, 0)).toEqual([]);
    expect(getPaginationPages(3, 5)).toEqual([1, 2, 3, 4, 5]);
    expect(getPaginationPages(1, 10)).toEqual([1, 2, "ellipsis", 10]);
    expect(getPaginationPages(5, 10)).toEqual([1, "ellipsis", 4, 5, 6, "ellipsis", 10]);
    expect(getPaginationPages(10, 10)).toEqual([1, "ellipsis", 9, 10]);
  });
});
