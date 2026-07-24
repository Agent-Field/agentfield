import { describe, it, expect } from "vitest";

import { getExecutionErrorCategoryMeta } from "./executionErrorCategory";

describe("getExecutionErrorCategoryMeta", () => {
  it("returns null for nullish input", () => {
    expect(getExecutionErrorCategoryMeta(null)).toBeNull();
    expect(getExecutionErrorCategoryMeta(undefined)).toBeNull();
    expect(getExecutionErrorCategoryMeta("")).toBeNull();
  });

  it("returns the canonical meta for a known category", () => {
    const meta = getExecutionErrorCategoryMeta("agent_timeout");
    expect(meta).not.toBeNull();
    expect(meta?.label).toBe("Agent timeout");
    expect(meta?.tooltip).toBe(meta?.description);
  });

  it("uses a short label for an unknown category whose raw form is short", () => {
    const meta = getExecutionErrorCategoryMeta("custom_thing");
    expect(meta?.label).toBe("custom thing");
    expect(meta?.tooltip).toBe("custom_thing");
  });

  it("collapses a long, free-form category to a fixed-length fallback label", () => {
    const longRaw =
      "Agent Restart Orphaned: Tier2-Test Re-Registered With New Instance (was 38ffec8279554ffa97aea6f0ab69a53c)";
    const meta = getExecutionErrorCategoryMeta(longRaw);
    expect(meta).not.toBeNull();
    // Label is bounded so the badge stays single-line in the Runs table.
    expect(meta!.label.length).toBeLessThanOrEqual(32);
    // Original string survives on the tooltip so context isn't lost on hover.
    expect(meta!.tooltip).toBe(longRaw);
  });
});
