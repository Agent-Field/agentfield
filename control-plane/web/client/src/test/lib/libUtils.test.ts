import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/api", () => ({
  getGlobalAdminToken: vi.fn(),
  getGlobalApiKey: vi.fn(),
}));

import { getGlobalAdminToken, getGlobalApiKey } from "@/services/api";
import { countPendingAgentTags, getAgentTagRowStatus } from "@/lib/governanceUtils";
import { areGovernanceAdminRoutesAvailable } from "@/lib/governanceProbe";
import { queryClient } from "@/lib/query-client";
import { getStatusBadgeClasses, getStatusTone, statusTone } from "@/lib/theme";
import { getNextTimeRange, TIME_RANGE_OPTIONS } from "@/lib/timeRanges";
import { cn } from "@/lib/utils";

describe("lib helpers", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.clearAllMocks();
  });

  it("class-merges utility strings", () => {
    expect(cn("px-2 py-1", { hidden: false, block: true }, "px-4")).toBe(
      "py-1 block px-4"
    );
  });

  it("exposes theme tokens and composed badge classes", () => {
    expect(getStatusTone("success")).toBe(statusTone.success);
    const classes = getStatusBadgeClasses("warning");
    expect(classes).toContain("inline-flex items-center");
    expect(classes).toContain("bg-status-warning/10");
    expect(classes).toContain("text-status-warning");
  });

  it("walks time ranges in order and exposes the options list", () => {
    expect(TIME_RANGE_OPTIONS).toEqual(["1h", "24h", "7d", "30d", "all"]);
    expect(getNextTimeRange("")).toBe("1h");
    expect(getNextTimeRange("1h")).toBe("24h");
    expect(getNextTimeRange("all")).toBeUndefined();
    expect(getNextTimeRange("weird")).toBe("all");
  });

  it("classifies governance agent rows and counts pending approvals", () => {
    expect(
      getAgentTagRowStatus({
        agent_id: "a",
        proposed_tags: [],
        approved_tags: [],
        lifecycle_status: "pending_approval",
        registered_at: "2026-01-01T00:00:00Z",
      })
    ).toBe("pending_approval");
    expect(
      getAgentTagRowStatus({
        agent_id: "b",
        proposed_tags: [],
        approved_tags: [],
        lifecycle_status: "offline",
        registered_at: "2026-01-01T00:00:00Z",
      })
    ).toBe("active");
    expect(
      getAgentTagRowStatus({
        agent_id: "c",
        proposed_tags: [],
        approved_tags: [],
        lifecycle_status: "retired",
        registered_at: "2026-01-01T00:00:00Z",
      })
    ).toBe("other");
    expect(
      countPendingAgentTags([
        {
          agent_id: "a",
          proposed_tags: [],
          approved_tags: [],
          lifecycle_status: "pending_approval",
          registered_at: "2026-01-01T00:00:00Z",
        },
        {
          agent_id: "b",
          proposed_tags: [],
          approved_tags: [],
          lifecycle_status: "active",
          registered_at: "2026-01-01T00:00:00Z",
        },
      ])
    ).toBe(1);
  });

  it("uses auth headers when probing governance admin routes", async () => {
    vi.mocked(getGlobalApiKey).mockReturnValue("api-key");
    vi.mocked(getGlobalAdminToken).mockReturnValue("admin-token");
    globalThis.fetch = vi.fn().mockResolvedValue({ status: 200 });

    await expect(areGovernanceAdminRoutesAvailable()).resolves.toBe(true);
    expect(globalThis.fetch).toHaveBeenCalledWith("/api/v1/admin/policies", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "X-Api-Key": "api-key",
        "X-Admin-Token": "admin-token",
      },
    });
  });

  it("treats a 404 governance probe as unavailable", async () => {
    vi.mocked(getGlobalApiKey).mockReturnValue(null);
    vi.mocked(getGlobalAdminToken).mockReturnValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue({ status: 404 });

    await expect(areGovernanceAdminRoutesAvailable()).resolves.toBe(false);
  });

  it("configures the shared query client defaults", () => {
    expect(queryClient.getDefaultOptions().queries).toMatchObject({
      staleTime: 30_000,
      gcTime: 300_000,
      retry: 1,
      refetchOnWindowFocus: false,
    });
  });
});
