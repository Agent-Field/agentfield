import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/api", () => ({
  getGlobalApiKey: vi.fn(() => "api-key-123"),
}));

const originalFetch = globalThis.fetch;

function buildTimelineResponse(cacheTimestamp = "2026-04-08T12:00:00Z") {
  return {
    timeline_data: [
      {
        timestamp: "2026-04-08T12:00:00Z",
        hour: "12:00",
        executions: 4,
        successful: 3,
        failed: 1,
        running: 0,
        success_rate: 75,
        avg_duration_ms: 25,
        total_duration_ms: 100,
      },
    ],
    cache_timestamp: cacheTimestamp,
    summary: {
      total_executions: 4,
      avg_success_rate: 75,
      total_errors: 1,
      peak_hour: "12:00",
      peak_executions: 4,
    },
  };
}

describe("executionTimelineService", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-08T12:05:00Z"));
    vi.resetModules();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("fetches timeline data, sends the API key header, and caches the result", async () => {
    const response = buildTimelineResponse();
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue(response),
    });

    const service = await import("@/services/executionTimelineService");
    const first = await service.getExecutionTimeline();
    const second = await service.getExecutionTimeline();

    expect(first).toEqual(response);
    expect(second).toBe(first);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect((options?.headers as Headers).get("X-API-Key")).toBe("api-key-123");

    const status = service.getTimelineCacheStatus();
    expect(status.hasData).toBe(true);
    expect(status.isValid).toBe(true);
    expect(status.ttl).toBe(300000);
  });

  it("forces refresh and clears cache state", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(buildTimelineResponse("2026-04-08T12:00:00Z")),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(buildTimelineResponse("2026-04-08T12:01:00Z")),
      });

    const service = await import("@/services/executionTimelineService");
    await service.getExecutionTimeline();
    const refreshed = await service.getExecutionTimeline(true);

    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    expect(refreshed.cache_timestamp).toBe("2026-04-08T12:01:00Z");

    service.clearTimelineCache();
    expect(service.getTimelineCacheStatus().hasData).toBe(false);
  });

  it("retries with backoff and eventually succeeds", async () => {
    globalThis.fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("first"))
      .mockRejectedValueOnce(new Error("second"))
      .mockResolvedValueOnce({
        ok: true,
        json: vi.fn().mockResolvedValue(buildTimelineResponse()),
      });

    const service = await import("@/services/executionTimelineService");
    const promise = service.getExecutionTimelineWithRetry(2, 100, true);
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.summary.total_executions).toBe(4);
    expect(globalThis.fetch).toHaveBeenCalledTimes(3);
  });

  it("throws API and timeout errors", async () => {
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: vi.fn().mockResolvedValue({ message: "backend error" }),
    });

    let service = await import("@/services/executionTimelineService");
    await expect(service.getExecutionTimeline(true)).rejects.toThrow("backend error");

    vi.resetModules();
    globalThis.fetch = vi.fn().mockRejectedValueOnce(
      Object.assign(new Error("aborted"), { name: "AbortError" })
    );

    service = await import("@/services/executionTimelineService");
    await expect(service.getExecutionTimeline(true)).rejects.toThrow("Request timeout after 10000ms");
  });

  it("reports freshness and cache age from response timestamps", async () => {
    const service = await import("@/services/executionTimelineService");
    const fresh = buildTimelineResponse("2026-04-08T12:04:30Z");
    const stale = buildTimelineResponse("2026-04-08T11:40:00Z");

    expect(service.isTimelineDataFresh(fresh, 60000)).toBe(true);
    expect(service.isTimelineDataFresh(stale, 60000)).toBe(false);
    expect(service.getTimelineCacheAge(fresh)).toBe(30000);
    expect(service.getTimelineCacheAge({ ...fresh, cache_timestamp: "" })).toBe(Infinity);
  });
});
