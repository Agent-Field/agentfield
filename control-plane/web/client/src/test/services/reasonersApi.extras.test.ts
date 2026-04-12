import { afterEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import { reasonersApi } from "@/services/reasonersApi";

function mockResponse(status: number, body: unknown, statusText = "OK") {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(
      typeof body === "string" ? body : JSON.stringify(body)
    ),
  } as unknown as Response;
}

describe("reasonersApi extras", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it("skips offset zero, encodes reasoner ids, and omits auth headers when unset", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          reasoners: [],
          total: 0,
          online_count: 0,
          offline_count: 0,
          nodes_count: 0,
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          reasoner_id: "team/core planner",
          name: "Planner",
          node_id: "node-1",
          node_status: "active",
          input_schema: {},
          output_schema: {},
          memory_config: { auto_inject: [], memory_retention: "1h", cache_results: false },
          last_updated: "2026-04-08T10:00:00Z",
        })
      );

    await expect(
      reasonersApi.getAllReasoners({
        status: "all",
        search: "plan + act",
        limit: 5,
        offset: 0,
      })
    ).resolves.toMatchObject({ total: 0 });
    await expect(reasonersApi.getReasonerDetails("team/core planner")).resolves.toMatchObject({
      reasoner_id: "team/core planner",
    });

    const [listUrl, listInit] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    const parsedList = new URL(listUrl, "http://localhost");
    expect(parsedList.pathname).toBe("/api/ui/v1/reasoners/all");
    expect(parsedList.searchParams.get("status")).toBeNull();
    expect(parsedList.searchParams.get("search")).toBe("plan + act");
    expect(parsedList.searchParams.get("limit")).toBe("5");
    expect(parsedList.searchParams.get("offset")).toBeNull();
    expect(new Headers(listInit.headers).get("X-API-Key")).toBeNull();

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe(
      "/api/ui/v1/reasoners/team%2Fcore%20planner/details"
    );
  });

  it("surfaces detailed execute and execution-status errors", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockResponse(502, "upstream failed", "Bad Gateway"))
      .mockResolvedValueOnce(mockResponse(404, { message: "missing" }, "Not Found"))
      .mockResolvedValueOnce(mockResponse(200, { status: "running" }));

    await expect(
      reasonersApi.executeReasoner("planner", { input: { prompt: "hello" } })
    ).rejects.toThrow("Failed to execute reasoner: Bad Gateway - upstream failed");
    await expect(reasonersApi.getExecutionStatus("exec-404")).rejects.toThrow("Execution not found");
    await expect(reasonersApi.getExecutionStatus("exec-bad")).rejects.toThrow(
      "Invalid execution status response format from server"
    );
  });

  it("posts async execution and template saves with auth headers", async () => {
    setGlobalApiKey("secret");
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          execution_id: "exec-1",
          status: "pending",
          message: "queued",
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          id: "tpl-1",
          name: "Starter",
          input: { prompt: "hi" },
          created_at: "2026-04-08T10:00:00Z",
        })
      );

    await expect(
      reasonersApi.executeReasonerAsync("planner", { input: { prompt: "hello" } })
    ).resolves.toMatchObject({ execution_id: "exec-1" });
    await expect(
      reasonersApi.saveExecutionTemplate("planner", { name: "Starter", input: { prompt: "hi" } })
    ).resolves.toMatchObject({ id: "tpl-1" });

    const [, asyncInit] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    expect(asyncInit.method).toBe("POST");
    expect(new Headers(asyncInit.headers).get("X-API-Key")).toBe("secret");
    expect(new Headers(asyncInit.headers).get("Content-Type")).toBe("application/json");

    const [, templateInit] = vi.mocked(globalThis.fetch).mock.calls[1] as [string, RequestInit];
    expect(templateInit.method).toBe("POST");
    expect(templateInit.body).toBe(JSON.stringify({ name: "Starter", input: { prompt: "hi" } }));
    expect(new Headers(templateInit.headers).get("X-API-Key")).toBe("secret");
  });
});
