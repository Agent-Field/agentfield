import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiState = vi.hoisted(() => ({
  getGlobalApiKey: vi.fn(),
}));

vi.mock("@/services/api", () => apiState);

const originalFetch = globalThis.fetch;

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("ardApi", () => {
  beforeEach(() => {
    vi.resetModules();
    apiState.getGlobalApiKey.mockReset();
    apiState.getGlobalApiKey.mockReturnValue("ard-key");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("calls every ARD UI endpoint with the expected method, body, and API key", async () => {
    globalThis.fetch = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({ ok: true })));

    const service = await import("@/services/ardApi");
    await service.getARDDashboard();
    await service.updateARDSettings({ enabled: true, publish_enabled: false });
    await service.saveARDPublication({
      target_kind: "reasoner",
      node_id: "node-1",
      target_id: "review_contract",
      published: true,
      display_name: "Review Contract",
      description: "Review contract language.",
      tags: ["legal"],
      capabilities: ["ContractReview"],
      representative_queries: ["review this MSA", "find risky clauses"],
      artifact_type: "application/openapi+json",
      validation_status: "valid",
    });
    await service.searchExternalARD({
      query: { text: "review", filter: { tags: ["legal"] } },
      federation: "none",
      pageSize: 20,
    });
    await service.importARDEntry(
      {
        identifier: "urn:ai:vendor.example:agent:review",
        displayName: "Vendor Review",
        type: "application/a2a-agent-card+json",
        url: "https://vendor.example/agent-card.json",
      },
      "https://registry.example.com/api/v1/ard",
    );
    await service.saveARDBinding("ext/id", {
      external_entry_id: "ext/id",
      callable: true,
      local_target: "external.vendor.review",
      adapter: "a2a",
      timeout_ms: 30000,
      allowed_operations: ["call"],
      policy: "review-only",
    });
    await service.saveARDRegistries([
      { url: "https://registry.example.com/api/v1/ard", name: "Registry" },
    ]);

    const calls = vi.mocked(globalThis.fetch).mock.calls;
    expect(calls.map(([url]) => url)).toEqual([
      "/api/ui/v1/ard",
      "/api/ui/v1/ard/settings",
      "/api/ui/v1/ard/publications",
      "/api/ui/v1/ard/external/search",
      "/api/ui/v1/ard/imports",
      "/api/ui/v1/ard/imports/ext%2Fid/binding",
      "/api/ui/v1/ard/registries",
    ]);
    expect(calls.map(([, options]) => options?.method || "GET")).toEqual([
      "GET",
      "PUT",
      "PUT",
      "POST",
      "POST",
      "PUT",
      "PUT",
    ]);
    for (const [, options] of calls) {
      expect((options?.headers as Headers).get("Content-Type")).toBe("application/json");
      expect((options?.headers as Headers).get("X-API-Key")).toBe("ard-key");
    }
    expect(JSON.parse(calls[1][1]?.body as string)).toEqual({
      enabled: true,
      publish_enabled: false,
    });
    expect(JSON.parse(calls[4][1]?.body as string)).toEqual({
      source_registry: "https://registry.example.com/api/v1/ard",
      entry: expect.objectContaining({
        identifier: "urn:ai:vendor.example:agent:review",
      }),
    });
    expect(JSON.parse(calls[6][1]?.body as string)).toEqual({
      registries: [{ url: "https://registry.example.com/api/v1/ard", name: "Registry" }],
    });
  });

  it("surfaces API errors and omits the API key when none is configured", async () => {
    apiState.getGlobalApiKey.mockReturnValue(null);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: "ARD disabled" }, { status: 403 }))
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: vi.fn().mockRejectedValue(new Error("bad json")),
      });

    const service = await import("@/services/ardApi");

    await expect(service.getARDDashboard()).rejects.toThrow("ARD disabled");
    await expect(service.getARDDashboard()).rejects.toThrow("ARD request failed with 500");

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect((options?.headers as Headers).get("X-API-Key")).toBeNull();
  });
});
