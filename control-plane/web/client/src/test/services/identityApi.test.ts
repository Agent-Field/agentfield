import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiState = vi.hoisted(() => ({
  getGlobalApiKey: vi.fn(),
}));

vi.mock("@/services/api", () => apiState);

const originalFetch = globalThis.fetch;

function buildJsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("identityApi", () => {
  beforeEach(() => {
    vi.resetModules();
    apiState.getGlobalApiKey.mockReset();
    apiState.getGlobalApiKey.mockReturnValue("identity-key");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("gets DID stats and applies the API key header", async () => {
    const payload = {
      total_agents: 5,
      total_reasoners: 10,
      total_skills: 4,
      total_dids: 19,
    };
    globalThis.fetch = vi.fn().mockResolvedValue(buildJsonResponse(payload));

    const service = await import("@/services/identityApi");
    await expect(service.getDIDStats()).resolves.toEqual(payload);

    const [url, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("/api/ui/v1/identity/dids/stats");
    expect((options?.headers as Headers).get("X-API-Key")).toBe("identity-key");
  });

  it("builds search and listing query strings with defaults and custom values", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        buildJsonResponse({
          results: [],
          total: 0,
          limit: 20,
          offset: 0,
          has_more: false,
        })
      )
      .mockResolvedValueOnce(
        buildJsonResponse({
          agents: [],
          total: 0,
          limit: 15,
          offset: 30,
          has_more: false,
        })
      )
      .mockResolvedValueOnce(
        buildJsonResponse({
          did: "did:agent:123",
          did_web: "did:web:agentfield.example",
          agent_node_id: "agent-123",
          status: "active",
          derivation_path: "/agents/agent-123",
          created_at: "2026-04-01T00:00:00Z",
          reasoner_count: 2,
          skill_count: 3,
        })
      )
      .mockResolvedValueOnce(
        buildJsonResponse({
          reasoners: [],
          skills: [],
          total_reasoners: 2,
          total_skills: 3,
        })
      );

    const service = await import("@/services/identityApi");

    await expect(service.searchDIDs("agent search")).resolves.toMatchObject({
      total: 0,
      limit: 20,
      offset: 0,
    });
    await expect(service.listAgents(15, 30)).resolves.toMatchObject({
      limit: 15,
      offset: 30,
    });
    await expect(service.getAgentDetails("agent-123", 7, 9)).resolves.toMatchObject({
      agent_node_id: "agent-123",
    });
    await expect(service.getAgentDIDs("agent-123", 8, 11)).resolves.toMatchObject({
      total_reasoners: 2,
      total_skills: 3,
    });

    const calls = vi.mocked(globalThis.fetch).mock.calls;
    expect(calls[0][0]).toBe(
      "/api/ui/v1/identity/dids/search?q=agent+search&type=all&limit=20&offset=0"
    );
    expect(calls[1][0]).toBe("/api/ui/v1/identity/agents?limit=15&offset=30");
    expect(calls[2][0]).toBe("/api/ui/v1/identity/agents/agent-123/details?limit=7&offset=9");
    expect(calls[3][0]).toBe("/api/ui/v1/identity/agents/agent-123/dids?limit=8&offset=11");
  });

  it("serializes all credential search filters and skips falsy optional values", async () => {
    const payload = {
      credentials: [],
      total: 0,
      limit: 99,
      offset: 3,
      has_more: false,
    };
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(buildJsonResponse(payload))
      .mockResolvedValueOnce(buildJsonResponse(payload));

    const { searchCredentials } = await import("@/services/identityApi");

    await expect(
      searchCredentials({
        workflow_id: "wf-1",
        session_id: "session-1",
        status: "verified",
        issuer_did: "did:issuer:1",
        agent_node_id: "agent-1",
        execution_id: "exec-1",
        caller_did: "did:caller:1",
        target_did: "did:target:1",
        query: "proof",
        start_time: "2026-04-01T00:00:00Z",
        end_time: "2026-04-02T00:00:00Z",
        limit: 99,
        offset: 3,
      })
    ).resolves.toEqual(payload);

    await expect(searchCredentials({ limit: 0, offset: 0, query: "" })).resolves.toEqual(payload);

    const [firstUrl] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(firstUrl).toBe(
      "/api/ui/v1/identity/credentials/search?workflow_id=wf-1&session_id=session-1&status=verified&issuer_did=did%3Aissuer%3A1&agent_node_id=agent-1&execution_id=exec-1&caller_did=did%3Acaller%3A1&target_did=did%3Atarget%3A1&query=proof&start_time=2026-04-01T00%3A00%3A00Z&end_time=2026-04-02T00%3A00%3A00Z&limit=99&offset=3"
    );

    const [secondUrl] = vi.mocked(globalThis.fetch).mock.calls[1];
    expect(secondUrl).toBe("/api/ui/v1/identity/credentials/search?");
  });

  it("uses response JSON messages and fallback messages for API failures", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        buildJsonResponse({ message: "DID lookup failed" }, { status: 404, statusText: "Not Found" })
      )
      .mockResolvedValueOnce({
        ok: false,
        status: 502,
        json: vi.fn().mockRejectedValue(new Error("bad json")),
      });

    const service = await import("@/services/identityApi");

    await expect(service.getDIDStats()).rejects.toThrow("DID lookup failed");
    await expect(service.listAgents()).rejects.toThrow("Request failed with status 502");
  });

  it("omits the API key header when no key is configured", async () => {
    apiState.getGlobalApiKey.mockReturnValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue(
      buildJsonResponse({
        results: [],
        total: 0,
        limit: 20,
        offset: 0,
        has_more: false,
      })
    );

    const { searchDIDs } = await import("@/services/identityApi");
    await searchDIDs("plain");

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect((options?.headers as Headers).get("X-API-Key")).toBeNull();
  });

  it("exports the compatibility default object", async () => {
    const service = await import("@/services/identityApi");

    expect(service.default.getDIDStats).toBe(service.getDIDStats);
    expect(service.default.searchDIDs).toBe(service.searchDIDs);
    expect(service.default.listAgents).toBe(service.listAgents);
    expect(service.default.getAgentDetails).toBe(service.getAgentDetails);
    expect(service.default.getAgentDIDs).toBe(service.getAgentDIDs);
    expect(service.default.searchCredentials).toBe(service.searchCredentials);
  });
});
