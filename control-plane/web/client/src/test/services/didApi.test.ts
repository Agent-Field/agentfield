import { afterEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import {
  copyDIDToClipboard,
  formatDIDForDisplay,
  getAgentDIDInfo,
  getDIDDocument,
  getDIDIdentifier,
  getDIDMethod,
  getDIDStatusSummary,
  getDIDSystemStatus,
  isValidDID,
  listAgentDIDs,
  registerAgentDID,
  resolveDID,
} from "@/services/didApi";

function mockResponse(status: number, body: unknown, statusText = "OK") {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

describe("didApi", () => {
  const originalFetch = globalThis.fetch;
  const originalClipboard = navigator.clipboard;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: originalClipboard,
    });
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it("registers DIDs with auth and JSON headers", async () => {
    setGlobalApiKey("secret");
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponse(200, {
        node_id: "node-1",
        did_document: { id: "did:key:abc" },
        registration_status: "registered",
        created_reasoner_dids: ["did:key:abc#reasoner"],
        created_skill_dids: ["did:key:abc#skill"],
      })
    );

    const request = {
      node_id: "node-1",
      node_name: "Node One",
      reasoners: ["planner"],
      skills: ["search"],
    } as any;

    await expect(registerAgentDID(request)).resolves.toMatchObject({
      registration_status: "registered",
    });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/ui/v1/did/register");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify(request));
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(new Headers(init.headers).get("X-API-Key")).toBe("secret");
  });

  it("encodes DIDs for resolve and document endpoints", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          did: "did:key:abc/123",
          public_key_jwk: {},
          component_type: "reasoner",
          derivation_path: "m/0",
        })
      )
      .mockResolvedValueOnce(
        mockResponse(200, {
          "@context": ["https://www.w3.org/ns/did/v1"],
          id: "did:key:abc/123",
          verificationMethod: [],
          authentication: [],
          assertionMethod: [],
          service: [],
        })
      );

    await expect(resolveDID("did:key:abc/123")).resolves.toMatchObject({
      component_type: "reasoner",
    });
    await expect(getDIDDocument("did:key:abc/123")).resolves.toMatchObject({
      id: "did:key:abc/123",
    });

    expect(vi.mocked(globalThis.fetch).mock.calls[0]?.[0]).toBe(
      "/api/ui/v1/did/resolve/did%3Akey%3Aabc%2F123"
    );
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe(
      "/api/ui/v1/did/document/did%3Akey%3Aabc%2F123"
    );
  });

  it("surfaces JSON error envelopes and falls back for status summaries", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockResponse(404, { message: "agent did missing" }, "Not Found"))
      .mockResolvedValueOnce(mockResponse(500, { message: "unavailable" }))
      .mockResolvedValueOnce(
        mockResponse(200, {
          node_id: "node-1",
          did: "did:key:abc",
          public_key_jwk: {},
          status: "active",
          registered_at: "2026-04-08T10:00:00Z",
          reasoners: { planner: { did: "did:key:abc#planner" } },
          skills: { search: { did: "did:key:abc#search" }, lookup: { did: "did:key:abc#lookup" } },
        })
      );

    await expect(getAgentDIDInfo("node-404")).rejects.toThrow("agent did missing");
    await expect(getDIDStatusSummary("node-down")).resolves.toEqual({
      has_did: false,
      did_status: "inactive",
      reasoner_count: 0,
      skill_count: 0,
      last_updated: "",
    });
    await expect(getDIDStatusSummary("node-1")).resolves.toEqual({
      has_did: true,
      did_status: "active",
      reasoner_count: 1,
      skill_count: 2,
      last_updated: "2026-04-08T10:00:00Z",
    });
  });

  it("lists DIDs with encoded filters and handles empty results", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          agent_dids: ["did:key:abc", "did:key:def"],
        })
      )
      .mockResolvedValueOnce(mockResponse(200, { agent_dids: [] }));

    await expect(
      listAgentDIDs({
        status: "active",
        search: "node one/two",
        limit: 2,
        cursor: "",
      } as any)
    ).resolves.toEqual(["did:key:abc", "did:key:def"]);
    await expect(listAgentDIDs()).resolves.toEqual([]);

    const parsed = new URL(
      vi.mocked(globalThis.fetch).mock.calls[0]?.[0] as string,
      "http://localhost"
    );
    expect(parsed.pathname).toBe("/api/ui/v1/did/agents");
    expect(parsed.searchParams.get("status")).toBe("active");
    expect(parsed.searchParams.get("search")).toBe("node one/two");
    expect(parsed.searchParams.get("limit")).toBe("2");
    expect(parsed.searchParams.get("cursor")).toBeNull();
  });

  it("fetches DID system status and supports clipboard utilities", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponse(200, {
        status: "ok",
        message: "ready",
        timestamp: "2026-04-08T10:00:00Z",
      })
    );

    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    await expect(getDIDSystemStatus()).resolves.toMatchObject({ status: "ok" });
    await expect(copyDIDToClipboard("did:key:abc")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("did:key:abc");

    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    writeText.mockRejectedValueOnce(new Error("denied"));
    await expect(copyDIDToClipboard("did:key:def")).resolves.toBe(false);
    expect(errorSpy).toHaveBeenCalled();
  });

  it("formats and parses DID helper values", () => {
    expect(formatDIDForDisplay("did:key:short", 30)).toBe("did:key:short");
    expect(formatDIDForDisplay("did:key:abcdefghijklmnopqrstuvwxyz", 16)).toBe("did:ke...uvwxyz");
    expect(isValidDID("did:key:agent_01")).toBe(true);
    expect(isValidDID("not-a-did")).toBe(false);
    expect(getDIDMethod("did:web:agentfield.dev")).toBe("web");
    expect(getDIDMethod("bad")).toBeNull();
    expect(getDIDIdentifier("did:web:agentfield.dev:planner")).toBe("agentfield.dev:planner");
    expect(getDIDIdentifier("did:key")).toBeNull();
  });
});
