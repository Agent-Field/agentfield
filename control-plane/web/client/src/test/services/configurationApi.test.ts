import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setGlobalApiKey } from "@/services/api";
import {
  ConfigurationApiError,
  deleteAgentEnvVar,
  getAgentPackages,
  getAgentStatus,
  getConfigurationSchema,
  getConfigurationStatusBadge,
  getRunningAgents,
  isAgentConfigured,
  isAgentPartiallyConfigured,
  patchAgentEnvFile,
  reconcileAgent,
  setAgentConfiguration,
  setAgentEnvFile,
  startAgent,
  stopAgent,
} from "@/services/configurationApi";

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

describe("configurationApi", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    setGlobalApiKey(null);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setGlobalApiKey(null);
    vi.restoreAllMocks();
  });

  it("issues env file mutations with auth headers and HTTP verbs", async () => {
    setGlobalApiKey("secret");
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, {
          agent_id: "agent-1",
          package_id: "pkg-1",
          variables: { TOKEN: "masked" },
          masked_keys: ["TOKEN"],
          file_exists: true,
        })
      )
      .mockResolvedValueOnce(mockResponse(200, {}))
      .mockResolvedValueOnce(mockResponse(200, {}))
      .mockResolvedValueOnce(mockResponse(200, {}));

    await setAgentEnvFile("agent-1", "pkg-1", { TOKEN: "abc" });
    await patchAgentEnvFile("agent-1", "pkg-1", { TOKEN: "def" });
    await deleteAgentEnvVar("agent-1", "pkg-1", "TOKEN");

    const putCall = vi.mocked(globalThis.fetch).mock.calls[0] as [string, RequestInit];
    const patchCall = vi.mocked(globalThis.fetch).mock.calls[1] as [string, RequestInit];
    const deleteCall = vi.mocked(globalThis.fetch).mock.calls[2] as [string, RequestInit];

    expect(putCall[0]).toContain("/agents/agent-1/env?packageId=pkg-1");
    expect(putCall[1].method).toBe("PUT");
    expect(new Headers(putCall[1].headers).get("X-API-Key")).toBe("secret");
    expect(patchCall[1].method).toBe("PATCH");
    expect(deleteCall[0]).toContain("/env/TOKEN?packageId=pkg-1");
    expect(deleteCall[1].method).toBe("DELETE");
  });

  it("parses JSON and plain-text API errors", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(500, { error: "schema unavailable" }, "Server Error")
      )
      .mockResolvedValueOnce(
        mockResponse(502, "gateway down", "Bad Gateway")
      );

    await expect(getConfigurationSchema("agent-1")).rejects.toEqual(
      new ConfigurationApiError("schema unavailable", 500)
    );
    await expect(getAgentStatus("agent-1")).rejects.toEqual(
      new ConfigurationApiError("gateway down", 502)
    );
  });

  it("builds package search URLs and posts configuration changes", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(mockResponse(200, []))
      .mockResolvedValueOnce(mockResponse(200, {}));

    await expect(getAgentPackages("planner")).resolves.toEqual([]);
    await setAgentConfiguration("agent-1", {
      package_id: "pkg-1",
      variables: {},
    } as never);

    const packagesUrl = vi.mocked(globalThis.fetch).mock.calls[0]?.[0] as string;
    expect(new URL(packagesUrl).searchParams.get("search")).toBe("planner");

    const configCall = vi.mocked(globalThis.fetch).mock.calls[1] as [string, RequestInit];
    expect(configCall[1].method).toBe("POST");
    expect(configCall[1].body).toBe(
      JSON.stringify({
        package_id: "pkg-1",
        variables: {},
      })
    );
  });

  it("handles lifecycle operations and timeouts", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(200, { id: "agent-1", status: "running" })
      )
      .mockResolvedValueOnce(mockResponse(200, {}))
      .mockResolvedValueOnce(mockResponse(200, { ok: true }))
      .mockResolvedValueOnce(mockResponse(200, [{ id: "agent-1", status: "running" }]))
      .mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }));

    await expect(startAgent("agent-1")).resolves.toMatchObject({ status: "running" });
    await expect(stopAgent("agent-1")).resolves.toBeUndefined();
    await expect(reconcileAgent("agent-1")).resolves.toEqual({ ok: true });
    await expect(getRunningAgents()).resolves.toEqual([{ id: "agent-1", status: "running" }]);

    await expect(startAgent("agent-2")).rejects.toEqual(
      new ConfigurationApiError("Request timeout after 5000ms", 408)
    );
  });

  it("exposes utility helpers for configuration and lifecycle badges", () => {
    expect(
      isAgentConfigured({ configuration_status: "configured" } as never)
    ).toBe(true);
    expect(
      isAgentPartiallyConfigured({ configuration_status: "partially_configured" } as never)
    ).toBe(true);

    expect(getConfigurationStatusBadge("configured")).toEqual({
      variant: "default",
      label: "Configured",
      color: "green",
    });
    expect(getConfigurationStatusBadge("partially_configured")).toEqual({
      variant: "secondary",
      label: "Partially Configured",
      color: "yellow",
    });
    expect(getConfigurationStatusBadge("not_configured")).toEqual({
      variant: "outline",
      label: "Not Configured",
      color: "gray",
    });
    expect(getConfigurationStatusBadge("unknown")).toEqual({
      variant: "outline",
      label: "Unknown",
      color: "gray",
    });
  });
});
