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

describe("observabilityWebhookApi", () => {
  beforeEach(() => {
    vi.resetModules();
    apiState.getGlobalApiKey.mockReset();
    apiState.getGlobalApiKey.mockReturnValue("obs-key");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("gets the current webhook config with auth and JSON headers", async () => {
    const payload = {
      configured: true,
      config: {
        id: "cfg-1",
        url: "https://example.com/webhook",
        has_secret: true,
        headers: { "X-Test": "one" },
        enabled: true,
        created_at: "2026-04-01T00:00:00Z",
        updated_at: "2026-04-01T00:00:00Z",
      },
    };
    globalThis.fetch = vi.fn().mockResolvedValue(buildJsonResponse(payload));

    const service = await import("@/services/observabilityWebhookApi");
    await expect(service.getObservabilityWebhook()).resolves.toEqual(payload);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      "/api/v1/settings/observability-webhook",
      expect.objectContaining({
        signal: expect.any(AbortSignal),
        headers: expect.any(Headers),
      })
    );

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    const headers = options?.headers as Headers;
    expect(headers.get("X-API-Key")).toBe("obs-key");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("sets the webhook config with a POST body", async () => {
    const request = {
      url: "https://example.com/new-webhook",
      secret: "super-secret",
      headers: { Authorization: "Bearer token" },
      enabled: false,
    };
    const payload = {
      success: true,
      message: "saved",
      config: {
        id: "cfg-2",
        url: request.url,
        has_secret: true,
        headers: request.headers,
        enabled: false,
        created_at: "2026-04-01T00:00:00Z",
        updated_at: "2026-04-02T00:00:00Z",
      },
    };
    globalThis.fetch = vi.fn().mockResolvedValue(buildJsonResponse(payload));

    const { setObservabilityWebhook } = await import("@/services/observabilityWebhookApi");
    await expect(setObservabilityWebhook(request)).resolves.toEqual(payload);

    const [url, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("/api/v1/settings/observability-webhook");
    expect(options?.method).toBe("POST");
    expect(options?.body).toBe(JSON.stringify(request));
  });

  it("deletes the webhook config", async () => {
    const payload = { success: true, message: "deleted" };
    globalThis.fetch = vi.fn().mockResolvedValue(buildJsonResponse(payload));

    const { deleteObservabilityWebhook } = await import("@/services/observabilityWebhookApi");
    await expect(deleteObservabilityWebhook()).resolves.toEqual(payload);

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(options?.method).toBe("DELETE");
  });

  it("gets webhook status, dead letter queue, redrive, and clear responses", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        buildJsonResponse({
          enabled: true,
          webhook_url: "https://example.com/webhook",
          queue_depth: 2,
          events_forwarded: 10,
          events_dropped: 1,
          dead_letter_count: 2,
          last_forwarded_at: "2026-04-03T00:00:00Z",
          last_error: "none",
        })
      )
      .mockResolvedValueOnce(
        buildJsonResponse({
          entries: [
            {
              id: 7,
              event_type: "execution.completed",
              event_source: "agentfield",
              event_timestamp: "2026-04-03T00:00:00Z",
              payload: "{}",
              error_message: "network",
              retry_count: 3,
              created_at: "2026-04-03T00:00:00Z",
            },
          ],
          total_count: 1,
        })
      )
      .mockResolvedValueOnce(
        buildJsonResponse({
          success: true,
          message: "redrive finished",
          processed: 3,
          failed: 1,
        })
      )
      .mockResolvedValueOnce(buildJsonResponse({ success: true, message: "cleared" }));

    const service = await import("@/services/observabilityWebhookApi");

    await expect(service.getObservabilityWebhookStatus()).resolves.toMatchObject({
      queue_depth: 2,
      dead_letter_count: 2,
    });
    await expect(service.getDeadLetterQueue(25, 50)).resolves.toMatchObject({
      total_count: 1,
    });
    await expect(service.redriveDeadLetterQueue()).resolves.toMatchObject({
      processed: 3,
      failed: 1,
    });
    await expect(service.clearDeadLetterQueue()).resolves.toEqual({
      success: true,
      message: "cleared",
    });

    const calls = vi.mocked(globalThis.fetch).mock.calls;
    expect(calls[0][0]).toBe("/api/v1/settings/observability-webhook/status");
    expect(calls[1][0]).toBe("/api/v1/settings/observability-webhook/dlq?limit=25&offset=50");
    expect(calls[2][0]).toBe("/api/v1/settings/observability-webhook/redrive");
    expect(calls[3][0]).toBe("/api/v1/settings/observability-webhook/dlq");
    expect(calls[2][1]?.method).toBe("POST");
    expect(calls[3][1]?.method).toBe("DELETE");
  });

  it("omits the API key header when no key is set", async () => {
    apiState.getGlobalApiKey.mockReturnValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue(buildJsonResponse({ configured: false }));

    const { getObservabilityWebhook } = await import("@/services/observabilityWebhookApi");
    await getObservabilityWebhook();

    const [, options] = vi.mocked(globalThis.fetch).mock.calls[0];
    const headers = options?.headers as Headers;
    expect(headers.get("X-API-Key")).toBeNull();
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("surfaces JSON error payloads and status on API failures", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        buildJsonResponse(
          { error: "invalid webhook" },
          { status: 400, statusText: "Bad Request" }
        )
      )
      .mockResolvedValueOnce(
        new Response("plain failure", { status: 500, statusText: "Server Error" })
      );

    const service = await import("@/services/observabilityWebhookApi");

    await expect(service.setObservabilityWebhook({ url: "https://bad.example" })).rejects.toMatchObject({
      name: "ObservabilityWebhookApiError",
      message: "invalid webhook",
      status: 400,
    });

    await expect(service.deleteObservabilityWebhook()).rejects.toMatchObject({
      name: "ObservabilityWebhookApiError",
      message: "plain failure",
      status: 500,
    });
  });

  it("converts aborts into timeout errors and leaves non-abort fetch errors untouched", async () => {
    globalThis.fetch = vi
      .fn()
      .mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }))
      .mockRejectedValueOnce(new Error("network exploded"));

    const service = await import("@/services/observabilityWebhookApi");

    await expect(service.redriveDeadLetterQueue()).rejects.toMatchObject({
      name: "ObservabilityWebhookApiError",
      message: "Request timeout after 60000ms",
      status: 408,
    });
    await expect(service.getObservabilityWebhook()).rejects.toThrow("network exploded");
  });
});
