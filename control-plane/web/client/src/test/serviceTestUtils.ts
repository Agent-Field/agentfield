import { vi } from "vitest";

export function mockJsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(typeof body === "string" ? body : JSON.stringify(body)),
    blob: vi.fn().mockResolvedValue(new Blob([JSON.stringify(body)], { type: "application/json" })),
    body: null,
  } as unknown as Response;
}

export function mockTextResponse(status: number, text: string, jsonBody?: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockImplementation(() =>
      jsonBody !== undefined ? Promise.resolve(jsonBody) : Promise.reject(new Error("no json body"))
    ),
    text: vi.fn().mockResolvedValue(text),
    body: null,
  } as unknown as Response;
}

export function installEventSourceMock() {
  const OriginalEventSource = globalThis.EventSource;
  const instances: Array<{ url: string; close: ReturnType<typeof vi.fn> }> = [];

  class MockEventSource {
    url: string;
    close = vi.fn();

    constructor(url: string | URL) {
      this.url = String(url);
      instances.push({ url: this.url, close: this.close });
    }
  }

  // @ts-expect-error test double
  globalThis.EventSource = MockEventSource;

  return {
    instances,
    restore() {
      globalThis.EventSource = OriginalEventSource;
    },
  };
}

export function installAnchorMock() {
  const originalCreateElement = document.createElement.bind(document);
  const anchor = originalCreateElement("a");
  const click = vi.fn();
  anchor.click = click;

  document.createElement = vi.fn((tagName: string) =>
    tagName === "a" ? anchor : originalCreateElement(tagName)
  ) as typeof document.createElement;

  return {
    anchor,
    click,
    restore() {
      document.createElement = originalCreateElement;
    },
  };
}

export function textStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}
