import { describe, expect, it } from "vitest";

import { getErrorMessage } from "@/services/errorMessage";

describe("getErrorMessage", () => {
  it("extracts a nested backend error message", () => {
    expect(
      getErrorMessage(
        {
          error: {
            code: "session_start_failed",
            message: "No active node can start this session",
          },
        },
        "fallback",
      ),
    ).toBe("No active node can start this session");
  });

  it("joins FastAPI validation messages", () => {
    expect(
      getErrorMessage(
        {
          detail: [
            { loc: ["body", "target"], msg: "Field required" },
            { loc: ["body", "model"], msg: "Unsupported model" },
          ],
        },
        "fallback",
      ),
    ).toBe("Field required; Unsupported model");
  });

  it("uses the fallback for unknown structured errors instead of coercing them", () => {
    expect(getErrorMessage({ code: "node_offline" }, "fallback")).toBe("fallback");
  });

  it("uses the fallback for empty and circular values", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;

    expect(getErrorMessage({}, "fallback")).toBe("fallback");
    expect(getErrorMessage(circular, "fallback")).toBe("fallback");
  });
});
