import { describe, expect, it } from "vitest";
import { safeExternalUrl } from "./safeExternalUrl";

describe("safeExternalUrl", () => {
    it.each([
        "javascript:alert(1)",
        "data:text/html,x",
        "file:///tmp/x",
        "mailto:a@b.c",
        "/relative",
        "https://user:pass@example.com",
        "",
        null,
    ])("rejects %s", (value) => expect(safeExternalUrl(value)).toBeNull());

    it.each(["http://x.test/a", "https://github.com/o/r"])(
        "accepts %s",
        (value) => expect(safeExternalUrl(value)).toBe(value),
    );
});

describe("markdown URL policy", () => {
    it("C2: mailto is opt-in and existing callers still reject it", () => {
        expect(safeExternalUrl("mailto:a@b.c", true)).toBe("mailto:a@b.c");
        expect(safeExternalUrl("mailto:a@b.c")).toBeNull();
    });

    it.each([
        "JaVaScRiPt:alert(1)", "java\tscript:alert(1)", " \x01javascript:alert(1)",
        "&#106;avascript:alert(1)", "javascript&colon;alert(1)", "%6aavascript:alert(1)",
        "javascript%3Aalert(1)", "data:text/html,x", "vbscript:x", "file:///tmp/x",
        "blob:https://example.com/x", "//evil", "/relative", "#fragment",
        "https://user:pass@example.com", "https://:pass@example.com", "mailto://user:pass@example.com",
        "https://", undefined,
    ])("C2: the parser rejects %s even with mailto enabled", (value) => {
        expect(safeExternalUrl(value, true)).toBeNull();
    });
});
