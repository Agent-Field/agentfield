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

    it.each([
        "http://x.test/a", "https://github.com/o/r", "https://example.com",
        "HTTPS://EXAMPLE.COM:443", " \x01https://example.com\t\n", "https://例え.test/a b",
    ])(
        "C2: returns the normalized validated URL for %s",
        (value) => expect(safeExternalUrl(value)).toBe(new URL(value).href),
    );

    it.each([
        "https:x",
        "https:\\x",
        "https:\\\\x",
        "https:\\\\\\\\x",
    ])("C2: shorthand or backslash URL %s navigates to the validated absolute destination", (value) => {
        const safe = safeExternalUrl(value);
        expect(safe).toBe("https://x/");
        const anchor = document.createElement("a");
        anchor.href = safe!;
        expect(anchor.href).toBe("https://x/");
    });
});

describe("markdown URL policy", () => {
    it("C2: mailto is opt-in and existing callers still reject it", () => {
        expect(safeExternalUrl("mailto:a@b.c", true)).toBe("mailto:a@b.c");
        expect(safeExternalUrl("MAILTO:a@b.c", true)).toBe("mailto:a@b.c");
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
