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
