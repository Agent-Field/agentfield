import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SafeMarkdown } from "./SafeMarkdown";
import { expectLegitimateMarkdown, expectSafeMarkdown, legitimateMarkdown } from "@/test/markdownSecurity";

afterEach(cleanup);

describe("SafeMarkdown", () => {
  it.each([
    ["img onerror", '<img src=x onerror="window.__pwned=1">'],
    ["script", '<script>window.__pwned=1</script>'],
    ["svg onload", '<svg onload="window.__pwned=1"></svg>'],
    ["iframe", '<iframe src="https://example.com"></iframe>'],
    ["object", '<object data="https://example.com"></object>'],
    ["embed", '<embed src="https://example.com">'],
    ["raw javascript anchor", '<a href="javascript:alert(1)">click</a>'],
    ["style", '<style>body { display: none }</style>'],
    ["form", '<form action="javascript:alert(1)">form</form>'],
    ["comment", '<!-- <img src=x onerror="window.__pwned=1"> -->'],
    ["unclosed tag", '<img src=x onerror="window.__pwned=1"'],
    ["malformed tag", '<img/src=x/onerror="window.__pwned=1">'],
  ])("C1: raw %s remains visible literal text", async (_name, payload) => {
    const { container } = render(<SafeMarkdown content={payload} />);
    expect(container.textContent).toBe(payload);
    expectSafeMarkdown(container);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(window).not.toHaveProperty("__pwned");
  });

  it.each([
    ['**<img src=x onerror="window.__pwned=1">**', '<img src=x onerror="window.__pwned=1">'],
    ['`<img src=x onerror="window.__pwned=1">`', '<img src=x onerror="window.__pwned=1">'],
    ['[<img src=x onerror="window.__pwned=1">](x)', '<img src=x onerror="window.__pwned=1">'],
    ['&lt;img src=x onerror="window.__pwned=1"&gt;', '<img src=x onerror="window.__pwned=1">'],
    ['&#60;script&#62;window.__pwned=1&#60;/script&#62;', '<script>window.__pwned=1</script>'],
    ['&amp;lt;img&amp;gt;', '&lt;img&gt;'],
  ])("C1: markdown and entity variant %s stays text", (content, expected) => {
    const { container } = render(<SafeMarkdown content={content} />);
    expect(container.textContent).toContain(expected);
    expectSafeMarkdown(container);
  });

  it("C1: HTML in a link title is literal attribute text", () => {
    const title = '<img src=x onerror="window.__pwned=1">';
    const { container } = render(<SafeMarkdown content={`[title](https://example.com '${title}')`} />);
    expect(screen.getByRole("link")).toHaveAttribute("title", title);
    expectSafeMarkdown(container);
  });

  it.each([
    "javascript:alert%281%29", "JaVaScRiPt:alert%281%29", "java\tscript:alert%281%29",
    " \x01javascript:alert%281%29", "&#106;avascript:alert%281%29", "javascript&colon;alert%281%29",
    "java&#x09;script:alert%281%29", "%6aavascript:alert%281%29", "javascript%3Aalert%281%29",
    "data:text/html,payload", "vbscript:alert%281%29", "file:///tmp/payload", "blob:https://example.com/id",
    "//evil.example.com", "/relative", "relative", "#fragment", "https://user:pass@example.com",
  ])("C2: rejects markdown destination %s", (url) => {
    const { container } = render(<SafeMarkdown content={`[blocked](<${url}>)`} />);
    expect(container.querySelector("a[href]")).toBeNull();
    expect(container.textContent).toContain("blocked");
    expectSafeMarkdown(container);
  });

  it.each(["http://example.com", "https://example.com/path?q=a&b=c", "HTTPS://example.com", "https:x", "http:/x", "mailto:operator@example.com"])(
    "C2: safe destination %s opens with isolation", (url) => {
      const { container } = render(<SafeMarkdown content={`[safe](${url})`} />);
      expect(screen.getByRole("link")).toHaveAttribute("href", new URL(url).href);
      expectSafeMarkdown(container);
    },
  );

  it("C3: images become alt text and GFM autolinks obey C2", () => {
    const { container } = render(<SafeMarkdown content={'![visible alt](https://example.com/tracker) ![](data:image/png,x)\n\nwww.example.com https://example.com/path https://user:pass@example.com'} />);
    expect(container.textContent).toContain("visible alt");
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getAllByRole("link").map((a) => a.getAttribute("href"))).toEqual(["http://www.example.com/", "https://example.com/path"]);
    expectSafeMarkdown(container);
  });

  it("C4: legitimate markdown retains all supported constructs and styling", () => {
    const { container } = render(<SafeMarkdown content={legitimateMarkdown} />);
    expectLegitimateMarkdown(container);
    expect(container.firstChild).toHaveClass("prose-sm");
  });
});
