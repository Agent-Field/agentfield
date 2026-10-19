import { expect } from "vitest";

export const attackMarkdown = `# Report <img src=x onerror="window.__pwned=1">

<script>window.__pwned=1</script>

**<svg onload="window.__pwned=1">**

[blocked](javascript:alert%281%29) [safe](https://example.com/report)

![remote image](https://example.com/track.png)`;

export const legitimateMarkdown = `# Heading one

## Heading two

**Bold** and *italic* and \`inline code\`.

Paragraph one
soft line break  
hard line break

Paragraph two

\`\`\`js
const value = "<img>";
\`\`\`

1. Ordered
2. Second

- Unordered
- Another

> Quoted

| Name | Value |
| --- | --- |
| item | value |

[Docs](https://example.com/docs) [Email](mailto:operator@example.com)`;

export function expectSafeMarkdown(container: Element) {
  expect(container.querySelector("script,iframe,object,embed,style,form,img,svg")).toBeNull();
  for (const element of container.querySelectorAll("*")) {
    expect(element.getAttributeNames().some((name) => /^on/i.test(name))).toBe(false);
  }
  for (const anchor of container.querySelectorAll("a[href]")) {
    const url = new URL(anchor.getAttribute("href")!);
    expect(["http:", "https:", "mailto:"]).toContain(url.protocol);
    expect(url.username).toBe("");
    expect(url.password).toBe("");
    expect(anchor).toHaveAttribute("target", "_blank");
    expect(anchor).toHaveAttribute("rel", "noopener noreferrer");
  }
}

export function expectLegitimateMarkdown(container: Element) {
  for (const tag of ["h1", "h2", "strong", "em", "code", "pre code", "ol li", "ul li", "p", "br", "blockquote", "table th", "table td"]) {
    expect(container.querySelector(tag), tag).not.toBeNull();
  }
  expect(container.querySelector("a")).toHaveClass("text-blue-500", "hover:underline");
  expectSafeMarkdown(container);
}
