import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { highlightJsonText, JsonHighlightedPre } from "./json-syntax-highlight";

afterEach(cleanup);

const payloads = [
  '<img src=x onerror=alert(1)>',
  '</pre><script>window.__pwned=1</script>',
  `"' & < > onmouseover="window.__pwned=1"`,
  '{}\n\n… truncated (<img src=x onerror=alert(1)> "\' & chars total)',
];

function expectEscaped(container: Element, original: string) {
  expect(container.textContent).toBe(original);
  for (const element of container.querySelectorAll("*")) {
    expect(element.tagName).toBe("SPAN");
    expect(element.getAttributeNames()).toEqual(["class"]);
    expect(["json-hl-key", "json-hl-string", "json-hl-number", "json-hl-boolean", "json-hl-null", "json-hl-punct"]).toContain(element.className);
  }
  expect(container.querySelector("img,script,iframe,svg")).toBeNull();
}

it.each(payloads)("C6: highlightJsonText escapes every input byte in %s", (payload) => {
  const template = document.createElement("template");
  const html = highlightJsonText(payload);
  expect(html.replace(/<span class="json-hl-(?:key|string|number|boolean|null|punct)">|<\/span>/g, "")).not.toMatch(/[<>"']/);
  template.innerHTML = html;
  const container = document.createElement("div");
  container.append(template.content);
  expectEscaped(container, payload);
});

it.each(payloads)("C6: JsonHighlightedPre text and data props escape %s", async (payload) => {
  const { container, rerender } = render(<JsonHighlightedPre text={payload} />);
  expectEscaped(container.querySelector("pre")!, payload);
  rerender(<JsonHighlightedPre data={{ [payload]: payload }} />);
  expectEscaped(container.querySelector("pre")!, JSON.stringify({ [payload]: payload }, null, 2));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(window).not.toHaveProperty("__pwned");
});

it("C6: escapes ampersands and both quote kinds even outside JSON tokens", () => {
  expect(highlightJsonText(`&'`)).toBe("&amp;&#39;");
  expect(highlightJsonText('"')).toBe('<span class="json-hl-string">&quot;</span>');
});
