import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { expectSafeMarkdown } from "@/test/markdownSecurity";
import { SafeMarkdown } from "./SafeMarkdown";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

// 20,000 nested blockquotes overflow the renderer under any vitest pool/stack size, so the
// fallback assertions are deterministic. Shallower nesting may or may not overflow depending on
// the runtime stack; for that case only the no-crash / no-live-markup invariant is asserted.
const overflowingContent = "> ".repeat(20000) + '**x** <img src=x onerror="window.__pwned=1">';

it("C7: nested markdown that may or may not overflow never crashes and never yields live markup", () => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  const content = "- ".repeat(1250) + '**x** <img src=x onerror="window.__pwned=1">';
  const { container } = render(
    <>
      <SafeMarkdown content={content} />
      <span>Surrounding page</span>
    </>,
  );
  expect(container.querySelector(".prose")?.textContent).toContain("x");
  expect(screen.getByText("Surrounding page")).toBeInTheDocument();
  expectSafeMarkdown(container);
}, 30000);

it.each([
  ["20,000 nested blockquotes", overflowingContent],
])("C7: deeply nested markdown falls back to text for %s without losing its sibling", (_name, content) => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  const { container } = render(
    <>
      <SafeMarkdown content={content} className="preview-style" />
      <span>Surrounding page</span>
    </>,
  );
  const preview = container.querySelector(".prose")!;
  expect(preview).toHaveClass("preview-style");
  expect(preview.querySelector("pre")?.textContent).toBe(content);
  expect(preview.querySelector("pre")?.childElementCount).toBe(0);
  expect(screen.getByText("Surrounding page")).toBeInTheDocument();
  expectSafeMarkdown(container);
}, 30000);

it("C7: the same SafeMarkdown resets its failed boundary when content changes", () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {});
  const content = overflowingContent;
  const { container, rerender } = render(<SafeMarkdown content={content} />);
  const preview = container.querySelector(".prose")!;
  expect(preview.querySelector("pre")?.textContent).toBe(content);

  // An unrelated prop change must keep the fallback without retrying the failed content.
  const errorCount = errors.mock.calls.length;
  rerender(<SafeMarkdown content={content} className="updated-style" />);
  expect(preview).toHaveClass("updated-style");
  expect(preview.querySelector("pre")?.textContent).toBe(content);
  expect(errors).toHaveBeenCalledTimes(errorCount);

  rerender(<SafeMarkdown content={"# Recovered\n\n**normal**\ntext"} />);
  expect(container.querySelector(".prose")).toBe(preview);
  expect(screen.getByRole("heading", { name: "Recovered" })).toBeInTheDocument();
  expect(preview.querySelector("strong")?.textContent).toBe("normal");
  expect(preview.querySelectorAll("br")).toHaveLength(1);
  expect(preview.querySelector("pre")).toBeNull();
  expectSafeMarkdown(container);
}, 30000);
