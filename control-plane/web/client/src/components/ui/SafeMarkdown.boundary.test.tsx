import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { expectSafeMarkdown } from "@/test/markdownSecurity";
import { SafeMarkdown } from "./SafeMarkdown";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it.each([
  ["1,250 nested list items", "- ".repeat(1250) + "**x**"],
  ["20,000 nested blockquotes", "> ".repeat(20000) + '**x** <img src=x onerror="window.__pwned=1">'],
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
  const content = "- ".repeat(1250) + "**x**";
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
