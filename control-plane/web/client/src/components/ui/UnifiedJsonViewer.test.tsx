import { TooltipProvider } from "@/components/ui/tooltip";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { UnifiedJsonViewer } from "./UnifiedJsonViewer";
import { attackMarkdown, expectLegitimateMarkdown, expectSafeMarkdown, legitimateMarkdown } from "@/test/markdownSecurity";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("C7: a deeply nested markdown preview falls back to text and keeps the JSON viewer mounted", async () => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  const content = "> ".repeat(20000) + '**x** <img src=x onerror="window.__pwned=1">';
  const user = userEvent.setup();
  const { container } = render(
    <UnifiedJsonViewer data={{ nested: { output: content, sibling: "Still mounted" } }} />,
    { wrapper: TooltipProvider },
  );
  await user.click(screen.getByRole("button", { name: "Show preview" }));
  const preview = container.querySelector(".prose")!;
  expect(preview.querySelector("pre")?.textContent).toBe(content);
  expect(screen.getByRole("button", { name: "Hide preview" })).toBeInTheDocument();
  expect(screen.getByText('"Still mounted"')).toBeInTheDocument();
  expectSafeMarkdown(preview);

  await user.click(screen.getByRole("button", { name: "Hide preview" }));
  expect(container.querySelector(".prose")).toBeNull();
  expect(screen.getByRole("button", { name: "Show preview" })).toBeInTheDocument();
}, 30000);

it("C1 C2 C3 C5: nested attacker string is safe through the actual Markdown Preview control", async () => {
  const user = userEvent.setup();
  const { container } = render(<UnifiedJsonViewer data={{ nested: { output: attackMarkdown } }} />, { wrapper: TooltipProvider });
  expect(container.querySelector(".prose")).toBeNull();
  // The viewer's existing icon control is titled 'Show preview'.
  await user.click(screen.getByRole("button", { name: "Show preview" }));
  const preview = container.querySelector(".prose")!;
  expect(preview.textContent).toContain('<img src=x onerror="window.__pwned=1">');
  expect(preview.textContent).toContain("blocked");
  expect(preview.querySelectorAll("a[href]")).toHaveLength(1);
  expectSafeMarkdown(preview);
  expect(preview).toHaveClass("bg-blue-50/30", "dark:bg-blue-950/20");
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(window).not.toHaveProperty("__pwned");
});

it("C4 C5: nested legitimate markdown renders in the viewer; short strings keep their inline rendering", async () => {
  const user = userEvent.setup();
  const { container, rerender } = render(<UnifiedJsonViewer data={{ nested: { output: legitimateMarkdown } }} />, { wrapper: TooltipProvider });
  await user.click(screen.getByRole("button", { name: "Show preview" }));
  expectLegitimateMarkdown(container.querySelector(".prose")!);
  rerender(<UnifiedJsonViewer data={{ output: "**short** <img src=x>" }} />);
  expect(screen.queryByRole("button", { name: "Show preview" })).toBeNull();
  expect(container.textContent).toContain("**short** <img src=x>");
  expect(container.querySelector("img")).toBeNull();
});

it("C5: ordinary strings keep their literal rendering", () => {
  const { container } = render(<UnifiedJsonViewer data={{ output: "plain <img src=x>" }} />, { wrapper: TooltipProvider });
  expect(container.textContent).toContain("plain <img src=x>");
  expect(screen.queryByRole("button", { name: "Show preview" })).toBeNull();
  expect(container.querySelector("img")).toBeNull();
});

it.each(["https://user:pass@host.example/", "https://", "javascript:alert(1)"])(
  "C2: viewer hides Open URL for unsafe destination %s", (url) => {
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    render(<UnifiedJsonViewer data={{ output: url }} />, { wrapper: TooltipProvider });
    expect(screen.queryByRole("button", { name: "Open URL" })).toBeNull();
    expect(open).not.toHaveBeenCalled();
  },
);

it.each(["https://host.example/", "https://HOST.example"])(
  "C2: viewer opens the validated normalized destination %s with isolation", async (url) => {
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    render(<UnifiedJsonViewer data={{ output: url }} />, { wrapper: TooltipProvider });
    await userEvent.click(screen.getByRole("button", { name: "Open URL" }));
    expect(open).toHaveBeenCalledExactlyOnceWith(new URL(url).href, "_blank", "noopener,noreferrer");
  },
);

it.each(["https:x", "https:\\x", "javascript:alert(1)"])(
  "C2: strings not classified as http(s) URLs get no Open URL control: %s", (value) => {
    render(<UnifiedJsonViewer data={{ output: value }} />, { wrapper: TooltipProvider });
    expect(screen.queryByRole("button", { name: "Open URL" })).toBeNull();
  },
);
