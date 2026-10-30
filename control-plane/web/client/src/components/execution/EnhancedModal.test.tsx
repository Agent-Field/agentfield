import { TooltipProvider } from "@/components/ui/tooltip";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { DataModal } from "./EnhancedModal";
import { attackMarkdown, expectLegitimateMarkdown, expectSafeMarkdown, legitimateMarkdown } from "@/test/markdownSecurity";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("C7: DataModal's deeply nested markdown tab falls back to text and keeps the dialog mounted", async () => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  const content = "> ".repeat(20000) + '**x** <img src=x onerror="window.__pwned=1">';
  const user = userEvent.setup();
  render(<DataModal isOpen onClose={() => {}} title="Output" data={content} />, { wrapper: TooltipProvider });
  await user.click(screen.getByRole("tab", { name: "Markdown Preview" }));
  const dialog = screen.getByRole("dialog");
  const preview = dialog.querySelector(".prose")!;
  expect(preview.querySelector("pre")?.textContent).toBe(content);
  expectSafeMarkdown(preview);
  expect(screen.getByRole("tab", { name: "Markdown Preview" })).toHaveAttribute("data-state", "active");

  await user.click(screen.getByRole("tab", { name: "Raw JSON" }));
  expect(screen.getByRole("dialog")).toBe(dialog);
  expect(screen.getByRole("tab", { name: "Raw JSON" })).toHaveAttribute("data-state", "active");
}, 30000);

it("C1 C2 C3 C5: DataModal's Markdown Preview tab safely renders an attacker string", async () => {
  const user = userEvent.setup();
  render(<DataModal isOpen onClose={() => {}} title="Output" data={attackMarkdown} />, { wrapper: TooltipProvider });
  await user.click(screen.getByRole("tab", { name: "Markdown Preview" }));
  const preview = screen.getByRole("dialog").querySelector(".prose")!;
  expect(preview.textContent).toContain('<img src=x onerror="window.__pwned=1">');
  expect(preview.textContent).toContain("blocked");
  expect(preview.querySelectorAll("a[href]")).toHaveLength(1);
  expectSafeMarkdown(preview);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(window).not.toHaveProperty("__pwned");
});

it("C4 C5: DataModal's markdown tab retains legitimate markdown", async () => {
  const user = userEvent.setup();
  render(<DataModal isOpen onClose={() => {}} title="Output" data={legitimateMarkdown} />, { wrapper: TooltipProvider });
  await user.click(screen.getByRole("tab", { name: "Markdown Preview" }));
  expectLegitimateMarkdown(screen.getByRole("dialog").querySelector(".prose")!);
});

it("C5: DataModal keeps non-markdown on the formatted and raw views", async () => {
  const user = userEvent.setup();
  render(<DataModal isOpen onClose={() => {}} title="Output" data="plain <img src=x>" />, { wrapper: TooltipProvider });
  expect(screen.queryByRole("tab", { name: "Markdown Preview" })).toBeNull();
  await user.click(screen.getByRole("tab", { name: "Raw JSON" }));
  expect(screen.getByRole("dialog").querySelector("pre")).toHaveTextContent('"plain <img src=x>"');
  expect(screen.getByRole("dialog").querySelector("img")).toBeNull();
});
