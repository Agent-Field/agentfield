// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(
    ({ className }, ref) => <svg ref={ref} data-testid="icon" className={className} />,
  );
  Icon.displayName = "Icon";
  return {
    ChevronDown: Icon,
    ChevronRight: Icon,
    Copy: Icon,
    Code: Icon,
    Document: Icon,
    Maximize: Icon,
    Launch: Icon,
  };
});

import { SmartStringRenderer } from "./SmartStringRenderer";

describe("SmartStringRenderer", () => {
  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn() },
    });
  });

  it("renders URL content as an external link and copies it", () => {
    render(
      <SmartStringRenderer content="https://example.com/docs" label="url" path={["root"]} className="url-shell" />,
    );

    const link = screen.getByRole("link", { name: "https://example.com/docs" });
    expect(link).toHaveAttribute("href", "https://example.com/docs");
    expect(document.querySelector(".url-shell")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button"));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("https://example.com/docs");
  });

  it("renders inline markdown, supports copy, and opens the modal action", () => {
    const onOpenModal = vi.fn();

    render(
      <SmartStringRenderer
        content="**bold** with `inline`"
        label="inline"
        path={["details"]}
        onOpenModal={onOpenModal}
      />,
    );

    expect(screen.getByText("bold")).toBeInTheDocument();
    expect(screen.getByText("inline")).toBeInTheDocument();

    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]);
    fireEvent.click(buttons[1]);

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("**bold** with `inline`");
    expect(onOpenModal).toHaveBeenCalledTimes(1);
  });

  it("renders compact markdown content with badges, expansion, copy, and modal controls", () => {
    const onOpenModal = vi.fn();
    const markdown = [
      "# Heading",
      "",
      "This is a long markdown paragraph with a [link](https://example.com) and enough text to exceed the preview length threshold.",
      "",
      "- item one",
      "- item two",
    ].join("\n");

    render(
      <SmartStringRenderer
        content={markdown}
        label="markdown"
        path={["content"]}
        maxInlineHeight={180}
        onOpenModal={onOpenModal}
      />,
    );

    expect(screen.getByText(/chars$/)).toBeInTheDocument();
    expect(screen.getByText("Markdown")).toBeInTheDocument();
    expect(screen.queryByText("Heading")).not.toBeInTheDocument();

    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]);

    expect(screen.getByText("Expanded")).toBeInTheDocument();
    expect(screen.getByText("Heading")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "link" })).toHaveAttribute("href", "https://example.com");
    expect(document.querySelector("[style*='max-height: 180px']")).toBeInTheDocument();

    fireEvent.click(buttons[1]);
    fireEvent.click(buttons[2]);

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(markdown);
    expect(onOpenModal).toHaveBeenCalledTimes(1);
  });

  it("renders compact code content with code badge and preformatted block", () => {
    const code = "export const answer = () => {\n  return { value: 42 };\n};";

    render(<SmartStringRenderer content={code} label="code" path={["result"]} />);

    fireEvent.click(screen.getAllByRole("button")[0]);
    expect(screen.getByText("Code")).toBeInTheDocument();
    expect(document.querySelector("pre")?.textContent).toContain("export const answer");
    expect(document.querySelector("pre")?.textContent).toContain("return { value: 42 }");
  });
});
