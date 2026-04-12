import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    ChevronRight: Icon,
    ChevronDown: Icon,
    FileText: Icon,
    Hash: Icon,
    Type: Icon,
    List: Icon,
    Braces: Icon,
    Quote: Icon,
    Eye: Icon,
    EyeOff: Icon,
    Search: Icon,
    Maximize2: Icon,
    Minimize2: Icon,
    Copy: Icon,
    Check: Icon,
  };
});

vi.mock("@/components/ui/copy-button", () => ({
  CopyButton: ({ value, tooltip, onClick, ...props }: { value: string; tooltip?: string; onClick?: React.MouseEventHandler<HTMLButtonElement> }) => (
    <button type="button" aria-label={tooltip ?? "Copy"} data-value={value} onClick={onClick} {...props}>
      copy
    </button>
  ),
}));

describe("UnifiedJsonViewer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state and supports flexible height without a header", () => {
    render(<UnifiedJsonViewer data={{}} showHeader={false} maxHeight="none" />);
    expect(screen.getByText("No data available")).toBeInTheDocument();
  });

  it("renders searchable JSON, expands long content, previews markdown, and opens urls", async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);

    render(
      <UnifiedJsonViewer
        title="Payload"
        data={{
          short: "hello",
          markdown: `**bold** [link](https://example.com) ${"markdown ".repeat(20)}`,
          url: "https://example.com",
          nested: { item: "visible value" },
          long: "x".repeat(120),
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Payload" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand" })).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("Search keys and values..."), "visible");
    expect(screen.getByText('"item":')).toBeInTheDocument();
    expect(screen.queryByText('"short":')).not.toBeInTheDocument();

    await user.clear(screen.getByPlaceholderText("Search keys and values..."));
    await user.click(screen.getByRole("button", { name: "Expand" }));
    expect(screen.getByRole("button", { name: "Collapse" })).toBeInTheDocument();

    const markdownRow = screen.getByText('"markdown":').closest(".group");
    expect(markdownRow).toBeTruthy();
    await user.click(within(markdownRow as HTMLElement).getByRole("button", { name: "Show preview" }));
    expect(await screen.findByRole("link", { name: "link" })).toHaveAttribute("href", "https://example.com");

    await user.click(screen.getByRole("button", { name: "Open URL" }));
    expect(openSpy).toHaveBeenCalledWith("https://example.com", "_blank", "noopener,noreferrer");

    const copyButtons = screen.getAllByRole("button", { name: /Copy /i });
    expect(copyButtons.length).toBeGreaterThan(0);
  });
});
