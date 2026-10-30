import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import { DiscoveryPage } from "./DiscoveryPage";
import type { ARDDashboard } from "@/types/ard";
import { getARDDashboard } from "@/services/ardApi";

vi.mock("@/services/ardApi", () => ({
  getARDDashboard: vi.fn(),
  updateARDSettings: vi.fn(), saveARDPublication: vi.fn(), searchExternalARD: vi.fn(),
  importARDEntry: vi.fn(), saveARDBinding: vi.fn(), saveARDRegistries: vi.fn(),
}));

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function renderCatalog(url: string) {
  const dashboard = {
    config: { locked: {}, allowed_registries: [] },
    state: { settings: {}, publications: {}, imports: [], bindings: {}, registries: [] },
    summary: { catalog_url: url }, publications: [],
  } as unknown as ARDDashboard;
  vi.mocked(getARDDashboard).mockResolvedValue(dashboard);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter><DiscoveryPage /></MemoryRouter></QueryClientProvider>);
}

it.each(["javascript:alert(1)", "java\tscript:alert(1)", "https://user:pass@example.com", "//evil.example.com"])(
  "C2: Discovery rejects control-plane catalog URL %s", async (url) => {
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    renderCatalog(url);
    const button = await screen.findByRole("button", { name: "Open" });
    expect(button).toBeDisabled();
    await userEvent.click(button);
    expect(open).not.toHaveBeenCalled();
  },
);

it.each(["https://example.com/.well-known/ai-catalog.json", "/.well-known/ai-catalog.json"])(
  "C2: Discovery opens safe catalog %s with isolation", async (url) => {
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    renderCatalog(url);
    await userEvent.click(await screen.findByRole("button", { name: "Open" }));
    expect(open).toHaveBeenCalledWith(new URL(url, window.location.origin).href, "_blank", "noopener,noreferrer");
  },
);
