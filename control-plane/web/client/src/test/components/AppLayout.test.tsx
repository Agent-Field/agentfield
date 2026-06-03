// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/components/AppLayout";

// Mock child components and hooks
vi.mock("@/components/AppSidebar", () => ({
  AppSidebar: () => <div data-testid="app-sidebar">AppSidebar</div>,
}));

vi.mock("@/components/HealthStrip", () => ({
  HealthStrip: () => <div data-testid="health-strip">HealthStrip</div>,
}));

vi.mock("@/components/CommandPalette", () => ({
  CommandPalette: () => <div data-testid="command-palette">CommandPalette</div>,
}));

vi.mock("@/components/NotificationBell", () => ({
  NotificationBell: () => (
    <div data-testid="notification-bell">NotificationBell</div>
  ),
}));

vi.mock("@/hooks/useSSEQuerySync", () => ({
  SSESyncProvider: ({ children }: React.PropsWithChildren) => (
    <>{children}</>
  ),
}));

// Mock UI library components
vi.mock("@/components/ui/sidebar", async (importOriginal) => {
  const original = await importOriginal();
  return {
    ...original,
    SidebarProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
    SidebarInset: ({ children }: React.PropsWithChildren) => <main>{children}</main>,
    SidebarTrigger: () => <button>Trigger</button>,
  };
});

const renderWithRouter = (initialEntries: string[]) => {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route path="/" element={<AppLayout />}>
           {/* This will render for any nested route */}
          <Route path="*" element={<div>Child Content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>
  );
};

describe("AppLayout", () => {
  it("renders main components and outlet content", () => {
    renderWithRouter(["/dashboard"]);
    expect(screen.getByTestId("app-sidebar")).toBeInTheDocument();
    expect(screen.getByTestId("health-strip")).toBeInTheDocument();
    expect(screen.getByTestId("notification-bell")).toBeInTheDocument();
    expect(screen.getByTestId("command-palette")).toBeInTheDocument();
    expect(screen.getByText("Child Content")).toBeInTheDocument();
  });

  it("renders skip to main content link", () => {
    renderWithRouter(["/dashboard"]);
    expect(screen.getByText("Skip to main content")).toBeInTheDocument();
  });

  it('hides breadcrumb on section index route like /dashboard', () => {
    renderWithRouter(['/dashboard']);
    const breadcrumbContainer = screen.getByRole('banner').querySelector('.min-w-0.flex-1.overflow-hidden.pr-1');
    expect(breadcrumbContainer?.querySelector('.min-h-5[aria-hidden=true]')).toBeInTheDocument();
    expect(screen.queryByRole('navigation', { name: /breadcrumb/i })).toBeNull();
  });

  it('shows breadcrumb trail for a nested run details page', () => {
    renderWithRouter(['/runs/run-abcd12345678']);
    expect(screen.getByText('Runs')).toBeInTheDocument();
    expect(screen.getAllByText('…12345678').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: 'Runs' })).toHaveAttribute('href', '/runs');
  });

  it('shows breadcrumb for compare runs page', () => {
    renderWithRouter(['/runs/compare']);
    expect(screen.getByText('Runs')).toBeInTheDocument();
    expect(screen.getAllByText('Compare').length).toBeGreaterThan(0);
  });

  it('shows breadcrumb for a nested playground page', () => {
    renderWithRouter(['/playground/reasoner-long-name-abcdefghijklmno']);
    expect(screen.getByText('Playground')).toBeInTheDocument();
    expect(screen.getAllByText('…bcdefghijklmno').length).toBeGreaterThan(0);
  });

  it('shows a generic breadcrumb for other nested pages', () => {
    renderWithRouter(['/settings/profile']);
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getAllByText('Profile').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: 'Settings' })).toHaveAttribute('href', '/settings');
  });
  
  it('shows a default breadcrumb for unknown paths', () => {
    renderWithRouter(['/some/unknown/path']);
    expect(screen.getAllByText('AgentField').length).toBeGreaterThan(0);
  });
});
