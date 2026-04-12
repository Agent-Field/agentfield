// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

const routeState = vi.hoisted(() => ({
  path: "/dashboard",
  reasonerId: "planner-1",
}));

vi.mock("react-router-dom", () => {
  const ReactRouterDom = {
    BrowserRouter: ({ children }: React.PropsWithChildren) => <>{children}</>,
    Routes: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    Route: ({
      element,
      children,
    }: React.PropsWithChildren<{ element?: React.ReactNode }>) => (
      <>
        {element}
        {children}
      </>
    ),
    Navigate: ({ to }: { to: string }) => <div>navigate:{to}</div>,
    useParams: () => ({ reasonerId: routeState.reasonerId }),
  };
  return ReactRouterDom;
});

vi.mock("@tanstack/react-query", () => ({
  QueryClientProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/theme-provider", () => ({
  ThemeProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/contexts/ModeContext", () => ({
  ModeProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/contexts/AuthContext", () => ({
  AuthProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/AuthGuard", () => ({
  AuthGuard: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/ErrorBoundary", () => ({
  ErrorBoundary: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/ui/notification", () => ({
  NotificationProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/lib/query-client", () => ({
  queryClient: {},
}));

vi.mock("@/components/AppLayout", () => ({
  AppLayout: () => <div>AppLayout</div>,
}));

vi.mock("@/components/RootRedirect", () => ({
  RootRedirect: () => <div>RootRedirect</div>,
}));

vi.mock("@/pages/NewDashboardPage", () => ({
  NewDashboardPage: () => <div>NewDashboardPage</div>,
}));

vi.mock("@/pages/NewSettingsPage", () => ({
  NewSettingsPage: () => <div>NewSettingsPage</div>,
}));

vi.mock("@/pages/AgentsPage", () => ({
  AgentsPage: () => <div>AgentsPage</div>,
}));

vi.mock("@/pages/RunsPage", () => ({
  RunsPage: () => <div>RunsPage</div>,
}));

vi.mock("@/pages/RunDetailPage", () => ({
  RunDetailPage: () => <div>RunDetailPage</div>,
}));

vi.mock("@/pages/VerifyProvenancePage", () => ({
  VerifyProvenancePage: () => <div>VerifyProvenancePage</div>,
}));

vi.mock("@/pages/ComparisonPage", () => ({
  ComparisonPage: () => <div>ComparisonPage</div>,
}));

vi.mock("@/pages/PlaygroundPage", () => ({
  PlaygroundPage: () => <div>PlaygroundPage</div>,
}));

vi.mock("@/pages/AccessManagementPage", () => ({
  AccessManagementPage: () => <div>AccessManagementPage</div>,
}));

describe("App", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("renders the routed application tree", async () => {
    const { default: App } = await import("@/App");
    render(<App />);

    expect(screen.getByText("AppLayout")).toBeInTheDocument();
    expect(screen.getByText("RootRedirect")).toBeInTheDocument();
    expect(screen.getByText("NewDashboardPage")).toBeInTheDocument();
    expect(screen.getByText("NewSettingsPage")).toBeInTheDocument();
    expect(screen.getByText("AgentsPage")).toBeInTheDocument();
    expect(screen.getByText("RunsPage")).toBeInTheDocument();
    expect(screen.getByText("RunDetailPage")).toBeInTheDocument();
    expect(screen.getByText("VerifyProvenancePage")).toBeInTheDocument();
    expect(screen.getByText("ComparisonPage")).toBeInTheDocument();
    expect(screen.getAllByText("PlaygroundPage").length).toBeGreaterThan(0);
    expect(screen.getByText("AccessManagementPage")).toBeInTheDocument();
    expect(screen.getAllByText("navigate:/runs").length).toBeGreaterThan(0);
    expect(screen.getAllByText("navigate:/agents").length).toBeGreaterThan(0);
    expect(screen.getAllByText("navigate:/settings").length).toBeGreaterThan(0);
    expect(screen.getAllByText("navigate:/access").length).toBeGreaterThan(0);
    expect(screen.getByText("navigate:/playground/planner-1")).toBeInTheDocument();
  });
});
