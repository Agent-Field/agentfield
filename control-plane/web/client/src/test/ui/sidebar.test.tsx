// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from "@/components/ui/sidebar";

const { useIsMobileMock } = vi.hoisted(() => ({
  useIsMobileMock: vi.fn(),
}));

vi.mock("@/hooks/use-mobile", () => ({
  useIsMobile: useIsMobileMock,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    PanelLeft: Icon,
    X: Icon,
  };
});

function SidebarHarness() {
  return (
    <SidebarProvider defaultOpen>
      <Sidebar>
        <SidebarHeader>
          <SidebarInput placeholder="Filter" />
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Projects</SidebarGroupLabel>
            <SidebarGroupAction aria-label="Add group action">+</SidebarGroupAction>
            <SidebarGroupContent>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton tooltip="Overview" isActive>
                    <span>Overview</span>
                  </SidebarMenuButton>
                  <SidebarMenuAction aria-label="Menu action" showOnHover>
                    *
                  </SidebarMenuAction>
                  <SidebarMenuBadge>3</SidebarMenuBadge>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
          <SidebarMenuSkeleton showIcon />
        </SidebarContent>
      </Sidebar>
      <div>
        <SidebarTrigger />
        <SidebarRail />
      </div>
    </SidebarProvider>
  );
}

describe("sidebar primitives", () => {
  beforeEach(() => {
    useIsMobileMock.mockReturnValue(false);
    document.cookie = "";
  });

  it("renders desktop sidebar parts and toggles collapsed state from trigger and keyboard shortcut", async () => {
    const user = userEvent.setup();

    render(<SidebarHarness />);

    expect(screen.getByPlaceholderText("Filter")).toBeInTheDocument();
    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("Overview")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Toggle Sidebar" })[0]);
    const desktopSidebar = document.querySelector('[data-side="left"]');
    expect(desktopSidebar).toHaveAttribute("data-state", "collapsed");
    expect(document.cookie).toContain("sidebar_state=false");

    fireEvent.keyDown(window, { key: "b", ctrlKey: true });
    expect(desktopSidebar).toHaveAttribute("data-state", "expanded");
    expect(document.cookie).toContain("sidebar_state=true");
  });

  it("renders the mobile sheet variant when the mobile hook is true", async () => {
    const user = userEvent.setup();
    useIsMobileMock.mockReturnValue(true);

    render(<SidebarHarness />);

    await user.click(screen.getAllByRole("button", { name: "Toggle Sidebar" })[0]);
    expect(await screen.findByText("Displays the mobile sidebar.")).toBeInTheDocument();
    expect(screen.getByText("Overview")).toBeInTheDocument();
  });
});