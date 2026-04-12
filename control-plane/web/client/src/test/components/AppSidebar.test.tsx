// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AppSidebar } from "@/components/AppSidebar";
import { Home, Link as LinkIcon, LucideProps } from "lucide-react";

// Mock config
vi.mock("@/config/navigation", () => ({
  navigation: [
    {
      label: "Main",
      items: [
        {
          title: "Dashboard",
          path: "/dashboard",
          icon: (props: LucideProps) => <Home {...props} />,
        },
        {
          title: "Runs",
          path: "/runs",
          icon: (props: LucideProps) => <Home {...props} />,
        },
      ],
    },
  ],
  resourceLinks: [
    {
      title: "Documentation",
      href: "https://docs.example.com",
      icon: (props: LucideProps) => <LinkIcon {...props} />,
    },
  ],
}));

// Mock UI components
vi.mock("@/components/ui/mode-toggle", () => ({
  ModeToggle: () => <button>Toggle Mode</button>,
}));

vi.mock("@/components/ui/sidebar", async (importOriginal) => {
    const original = await importOriginal();
    return {
        ...original,
        Sidebar: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
        SidebarHeader: ({ children }: React.PropsWithChildren) => <header>{children}</header>,
        SidebarContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
        SidebarGroup: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
        SidebarGroupLabel: ({ children }: React.PropsWithChildren) => <h5>{children}</h5>,
        SidebarGroupContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
        SidebarMenu: ({ children }: React.PropsWithChildren) => <ul>{children}</ul>,
        SidebarMenuItem: ({ children }: React.PropsWithChildren) => <li>{children}</li>,
        SidebarMenuButton: ({ children, isActive }: React.PropsWithChildren<{isActive?: boolean}>) => <div data-active={isActive}>{children}</div>,
        SidebarSeparator: () => <hr />,
        SidebarRail: () => <div>Rail</div>,
        useSidebar: () => ({ state: 'expanded' }),
    };
});


vi.mock("@/lib/utils", () => ({
  cn: (...inputs: any[]) => inputs.filter(Boolean).join(" "),
}));

const renderWithRouter = (initialEntries: string[]) => {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <AppSidebar />
    </MemoryRouter>,
  );
};

describe("AppSidebar", () => {
  it("renders the logo, navigation, and resource links", () => {
    renderWithRouter(["/dashboard"]);
    expect(screen.getByText("AgentField")).toBeInTheDocument();
    expect(screen.getByText("Control Plane")).toBeInTheDocument();
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(screen.getByText("Resources")).toBeInTheDocument();
    expect(screen.getByText("Documentation")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Toggle Mode" })).toBeInTheDocument();
  });

  it("marks the active navigation item based on exact path", () => {
    renderWithRouter(["/dashboard"]);
    const dashboardLink = screen.getByText("Dashboard").closest('[data-active]');
    const runsLink = screen.getByText("Runs").closest('[data-active]');
    expect(dashboardLink).toHaveAttribute('data-active', 'true');
    expect(runsLink).toHaveAttribute('data-active', 'false');
  });

  it("marks the active navigation item based on a prefix path", () => {
    renderWithRouter(["/runs/some-run-id"]);
    const dashboardLink = screen.getByText("Dashboard").closest('[data-active]');
    const runsLink = screen.getByText("Runs").closest('[data-active]');
    expect(dashboardLink).toHaveAttribute('data-active', 'false');
    expect(runsLink).toHaveAttribute('data-active', 'true');
  });

  it("renders external resource links with correct attributes", () => {
    renderWithRouter(["/dashboard"]);
    const docLink = screen.getByText("Documentation").closest("a");
    expect(docLink).toHaveAttribute("href", "https://docs.example.com");
    expect(docLink).toHaveAttribute("target", "_blank");
    expect(docLink).toHaveAttribute("rel", "noopener noreferrer");
  });
});
