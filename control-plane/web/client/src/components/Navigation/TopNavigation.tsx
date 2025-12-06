import React from "react";
import { useLocation, Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { ModeToggle } from "@/components/ui/mode-toggle";

export function TopNavigation() {
  const location = useLocation();

  // Generate breadcrumbs from current path
  const generateBreadcrumbs = () => {
    const pathSegments = location.pathname.split("/").filter(Boolean);
    const breadcrumbs = [{ label: "Home", href: "/" }];

    // Define route mappings for better breadcrumb navigation
    const routeMappings: Record<
      string,
      { label: string; href: string; parent?: string }
    > = {
      reasoners: { label: "Reasoners", href: "/reasoners/all" },
      "reasoners/all": {
        label: "All Reasoners",
        href: "/reasoners/all",
        parent: "reasoners",
      },
      "reasoners/executions": {
        label: "Recent Executions",
        href: "/reasoners/executions",
        parent: "reasoners",
      },
      "reasoners/workflows": {
        label: "Workflows",
        href: "/reasoners/workflows",
        parent: "reasoners",
      },
      nodes: { label: "Agent Nodes", href: "/nodes" },
      packages: { label: "Packages", href: "/packages" },
      settings: { label: "Settings", href: "/settings" },
      agents: { label: "My Agents", href: "/agents" },
      dashboard: { label: "Dashboard", href: "/dashboard" },
      "dashboard/enhanced": {
        label: "Enhanced Dashboard",
        href: "/dashboard/enhanced",
        parent: "dashboard",
      },
      identity: { label: "Identity & Trust", href: "/identity/dids" },
      "identity/dids": { label: "DID Explorer", href: "/identity/dids" },
      "identity/credentials": { label: "Credentials", href: "/identity/credentials" },
    };

    let currentPath = "";
    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const routeKey = pathSegments.slice(0, index + 1).join("/");

      // Check if we have a specific mapping for this route
      if (routeMappings[routeKey]) {
        const mapping = routeMappings[routeKey];

        // For reasoners/all, we want to show just "Reasoners" in breadcrumb
        if (routeKey === "reasoners/all") {
          // Replace the previous "Reasoners" breadcrumb if it exists
          const existingReasonersIndex = breadcrumbs.findIndex(
            (b) => b.label === "Reasoners"
          );
          if (existingReasonersIndex !== -1) {
            breadcrumbs[existingReasonersIndex] = {
              label: "Reasoners",
              href: "/reasoners/all",
            };
          } else {
            breadcrumbs.push({ label: "Reasoners", href: "/reasoners/all" });
          }
          return; // Skip adding "All Reasoners" as a separate breadcrumb
        }

        breadcrumbs.push({ label: mapping.label, href: mapping.href });
      } else {
        // Handle dynamic routes (like reasoner detail pages)
        let label =
          segment.charAt(0).toUpperCase() + segment.slice(1).replace("-", " ");
        const href = currentPath;

        // Special handling for reasoner detail pages
        if (
          pathSegments[index - 1] === "reasoners" &&
          segment !== "all" &&
          segment !== "executions" &&
          segment !== "workflows"
        ) {
          // This is a reasoner detail page - decode the reasoner ID and format it nicely
          try {
            const decodedId = decodeURIComponent(segment);
            const parts = decodedId.split(".");
            if (parts.length >= 2) {
              label = parts[parts.length - 1]; // Use the reasoner name part
            } else {
              label = decodedId;
            }
          } catch {
            label = segment;
          }

          // Ensure the parent "Reasoners" breadcrumb points to /reasoners/all
          const reasonersIndex = breadcrumbs.findIndex(
            (b) => b.label === "Reasoners"
          );
          if (reasonersIndex !== -1) {
            breadcrumbs[reasonersIndex].href = "/reasoners/all";
          }
        }
        // Handle node detail pages
        else if (pathSegments[index - 1] === "nodes") {
          label = `Node ${segment}`;
        }

        breadcrumbs.push({ label, href });
      }
    });

    return breadcrumbs;
  };

  const breadcrumbs = generateBreadcrumbs();

  return (
    <header
      className={cn(
        "h-16 flex items-center justify-between sticky top-0 z-50",
        "bg-gradient-to-r from-bg-base via-bg-subtle to-bg-base",
        "backdrop-blur-xl border-none", // Borderless design
        "shadow-soft transition-all duration-200",
        "px-4 md:px-6 lg:px-8" // Better horizontal padding
      )}
    >
      {/* Left Section - Refined Breadcrumbs */}
      <div className="flex items-center gap-4 flex-1 min-w-0">
        {/* Sidebar Toggle */}
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="h-4" />

        {/* Enhanced Breadcrumbs using shadcn component */}
        <Breadcrumb>
          <BreadcrumbList>
            {breadcrumbs.map((crumb, index) => {
              const isFirst = index === 0;
              const isLast = index === breadcrumbs.length - 1;
              const isHiddenOnMobile = !isFirst && !isLast;

              return (
                <React.Fragment key={crumb.href}>
                  <BreadcrumbItem className={cn(isHiddenOnMobile && "hidden md:inline-flex")}>
                    {isLast ? (
                      <BreadcrumbPage className="max-w-[150px] md:max-w-[200px] truncate" title={crumb.label}>
                        {crumb.label}
                      </BreadcrumbPage>
                    ) : (
                      <BreadcrumbLink asChild>
                        <Link
                          to={crumb.href}
                          className="max-w-[150px] md:max-w-[200px] truncate"
                          title={crumb.label}
                        >
                          {crumb.label}
                        </Link>
                      </BreadcrumbLink>
                    )}
                  </BreadcrumbItem>
                  {index < breadcrumbs.length - 1 && (
                    <BreadcrumbSeparator className={cn(isHiddenOnMobile && "hidden md:list-item")} />
                  )}
                </React.Fragment>
              );
            })}
          </BreadcrumbList>
        </Breadcrumb>
      </div>

      {/* Right Section - Theme Toggle & Future Extensions */}
      <div className="flex items-center gap-3">
        <ModeToggle />
      </div>
    </header>
  );
}
