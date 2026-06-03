import { Link, Outlet, useLocation, useParams } from "react-router-dom";
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { AppSidebar } from "./AppSidebar";
import { HealthStrip } from "./HealthStrip";
import { CommandPalette } from "./CommandPalette";
import { NotificationBell } from "./NotificationBell";
import { SSESyncProvider } from "@/hooks/useSSEQuerySync";

const routeNames: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/runs": "Runs",
  "/agents": "Agent nodes",
  "/playground": "Playground",
  "/verify": "Provenance",
  "/access": "Access management",
  "/settings": "Settings",
};

/** Match the longest configured section prefix so `/dashboard/legacy` maps to `/dashboard`. */
function longestSectionPath(pathname: string): string | null {
  const paths = Object.keys(routeNames).sort((a, b) => b.length - a.length);
  for (const p of paths) {
    if (pathname === p || pathname.startsWith(`${p}/`)) {
      return p;
    }
  }
  return null;
}

function shortResourceId(id: string, tailChars = 8): string {
  const t = Math.max(4, tailChars);
  if (id.length <= t + 1) return id;
  return `…${id.slice(-t)}`;
}

type HeaderCrumb = { label: string; to?: string };

/**
 * On section index routes the sidebar already shows the active item, so the header
 * title was redundant. Hide it there; use a real trail on nested routes (back via link).
 */
function resolveHeaderCrumbs(
  pathname: string,
  params: Readonly<Partial<Record<string, string | undefined>>>,
): { mode: "hidden" } | { mode: "trail"; crumbs: HeaderCrumb[] } {
  const section = longestSectionPath(pathname);
  if (!section) {
    return { mode: "trail", crumbs: [{ label: "AgentField" }] };
  }

  const sectionTitle = routeNames[section]!;
  const rest = pathname.slice(section.length).replace(/^\//, "");
  if (!rest) {
    return { mode: "hidden" };
  }

  const parts = rest.split("/").filter(Boolean);

  if (section === "/runs") {
    if (parts[0] === "compare") {
      return {
        mode: "trail",
        crumbs: [
          { label: sectionTitle, to: section },
          { label: "Compare" },
        ],
      };
    }
    const runId = params.runId ?? parts[0];
    if (runId) {
      return {
        mode: "trail",
        crumbs: [
          { label: sectionTitle, to: section },
          { label: shortResourceId(runId) },
        ],
      };
    }
  }

  if (section === "/playground" && parts[0]) {
    const reasonerId = params.reasonerId ?? parts[0];
    return {
      mode: "trail",
      crumbs: [
        { label: sectionTitle, to: section },
        { label: shortResourceId(reasonerId, 14) },
      ],
    };
  }

  const last = parts[parts.length - 1] ?? rest;
  const display =
    last.length > 24 ? shortResourceId(last, 10) : last.replace(/-/g, " ");
  const pageLabel =
    display.length > 0
      ? display.charAt(0).toUpperCase() + display.slice(1)
      : "Page";

  return {
    mode: "trail",
    crumbs: [{ label: sectionTitle, to: section }, { label: pageLabel }],
  };
}

export function AppLayout() {
  return (
    <SSESyncProvider>
      <AppLayoutShell />
    </SSESyncProvider>
  );
}

function AppLayoutShell() {
  const location = useLocation();
  const params = useParams();
  const header = resolveHeaderCrumbs(location.pathname, params);
  return (
    // h-svh + overflow-hidden on the wrapper forces the inner content div
    // to be the scroll container (not the body). That way the top header
    // below stays pinned at the top of the viewport regardless of scroll
    // position — no sticky hack needed.
    <SidebarProvider
      defaultOpen={true}
      className="h-svh overflow-hidden"
    >
      <AppSidebar />
      <SidebarInset className="min-h-0 overflow-hidden">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
        >
          Skip to main content
        </a>
        <header className="sticky top-0 z-30 flex h-16 min-w-0 shrink-0 items-center gap-2 overflow-hidden border-b border-border/60 bg-background/85 px-4 backdrop-blur transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
          <SidebarTrigger className="-ml-1 shrink-0" />
          <Separator orientation="vertical" className="mr-2 h-4 shrink-0" />
          <div className="min-w-0 flex-1 overflow-hidden pr-1">
            {header.mode === "hidden" ? (
              <div className="min-h-5" aria-hidden="true" />
            ) : (
              <>
                {/* Narrow viewports: only the leaf — sidebar already shows section; saves space for status strip. */}
                <Breadcrumb className="min-w-0 overflow-hidden sm:hidden">
                  <BreadcrumbList className="min-w-0">
                    <BreadcrumbItem className="min-w-0 max-w-full">
                      <BreadcrumbPage className="block truncate">
                        {header.crumbs[header.crumbs.length - 1]!.label}
                      </BreadcrumbPage>
                    </BreadcrumbItem>
                  </BreadcrumbList>
                </Breadcrumb>
                <Breadcrumb className="hidden min-w-0 overflow-hidden sm:block">
                  <BreadcrumbList className="min-w-0 flex-nowrap overflow-hidden">
                    {header.crumbs.map((crumb, i) => (
                      <span key={`${crumb.label}-${i}`} className="contents">
                        {i > 0 ? <BreadcrumbSeparator className="shrink-0" /> : null}
                        <BreadcrumbItem
                          className={
                            i === header.crumbs.length - 1
                              ? "min-w-0 max-w-full"
                              : "shrink-0"
                          }
                        >
                          {crumb.to && i < header.crumbs.length - 1 ? (
                            <BreadcrumbLink asChild>
                              <Link to={crumb.to}>{crumb.label}</Link>
                            </BreadcrumbLink>
                          ) : (
                            <BreadcrumbPage className="block truncate">
                              {crumb.label}
                            </BreadcrumbPage>
                          )}
                        </BreadcrumbItem>
                      </span>
                    ))}
                  </BreadcrumbList>
                </Breadcrumb>
              </>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2 sm:gap-3">
            <HealthStrip />
            <Separator orientation="vertical" className="hidden h-4 sm:block" />
            <kbd className="hidden md:inline-flex h-5 shrink-0 items-center gap-1 rounded border border-border bg-muted px-1.5 text-micro font-mono text-muted-foreground">
              ⌘K
            </kbd>
            <Separator orientation="vertical" className="hidden h-4 sm:block" />
            <NotificationBell />
          </div>
        </header>
        <CommandPalette />
        <div
          id="main-content"
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto p-4 sm:p-6"
        >
          <Outlet />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
