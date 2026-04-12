import { Link, useLocation } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import logoShortLight from "@/assets/logos/logo-short-light-v2.svg?url";
import logoShortDark from "@/assets/logos/logo-short-dark-v2.svg?url";
import { navigation, resourceLinks } from "@/config/navigation";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { cn } from "@/lib/utils";

function SidebarLogo() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <SidebarMenu className="group-data-[collapsible=icon]:items-center">
      <SidebarMenuItem>
        <SidebarMenuButton
          asChild
          size="lg"
          tooltip="AgentField"
          className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
        >
          <Link to="/dashboard">
            <span className="relative size-8 shrink-0">
              <img
                src={logoShortLight}
                alt=""
                width={32}
                height={32}
                decoding="async"
                className="size-8 rounded-xl object-cover dark:hidden"
              />
              <img
                src={logoShortDark}
                alt=""
                width={32}
                height={32}
                decoding="async"
                className="hidden size-8 rounded-xl object-cover dark:block"
              />
            </span>
            {!isCollapsed && (
              <div className="flex min-w-0 flex-col gap-0.5 leading-none">
                <span className="truncate font-semibold tracking-tight text-sidebar-foreground">
                  AgentField
                </span>
                <span className="truncate text-xs font-normal text-sidebar-foreground/65">
                  Control Plane
                </span>
              </div>
            )}
          </Link>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}

export function AppSidebar() {
  const location = useLocation();

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader className="gap-0 group-data-[collapsible=icon]:px-1 group-data-[collapsible=icon]:py-1.5">
        <div
          className={cn(
            "flex w-full min-w-0 items-center gap-2",
            "group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:gap-2"
          )}
        >
          <div className="min-w-0 flex-1 group-data-[collapsible=icon]:flex-none group-data-[collapsible=icon]:w-full">
            <SidebarLogo />
          </div>
          <ModeToggle
            className={cn(
              "size-10 shrink-0 rounded-md text-sidebar-foreground",
              "group-data-[collapsible=icon]:size-8",
              "hover:bg-[var(--sidebar-hover)] hover:text-sidebar-accent-foreground",
              "focus-visible:ring-2 focus-visible:ring-sidebar-ring"
            )}
          />
        </div>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        {navigation.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => {
                  const active =
                    item.path === location.pathname ||
                    location.pathname.startsWith(`${item.path}/`);

                  return (
                    <SidebarMenuItem key={item.path}>
                      <SidebarMenuButton
                        asChild
                        isActive={active}
                        tooltip={item.title}
                      >
                        <Link to={item.path}>
                          <item.icon />
                          <span>{item.title}</span>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}

        <SidebarGroup>
          <SidebarGroupLabel>Resources</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {resourceLinks.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton asChild tooltip={item.title}>
                    <a
                      href={item.href}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <item.icon />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
