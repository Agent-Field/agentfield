import { NavLink } from "react-router-dom";

import type { NavigationSection } from "./types";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { Icon } from "@/components/ui/icon";
import { cn } from "@/lib/utils";

interface SidebarNewProps {
  sections: NavigationSection[];
}

export function SidebarNew({ sections }: SidebarNewProps) {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <Sidebar collapsible="icon" className="border-r border-border/40 bg-sidebar/95 backdrop-blur supports-[backdrop-filter]:bg-sidebar/60">
      {/* Header - Add bottom spacing and subtle border separator for visual hierarchy */}
      <SidebarHeader className="pb-3 border-b border-border/40">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild className="active:scale-[0.98] transition-transform">
              <NavLink to="/dashboard">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground shadow-sm">
                  <Icon name="dashboard" size={16} />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold tracking-tight">AgentField</span>
                  <span className="truncate text-[10px] text-muted-foreground font-mono">v1.0.0</span>
                </div>
              </NavLink>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      {/* Content — groups provide their own p-2; no extra px here so
           collapsed icons (32 px) centre inside the 48 px rail. */}
      <SidebarContent className="space-y-4">
        {sections.map((section) => (
          <SidebarGroup key={section.id} className="space-y-0.5">
            {/* Apply caption styling for clear header differentiation */}
            <SidebarGroupLabel className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground/70 px-2 mb-1">
              {section.title}
            </SidebarGroupLabel>
            {/* Add gap after header */}
            <SidebarGroupContent>
              <SidebarMenu>
                {section.items.map((item) => (
                  <SidebarMenuItem key={item.id}>
                    {item.disabled ? (
                      <SidebarMenuButton
                        isActive={false}
                        tooltip={isCollapsed ? item.label : undefined}
                        disabled
                        className="h-8 text-[13px]"
                      >
                        {item.icon && <Icon name={item.icon} size={15} />}
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    ) : (
                      <NavLink to={item.href} className="block">
                        {({ isActive }) => (
                          <SidebarMenuButton
                            asChild
                            isActive={isActive}
                            tooltip={isCollapsed ? item.label : undefined}
                            className={cn(
                              "h-8 text-[13px] transition-all duration-200 relative",
                              isActive
                                ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium shadow-sm"
                                : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50"
                            )}
                          >
                            <span className="flex items-center gap-2.5">
                              {isActive && (
                                <div className="absolute left-0 top-1/2 -translate-y-1/2 h-4 w-0.5 bg-primary rounded-r-full" />
                              )}
                              {item.icon && <Icon name={item.icon} size={15} className={cn(isActive ? "text-primary" : "text-muted-foreground")} />}
                              <span>{item.label}</span>
                            </span>
                          </SidebarMenuButton>
                        )}
                      </NavLink>
                    )}
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter className="border-t border-border/40 pt-2">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Documentation">
              <a href="https://agentfield.ai/docs" target="_blank" rel="noopener noreferrer">
                <Icon name="documentation" size={15} className="text-muted-foreground" />
                <span>Documentation</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="GitHub">
              <a href="https://github.com/Agent-Field/agentfield" target="_blank" rel="noopener noreferrer">
                <Icon name="github" size={15} className="text-muted-foreground" />
                <span>GitHub</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Support">
              <a href="https://github.com/Agent-Field/agentfield/issues" target="_blank" rel="noopener noreferrer">
                <Icon name="support" size={15} className="text-muted-foreground" />
                <span>Support</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
