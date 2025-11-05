import { NavLink } from "react-router-dom";

import type { NavigationSection } from "./types";
import {
  Sidebar,
  SidebarContent,
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

interface SidebarNewProps {
  sections: NavigationSection[];
}

export function SidebarNew({ sections }: SidebarNewProps) {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <Sidebar collapsible="icon">
      {/* Header - Add bottom spacing and subtle border separator for visual hierarchy */}
      <SidebarHeader className="pb-4 border-b border-border-secondary">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <NavLink to="/dashboard">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <Icon name="dashboard" size={16} />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Haxen</span>
                  <span className="truncate text-xs">Open Control Plane</span>
                </div>
              </NavLink>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      {/* Content - Add spacing between groups */}
      <SidebarContent className="space-y-6">
        {sections.map((section) => (
          <SidebarGroup key={section.id} className="space-y-1">
            {/* Apply caption styling for clear header differentiation */}
            <SidebarGroupLabel className="text-caption text-nav-text-tertiary">
              {section.title}
            </SidebarGroupLabel>
            {/* Add gap after header */}
            <SidebarGroupContent className="mt-4">
              <SidebarMenu>
                {section.items.map((item) => (
                  <SidebarMenuItem key={item.id}>
                    {item.disabled ? (
                      <SidebarMenuButton
                        isActive={false}
                        tooltip={isCollapsed ? item.label : undefined}
                        disabled
                      >
                        {item.icon && <Icon name={item.icon} size={16} />}
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    ) : (
                      <NavLink to={item.href} className="block">
                        {({ isActive }) => (
                          <SidebarMenuButton
                            asChild
                            isActive={isActive}
                            tooltip={isCollapsed ? item.label : undefined}
                          >
                            <span className="flex items-center gap-2">
                              {item.icon && <Icon name={item.icon} size={16} />}
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
    </Sidebar>
  );
}
