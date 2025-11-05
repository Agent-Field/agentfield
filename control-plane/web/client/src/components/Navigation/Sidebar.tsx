import { NavLink } from 'react-router-dom';
import { Icon } from '@/components/ui/icon';
import { cn } from '../../lib/utils';
import type { NavigationSection } from './types';
import { NavigationSectionComponent } from './NavigationSection';
import { typography } from '../../styles/theme';

interface SidebarProps {
  sections: NavigationSection[];
  isCollapsed?: boolean;
  onToggle?: () => void;
}

export function Sidebar({ sections, isCollapsed = false }: SidebarProps) {
  return (
    <aside className={cn(
      "h-screen bg-gradient-to-b from-bg-base to-bg-subtle backdrop-blur-sm",
      "shadow-none border-none", // Borderless design
      isCollapsed ? "w-14" : "w-56", // Slightly narrower for better proportions
      "transition-all duration-200 ease-out"
    )}>
      <div className="flex flex-col h-full">
        {/* Header - Modernized with subtle separation */}
        <div className={cn(
          "relative", // Remove border, use background separation
          "bg-gradient-to-r from-bg-elevated/50 to-bg-overlay/30",
          "backdrop-blur-sm",
          isCollapsed ? "p-3" : "px-6 py-4",
          "transition-all duration-200"
        )}>
          <NavLink
            to="/dashboard"
            className={cn(
              "flex items-center",
              isCollapsed ? "justify-center" : "space-x-3",
              "group transition-colors duration-100"
            )}
          >
            <div className={cn(
              "rounded-lg bg-primary/10",
              "transition-colors duration-100",
              "group-hover:bg-primary/15",
              isCollapsed ? "p-2" : "p-2.5"
            )}>
              <Icon
                name="dashboard"
                size={isCollapsed ? 16 : 20}
                className="text-primary"
              />
            </div>
            {!isCollapsed && (
              <div>
                <h1 className={cn(
                  typography["heading-sm"],
                  "text-foreground font-semibold"
                )}>
                  Haxen
                </h1>
                <p className={cn(
                  typography["helper-text"],
                  "text-muted-foreground/70"
                )}>
                  Open Control Plane
                </p>
              </div>
            )}
          </NavLink>

          {/* Subtle bottom separator using gradient */}
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/20 to-transparent" />
        </div>

        {/* Navigation - Refined spacing with visual hierarchy */}
        <nav className={cn(
          "flex-1 overflow-y-auto",
          "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border",
          isCollapsed ? "py-4 px-2" : "py-6 px-2" // More generous top/bottom padding
        )}>
          <div className={cn(
            "space-y-8", // Optimal section spacing (32px) for clear grouping
            "transition-all duration-200"
          )}>
            {sections.map((section) => (
              <div
                key={section.id}
                className="transition-all duration-200"
              >
                <NavigationSectionComponent
                  section={section}
                  isCollapsed={isCollapsed}
                />
              </div>
            ))}
          </div>
        </nav>

        {/* Footer - Optional subtle branding */}
        <div className={cn(
          "relative mt-auto",
          "bg-gradient-to-r from-bg-elevated/30 to-bg-overlay/20",
          isCollapsed ? "p-2" : "p-4",
          "transition-all duration-200"
        )}>
          {/* Subtle top separator */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/20 to-transparent" />

          {!isCollapsed && (
            <div className="text-body-small/50 text-center">
              v{import.meta.env.VITE_APP_VERSION || "1.0.0"}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
