import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
  headerClassName?: string;
  contentClassName?: string;
  icon?: React.ComponentType<{ className?: string }>;
  badge?: React.ReactNode;
}

export function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
  className,
  headerClassName,
  contentClassName,
  icon: Icon,
  badge
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={cn("border border-border rounded-lg overflow-hidden", className)}>
      {/* Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-full flex items-center justify-between p-4",
          "hover:bg-muted/50 transition-colors duration-150",
          "text-left focus:outline-none focus:ring-2 focus:ring-ring focus:ring-inset",
          headerClassName
        )}
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="w-4 h-4 text-muted-foreground" />}
          <h3 className="text-sm font-medium text-foreground">{title}</h3>
          {badge}
        </div>

        <div className="flex items-center gap-2">
          {isOpen ? (
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          )}
        </div>
      </button>

      {/* Content */}
      {isOpen && (
        <div className={cn("border-t border-border bg-background", contentClassName)}>
          {children}
        </div>
      )}
    </div>
  );
}
