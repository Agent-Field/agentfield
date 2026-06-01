import React from "react";
import { cn } from "../../lib/utils";

interface ExecutionDetailsLayoutProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * Main layout container for execution details - follows Linear.app centered layout pattern
 * Provides consistent max-width and spacing for all execution detail views
 */
export function ExecutionDetailsLayout({ children, className }: ExecutionDetailsLayoutProps) {
  return (
    <div className={cn("min-h-screen bg-background", className)}>
      {/* Centered container with max width - matches dashboard pattern */}
      <div className="flex justify-center">
        <div className="max-w-4xl w-full px-4 py-6 space-y-6">
          {children}
        </div>
      </div>
    </div>
  );
}
