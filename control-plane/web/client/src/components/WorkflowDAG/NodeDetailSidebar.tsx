import { Close } from "@/components/ui/icon-bridge";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { statusTone } from "../../lib/theme";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader } from "../ui/card";
import { Skeleton } from "../ui/skeleton";
import { useNodeDetails } from "./hooks/useNodeDetails";
import { DataSection } from "./sections/DataSection";
import { ExecutionHeader } from "./sections/ExecutionHeader";
import { TechnicalSection } from "./sections/TechnicalSection";
import { TimingSection } from "./sections/TimingSection";

interface WorkflowNodeData {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  workflow_depth: number;
  task_name?: string;
  agent_name?: string;
}

interface NodeDetailSidebarProps {
  node: WorkflowNodeData | null;
  isOpen: boolean;
  onClose: () => void;
}

export function NodeDetailSidebar({
  node,
  isOpen,
  onClose,
}: NodeDetailSidebarProps) {
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const { nodeDetails, loading, error, refetch } = useNodeDetails(
    node?.execution_id
  );

  // Handle copy to clipboard
  const handleCopy = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(label);
      setTimeout(() => setCopySuccess(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  // Focus management
  useEffect(() => {
    if (isOpen) {
      // Focus the close button when sidebar opens
      const closeButton = document.querySelector(
        "[data-sidebar-close]"
      ) as HTMLElement;
      closeButton?.focus();
    }
  }, [isOpen]);

  // Refresh data when node changes
  useEffect(() => {
    if (node && isOpen) {
      refetch();
    }
  }, [node?.execution_id, isOpen, refetch]);

  if (!isOpen || !node) {
    return null;
  }

  const sidebarContent = (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 z-[70] bg-background/80 backdrop-blur-sm transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div
        className={cn(
          "fixed top-0 right-0 z-[80] flex h-full w-full max-w-full flex-col transition-transform duration-300 ease-out",
          "border-l border-border bg-card/95 backdrop-blur-xl",
          "shadow-[0px_24px_60px_-28px_color-mix(in_srgb,_var(--foreground)_18%,_transparent)]",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby="sidebar-title"
        style={{ width: "min(560px, 100vw)" }}
      >
        {/* Header - Fixed */}
        <div className="flex flex-shrink-0 items-center justify-between border-b border-border/50 px-5 py-4 md:px-6">
          <div className="flex-1 min-w-0">
            <h2
              id="sidebar-title"
              className="truncate text-base font-semibold text-foreground"
            >
              Execution Details
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {node.task_name || node.reasoner_id}
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="ml-4 h-8 w-8 p-0 hover:bg-muted"
            data-sidebar-close
            aria-label="Close sidebar"
          >
            <Close size={16} className="text-muted-foreground" />
          </Button>
        </div>

        {/* Content - Scrollable */}
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
          <div className="space-y-5 px-5 py-5 md:space-y-6 md:px-6 md:py-6">
            {loading ? (
              <SidebarSkeleton />
            ) : error ? (
              <ErrorState error={error} onRetry={refetch} />
            ) : (
              <>
                {/* Execution Header with Navigation */}
                <ExecutionHeader
                  node={node}
                  details={nodeDetails}
                  onCopy={handleCopy}
                  copySuccess={copySuccess}
                />

                {/* Data Section - Input/Output (PRIORITY #1) */}
                <DataSection node={node} details={nodeDetails} />

                {/* Timing Section */}
                <TimingSection node={node} details={nodeDetails} />

                {/* Technical Section */}
                <TechnicalSection
                  node={node}
                  details={nodeDetails}
                  onCopy={handleCopy}
                  copySuccess={copySuccess}
                />
              </>
            )}
          </div>
        </div>

        {/* Footer - Fixed */}
        <div className="flex-shrink-0 border-t border-border/50 px-5 py-4 md:px-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground/70">
            <span>Last updated: {new Date().toLocaleTimeString()}</span>
            {node.status === "running" && (
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-status-info" />
                <span>Live updates</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );

  // Render using portal to escape parent stacking context
  return createPortal(sidebarContent, document.body);
}

// Loading skeleton
function SidebarSkeleton() {
  return (
    <div className="space-y-6">
      {[...Array(5)].map((_, i) => (
        <Card
          key={i}
          className="border border-border bg-card"
        >
          <CardHeader className="pb-2">
            <Skeleton className="h-4 w-24 bg-muted/50" />
          </CardHeader>
          <CardContent className="space-y-2">
            <Skeleton className="h-3 w-full bg-muted/50" />
            <Skeleton className="h-3 w-3/4 bg-muted/50" />
            <Skeleton className="h-3 w-1/2 bg-muted/50" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Error state
function ErrorState({
  error,
  onRetry,
}: {
  error: string;
  onRetry: () => void;
}) {
  const errorTone = statusTone.error;

  return (
    <Card className={cn(errorTone.bg, errorTone.border)}>
      <CardContent className="py-8 text-center">
        <div className={cn("mb-4 text-2xl font-semibold tracking-tight", errorTone.accent)}>
          <Close size={24} className="mx-auto" />
        </div>
        <h3 className="mb-2 text-base font-semibold">
          Failed to load execution details
        </h3>
        <p className="mb-4 text-sm text-muted-foreground">{error}</p>
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          className="text-xs"
        >
          Try Again
        </Button>
      </CardContent>
    </Card>
  );
}
