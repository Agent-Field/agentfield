import React from "react";
import {
  ArrowDown,
  ArrowUp,
  Database,
  Maximize,
  InProgress,
} from "@/components/ui/icon-bridge";
import { Button } from "./button";
import { Badge } from "./badge";
import { UnifiedJsonViewer } from "./UnifiedJsonViewer";
import { CopyButton } from "./copy-button";
import { cn } from "../../lib/utils";

interface UnifiedDataPanelProps {
  data: any;
  title: string;
  type: "input" | "output";
  size?: number;
  className?: string;
  isLoading?: boolean;
  error?: string;
  emptyStateConfig?: {
    icon?: React.ComponentType<{ className?: string }>;
    title?: string;
    description?: string;
  };
  onModalOpen?: () => void;
  showModalButton?: boolean;
  maxHeight?: string;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function UnifiedDataPanel({
  data,
  title,
  type,
  size,
  className,
  isLoading = false,
  error,
  emptyStateConfig,
  onModalOpen,
  showModalButton = true,
  maxHeight = "none",
}: UnifiedDataPanelProps) {
  const IconComponent = type === "input" ? ArrowDown : ArrowUp;
  const hasData =
    data && (typeof data === "object" ? Object.keys(data).length > 0 : true);
  const jsonString = hasData ? JSON.stringify(data, null, 2) : "";

  // Default empty state configuration
  const defaultEmptyState = {
    icon: Database,
    title: type === "input" ? "No input data" : "No output data",
    description:
      type === "input"
        ? "This execution was started without input parameters"
        : "No output data was generated",
  };

  const emptyState = { ...defaultEmptyState, ...emptyStateConfig };
  const EmptyIcon = emptyState.icon;

  const handleHeaderClick = () => {
    if (hasData && onModalOpen) {
      onModalOpen();
    }
  };

  return (
    <div className={cn(
      "h-full min-h-0 flex flex-col overflow-hidden",
      className
    )}>
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between p-4 border-b border-border bg-muted/20 flex-shrink-0",
          hasData && onModalOpen && "cursor-pointer hover:bg-muted/30 transition-colors"
        )}
        onClick={handleHeaderClick}
        title={hasData && onModalOpen ? "Click to expand in modal" : undefined}
      >
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-full bg-background border border-border">
            <IconComponent className="w-4 h-4 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-medium text-foreground flex items-center gap-2">
              {title}
              {isLoading && (
                <InProgress className="w-3 h-3 animate-spin text-muted-foreground" />
              )}
            </h3>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              {size && <span>{formatBytes(size)}</span>}
              {hasData && (
                <>
                  {size && <span>•</span>}
                  <Badge variant="secondary" className="text-micro h-4 px-1.5">
                    {typeof data === "object" && !Array.isArray(data)
                      ? `${Object.keys(data).length} keys`
                      : Array.isArray(data)
                        ? `${data.length} items`
                        : "Data available"}
                  </Badge>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {error && (
            <Badge variant="destructive" className="text-micro h-5 px-2">
              Error
            </Badge>
          )}
          {hasData && (
            <CopyButton
              value={jsonString}
              tooltip="Copy data"
              className="h-6 w-6 [&_svg]:!h-3 [&_svg]:!w-3"
              onClick={(event: React.MouseEvent<HTMLButtonElement>) => {
                event.stopPropagation();
              }}
            />
          )}
          {hasData && showModalButton && onModalOpen && (
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onModalOpen();
              }}
              className="h-6 w-6 p-0"
              title="Expand in modal (or click header)"
            >
              <Maximize className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
        {isLoading ? (
          <div className="h-full flex items-center justify-center p-6 text-center text-muted-foreground">
            <div className="flex flex-col items-center gap-3">
              <InProgress className="w-8 h-8 animate-spin text-muted-foreground/50" />
              <p className="text-sm">Loading data...</p>
            </div>
          </div>
        ) : error ? (
          <div className="h-full flex items-center justify-center p-6 text-center">
            <div className="flex flex-col items-center gap-3 text-destructive">
              <Database className="w-8 h-8 text-destructive/50" />
              <div>
                <p className="text-sm font-medium">Failed to load data</p>
                <p className="text-sm text-muted-foreground mt-1">{error}</p>
              </div>
            </div>
          </div>
        ) : hasData ? (
          <div className="h-full min-h-0 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
            <UnifiedJsonViewer
              data={data}
              maxHeight={maxHeight}
              searchable={true}
              showHeader={false}
              className="border-0 rounded-none h-full"
            />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center p-6 text-center text-muted-foreground">
            <div className="flex flex-col items-center gap-3">
              <EmptyIcon className="w-8 h-8 text-muted-foreground/50" />
              <div>
                <p className="text-sm font-medium">{emptyState.title}</p>
                <p className="text-xs mt-1 max-w-xs">
                  {emptyState.description}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
