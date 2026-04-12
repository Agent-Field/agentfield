import { useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Database,
  FileText,
  Eye,
  Maximize2,
} from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { normalizeExecutionStatus } from "../../utils/status";
import { CollapsibleSection } from "./CollapsibleSection";
import { Button } from "../ui/button";
import { CopyButton } from "../ui/copy-button";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import { DataModal } from "./EnhancedModal";

interface EnhancedDataPanelProps {
  execution: WorkflowExecution;
  type: "input" | "output";
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function EnhancedDataPanel({ execution, type }: EnhancedDataPanelProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const data = type === "input" ? execution.input_data : execution.output_data;
  const size = type === "input" ? execution.input_size : execution.output_size;
  const icon = type === "input" ? ArrowDown : ArrowUp;
  const title = type === "input" ? "Input Data" : "Output Data";

  const hasData = data && Object.keys(data).length > 0;
  const jsonString = hasData ? JSON.stringify(data, null, 2) : "";
  const isLargeContent = jsonString.length > 1000; // Show modal button for large content

  const status = normalizeExecutionStatus(execution.status);

  const getEmptyStateContent = () => {
    if (type === "output") {
      if (status === "running") {
        return {
          icon: FileText,
          title: "Execution in progress",
          description:
            "Output data will appear here when the execution completes",
        };
      }
      if (status === "failed") {
        return {
          icon: Database,
          title: "Execution failed",
          description: "No output data was generated due to execution failure",
        };
      }
      if (status === "succeeded") {
        return {
          icon: Database,
          title: "No output data",
          description:
            "This execution completed successfully but didn't return any data",
        };
      }
    }

    return {
      icon: Database,
      title: "No input data",
      description: "This execution was started without input parameters",
    };
  };

  const emptyState = getEmptyStateContent();
  const EmptyIcon = emptyState.icon;

  const badge = (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">
        {formatBytes(size)}
      </span>
      {hasData && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsModalOpen(true)}
          className="h-6 w-6 p-0"
          title="Expand in modal"
        >
          <Maximize2 className="h-3 w-3" />
        </Button>
      )}
    </div>
  );

  return (
    <>
      <CollapsibleSection
        title={title}
        icon={icon}
        badge={badge}
        defaultOpen={true}
        contentClassName="p-0"
      >
        {hasData ? (
          <div className="p-4">
            <div className="border border-border rounded-lg overflow-hidden">
              <div className="flex items-center justify-between p-3 border-b border-border bg-muted/30">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">
                    Data Preview
                  </span>
                  {isLargeContent && (
                    <span className="text-sm text-muted-foreground">
                      (Showing preview)
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <CopyButton
                    value={jsonString}
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                    tooltip="Copy data"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsModalOpen(true)}
                    className="h-6 text-xs px-2"
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    View Full
                  </Button>
                </div>
              </div>
              <div className="p-4 bg-background max-h-[300px] overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                <UnifiedJsonViewer
                  data={data}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 text-center text-muted-foreground">
            <EmptyIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">{emptyState.title}</p>
            <p className="text-xs mt-1">{emptyState.description}</p>
          </div>
        )}
      </CollapsibleSection>

      {/* Enhanced Modal with Advanced JSON Viewer */}
      <DataModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={title}
        icon={icon}
        data={data}
      />
    </>
  );
}
