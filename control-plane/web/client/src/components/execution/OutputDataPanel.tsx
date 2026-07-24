
import { ArrowUp, FileText, CheckCircle, XCircle } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { CollapsibleSection } from "./CollapsibleSection";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";

interface OutputDataPanelProps {
  execution: WorkflowExecution;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function OutputDataPanel({ execution }: OutputDataPanelProps) {
  const outputData = execution.output_data;
  const hasOutputData = (() => {
    if (outputData === null || outputData === undefined) return false;
    if (typeof outputData === "string") return outputData.trim().length > 0;
    if (Array.isArray(outputData)) return outputData.length > 0;
    if (typeof outputData === "object") return Object.keys(outputData).length > 0;
    return Boolean(outputData);
  })();

  const normalizedStatus = execution.status?.toLowerCase() ?? "";
  const isCompleted = normalizedStatus === "succeeded";
  const isFailed = ["failed", "error", "timeout"].includes(normalizedStatus);
  const isRunning = ["running", "pending"].includes(normalizedStatus);

  const badge = (
    <span className="text-sm text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">
      {formatBytes(execution.output_size)}
    </span>
  );

  const getEmptyStateContent = () => {
    if (isRunning) {
      return {
        icon: FileText,
        title: "Execution in progress",
        description: "Output data will appear here when the execution completes"
      };
    }

    if (isFailed) {
      return {
        icon: XCircle,
        title: "Execution failed",
        description: "No output data was generated due to execution failure"
      };
    }

    if (isCompleted && !hasOutputData) {
      return {
        icon: CheckCircle,
        title: "No output data",
        description: "This execution completed successfully but didn't return any data"
      };
    }

    return {
      icon: FileText,
      title: "No output data",
      description: "Output data will appear here when available"
    };
  };

  const emptyState = getEmptyStateContent();
  const EmptyIcon = emptyState.icon;

  return (
    <CollapsibleSection
      title="Output Data"
      icon={ArrowUp}
      badge={badge}
      defaultOpen={true}
      contentClassName="p-0"
    >
      {hasOutputData ? (
        <UnifiedJsonViewer
          data={outputData}
        />
      ) : (
        <div className="p-6 text-center text-muted-foreground">
          <EmptyIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">{emptyState.title}</p>
          <p className="text-xs mt-1">{emptyState.description}</p>
        </div>
      )}
    </CollapsibleSection>
  );
}

// Alias for backwards compatibility
export { OutputDataPanel as RedesignedOutputDataPanel };
