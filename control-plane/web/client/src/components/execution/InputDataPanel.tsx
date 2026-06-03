
import { ArrowDown, Database } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { CollapsibleSection } from "./CollapsibleSection";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";

interface InputDataPanelProps {
  execution: WorkflowExecution;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function InputDataPanel({ execution }: InputDataPanelProps) {
  const inputData = execution.input_data;
  const hasInputData = (() => {
    if (inputData === null || inputData === undefined) return false;
    if (typeof inputData === "string") return inputData.trim().length > 0;
    if (Array.isArray(inputData)) return inputData.length > 0;
    if (typeof inputData === "object") return Object.keys(inputData).length > 0;
    return Boolean(inputData);
  })();

  const badge = (
    <span className="text-sm text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">
      {formatBytes(execution.input_size)}
    </span>
  );

  return (
    <CollapsibleSection
      title="Input Data"
      icon={ArrowDown}
      badge={badge}
      defaultOpen={true}
      contentClassName="p-0"
    >
      {hasInputData ? (
        <UnifiedJsonViewer
          data={inputData}
        />
      ) : (
        <div className="p-6 text-center text-muted-foreground">
          <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No input data</p>
          <p className="text-xs mt-1">This execution was started without input parameters</p>
        </div>
      )}
    </CollapsibleSection>
  );
}

// Alias for backwards compatibility
export { InputDataPanel as RedesignedInputDataPanel };
