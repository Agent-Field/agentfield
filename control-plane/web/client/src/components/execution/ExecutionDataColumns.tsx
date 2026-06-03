import { useState } from "react";
import { FileText, Database } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { normalizeExecutionStatus } from "../../utils/status";
import { UnifiedDataPanel } from "../ui/UnifiedDataPanel";
import { DataModal } from "./EnhancedModal";
import { ResizableSplitPane, useResponsiveSplitPane } from "../ui/ResizableSplitPane";

interface ExecutionDataColumnsProps {
  execution: WorkflowExecution;
}

interface DataPanelProps {
  execution: WorkflowExecution;
  type: "input" | "output";
}

function DataPanel({ execution, type }: DataPanelProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const data = type === "input" ? execution.input_data : execution.output_data;
  const size = type === "input" ? execution.input_size : execution.output_size;
  const title = type === "input" ? "Input Data" : "Output Data";

  const status = normalizeExecutionStatus(execution.status);

  const getEmptyStateConfig = () => {
    if (type === "output") {
      if (status === "running") {
        return {
          icon: FileText,
          title: "Execution in progress",
          description: "Output data will appear here when the execution completes",
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
          description: "This execution completed successfully but didn't return any data",
        };
      }
    }

    return {
      icon: Database,
      title: "No input data",
      description: "This execution was started without input parameters",
    };
  };

  return (
    <div className="h-full min-h-0 flex flex-col">
      <UnifiedDataPanel
        data={data}
        title={title}
        type={type}
        size={size}
        emptyStateConfig={getEmptyStateConfig()}
        onModalOpen={() => setIsModalOpen(true)}
        maxHeight="none"
        className="flex-1 min-h-0"
      />

      {/* Enhanced Modal */}
      <DataModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={title}
        data={data}
      />
    </div>
  );
}

export function ExecutionDataColumns({ execution }: ExecutionDataColumnsProps) {
  // Use responsive behavior - vertical on small screens, horizontal on large screens
  const { isSmallScreen } = useResponsiveSplitPane(1024);
  const leftPanelPadding = isSmallScreen ? "pb-4" : "pr-6";
  const rightPanelPadding = isSmallScreen ? "pt-4" : "pl-6";

  return (
    <div className="h-full min-h-0 overflow-hidden">
      <ResizableSplitPane
        defaultSizePercent={isSmallScreen ? 100 : 50}
        minSizePercent={25}
        maxSizePercent={75}
        orientation={isSmallScreen ? "vertical" : "horizontal"}
        className="min-h-0"
        collapsible={true}
        leftPanelClassName={leftPanelPadding}
        rightPanelClassName={rightPanelPadding}
      >
        <div className="h-full min-h-0 flex flex-col">
          <DataPanel execution={execution} type="input" />
        </div>
        <div className="h-full min-h-0 flex flex-col">
          <DataPanel execution={execution} type="output" />
        </div>
      </ResizableSplitPane>
    </div>
  );
}
