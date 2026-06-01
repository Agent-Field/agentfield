import { ArrowLeft } from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import type { WorkflowExecution } from "../../types/executions";

interface VCStatus {
  has_vc: boolean;
  vc_id?: string;
  status: string;
  created_at?: string;
  vc_document?: unknown;
}

interface ExecutionHeaderProps {
  execution: WorkflowExecution;
  vcStatus?: VCStatus | null;
  vcLoading?: boolean;
  onNavigateBack?: () => void;
}

export function ExecutionHeader({
  execution,
  onNavigateBack,
}: ExecutionHeaderProps) {
  const statusVariantMap: Record<string, "success" | "failed" | "running" | "pending" | "secondary"> = {
    completed: "success",
    failed: "failed",
    running: "running",
    pending: "pending",
  };

  const variant = statusVariantMap[execution.status] ?? "secondary";

  return (
    <div className="flex items-center gap-3 pb-2">
      {onNavigateBack && (
        <Button variant="ghost" size="sm" onClick={onNavigateBack} className="h-8 w-8 p-0">
          <ArrowLeft className="h-4 w-4" />
        </Button>
      )}
      <div className="flex flex-col gap-0.5 min-w-0">
        <h1 className="text-lg font-semibold truncate">
          Execution: {execution.execution_id}
        </h1>
        <div className="flex items-center gap-2">
          <Badge variant={variant} className="text-xs">
            {execution.status}
          </Badge>
          {execution.reasoner_id && (
            <span className="text-xs text-muted-foreground truncate">
              {execution.reasoner_id}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
