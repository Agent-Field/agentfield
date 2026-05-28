import {
  Activity,
  Check,
  ChevronLeft,
  ChevronRight,
  Users,
} from "@/components/ui/icon-bridge";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";

interface WorkflowBreadcrumbProps {
  execution: WorkflowExecution;
  onNavigateBack: () => void;
}

function CopyableId({ id, label }: { id: string; label: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const displayId = id.length > 8 ? `${id.slice(0, 8)}...` : id;

  return (
    <div className="flex items-center gap-1">
      <span>{label} </span>
      <code
        className="font-mono text-xs bg-muted/50 px-1 py-0.5 rounded cursor-pointer hover:bg-muted/80 transition-colors"
        title={`${label}: ${id} (click to copy)`}
        onClick={handleCopy}
      >
        {displayId}
      </code>
      {copied && <Check className="w-3 h-3 text-green-500" />}
    </div>
  );
}

export function WorkflowBreadcrumb({
  execution,
  onNavigateBack,
}: WorkflowBreadcrumbProps) {
  const navigate = useNavigate();

  const navigateToWorkflow = () => {
    navigate(`/workflows/${execution.workflow_id}`);
  };

  const navigateToSession = () => {
    if (execution.session_id) {
      // For now, just navigate to executions filtered by session
      navigate(`/executions?session_id=${execution.session_id}`);
    }
  };

  return (
    <Card className="bg-muted/30 border-border/50">
      <CardContent className="py-3 px-4">
        <nav className="flex items-center gap-2 text-sm flex-wrap">
          <Button
            variant="ghost"
            size="sm"
            onClick={onNavigateBack}
            className="shrink-0"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Executions
          </Button>

          <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />

          <Button
            variant="ghost"
            size="sm"
            onClick={navigateToWorkflow}
            className="shrink-0"
          >
            <Activity className="w-4 h-4 mr-1" />
            <CopyableId id={execution.workflow_id} label="Workflow" />
          </Button>

          {execution.session_id && (
            <>
              <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
              <Button
                variant="ghost"
                size="sm"
                onClick={navigateToSession}
                className="shrink-0"
              >
                <Users className="w-4 h-4 mr-1" />
                <CopyableId id={execution.session_id} label="Session" />
              </Button>
            </>
          )}

          <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
          <span className="text-muted-foreground">Execution Detail</span>
        </nav>
      </CardContent>
    </Card>
  );
}
