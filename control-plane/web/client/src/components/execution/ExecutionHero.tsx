import {
  ArrowDown,
  ArrowUp,
  Check,
  Clock,
  SpinnerGap,
  X,
  BracketsCurly,
  Stack,
  ShareNetwork,
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import type { VCStatusData } from "../../types/did";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import StatusIndicator from "../ui/status-indicator";
import { VerifiableCredentialBadge } from "../vc";
import { CopyButton } from "../ui/copy-button";
import { ExecutionTimeline } from "./ExecutionTimeline";
import { statusTone } from "@/lib/theme";
import { cn } from "@/lib/utils";

interface ExecutionHeroProps {
  execution: WorkflowExecution;
  vcStatus?: VCStatusData | null;
  vcLoading?: boolean;
}

function VCStatusCard({
  vcStatus,
  vcLoading,
  executionId,
}: {
  vcStatus?: VCStatusData | null;
  vcLoading?: boolean;
  executionId: string;
}) {
  if (vcLoading) {
    return (
      <div className="flex items-center gap-2 text-body-small">
        <SpinnerGap className="w-3 h-3 animate-spin" />
        <span>Checking VC...</span>
      </div>
    );
  }

  if (vcStatus?.has_vc) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <div className="flex items-center gap-1 text-muted-foreground">
          <Check className={cn("w-3 h-3", statusTone.success.accent)} />
          <span>VC</span>
        </div>
        {/* Use the proper VerifiableCredentialBadge component with custom styling */}
        <div className="[&_button]:h-6 [&_button]:px-2 [&_button]:text-xs [&_button]:bg-transparent [&_button]:border-0 [&_button]:text-primary [&_button:hover]:text-primary/80 [&_button:hover]:bg-muted/50 [&_button]:transition-colors">
          <VerifiableCredentialBadge
            hasVC={vcStatus.has_vc}
            status={vcStatus.status}
            vcData={vcStatus}
            executionId={executionId}
            showCopyButton={true}
            showVerifyButton={true}
            variant="detail"
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 text-body-small">
      <X className="w-3 h-3" />
      <span>No VC</span>
    </div>
  );
}

function formatDuration(durationMs?: number): string {
  if (!durationMs) return "N/A";

  if (durationMs < 1000) {
    return `${durationMs}ms`;
  } else if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";

  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function ExecutionHero({
  execution,
  vcStatus,
  vcLoading,
}: ExecutionHeroProps) {
  const navigate = useNavigate();

  const handleViewWorkflow = () => {
    navigate(`/workflows/${execution.workflow_id}`);
  };

  return (
    <Card className="shadow-lg">
      <CardContent className="p-6">
        {/* Main Header Section */}
        <div className="flex items-center justify-between mb-6">
          {/* Left: Primary Info */}
          <div className="flex items-center gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-heading-2">
                  {execution.reasoner_id}
                </h1>
                <StatusIndicator
                  status={execution.status}
                  animated={execution.status === "running"}
                  className="text-base"
                />
              </div>

              {/* Clean Identity Row - No Confusing "Verified" Labels */}
              <div className="flex items-center gap-4 text-body">
                <div className="flex items-center gap-2">
                  <span>Agent:</span>
                  <code className="font-mono bg-muted/50 px-2 py-0.5 rounded text-body-small">
                    {execution.agent_node_id}
                  </code>
                  <CopyButton
                    value={execution.agent_node_id}
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 p-0 hover:bg-muted/80 [&_svg]:h-3 [&_svg]:w-3"
                    tooltip="Copy agent node ID"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <span>DID:</span>
                  <DIDDisplay
                    nodeId={execution.agent_node_id}
                    variant="inline"
                    className="text-body-small"
                  />
                </div>

                {/* VC Status - Integrated inline */}
                <VCStatusCard
                  vcStatus={vcStatus}
                  vcLoading={vcLoading}
                  executionId={execution.execution_id}
                />
              </div>
            </div>
          </div>

          {/* Right: Key Metrics - Horizontal Layout */}
          <div className="flex items-center gap-6">
            <div className="text-center">
              <div className="text-heading-3 text-foreground">
                {formatDuration(execution.duration_ms)}
              </div>
              <div className="text-body-small flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Duration
              </div>
            </div>

            <div className="text-center">
              <div className="text-heading-3 text-foreground">
                {formatBytes(execution.input_size)}
              </div>
              <div className="text-body-small flex items-center gap-1">
                <ArrowDown className="w-3 h-3" />
                Input
              </div>
            </div>

            <div className="text-center">
              <div className="text-heading-3 text-foreground">
                {formatBytes(execution.output_size)}
              </div>
              <div className="text-body-small flex items-center gap-1">
                <ArrowUp className="w-3 h-3" />
                Output
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Actions - Clean Row */}
        <div className="flex items-center justify-between border-t border-border/50 pt-4">
          {/* Left: Navigation Buttons */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleViewWorkflow}
              className="flex items-center gap-2 h-8"
            >
              <Stack className="w-4 h-4" />
              Workflow
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                navigate(
                  `/reasoners/${execution.agent_node_id}.${execution.reasoner_id}`
                )
              }
              className="flex items-center gap-2 h-8"
            >
              <BracketsCurly className="w-4 h-4" />
              Reasoner
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate(`/nodes/${execution.agent_node_id}`)}
              className="flex items-center gap-2 h-8"
            >
              <ShareNetwork className="w-4 h-4" />
              Agent Node
            </Button>
          </div>

          {/* Right: Execution ID */}
          <div className="flex items-center gap-2 text-body-small">
            <span>ID:</span>
            <code
              className="font-mono bg-muted/50 px-2 py-0.5 rounded text-xs"
              title={execution.execution_id}
            >
              {execution.execution_id.slice(0, 8)}...
              {execution.execution_id.slice(-4)}
            </code>
            <CopyButton
              value={execution.execution_id}
              variant="ghost"
              size="icon"
              className="h-6 w-6 p-0 hover:bg-muted/80 [&_svg]:h-3 [&_svg]:w-3"
              tooltip="Copy execution ID"
            />
          </div>
        </div>

        {/* Timeline Bar - Minimal */}
        <div className="mt-4">
          <ExecutionTimeline execution={execution} />
        </div>
      </CardContent>
    </Card>
  );
}
