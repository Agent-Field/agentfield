import {
  CheckCircle,
  XCircle,
  Clock,
  ExternalLink,
  PauseCircle,
  Timer,
} from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { Badge } from "../ui/badge";
import { CollapsibleSection } from "./CollapsibleSection";
import { CopyButton } from "../ui/copy-button";
import { cn } from "../../lib/utils";

interface ExecutionApprovalPanelProps {
  execution: WorkflowExecution;
}

function getApprovalStatusConfig(status?: string) {
  switch (status) {
    case "approved":
      return {
        label: "Approved",
        icon: CheckCircle,
        badgeVariant: "success" as const,
        color: "text-green-500",
        description: "The plan was approved by a human reviewer.",
      };
    case "rejected":
      return {
        label: "Rejected",
        icon: XCircle,
        badgeVariant: "failed" as const,
        color: "text-red-500",
        description: "The plan was rejected by a human reviewer.",
      };
    case "expired":
      return {
        label: "Expired",
        icon: Timer,
        badgeVariant: "pending" as const,
        color: "text-muted-foreground",
        description: "The approval request expired before a decision was made.",
      };
    case "pending":
    default:
      return {
        label: "Pending",
        icon: PauseCircle,
        badgeVariant: "pending" as const,
        color: "text-amber-500",
        description: "Waiting for a human reviewer to approve or reject the plan.",
      };
  }
}

function formatTimestamp(ts?: string): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function parseFeedback(response?: string): string | null {
  if (!response) return null;
  try {
    const parsed = JSON.parse(response);
    if (parsed.feedback) return parsed.feedback;
    if (parsed.decision) return null; // just the decision, no feedback text
    return response;
  } catch {
    return response;
  }
}

export function ExecutionApprovalPanel({ execution }: ExecutionApprovalPanelProps) {
  const hasApproval = !!execution.approval_request_id;

  if (!hasApproval) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <PauseCircle className="w-10 h-10 text-muted-foreground/40 mb-3" />
        <p className="text-sm text-muted-foreground text-muted-foreground">
          No approval request for this execution.
        </p>
      </div>
    );
  }

  const config = getApprovalStatusConfig(execution.approval_status);
  const StatusIcon = config.icon;
  const feedback = parseFeedback(execution.approval_response);

  return (
    <div className="space-y-6">
      {/* Approval Status Hero */}
      <CollapsibleSection
        title="Approval Status"
        icon={PauseCircle}
        defaultOpen={true}
        badge={
          <Badge variant={config.badgeVariant} size="sm">
            <StatusIcon className="w-3 h-3 mr-1" />
            {config.label}
          </Badge>
        }
      >
        <div className="p-4 space-y-4">
          <p className="text-sm text-muted-foreground text-muted-foreground">
            {config.description}
          </p>

          {/* Timeline */}
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 rounded-full bg-amber-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Clock className="w-3 h-3 text-amber-500" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground">Approval Requested</p>
                <p className="text-sm text-muted-foreground text-muted-foreground font-mono">
                  {formatTimestamp(execution.approval_requested_at)}
                </p>
              </div>
            </div>

            {execution.approval_responded_at && (
              <div className="flex items-start gap-3">
                <div className={cn(
                  "w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5",
                  execution.approval_status === "approved" ? "bg-green-500/10" : "bg-red-500/10"
                )}>
                  <StatusIcon className={cn("w-3 h-3", config.color)} />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground">
                    Decision: {config.label}
                  </p>
                  <p className="text-sm text-muted-foreground text-muted-foreground font-mono">
                    {formatTimestamp(execution.approval_responded_at)}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Feedback */}
          {feedback && (
            <div className="border-t border-border pt-3">
              <h4 className="text-sm font-medium text-foreground mb-2">Reviewer Feedback</h4>
              <div className="bg-muted/30 rounded-md p-3">
                <p className="text-sm text-muted-foreground text-foreground whitespace-pre-wrap">{feedback}</p>
              </div>
            </div>
          )}
        </div>
      </CollapsibleSection>

      {/* Request Details */}
      <CollapsibleSection
        title="Request Details"
        icon={Clock}
        defaultOpen={false}
      >
        <div className="p-4 space-y-3 text-sm">
          {/* Request ID */}
          <div className="flex items-center justify-between gap-2">
            <label className="text-muted-foreground flex-shrink-0">Request ID</label>
            <div className="flex items-center gap-1.5 min-w-0">
              <code className="font-mono text-xs text-foreground bg-muted/30 px-1.5 py-0.5 rounded truncate">
                {execution.approval_request_id}
              </code>
              <CopyButton
                value={execution.approval_request_id!}
                tooltip="Copy request ID"
                className="h-6 w-6 rounded-md [&_svg]:!h-3 [&_svg]:!w-3 flex-shrink-0"
              />
            </div>
          </div>

          {/* Review URL */}
          {execution.approval_request_url && (
            <div className="flex items-center justify-between gap-2">
              <label className="text-muted-foreground flex-shrink-0">Review Page</label>
              <a
                href={execution.approval_request_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs font-medium text-foreground hover:underline flex items-center gap-1"
              >
                Open in Hub
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}

          {/* Timestamps */}
          <div className="flex items-center justify-between gap-2">
            <label className="text-muted-foreground flex-shrink-0">Requested At</label>
            <span className="font-mono text-xs text-foreground">
              {formatTimestamp(execution.approval_requested_at)}
            </span>
          </div>

          {execution.approval_responded_at && (
            <div className="flex items-center justify-between gap-2">
              <label className="text-muted-foreground flex-shrink-0">Responded At</label>
              <span className="font-mono text-xs text-foreground">
                {formatTimestamp(execution.approval_responded_at)}
              </span>
            </div>
          )}
        </div>
      </CollapsibleSection>
    </div>
  );
}
