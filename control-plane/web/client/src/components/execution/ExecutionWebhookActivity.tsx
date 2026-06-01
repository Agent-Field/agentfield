import { RadioTower, AlertTriangle, CheckCircle2 } from "@/components/ui/icon-bridge";
import type { WorkflowExecution, ExecutionWebhookEvent } from "../../types/executions";
import { Badge } from "../ui/badge";
import { cn } from "../../lib/utils";
import { formatWebhookStatusLabel } from "../../utils/webhook";
import { Button } from "../ui/button";

interface ExecutionWebhookActivityProps {
  execution: WorkflowExecution;
  onRetry?: () => Promise<void> | void;
  isRetrying?: boolean;
  retryError?: string | null;
}

function sortEvents(events: ExecutionWebhookEvent[]): ExecutionWebhookEvent[] {
  return [...events].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  );
}

function getStatusTone(status?: string, hasError?: boolean): "success" | "danger" | "muted" {
  if (hasError) return "danger";
  const normalized = status?.toLowerCase();
  if (normalized === "failed") return "danger";
  if (normalized === "succeeded" || normalized === "delivered" || normalized === "success") {
    return "success";
  }
  return "muted";
}

export function ExecutionWebhookActivity({ execution, onRetry, isRetrying, retryError }: ExecutionWebhookActivityProps) {
  const events = Array.isArray(execution.webhook_events)
    ? sortEvents(execution.webhook_events)
    : [];

  const registered = Boolean(execution.webhook_registered || events.length > 0);
  const successCount = events.filter((event) => {
    const status = event.status?.toLowerCase();
    return status === "succeeded" || status === "success" || status === "delivered";
  }).length;
  const failureCount = events.filter((event) => {
    const status = event.status?.toLowerCase();
    return status === "failed" || Boolean(event.error_message);
  }).length;
  const pending = registered && events.length === 0 && failureCount === 0;

  return (
    <div className="border border-border rounded-lg bg-card/60 overflow-hidden">
      <div className="px-4 py-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
            <RadioTower className="h-4 w-4" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-semibold text-foreground">Webhook Activity</p>
            <p className="text-sm text-muted-foreground">
              Monitor callback deliveries and diagnose errors at a glance.
            </p>
          </div>
        </div>
        {onRetry && (
          <div className="flex flex-col sm:items-end gap-1 text-xs">
            <Button
              variant="secondary"
              size="sm"
              className="h-8"
              onClick={() => void onRetry()}
              disabled={Boolean(isRetrying)}
            >
              {isRetrying ? (
                "Retrying…"
              ) : (
                <span className="inline-flex items-center gap-1">
                  <RadioTower className="h-3 w-3" /> Retry webhook
                </span>
              )}
            </Button>
            {retryError && (
              <span className="text-destructive text-sm text-muted-foreground">{retryError}</span>
            )}
          </div>
        )}
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="outline" className={cn(
            "border-emerald-500/40 text-emerald-500",
            successCount === 0 && "border-border text-muted-foreground"
          )}>
            <CheckCircle2 className="h-3 w-3" /> {successCount} delivered
          </Badge>
          <Badge variant="outline" className={cn(
            "border-destructive/40 text-destructive",
            failureCount === 0 && "border-border text-muted-foreground"
          )}>
            <AlertTriangle className="h-3 w-3" /> {failureCount} failed
          </Badge>
          {pending && (
            <Badge variant="outline" className="border-border text-muted-foreground">
              pending
            </Badge>
          )}
        </div>
      </div>

      <div className="border-t border-border">
        {events.length > 0 ? (
          <div className="divide-y divide-border/60">
            {events.map((event) => {
              const tone = getStatusTone(event.status, Boolean(event.error_message));
              return (
                <div
                  key={event.id}
                  className="px-4 py-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="flex items-center gap-3">
                    <Badge
                      variant="outline"
                      className={cn(
                        "text-sm text-muted-foreground uppercase tracking-wide",
                        tone === "success" && "border-emerald-500/40 text-emerald-500",
                        tone === "danger" && "border-destructive/40 text-destructive",
                      )}
                    >
                      {formatWebhookStatusLabel(event.status)}
                    </Badge>
                    {event.http_status && (
                      <span className="text-sm text-muted-foreground text-muted-foreground">
                        HTTP {event.http_status}
                      </span>
                    )}
                  </div>
                  <div className="flex-1 text-sm text-muted-foreground sm:text-right">
                    {new Date(event.created_at).toLocaleString()}
                  </div>
                  {event.error_message && (
                    <div className="text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded px-3 py-2 sm:max-w-xs">
                      {event.error_message}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-4 text-sm text-muted-foreground">
            {pending && "Webhook registered – waiting for the first delivery."}
            {!pending && !registered && "No webhook was registered for this execution."}
            {!pending && registered && !events.length &&
              "Webhook registered – no deliveries were recorded for this run."}
          </div>
        )}
      </div>
    </div>
  );
}
