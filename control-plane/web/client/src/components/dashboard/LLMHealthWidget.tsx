import { AlertTriangle, CircleAlert, CircleCheck, Clock, RefreshCw } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { formatRelativeTime } from "@/utils/dateFormat";
import type { LLMCircuitState, LLMEndpointHealth, LLMHealthResponse } from "@/hooks/queries";

type LLMHealthWidgetProps = {
  health: LLMHealthResponse | undefined;
  loading?: boolean;
  className?: string;
};

type BackendState = "healthy" | "degraded" | "down" | "disabled" | "unknown";

type EndpointLike = Partial<LLMEndpointHealth> & { name: string };

function getBackendState(health: LLMHealthResponse | undefined): BackendState {
  if (!health) return "unknown";
  if (!health.enabled) return "disabled";
  if (health.endpoints.length === 0) return "unknown";

  const healthyEndpoints = health.endpoints.filter((endpoint) => endpoint.healthy);
  if (healthyEndpoints.length === health.endpoints.length) return "healthy";
  if (healthyEndpoints.length > 0) return "degraded";
  return "down";
}

function stateMeta(state: BackendState) {
  switch (state) {
    case "healthy":
      return {
        label: "Healthy",
        badgeVariant: "success" as const,
        tone: "text-green-500",
        description: "All configured LLM endpoints are responding.",
      };
    case "degraded":
      return {
        label: "Degraded",
        badgeVariant: "pending" as const,
        tone: "text-amber-500",
        description: "At least one LLM endpoint is unhealthy or in recovery.",
      };
    case "down":
      return {
        label: "Down",
        badgeVariant: "failed" as const,
        tone: "text-destructive",
        description: "No configured LLM endpoints are currently healthy.",
      };
    case "disabled":
      return {
        label: "Disabled",
        badgeVariant: "outline" as const,
        tone: "text-muted-foreground",
        description: "LLM health monitoring is not configured.",
      };
    default:
      return {
        label: "Unknown",
        badgeVariant: "unknown" as const,
        tone: "text-muted-foreground",
        description: "LLM health status is unavailable.",
      };
  }
}

function circuitStateLabel(state: LLMCircuitState): string {
  switch (state) {
    case "closed":
      return "Closed";
    case "open":
      return "Open";
    case "half_open":
      return "Half-open";
  }
}

function circuitStateVariant(state: LLMCircuitState) {
  switch (state) {
    case "closed":
      return "success" as const;
    case "open":
      return "failed" as const;
    case "half_open":
      return "pending" as const;
  }
}

function endpointSummary(endpoint: LLMEndpointHealth) {
  if (endpoint.last_error?.trim()) {
    return endpoint.last_error.trim();
  }
  if (endpoint.last_success) {
    return `Last success ${formatRelativeTime(endpoint.last_success)}`;
  }
  return "No checks recorded yet.";
}

function normalizeEndpoint(endpoint: EndpointLike): LLMEndpointHealth {
  const healthy = Boolean(endpoint.healthy);
  const consecutive_failures = Number(endpoint.consecutive_failures ?? 0);
  const total_checks = Number(endpoint.total_checks ?? 0);
  const total_failures = Number(endpoint.total_failures ?? 0);

  return {
    name: endpoint.name,
    healthy,
    circuit_state:
      endpoint.circuit_state ??
      (healthy ? "closed" : "open"),
    consecutive_failures: Number.isFinite(consecutive_failures) ? consecutive_failures : 0,
    last_error: endpoint.last_error ?? endpoint.error,
    last_success: endpoint.last_success,
    last_checked: endpoint.last_checked,
    total_checks: Number.isFinite(total_checks) ? total_checks : undefined,
    total_failures: Number.isFinite(total_failures) ? total_failures : undefined,
    url: endpoint.url,
    model: endpoint.model,
    latency_ms: endpoint.latency_ms,
    error: endpoint.error,
  };
}

function EndpointRow({ endpoint }: { endpoint: EndpointLike }) {
  const normalized = normalizeEndpoint(endpoint);
  const circuitOpen = normalized.circuit_state === "open";
  const stateTone = normalized.healthy ? "text-foreground" : "text-destructive";

  return (
    <li
      className={cn(
        "rounded-lg border p-3 shadow-sm",
        circuitOpen ? "border-destructive/40 bg-destructive/5" : "border-border bg-card",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-medium" title={normalized.name}>
              {normalized.name}
            </span>
            <Badge variant={circuitStateVariant(normalized.circuit_state)} className="shrink-0">
              {circuitStateLabel(normalized.circuit_state)}
            </Badge>
            <Badge variant={normalized.healthy ? "success" : "failed"} className="shrink-0">
              {normalized.healthy ? "Healthy" : "Unhealthy"}
            </Badge>
          </div>
          <p className={cn("text-sm", stateTone)}>{endpointSummary(normalized)}</p>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="font-mono tabular-nums">
              {normalized.consecutive_failures} failure
              {normalized.consecutive_failures === 1 ? "" : "s"}
            </span>
            {normalized.last_checked ? (
              <>
                <Separator orientation="vertical" className="h-3" />
                <span>Checked {formatRelativeTime(normalized.last_checked)}</span>
              </>
            ) : null}
            {normalized.last_success ? (
              <>
                <Separator orientation="vertical" className="h-3" />
                <span>Last success {formatRelativeTime(normalized.last_success)}</span>
              </>
            ) : null}
          </div>
        </div>
        {circuitOpen ? (
          <Badge variant="destructive" showIcon={false} className="shrink-0 gap-1.5">
            <AlertTriangle className="size-3.5" aria-hidden />
            Circuit open
          </Badge>
        ) : (
          <Badge variant="outline" showIcon={false} className="shrink-0 gap-1.5">
            {normalized.healthy ? (
              <CircleCheck className="size-3.5 text-green-500" aria-hidden />
            ) : (
              <CircleAlert className="size-3.5 text-amber-500" aria-hidden />
            )}
            {normalized.healthy ? "Responding" : "Check failed"}
          </Badge>
        )}
      </div>
    </li>
  );
}

export function LLMHealthWidget({ health, loading = false, className }: LLMHealthWidgetProps) {
  if (loading) {
    return (
      <Card className={className}>
        <CardHeader className="space-y-2">
          <div className="h-4 w-40 animate-pulse rounded bg-muted" />
          <div className="h-3 w-full max-w-md animate-pulse rounded bg-muted" />
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="h-24 w-full animate-pulse rounded-lg bg-muted" />
        </CardContent>
      </Card>
    );
  }

  const state = getBackendState(health);
  const meta = stateMeta(state);
  const endpoints = (health?.endpoints ?? []).map((endpoint) => normalizeEndpoint(endpoint));
  const hasOpenCircuit = endpoints.some((endpoint) => endpoint.circuit_state === "open");
  const openEndpoints = endpoints.filter((endpoint) => endpoint.circuit_state === "open");

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base font-semibold">LLM backend health</CardTitle>
            <CardDescription>{meta.description}</CardDescription>
          </div>
          <Badge variant={meta.badgeVariant} showIcon={false} className="shrink-0 gap-1.5">
            {state === "unknown" ? (
              <Clock className={cn("size-3.5", meta.tone)} aria-hidden />
            ) : state === "down" || hasOpenCircuit ? (
              <AlertTriangle className={cn("size-3.5", meta.tone)} aria-hidden />
            ) : (
              <RefreshCw className={cn("size-3.5", meta.tone)} aria-hidden />
            )}
            {meta.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        {hasOpenCircuit ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>Circuit breaker open</AlertTitle>
            <AlertDescription className="mt-1">
              {openEndpoints.length === 1
                ? `${openEndpoints[0]?.name} is failing fast until recovery checks pass.`
                : `${openEndpoints.length} endpoints are failing fast until recovery checks pass.`}
            </AlertDescription>
          </Alert>
        ) : null}

        {health?.enabled === false ? (
          <p className="text-sm text-muted-foreground">
            LLM health monitoring is disabled for this deployment.
          </p>
        ) : endpoints.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No LLM endpoints are configured yet.
          </p>
        ) : (
          <>
            <ul className="space-y-2.5">
              {endpoints.map((endpoint) => (
                <EndpointRow key={endpoint.name} endpoint={endpoint} />
              ))}
            </ul>
            {health?.checked_at ? (
              <p className="text-xs text-muted-foreground">
                Last health poll {formatRelativeTime(health.checked_at)}
              </p>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}
