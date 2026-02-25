import { useMemo, useState } from "react";
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronUp,
  Circle,
  CircleCheck,
  CircleX,
  Copy,
  Download,
  FileJson,
  Hash,
  InProgress,
  Link,
  ShieldCheck,
  Timer,
  User,
} from "@/components/ui/icon-bridge";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge, StatusBadge } from "../ui/badge";
import { Separator } from "../ui/separator";
import { CopyButton } from "../ui/copy-button";
import { cn } from "../../lib/utils";
import {
  getStatusLabel,
  getStatusTheme,
  normalizeExecutionStatus,
} from "../../utils/status";
import type { CanonicalStatus } from "../../utils/status";
import {
  copyVCToClipboard,
  downloadVCDocument,
  downloadDIDResolutionBundle,
  exportWorkflowComplianceReport,
} from "../../services/vcApi";
import type { WorkflowSummary } from "../../types/workflows";
import type { ExecutionVC, WorkflowVCChainResponse, DIDResolutionEntry } from "../../types/did";

interface EnhancedWorkflowIdentityProps {
  workflow: WorkflowSummary;
  vcChain: WorkflowVCChainResponse | null;
}

type StatusBadgeStatus = "success" | "failed" | "running" | "pending" | "degraded" | "unknown";

const STATUS_BADGE_VARIANT_MAP: Record<CanonicalStatus, StatusBadgeStatus> = {
  pending: "pending",
  queued: "pending",
  waiting: "pending",
  running: "running",
  succeeded: "success",
  failed: "failed",
  cancelled: "unknown",
  timeout: "failed",
  unknown: "unknown",
};

const STATUS_ICON_COMPONENT: Record<CanonicalStatus, typeof Circle> = {
  pending: Circle,
  queued: Circle,
  waiting: Circle,
  running: InProgress,
  succeeded: CircleCheck,
  failed: CircleX,
  cancelled: Circle,
  timeout: CircleX,
  unknown: Circle,
};

const EXTRA_ICON_CLASS: Partial<Record<CanonicalStatus, string>> = {
  running: "animate-spin",
  pending: "animate-pulse",
  queued: "animate-pulse",
  waiting: "animate-pulse",
};

function getStatusBadgeVariant(status: string): StatusBadgeStatus {
  const normalized = normalizeExecutionStatus(status);
  return STATUS_BADGE_VARIANT_MAP[normalized] ?? "unknown";
}

function getStatusIconConfig(status: string) {
  const normalized = normalizeExecutionStatus(status);
  const IconComponent = STATUS_ICON_COMPONENT[normalized] ?? Circle;
  const theme = getStatusTheme(normalized);
  const extra = EXTRA_ICON_CLASS[normalized];

  return {
    Icon: IconComponent,
    className: extra ? cn(theme.iconClass, extra) : theme.iconClass,
  };
}

const formatDateTime = (value?: string) => {
  if (!value) return "—";
  try {
    const date = new Date(value);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  } catch {
    return value;
  }
};

const formatRelativeTime = (value?: string) => {
  if (!value) return "—";
  try {
    const target = new Date(value).getTime();
    if (Number.isNaN(target)) return value;
    const delta = target - Date.now();
    const abs = Math.abs(delta);
    const minutes = Math.round(abs / 60000);
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ${delta < 0 ? "ago" : "from now"}`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ${delta < 0 ? "ago" : "from now"}`;
    const days = Math.round(hours / 24);
    return `${days} day${days === 1 ? "" : "s"} ${delta < 0 ? "ago" : "from now"}`;
  } catch {
    return value;
  }
};

export function EnhancedWorkflowIdentity({ workflow, vcChain }: EnhancedWorkflowIdentityProps) {
  const [expandedVcId, setExpandedVcId] = useState<string | null>(null);
  const [exportingFormat, setExportingFormat] = useState<"json" | "csv" | null>(null);
  const [downloadingVcId, setDownloadingVcId] = useState<string | null>(null);
  const [copyingVcId, setCopyingVcId] = useState<string | null>(null);
  const [downloadingDid, setDownloadingDid] = useState<string | null>(null);

  const componentVCs = useMemo<ExecutionVC[]>(() => vcChain?.component_vcs ?? [], [vcChain?.component_vcs]);
  const componentVCInfos = useMemo(
    () =>
      componentVCs.map((vc) => {
        let document: any = null;
        if (vc.vc_document) {
          try {
            document = typeof vc.vc_document === "string" ? JSON.parse(vc.vc_document) : vc.vc_document;
          } catch (error) {
            console.warn("Failed to parse VC document for", vc.vc_id, error);
            document = null;
          }
        }

        return {
          vc,
          document,
          credentialSubject: document?.credentialSubject ?? null,
        };
      }),
    [componentVCs]
  );
  const verifiedVCs = useMemo(
    () => componentVCs.filter((vc) => normalizeExecutionStatus(vc.status) === "succeeded"),
    [componentVCs]
  );
  const failedVCs = useMemo(
    () => componentVCs.filter((vc) => {
      const normalized = normalizeExecutionStatus(vc.status);
      return normalized === "failed" || normalized === "timeout";
    }),
    [componentVCs]
  );

  const workflowCredentialStatus = normalizeExecutionStatus(
    vcChain?.workflow_vc?.status || vcChain?.status || workflow.status || "unknown"
  );
  const workflowCredentialLabel = getStatusLabel(workflowCredentialStatus);
  const workflowBadgeVariant = getStatusBadgeVariant(workflowCredentialStatus);

  const latestCredentialIssuedAt = useMemo(() => {
    if (!componentVCs.length) return vcChain?.workflow_vc?.start_time;
    return componentVCs
      .map((vc) => vc.created_at)
      .filter(Boolean)
      .sort((a, b) => new Date(b || 0).getTime() - new Date(a || 0).getTime())[0];
  }, [componentVCs, vcChain?.workflow_vc?.start_time]);

  const didEntries = useMemo<Array<[string, DIDResolutionEntry]>>(() => {
    if (!vcChain?.did_resolution_bundle) return [];
    return Object.entries(vcChain.did_resolution_bundle);
  }, [vcChain?.did_resolution_bundle]);

  const handleToggleVc = (vcId: string) => {
    setExpandedVcId((prev) => (prev === vcId ? null : vcId));
  };

  const handleExportBundle = async (format: "json" | "csv") => {
    if (!workflow.workflow_id) return;
    setExportingFormat(format);
    try {
      await exportWorkflowComplianceReport(workflow.workflow_id, format);
    } catch (error) {
      console.error("Failed to export workflow credential bundle:", error);
    } finally {
      setExportingFormat(null);
    }
  };

  const handleDownloadVC = async (vc: ExecutionVC) => {
    if (!vc || downloadingVcId) return;
    setDownloadingVcId(vc.vc_id);
    try {
      await downloadVCDocument(vc);
    } catch (error) {
      console.error("Failed to download VC document:", error);
    } finally {
      setDownloadingVcId(null);
    }
  };

  const handleCopyVC = async (vc: ExecutionVC) => {
    if (!vc) return;
    setCopyingVcId(vc.vc_id);
    try {
      const success = await copyVCToClipboard(vc);
      if (!success) {
        console.warn("VC copy returned false");
      }
    } catch (error) {
      console.error("Failed to copy VC document:", error);
    } finally {
      setTimeout(() => setCopyingVcId(null), 1800);
    }
  };

  const handleDownloadDid = async (did: string) => {
    setDownloadingDid(did);
    try {
      await downloadDIDResolutionBundle(did);
    } catch (error) {
      console.error("Failed to download DID bundle:", error);
    } finally {
      setDownloadingDid(null);
    }
  };

  const identitySummary = [
    {
      title: "Verification",
      icon: ShieldCheck,
      value: componentVCs.length
        ? `${verifiedVCs.length}/${componentVCs.length} complete`
        : "0 credentials",
      hint: componentVCs.length
        ? `${Math.round((verifiedVCs.length / componentVCs.length) * 100)}% of executions verified`
        : "No component credentials yet",
      badgeLabel: workflowCredentialLabel,
      badgeTone: workflowCredentialStatus,
    },
    {
      title: "Issues",
      icon: AlertCircle,
      value: failedVCs.length,
      hint: failedVCs.length ? "Action required on failing credentials" : "No failures detected",
      badgeLabel: failedVCs.length ? "Attention" : "Clear",
      badgeTone: failedVCs.length ? "failed" : "succeeded",
    },
    {
      title: "Identity",
      icon: User,
      value: workflow.agent_name || "Anonymous agent",
      hint: workflow.session_id ? `Session ${workflow.session_id.slice(0, 8)}…` : "Session pending",
      badgeLabel: workflow.actor_id ? "Actor linked" : "Agent only",
      badgeTone: workflow.actor_id ? "succeeded" : "pending",
    },
    {
      title: "Last issued",
      icon: Timer,
      value: latestCredentialIssuedAt ? formatRelativeTime(latestCredentialIssuedAt) : "—",
      hint: latestCredentialIssuedAt ? formatDateTime(latestCredentialIssuedAt) : "No credentials issued",
      badgeLabel: componentVCs.length ? "Auditable" : "Waiting",
      badgeTone: componentVCs.length ? "succeeded" : "pending",
    },
  ];

  return (
    <div className="h-full overflow-hidden">
      <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
        <div className="p-6 space-y-6">
          <ResponsiveGrid variant="dashboard" align="start">
            {identitySummary.map((item, index) => {
              const Icon = item.icon;
              const statusVariant = getStatusBadgeVariant(item.badgeTone);

              return (
                <div
                  key={`${item.title}-${index}`}
                  className="rounded-xl border border-border bg-card/80 p-4 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 text-body font-medium text-text-primary">
                      <span className="rounded-md bg-muted p-1.5">
                        <Icon className="h-4 w-4" />
                      </span>
                      {item.title}
                    </div>
                    <StatusBadge
                      status={statusVariant}
                      size="sm"
                      showIcon={false}
                      className="font-medium"
                    >
                      {item.badgeLabel}
                    </StatusBadge>
                  </div>
                  <div className="mt-4 text-heading-1">
                    {item.value}
                  </div>
                  <p className="mt-1 text-body-small">{item.hint}</p>
                </div>
              );
            })}
          </ResponsiveGrid>

          <ResponsiveGrid columns={{ base: 1, lg: 12 }} gap="lg" align="start">
            <ResponsiveGrid.Item span={{ lg: 4 }}>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4" />
                    Digital Identity
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-body">
                  <div>
                    <span className="text-caption">Workflow ID</span>
                    <div className="mt-1 flex items-center gap-2">
                      <code className="rounded bg-muted/40 px-2 py-1 font-mono text-body-small text-foreground">
                        {workflow.workflow_id}
                      </code>
                      <CopyButton
                        value={workflow.workflow_id}
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                      />
                    </div>
                  </div>

                  {workflow.session_id && (
                    <div>
                      <span className="text-caption">Session ID</span>
                      <div className="mt-1 flex items-center gap-2">
                        <code className="rounded bg-muted/40 px-2 py-1 font-mono text-body-small text-foreground">
                          {workflow.session_id}
                        </code>
                        <CopyButton
                          value={workflow.session_id}
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                        />
                      </div>
                    </div>
                  )}

                  {vcChain?.workflow_vc?.workflow_vc_id && (
                    <div>
                      <span className="text-caption">Workflow Credential</span>
                      <div className="mt-1 flex items-center gap-2">
                        <code className="rounded bg-muted/40 px-2 py-1 font-mono text-body-small text-foreground">
                          {vcChain.workflow_vc.workflow_vc_id}
                        </code>
                        <CopyButton
                          value={vcChain.workflow_vc.workflow_vc_id}
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                        />
                      </div>
                    </div>
                  )}

                  <div>
                    <span className="text-caption">Agent</span>
                    <div className="mt-1 flex items-center gap-2 text-body text-text-primary">
                      <User className="h-4 w-4 text-muted-foreground" />
                      <span>{workflow.agent_name || "Unknown agent"}</span>
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <span className="text-caption">Trust bundle</span>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportBundle("json")}
                        disabled={exportingFormat !== null}
                        className="flex items-center gap-2"
                      >
                        {exportingFormat === "json" ? (
                          <InProgress className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <FileJson className="h-3.5 w-3.5" />
                        )}
                        JSON
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportBundle("csv")}
                        disabled={exportingFormat !== null}
                        className="flex items-center gap-2"
                      >
                        {exportingFormat === "csv" ? (
                          <InProgress className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Download className="h-3.5 w-3.5" />
                        )}
                        CSV
                      </Button>
                    </div>
                    <p className="text-body-small">
                      Export the complete credential set for independent verification.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </ResponsiveGrid.Item>

            <ResponsiveGrid.Item span={{ lg: 8 }}>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Link className="h-4 w-4" />
                      Workflow Credential Overview
                    </div>
                    <StatusBadge status={workflowBadgeVariant} size="sm">
                      {workflowCredentialLabel}
                    </StatusBadge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveGrid
                    preset="halves"
                    gap="sm"
                    align="start"
                    className="text-body"
                  >
                    <div className="grid gap-1">
                      <span className="text-caption">Status</span>
                      <span className="text-body font-medium text-text-primary">{workflowCredentialLabel}</span>
                    </div>
                    <div className="grid gap-1">
                      <span className="text-caption">Issued</span>
                      <span className="text-body">{formatDateTime(vcChain?.workflow_vc?.start_time)}</span>
                    </div>
                    <div className="grid gap-1">
                      <span className="text-caption">Session</span>
                      <span className="text-body">{workflow.session_id || "—"}</span>
                    </div>
                    <div className="grid gap-1">
                      <span className="text-caption">Total steps</span>
                      <span className="text-body">{vcChain?.total_steps ?? componentVCs.length}</span>
                    </div>
                    <div className="grid gap-1">
                      <span className="text-caption">Verified steps</span>
                      <span className="text-body">{verifiedVCs.length}</span>
                    </div>
                    <div className="grid gap-1">
                      <span className="text-caption">Failures</span>
                      <span className="text-body">{failedVCs.length}</span>
                    </div>
                  </ResponsiveGrid>
                </CardContent>
              </Card>
            </ResponsiveGrid.Item>

            <ResponsiveGrid.Item span={{ lg: 12 }}>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base font-semibold">
                    <Link className="h-4 w-4" />
                    Workflow Path
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {componentVCInfos.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-muted/10 p-6 text-center text-body-small">
                      No credentialed executions available to assemble a workflow path yet.
                    </div>
                  ) : (
                    componentVCInfos.map(({ vc, credentialSubject }, index) => {
                      const normalized = normalizeExecutionStatus(vc.status);
                      const statusVariant = getStatusBadgeVariant(normalized);
                      const statusLabel = getStatusLabel(normalized);
                      const callerDid = credentialSubject?.caller?.did ?? vc.caller_did;
                      const targetDid = credentialSubject?.target?.did ?? vc.target_did;
                      const functionName = credentialSubject?.target?.functionName || "—";
                      const timestamp = credentialSubject?.execution?.timestamp ?? vc.created_at;

                      return (
                        <div key={vc.vc_id} className="rounded-xl border border-border bg-card/70 p-4">
                          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <div className="flex items-center gap-3 text-sm font-semibold text-foreground">
                              <Badge variant="secondary" className="text-xs">
                                Step {index + 1}
                              </Badge>
                              <span>Execution {vc.execution_id.slice(0, 8)}…</span>
                              <span className="text-body-small">{formatDateTime(timestamp)}</span>
                            </div>
                            <StatusBadge status={statusVariant} size="sm">
                              {statusLabel}
                            </StatusBadge>
                          </div>

                          <ResponsiveGrid
                            preset="halves"
                            gap="sm"
                            align="start"
                            className="mt-3"
                          >
                            <div>
                              <span className="text-xs uppercase tracking-wide text-muted-foreground">Caller DID</span>
                              <div className="mt-1 flex items-center gap-2">
                                <code className="rounded bg-muted/40 px-2 py-1 font-mono text-xs text-foreground break-all">
                                  {callerDid || "—"}
                                </code>
                                {callerDid && (
                                  <CopyButton
                                    value={callerDid}
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                  />
                                )}
                              </div>
                              {credentialSubject?.caller?.agentNodeDid && (
                                <p className="mt-1 text-body-small">
                                  Agent node: {credentialSubject.caller.agentNodeDid}
                                </p>
                              )}
                            </div>
                            <div>
                              <span className="text-xs uppercase tracking-wide text-muted-foreground">Target DID</span>
                              <div className="mt-1 flex items-center gap-2">
                                <code className="rounded bg-muted/40 px-2 py-1 font-mono text-xs text-foreground break-all">
                                  {targetDid || "—"}
                                </code>
                                {targetDid && (
                                  <CopyButton
                                    value={targetDid}
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                  />
                                )}
                              </div>
                              {credentialSubject?.target?.agentNodeDid && (
                                <p className="mt-1 text-body-small">
                                  Agent node: {credentialSubject.target.agentNodeDid}
                                </p>
                              )}
                            </div>
                            <div>
                              <span className="text-xs uppercase tracking-wide text-muted-foreground">Function</span>
                              <p className="mt-1 font-mono text-xs text-foreground break-all">{functionName}</p>
                            </div>
                            <div>
                              <span className="text-xs uppercase tracking-wide text-muted-foreground">Input → Output</span>
                              <p className="mt-1 font-mono text-body-small break-all">
                                {vc.input_hash || "—"}
                              </p>
                              <p className="mt-1 font-mono text-body-small break-all">
                                {vc.output_hash || "—"}
                              </p>
                            </div>
                          </ResponsiveGrid>
                        </div>
                      );
                    })
                  )}
                </CardContent>
              </Card>
            </ResponsiveGrid.Item>
          </ResponsiveGrid>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <Hash className="h-4 w-4" />
                Component Credentials
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {componentVCs.length === 0 && (
                <div className="rounded-lg border border-dashed border-border bg-muted/10 p-6 text-center text-body-small">
                  No verifiable credentials have been issued for individual executions yet.
                </div>
              )}

              {componentVCInfos.map(({ vc, document, credentialSubject }) => {
                const normalized = normalizeExecutionStatus(vc.status);
                const statusVariant = getStatusBadgeVariant(normalized);
                const { Icon: StatusIcon, className: statusIconClass } = getStatusIconConfig(normalized);
                const isExpanded = expandedVcId === vc.vc_id;
                const documentPreview = (() => {
                  if (document) return JSON.stringify(document, null, 2);
                  if (!vc.vc_document) return null;
                  return typeof vc.vc_document === "string"
                    ? vc.vc_document
                    : JSON.stringify(vc.vc_document, null, 2);
                })();
                const executionDetails = credentialSubject?.execution;
                const duration = executionDetails?.durationMs ?? 0;
                const timestamp = executionDetails?.timestamp ?? vc.created_at;
                const callerDid = credentialSubject?.caller?.did ?? vc.caller_did;
                const callerAgent = credentialSubject?.caller?.agentNodeDid ?? "";
                const targetDid = credentialSubject?.target?.did ?? vc.target_did;
                const targetAgent = credentialSubject?.target?.agentNodeDid ?? "";
                const functionName = credentialSubject?.target?.functionName || "—";
                const issuerDid = document?.issuer ?? vc.issuer_did;
                const proofValue = document?.proof?.proofValue ?? vc.signature;
                const inputHash = executionDetails?.inputHash ?? vc.input_hash;
                const outputHash = executionDetails?.outputHash ?? vc.output_hash;
                const statusLabel = getStatusLabel(normalized);

                return (
                  <div key={vc.vc_id} className="rounded-xl border border-border bg-card/70">
                    <div className="flex flex-col gap-2 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-start gap-3">
                        <span className="mt-0.5 rounded-md bg-muted p-1.5">
                          <StatusIcon className={cn("h-4 w-4", statusIconClass)} />
                        </span>
                        <div>
                          <div className="text-sm font-semibold text-foreground">
                            Execution {vc.execution_id.slice(0, 8)}…
                          </div>
                          <div className="text-body-small">
                            Issued {formatDateTime(timestamp)}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <StatusBadge status={statusVariant} size="sm">
                          {statusLabel}
                        </StatusBadge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleToggleVc(vc.vc_id)}
                          className="h-8 w-8 p-0"
                          aria-label={isExpanded ? "Collapse credential" : "Expand credential"}
                        >
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="border-t border-border bg-muted/10 px-4 py-4 text-sm">
                        <ResponsiveGrid preset="halves" gap="sm" align="start">
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Caller DID</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-xs text-foreground break-all">{callerDid || "—"}</p>
                              {callerDid && (
                                <CopyButton
                                  value={callerDid}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                            {callerAgent && (
                              <p className="mt-1 text-body-small">Agent node: {callerAgent}</p>
                            )}
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Target DID</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-xs text-foreground break-all">{targetDid || "—"}</p>
                              {targetDid && (
                                <CopyButton
                                  value={targetDid}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                            {targetAgent && (
                              <p className="mt-1 text-body-small">Agent node: {targetAgent}</p>
                            )}
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Function</span>
                            <p className="mt-1 font-mono text-xs text-foreground break-all">{functionName}</p>
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Duration</span>
                            <p className="mt-1 font-mono text-xs text-foreground">
                              {duration ? `${duration} ms` : "—"}
                            </p>
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Issuer DID</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-xs text-foreground break-all">{issuerDid || "—"}</p>
                              {issuerDid && (
                                <CopyButton
                                  value={issuerDid}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Signature</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-xs text-foreground break-all">
                                {proofValue || "—"}
                              </p>
                              {proofValue && (
                                <CopyButton
                                  value={proofValue}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Input Hash</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-body-small break-all">{inputHash || "—"}</p>
                              {inputHash && (
                                <CopyButton
                                  value={inputHash}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                          </div>
                          <div>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">Output Hash</span>
                            <div className="mt-1 flex items-center gap-2">
                              <p className="font-mono text-body-small break-all">{outputHash || "—"}</p>
                              {outputHash && (
                                <CopyButton
                                  value={outputHash}
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 [&_svg]:h-3.5 [&_svg]:w-3.5"
                                />
                              )}
                            </div>
                          </div>
                        </ResponsiveGrid>

                        <Separator className="my-4" />

                        <div className="flex flex-wrap items-center gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDownloadVC(vc)}
                            disabled={downloadingVcId === vc.vc_id}
                            className="flex items-center gap-2"
                          >
                            {downloadingVcId === vc.vc_id ? (
                              <InProgress className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Download className="h-3.5 w-3.5" />
                            )}
                            Download JSON
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleCopyVC(vc)}
                            disabled={copyingVcId === vc.vc_id}
                            className="flex items-center gap-2"
                          >
                            {copyingVcId === vc.vc_id ? (
                              <Check className="h-3.5 w-3.5 text-emerald-500" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                            Copy JSON
                          </Button>
                        </div>

                        {documentPreview && (
                          <div className="mt-4 max-h-64 overflow-auto rounded-lg bg-background p-3 text-xs">
                            <pre className="font-mono text-muted-foreground whitespace-pre-wrap">
                              {documentPreview}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <FileJson className="h-4 w-4" />
                DID Resolution Bundle
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              {didEntries.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border bg-muted/10 p-6 text-center text-body-small">
                  No decentralized identifier bundle has been attached to this workflow credential yet.
                </div>
              ) : (
                didEntries.map(([did, entry]) => (
                  <div key={did} className="rounded-lg border border-border bg-card/70 p-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div className="space-y-1">
                        <p className="font-mono text-xs text-foreground break-all">{did}</p>
                        <p className="text-body-small">
                          Method: {entry?.method || "unknown"} • Resolved {formatRelativeTime(entry?.resolved_at)}
                        </p>
                        <p className="text-body-small">
                          Source: {entry?.resolved_from || "not provided"}
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDownloadDid(did)}
                        disabled={downloadingDid === did}
                        className="flex items-center gap-2 self-start md:self-auto"
                      >
                        {downloadingDid === did ? (
                          <InProgress className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Download className="h-3.5 w-3.5" />
                        )}
                        Download bundle
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
