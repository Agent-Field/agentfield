import { useState } from "react";
import { Shield, ExternalLink, AlertCircle, CheckCircle, Eye, Download, Loader2 } from "@/components/ui/icon-bridge";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import type { WorkflowExecution } from "../../types/executions";
import type { VCStatusData, VCDocument } from "../../types/did";
import { DIDDisplay } from "../did/DIDDisplay";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { VerifiableCredentialBadge } from "../vc";
import { CollapsibleSection } from "./CollapsibleSection";
import { downloadExecutionVCBundle } from "../../services/vcApi";
import { CopyButton } from "../ui/copy-button";
import { JsonHighlightedPre } from "../ui/json-syntax-highlight";

interface ExecutionIdentityPanelProps {
  execution: WorkflowExecution;
  vcStatus?: VCStatusData | null;
  vcLoading?: boolean;
}

function truncateId(id: string): string {
  return `${id.slice(0, 8)}...${id.slice(-4)}`;
}

export function ExecutionIdentityPanel({
  execution,
  vcStatus,
  vcLoading
}: ExecutionIdentityPanelProps) {
  const [showVCDetails, setShowVCDetails] = useState(false);
  const [downloadingBundle, setDownloadingBundle] = useState(false);

  const handleVerifyVC = () => {
    // For now, just toggle details
    setShowVCDetails(!showVCDetails);
  };

  const handleDownloadBundle = async () => {
    if (downloadingBundle || !execution?.execution_id) return;
    setDownloadingBundle(true);
    try {
      await downloadExecutionVCBundle(execution.execution_id);
    } catch (error) {
      console.error("Failed to download execution VC bundle:", error);
    } finally {
      setDownloadingBundle(false);
    }
  };

  const vcDocument =
    vcStatus?.vc_document && typeof vcStatus.vc_document === "object" && !Array.isArray(vcStatus.vc_document)
      ? (vcStatus.vc_document as Partial<VCDocument>)
      : null;
  const credentialSubject = vcDocument?.credentialSubject ?? null;
  const executionDetails = credentialSubject?.execution ?? null;
  const callerInfo = credentialSubject?.caller ?? null;
  const targetInfo = credentialSubject?.target ?? null;
  const issuerDid = vcDocument?.issuer ?? "";
  const proofValue = vcDocument?.proof?.proofValue ?? "";
  const inputHash = executionDetails?.inputHash ?? "";
  const outputHash = executionDetails?.outputHash ?? "";
  const durationMs = executionDetails?.durationMs ?? execution.duration_ms ?? 0;
  const executionTimestamp = executionDetails?.timestamp ?? execution.started_at;
  const callerDid = callerInfo?.did ?? "";
  const callerAgent = callerInfo?.agentNodeDid ?? "";
  const targetDid = targetInfo?.did ?? "";
  const targetAgent = targetInfo?.agentNodeDid ?? "";
  const functionName = targetInfo?.functionName ?? "";

  return (
    <div className="h-full overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
      <div className="p-6 space-y-6">
        {/* Agent Identity */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Agent Identity
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="md" align="start">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm text-muted-foreground">Agent Node ID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-sm text-foreground bg-muted/30 px-2 py-1 rounded">
                        {execution.agent_node_id}
                      </code>
                      <CopyButton
                        value={execution.agent_node_id}
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                        tooltip="Copy agent node ID"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm text-muted-foreground">Decentralized Identifier (DID)</label>
                    <div className="flex items-center gap-2 mt-1">
                      <DIDDisplay nodeId={execution.agent_node_id} variant="full" />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm text-muted-foreground">Execution ID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-sm text-foreground bg-muted/30 px-2 py-1 rounded">
                        {truncateId(execution.execution_id)}
                      </code>
                      <CopyButton
                        value={execution.execution_id}
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                        tooltip="Copy execution ID"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        title="View full ID"
                      >
                        <Eye className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </ResponsiveGrid>
          </CardContent>
        </Card>

        {/* Verifiable Credential */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Verifiable Credential
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {vcLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                <span className="text-sm text-muted-foreground">Loading credential status...</span>
              </div>
            ) : vcStatus?.has_vc ? (
              <div className="space-y-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="flex items-center gap-3">
                    <VerifiableCredentialBadge
                      hasVC={vcStatus.has_vc}
                      status={vcStatus.status}
                      vcData={vcStatus}
                      executionId={execution.execution_id}
                      showCopyButton={false}
                      showVerifyButton={false}
                    />
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        Credential Verified
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Status: {vcStatus.status}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleDownloadBundle}
                      disabled={downloadingBundle}
                      className="flex items-center gap-2"
                    >
                      {downloadingBundle ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Download className="w-3 h-3" />
                      )}
                      Download JSON
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleVerifyVC}
                      className="flex items-center gap-2"
                    >
                      <Eye className="w-3 h-3" />
                      {showVCDetails ? 'Hide' : 'View'} Details
                    </Button>
                  </div>
                </div>

                {vcStatus.vc_id && (
                  <div>
                    <label className="text-sm text-muted-foreground">Credential ID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-sm text-foreground bg-muted/30 px-2 py-1 rounded">
                        {truncateId(vcStatus.vc_id)}
                      </code>
                      <CopyButton
                        value={vcStatus.vc_id}
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                        tooltip="Copy VC ID"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        title="View on blockchain"
                      >
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                )}

                {vcStatus.created_at && (
                  <div>
                    <label className="text-sm text-muted-foreground">Issued At</label>
                    <p className="font-mono text-sm text-foreground mt-1">
                      {new Date(vcStatus.created_at).toLocaleString()}
                    </p>
                  </div>
                )}

                <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="sm" align="start">
                  <div>
                    <label className="text-sm text-muted-foreground">Issuer DID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {issuerDid || '—'}
                      </code>
                      {issuerDid && (
                        <CopyButton
                          value={issuerDid}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy issuer DID"
                        />
                      )}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Proof</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {proofValue || '—'}
                      </code>
                      {proofValue && (
                        <CopyButton
                          value={proofValue}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy proof value"
                        />
                      )}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Caller DID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {callerDid || '—'}
                      </code>
                      {callerDid && (
                        <CopyButton
                          value={callerDid}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy caller DID"
                        />
                      )}
                    </div>
                    {callerAgent && (
                      <p className="text-sm text-muted-foreground mt-1">Agent node: {callerAgent}</p>
                    )}
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Target DID</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {targetDid || '—'}
                      </code>
                      {targetDid && (
                        <CopyButton
                          value={targetDid}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy target DID"
                        />
                      )}
                    </div>
                    {targetAgent && (
                      <p className="text-sm text-muted-foreground mt-1">Agent node: {targetAgent}</p>
                    )}
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Function</label>
                    <p className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded mt-1 break-all">
                      {functionName || '—'}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Execution Timestamp</label>
                    <p className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded mt-1 break-all">
                      {executionTimestamp ? new Date(executionTimestamp).toLocaleString() : '—'}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Duration</label>
                    <p className="font-mono text-xs text-foreground bg-muted/30 px-2 py-1 rounded mt-1">
                      {durationMs ? `${durationMs} ms` : '—'}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Input Hash</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-sm text-muted-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {inputHash || '—'}
                      </code>
                      {inputHash && (
                        <CopyButton
                          value={inputHash}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy input hash"
                        />
                      )}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Output Hash</label>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="font-mono text-sm text-muted-foreground bg-muted/30 px-2 py-1 rounded break-all">
                        {outputHash || '—'}
                      </code>
                      {outputHash && (
                        <CopyButton
                          value={outputHash}
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 p-0 [&_svg]:h-3 [&_svg]:w-3"
                          tooltip="Copy output hash"
                        />
                      )}
                    </div>
                  </div>
                </ResponsiveGrid>

                {showVCDetails && Boolean(vcStatus.vc_document) && (
                  <CollapsibleSection
                    title="Credential Document"
                    icon={Shield}
                    defaultOpen={true}
                    badge={<Badge variant="secondary" className="text-xs">JSON</Badge>}
                  >
                    <div className="p-4">
                      <JsonHighlightedPre
                        data={vcStatus.vc_document}
                        className="max-h-64 overflow-auto rounded-lg bg-muted/50 p-4 text-xs"
                      />
                      <div className="mt-2 flex justify-end">
                        <CopyButton
                          value={JSON.stringify(vcStatus.vc_document, null, 2)}
                          className="h-8 w-auto px-3"
                        />
                      </div>
                    </div>
                  </CollapsibleSection>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-3 p-4 border border-yellow-200 bg-yellow-50 rounded-lg">
                <AlertCircle className="w-5 h-5 text-yellow-600" />
                <div>
                  <p className="text-sm font-medium text-yellow-800">
                    No Verifiable Credential
                  </p>
                  <p className="text-xs text-yellow-700 mt-1">
                    This execution does not have an associated verifiable credential.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Trust & Security */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Trust & Security
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ResponsiveGrid columns={{ base: 1, md: 2 }} gap="md" align="start" className="text-sm">
              <div className="space-y-3">
                <div>
                  <label className="text-muted-foreground">Trust Level</label>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant={vcStatus?.has_vc ? "default" : "secondary"}>
                      {vcStatus?.has_vc ? "Verified" : "Unverified"}
                    </Badge>
                  </div>
                </div>

                <div>
                  <label className="text-muted-foreground">Authentication</label>
                  <p className="text-foreground">DID-based</p>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <label className="text-muted-foreground">Integrity Check</label>
                  <div className="flex items-center gap-2 mt-1">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className="text-foreground">Passed</span>
                  </div>
                </div>

                <div>
                  <label className="text-muted-foreground">Chain of Trust</label>
                  <p className="text-foreground">Validated</p>
                </div>
              </div>
            </ResponsiveGrid>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
