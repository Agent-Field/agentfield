import type {
  ExecutionVC,
  WorkflowVC,
  WorkflowVCChainResponse,
  VCVerificationRequest,
  VCVerificationResponse,
  VCFilters,
  VCExportResponse,
  VCStatusSummary,
  AuditTrailEntry,
  ComprehensiveVCVerificationResult,
  WorkflowVCStatusBatchResponse
} from '../types/did';
import { normalizeExecutionStatus, isSuccessStatus, isFailureStatus } from '../utils/status';
import { getGlobalApiKey } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/ui/v1';

async function fetchWrapper<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers || {});
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set('X-API-Key', apiKey);
  }
  const response = await fetch(`${API_BASE_URL}${url}`, { ...options, headers });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      message: 'Request failed with status ' + response.status
    }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// VC Management API Functions

function normalizeJsonDocument(value: unknown): unknown {
  if (!value) {
    return value ?? null;
  }

  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch (error) {
      console.warn('Failed to parse JSON document, returning raw string', error);
      return value;
    }
  }

  return value;
}

function createDefaultVCStatusSummary(): VCStatusSummary {
  return {
    has_vcs: false,
    vc_count: 0,
    verified_count: 0,
    failed_count: 0,
    last_vc_created: '',
    verification_status: 'none'
  };
}

function deriveVCStatusFromChain(vcChain: WorkflowVCChainResponse): VCStatusSummary {
  const vcCount = vcChain.component_vcs.length;
  const verifiedCount = vcChain.component_vcs.filter(vc => isSuccessStatus(vc.status)).length;
  const failedCount = vcChain.component_vcs.filter(vc => isFailureStatus(vc.status)).length;
  const lastVCCreated = vcChain.component_vcs.length > 0
    ? vcChain.component_vcs[vcChain.component_vcs.length - 1].created_at
    : '';

  let verificationStatus: 'verified' | 'pending' | 'failed' | 'none' = 'none';
  if (vcCount === 0) {
    verificationStatus = 'none';
  } else if (failedCount > 0) {
    verificationStatus = 'failed';
  } else if (verifiedCount === vcCount) {
    verificationStatus = 'verified';
  } else {
    verificationStatus = 'pending';
  }

  return {
    has_vcs: vcCount > 0,
    vc_count: vcCount,
    verified_count: verifiedCount,
    failed_count: failedCount,
    last_vc_created: lastVCCreated,
    verification_status: verificationStatus
  };
}

/**
 * Verify a Verifiable Credential
 */
export async function verifyVC(vcDocument: any): Promise<VCVerificationResponse> {
  const request: VCVerificationRequest = {
    vc_document: vcDocument
  };

  return fetchWrapper<VCVerificationResponse>('/did/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
}

/**
 * Get workflow VC chain for a specific workflow
 */
export async function getWorkflowVCChain(workflowId: string): Promise<WorkflowVCChainResponse> {
  return fetchWrapper<WorkflowVCChainResponse>(`/workflows/${workflowId}/vc-chain`);
}

/**
 * Create a workflow-level VC
 */
export async function createWorkflowVC(
  workflowId: string,
  sessionId: string,
  executionVCIds: string[]
): Promise<WorkflowVC> {
  return fetchWrapper<WorkflowVC>(`/did/workflow/${workflowId}/vc`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      execution_vc_ids: executionVCIds
    })
  });
}

/**
 * Export VCs with optional filtering
 */
export async function exportVCs(filters?: VCFilters): Promise<VCExportResponse> {
  const queryParams = new URLSearchParams();

  if (filters) {
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        queryParams.append(key, value.toString());
      }
    });
  }

  const queryString = queryParams.toString();
  const url = `/did/export/vcs${queryString ? `?${queryString}` : ''}`;

  return fetchWrapper<VCExportResponse>(url);
}

async function fetchWorkflowVCStatusBatch(workflowIds: string[]): Promise<Record<string, VCStatusSummary>> {
  if (workflowIds.length === 0) {
    return {};
  }

  const response = await fetchWrapper<WorkflowVCStatusBatchResponse>(
    `/workflows/vc-status`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow_ids: workflowIds }),
    }
  );

  const summaries: Record<string, VCStatusSummary> = {};
  response.summaries?.forEach((summary) => {
    summaries[summary.workflow_id] = {
      has_vcs: summary.has_vcs,
      vc_count: summary.vc_count,
      verified_count: summary.verified_count,
      failed_count: summary.failed_count ?? 0,
      last_vc_created: summary.last_vc_created,
      verification_status: summary.verification_status,
    };
  });

  return summaries;
}

/**
 * Fetch VC status summaries for a batch of workflows, falling back to legacy calls if needed.
 */
export async function getWorkflowVCStatuses(workflowIds: string[]): Promise<Record<string, VCStatusSummary>> {
  if (!workflowIds || workflowIds.length === 0) {
    return {};
  }

  try {
    const summaries = await fetchWorkflowVCStatusBatch(workflowIds);
    workflowIds.forEach((id) => {
      if (!summaries[id]) {
        summaries[id] = createDefaultVCStatusSummary();
      }
    });
    return summaries;
  } catch (error) {
    console.warn('Workflow VC status batch endpoint unavailable, falling back to legacy per-workflow fetch', error);

    const entries = await Promise.all(
      workflowIds.map(async (workflowId) => {
        try {
          const vcChain = await getWorkflowVCChain(workflowId);
          return [workflowId, deriveVCStatusFromChain(vcChain)] as const;
        } catch (chainError) {
          console.warn(`Failed to derive VC status for workflow ${workflowId}:`, chainError);
          return [workflowId, createDefaultVCStatusSummary()] as const;
        }
      })
    );

    return Object.fromEntries(entries);
  }
}

/**
 * Get VC status summary for a single workflow
 */
export async function getVCStatusSummary(workflowId: string): Promise<VCStatusSummary> {
  if (!workflowId) {
    return createDefaultVCStatusSummary();
  }

  const summaries = await getWorkflowVCStatuses([workflowId]);
  return summaries[workflowId] ?? createDefaultVCStatusSummary();
}

/**
 * Get VC status summary for an execution
 */
export async function getExecutionVCStatus(executionId: string): Promise<{
  has_vc: boolean;
  vc_id?: string;
  status: string;
  created_at?: string;
  vc_document?: any; // Add vc_document to the return type
  storage_uri?: string;
  document_size_bytes?: number;
  original_status?: string;
}> {
  try {
    // DEBUG: Log the execution ID being requested
    console.log('DEBUG: getExecutionVCStatus called for executionId:', executionId);

    // Try to get execution VC directly from a dedicated endpoint first
    try {
      const result = await fetchWrapper<{
        has_vc: boolean;
        vc_id?: string;
        status: string;
        created_at?: string;
        vc_document?: any; // Include vc_document in response
        storage_uri?: string;
        document_size_bytes?: number;
        original_status?: string;
      }>(`/executions/${executionId}/vc-status`);

      console.log('DEBUG: Direct endpoint response:', result);
      return result;
    } catch (directError) {
      // If direct endpoint doesn't exist, fall back to export method
      console.warn('Direct VC status endpoint not available, using fallback method');
      console.log('DEBUG: Direct endpoint error:', directError);

      const response = await exportVCs({
        limit: 100, // Get more VCs to search through
      });

      console.log('DEBUG: Export VCs response:', response);

      const executionVC = response.execution_vcs.find(vc => vc.execution_id === executionId);

      if (executionVC) {
        console.log('DEBUG: Found execution VC in export:', executionVC);
        console.log('DEBUG: ExecutionVCInfo does not include vc_document field');
        return {
          has_vc: true,
          vc_id: executionVC.vc_id,
          status: executionVC.status,
          created_at: executionVC.created_at,
          storage_uri: (executionVC as any).storage_uri,
          document_size_bytes: (executionVC as any).document_size_bytes,
          original_status: executionVC.status
          // Note: vc_document is not available in ExecutionVCInfo type
        };
      }

      console.log('DEBUG: No execution VC found in export for executionId:', executionId);
      return {
        has_vc: false,
        status: 'none'
      };
    }
  } catch (error) {
    console.error('Failed to get execution VC status:', error);
    return {
      has_vc: false,
      status: 'none' // Changed from 'error' to 'none' to prevent UI panic
    };
  }
}

/**
 * Get execution VC document with full VC data for download in enhanced format
 */
export async function getExecutionVCDocument(executionId: string): Promise<ExecutionVC> {
  try {
    console.log('DEBUG: getExecutionVCDocument called for executionId:', executionId);

    // Try to get the full execution VC from the backend
    const result = await fetchWrapper<ExecutionVC>(`/executions/${executionId}/vc`);

    console.log('DEBUG: getExecutionVCDocument response:', result);

    if (!result.vc_document) {
      throw new Error('VC document not found or not available for download');
    }

    return result;
  } catch (error) {
    console.error('Failed to get execution VC document:', error);

    // Provide more specific error messages based on the error
    if (error instanceof Error) {
      if (error.message.includes('404') || error.message.includes('not found')) {
        throw new Error('VC not found for this execution. The execution may not have completed or VC generation may have failed.');
      } else if (error.message.includes('503') || error.message.includes('not available')) {
        throw new Error('VC service is currently unavailable. Please try again later.');
      } else if (error.message.includes('500')) {
        throw new Error('Server error while fetching VC. Please check the server logs.');
      }
    }

    throw new Error('Failed to fetch execution VC document for download');
  }
}

/**
 * Get execution VC document in enhanced format with DID resolution bundle for CLI verification
 */
export async function getExecutionVCDocumentEnhanced(executionId: string): Promise<any> {
  try {
    console.log('DEBUG: getExecutionVCDocumentEnhanced called for executionId:', executionId);

    // Get the execution VC
    const executionVC = await getExecutionVCDocument(executionId);

    // Parse the VC document to extract DIDs
    const vcDocument = typeof executionVC.vc_document === 'string'
      ? JSON.parse(executionVC.vc_document)
      : executionVC.vc_document;

    // Collect unique DIDs from the VC
    const uniqueDIDs = new Set<string>();
    if (vcDocument.issuer) uniqueDIDs.add(vcDocument.issuer);
    if (vcDocument.credentialSubject?.caller?.did) uniqueDIDs.add(vcDocument.credentialSubject.caller.did);
    if (vcDocument.credentialSubject?.target?.did) uniqueDIDs.add(vcDocument.credentialSubject.target.did);

    // Create DID resolution bundle
    const didResolutionBundle: Record<string, any> = {};

    for (const did of uniqueDIDs) {
      if (did && did.trim() !== '') {
        try {
          // Try to get DID resolution bundle for each DID
          const resolution = await getDIDResolutionBundle(did);
          didResolutionBundle[did] = {
            method: did.startsWith('did:key:') ? 'key' : 'unknown',
            public_key_jwk: resolution.verification_keys?.[0]?.publicKeyJwk || {},
            resolved_from: 'bundled',
            resolved_at: new Date().toISOString()
          };
        } catch (error) {
          console.warn(`Failed to resolve DID ${did}:`, error);
          // Create a placeholder entry for failed resolution
          didResolutionBundle[did] = {
            method: did.startsWith('did:key:') ? 'key' : 'unknown',
            public_key_jwk: {},
            resolved_from: 'failed',
            resolved_at: new Date().toISOString(),
            error: 'Resolution failed'
          };
        }
      }
    }

    // Create enhanced VC chain format compatible with CLI verification
    const enhancedChain = {
      workflow_id: executionVC.workflow_id || 'single-execution',
      generated_at: new Date().toISOString(),
      total_executions: 1,
      completed_executions: 1,
      workflow_status: normalizeExecutionStatus(executionVC.status),
      execution_vcs: [executionVC],
      workflow_vc: {
        workflow_id: executionVC.workflow_id || 'single-execution',
        session_id: executionVC.session_id,
        component_vcs: [executionVC.vc_id],
        workflow_vc_id: '',
        status: executionVC.status,
        start_time: executionVC.created_at,
        end_time: executionVC.created_at,
        total_steps: 1,
        completed_steps: 1,
        vc_document: null, // Single execution doesn't have workflow VC
        signature: '',
        issuer_did: executionVC.issuer_did,
        snapshot_time: new Date().toISOString()
      },
      did_resolution_bundle: didResolutionBundle,
      verification_metadata: {
        export_version: '1.0',
        total_signatures: 1,
        bundled_dids: Object.keys(didResolutionBundle).length,
        export_timestamp: new Date().toISOString()
      }
    };

    console.log('DEBUG: Enhanced execution VC chain created:', enhancedChain);
    return enhancedChain;
  } catch (error) {
    console.error('Failed to get enhanced execution VC document:', error);
    throw error;
  }
}

/**
 * Get audit trail for a workflow
 */
export async function getWorkflowAuditTrail(workflowId: string): Promise<AuditTrailEntry[]> {
  try {
    const vcChain = await getWorkflowVCChain(workflowId);

    return vcChain.component_vcs.map(vc => ({
      vc_id: vc.vc_id,
      execution_id: vc.execution_id,
      timestamp: vc.created_at,
      caller_did: vc.caller_did,
      target_did: vc.target_did,
      status: vc.status,
      input_hash: vc.input_hash,
      output_hash: vc.output_hash,
      signature: vc.signature
    }));
  } catch (error) {
    console.error('Failed to get audit trail:', error);
    return [];
  }
}

/**
 * Download VC document as JSON
 */
export async function downloadVCDocument(vc: ExecutionVC): Promise<void> {
  try {
    // DEBUG: Log the VC object to understand what data we have
    console.log('DEBUG: downloadVCDocument called with VC:', vc);
    console.log('DEBUG: vc.vc_document type:', typeof vc.vc_document);
    console.log('DEBUG: vc.vc_document value:', vc.vc_document);

    if (!vc.vc_document) {
      console.error('DEBUG: vc_document is missing or undefined');
      throw new Error('VC document is missing - cannot download');
    }

    const vcDocument = typeof vc.vc_document === 'string'
      ? JSON.parse(vc.vc_document)
      : vc.vc_document;

    console.log('DEBUG: Parsed VC document:', vcDocument);

    const blob = new Blob([JSON.stringify(vcDocument, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vc-${vc.vc_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download VC document:', error);
    throw new Error('Failed to download VC document');
  }
}

/**
 * Download a full execution VC bundle (execution VC + workflow context + DID bundle)
 */
export async function downloadExecutionVCBundle(executionId: string): Promise<void> {
  try {
    const bundle = await getExecutionVCDocumentEnhanced(executionId);
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `execution-vc-${executionId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download execution VC bundle:', error);
    throw new Error('Failed to download execution VC bundle');
  }
}

/**
 * Copy VC document to clipboard
 */
export async function copyVCToClipboard(vc: ExecutionVC): Promise<boolean> {
  try {
    const vcDocument = typeof vc.vc_document === 'string'
      ? JSON.parse(vc.vc_document)
      : vc.vc_document;

    await navigator.clipboard.writeText(JSON.stringify(vcDocument, null, 2));
    return true;
  } catch (error) {
    console.error('Failed to copy VC to clipboard:', error);
    return false;
  }
}

/**
 * Export workflow VCs as a compliance report
 */
export async function exportWorkflowComplianceReport(
  workflowId: string,
  format: 'json' | 'csv' = 'json'
): Promise<void> {
  try {
    const vcChain = await getWorkflowVCChain(workflowId);

    const sortedComponentVCs = [...vcChain.component_vcs].sort((a, b) => {
      const aTime = new Date(a.created_at).getTime();
      const bTime = new Date(b.created_at).getTime();
      return aTime - bTime;
    });

    const normalizedExecutionVCs = sortedComponentVCs.map((vc) => ({
      ...vc,
      vc_document: normalizeJsonDocument(vc.vc_document),
    }));

    const workflowCredential = vcChain.workflow_vc
      ? {
          ...vcChain.workflow_vc,
          vc_document: normalizeJsonDocument(vcChain.workflow_vc.vc_document),
        }
      : null;

    const generatedAt = new Date().toISOString();
    const workflowStatus = normalizeExecutionStatus(vcChain.status);
    const totalExecutions = workflowCredential?.total_steps ?? normalizedExecutionVCs.length;
    const completedExecutions = normalizedExecutionVCs.filter((vc) => isSuccessStatus(vc.status)).length;
    const bundledDids = vcChain.did_resolution_bundle ? Object.keys(vcChain.did_resolution_bundle).length : 0;
    const totalSignatures = normalizedExecutionVCs.reduce((count, vc) => (vc.signature ? count + 1 : count), 0)
      + (workflowCredential?.signature ? 1 : 0);

    const workflowPath = normalizedExecutionVCs.map((vc) => {
      const document: any = typeof vc.vc_document === 'object' && vc.vc_document !== null ? vc.vc_document : null;
      const subject = document?.credentialSubject ?? {};

      return {
        vc_id: vc.vc_id,
        execution_id: vc.execution_id,
        caller: {
          did: subject?.caller?.did ?? vc.caller_did,
          agentNodeDid: subject?.caller?.agentNodeDid ?? '',
          type: subject?.caller?.type ?? '',
        },
        target: {
          did: subject?.target?.did ?? vc.target_did,
          agentNodeDid: subject?.target?.agentNodeDid ?? '',
          functionName: subject?.target?.functionName ?? '',
        },
        timestamp: subject?.execution?.timestamp ?? vc.created_at,
        status: subject?.execution?.status ?? vc.status,
      };
    });

    if (format === 'json') {
      const report = {
        workflow_id: workflowId,
        generated_at: generatedAt,
        total_executions: totalExecutions,
        completed_executions: completedExecutions,
        workflow_status: workflowStatus,
        execution_vcs: normalizedExecutionVCs,
        workflow_vc: workflowCredential,
        workflow_path: workflowPath,
        did_resolution_bundle: vcChain.did_resolution_bundle ?? {},
        verification_metadata: {
          export_version: '1.0',
          total_signatures: totalSignatures,
          bundled_dids: bundledDids,
          export_timestamp: generatedAt,
        },
      };

      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json',
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `workflow-compliance-${workflowId}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else if (format === 'csv') {
      const headers = [
        'VC ID',
        'Execution ID',
        'Created At',
        'Status',
        'Caller DID',
        'Target DID',
        'Input Hash',
        'Output Hash',
        'Signature',
      ];

      const rows = normalizedExecutionVCs.map((vc) => [
        vc.vc_id,
        vc.execution_id,
        vc.created_at,
        vc.status,
        vc.caller_did,
        vc.target_did,
        vc.input_hash,
        vc.output_hash,
        vc.signature,
      ]);

      const csvContent = [headers, ...rows]
        .map((row) => row.map((field) => `"${field}"`).join(','))
        .join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `workflow-compliance-${workflowId}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.error('Failed to export compliance report:', error);
    throw new Error('Failed to export compliance report');
  }
}

/**
 * Validate VC document format
 */
export function isValidVCDocument(vcDocument: any): boolean {
  try {
    const doc = typeof vcDocument === 'string' ? JSON.parse(vcDocument) : vcDocument;

    // Basic VC structure validation
    return (
      doc &&
      Array.isArray(doc['@context']) &&
      Array.isArray(doc.type) &&
      typeof doc.id === 'string' &&
      typeof doc.issuer === 'string' &&
      typeof doc.issuanceDate === 'string' &&
      doc.credentialSubject &&
      doc.proof
    );
  } catch (error) {
    return false;
  }
}

/**
 * Get DID resolution bundle for a specific DID
 */
export async function getDIDResolutionBundle(did: string): Promise<{
  did: string;
  resolution_status: string;
  did_document: any;
  verification_keys: any[];
  service_endpoints: any[];
  related_vcs: any[];
  component_dids: any[];
  resolution_metadata: any;
}> {
  return fetchWrapper(`/did/${encodeURIComponent(did)}/resolution-bundle`);
}

/**
 * Download DID resolution bundle as JSON
 */
export async function downloadDIDResolutionBundle(did: string): Promise<void> {
  try {
    console.log('DEBUG: downloadDIDResolutionBundle called for DID:', did);

    const response = await fetch(`${API_BASE_URL}/did/${encodeURIComponent(did)}/resolution-bundle/download`);

    if (!response.ok) {
      throw new Error(`Failed to download DID resolution bundle: ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `did-resolution-bundle-${did.replace(/[^a-zA-Z0-9]/g, '_')}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download DID resolution bundle:', error);
    throw new Error('Failed to download DID resolution bundle');
  }
}

/**
 * Perform comprehensive VC verification for an execution
 */
export async function verifyExecutionVCComprehensive(executionId: string): Promise<ComprehensiveVCVerificationResult> {
  return fetchWrapper<ComprehensiveVCVerificationResult>(`/executions/${executionId}/verify-vc`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Perform comprehensive VC verification for a workflow
 */
export async function verifyWorkflowVCComprehensive(workflowId: string): Promise<ComprehensiveVCVerificationResult> {
  return fetchWrapper<ComprehensiveVCVerificationResult>(`/workflows/${workflowId}/verify-vc`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Format VC status for display
 */
export function formatVCStatus(status: string): {
  label: string;
  variant: 'default' | 'secondary' | 'destructive' | 'outline';
} {
  switch (status.toLowerCase()) {
    case 'completed':
    case 'verified':
      return { label: 'Verified', variant: 'default' };
    case 'pending':
    case 'processing':
      return { label: 'Pending', variant: 'secondary' };
    case 'failed':
    case 'error':
      return { label: 'Failed', variant: 'destructive' };
    default:
      return { label: status, variant: 'outline' };
  }
}
