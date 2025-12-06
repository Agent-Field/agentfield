import { Buffer } from 'node:buffer';
import axios, { type AxiosInstance } from 'axios';

// ============================================================================
// DID Identity Types
// ============================================================================

export interface DIDIdentity {
  did: string;
  privateKeyJwk: string;
  publicKeyJwk: string;
  derivationPath: string;
  componentType: string;
  functionName?: string;
}

export interface DIDIdentityPackage {
  agentDid: DIDIdentity;
  reasonerDids: Record<string, DIDIdentity>;
  skillDids: Record<string, DIDIdentity>;
  agentfieldServerId: string;
}

export interface DIDRegistrationRequest {
  agentNodeId: string;
  reasoners: Array<{ id: string; [key: string]: any }>;
  skills: Array<{ id: string; [key: string]: any }>;
}

export interface DIDRegistrationResponse {
  success: boolean;
  identityPackage?: DIDIdentityPackage;
  message?: string;
  error?: string;
}

// ============================================================================
// Credential Types
// ============================================================================

export interface ExecutionCredential {
  vcId: string;
  executionId: string;
  workflowId: string;
  sessionId?: string;
  issuerDid?: string;
  targetDid?: string;
  callerDid?: string;
  vcDocument: any;
  signature?: string;
  inputHash?: string;
  outputHash?: string;
  status: string;
  createdAt: string;
}

export interface WorkflowCredential {
  workflowId: string;
  sessionId?: string;
  componentVcs: string[];
  workflowVcId: string;
  status: string;
  startTime: string;
  endTime?: string;
  totalSteps: number;
  completedSteps: number;
}

export interface AuditTrailFilters {
  workflowId?: string;
  sessionId?: string;
  issuerDid?: string;
  status?: string;
  limit?: number;
}

export interface AuditTrailExport {
  agentDids: string[];
  executionVcs: Array<{
    vcId: string;
    executionId: string;
    workflowId: string;
    sessionId?: string;
    issuerDid?: string;
    targetDid?: string;
    callerDid?: string;
    status: string;
    createdAt: string;
  }>;
  workflowVcs: WorkflowCredential[];
  totalCount: number;
  filtersApplied?: Record<string, any>;
}

export interface GenerateCredentialParams {
  executionContext: {
    executionId: string;
    workflowId?: string;
    sessionId?: string;
    callerDid?: string;
    targetDid?: string;
    agentNodeDid?: string;
    timestamp?: string | Date;
  };
  inputData?: any;
  outputData?: any;
  status?: string;
  errorMessage?: string;
  durationMs?: number;
  headers?: Record<string, string>;
}

export class DidClient {
  private readonly http: AxiosInstance;
  private readonly defaultHeaders: Record<string, string>;

  constructor(baseUrl: string, defaultHeaders?: Record<string, string | number | boolean | undefined>) {
    this.http = axios.create({ baseURL: baseUrl.replace(/\/$/, '') });
    this.defaultHeaders = this.sanitizeHeaders(defaultHeaders ?? {});
  }

  /**
   * Register an agent with the DID system and obtain an identity package.
   * This must be called before generating VCs to get the caller/target DIDs.
   */
  async registerAgent(request: DIDRegistrationRequest): Promise<DIDRegistrationResponse> {
    const payload = {
      agent_node_id: request.agentNodeId,
      reasoners: request.reasoners,
      skills: request.skills
    };

    const res = await this.http.post('/api/v1/did/register', payload, {
      headers: this.mergeHeaders()
    });

    const data = res.data ?? {};
    if (!data.success) {
      return {
        success: false,
        error: data.error ?? 'DID registration failed'
      };
    }

    return {
      success: true,
      identityPackage: this.parseIdentityPackage(data.identity_package),
      message: data.message
    };
  }

  private parseIdentityPackage(pkg: any): DIDIdentityPackage {
    const parseIdentity = (data: any): DIDIdentity => ({
      did: data?.did ?? '',
      privateKeyJwk: data?.private_key_jwk ?? '',
      publicKeyJwk: data?.public_key_jwk ?? '',
      derivationPath: data?.derivation_path ?? '',
      componentType: data?.component_type ?? '',
      functionName: data?.function_name
    });

    const reasonerDids: Record<string, DIDIdentity> = {};
    if (pkg?.reasoner_dids) {
      for (const [name, data] of Object.entries(pkg.reasoner_dids)) {
        reasonerDids[name] = parseIdentity(data);
      }
    }

    const skillDids: Record<string, DIDIdentity> = {};
    if (pkg?.skill_dids) {
      for (const [name, data] of Object.entries(pkg.skill_dids)) {
        skillDids[name] = parseIdentity(data);
      }
    }

    return {
      agentDid: parseIdentity(pkg?.agent_did),
      reasonerDids,
      skillDids,
      agentfieldServerId: pkg?.agentfield_server_id ?? ''
    };
  }

  async generateCredential(params: GenerateCredentialParams): Promise<ExecutionCredential> {
    const ctx = params.executionContext;
    const timestamp =
      ctx.timestamp instanceof Date
        ? ctx.timestamp.toISOString()
        : ctx.timestamp ?? new Date().toISOString();

    const payload = {
      execution_context: {
        execution_id: ctx.executionId,
        workflow_id: ctx.workflowId,
        session_id: ctx.sessionId,
        caller_did: ctx.callerDid,
        target_did: ctx.targetDid,
        agent_node_did: ctx.agentNodeDid,
        timestamp
      },
      input_data: this.serializeDataForJson(params.inputData),
      output_data: this.serializeDataForJson(params.outputData),
      status: params.status ?? 'succeeded',
      error_message: params.errorMessage,
      duration_ms: params.durationMs ?? 0
    };

    const res = await this.http.post('/api/v1/execution/vc', payload, {
      headers: this.mergeHeaders(params.headers)
    });

    return this.mapExecutionCredential(res.data);
  }

  async exportAuditTrail(filters: AuditTrailFilters = {}): Promise<AuditTrailExport> {
    const res = await this.http.get('/api/v1/did/export/vcs', {
      params: this.cleanFilters(filters),
      headers: this.mergeHeaders()
    });

    const data = res.data ?? {};
    return {
      agentDids: data.agent_dids ?? [],
      executionVcs: (data.execution_vcs ?? []).map((vc: any) => ({
        vcId: vc.vc_id,
        executionId: vc.execution_id,
        workflowId: vc.workflow_id,
        sessionId: vc.session_id,
        issuerDid: vc.issuer_did,
        targetDid: vc.target_did,
        callerDid: vc.caller_did,
        status: vc.status,
        createdAt: vc.created_at
      })),
      workflowVcs: (data.workflow_vcs ?? []).map((vc: any) => ({
        workflowId: vc.workflow_id,
        sessionId: vc.session_id,
        componentVcs: vc.component_vcs ?? [],
        workflowVcId: vc.workflow_vc_id ?? vc.workflowVcId ?? vc.workflow_id,
        status: vc.status,
        startTime: vc.start_time,
        endTime: vc.end_time,
        totalSteps: vc.total_steps ?? 0,
        completedSteps: vc.completed_steps ?? 0
      })),
      totalCount: data.total_count ?? 0,
      filtersApplied: data.filters_applied
    };
  }

  private serializeDataForJson(data: any) {
    if (data === undefined || data === null) return '';
    let value: string;
    if (typeof data === 'string') {
      value = data;
    } else if (data instanceof Uint8Array) {
      value = Buffer.from(data).toString('utf-8');
    } else if (typeof data === 'object') {
      try {
        value = JSON.stringify(data, Object.keys(data).sort());
      } catch {
        value = String(data);
      }
    } else {
      value = String(data);
    }
    return Buffer.from(value, 'utf-8').toString('base64');
  }

  private mapExecutionCredential(data: any): ExecutionCredential {
    return {
      vcId: data?.vc_id ?? '',
      executionId: data?.execution_id ?? '',
      workflowId: data?.workflow_id ?? '',
      sessionId: data?.session_id,
      issuerDid: data?.issuer_did,
      targetDid: data?.target_did,
      callerDid: data?.caller_did,
      vcDocument: data?.vc_document,
      signature: data?.signature,
      inputHash: data?.input_hash,
      outputHash: data?.output_hash,
      status: data?.status ?? '',
      createdAt: data?.created_at ?? ''
    };
  }

  private cleanFilters(filters: AuditTrailFilters) {
    const cleaned: Record<string, any> = {};
    if (filters.workflowId) cleaned.workflow_id = filters.workflowId;
    if (filters.sessionId) cleaned.session_id = filters.sessionId;
    if (filters.issuerDid) cleaned.issuer_did = filters.issuerDid;
    if (filters.status) cleaned.status = filters.status;
    if (filters.limit !== undefined) cleaned.limit = filters.limit;
    return cleaned;
  }

  private mergeHeaders(headers?: Record<string, any>) {
    return {
      ...this.defaultHeaders,
      ...this.sanitizeHeaders(headers ?? {})
    };
  }

  private sanitizeHeaders(headers: Record<string, any>): Record<string, string> {
    const sanitized: Record<string, string> = {};
    Object.entries(headers).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      sanitized[key] = typeof value === 'string' ? value : String(value);
    });
    return sanitized;
  }
}
