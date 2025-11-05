// DID/VC TypeScript type definitions based on backend Go types

export interface AgentDIDInfo {
  did: string;
  agent_node_id: string;
  agentfield_server_id: string;
  public_key_jwk: any;
  derivation_path: string;
  reasoners: Record<string, ReasonerDIDInfo>;
  skills: Record<string, SkillDIDInfo>;
  status: AgentDIDStatus;
  registered_at: string;
}

export interface ReasonerDIDInfo {
  did: string;
  function_name: string;
  public_key_jwk: any;
  derivation_path: string;
  capabilities: string[];
  exposure_level: string;
  created_at: string;
}

export interface SkillDIDInfo {
  did: string;
  function_name: string;
  public_key_jwk: any;
  derivation_path: string;
  tags: string[];
  exposure_level: string;
  created_at: string;
}

export type AgentDIDStatus = 'active' | 'inactive' | 'revoked';

export interface ExecutionVC {
  vc_id: string;
  execution_id: string;
  workflow_id: string;
  session_id: string;
  issuer_did: string;
  target_did: string;
  caller_did: string;
  vc_document: any;
  signature: string;
  storage_uri?: string;
  document_size_bytes?: number;
  input_hash: string;
  output_hash: string;
  status: string;
  created_at: string;
  parent_vc_id?: string;
  child_vc_ids?: string[];
}

export interface WorkflowVC {
  workflow_id: string;
  session_id: string;
  component_vcs: string[];
  workflow_vc_id: string;
  status: string;
  start_time: string;
  end_time?: string;
  total_steps: number;
  completed_steps: number;
  vc_document?: any;
  signature?: string;
  issuer_did?: string;
  snapshot_time?: string;
  storage_uri?: string;
  document_size_bytes?: number;
}

export interface DIDIdentityPackage {
  agent_did: DIDIdentity;
  reasoner_dids: Record<string, DIDIdentity>;
  skill_dids: Record<string, DIDIdentity>;
  agentfield_server_id: string;
}

export interface DIDIdentity {
  did: string;
  private_key_jwk: string;
  public_key_jwk: string;
  derivation_path: string;
  component_type: string;
  function_name?: string;
}

export interface ExecutionContext {
  execution_id: string;
  workflow_id: string;
  session_id: string;
  caller_did: string;
  target_did: string;
  agent_node_did: string;
  timestamp: string;
}

export interface VCDocument {
  '@context': string[];
  type: string[];
  id: string;
  issuer: string;
  issuanceDate: string;
  credentialSubject: VCCredentialSubject;
  proof: VCProof;
}

export interface VCCredentialSubject {
  executionId: string;
  workflowId: string;
  sessionId: string;
  caller: VCCaller;
  target: VCTarget;
  execution: VCExecution;
  audit: VCAudit;
}

export interface VCCaller {
  did: string;
  type: string;
  agentNodeDid: string;
}

export interface VCTarget {
  did: string;
  agentNodeDid: string;
  functionName: string;
}

export interface VCExecution {
  inputHash: string;
  outputHash: string;
  timestamp: string;
  durationMs: number;
  status: string;
  errorMessage?: string;
}

export interface VCAudit {
  inputDataHash: string;
  outputDataHash: string;
  metadata: Record<string, any>;
}

export interface VCProof {
  type: string;
  created: string;
  verificationMethod: string;
  proofPurpose: string;
  proofValue: string;
}

// Request/Response types for API calls
export interface DIDRegistrationRequest {
  agent_node_id: string;
  reasoners: ReasonerDefinition[];
  skills: SkillDefinition[];
}

export interface ReasonerDefinition {
  function_name: string;
  capabilities: string[];
  exposure_level: string;
}

export interface SkillDefinition {
  function_name: string;
  tags: string[];
  exposure_level: string;
}

export interface DIDRegistrationResponse {
  success: boolean;
  identity_package: DIDIdentityPackage;
  message?: string;
  error?: string;
}

export interface VCVerificationRequest {
  vc_document: any;
}

export interface VCVerificationResponse {
  valid: boolean;
  issuer_did?: string;
  issued_at?: string;
  message?: string;
  error?: string;
}

export interface VerificationIssue {
  type: string;
  severity: 'critical' | 'warning' | 'info';
  component: string;
  field?: string;
  expected?: string;
  actual?: string;
  description: string;
}

export interface IntegrityCheckResults {
  metadata_consistency: boolean;
  field_consistency: boolean;
  timestamp_validation: boolean;
  hash_validation: boolean;
  structural_integrity: boolean;
  issues: VerificationIssue[];
}

export interface SecurityAnalysis {
  signature_strength: string;
  key_validation: boolean;
  did_authenticity: boolean;
  replay_protection: boolean;
  tamper_evidence: string[];
  security_score: number;
  issues: VerificationIssue[];
}

export interface ComplianceChecks {
  w3c_compliance: boolean;
  agentfield_standard_compliance: boolean;
  audit_trail_integrity: boolean;
  data_integrity_checks: boolean;
  issues: VerificationIssue[];
}

export interface ComprehensiveVCVerificationResult {
  valid: boolean;
  overall_score: number; // 0-100
  critical_issues: VerificationIssue[];
  warnings: VerificationIssue[];
  integrity_checks: IntegrityCheckResults;
  security_analysis: SecurityAnalysis;
  compliance_checks: ComplianceChecks;
  verification_timestamp: string;
}

export interface WorkflowVCChainRequest {
  workflow_id: string;
}

export interface WorkflowVCChainResponse {
  workflow_id: string;
  component_vcs: ExecutionVC[];
  workflow_vc: WorkflowVC;
  total_steps: number;
  status: string;
  did_resolution_bundle?: DIDResolutionBundle;
}

export interface DIDResolutionBundle {
  [did: string]: DIDResolutionEntry;
}

export interface DIDResolutionEntry {
  method: string;
  public_key_jwk: any;
  resolved_from: string;
  resolved_at: string;
  error?: string;
}

export interface DIDFilters {
  agentfield_server_id?: string;
  agent_node_id?: string;
  component_type?: string;
  status?: AgentDIDStatus;
  exposure_level?: string;
  created_after?: string;
  created_before?: string;
  limit?: number;
  offset?: number;
}

export interface VCFilters {
  execution_id?: string;
  workflow_id?: string;
  session_id?: string;
  issuer_did?: string;
  caller_did?: string;
  target_did?: string;
  status?: string;
  created_after?: string;
  created_before?: string;
  limit?: number;
  offset?: number;
}

export interface VCExportResponse {
  agent_dids: string[];
  execution_vcs: ExecutionVCInfo[];
  workflow_vcs: WorkflowVC[];
  total_count: number;
  filters_applied: VCFilters;
}

export interface ExecutionVCInfo {
  vc_id: string;
  execution_id: string;
  workflow_id: string;
  session_id: string;
  issuer_did: string;
  target_did: string;
  caller_did: string;
  status: string;
  created_at: string;
  storage_uri?: string;
  document_size_bytes?: number;
}

// UI-specific types
export interface DIDStatusSummary {
  has_did: boolean;
  did_status: AgentDIDStatus;
  reasoner_count: number;
  skill_count: number;
  last_updated: string;
}

export interface VCStatusSummary {
  has_vcs: boolean;
  vc_count: number;
  verified_count: number;
  last_vc_created: string;
  verification_status: 'verified' | 'pending' | 'failed' | 'none';
}

export interface AuditTrailEntry {
  vc_id: string;
  execution_id: string;
  timestamp: string;
  caller_did: string;
  target_did: string;
  status: string;
  input_hash: string;
  output_hash: string;
  signature: string;
}
