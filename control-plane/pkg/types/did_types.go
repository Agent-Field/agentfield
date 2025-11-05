package types

import (
	"encoding/json"
	"time"
)

// DIDRegistry represents the master DID registry for a af server.
type DIDRegistry struct {
	AgentFieldServerID string                  `json:"agentfield_server_id" db:"agentfield_server_id"`
	MasterSeed         []byte                  `json:"master_seed" db:"master_seed_encrypted"`
	RootDID            string                  `json:"root_did" db:"root_did"`
	AgentNodes         map[string]AgentDIDInfo `json:"agent_nodes" db:"agent_nodes"`
	TotalDIDs          int                     `json:"total_dids" db:"total_dids"`
	CreatedAt          time.Time               `json:"created_at" db:"created_at"`
	LastKeyRotation    time.Time               `json:"last_key_rotation" db:"last_key_rotation"`
}

// AgentDIDInfo represents DID information for an agent node.
type AgentDIDInfo struct {
	DID                string                     `json:"did" db:"did"`
	AgentNodeID        string                     `json:"agent_node_id" db:"agent_node_id"`
	AgentFieldServerID string                     `json:"agentfield_server_id" db:"agentfield_server_id"`
	PublicKeyJWK       json.RawMessage            `json:"public_key_jwk" db:"public_key_jwk"`
	DerivationPath     string                     `json:"derivation_path" db:"derivation_path"`
	Reasoners          map[string]ReasonerDIDInfo `json:"reasoners" db:"reasoners"`
	Skills             map[string]SkillDIDInfo    `json:"skills" db:"skills"`
	Status             AgentDIDStatus             `json:"status" db:"status"`
	RegisteredAt       time.Time                  `json:"registered_at" db:"registered_at"`
}

// ReasonerDIDInfo represents DID information for a reasoner.
type ReasonerDIDInfo struct {
	DID            string          `json:"did" db:"did"`
	FunctionName   string          `json:"function_name" db:"function_name"`
	PublicKeyJWK   json.RawMessage `json:"public_key_jwk" db:"public_key_jwk"`
	DerivationPath string          `json:"derivation_path" db:"derivation_path"`
	Capabilities   []string        `json:"capabilities" db:"capabilities"`
	ExposureLevel  string          `json:"exposure_level" db:"exposure_level"`
	CreatedAt      time.Time       `json:"created_at" db:"created_at"`
}

// SkillDIDInfo represents DID information for a skill.
type SkillDIDInfo struct {
	DID            string          `json:"did" db:"did"`
	FunctionName   string          `json:"function_name" db:"function_name"`
	PublicKeyJWK   json.RawMessage `json:"public_key_jwk" db:"public_key_jwk"`
	DerivationPath string          `json:"derivation_path" db:"derivation_path"`
	Tags           []string        `json:"tags" db:"tags"`
	ExposureLevel  string          `json:"exposure_level" db:"exposure_level"`
	CreatedAt      time.Time       `json:"created_at" db:"created_at"`
}

// AgentDIDStatus represents the status of an agent DID.
type AgentDIDStatus string

const (
	AgentDIDStatusActive   AgentDIDStatus = "active"
	AgentDIDStatusInactive AgentDIDStatus = "inactive"
	AgentDIDStatusRevoked  AgentDIDStatus = "revoked"
)

// ExecutionVC represents a verifiable credential for an execution.
type ExecutionVC struct {
	VCID         string          `json:"vc_id" db:"vc_id"`
	ExecutionID  string          `json:"execution_id" db:"execution_id"`
	WorkflowID   string          `json:"workflow_id" db:"workflow_id"`
	SessionID    string          `json:"session_id" db:"session_id"`
	IssuerDID    string          `json:"issuer_did" db:"issuer_did"`
	TargetDID    string          `json:"target_did" db:"target_did"`
	CallerDID    string          `json:"caller_did" db:"caller_did"`
	VCDocument   json.RawMessage `json:"vc_document" db:"vc_document"`
	Signature    string          `json:"signature" db:"signature"`
	StorageURI   string          `json:"storage_uri" db:"storage_uri"`
	DocumentSize int64           `json:"document_size_bytes" db:"document_size_bytes"`
	InputHash    string          `json:"input_hash" db:"input_hash"`
	OutputHash   string          `json:"output_hash" db:"output_hash"`
	Status       string          `json:"status" db:"status"`
	CreatedAt    time.Time       `json:"created_at" db:"created_at"`
	ParentVCID   *string         `json:"parent_vc_id,omitempty" db:"parent_vc_id"`
	ChildVCIDs   []string        `json:"child_vc_ids,omitempty" db:"child_vc_ids"`
}

// WorkflowVC represents a workflow-level verifiable credential.
type WorkflowVC struct {
	WorkflowID     string          `json:"workflow_id" db:"workflow_id"`
	SessionID      string          `json:"session_id" db:"session_id"`
	ComponentVCs   []string        `json:"component_vcs" db:"component_vcs"`
	WorkflowVCID   string          `json:"workflow_vc_id" db:"workflow_vc_id"`
	Status         string          `json:"status" db:"status"`
	StartTime      time.Time       `json:"start_time" db:"start_time"`
	EndTime        *time.Time      `json:"end_time,omitempty" db:"end_time"`
	TotalSteps     int             `json:"total_steps" db:"total_steps"`
	CompletedSteps int             `json:"completed_steps" db:"completed_steps"`
	VCDocument     json.RawMessage `json:"vc_document,omitempty" db:"vc_document"`
	Signature      string          `json:"signature,omitempty" db:"signature"`
	IssuerDID      string          `json:"issuer_did,omitempty" db:"issuer_did"`
	SnapshotTime   time.Time       `json:"snapshot_time,omitempty" db:"snapshot_time"`
	StorageURI     string          `json:"storage_uri" db:"storage_uri"`
	DocumentSize   int64           `json:"document_size_bytes" db:"document_size_bytes"`
}

// DIDIdentityPackage represents the complete DID identity package for an agent.
type DIDIdentityPackage struct {
	AgentDID           DIDIdentity            `json:"agent_did"`
	ReasonerDIDs       map[string]DIDIdentity `json:"reasoner_dids"`
	SkillDIDs          map[string]DIDIdentity `json:"skill_dids"`
	AgentFieldServerID string                 `json:"agentfield_server_id"`
}

// DIDIdentity represents a single DID identity with keys.
type DIDIdentity struct {
	DID            string `json:"did"`
	PrivateKeyJWK  string `json:"private_key_jwk"`
	PublicKeyJWK   string `json:"public_key_jwk"`
	DerivationPath string `json:"derivation_path"`
	ComponentType  string `json:"component_type"`
	FunctionName   string `json:"function_name,omitempty"`
}

// ExecutionContext represents the context for DID-enabled execution.
type ExecutionContext struct {
	ExecutionID  string    `json:"execution_id"`
	WorkflowID   string    `json:"workflow_id"`
	SessionID    string    `json:"session_id"`
	CallerDID    string    `json:"caller_did"`
	TargetDID    string    `json:"target_did"`
	AgentNodeDID string    `json:"agent_node_did"`
	Timestamp    time.Time `json:"timestamp"`
}

// VCDocument represents a complete verifiable credential document.
type VCDocument struct {
	Context           []string            `json:"@context"`
	Type              []string            `json:"type"`
	ID                string              `json:"id"`
	Issuer            string              `json:"issuer"`
	IssuanceDate      string              `json:"issuanceDate"`
	CredentialSubject VCCredentialSubject `json:"credentialSubject"`
	Proof             VCProof             `json:"proof"`
}

// WorkflowVCDocument represents a complete workflow-level verifiable credential document.
type WorkflowVCDocument struct {
	Context           []string                    `json:"@context"`
	Type              []string                    `json:"type"`
	ID                string                      `json:"id"`
	Issuer            string                      `json:"issuer"`
	IssuanceDate      string                      `json:"issuanceDate"`
	CredentialSubject WorkflowVCCredentialSubject `json:"credentialSubject"`
	Proof             VCProof                     `json:"proof"`
}

// VCCredentialSubject represents the subject of a verifiable credential.
type VCCredentialSubject struct {
	ExecutionID string      `json:"executionId"`
	WorkflowID  string      `json:"workflowId"`
	SessionID   string      `json:"sessionId"`
	Caller      VCCaller    `json:"caller"`
	Target      VCTarget    `json:"target"`
	Execution   VCExecution `json:"execution"`
	Audit       VCAudit     `json:"audit"`
}

// WorkflowVCCredentialSubject represents the subject of a workflow-level verifiable credential.
type WorkflowVCCredentialSubject struct {
	WorkflowID     string   `json:"workflowId"`
	SessionID      string   `json:"sessionId"`
	ComponentVCIDs []string `json:"componentVcIds"`
	TotalSteps     int      `json:"totalSteps"`
	CompletedSteps int      `json:"completedSteps"`
	Status         string   `json:"status"`
	StartTime      string   `json:"startTime"`
	EndTime        *string  `json:"endTime,omitempty"`
	SnapshotTime   string   `json:"snapshotTime"`
	Orchestrator   VCCaller `json:"orchestrator"`
	Audit          VCAudit  `json:"audit"`
}

// VCCaller represents the caller information in a VC.
type VCCaller struct {
	DID          string `json:"did"`
	Type         string `json:"type"`
	AgentNodeDID string `json:"agentNodeDid"`
}

// VCTarget represents the target information in a VC.
type VCTarget struct {
	DID          string `json:"did"`
	AgentNodeDID string `json:"agentNodeDid"`
	FunctionName string `json:"functionName"`
}

// VCExecution represents the execution information in a VC.
type VCExecution struct {
	InputHash    string `json:"inputHash"`
	OutputHash   string `json:"outputHash"`
	Timestamp    string `json:"timestamp"`
	DurationMS   int    `json:"durationMs"`
	Status       string `json:"status"`
	ErrorMessage string `json:"errorMessage,omitempty"`
}

// VCAudit represents the audit information in a VC.
type VCAudit struct {
	InputDataHash  string                 `json:"inputDataHash"`
	OutputDataHash string                 `json:"outputDataHash"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// VCProof represents the cryptographic proof in a VC.
type VCProof struct {
	Type               string `json:"type"`
	Created            string `json:"created"`
	VerificationMethod string `json:"verificationMethod"`
	ProofPurpose       string `json:"proofPurpose"`
	ProofValue         string `json:"proofValue"`
}

// DIDFilters holds filters for querying DIDs.
type DIDFilters struct {
	AgentFieldServerID *string         `json:"agentfield_server_id,omitempty"`
	AgentNodeID        *string         `json:"agent_node_id,omitempty"`
	ComponentType      *string         `json:"component_type,omitempty"`
	Status             *AgentDIDStatus `json:"status,omitempty"`
	ExposureLevel      *string         `json:"exposure_level,omitempty"`
	CreatedAfter       *time.Time      `json:"created_after,omitempty"`
	CreatedBefore      *time.Time      `json:"created_before,omitempty"`
	Limit              int             `json:"limit,omitempty"`
	Offset             int             `json:"offset,omitempty"`
}

// VCFilters holds filters for querying VCs.
type VCFilters struct {
	ExecutionID   *string    `json:"execution_id,omitempty"`
	WorkflowID    *string    `json:"workflow_id,omitempty"`
	SessionID     *string    `json:"session_id,omitempty"`
	IssuerDID     *string    `json:"issuer_did,omitempty"`
	CallerDID     *string    `json:"caller_did,omitempty"`
	TargetDID     *string    `json:"target_did,omitempty"`
	Status        *string    `json:"status,omitempty"`
	CreatedAfter  *time.Time `json:"created_after,omitempty"`
	CreatedBefore *time.Time `json:"created_before,omitempty"`
	Limit         int        `json:"limit,omitempty"`
	Offset        int        `json:"offset,omitempty"`
}

// DIDRegistrationRequest represents a request to register an agent with DIDs.
type DIDRegistrationRequest struct {
	AgentNodeID string               `json:"agent_node_id"`
	Reasoners   []ReasonerDefinition `json:"reasoners"`
	Skills      []SkillDefinition    `json:"skills"`
}

// DIDRegistrationResponse represents the response to a DID registration request.
type DIDRegistrationResponse struct {
	Success         bool               `json:"success"`
	IdentityPackage DIDIdentityPackage `json:"identity_package"`
	Message         string             `json:"message,omitempty"`
	Error           string             `json:"error,omitempty"`
}

// VCVerificationRequest represents a request to verify a VC.
type VCVerificationRequest struct {
	VCDocument json.RawMessage `json:"vc_document"`
}

// VCVerificationResponse represents the response to a VC verification request.
type VCVerificationResponse struct {
	Valid     bool   `json:"valid"`
	IssuerDID string `json:"issuer_did,omitempty"`
	IssuedAt  string `json:"issued_at,omitempty"`
	Message   string `json:"message,omitempty"`
	Error     string `json:"error,omitempty"`
}

// WorkflowVCChainRequest represents a request to get a workflow VC chain.
type WorkflowVCChainRequest struct {
	WorkflowID string `json:"workflow_id"`
}

// WorkflowVCChainResponse represents the response containing a workflow VC chain.
type WorkflowVCChainResponse struct {
	WorkflowID   string        `json:"workflow_id"`
	ComponentVCs []ExecutionVC `json:"component_vcs"`
	WorkflowVC   WorkflowVC    `json:"workflow_vc"`
	TotalSteps   int           `json:"total_steps"`
	Status       string        `json:"status"`
	// Enhanced: DID resolution bundle for offline verification
	DIDResolutionBundle map[string]DIDResolutionEntry `json:"did_resolution_bundle,omitempty"`
}

// DIDResolutionEntry represents a resolved DID with its public key for offline verification.
type DIDResolutionEntry struct {
	Method       string          `json:"method"` // "key", "web", etc.
	PublicKeyJWK json.RawMessage `json:"public_key_jwk"`
	ResolvedFrom string          `json:"resolved_from"` // "bundled", "web", "resolver"
	ResolvedAt   string          `json:"resolved_at"`   // ISO 8601 timestamp
}

// DIDRegistryEntry represents a single entry in the DID registry.
type DIDRegistryEntry struct {
	DID            string    `json:"did" db:"did"`
	DIDDocument    string    `json:"did_document" db:"did_document"`
	PublicKey      string    `json:"public_key" db:"public_key"`
	PrivateKeyRef  string    `json:"private_key_ref" db:"private_key_ref"`
	DerivationPath string    `json:"derivation_path" db:"derivation_path"`
	CreatedAt      time.Time `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time `json:"updated_at" db:"updated_at"`
	Status         string    `json:"status" db:"status"`
}

// ComponentDIDInfo represents DID information for a component (reasoner or skill).
type ComponentDIDInfo struct {
	ComponentID     string    `json:"component_id" db:"component_id"`
	ComponentDID    string    `json:"component_did" db:"component_did"`
	AgentDID        string    `json:"agent_did" db:"agent_did"`
	ComponentType   string    `json:"component_type" db:"component_type"`
	ComponentName   string    `json:"component_name" db:"component_name"`
	DerivationIndex int       `json:"derivation_index" db:"derivation_index"`
	CreatedAt       time.Time `json:"created_at" db:"created_at"`
}

// ExecutionVCInfo represents information about an execution VC stored in database.
type ExecutionVCInfo struct {
	VCID         string    `json:"vc_id" db:"vc_id"`
	ExecutionID  string    `json:"execution_id" db:"execution_id"`
	WorkflowID   string    `json:"workflow_id" db:"workflow_id"`
	SessionID    string    `json:"session_id" db:"session_id"`
	IssuerDID    string    `json:"issuer_did" db:"issuer_did"`
	TargetDID    string    `json:"target_did" db:"target_did"`
	CallerDID    string    `json:"caller_did" db:"caller_did"`
	InputHash    string    `json:"input_hash" db:"input_hash"`
	OutputHash   string    `json:"output_hash" db:"output_hash"`
	Status       string    `json:"status" db:"status"`
	CreatedAt    time.Time `json:"created_at" db:"created_at"`
	StorageURI   string    `json:"storage_uri" db:"storage_uri"`
	DocumentSize int64     `json:"document_size_bytes" db:"document_size_bytes"`
}

// WorkflowVCInfo represents information about a workflow VC.
type WorkflowVCInfo struct {
	WorkflowVCID   string     `json:"workflow_vc_id" db:"workflow_vc_id"`
	WorkflowID     string     `json:"workflow_id" db:"workflow_id"`
	SessionID      string     `json:"session_id" db:"session_id"`
	ComponentVCIDs []string   `json:"component_vc_ids" db:"component_vc_ids"`
	Status         string     `json:"status" db:"status"`
	StartTime      time.Time  `json:"start_time" db:"start_time"`
	EndTime        *time.Time `json:"end_time" db:"end_time"`
	TotalSteps     int        `json:"total_steps" db:"total_steps"`
	CompletedSteps int        `json:"completed_steps" db:"completed_steps"`
	StorageURI     string     `json:"storage_uri" db:"storage_uri"`
	DocumentSize   int64      `json:"document_size_bytes" db:"document_size_bytes"`
}

// AgentFieldServerDIDInfo represents af server-level DID information stored in the database.
type AgentFieldServerDIDInfo struct {
	AgentFieldServerID string    `json:"agentfield_server_id" db:"agentfield_server_id"`
	RootDID            string    `json:"root_did" db:"root_did"`
	MasterSeed         []byte    `json:"master_seed" db:"master_seed_encrypted"`
	CreatedAt          time.Time `json:"created_at" db:"created_at"`
	LastKeyRotation    time.Time `json:"last_key_rotation" db:"last_key_rotation"`
}

// RegistrationType represents the type of DID registration being performed.
type RegistrationType string

const (
	RegistrationTypeNew         RegistrationType = "new"
	RegistrationTypeUpdate      RegistrationType = "update"
	RegistrationTypeForceUpdate RegistrationType = "force_update"
)

// EnhancedDIDRegistrationRequest represents an enhanced request to register an agent with DIDs.
type EnhancedDIDRegistrationRequest struct {
	AgentNodeID      string               `json:"agent_node_id"`
	Reasoners        []ReasonerDefinition `json:"reasoners"`
	Skills           []SkillDefinition    `json:"skills"`
	RegistrationType RegistrationType     `json:"registration_type"`
	ForceOverwrite   bool                 `json:"force_overwrite,omitempty"`
}

// DifferentialAnalysisResult represents the result of comparing existing vs new reasoners/skills.
type DifferentialAnalysisResult struct {
	NewReasonerIDs     []string `json:"new_reasoner_ids"`
	UpdatedReasonerIDs []string `json:"updated_reasoner_ids"`
	RemovedReasonerIDs []string `json:"removed_reasoner_ids"`
	NewSkillIDs        []string `json:"new_skill_ids"`
	UpdatedSkillIDs    []string `json:"updated_skill_ids"`
	RemovedSkillIDs    []string `json:"removed_skill_ids"`
	RequiresUpdate     bool     `json:"requires_update"`
}

// PartialDIDRegistrationRequest represents a request for partial DID registration.
type PartialDIDRegistrationRequest struct {
	AgentNodeID        string               `json:"agent_node_id"`
	NewReasonerIDs     []string             `json:"new_reasoner_ids"`
	NewSkillIDs        []string             `json:"new_skill_ids"`
	UpdatedReasonerIDs []string             `json:"updated_reasoner_ids"`
	UpdatedSkillIDs    []string             `json:"updated_skill_ids"`
	AllReasoners       []ReasonerDefinition `json:"all_reasoners"`
	AllSkills          []SkillDefinition    `json:"all_skills"`
}

// ComponentDeregistrationRequest represents a request to deregister specific components.
type ComponentDeregistrationRequest struct {
	AgentNodeID         string   `json:"agent_node_id"`
	ReasonerIDsToRemove []string `json:"reasoner_ids_to_remove"`
	SkillIDsToRemove    []string `json:"skill_ids_to_remove"`
}

// PartialDIDRegistrationResponse represents the response to a partial DID registration request.
type PartialDIDRegistrationResponse struct {
	Success         bool               `json:"success"`
	IdentityPackage DIDIdentityPackage `json:"identity_package"`
	Message         string             `json:"message,omitempty"`
	Error           string             `json:"error,omitempty"`
	NewReasonerDIDs int                `json:"new_reasoner_dids"`
	NewSkillDIDs    int                `json:"new_skill_dids"`
}

// ComponentDeregistrationResponse represents the response to a component deregistration request.
type ComponentDeregistrationResponse struct {
	Success      bool   `json:"success"`
	RemovedCount int    `json:"removed_count"`
	Message      string `json:"message,omitempty"`
	Error        string `json:"error,omitempty"`
}
