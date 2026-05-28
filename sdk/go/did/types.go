// Package did provides DID (Decentralized Identifier) authentication and
// Verifiable Credential generation for AgentField Go SDK agents.
package did

// DIDIdentity represents a single DID with associated cryptographic keys.
type DIDIdentity struct {
	DID            string `json:"did"`
	PrivateKeyJWK  string `json:"private_key_jwk,omitempty"`
	PublicKeyJWK   string `json:"public_key_jwk"`
	DerivationPath string `json:"derivation_path"`
	ComponentType  string `json:"component_type"` // "agent", "reasoner", "skill"
	FunctionName   string `json:"function_name,omitempty"`
}

// DIDIdentityPackage is the complete set of DIDs returned by the control plane
// after agent registration. It includes the agent-level DID and per-function DIDs.
type DIDIdentityPackage struct {
	AgentDID           DIDIdentity            `json:"agent_did"`
	ReasonerDIDs       map[string]DIDIdentity `json:"reasoner_dids"`
	SkillDIDs          map[string]DIDIdentity `json:"skill_dids"`
	AgentFieldServerID string                 `json:"agentfield_server_id"`
}

// RegistrationRequest is sent to the control plane to register agent DIDs.
type RegistrationRequest struct {
	AgentNodeID string        `json:"agent_node_id"`
	Reasoners   []FunctionDef `json:"reasoners"`
	Skills      []FunctionDef `json:"skills"`
}

// FunctionDef identifies a reasoner or skill during DID registration.
type FunctionDef struct {
	ID string `json:"id"`
}

// RegistrationResponse is the response from the DID registration endpoint.
type RegistrationResponse struct {
	Success         bool               `json:"success"`
	IdentityPackage DIDIdentityPackage `json:"identity_package"`
	Error           string             `json:"error,omitempty"`
}

// ExecutionContext carries DID-specific metadata for a single execution,
// used when generating Verifiable Credentials.
type ExecutionContext struct {
	ExecutionID  string `json:"execution_id"`
	WorkflowID   string `json:"workflow_id,omitempty"`
	SessionID    string `json:"session_id,omitempty"`
	CallerDID    string `json:"caller_did,omitempty"`
	TargetDID    string `json:"target_did,omitempty"`
	AgentNodeDID string `json:"agent_node_did,omitempty"`
	Timestamp    string `json:"timestamp,omitempty"`
}

// VCGenerationRequest is the payload for generating a Verifiable Credential.
type VCGenerationRequest struct {
	ExecutionContext ExecutionContext `json:"execution_context"`
	InputData        string          `json:"input_data"`
	OutputData       string          `json:"output_data"`
	Status           string          `json:"status"`
	ErrorMessage     string          `json:"error_message,omitempty"`
	DurationMS       int64           `json:"duration_ms,omitempty"`
}

// ExecutionVC represents a Verifiable Credential generated for an execution.
type ExecutionVC struct {
	VCID        string `json:"vc_id"`
	ExecutionID string `json:"execution_id"`
	WorkflowID  string `json:"workflow_id"`
	SessionID   string `json:"session_id,omitempty"`
	IssuerDID   string `json:"issuer_did"`
	TargetDID   string `json:"target_did"`
	CallerDID   string `json:"caller_did,omitempty"`
	VCDocument  any    `json:"vc_document"`
	Signature   string `json:"signature"`
	InputHash   string `json:"input_hash"`
	OutputHash  string `json:"output_hash"`
	Status      string `json:"status"`
	CreatedAt   string `json:"created_at"`
}

// WorkflowVCChain is the audit trail for a workflow, containing all execution VCs.
type WorkflowVCChain struct {
	WorkflowID   string        `json:"workflow_id"`
	ExecutionVCs []ExecutionVC `json:"execution_vcs"`
	WorkflowVC   any           `json:"workflow_vc,omitempty"`
}
