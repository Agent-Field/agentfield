package types

import (
	"time"
)

// AccessPolicy defines a tag-based authorization policy for cross-agent calls.
type AccessPolicy struct {
	ID             int64                        `json:"id" db:"id"`
	Name           string                       `json:"name" db:"name"`
	CallerTags     []string                     `json:"caller_tags"`
	TargetTags     []string                     `json:"target_tags"`
	AllowFunctions []string                     `json:"allow_functions"`
	DenyFunctions  []string                     `json:"deny_functions"`
	Constraints    map[string]AccessConstraint  `json:"constraints,omitempty"`
	Action         string                       `json:"action" db:"action"`       // "allow" or "deny"
	Priority       int                          `json:"priority" db:"priority"`
	Enabled        bool                         `json:"enabled" db:"enabled"`
	Description    *string                      `json:"description,omitempty" db:"description"`
	CreatedAt      time.Time                    `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time                    `json:"updated_at" db:"updated_at"`
}

// AccessConstraint defines a parameter constraint for a policy.
type AccessConstraint struct {
	Operator string `json:"operator"` // "<=", ">=", "==", "!=", "<", ">"
	Value    any    `json:"value"`
}

// AccessPolicyRequest represents a request to create or update an access policy.
type AccessPolicyRequest struct {
	Name           string                       `json:"name" binding:"required"`
	CallerTags     []string                     `json:"caller_tags" binding:"required"`
	TargetTags     []string                     `json:"target_tags" binding:"required"`
	AllowFunctions []string                     `json:"allow_functions,omitempty"`
	DenyFunctions  []string                     `json:"deny_functions,omitempty"`
	Constraints    map[string]AccessConstraint  `json:"constraints,omitempty"`
	Action         string                       `json:"action" binding:"required"`
	Priority       int                          `json:"priority,omitempty"`
	Description    string                       `json:"description,omitempty"`
}

// PolicyEvaluationResult represents the outcome of evaluating access policies.
type PolicyEvaluationResult struct {
	Allowed    bool   `json:"allowed"`
	Matched    bool   `json:"matched"`      // true if a policy matched
	PolicyName string `json:"policy_name"`  // which policy matched
	PolicyID   int64  `json:"policy_id"`
	Reason     string `json:"reason"`       // why allow/deny
}

// AccessPolicyListResponse represents the response for listing access policies.
type AccessPolicyListResponse struct {
	Policies []*AccessPolicy `json:"policies"`
	Total    int             `json:"total"`
}

// AgentTagVCDocument is a W3C Verifiable Credential certifying an agent's approved tags.
// Issued when an admin approves an agent's tags. Verified at call time.
type AgentTagVCDocument struct {
	Context           []string                    `json:"@context"`
	Type              []string                    `json:"type"`
	ID                string                      `json:"id"`
	Issuer            string                      `json:"issuer"`
	IssuanceDate      string                      `json:"issuanceDate"`
	ExpirationDate    string                      `json:"expirationDate,omitempty"`
	CredentialSubject AgentTagVCCredentialSubject  `json:"credentialSubject"`
	Proof             *VCProof                    `json:"proof,omitempty"`
}

// AgentTagVCCredentialSubject is the credentialSubject of an AgentTagVC.
type AgentTagVCCredentialSubject struct {
	ID          string                `json:"id"`        // Agent's DID
	AgentID     string                `json:"agent_id"`
	Permissions AgentTagVCPermissions `json:"permissions"`
	ApprovedBy  string                `json:"approved_by,omitempty"`
	ApprovedAt  string                `json:"approved_at,omitempty"`
}

// AgentTagVCPermissions contains the approved tags and callee permissions.
type AgentTagVCPermissions struct {
	Tags           []string `json:"tags"`             // Approved tags
	AllowedCallees []string `json:"allowed_callees"`  // ["*"] = policy decides
}

// AgentTagVCRecord is the DB record for a stored Agent Tag VC.
type AgentTagVCRecord struct {
	ID         int64      `json:"id"`
	AgentID    string     `json:"agent_id"`
	AgentDID   string     `json:"agent_did"`
	VCID       string     `json:"vc_id"`
	VCDocument string     `json:"vc_document"`
	Signature  string     `json:"signature"`
	IssuedAt   time.Time  `json:"issued_at"`
	ExpiresAt  *time.Time `json:"expires_at,omitempty"`
	RevokedAt  *time.Time `json:"revoked_at,omitempty"`
}

// TagApprovalRequest represents a request to approve an agent's tags.
type TagApprovalRequest struct {
	ApprovedTags []string            `json:"approved_tags" binding:"required"`
	SkillTags    map[string][]string `json:"skill_tags,omitempty"`
	ReasonerTags map[string][]string `json:"reasoner_tags,omitempty"`
	Reason       string              `json:"reason,omitempty"`
}

// TagRejectionRequest represents a request to reject an agent's tags.
type TagRejectionRequest struct {
	Reason string `json:"reason,omitempty"`
}

// PendingAgentResponse represents the response for a pending agent's tag info.
type PendingAgentResponse struct {
	AgentID      string   `json:"agent_id"`
	ProposedTags []string `json:"proposed_tags"`
	ApprovedTags []string `json:"approved_tags,omitempty"`
	Status       string   `json:"status"`
	RegisteredAt string   `json:"registered_at"`
}
