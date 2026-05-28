package services

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// TagVCStorage defines the storage operations needed by the tag VC verifier.
type TagVCStorage interface {
	GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error)
}

// TagVCVerifier loads and verifies Agent Tag VCs at call time.
type TagVCVerifier struct {
	storage   TagVCStorage
	vcService *VCService
}

// NewTagVCVerifier creates a new tag VC verifier.
func NewTagVCVerifier(storage TagVCStorage, vcService *VCService) *TagVCVerifier {
	return &TagVCVerifier{
		storage:   storage,
		vcService: vcService,
	}
}

// VerifyAgentTagVC loads an agent's tag VC from storage, verifies the signature,
// checks expiration/revocation, and returns the parsed VC document.
func (v *TagVCVerifier) VerifyAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCDocument, error) {
	// Load VC record from storage
	record, err := v.storage.GetAgentTagVC(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("no tag VC for agent %s: %w", agentID, err)
	}

	// Check revocation
	if record.RevokedAt != nil {
		return nil, fmt.Errorf("tag VC for agent %s was revoked at %s", agentID, record.RevokedAt.Format(time.RFC3339))
	}

	// Check expiration
	if record.ExpiresAt != nil && record.ExpiresAt.Before(time.Now()) {
		return nil, fmt.Errorf("tag VC for agent %s expired at %s", agentID, record.ExpiresAt.Format(time.RFC3339))
	}

	// Parse VC document
	var vc types.AgentTagVCDocument
	if err := json.Unmarshal([]byte(record.VCDocument), &vc); err != nil {
		return nil, fmt.Errorf("failed to parse tag VC document for agent %s: %w", agentID, err)
	}

	// Verify Ed25519 signature — vcService is required for signature verification
	if v.vcService == nil {
		return nil, fmt.Errorf("cannot verify tag VC for agent %s: VC service not available", agentID)
	}

	valid, err := v.vcService.VerifyAgentTagVCSignature(&vc)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Tag VC signature verification failed")
		return nil, fmt.Errorf("tag VC signature verification failed for agent %s: %w", agentID, err)
	}
	if !valid {
		return nil, fmt.Errorf("tag VC signature is invalid for agent %s", agentID)
	}

	// Validate issuer-subject binding: the VC's agent ID must match the requested agent
	if vc.CredentialSubject.AgentID != "" && vc.CredentialSubject.AgentID != agentID {
		return nil, fmt.Errorf("tag VC subject mismatch: VC is for agent %s but verification requested for %s", vc.CredentialSubject.AgentID, agentID)
	}

	return &vc, nil
}
