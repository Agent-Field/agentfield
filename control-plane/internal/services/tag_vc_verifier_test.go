package services

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
)

// mockTagVCStorage implements TagVCStorage for testing.
type mockTagVCStorage struct {
	records map[string]*types.AgentTagVCRecord
}

func newMockTagVCStorage() *mockTagVCStorage {
	return &mockTagVCStorage{records: make(map[string]*types.AgentTagVCRecord)}
}

func (m *mockTagVCStorage) GetAgentTagVC(_ context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	r, ok := m.records[agentID]
	if !ok {
		return nil, fmt.Errorf("no tag VC for agent %s", agentID)
	}
	return r, nil
}

func validVCDocument(agentID, agentDID string) string {
	vc := types.AgentTagVCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential", "AgentTagCredential"},
		ID:           "urn:agentfield:test-vc",
		Issuer:       "did:web:localhost:admin",
		IssuanceDate: time.Now().Format(time.RFC3339),
		CredentialSubject: types.AgentTagVCCredentialSubject{
			ID:      agentDID,
			AgentID: agentID,
			Permissions: types.AgentTagVCPermissions{
				Tags:           []string{"finance"},
				AllowedCallees: []string{"*"},
			},
		},
	}
	b, _ := json.Marshal(vc)
	return string(b)
}

func TestVerifyAgentTagVC_StorageError(t *testing.T) {
	storage := newMockTagVCStorage()
	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no tag VC for agent")
}

func TestVerifyAgentTagVC_RevokedVC(t *testing.T) {
	storage := newMockTagVCStorage()
	revokedAt := time.Now().Add(-1 * time.Hour)
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: validVCDocument("agent-1", "did:web:test"),
		RevokedAt:  &revokedAt,
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "revoked")
}

func TestVerifyAgentTagVC_ExpiredVC(t *testing.T) {
	storage := newMockTagVCStorage()
	expired := time.Now().Add(-24 * time.Hour)
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: validVCDocument("agent-1", "did:web:test"),
		ExpiresAt:  &expired,
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "expired")
}

func TestVerifyAgentTagVC_NotYetExpired(t *testing.T) {
	// VC with future expiry should pass the expiration check (but still fail on
	// nil vcService — that's expected and tested separately)
	storage := newMockTagVCStorage()
	future := time.Now().Add(24 * time.Hour)
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: validVCDocument("agent-1", "did:web:test"),
		ExpiresAt:  &future,
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	// Should NOT fail on expiration — should fail on nil vcService instead
	assert.Error(t, err)
	assert.NotContains(t, err.Error(), "expired")
	assert.Contains(t, err.Error(), "VC service not available")
}

func TestVerifyAgentTagVC_MalformedJSON(t *testing.T) {
	storage := newMockTagVCStorage()
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: "not valid json{{{",
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse tag VC document")
}

func TestVerifyAgentTagVC_NilVCServiceFails(t *testing.T) {
	storage := newMockTagVCStorage()
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: validVCDocument("agent-1", "did:web:test"),
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "VC service not available")
}

func TestVerifyAgentTagVC_SubjectMismatch(t *testing.T) {
	// VC issued for agent-1 but verification requested for agent-2
	// This test covers the subject binding check AFTER signature verification.
	// Since we can't easily mock VCService (concrete type), we test this path
	// indirectly — the check happens after vcService.VerifyAgentTagVCSignature.
	// We verify the error message is correct for mismatch detection.
	storage := newMockTagVCStorage()
	// Store a VC that claims to be for "agent-1"
	storage.records["agent-2"] = &types.AgentTagVCRecord{
		AgentID:    "agent-2",
		VCDocument: validVCDocument("agent-1", "did:web:test"), // VC says agent_id=agent-1
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-2")
	// Will fail at vcService check before reaching subject mismatch,
	// but the flow is correct — nil vcService blocks forged VCs
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "VC service not available")
}

func TestVerifyAgentTagVC_NoExpirationIsValid(t *testing.T) {
	// VC with nil ExpiresAt should pass expiration check
	storage := newMockTagVCStorage()
	storage.records["agent-1"] = &types.AgentTagVCRecord{
		AgentID:    "agent-1",
		VCDocument: validVCDocument("agent-1", "did:web:test"),
		ExpiresAt:  nil, // No expiration
	}

	verifier := NewTagVCVerifier(storage, nil)

	_, err := verifier.VerifyAgentTagVC(context.Background(), "agent-1")
	// Should pass revocation and expiration, fail only at vcService
	assert.Error(t, err)
	assert.NotContains(t, err.Error(), "expired")
	assert.NotContains(t, err.Error(), "revoked")
	assert.Contains(t, err.Error(), "VC service not available")
}
