package services

import (
	"context"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type mockTagApprovalVCService struct {
	didService *DIDService
	signErr    error
}

func (m *mockTagApprovalVCService) GetDIDService() *DIDService {
	return m.didService
}

func (m *mockTagApprovalVCService) SignAgentTagVC(_ *types.AgentTagVCDocument) (*types.VCProof, error) {
	if m.signErr != nil {
		return nil, m.signErr
	}
	return &types.VCProof{
		Type:               "Ed25519Signature2020",
		Created:            time.Now().UTC().Format(time.RFC3339),
		VerificationMethod: "did:key:test#key-1",
		ProofPurpose:       "assertionMethod",
		ProofValue:         "signature",
	}, nil
}

func TestTagApprovalServiceRevokeAndIssueTagVC(t *testing.T) {
	didService, _, _, _, _ := setupDIDTestEnvironment(t)
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
		ApprovedTags:    []string{"finance"},
		Reasoners:       []types.ReasonerDefinition{{ID: "r1", ApprovedTags: []string{"finance"}}},
		Skills:          []types.SkillDefinition{{ID: "s1", ApprovedTags: []string{"finance"}}},
	}
	storage.agentDID["agent-1"] = &types.AgentDIDInfo{DID: "did:key:agent-1"}

	service := NewTagApprovalService(config.TagApprovalRulesConfig{}, storage)
	var revokedAgentID string
	service.SetOnRevokeCallback(func(_ context.Context, agentID string) {
		revokedAgentID = agentID
	})
	service.SetVCService(&mockTagApprovalVCService{didService: didService})

	service.IssueAutoApprovedTagsVC(context.Background(), "agent-1", []string{"finance"})
	require.Contains(t, storage.tagVCs, "agent-1")
	require.Equal(t, "did:key:agent-1", storage.tagVCs["agent-1"].AgentDID)
	require.NotEmpty(t, storage.tagVCs["agent-1"].Signature)

	err := service.RevokeAgentTags(context.Background(), "agent-1", "admin", "policy change")
	require.NoError(t, err)
	require.Equal(t, "agent-1", revokedAgentID)
	require.Equal(t, types.AgentStatusPendingApproval, storage.agents["agent-1"].LifecycleStatus)
	require.Empty(t, storage.agents["agent-1"].ApprovedTags)
	require.Nil(t, storage.agents["agent-1"].Reasoners[0].ApprovedTags)
	require.Nil(t, storage.agents["agent-1"].Skills[0].ApprovedTags)
	require.NotContains(t, storage.tagVCs, "agent-1")

	err = service.RevokeAgentTags(context.Background(), "agent-1", "admin", "repeat")
	require.ErrorContains(t, err, "already_revoked")
}
