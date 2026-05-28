package services

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockTagApprovalStorage implements TagApprovalStorage for testing.
type mockTagApprovalStorage struct {
	agents   map[string]*types.AgentNode
	agentDID map[string]*types.AgentDIDInfo
	tagVCs   map[string]*types.AgentTagVCRecord

	registerErr error
}

func newMockTagApprovalStorage() *mockTagApprovalStorage {
	return &mockTagApprovalStorage{
		agents:   make(map[string]*types.AgentNode),
		agentDID: make(map[string]*types.AgentDIDInfo),
		tagVCs:   make(map[string]*types.AgentTagVCRecord),
	}
}

func (m *mockTagApprovalStorage) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	agent, ok := m.agents[id]
	if !ok {
		return nil, fmt.Errorf("agent %s not found", id)
	}
	return agent, nil
}

func (m *mockTagApprovalStorage) ListAgentVersions(_ context.Context, id string) ([]*types.AgentNode, error) {
	// Mock returns nothing; tests use unversioned agents stored via GetAgent key
	return nil, nil
}

func (m *mockTagApprovalStorage) RegisterAgent(_ context.Context, node *types.AgentNode) error {
	if m.registerErr != nil {
		return m.registerErr
	}
	m.agents[node.ID] = node
	return nil
}

func (m *mockTagApprovalStorage) ListAgentsByLifecycleStatus(_ context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	var result []*types.AgentNode
	for _, agent := range m.agents {
		if agent.LifecycleStatus == status {
			result = append(result, agent)
		}
	}
	return result, nil
}

func (m *mockTagApprovalStorage) GetAgentDID(_ context.Context, agentID string) (*types.AgentDIDInfo, error) {
	info, ok := m.agentDID[agentID]
	if !ok {
		return nil, fmt.Errorf("DID not found for agent %s", agentID)
	}
	return info, nil
}

func (m *mockTagApprovalStorage) StoreAgentTagVC(_ context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	m.tagVCs[agentID] = &types.AgentTagVCRecord{
		AgentID:    agentID,
		AgentDID:   agentDID,
		VCID:       vcID,
		VCDocument: vcDocument,
		Signature:  signature,
		IssuedAt:   issuedAt,
		ExpiresAt:  expiresAt,
	}
	return nil
}

func (m *mockTagApprovalStorage) RevokeAgentTagVC(_ context.Context, agentID string) error {
	delete(m.tagVCs, agentID)
	return nil
}

func testApprovalConfig(rules ...config.TagApprovalRule) config.TagApprovalRulesConfig {
	return config.TagApprovalRulesConfig{
		DefaultMode: "manual",
		Rules:       rules,
	}
}

// ============================================================================
// EvaluateTags
// ============================================================================

func TestEvaluateTags_AutoApprovedTags(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"internal", "beta"}, Approval: "auto"},
	), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"internal", "beta"})
	assert.Equal(t, []string{"internal", "beta"}, result.AutoApproved)
	assert.Empty(t, result.ManualReview)
	assert.Empty(t, result.Forbidden)
	assert.True(t, result.AllAutoApproved)
}

func TestEvaluateTags_ManualReviewTags(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"finance"}, Approval: "manual"},
	), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"finance"})
	assert.Empty(t, result.AutoApproved)
	assert.Equal(t, []string{"finance"}, result.ManualReview)
	assert.False(t, result.AllAutoApproved)
}

func TestEvaluateTags_ForbiddenTags(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"root", "superuser"}, Approval: "forbidden"},
	), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"root"})
	assert.Empty(t, result.AutoApproved)
	assert.Empty(t, result.ManualReview)
	assert.Equal(t, []string{"root"}, result.Forbidden)
	assert.False(t, result.AllAutoApproved)
}

func TestEvaluateTags_MixedModes(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"internal"}, Approval: "auto"},
		config.TagApprovalRule{Tags: []string{"finance"}, Approval: "manual"},
		config.TagApprovalRule{Tags: []string{"root"}, Approval: "forbidden"},
	), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"internal", "finance", "root"})
	assert.Equal(t, []string{"internal"}, result.AutoApproved)
	assert.Equal(t, []string{"finance"}, result.ManualReview)
	assert.Equal(t, []string{"root"}, result.Forbidden)
	assert.False(t, result.AllAutoApproved)
}

func TestEvaluateTags_DefaultModeFallback(t *testing.T) {
	// Tags not in any rule should use the default mode
	svc := NewTagApprovalService(config.TagApprovalRulesConfig{
		DefaultMode: "manual",
		Rules:       nil,
	}, newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"unknown-tag"})
	assert.Equal(t, []string{"unknown-tag"}, result.ManualReview)
	assert.False(t, result.AllAutoApproved)
}

func TestEvaluateTags_DefaultModeAutoIfEmpty(t *testing.T) {
	// If no default mode configured, defaults to "auto"
	svc := NewTagApprovalService(config.TagApprovalRulesConfig{}, newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"any-tag"})
	assert.Equal(t, []string{"any-tag"}, result.AutoApproved)
	assert.True(t, result.AllAutoApproved)
}

func TestEvaluateTags_CaseInsensitive(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"Finance"}, Approval: "manual"},
	), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"FINANCE"})
	assert.Equal(t, []string{"finance"}, result.ManualReview)
}

func TestEvaluateTags_EmptyAndWhitespaceTags(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(), newMockTagApprovalStorage())

	result := svc.EvaluateTags([]string{"", "  ", "valid"})
	// Empty/whitespace tags should be skipped; "valid" goes to default mode (manual)
	assert.Equal(t, []string{"valid"}, result.ManualReview)
	assert.Empty(t, result.AutoApproved)
}

// ============================================================================
// IsEnabled
// ============================================================================

func TestIsEnabled_WithRules(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"admin"}, Approval: "manual"},
	), newMockTagApprovalStorage())
	assert.True(t, svc.IsEnabled())
}

func TestIsEnabled_WithNonAutoDefault(t *testing.T) {
	svc := NewTagApprovalService(config.TagApprovalRulesConfig{
		DefaultMode: "manual",
	}, newMockTagApprovalStorage())
	assert.True(t, svc.IsEnabled())
}

func TestIsEnabled_DefaultAutoNoRules(t *testing.T) {
	svc := NewTagApprovalService(config.TagApprovalRulesConfig{}, newMockTagApprovalStorage())
	assert.False(t, svc.IsEnabled())
}

// ============================================================================
// CollectAllProposedTags
// ============================================================================

func TestCollectAllProposedTags_FromReasonersAndSkills(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", Tags: []string{"finance"}, ProposedTags: []string{"finance", "internal"}},
			{ID: "r2", Tags: []string{"billing"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", Tags: []string{"payment"}, ProposedTags: []string{"payment"}},
		},
	}

	tags := CollectAllProposedTags(agent)
	// Should use ProposedTags when available, fall back to Tags
	assert.Contains(t, tags, "finance")
	assert.Contains(t, tags, "internal")
	assert.Contains(t, tags, "billing")
	assert.Contains(t, tags, "payment")
}

func TestCollectAllProposedTags_DeduplicatesAndNormalizes(t *testing.T) {
	agent := &types.AgentNode{
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", ProposedTags: []string{"Finance", "internal"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"FINANCE", "payment"}},
		},
	}

	tags := CollectAllProposedTags(agent)
	// "Finance" and "FINANCE" should deduplicate to one entry
	finCount := 0
	for _, t := range tags {
		if t == "finance" {
			finCount++
		}
	}
	assert.Equal(t, 1, finCount, "should deduplicate case-insensitive tags")
	assert.Len(t, tags, 3) // finance, internal, payment
}

// ============================================================================
// ApproveAgentTags
// ============================================================================

func TestApproveAgentTags_HappyPath(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"finance", "payment"},
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", ProposedTags: []string{"finance"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"finance", "payment"}},
		},
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)

	err := svc.ApproveAgentTags(context.Background(), "agent-1", []string{"finance", "payment"}, "admin-user")
	require.NoError(t, err)

	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusStarting, agent.LifecycleStatus)
	assert.Equal(t, []string{"finance", "payment"}, agent.ApprovedTags)
	// Per-skill: reasoner r1 proposed ["finance"], approved set has "finance" → approved
	assert.Equal(t, []string{"finance"}, agent.Reasoners[0].ApprovedTags)
	// Skill s1 proposed ["finance", "payment"], both in approved set
	assert.Equal(t, []string{"finance", "payment"}, agent.Skills[0].ApprovedTags)
}

func TestApproveAgentTags_PartialApproval(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"finance", "admin"},
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"finance", "admin"}},
		},
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)

	// Admin approves only "finance", not "admin"
	err := svc.ApproveAgentTags(context.Background(), "agent-1", []string{"finance"}, "admin-user")
	require.NoError(t, err)

	agent := storage.agents["agent-1"]
	assert.Equal(t, []string{"finance"}, agent.ApprovedTags)
	// Skill proposed ["finance", "admin"], only "finance" is in approved set
	assert.Equal(t, []string{"finance"}, agent.Skills[0].ApprovedTags)
}

func TestApproveAgentTags_NonPendingAgentFails(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)

	err := svc.ApproveAgentTags(context.Background(), "agent-1", []string{"finance"}, "admin")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not pending approval")
}

func TestApproveAgentTags_AgentNotFoundFails(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(), newMockTagApprovalStorage())
	err := svc.ApproveAgentTags(context.Background(), "nonexistent", []string{"finance"}, "admin")
	assert.Error(t, err)
}

// ============================================================================
// ApproveAgentTagsPerSkill
// ============================================================================

func TestApproveAgentTagsPerSkill_HappyPath(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", ProposedTags: []string{"finance", "internal"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"payment"}},
			{ID: "s2", ProposedTags: []string{"billing"}},
		},
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)

	err := svc.ApproveAgentTagsPerSkill(context.Background(), "agent-1",
		map[string][]string{
			"s1": {"payment"},
			// s2 not approved
		},
		map[string][]string{
			"r1": {"finance"}, // internal not approved
		},
		"admin-user",
	)
	require.NoError(t, err)

	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusStarting, agent.LifecycleStatus)
	assert.Equal(t, []string{"finance"}, agent.Reasoners[0].ApprovedTags)
	assert.Equal(t, []string{"payment"}, agent.Skills[0].ApprovedTags)
	assert.Nil(t, agent.Skills[1].ApprovedTags) // s2 not in approval map

	// Agent-level approved tags = union of all per-skill approved tags
	assert.Contains(t, agent.ApprovedTags, "finance")
	assert.Contains(t, agent.ApprovedTags, "payment")
}

func TestApproveAgentTagsPerSkill_NonPendingFails(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)
	err := svc.ApproveAgentTagsPerSkill(context.Background(), "agent-1", nil, nil, "admin")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not pending approval")
}

// ============================================================================
// RejectAgentTags
// ============================================================================

func TestRejectAgentTags_HappyPath(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
		ProposedTags:    []string{"root"},
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", ProposedTags: []string{"root"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", ProposedTags: []string{"root"}},
		},
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)

	err := svc.RejectAgentTags(context.Background(), "agent-1", "admin", "Forbidden tags")
	require.NoError(t, err)

	agent := storage.agents["agent-1"]
	assert.Equal(t, types.AgentStatusOffline, agent.LifecycleStatus)
	assert.Nil(t, agent.ApprovedTags)
	assert.Nil(t, agent.Reasoners[0].ApprovedTags)
	assert.Nil(t, agent.Skills[0].ApprovedTags)
}

func TestRejectAgentTags_NonPendingFails(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["agent-1"] = &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusStarting,
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)
	err := svc.RejectAgentTags(context.Background(), "agent-1", "admin", "reason")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not pending approval")
}

// ============================================================================
// ListPendingAgents
// ============================================================================

func TestListPendingAgents_ReturnsOnlyPending(t *testing.T) {
	storage := newMockTagApprovalStorage()
	storage.agents["pending-1"] = &types.AgentNode{
		ID:              "pending-1",
		LifecycleStatus: types.AgentStatusPendingApproval,
	}
	storage.agents["ready-1"] = &types.AgentNode{
		ID:              "ready-1",
		LifecycleStatus: types.AgentStatusReady,
	}
	storage.agents["pending-2"] = &types.AgentNode{
		ID:              "pending-2",
		LifecycleStatus: types.AgentStatusPendingApproval,
	}

	svc := NewTagApprovalService(testApprovalConfig(), storage)
	agents, err := svc.ListPendingAgents(context.Background())
	require.NoError(t, err)

	assert.Len(t, agents, 2)
	for _, a := range agents {
		assert.Equal(t, types.AgentStatusPendingApproval, a.LifecycleStatus)
	}
}

// ============================================================================
// ProcessRegistrationTags
// ============================================================================

func TestProcessRegistrationTags_AllAutoApproved(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"internal", "beta"}, Approval: "auto"},
	), newMockTagApprovalStorage())

	agent := &types.AgentNode{
		ID: "agent-1",
		Reasoners: []types.ReasonerDefinition{
			{ID: "r1", Tags: []string{"internal"}},
		},
		Skills: []types.SkillDefinition{
			{ID: "s1", Tags: []string{"beta"}},
		},
	}

	result := svc.ProcessRegistrationTags(agent)
	assert.True(t, result.AllAutoApproved)
	assert.Equal(t, []string{"internal", "beta"}, agent.ProposedTags)
	assert.Equal(t, []string{"internal", "beta"}, agent.ApprovedTags)
	// Lifecycle status should NOT be set to pending
	assert.NotEqual(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
}

func TestProcessRegistrationTags_NeedsApproval(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"internal"}, Approval: "auto"},
		config.TagApprovalRule{Tags: []string{"finance"}, Approval: "manual"},
	), newMockTagApprovalStorage())

	agent := &types.AgentNode{
		ID: "agent-1",
		Skills: []types.SkillDefinition{
			{ID: "s1", Tags: []string{"internal", "finance"}},
		},
	}

	result := svc.ProcessRegistrationTags(agent)
	assert.False(t, result.AllAutoApproved)
	assert.Equal(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
	// Only auto-approved tags should be set
	assert.Equal(t, []string{"internal"}, agent.ApprovedTags)
}

func TestProcessRegistrationTags_ForbiddenTagBlocksApproval(t *testing.T) {
	svc := NewTagApprovalService(testApprovalConfig(
		config.TagApprovalRule{Tags: []string{"root"}, Approval: "forbidden"},
	), newMockTagApprovalStorage())

	agent := &types.AgentNode{
		ID: "agent-1",
		Skills: []types.SkillDefinition{
			{ID: "s1", Tags: []string{"root"}},
		},
	}

	result := svc.ProcessRegistrationTags(agent)
	assert.False(t, result.AllAutoApproved)
	assert.Equal(t, []string{"root"}, result.Forbidden)
	assert.Equal(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
}
