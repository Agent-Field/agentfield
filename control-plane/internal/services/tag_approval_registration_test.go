package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestProcessRegistrationTags_PreservesExistingApprovalWhenCovered(t *testing.T) {
	service := NewTagApprovalService(config.TagApprovalRulesConfig{
		DefaultMode: "manual",
		Rules: []config.TagApprovalRule{
			{Tags: []string{"finance"}, Approval: "manual"},
		},
	}, newMockTagApprovalStorage())

	agent := &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
		ApprovedTags:    []string{"finance"},
		Skills:          []types.SkillDefinition{{ID: "s1", Tags: []string{"finance"}}},
	}

	result := service.ProcessRegistrationTags(agent)
	require.False(t, result.AllAutoApproved)
	require.Equal(t, []string{"finance"}, result.ManualReview)
	require.Equal(t, types.AgentStatusReady, agent.LifecycleStatus)
	require.Equal(t, []string{"finance"}, agent.ApprovedTags)
}

func TestProcessRegistrationTags_MergesExistingApprovalWithNewAutoApprovedTags(t *testing.T) {
	service := NewTagApprovalService(config.TagApprovalRulesConfig{
		DefaultMode: "manual",
		Rules: []config.TagApprovalRule{
			{Tags: []string{"finance"}, Approval: "manual"},
			{Tags: []string{"beta"}, Approval: "auto"},
			{Tags: []string{"security"}, Approval: "manual"},
		},
	}, newMockTagApprovalStorage())

	agent := &types.AgentNode{
		ID:              "agent-1",
		LifecycleStatus: types.AgentStatusReady,
		ApprovedTags:    []string{"finance"},
		Skills: []types.SkillDefinition{
			{ID: "s1", Tags: []string{"finance", "beta", "security"}},
		},
	}

	result := service.ProcessRegistrationTags(agent)
	require.False(t, result.AllAutoApproved)
	require.Equal(t, []string{"beta"}, result.AutoApproved)
	require.Equal(t, []string{"finance", "security"}, result.ManualReview)
	require.Equal(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
	require.ElementsMatch(t, []string{"finance", "beta"}, agent.ApprovedTags)
}
