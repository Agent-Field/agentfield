package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDIDServiceHandlePartialRegistrationRemovalsOnly(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	registration := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-removals-only",
		Reasoners: []types.ReasonerDefinition{
			{ID: "reasoner-1"},
			{ID: "reasoner-2"},
		},
		Skills: []types.SkillDefinition{
			{ID: "skill-1"},
		},
	}

	resp, err := service.RegisterAgent(registration)
	require.NoError(t, err)
	require.True(t, resp.Success)

	diff := &types.DifferentialAnalysisResult{
		RemovedReasonerIDs: []string{"reasoner-2"},
		UpdatedReasonerIDs: []string{"reasoner-1"},
		UpdatedSkillIDs:    []string{"skill-1"},
		RequiresUpdate:     true,
	}

	resp, err = service.handlePartialRegistration(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-removals-only",
		Reasoners:   registration.Reasoners[:1],
		Skills:      registration.Skills,
	}, diff)
	require.NoError(t, err)
	require.True(t, resp.Success)
	require.Contains(t, resp.Message, "removed 1 reasoners, 0 skills")
	require.Equal(t, "agent-removals-only", service.ResolveAgentIDByDID(resp.IdentityPackage.AgentDID.DID))

	existing, err := service.GetExistingAgentDID("agent-removals-only")
	require.NoError(t, err)
	require.Contains(t, existing.Reasoners, "reasoner-1")
	require.NotContains(t, existing.Reasoners, "reasoner-2")
	require.Contains(t, existing.Skills, "skill-1")
}
