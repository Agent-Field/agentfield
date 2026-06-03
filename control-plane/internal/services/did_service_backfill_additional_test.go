package services

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDIDServiceRegisterAgentUpdatesExistingComponents(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	initial := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-update",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
		Skills:      []types.SkillDefinition{{ID: "skill-1"}},
	}
	resp, err := service.RegisterAgent(initial)
	require.NoError(t, err)
	require.True(t, resp.Success)

	updated := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-update",
		Reasoners: []types.ReasonerDefinition{
			{ID: "reasoner-1"},
			{ID: "reasoner-2"},
		},
		Skills: []types.SkillDefinition{
			{ID: "skill-2", Tags: []string{"approved"}},
		},
	}

	resp, err = service.RegisterAgent(updated)
	require.NoError(t, err)
	require.True(t, resp.Success)
	require.Contains(t, resp.Message, "Partial registration successful")
	require.Contains(t, resp.IdentityPackage.ReasonerDIDs, "reasoner-2")
	require.Contains(t, resp.IdentityPackage.SkillDIDs, "skill-2")

	existing, err := service.GetExistingAgentDID("agent-update")
	require.NoError(t, err)
	require.Contains(t, existing.Reasoners, "reasoner-1")
	require.Contains(t, existing.Reasoners, "reasoner-2")
	require.NotContains(t, existing.Skills, "skill-1")
	require.Contains(t, existing.Skills, "skill-2")
	require.Equal(t, []string{"approved"}, existing.Skills["skill-2"].Tags)
}

func TestDIDServiceBackfillExistingNodes(t *testing.T) {
	t.Run("disabled did system is a no-op", func(t *testing.T) {
		provider, ctx := setupTestStorage(t)
		registry := NewDIDRegistryWithStorage(provider)
		require.NoError(t, registry.Initialize())

		keystoreDir := filepath.Join(t.TempDir(), "keys")
		ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
		require.NoError(t, err)

		service := NewDIDService(&config.DIDConfig{
			Enabled:  false,
			Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"},
		}, ks, registry)

		require.NoError(t, service.BackfillExistingNodes(ctx, provider))
	})

	t.Run("backfills only missing nodes", func(t *testing.T) {
		service, _, provider, ctx, _ := setupDIDTestEnvironment(t)

		existingNode := &types.AgentNode{
			ID:       "existing-node",
			BaseURL:  "http://existing.example",
			Reasoners: []types.ReasonerDefinition{{ID: "existing-reasoner"}},
			Skills:   []types.SkillDefinition{{ID: "existing-skill"}},
		}
		missingNode := &types.AgentNode{
			ID:       "missing-node",
			BaseURL:  "http://missing.example",
			Reasoners: []types.ReasonerDefinition{{ID: "missing-reasoner"}},
			Skills:   []types.SkillDefinition{{ID: "missing-skill"}},
		}

		require.NoError(t, provider.RegisterAgent(ctx, existingNode))
		require.NoError(t, provider.RegisterAgent(ctx, missingNode))

		resp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
			AgentNodeID: existingNode.ID,
			Reasoners:   existingNode.Reasoners,
			Skills:      existingNode.Skills,
		})
		require.NoError(t, err)
		require.True(t, resp.Success)

		before, err := service.ListAllAgentDIDs()
		require.NoError(t, err)
		require.Len(t, before, 1)

		require.NoError(t, service.BackfillExistingNodes(context.Background(), provider))

		after, err := service.ListAllAgentDIDs()
		require.NoError(t, err)
		require.Len(t, after, 2)

		backfilled, err := service.GetExistingAgentDID(missingNode.ID)
		require.NoError(t, err)
		require.Equal(t, missingNode.ID, backfilled.AgentNodeID)
		require.Contains(t, backfilled.Reasoners, "missing-reasoner")
		require.Contains(t, backfilled.Skills, "missing-skill")
	})
}
