package services

import (
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDIDServiceAdditionalHelperBranches(t *testing.T) {
	t.Run("initialize preserves existing registry and get registry disabled", func(t *testing.T) {
		service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

		before, err := registry.GetRegistry(agentfieldID)
		require.NoError(t, err)
		require.NotNil(t, before)

		require.NoError(t, service.Initialize(agentfieldID))

		after, err := registry.GetRegistry(agentfieldID)
		require.NoError(t, err)
		require.Equal(t, before.RootDID, after.RootDID)
		require.Equal(t, before.TotalDIDs, after.TotalDIDs)

		disabled := NewDIDService(&config.DIDConfig{Enabled: false}, nil, registry)
		require.NoError(t, disabled.Initialize("ignored"))
		_, err = disabled.GetRegistry(agentfieldID)
		require.ErrorContains(t, err, "DID system is disabled")
	})

	t.Run("path generators and identity package cover success and fallback branches", func(t *testing.T) {
		service, _, _, _, _ := setupDIDTestEnvironment(t)

		resp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
			AgentNodeID: "agent-identity-package",
			Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
			Skills:      []types.SkillDefinition{{ID: "skill-1"}},
		})
		require.NoError(t, err)
		require.True(t, resp.Success)

		existing, err := service.GetExistingAgentDID("agent-identity-package")
		require.NoError(t, err)

		pkg := service.buildExistingIdentityPackage(existing)
		require.Equal(t, resp.IdentityPackage.AgentDID.DID, pkg.AgentDID.DID)
		require.NotEmpty(t, pkg.AgentDID.PrivateKeyJWK)
		require.NotEmpty(t, pkg.ReasonerDIDs["reasoner-1"].PrivateKeyJWK)
		require.NotEmpty(t, pkg.SkillDIDs["skill-1"].PrivateKeyJWK)

		require.NotNil(t, service.findReasonerByID([]types.ReasonerDefinition{{ID: "reasoner-1"}}, "reasoner-1"))
		require.Nil(t, service.findReasonerByID([]types.ReasonerDefinition{{ID: "reasoner-1"}}, "missing"))
		require.NotNil(t, service.findSkillByID([]types.SkillDefinition{{ID: "skill-1"}}, "skill-1"))
		require.Nil(t, service.findSkillByID([]types.SkillDefinition{{ID: "skill-1"}}, "missing"))

		uninitialized := NewDIDService(&config.DIDConfig{Enabled: true}, nil, service.registry)
		require.Empty(t, uninitialized.generateReasonerPath("agent-identity-package", "reasoner-2"))
		require.Empty(t, uninitialized.generateSkillPath("agent-identity-package", "skill-2"))
	})
}

func TestDIDServiceInitializeExistingRegistryAndKeystoreBranches(t *testing.T) {
	provider, _ := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	service := NewDIDService(&config.DIDConfig{Enabled: true}, ks, registry)
	require.NoError(t, service.Initialize("agentfield-extra"))
	require.NoError(t, service.Initialize("agentfield-extra"))

	loaded, err := service.GetRegistry("agentfield-extra")
	require.NoError(t, err)
	require.NotNil(t, loaded)
}
