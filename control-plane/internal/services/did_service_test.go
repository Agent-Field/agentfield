package services

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/your-org/haxen/control-plane/internal/config"
	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func setupDIDTestEnvironment(t *testing.T) (*DIDService, *DIDRegistry, storage.StorageProvider, context.Context, string) {
	t.Helper()

	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}

	service := NewDIDService(cfg, ks, registry)

	haxenID := "haxen-test"
	require.NoError(t, service.Initialize(haxenID))

	return service, registry, provider, ctx, haxenID
}

func TestDIDServiceRegisterAgentAndResolve(t *testing.T) {
	service, registry, provider, ctx, haxenID := setupDIDTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-alpha",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{{ID: "skill.fn", Tags: []string{"analysis"}}},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)
	require.NotEmpty(t, resp.IdentityPackage.AgentDID.DID)
	require.Contains(t, resp.IdentityPackage.ReasonerDIDs, "reasoner.fn")
	require.Contains(t, resp.IdentityPackage.SkillDIDs, "skill.fn")

	storedRegistry, err := registry.GetRegistry(haxenID)
	require.NoError(t, err)
	require.NotNil(t, storedRegistry)
	require.Contains(t, storedRegistry.AgentNodes, "agent-alpha")

	agents, err := provider.ListAgentDIDs(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, agents)

	agentIdentity := resp.IdentityPackage.AgentDID
	resolved, err := service.ResolveDID(agentIdentity.DID)
	require.NoError(t, err)
	require.Equal(t, agentIdentity.DID, resolved.DID)

	reasonerIdentity := resp.IdentityPackage.ReasonerDIDs["reasoner.fn"]
	resolvedReasoner, err := service.ResolveDID(reasonerIdentity.DID)
	require.NoError(t, err)
	require.Equal(t, reasonerIdentity.DID, resolvedReasoner.DID)

	skillIdentity := resp.IdentityPackage.SkillDIDs["skill.fn"]
	resolvedSkill, err := service.ResolveDID(skillIdentity.DID)
	require.NoError(t, err)
	require.Equal(t, skillIdentity.DID, resolvedSkill.DID)
}

func TestDIDServiceValidateRegistryFailure(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	err = service.validateHaxenServerRegistry()
	require.Error(t, err)

	haxenID := "haxen-validate"
	require.NoError(t, service.Initialize(haxenID))
	require.NoError(t, service.validateHaxenServerRegistry())

	stored, err := registry.GetRegistry(haxenID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	require.False(t, stored.CreatedAt.IsZero())
	require.False(t, stored.LastKeyRotation.IsZero())
	_ = ctx
}
