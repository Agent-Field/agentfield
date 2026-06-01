package services

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

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

	agentfieldID := "agentfield-test"
	require.NoError(t, service.Initialize(agentfieldID))

	return service, registry, provider, ctx, agentfieldID
}

func TestDIDServiceRegisterAgentAndResolve(t *testing.T) {
	service, registry, provider, ctx, agentfieldID := setupDIDTestEnvironment(t)

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

	storedRegistry, err := registry.GetRegistry(agentfieldID)
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

	err = service.validateAgentFieldServerRegistry()
	require.Error(t, err)

	agentfieldID := "agentfield-validate"
	require.NoError(t, service.Initialize(agentfieldID))
	require.NoError(t, service.validateAgentFieldServerRegistry())

	stored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	require.False(t, stored.CreatedAt.IsZero())
	require.False(t, stored.LastKeyRotation.IsZero())
	_ = ctx
}

func TestDIDService_ResolveDID_RootDID(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	// Get the root DID from registry
	storedRegistry, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	require.NotNil(t, storedRegistry)
	require.NotEmpty(t, storedRegistry.RootDID)

	// Resolve root DID
	resolved, err := service.ResolveDID(storedRegistry.RootDID)
	require.NoError(t, err)
	require.NotNil(t, resolved)
	require.Equal(t, storedRegistry.RootDID, resolved.DID)
	require.Equal(t, "agentfield_server", resolved.ComponentType)
	require.Equal(t, "m/44'/0'", resolved.DerivationPath)
	require.NotEmpty(t, resolved.PrivateKeyJWK)
	require.NotEmpty(t, resolved.PublicKeyJWK)
}

func TestDIDService_ResolveDID_NotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	resolved, err := service.ResolveDID("did:key:invalid")
	require.Error(t, err)
	require.Nil(t, resolved)
	require.Contains(t, err.Error(), "DID not found")
}

func TestDIDService_ResolveDID_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: false, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	resolved, err := service.ResolveDID("did:key:test")
	require.Error(t, err)
	require.Nil(t, resolved)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestDIDService_RegisterAgent_ExistingAgent_NoChanges(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent first time
	req1 := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-existing",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	resp1, err := service.RegisterAgent(req1)
	require.NoError(t, err)
	require.True(t, resp1.Success)

	// Register same agent again with same components
	req2 := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-existing",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	resp2, err := service.RegisterAgent(req2)
	require.NoError(t, err)
	require.True(t, resp2.Success)
	require.Contains(t, resp2.Message, "No changes detected")
	require.Equal(t, resp1.IdentityPackage.AgentDID.DID, resp2.IdentityPackage.AgentDID.DID)
}

func TestDIDService_PartialRegisterAgent_NewComponents(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent with initial components
	req1 := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-partial",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	resp1, err := service.RegisterAgent(req1)
	require.NoError(t, err)
	require.True(t, resp1.Success)

	// Partial registration with new components
	partialReq := &types.PartialDIDRegistrationRequest{
		AgentNodeID:    "agent-partial",
		NewReasonerIDs: []string{"reasoner2"},
		NewSkillIDs:    []string{"skill2"},
		AllReasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}, {ID: "reasoner2"}},
		AllSkills:      []types.SkillDefinition{{ID: "skill1"}, {ID: "skill2"}},
	}

	resp2, err := service.PartialRegisterAgent(partialReq)
	require.NoError(t, err)
	require.True(t, resp2.Success)
	require.Contains(t, resp2.Message, "Partial registration successful")
	require.Len(t, resp2.IdentityPackage.ReasonerDIDs, 1) // Only new ones
	require.Len(t, resp2.IdentityPackage.SkillDIDs, 1)     // Only new ones
	require.Contains(t, resp2.IdentityPackage.ReasonerDIDs, "reasoner2")
	require.Contains(t, resp2.IdentityPackage.SkillDIDs, "skill2")
}

func TestDIDService_PartialRegisterAgent_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: false, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	partialReq := &types.PartialDIDRegistrationRequest{
		AgentNodeID:    "agent-test",
		NewReasonerIDs: []string{"reasoner1"},
		AllReasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		AllSkills:      []types.SkillDefinition{},
	}

	resp, err := service.PartialRegisterAgent(partialReq)
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "DID system is disabled")
	_ = ctx
}

func TestDIDService_DeregisterComponents_Success(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent with multiple components
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-deregister",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}, {ID: "reasoner2"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}, {ID: "skill2"}},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)

	// Deregister some components
	deregReq := &types.ComponentDeregistrationRequest{
		AgentNodeID:         "agent-deregister",
		ReasonerIDsToRemove: []string{"reasoner1"},
		SkillIDsToRemove:    []string{"skill1"},
	}

	deregResp, err := service.DeregisterComponents(deregReq)
	require.NoError(t, err)
	require.True(t, deregResp.Success)
	require.Equal(t, 2, deregResp.RemovedCount)

	// Verify components were removed
	existingAgent, err := service.GetExistingAgentDID("agent-deregister")
	require.NoError(t, err)
	require.NotContains(t, existingAgent.Reasoners, "reasoner1")
	require.Contains(t, existingAgent.Reasoners, "reasoner2")
	require.NotContains(t, existingAgent.Skills, "skill1")
	require.Contains(t, existingAgent.Skills, "skill2")
}

func TestDIDService_DeregisterComponents_NotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-deregister-notfound",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	_, err := service.RegisterAgent(req)
	require.NoError(t, err)

	// Try to deregister non-existent components
	deregReq := &types.ComponentDeregistrationRequest{
		AgentNodeID:         "agent-deregister-notfound",
		ReasonerIDsToRemove: []string{"nonexistent-reasoner"},
		SkillIDsToRemove:    []string{"nonexistent-skill"},
	}

	deregResp, err := service.DeregisterComponents(deregReq)
	require.NoError(t, err)
	require.True(t, deregResp.Success)
	require.Equal(t, 0, deregResp.RemovedCount) // No components removed
}

func TestDIDService_DeregisterComponents_AgentNotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	deregReq := &types.ComponentDeregistrationRequest{
		AgentNodeID:         "nonexistent-agent",
		ReasonerIDsToRemove: []string{"reasoner1"},
		SkillIDsToRemove:    []string{"skill1"},
	}

	deregResp, err := service.DeregisterComponents(deregReq)
	require.NoError(t, err)
	require.False(t, deregResp.Success)
	require.Contains(t, deregResp.Error, "agent not found")
}

func TestDIDService_PerformDifferentialAnalysis_NoChanges(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-diff",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	_, err := service.RegisterAgent(req)
	require.NoError(t, err)

	// Perform differential analysis with same components
	result, err := service.PerformDifferentialAnalysis("agent-diff", []string{"reasoner1"}, []string{"skill1"})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.RequiresUpdate)
	require.Empty(t, result.NewReasonerIDs)
	require.Empty(t, result.RemovedReasonerIDs)
	require.Empty(t, result.NewSkillIDs)
	require.Empty(t, result.RemovedSkillIDs)
	require.Len(t, result.UpdatedReasonerIDs, 1)
	require.Len(t, result.UpdatedSkillIDs, 1)
}

func TestDIDService_PerformDifferentialAnalysis_NewComponents(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-diff-new",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	_, err := service.RegisterAgent(req)
	require.NoError(t, err)

	// Perform differential analysis with new components
	result, err := service.PerformDifferentialAnalysis("agent-diff-new", []string{"reasoner1", "reasoner2"}, []string{"skill1", "skill2"})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.RequiresUpdate)
	require.Len(t, result.NewReasonerIDs, 1)
	require.Contains(t, result.NewReasonerIDs, "reasoner2")
	require.Len(t, result.NewSkillIDs, 1)
	require.Contains(t, result.NewSkillIDs, "skill2")
	require.Empty(t, result.RemovedReasonerIDs)
	require.Empty(t, result.RemovedSkillIDs)
}

func TestDIDService_PerformDifferentialAnalysis_RemovedComponents(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent with multiple components
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-diff-removed",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}, {ID: "reasoner2"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}, {ID: "skill2"}},
	}

	_, err := service.RegisterAgent(req)
	require.NoError(t, err)

	// Perform differential analysis with fewer components
	result, err := service.PerformDifferentialAnalysis("agent-diff-removed", []string{"reasoner1"}, []string{"skill1"})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.RequiresUpdate)
	require.Empty(t, result.NewReasonerIDs)
	require.Empty(t, result.NewSkillIDs)
	require.Len(t, result.RemovedReasonerIDs, 1)
	require.Contains(t, result.RemovedReasonerIDs, "reasoner2")
	require.Len(t, result.RemovedSkillIDs, 1)
	require.Contains(t, result.RemovedSkillIDs, "skill2")
}

func TestDIDService_PerformDifferentialAnalysis_AgentNotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	result, err := service.PerformDifferentialAnalysis("nonexistent-agent", []string{"reasoner1"}, []string{"skill1"})
	require.Error(t, err)
	require.Nil(t, result)
	require.Contains(t, err.Error(), "failed to get existing agent")
}

func TestDIDService_GetExistingAgentDID_Success(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-get-existing",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	regResp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	// Get existing agent
	existingAgent, err := service.GetExistingAgentDID("agent-get-existing")
	require.NoError(t, err)
	require.NotNil(t, existingAgent)
	require.Equal(t, "agent-get-existing", existingAgent.AgentNodeID)
	require.Equal(t, regResp.IdentityPackage.AgentDID.DID, existingAgent.DID)
	require.Len(t, existingAgent.Reasoners, 1)
	require.Len(t, existingAgent.Skills, 1)
}

func TestDIDService_GetExistingAgentDID_NotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	existingAgent, err := service.GetExistingAgentDID("nonexistent-agent")
	require.Error(t, err)
	require.Nil(t, existingAgent)
	require.Contains(t, err.Error(), "agent not found")
}

func TestDIDService_GetExistingAgentDID_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: false, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	existingAgent, err := service.GetExistingAgentDID("agent-test")
	require.Error(t, err)
	require.Nil(t, existingAgent)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestDIDService_ListAllAgentDIDs_Success(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register multiple agents
	req1 := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-list-1",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{},
	}

	_, err := service.RegisterAgent(req1)
	require.NoError(t, err)

	req2 := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-list-2",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{},
	}

	_, err = service.RegisterAgent(req2)
	require.NoError(t, err)

	// List all agent DIDs
	agentDIDs, err := service.ListAllAgentDIDs()
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(agentDIDs), 2)
}

func TestDIDService_ListAllAgentDIDs_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: false, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	agentDIDs, err := service.ListAllAgentDIDs()
	require.Error(t, err)
	require.Nil(t, agentDIDs)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestDIDService_RegisterAgent_EmptyReasonerID(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent with empty reasoner ID (should be skipped)
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-empty-reasoner",
		Reasoners:   []types.ReasonerDefinition{{ID: ""}, {ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)
	require.Len(t, resp.IdentityPackage.ReasonerDIDs, 1) // Only non-empty reasoner
	require.Contains(t, resp.IdentityPackage.ReasonerDIDs, "reasoner1")
}

func TestDIDService_RegisterAgent_EmptySkillID(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	// Register agent with empty skill ID (should be skipped)
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-empty-skill",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: ""}, {ID: "skill1"}},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)
	require.Len(t, resp.IdentityPackage.SkillDIDs, 1) // Only non-empty skill
	require.Contains(t, resp.IdentityPackage.SkillDIDs, "skill1")
}

func TestDIDService_RegisterAgent_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: false, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-disabled",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.False(t, resp.Success)
	require.Contains(t, resp.Error, "DID system is disabled")
	_ = ctx
}

func TestDIDService_GetAgentFieldServerID(t *testing.T) {
	service, _, _, _, agentfieldID := setupDIDTestEnvironment(t)

	serverID, err := service.GetAgentFieldServerID()
	require.NoError(t, err)
	require.Equal(t, agentfieldID, serverID)
}

func TestDIDService_GetAgentFieldServerID_NotInitialized(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	cfg := &config.DIDConfig{Enabled: true, Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"}}
	service := NewDIDService(cfg, ks, registry)

	serverID, err := service.GetAgentFieldServerID()
	require.Error(t, err)
	require.Empty(t, serverID)
	require.Contains(t, err.Error(), "not initialized")
	_ = ctx
}
