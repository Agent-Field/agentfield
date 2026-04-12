package services

import (
	"encoding/json"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDIDServiceKeyGenerationHelpers(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	storedRegistry, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)

	derivationPath := "m/44'/123'/0'/0'"
	did, privateJWK, publicJWK, err := service.generateDIDWithKeys(storedRegistry.MasterSeed, derivationPath)
	require.NoError(t, err)
	require.NotEmpty(t, did)

	derivedDID, err := service.generateDIDFromSeed(storedRegistry.MasterSeed, derivationPath)
	require.NoError(t, err)
	require.Equal(t, did, derivedDID)

	regeneratedPrivateJWK, err := service.regeneratePrivateKeyJWK(storedRegistry.MasterSeed, derivationPath)
	require.NoError(t, err)
	require.JSONEq(t, privateJWK, regeneratedPrivateJWK)

	regeneratedPublicJWK, err := service.regeneratePublicKeyJWK(storedRegistry.MasterSeed, derivationPath)
	require.NoError(t, err)
	require.JSONEq(t, publicJWK, regeneratedPublicJWK)

	var privateKeyFields map[string]string
	require.NoError(t, json.Unmarshal([]byte(privateJWK), &privateKeyFields))
	require.Equal(t, "OKP", privateKeyFields["kty"])
	require.Equal(t, "Ed25519", privateKeyFields["crv"])
	require.NotEmpty(t, privateKeyFields["x"])
	require.NotEmpty(t, privateKeyFields["d"])

	var publicKeyFields map[string]string
	require.NoError(t, json.Unmarshal([]byte(publicJWK), &publicKeyFields))
	require.Equal(t, privateKeyFields["x"], publicKeyFields["x"])
}

func TestDIDServiceResolveAgentIDAndDerivationPaths(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-paths",
		Reasoners: []types.ReasonerDefinition{
			{ID: "reasoner-1"},
		},
		Skills: []types.SkillDefinition{
			{ID: "skill-1"},
		},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)

	require.Equal(t, "agent-paths", service.ResolveAgentIDByDID(resp.IdentityPackage.AgentDID.DID))
	require.Empty(t, service.ResolveAgentIDByDID("did:key:missing"))

	reasonerPath := service.generateReasonerPath("agent-paths", "reasoner-2")
	require.NotEmpty(t, reasonerPath)
	require.Contains(t, reasonerPath, "/0'/1'")

	skillPath := service.generateSkillPath("agent-paths", "skill-2")
	require.NotEmpty(t, skillPath)
	require.Contains(t, skillPath, "/1'/1'")
}
