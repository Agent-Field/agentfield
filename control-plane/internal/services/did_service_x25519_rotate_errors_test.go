package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// TestRotateAgentX25519Key_EmptyDID rejects an empty DID up front.
func TestRotateAgentX25519Key_EmptyDID(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	_, _, err := service.RotateAgentX25519Key("")
	require.Error(t, err)
	require.Contains(t, err.Error(), "did is required")
}

// TestRotateAgentX25519Key_NotFound rejects a DID that is not in the registry.
func TestRotateAgentX25519Key_NotFound(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	_, _, err := service.RotateAgentX25519Key("did:key:zNotARegisteredDID")
	require.Error(t, err)
	require.Contains(t, err.Error(), "not found")
}

// TestRotateAgentX25519Key_RootDIDRejected rejects rotation of the control-plane
// root DID — it carries no rotatable keyAgreement key.
func TestRotateAgentX25519Key_RootDIDRejected(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	stored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	require.NotEmpty(t, stored.RootDID)

	_, _, err = service.RotateAgentX25519Key(stored.RootDID)
	require.Error(t, err)
	require.Contains(t, err.Error(), "root DID")
}

// TestRotateAgentX25519Key_ReasonerDIDRejected rejects rotation of a reasoner
// DID with a precise, reasoner-specific error.
func TestRotateAgentX25519Key_ReasonerDIDRejected(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	resp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-reasoner-reject",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)
	require.True(t, resp.Success)

	reasonerDID := resp.IdentityPackage.ReasonerDIDs["reasoner.fn"].DID
	require.NotEmpty(t, reasonerDID)

	_, _, err = service.RotateAgentX25519Key(reasonerDID)
	require.Error(t, err)
	require.Contains(t, err.Error(), "reasoner DID")
	require.Contains(t, err.Error(), "agent-node DIDs")
}

// TestRotateAgentX25519Key_SkillDIDRejected rejects rotation of a skill DID with
// a precise, skill-specific error.
func TestRotateAgentX25519Key_SkillDIDRejected(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	resp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-skill-reject",
		Reasoners:   []types.ReasonerDefinition{},
		Skills:      []types.SkillDefinition{{ID: "skill.fn"}},
	})
	require.NoError(t, err)
	require.True(t, resp.Success)

	skillDID := resp.IdentityPackage.SkillDIDs["skill.fn"].DID
	require.NotEmpty(t, skillDID)

	_, _, err = service.RotateAgentX25519Key(skillDID)
	require.Error(t, err)
	require.Contains(t, err.Error(), "skill DID")
	require.Contains(t, err.Error(), "agent-node DIDs")
}

// TestRotateAgentX25519Key_RetiresPriorKey covers the rotation invariant that a
// rotated key cannot be re-derived at the prior epoch: after rotating to epoch
// N+1, the private key derived at epoch N+1 differs from the one at epoch N, so
// a payload encrypted to the epoch-N public key is no longer decryptable by the
// re-derived (current) private key.
func TestRotateAgentX25519Key_RetiresPriorKey(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	resp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-retire",
		Reasoners:   []types.ReasonerDefinition{},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)
	require.True(t, resp.Success)
	agentDID := resp.IdentityPackage.AgentDID.DID

	stored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	info := stored.AgentNodes["agent-retire"]

	// Private scalar at the original epoch (0).
	epoch0Priv, err := service.deriveX25519PrivateKeyAtEpoch(stored.MasterSeed, info.DerivationPath, 0)
	require.NoError(t, err)

	// Rotate to epoch 1.
	_, newEpoch, err := service.RotateAgentX25519Key(agentDID)
	require.NoError(t, err)
	require.Equal(t, 1, newEpoch)

	// Private scalar at the new epoch differs from epoch 0 — the old key is
	// retired and cannot be reproduced at the new epoch.
	epoch1Priv, err := service.deriveX25519PrivateKeyAtEpoch(stored.MasterSeed, info.DerivationPath, 1)
	require.NoError(t, err)
	require.NotEqual(t, epoch0Priv.Bytes(), epoch1Priv.Bytes(),
		"rotated private key must differ from the retired epoch-0 key")
	require.NotEqual(t, epoch0Priv.PublicKey().Bytes(), epoch1Priv.PublicKey().Bytes(),
		"rotated public key must differ from the retired epoch-0 key")
}
