package services

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// parseJWK is a small helper that unmarshals a JWK string into a map for
// field-level assertions in the X25519 keyAgreement tests.
func parseJWK(t *testing.T, raw string) map[string]interface{} {
	t.Helper()
	require.NotEmpty(t, raw, "JWK string must not be empty")
	var jwk map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(raw), &jwk), "JWK must parse as JSON")
	return jwk
}

// TestDIDService_RegisterAgent_X25519KeyAgreement asserts that RegisterAgent
// populates a complete X25519 keyAgreement keypair on the returned identity
// package, and that the private JWK is well-formed (OKP/X25519 with a private
// scalar `d`). This is requirement (1) of the X25519 plumbing contract.
func TestDIDService_RegisterAgent_X25519KeyAgreement(t *testing.T) {
	service, _, _, _, _ := setupDIDTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-x25519",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{{ID: "skill.fn"}},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)

	agentDID := resp.IdentityPackage.AgentDID

	// Both keyAgreement JWKs must be present on the returned package.
	require.NotEmpty(t, agentDID.X25519PrivateKeyJWK, "AgentDID must carry an X25519 private JWK")
	require.NotEmpty(t, agentDID.X25519PublicKeyJWK, "AgentDID must carry an X25519 public JWK")

	// The private JWK must parse as a valid RFC 8037 X25519 OKP key with a
	// non-empty private component `d`.
	privJWK := parseJWK(t, agentDID.X25519PrivateKeyJWK)
	require.Equal(t, "OKP", privJWK["kty"], "private JWK kty must be OKP")
	require.Equal(t, "X25519", privJWK["crv"], "private JWK crv must be X25519")
	d, ok := privJWK["d"].(string)
	require.True(t, ok, "private JWK must have a string `d` component")
	require.NotEmpty(t, d, "private JWK `d` (private scalar) must be non-empty")

	// Sanity: the public JWK must also be a valid X25519 OKP key with `x`.
	pubJWK := parseJWK(t, agentDID.X25519PublicKeyJWK)
	require.Equal(t, "OKP", pubJWK["kty"])
	require.Equal(t, "X25519", pubJWK["crv"])
	x, ok := pubJWK["x"].(string)
	require.True(t, ok, "public JWK must have a string `x` component")
	require.NotEmpty(t, x, "public JWK `x` must be non-empty")
}

// TestDIDService_X25519_DeterministicAndIndependent covers requirements (3) and
// (4): deriving the X25519 keypair twice from the same master seed + path yields
// identical public keys (determinism), and the X25519 public key bytes differ
// from the Ed25519 public key bytes for the same DID (independent derivation).
func TestDIDService_X25519_DeterministicAndIndependent(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-determinism",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{},
	}

	resp, err := service.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, resp.Success)

	storedRegistry, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	require.NotNil(t, storedRegistry)

	agentInfo, ok := storedRegistry.AgentNodes["agent-determinism"]
	require.True(t, ok, "registered agent must be present in the stored registry")
	require.NotEmpty(t, agentInfo.DerivationPath)

	// (3) Determinism: derive the X25519 keypair twice and compare public keys.
	pub1, priv1, err := service.regenerateX25519KeyPairJWK(storedRegistry.MasterSeed, agentInfo.DerivationPath)
	require.NoError(t, err)
	pub2, priv2, err := service.regenerateX25519KeyPairJWK(storedRegistry.MasterSeed, agentInfo.DerivationPath)
	require.NoError(t, err)
	require.Equal(t, pub1, pub2, "X25519 public JWK must be deterministic for the same seed + path")
	require.Equal(t, priv1, priv2, "X25519 private JWK must be deterministic for the same seed + path")

	// (4) Independence: the X25519 keyAgreement public key bytes must differ from
	// the Ed25519 signing public key bytes for the same DID — they are derived
	// with distinct HKDF salts and must not collide.
	ed25519PubJWK, err := service.regeneratePublicKeyJWK(storedRegistry.MasterSeed, agentInfo.DerivationPath)
	require.NoError(t, err)

	x25519XBytes := decodeJWKX(t, pub1)
	ed25519XBytes := decodeJWKX(t, ed25519PubJWK)
	require.NotEmpty(t, x25519XBytes)
	require.NotEmpty(t, ed25519XBytes)
	require.NotEqual(t, ed25519XBytes, x25519XBytes,
		"X25519 keyAgreement public key bytes must differ from Ed25519 signing public key bytes")
}

// TestDIDService_RotateAgentX25519Key covers the rotation contract: rotating an
// agent's keyAgreement key increments the stored epoch, changes the stored public
// key, and makes ResolveDID return the NEW pub + a matching private key.
func TestDIDService_RotateAgentX25519Key(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	regResp, err := service.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-rotate",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)
	agentDID := regResp.IdentityPackage.AgentDID.DID
	require.Equal(t, 0, regResp.IdentityPackage.AgentDID.X25519Epoch, "new agents start at epoch 0")

	// Capture the pre-rotation stored public key + the resolved keypair.
	preResolve, err := service.ResolveDID(agentDID)
	require.NoError(t, err)
	require.Equal(t, 0, preResolve.X25519Epoch)
	prePubX := decodeJWKX(t, preResolve.X25519PublicKeyJWK)

	preStored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	preStoredInfo := preStored.AgentNodes["agent-rotate"]
	require.Equal(t, 0, preStoredInfo.X25519Epoch)

	// Rotate.
	newPub, newEpoch, err := service.RotateAgentX25519Key(agentDID)
	require.NoError(t, err)
	require.Equal(t, 1, newEpoch, "rotation must increment the epoch to 1")
	require.NotEmpty(t, newPub)

	// The new pub must differ from the pre-rotation pub.
	newPubX := decodeJWKX(t, newPub)
	require.NotEqual(t, prePubX, newPubX, "rotated public key bytes must differ from the original")

	// The stored registry must reflect the new epoch + new public key.
	postStored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	postStoredInfo := postStored.AgentNodes["agent-rotate"]
	require.Equal(t, 1, postStoredInfo.X25519Epoch, "stored epoch must increment")
	require.Equal(t, newPubX, decodeJWKX(t, string(postStoredInfo.X25519PublicKeyJWK)),
		"stored public key must equal the rotated public key")
	require.NotEqual(t, prePubX, decodeJWKX(t, string(postStoredInfo.X25519PublicKeyJWK)),
		"stored public key must have changed")

	// ResolveDID after rotation returns the new pub + a priv whose public
	// component matches the new pub (derive-and-compare via the `x` field).
	postResolve, err := service.ResolveDID(agentDID)
	require.NoError(t, err)
	require.Equal(t, 1, postResolve.X25519Epoch)
	require.Equal(t, newPubX, decodeJWKX(t, postResolve.X25519PublicKeyJWK),
		"resolve must return the rotated public key")
	require.Equal(t, newPubX, decodeJWKX(t, postResolve.X25519PrivateKeyJWK),
		"the resolved private key's public component must correspond to the new public key")
}

// TestDIDService_X25519_EpochDeterminismAndDivergence covers the epoch-aware
// derivation contract: deriving at the same (seed, path, epoch) twice is
// identical, while different epochs produce different keys.
func TestDIDService_X25519_EpochDeterminismAndDivergence(t *testing.T) {
	service, registry, _, _, agentfieldID := setupDIDTestEnvironment(t)

	_, err := service.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-epoch",
		Reasoners:   []types.ReasonerDefinition{},
		Skills:      []types.SkillDefinition{},
	})
	require.NoError(t, err)

	stored, err := registry.GetRegistry(agentfieldID)
	require.NoError(t, err)
	info := stored.AgentNodes["agent-epoch"]
	require.NotEmpty(t, info.DerivationPath)

	// Determinism at a fixed epoch.
	e0a, _, err := service.regenerateX25519KeyPairJWKAtEpoch(stored.MasterSeed, info.DerivationPath, 0)
	require.NoError(t, err)
	e0b, _, err := service.regenerateX25519KeyPairJWKAtEpoch(stored.MasterSeed, info.DerivationPath, 0)
	require.NoError(t, err)
	require.Equal(t, e0a, e0b, "same (seed, path, epoch) must derive identically")

	// Divergence across epochs.
	e1, _, err := service.regenerateX25519KeyPairJWKAtEpoch(stored.MasterSeed, info.DerivationPath, 1)
	require.NoError(t, err)
	e2, _, err := service.regenerateX25519KeyPairJWKAtEpoch(stored.MasterSeed, info.DerivationPath, 2)
	require.NoError(t, err)
	require.NotEqual(t, e0a, e1, "epoch 0 and epoch 1 must derive different keys")
	require.NotEqual(t, e1, e2, "epoch 1 and epoch 2 must derive different keys")

	// The epoch-less helper must equal epoch 0 (well-defined delegation).
	eLess, _, err := service.regenerateX25519KeyPairJWK(stored.MasterSeed, info.DerivationPath)
	require.NoError(t, err)
	require.Equal(t, e0a, eLess, "epoch-less helper must delegate to epoch 0")
}

// decodeJWKX extracts and base64url-decodes the `x` (public key) component of a
// JWK string, asserting it is present.
func decodeJWKX(t *testing.T, raw string) []byte {
	t.Helper()
	jwk := parseJWK(t, raw)
	x, ok := jwk["x"].(string)
	require.True(t, ok, "JWK must have a string `x` component")
	require.NotEmpty(t, x)
	decoded, err := base64.RawURLEncoding.DecodeString(x)
	require.NoError(t, err, "JWK `x` must be valid base64url")
	return decoded
}
