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
