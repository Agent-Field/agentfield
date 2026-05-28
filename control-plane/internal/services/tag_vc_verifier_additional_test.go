package services

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestTagVCVerifierWithVCService(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)
	issuerDID, err := didService.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	newSignedVC := func(agentID string) string {
		t.Helper()
		vcDocument := &types.AgentTagVCDocument{
			Context:      []string{"https://www.w3.org/2018/credentials/v1"},
			Type:         []string{"VerifiableCredential", "AgentTagCredential"},
			ID:           "urn:agentfield:vc:" + agentID,
			Issuer:       issuerDID,
			IssuanceDate: time.Now().UTC().Format(time.RFC3339),
			CredentialSubject: types.AgentTagVCCredentialSubject{
				ID:      "did:key:" + agentID,
				AgentID: agentID,
				Permissions: types.AgentTagVCPermissions{
					Tags:           []string{"finance"},
					AllowedCallees: []string{"*"},
				},
			},
		}
		proof, err := vcService.SignAgentTagVC(vcDocument)
		require.NoError(t, err)
		vcDocument.Proof = proof
		raw, err := json.Marshal(vcDocument)
		require.NoError(t, err)
		return string(raw)
	}

	verifier := NewTagVCVerifier(provider, vcService)

	require.NoError(t, provider.StoreAgentTagVC(ctx, "agent-ok", "did:key:agent-ok", "vc-ok", newSignedVC("agent-ok"), "sig", time.Now().UTC(), nil))
	verified, err := verifier.VerifyAgentTagVC(ctx, "agent-ok")
	require.NoError(t, err)
	require.Equal(t, "agent-ok", verified.CredentialSubject.AgentID)

	require.NoError(t, provider.StoreAgentTagVC(ctx, "agent-mismatch", "did:key:agent-mismatch", "vc-mismatch", newSignedVC("different-agent"), "sig", time.Now().UTC(), nil))
	_, err = verifier.VerifyAgentTagVC(ctx, "agent-mismatch")
	require.ErrorContains(t, err, "tag VC subject mismatch")
}
