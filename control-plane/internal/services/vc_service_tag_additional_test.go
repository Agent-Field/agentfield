package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestVCServiceAgentTagVCSigningVerificationAndListing(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	issuerDID, err := didService.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	vcDocument := &types.AgentTagVCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential", "AgentTagCredential"},
		ID:           "urn:agentfield:vc:tag-1",
		Issuer:       issuerDID,
		IssuanceDate: time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: types.AgentTagVCCredentialSubject{
			ID:      "did:key:agent-1",
			AgentID: "agent-1",
			Permissions: types.AgentTagVCPermissions{
				Tags:           []string{"finance"},
				AllowedCallees: []string{"*"},
			},
		},
	}

	proof, err := vcService.SignAgentTagVC(vcDocument)
	require.NoError(t, err)
	vcDocument.Proof = proof

	valid, err := vcService.VerifyAgentTagVCSignature(vcDocument)
	require.NoError(t, err)
	require.True(t, valid)

	vcDocumentJSON, err := json.Marshal(vcDocument)
	require.NoError(t, err)
	require.NoError(t, provider.StoreAgentTagVC(ctx, "agent-1", vcDocument.CredentialSubject.ID, vcDocument.ID, string(vcDocumentJSON), proof.ProofValue, time.Now().UTC(), nil))

	records, err := vcService.ListAgentTagVCs()
	require.NoError(t, err)
	require.Len(t, records, 1)
	require.Equal(t, "agent-1", records[0].AgentID)

	storageRecords, err := vcService.vcStorage.ListAgentTagVCs(context.Background())
	require.NoError(t, err)
	require.Len(t, storageRecords, 1)
}

func TestVCServiceVerifyAgentTagVCSignatureErrors(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)
	issuerDID, err := didService.GetControlPlaneIssuerDID()
	require.NoError(t, err)

	tests := []struct {
		name    string
		vc      *types.AgentTagVCDocument
		wantErr string
	}{
		{
			name: "missing proof",
			vc: &types.AgentTagVCDocument{
				Issuer: issuerDID,
			},
			wantErr: "VC has no valid signature",
		},
		{
			name: "invalid signature encoding",
			vc: &types.AgentTagVCDocument{
				Context:      []string{"https://www.w3.org/2018/credentials/v1"},
				Type:         []string{"VerifiableCredential"},
				ID:           "urn:agentfield:vc:bad-signature",
				Issuer:       issuerDID,
				IssuanceDate: time.Now().UTC().Format(time.RFC3339),
				CredentialSubject: types.AgentTagVCCredentialSubject{
					ID:      "did:key:agent-1",
					AgentID: "agent-1",
				},
				Proof: &types.VCProof{
					Type:       "Ed25519Signature2020",
					ProofValue: "%%%not-base64%%%",
				},
			},
			wantErr: "failed to decode signature",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, err := vcService.VerifyAgentTagVCSignature(tt.vc)
			require.ErrorContains(t, err, tt.wantErr)
			require.False(t, valid)
		})
	}
}
