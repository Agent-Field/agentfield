package did

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDIDIdentityJSONRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		original DIDIdentity
	}{
		{
			name: "full agent DID",
			original: DIDIdentity{
				DID:            "did:key:z6Mktest123",
				PrivateKeyJWK:  `{"kty":"OKP","crv":"Ed25519","x":"abc","d":"xyz"}`,
				PublicKeyJWK:   `{"kty":"OKP","crv":"Ed25519","x":"abc"}`,
				DerivationPath: "m/0'/0'",
				ComponentType:  "agent",
				FunctionName:   "",
			},
		},
		{
			name: "reasoner DID with function name",
			original: DIDIdentity{
				DID:            "did:key:z6Mkreasoner456",
				PublicKeyJWK:   `{"kty":"OKP","crv":"Ed25519","x":"def"}`,
				DerivationPath: "m/0'/1'",
				ComponentType:  "reasoner",
				FunctionName:   "my-function",
			},
		},
		{
			name: "skill DID",
			original: DIDIdentity{
				DID:           "did:key:z6Mkskill789",
				PublicKeyJWK:  `{"kty":"OKP","crv":"Ed25519","x":"ghi"}`,
				ComponentType: "skill",
				FunctionName:  "my-skill",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.original)
			require.NoError(t, err)

			var decoded DIDIdentity
			err = json.Unmarshal(data, &decoded)
			require.NoError(t, err)

			assert.Equal(t, tt.original.DID, decoded.DID)
			assert.Equal(t, tt.original.PublicKeyJWK, decoded.PublicKeyJWK)
			assert.Equal(t, tt.original.DerivationPath, decoded.DerivationPath)
			assert.Equal(t, tt.original.ComponentType, decoded.ComponentType)
			assert.Equal(t, tt.original.FunctionName, decoded.FunctionName)
		})
	}
}

func TestDIDIdentityPrivateKeyOmitEmpty(t *testing.T) {
	// PrivateKeyJWK is omitempty — should not appear when empty
	d := DIDIdentity{
		DID:           "did:key:z6Mk123",
		PublicKeyJWK:  `{"kty":"OKP"}`,
		ComponentType: "agent",
		// PrivateKeyJWK intentionally left empty
	}

	data, err := json.Marshal(d)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasPrivate := m["private_key_jwk"]
	assert.False(t, hasPrivate, "private_key_jwk should be omitted when empty")
}

func TestDIDIdentityPackageJSONRoundTrip(t *testing.T) {
	original := DIDIdentityPackage{
		AgentDID: DIDIdentity{
			DID:           "did:key:agent",
			PublicKeyJWK:  `{"kty":"OKP","x":"agent-pub"}`,
			ComponentType: "agent",
		},
		ReasonerDIDs: map[string]DIDIdentity{
			"r1": {
				DID:           "did:key:reasoner1",
				PublicKeyJWK:  `{"kty":"OKP","x":"r1-pub"}`,
				ComponentType: "reasoner",
				FunctionName:  "r1",
			},
		},
		SkillDIDs: map[string]DIDIdentity{
			"s1": {
				DID:           "did:key:skill1",
				PublicKeyJWK:  `{"kty":"OKP","x":"s1-pub"}`,
				ComponentType: "skill",
				FunctionName:  "s1",
			},
		},
		AgentFieldServerID: "server-id-abc",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded DIDIdentityPackage
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.AgentDID.DID, decoded.AgentDID.DID)
	assert.Equal(t, original.AgentFieldServerID, decoded.AgentFieldServerID)

	require.Contains(t, decoded.ReasonerDIDs, "r1")
	assert.Equal(t, original.ReasonerDIDs["r1"].DID, decoded.ReasonerDIDs["r1"].DID)
	assert.Equal(t, original.ReasonerDIDs["r1"].FunctionName, decoded.ReasonerDIDs["r1"].FunctionName)

	require.Contains(t, decoded.SkillDIDs, "s1")
	assert.Equal(t, original.SkillDIDs["s1"].DID, decoded.SkillDIDs["s1"].DID)
}

func TestRegistrationRequestJSONRoundTrip(t *testing.T) {
	original := RegistrationRequest{
		AgentNodeID: "node-1",
		Reasoners: []FunctionDef{
			{ID: "r1"},
			{ID: "r2"},
		},
		Skills: []FunctionDef{
			{ID: "s1"},
		},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded RegistrationRequest
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.AgentNodeID, decoded.AgentNodeID)
	require.Len(t, decoded.Reasoners, 2)
	assert.Equal(t, "r1", decoded.Reasoners[0].ID)
	assert.Equal(t, "r2", decoded.Reasoners[1].ID)
	require.Len(t, decoded.Skills, 1)
	assert.Equal(t, "s1", decoded.Skills[0].ID)
}

func TestFunctionDefJSONRoundTrip(t *testing.T) {
	original := FunctionDef{ID: "my-function"}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded FunctionDef
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ID, decoded.ID)
}

func TestRegistrationResponseJSONRoundTrip(t *testing.T) {
	original := RegistrationResponse{
		Success: true,
		IdentityPackage: DIDIdentityPackage{
			AgentDID: DIDIdentity{
				DID:           "did:key:agent",
				PublicKeyJWK:  `{"kty":"OKP","x":"pub"}`,
				ComponentType: "agent",
			},
			ReasonerDIDs:       map[string]DIDIdentity{},
			SkillDIDs:          map[string]DIDIdentity{},
			AgentFieldServerID: "server-1",
		},
		Error: "",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded RegistrationResponse
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.Success, decoded.Success)
	assert.Equal(t, original.Error, decoded.Error)
	assert.Equal(t, original.IdentityPackage.AgentDID.DID, decoded.IdentityPackage.AgentDID.DID)
	assert.Equal(t, original.IdentityPackage.AgentFieldServerID, decoded.IdentityPackage.AgentFieldServerID)
}

func TestRegistrationResponseErrorOmitEmpty(t *testing.T) {
	// Error is omitempty — should not appear when empty
	r := RegistrationResponse{
		Success: true,
		IdentityPackage: DIDIdentityPackage{
			AgentDID:     DIDIdentity{DID: "did:key:x", PublicKeyJWK: "{}", ComponentType: "agent"},
			ReasonerDIDs: map[string]DIDIdentity{},
			SkillDIDs:    map[string]DIDIdentity{},
		},
	}

	data, err := json.Marshal(r)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasError := m["error"]
	assert.False(t, hasError, "error field should be omitted when empty")
}

func TestExecutionContextJSONRoundTrip(t *testing.T) {
	original := ExecutionContext{
		ExecutionID:  "exec-1",
		WorkflowID:   "wf-1",
		SessionID:    "session-1",
		CallerDID:    "did:key:caller",
		TargetDID:    "did:key:target",
		AgentNodeDID: "did:key:node",
		Timestamp:    "2026-04-05T10:00:00Z",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded ExecutionContext
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original, decoded)
}

func TestExecutionContextOptionalFieldsOmitEmpty(t *testing.T) {
	// workflow_id, session_id, caller_did, target_did, agent_node_did, timestamp are omitempty
	ec := ExecutionContext{
		ExecutionID: "exec-1",
	}

	data, err := json.Marshal(ec)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	for _, field := range []string{"workflow_id", "session_id", "caller_did", "target_did", "agent_node_did", "timestamp"} {
		_, has := m[field]
		assert.False(t, has, "field %q should be omitted when empty", field)
	}
}

func TestVCGenerationRequestJSONRoundTrip(t *testing.T) {
	original := VCGenerationRequest{
		ExecutionContext: ExecutionContext{
			ExecutionID: "exec-1",
			WorkflowID:  "wf-1",
		},
		InputData:    `{"input":"data"}`,
		OutputData:   `{"output":"result"}`,
		Status:       "succeeded",
		ErrorMessage: "",
		DurationMS:   1500,
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded VCGenerationRequest
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ExecutionContext.ExecutionID, decoded.ExecutionContext.ExecutionID)
	assert.Equal(t, original.InputData, decoded.InputData)
	assert.Equal(t, original.OutputData, decoded.OutputData)
	assert.Equal(t, original.Status, decoded.Status)
	assert.Equal(t, original.DurationMS, decoded.DurationMS)
}

func TestVCGenerationRequestErrorMessageOmitEmpty(t *testing.T) {
	req := VCGenerationRequest{
		ExecutionContext: ExecutionContext{ExecutionID: "exec-1"},
		InputData:       "{}",
		OutputData:      "{}",
		Status:          "succeeded",
		// ErrorMessage intentionally empty
	}

	data, err := json.Marshal(req)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, has := m["error_message"]
	assert.False(t, has, "error_message should be omitted when empty")
}

func TestExecutionVCJSONRoundTrip(t *testing.T) {
	original := ExecutionVC{
		VCID:        "vc-1",
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		SessionID:   "session-1",
		IssuerDID:   "did:key:issuer",
		TargetDID:   "did:key:target",
		CallerDID:   "did:key:caller",
		VCDocument:  map[string]any{"@context": []string{"https://www.w3.org/2018/credentials/v1"}, "type": []string{"VerifiableCredential"}},
		Signature:   "sig-abc",
		InputHash:   "input-hash-abc",
		OutputHash:  "output-hash-xyz",
		Status:      "succeeded",
		CreatedAt:   "2026-04-05T10:00:00Z",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded ExecutionVC
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.VCID, decoded.VCID)
	assert.Equal(t, original.ExecutionID, decoded.ExecutionID)
	assert.Equal(t, original.WorkflowID, decoded.WorkflowID)
	assert.Equal(t, original.SessionID, decoded.SessionID)
	assert.Equal(t, original.IssuerDID, decoded.IssuerDID)
	assert.Equal(t, original.TargetDID, decoded.TargetDID)
	assert.Equal(t, original.CallerDID, decoded.CallerDID)
	assert.Equal(t, original.Signature, decoded.Signature)
	assert.Equal(t, original.InputHash, decoded.InputHash)
	assert.Equal(t, original.OutputHash, decoded.OutputHash)
	assert.Equal(t, original.Status, decoded.Status)
	assert.Equal(t, original.CreatedAt, decoded.CreatedAt)
}

func TestWorkflowVCChainJSONRoundTrip(t *testing.T) {
	original := WorkflowVCChain{
		WorkflowID: "wf-1",
		ExecutionVCs: []ExecutionVC{
			{
				VCID:        "vc-1",
				ExecutionID: "exec-1",
				WorkflowID:  "wf-1",
				IssuerDID:   "did:key:issuer",
				TargetDID:   "did:key:target",
				Signature:   "sig-1",
				InputHash:   "hash-in-1",
				OutputHash:  "hash-out-1",
				Status:      "succeeded",
				CreatedAt:   "2026-04-05T10:00:00Z",
			},
		},
		WorkflowVC: map[string]any{"type": "WorkflowVC"},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded WorkflowVCChain
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.WorkflowID, decoded.WorkflowID)
	require.Len(t, decoded.ExecutionVCs, 1)
	assert.Equal(t, "vc-1", decoded.ExecutionVCs[0].VCID)
	assert.Equal(t, "exec-1", decoded.ExecutionVCs[0].ExecutionID)
	assert.NotNil(t, decoded.WorkflowVC)
}

func TestWorkflowVCChainNilWorkflowVC(t *testing.T) {
	chain := WorkflowVCChain{
		WorkflowID:   "wf-1",
		ExecutionVCs: []ExecutionVC{},
		// WorkflowVC intentionally nil — omitempty
	}

	data, err := json.Marshal(chain)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, has := m["workflow_vc"]
	assert.False(t, has, "workflow_vc should be omitted when nil")
}
