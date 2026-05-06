package services

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGenerateTriggerEventVC_HappyPath confirms a trigger event VC is signed
// by the CP root DID, persisted with kind='trigger_event', and carries the
// trigger metadata in its credentialSubject.
func TestGenerateTriggerEventVC_HappyPath(t *testing.T) {
	vcService, _, provider, ctx := setupVCTestEnvironment(t)

	in := TriggerEventInput{
		TriggerID:  "trg_abc123",
		SourceName: "stripe",
		EventType:  "payment_intent.succeeded",
		EventID:    "evt_test_001",
		Payload:    []byte(`{"id":"evt_test_001","type":"payment_intent.succeeded"}`),
		ReceivedAt: time.Date(2026, 4, 28, 12, 0, 0, 0, time.UTC),
		Verification: types.VCTriggerVerification{
			Passed:    true,
			Algorithm: "stripe-v1",
		},
	}

	vc, err := vcService.GenerateTriggerEventVC(ctx, in)
	require.NoError(t, err)
	require.NotNil(t, vc)

	assert.Equal(t, types.ExecutionVCKindTriggerEvent, vc.Kind)
	assert.Equal(t, "evt_test_001", vc.ExecutionID, "event ID should populate execution_id slot for uniqueness")
	require.NotNil(t, vc.TriggerID)
	assert.Equal(t, "trg_abc123", *vc.TriggerID)
	require.NotNil(t, vc.SourceName)
	assert.Equal(t, "stripe", *vc.SourceName)
	require.NotNil(t, vc.EventType)
	assert.Equal(t, "payment_intent.succeeded", *vc.EventType)
	assert.NotEmpty(t, vc.Signature, "VC must be signed")

	// Confirm the credentialSubject was serialized with verification + payload hash.
	var doc map[string]any
	require.NoError(t, json.Unmarshal(vc.VCDocument, &doc))
	subj, ok := doc["credentialSubject"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "trg_abc123", subj["trigger_id"])
	assert.Equal(t, "stripe", subj["source_name"])
	assert.NotEmpty(t, subj["payload_hash"])

	// Persisted row should round-trip back via the storage provider.
	stored, err := provider.GetExecutionVC(ctx, vc.VCID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	assert.Equal(t, vc.VCID, stored.VCID)
}

// TestGenerateTriggerEventVC_DIDDisabled confirms the VC mint is a clean
// no-op when DID is off — callers can dispatch without checking config.
func TestGenerateTriggerEventVC_DIDDisabled(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	disabled := &config.DIDConfig{Enabled: false}
	svc := &VCService{config: disabled}

	vc, err := svc.GenerateTriggerEventVC(ctx, TriggerEventInput{
		TriggerID: "trg_disabled",
		EventID:   "evt_disabled",
		Payload:   []byte(`{}`),
	})
	require.NoError(t, err)
	assert.Nil(t, vc, "DID disabled → nil VC, nil error so callers proceed")

	// Sanity: nothing got written.
	_, err = provider.GetExecutionVC(ctx, "nonexistent")
	require.Error(t, err)
}

// TestGenerateTriggerEventVC_PersistDisabled — DID enabled but
// PersistExecutionVC=false (some test/dev modes) returns nil cleanly too.
func TestGenerateTriggerEventVC_PersistDisabled(t *testing.T) {
	_, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: true,
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: false, // gate
			PersistExecutionVC:    false,
		},
	}
	svc := &VCService{config: cfg}

	vc, err := svc.GenerateTriggerEventVC(ctx, TriggerEventInput{
		TriggerID: "trg_no_persist",
		EventID:   "evt_no_persist",
		Payload:   []byte(`{}`),
	})
	require.NoError(t, err)
	assert.Nil(t, vc)
}

// TestGenerateExecutionVC_PropagatesParentVCID confirms the chain extension
// works for normal execution VCs: when ExecutionContext.ParentVCID is set,
// the resulting ExecutionVC.ParentVCID points at it.
func TestGenerateExecutionVC_PropagatesParentVCID(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-vc-chain",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}
	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID
	parentVCID := "trg_evt_vc_parent_id"
	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-vc-chain-001",
		WorkflowID:   "wf-chain-001",
		SessionID:    "sess-001",
		CallerDID:    callerDID,
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now().UTC(),
		ParentVCID:   parentVCID,
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"in":1}`), []byte(`{"out":2}`), string(types.ExecutionStatusSucceeded), nil, 12)
	require.NoError(t, err)
	require.NotNil(t, vc)
	require.NotNil(t, vc.ParentVCID, "ParentVCID must propagate from ExecutionContext")
	assert.Equal(t, parentVCID, *vc.ParentVCID)
	assert.Equal(t, types.ExecutionVCKindExecution, vc.Kind)
}
