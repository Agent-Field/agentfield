package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestVCChainVerifiableFlag(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-chain-verifiable",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)
	agentDID := regResp.IdentityPackage.AgentDID.DID
	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	_, err = vcService.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID:  "exec-verifiable",
		WorkflowID:   "workflow-verifiable",
		SessionID:    "session-verifiable",
		CallerDID:    callerDID,
		AgentNodeDID: agentDID,
		Timestamp:    time.Now(),
	}, []byte(`{"input":1}`), []byte(`{"output":1}`), "succeeded", nil, 10)
	require.NoError(t, err)

	chain, err := vcService.GetWorkflowVCChain("workflow-verifiable")
	require.NoError(t, err)
	require.True(t, chain.Verifiable)
	require.Empty(t, chain.UnresolvedDIDs)

	const unknownDID = "did:web:example.com:agents:gone"
	require.NoError(t, provider.StoreExecutionVC(ctx,
		"vc-unverifiable", "exec-unverifiable", "workflow-unverifiable", "session-unverifiable",
		agentDID, unknownDID, callerDID,
		"hash-in", "hash-out", "succeeded",
		[]byte(`{"id":"vc-unverifiable","issuer":"`+agentDID+`"}`), "sig", "", 24,
	))

	chain, err = vcService.GetWorkflowVCChain("workflow-unverifiable")
	require.NoError(t, err)
	require.False(t, chain.Verifiable)
	require.Equal(t, []string{unknownDID}, chain.UnresolvedDIDs)
}

func TestC8F8RequiredIssuersAndOptionalParticipants(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	valid := register(t, svc, "node", nil, nil).IdentityPackage.AgentDID.DID
	vc := NewVCService(svc.config, svc, env.provider)
	documentFor := func(issuer string) json.RawMessage {
		raw, err := json.Marshal(map[string]string{"issuer": issuer})
		require.NoError(t, err)
		return raw
	}
	for _, bad := range []string{"", " ", "did:key:", "did:", "garbage", "did::missing", "did:web:invalid host"} {
		t.Run("execution/"+bad, func(t *testing.T) {
			bundle, unresolved := vc.collectDIDResolutionBundle([]types.ExecutionVC{{IssuerDID: bad, VCDocument: documentFor(bad)}}, &types.WorkflowVC{IssuerDID: valid})
			require.Equal(t, []string{bad}, unresolved)
			require.Contains(t, bundle, valid)
		})
		t.Run("workflow/"+bad, func(t *testing.T) {
			_, unresolved := vc.collectDIDResolutionBundle([]types.ExecutionVC{{IssuerDID: valid, VCDocument: documentFor(valid)}}, &types.WorkflowVC{IssuerDID: bad})
			require.Equal(t, []string{bad}, unresolved)
		})
		// Even an invalid first execution issuer must not prevent exporting
		// a wrapper; its required unresolved DID marks the chain false.
		workflowID := "wf-" + bad
		require.NoError(t, env.provider.StoreExecutionVC(context.Background(), "vc-"+bad, "exec-"+bad, workflowID, "session", bad, "", "", "in", "out", "succeeded", documentFor(bad), "signature", "", 0))
		chain, err := vc.GetWorkflowVCChain(workflowID)
		require.NoError(t, err)
		require.False(t, chain.Verifiable)
		require.Equal(t, []string{bad}, chain.UnresolvedDIDs)
	}
	_, unresolved := vc.collectDIDResolutionBundle([]types.ExecutionVC{{IssuerDID: valid, VCDocument: documentFor(valid)}}, &types.WorkflowVC{IssuerDID: valid})
	require.Empty(t, unresolved, "absent optional caller and target are not required DIDs")
}
