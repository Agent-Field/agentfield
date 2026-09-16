package services

import (
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
		[]byte(`{"id":"vc-unverifiable"}`), "sig", "", 24,
	))

	chain, err = vcService.GetWorkflowVCChain("workflow-unverifiable")
	require.NoError(t, err)
	require.False(t, chain.Verifiable)
	require.Equal(t, []string{unknownDID}, chain.UnresolvedDIDs)
}
