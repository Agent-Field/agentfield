package services

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func setAgentLegacyStatus(t *testing.T, provider interface {
	UpdateAgentHealth(context.Context, string, types.HealthStatus) error
	UpdateAgentLifecycleStatus(context.Context, string, types.AgentLifecycleStatus) error
	UpdateAgentHeartbeat(context.Context, string, string, time.Time) error
}, ctx context.Context, nodeID, version string, health types.HealthStatus, lifecycle types.AgentLifecycleStatus, heartbeat time.Time) {
	t.Helper()
	require.NoError(t, provider.UpdateAgentHealth(ctx, nodeID, health))
	require.NoError(t, provider.UpdateAgentLifecycleStatus(ctx, nodeID, lifecycle))
	require.NoError(t, provider.UpdateAgentHeartbeat(ctx, nodeID, version, heartbeat))
}

func TestStatusManagerPendingApprovalPreservesAdminLifecycle(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-pending-approval")
	setAgentLegacyStatus(t, provider, ctx, "node-pending-approval", "1.0.0", types.HealthStatusInactive, types.AgentStatusPendingApproval, time.Now().Add(-time.Minute))

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	sm.statusCache["node-pending-approval"] = &cachedAgentStatus{
		Status: &types.AgentStatus{
			State:           types.AgentStateInactive,
			HealthScore:     10,
			HealthStatus:    types.HealthStatusInactive,
			LifecycleStatus: types.AgentStatusPendingApproval,
		},
		Timestamp: time.Now(),
	}

	activeState := types.AgentStateActive
	healthScore := 77
	require.NoError(t, sm.UpdateAgentStatus(ctx, "node-pending-approval", &types.AgentStatusUpdate{
		State:       &activeState,
		HealthScore: &healthScore,
		Source:      types.StatusSourceHeartbeat,
		Reason:      "should be ignored",
	}))

	agent, err := provider.GetAgent(ctx, "node-pending-approval")
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusPendingApproval, agent.LifecycleStatus)
	require.Equal(t, types.HealthStatusInactive, agent.HealthStatus)
	require.Equal(t, 77, sm.statusCache["node-pending-approval"].Status.HealthScore)
}

func TestStatusManagerUpdateAgentStatusEnforcesLifecycleConsistency(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-enforce-lifecycle")

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	activeState := types.AgentStateActive
	offlineLifecycle := types.AgentStatusOffline
	require.NoError(t, sm.UpdateAgentStatus(ctx, "node-enforce-lifecycle", &types.AgentStatusUpdate{
		State:           &activeState,
		LifecycleStatus: &offlineLifecycle,
		Source:          types.StatusSourceHeartbeat,
		Reason:          "activate agent",
	}))

	agent, err := provider.GetAgent(ctx, "node-enforce-lifecycle")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusActive, agent.HealthStatus)
	require.Equal(t, types.AgentStatusReady, agent.LifecycleStatus)
}

func TestStatusManagerPerformReconciliationUpdatesStaleAgents(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-stale-active")
	registerTestAgent(t, provider, ctx, "node-stale-starting")
	registerTestAgent(t, provider, ctx, "node-fresh")

	setAgentLegacyStatus(t, provider, ctx, "node-stale-active", "1.0.0", types.HealthStatusActive, types.AgentStatusReady, time.Now().Add(-2*time.Minute))
	setAgentLegacyStatus(t, provider, ctx, "node-stale-starting", "1.0.0", types.HealthStatusActive, types.AgentStatusStarting, time.Now().Add(-2*time.Minute))
	setAgentLegacyStatus(t, provider, ctx, "node-fresh", "1.0.0", types.HealthStatusActive, types.AgentStatusReady, time.Now())

	sm := NewStatusManager(provider, StatusManagerConfig{
		HeartbeatStaleThreshold: 30 * time.Second,
		MaxTransitionTime:       45 * time.Second,
	}, nil, nil)

	sm.performReconciliation()

	staleActive, err := provider.GetAgent(ctx, "node-stale-active")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, staleActive.HealthStatus)
	require.Equal(t, types.AgentStatusOffline, staleActive.LifecycleStatus)

	staleStarting, err := provider.GetAgent(ctx, "node-stale-starting")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, staleStarting.HealthStatus)
	require.Equal(t, types.AgentStatusOffline, staleStarting.LifecycleStatus)

	fresh, err := provider.GetAgent(ctx, "node-fresh")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusActive, fresh.HealthStatus)
	require.Equal(t, types.AgentStatusReady, fresh.LifecycleStatus)
}

func TestVCServiceWorkflowStatusSummariesPropagatesStorageErrors(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-summary-branches",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
	})
	require.NoError(t, err)

	now := time.Now().UTC()
	for _, tc := range []struct {
		vcID       string
		workflowID string
		status     string
		createdAt  time.Time
	}{
		{vcID: "vc-verified-1", workflowID: "workflow-verified", status: string(types.ExecutionStatusSucceeded), createdAt: now.Add(-2 * time.Minute)},
		{vcID: "vc-verified-2", workflowID: "workflow-verified", status: string(types.ExecutionStatusSucceeded), createdAt: now.Add(-1 * time.Minute)},
		{vcID: "vc-failed-1", workflowID: "workflow-failed", status: string(types.ExecutionStatusSucceeded), createdAt: now.Add(-3 * time.Minute)},
		{vcID: "vc-failed-2", workflowID: "workflow-failed", status: string(types.ExecutionStatusFailed), createdAt: now.Add(-time.Minute)},
		{vcID: "vc-pending-1", workflowID: "workflow-pending", status: string(types.ExecutionStatusRunning), createdAt: now.Add(-30 * time.Second)},
	} {
		require.NoError(t, vcService.vcStorage.StoreExecutionVC(context.Background(), &types.ExecutionVC{
			VCID:         tc.vcID,
			ExecutionID:  tc.vcID,
			WorkflowID:   tc.workflowID,
			SessionID:    "session-summary",
			IssuerDID:    regResp.IdentityPackage.AgentDID.DID,
			CallerDID:    regResp.IdentityPackage.AgentDID.DID,
			VCDocument:   json.RawMessage(`{"id":"` + tc.vcID + `"}`),
			Signature:    "sig",
			DocumentSize: 16,
			Status:       tc.status,
			CreatedAt:    tc.createdAt,
		}))
	}

	_, err = vcService.GetWorkflowVCStatusSummaries([]string{
		"workflow-verified",
		"workflow-failed",
		"workflow-pending",
		"workflow-empty",
	})
	require.ErrorContains(t, err, "failed to scan workflow VC status summary")
}

func TestVCServiceVerifyWorkflowVCComprehensiveErrorBranches(t *testing.T) {
	t.Run("disabled system", func(t *testing.T) {
		service := NewVCService(&config.DIDConfig{Enabled: false}, nil, nil)
		result, err := service.VerifyWorkflowVCComprehensive("workflow-disabled")
		require.NoError(t, err)
		require.False(t, result.Valid)
		require.Len(t, result.CriticalIssues, 1)
		require.Equal(t, "system_disabled", result.CriticalIssues[0].Type)
	})

	t.Run("workflow chain error", func(t *testing.T) {
		service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)
		result, err := service.VerifyWorkflowVCComprehensive("workflow-chain-error")
		require.NoError(t, err)
		require.False(t, result.Valid)
		require.Len(t, result.CriticalIssues, 1)
		require.Equal(t, "workflow_chain_error", result.CriticalIssues[0].Type)
	})

	t.Run("malformed execution and workflow vcs", func(t *testing.T) {
		vcService, didService, _, _ := setupVCTestEnvironment(t)

		regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
			AgentNodeID: "agent-bad-workflow",
			Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
		})
		require.NoError(t, err)

		require.NoError(t, vcService.vcStorage.StoreExecutionVC(context.Background(), &types.ExecutionVC{
			VCID:         "vc-bad-json",
			ExecutionID:  "exec-bad-json",
			WorkflowID:   "workflow-bad-json",
			SessionID:    "session-bad-json",
			IssuerDID:    regResp.IdentityPackage.AgentDID.DID,
			CallerDID:    regResp.IdentityPackage.AgentDID.DID,
			VCDocument:   json.RawMessage(`not-json`),
			Signature:    "sig",
			DocumentSize: 8,
			Status:       string(types.ExecutionStatusSucceeded),
			CreatedAt:    time.Now(),
		}))

		require.NoError(t, vcService.vcStorage.StoreWorkflowVC(context.Background(), &types.WorkflowVC{
			WorkflowID:     "workflow-bad-json",
			SessionID:      "session-bad-json",
			ComponentVCs:   []string{"vc-bad-json"},
			WorkflowVCID:   "workflow-vc-bad-json",
			Status:         string(types.ExecutionStatusSucceeded),
			StartTime:      time.Now().Add(-time.Second),
			EndTime:        func() *time.Time { t := time.Now(); return &t }(),
			TotalSteps:     1,
			CompletedSteps: 1,
			VCDocument:     json.RawMessage(`not-json`),
			Signature:      "sig",
			IssuerDID:      regResp.IdentityPackage.AgentDID.DID,
			DocumentSize:   8,
		}))

		result, err := vcService.VerifyWorkflowVCComprehensive("workflow-bad-json")
		require.NoError(t, err)
		require.False(t, result.Valid)

		issueTypes := make([]string, 0, len(result.CriticalIssues))
		for _, issue := range result.CriticalIssues {
			issueTypes = append(issueTypes, issue.Type)
		}
		require.Contains(t, issueTypes, "parse_error")
		require.Less(t, result.OverallScore, 100.0)
	})
}

func TestVCServiceVerificationHelpersAdditionalBranches(t *testing.T) {
	service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)

	tests := []struct {
		name    string
		doc     *types.VCDocument
		wantErr string
	}{
		{
			name:    "missing type",
			doc:     &types.VCDocument{Context: []string{"ctx"}},
			wantErr: "missing type",
		},
		{
			name:    "missing id",
			doc:     &types.VCDocument{Context: []string{"ctx"}, Type: []string{"type"}},
			wantErr: "missing id",
		},
		{
			name:    "missing issuer",
			doc:     &types.VCDocument{Context: []string{"ctx"}, Type: []string{"type"}, ID: "id"},
			wantErr: "missing issuer",
		},
		{
			name:    "missing issuance date",
			doc:     &types.VCDocument{Context: []string{"ctx"}, Type: []string{"type"}, ID: "id", Issuer: "did:key:test"},
			wantErr: "missing issuanceDate",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.ErrorContains(t, service.validateVCStructure(tt.doc), tt.wantErr)
		})
	}

	t.Run("verify workflow vc signature invalid jwk", func(t *testing.T) {
		validDoc := &types.WorkflowVCDocument{
			Context: []string{"https://www.w3.org/2018/credentials/v1"},
			Type:    []string{"VerifiableCredential", "AgentFieldWorkflowCredential"},
			Proof:   types.VCProof{ProofValue: "not-base64"},
		}

		_, err := service.verifyWorkflowVCSignature(validDoc, &types.DIDIdentity{PublicKeyJWK: `{"kty":"OKP"}`})
		require.ErrorContains(t, err, "missing 'x' parameter")

		_, err = service.verifyWorkflowVCSignature(validDoc, &types.DIDIdentity{PublicKeyJWK: `{"x":"AQI"}`})
		require.ErrorContains(t, err, "invalid public key length")
	})
}

func TestPayloadStoreAdditionalErrorBranches(t *testing.T) {
	store := NewFilePayloadStore(t.TempDir())

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := store.SaveFromReader(cancelledCtx, bytes.NewReader([]byte("data")))
	require.ErrorIs(t, err, context.Canceled)

	_, err = store.Open(cancelledCtx, payloadURIPrefix+"abc")
	require.ErrorIs(t, err, context.Canceled)

	err = store.Remove(cancelledCtx, payloadURIPrefix+"abc")
	require.ErrorIs(t, err, context.Canceled)

	err = store.Remove(context.Background(), payloadURIPrefix)
	require.ErrorContains(t, err, "invalid payload URI")
}

func TestDIDRegistryAdditionalErrorBranches(t *testing.T) {
	registry := NewDIDRegistryWithStorage(nil)
	require.ErrorContains(t, registry.Initialize(), "storage provider not available")

	_, err := registry.FindDIDByComponent("missing-registry", "agent", "agent-1")
	require.ErrorContains(t, err, "registry not found")

	registry.registries["agentfield-test"] = &types.DIDRegistry{
		AgentFieldServerID: "agentfield-test",
		AgentNodes:         map[string]types.AgentDIDInfo{},
	}

	require.ErrorContains(t, registry.UpdateAgentStatus("agentfield-test", "missing-agent", types.AgentDIDStatusActive), "agent not found")
	_, err = registry.FindDIDByComponent("agentfield-test", "agent", "missing-agent")
	require.ErrorContains(t, err, "DID not found")
}
