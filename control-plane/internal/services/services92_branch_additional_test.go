package services

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type statusManagerListErrStorage struct {
	storage.StorageProvider
	listErr error
}

func (s *statusManagerListErrStorage) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	return nil, s.listErr
}

type vcStorageUnsupportedProviderStub struct {
	storage.StorageProvider
	listResp []*types.ExecutionVCInfo
	listErr  error
}

func (s *vcStorageUnsupportedProviderStub) ListExecutionVCs(context.Context, types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.listResp, nil
}

func TestUIServiceGetReconciledNodeStatusFallbackBranches(t *testing.T) {
	tests := []struct {
		name          string
		node          *types.AgentNode
		agentService  *uiAgentServiceStub
		wantLifecycle types.AgentLifecycleStatus
		wantHealth    types.HealthStatus
	}{
		{
			name:          "agent service marks node offline",
			node:          &types.AgentNode{ID: "node-off", LifecycleStatus: types.AgentStatusReady, HealthStatus: types.HealthStatusActive},
			agentService:  &uiAgentServiceStub{statusByName: map[string]*domain.AgentStatus{"node-off": {IsRunning: false}}},
			wantLifecycle: types.AgentStatusOffline,
			wantHealth:    types.HealthStatusInactive,
		},
		{
			name:          "agent service preserves degraded lifecycle",
			node:          &types.AgentNode{ID: "node-degraded", LifecycleStatus: types.AgentStatusDegraded, HealthStatus: types.HealthStatusInactive},
			agentService:  &uiAgentServiceStub{statusByName: map[string]*domain.AgentStatus{"node-degraded": {IsRunning: true}}},
			wantLifecycle: types.AgentStatusDegraded,
			wantHealth:    types.HealthStatusActive,
		},
		{
			name:          "fallback normalizes inactive ready node",
			node:          &types.AgentNode{ID: "node-inactive", LifecycleStatus: types.AgentStatusReady, HealthStatus: types.HealthStatusInactive},
			wantLifecycle: types.AgentStatusOffline,
			wantHealth:    types.HealthStatusInactive,
		},
		{
			name:          "fallback normalizes active offline node",
			node:          &types.AgentNode{ID: "node-active", LifecycleStatus: types.AgentStatusOffline, HealthStatus: types.HealthStatusActive},
			wantLifecycle: types.AgentStatusReady,
			wantHealth:    types.HealthStatusActive,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := &UIService{}
			if tt.agentService != nil {
				service.agentService = tt.agentService
			}
			lifecycle, health := service.getReconciledNodeStatus(tt.node.ID, tt.node)
			require.Equal(t, tt.wantLifecycle, lifecycle)
			require.Equal(t, tt.wantHealth, health)
		})
	}
}

func TestStatusManagerReconciliationBranchCoverage(t *testing.T) {
	t.Run("list agents error is ignored", func(t *testing.T) {
		sm := NewStatusManager(&statusManagerListErrStorage{listErr: errors.New("list failed")}, StatusManagerConfig{}, nil, nil)
		sm.performReconciliation()
	})

	t.Run("reconcile transitions and no-op branch", func(t *testing.T) {
		provider, ctx := setupStatusManagerStorage(t)
		registerTestAgent(t, provider, ctx, "node-ready-offline")
		registerTestAgent(t, provider, ctx, "node-noop")

		setAgentLegacyStatus(t, provider, ctx, "node-ready-offline", "1.0.0", types.HealthStatusActive, types.AgentStatusOffline, time.Now())
		setAgentLegacyStatus(t, provider, ctx, "node-noop", "1.0.0", types.HealthStatusActive, types.AgentStatusReady, time.Now())

		sm := NewStatusManager(provider, StatusManagerConfig{
			HeartbeatStaleThreshold: time.Minute,
			MaxTransitionTime:       time.Minute,
		}, nil, nil)

		readyOffline, err := provider.GetAgent(ctx, "node-ready-offline")
		require.NoError(t, err)
		require.NoError(t, sm.reconcileAgentStatus(ctx, readyOffline))

		updated, err := provider.GetAgent(ctx, "node-ready-offline")
		require.NoError(t, err)
		require.Equal(t, types.HealthStatusActive, updated.HealthStatus)
		require.Equal(t, types.AgentStatusReady, updated.LifecycleStatus)

		noop, err := provider.GetAgent(ctx, "node-noop")
		require.NoError(t, err)
		require.NoError(t, sm.reconcileAgentStatus(ctx, noop))

		unchanged, err := provider.GetAgent(ctx, "node-noop")
		require.NoError(t, err)
		require.Equal(t, types.HealthStatusActive, unchanged.HealthStatus)
		require.Equal(t, types.AgentStatusReady, unchanged.LifecycleStatus)
	})
}

func TestObservabilityForwarderTransformExecutionEventPayloadBranches(t *testing.T) {
	forwarder := NewObservabilityForwarder(newMockObservabilityStore(), ObservabilityForwarderConfig{}).(*observabilityForwarder)
	now := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name        string
		data        interface{}
		wantPayload bool
	}{
		{name: "non map payload stored under payload key", data: []string{"a", "b"}, wantPayload: true},
		{name: "nil payload omitted", data: nil, wantPayload: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := forwarder.transformExecutionEvent(events.ExecutionEvent{
				Type:        events.ExecutionStarted,
				ExecutionID: "exec-1",
				WorkflowID:  "wf-1",
				AgentNodeID: "node-1",
				Status:      "running",
				Timestamp:   now,
				Data:        tt.data,
			})

			data := event.Data.(map[string]interface{})
			require.Equal(t, now.Format(time.RFC3339), event.Timestamp)
			_, hasPayload := data["payload"]
			require.Equal(t, tt.wantPayload, hasPayload)
		})
	}
}

func TestKeystoreServiceAdditionalErrorBranches(t *testing.T) {
	t.Run("new keystore fails when path is a file", func(t *testing.T) {
		baseDir := t.TempDir()
		filePath := filepath.Join(baseDir, "not-a-directory")
		require.NoError(t, os.WriteFile(filePath, []byte("x"), 0600))

		_, err := NewKeystoreService(&config.KeystoreConfig{Path: filePath, Type: "local"})
		require.ErrorContains(t, err, "failed to create keystore directory")
	})

	t.Run("decrypt short ciphertext and disabled backup", func(t *testing.T) {
		svc, err := NewKeystoreService(&config.KeystoreConfig{Path: t.TempDir(), Type: "local", BackupEnabled: false})
		require.NoError(t, err)

		_, err = svc.DecryptData([]byte("short"))
		require.ErrorContains(t, err, "ciphertext too short")
		require.NoError(t, svc.BackupKeys())
	})
}

func TestVCStorageUnsupportedProviderBranches(t *testing.T) {
	provider := &vcStorageUnsupportedProviderStub{
		listResp: []*types.ExecutionVCInfo{
			{
				VCID:        "vc-unsupported",
				ExecutionID: "exec-unsupported",
				WorkflowID:  "wf-unsupported",
				SessionID:   "session-unsupported",
				Status:      string(types.ExecutionStatusSucceeded),
				CreatedAt:   time.Now().UTC(),
			},
		},
	}
	vcStorage := NewVCStorageWithStorage(provider)

	_, _, err := vcStorage.getFullVCFromDatabase("vc-unsupported")
	require.ErrorContains(t, err, "unsupported storage provider")

	records, err := vcStorage.loadExecutionVCsFromDatabaseWithFilters(types.VCFilters{})
	require.NoError(t, err)
	require.Empty(t, records)

	provider.listErr = errors.New("list failed")
	_, err = vcStorage.loadExecutionVCsFromDatabaseWithFilters(types.VCFilters{})
	require.ErrorContains(t, err, "failed to list execution VCs from database")
}
