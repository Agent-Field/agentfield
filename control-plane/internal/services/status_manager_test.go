package services

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/your-org/haxen/control-plane/internal/core/interfaces"
	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

type fakeAgentClient struct {
	statusResponse *interfaces.AgentStatusResponse
	err            error
	calls          int
}

func (f *fakeAgentClient) setError(err error) {
	f.err = err
}

func (f *fakeAgentClient) GetAgentStatus(ctx context.Context, nodeID string) (*interfaces.AgentStatusResponse, error) {
	f.calls++
	if f.err != nil {
		err := f.err
		f.err = nil
		return nil, err
	}
	return f.statusResponse, nil
}

func (f *fakeAgentClient) GetMCPHealth(ctx context.Context, nodeID string) (*interfaces.MCPHealthResponse, error) {
	return nil, nil
}

func (f *fakeAgentClient) RestartMCPServer(ctx context.Context, nodeID, alias string) error {
	return nil
}

func (f *fakeAgentClient) GetMCPTools(ctx context.Context, nodeID, alias string) (*interfaces.MCPToolsResponse, error) {
	return nil, nil
}

func (f *fakeAgentClient) ShutdownAgent(ctx context.Context, nodeID string, graceful bool, timeoutSeconds int) (*interfaces.AgentShutdownResponse, error) {
	return nil, nil
}

func setupStatusManagerStorage(t *testing.T) (storage.StorageProvider, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "haxen.db"),
			KVStorePath:  filepath.Join(tempDir, "haxen.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := provider.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping status manager test")
		}
		require.NoError(t, err)
	}
	t.Cleanup(func() { _ = provider.Close(ctx) })

	return provider, ctx
}

func registerTestAgent(t *testing.T, provider storage.StorageProvider, ctx context.Context, nodeID string) {
	t.Helper()

	node := &types.AgentNode{
		ID:              nodeID,
		TeamID:          "team",
		BaseURL:         "http://localhost",
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		LastHeartbeat:   time.Now().Add(-1 * time.Minute),
		Reasoners:       []types.ReasonerDefinition{},
		Skills:          []types.SkillDefinition{},
	}

	require.NoError(t, provider.RegisterAgent(ctx, node))
}

func ptrAgentState(state types.AgentState) *types.AgentState {
	return &state
}

func TestStatusManagerCachingAndFallback(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-1")

	fakeClient := &fakeAgentClient{statusResponse: &interfaces.AgentStatusResponse{Status: "running"}}
	sm := NewStatusManager(provider, StatusManagerConfig{
		ReconcileInterval: 100 * time.Millisecond,
		StatusCacheTTL:    30 * time.Second,
		MaxTransitionTime: time.Second,
	}, nil, fakeClient)

	status, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, status.State)
	require.Equal(t, 1, fakeClient.calls)

	// Subsequent call within cache window should not re-hit client even if error is configured.
	fakeClient.setError(errors.New("boom"))
	statusCached, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, statusCached.State)
	require.Equal(t, 1, fakeClient.calls)

	// After cache expiry, a new health check should occur and fall back to inactive state on failure.
	time.Sleep(1100 * time.Millisecond)
	fakeClient.setError(errors.New("still failing"))
	statusAfterError, err := sm.GetAgentStatus(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateInactive, statusAfterError.State)
	require.Equal(t, 2, fakeClient.calls)

	storedAgent, err := provider.GetAgent(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, types.HealthStatusInactive, storedAgent.HealthStatus)
}

func TestStatusManagerAllowsInactiveToActiveTransition(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-transition")

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

	update := &types.AgentStatusUpdate{
		State:  ptrAgentState(types.AgentStateActive),
		Source: types.StatusSourceHeartbeat,
		Reason: "heartbeat indicates agent active",
	}

	require.NoError(t, sm.UpdateAgentStatus(ctx, "node-transition", update))

	status, err := sm.GetAgentStatus(ctx, "node-transition")
	require.NoError(t, err)
	require.Equal(t, types.AgentStateActive, status.State)
}

func TestStatusManagerSnapshotUsesStorage(t *testing.T) {
	provider, ctx := setupStatusManagerStorage(t)
	registerTestAgent(t, provider, ctx, "node-snapshot")

	sm := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)

	snapshot, err := sm.GetAgentStatusSnapshot(ctx, "node-snapshot", nil)
	require.NoError(t, err)
	require.Equal(t, types.StatusSourceReconcile, snapshot.Source)
	require.Equal(t, types.AgentStatusOffline, snapshot.LifecycleStatus)

	// Ensure snapshot is cached and returned without additional storage lookups when provided with cached node data.
	smNoCache := NewStatusManager(provider, StatusManagerConfig{}, nil, nil)
	node := &types.AgentNode{ID: "node-snapshot", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: time.Now()}
	snapshot2, err := smNoCache.GetAgentStatusSnapshot(ctx, "node-snapshot", node)
	require.NoError(t, err)
	require.Equal(t, types.AgentStatusReady, snapshot2.LifecycleStatus)
}
