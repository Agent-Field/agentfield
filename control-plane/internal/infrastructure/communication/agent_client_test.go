package communication

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestStorage(t *testing.T, ctx context.Context) storage.StorageProvider {
	t.Helper()

	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "test.db"),
			KVStorePath:  filepath.Join(tempDir, "test.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := provider.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5 support")
		}
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	return provider
}

func registerAgent(t *testing.T, ctx context.Context, provider storage.StorageProvider, baseURL string) string {
	t.Helper()

	agent := &types.AgentNode{
		ID:              "agent-test",
		TeamID:          "team-1",
		BaseURL:         baseURL,
		Version:         "1.0.0",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   time.Now(),
		RegisteredAt:    time.Now(),
	}

	require.NoError(t, provider.RegisterAgent(ctx, agent))
	return agent.ID
}

type flakyTransport struct {
	attempts int32
}

func (ft *flakyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	attempt := atomic.AddInt32(&ft.attempts, 1)
	if attempt == 1 {
		return nil, errors.New("connection reset by peer")
	}

	recorder := httptest.NewRecorder()
	recorder.Code = http.StatusOK
	recorder.Header().Set("Content-Type", "application/json")
	recorder.Body.WriteString(`{
		"status":"running",
		"uptime":"1s",
		"uptime_seconds":1,
		"pid":123,
		"version":"1.0.0",
		"node_id":"agent-test",
		"last_activity":"2024-01-01T00:00:00Z",
		"resources":{}
	}`)
	return recorder.Result(), nil
}

func TestHTTPAgentClient_GetAgentStatusRetriesNetworkErrors(t *testing.T) {
	ctx := context.Background()

	provider := setupTestStorage(t, ctx)
	agentID := registerAgent(t, ctx, provider, "http://agent.local")
	client := NewHTTPAgentClient(provider, 0)

	flaky := &flakyTransport{}
	client.httpClient.Transport = flaky
	client.httpClient.Timeout = 0

	resp, err := client.GetAgentStatus(ctx, agentID)
	require.NoError(t, err)
	assert.Equal(t, "running", resp.Status)
	assert.Equal(t, int32(2), atomic.LoadInt32(&flaky.attempts), "client should retry once after transient network failure")
}

type storageOverride struct {
	storage.StorageProvider
	override func(ctx context.Context, id string) (*types.AgentNode, error)
}

func (s *storageOverride) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.override != nil {
		return s.override(ctx, id)
	}
	return s.StorageProvider.GetAgent(ctx, id)
}

func TestHTTPAgentClient_GetAgentStatusHandlesMissingAgents(t *testing.T) {
	ctx := context.Background()

	provider := setupTestStorage(t, ctx)
	agentID := registerAgent(t, ctx, provider, "http://agent.local")

	override := &storageOverride{
		StorageProvider: provider,
		override: func(ctx context.Context, id string) (*types.AgentNode, error) {
			return nil, nil
		},
	}

	client := NewHTTPAgentClient(override, time.Second)

	_, err := client.GetAgentStatus(ctx, agentID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestHTTPAgentClient_GetAgentStatusRejectsMismatchedNodeID(t *testing.T) {
	ctx := context.Background()

	// Fake agent that returns a different node_id than requested
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"status":"running",
			"uptime":"1s",
			"uptime_seconds":1,
			"pid":123,
			"version":"1.0.0",
			"node_id":"other-agent",
			"last_activity":"2024-01-01T00:00:00Z",
			"resources":{}
		}`))
	}))
	defer server.Close()

	provider := setupTestStorage(t, ctx)
	agentID := registerAgent(t, ctx, provider, server.URL)
	client := NewHTTPAgentClient(provider, time.Second)

	_, err := client.GetAgentStatus(ctx, agentID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent ID mismatch")
}
