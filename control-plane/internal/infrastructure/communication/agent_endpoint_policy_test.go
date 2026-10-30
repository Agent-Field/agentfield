package communication

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// C5: status (also the health-monitor poll) and shutdown land on fixed paths
// under the base prefix, with no query or fragment.
func TestAgentEndpointStatusShutdownPaths(t *testing.T) {
	paths := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths <- r.URL.Path
		require.Empty(t, r.URL.RawQuery)
		if _, err := w.Write([]byte(`{"status":"running","node_id":"agent-test"}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	ctx := context.Background()
	store := setupTestStorage(t, ctx)
	id := registerAgent(t, ctx, store, server.URL+"/prefix/")
	client := NewHTTPAgentClient(store, time.Second)
	_, err := client.GetAgentStatus(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "/prefix/status", <-paths)
	_, err = client.ShutdownAgent(ctx, id, true, 1)
	require.NoError(t, err)
	require.Equal(t, "/prefix/shutdown", <-paths)
}

// Review finding 3: status/shutdown still require base_url and must not send
// their lifecycle payloads to an invocation-only node's execution endpoint.
func TestReviewInvocationOnlyStatusShutdown(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
	}))
	defer server.Close()
	ctx := context.Background()
	store := setupTestStorage(t, ctx)
	invocation := server.URL + "/invoke"
	require.NoError(t, store.RegisterAgent(ctx, &types.AgentNode{ID: "invocation-only", DeploymentType: "serverless", InvocationURL: &invocation, RegisteredAt: time.Now(), LastHeartbeat: time.Now()}))
	client := NewHTTPAgentClient(store, time.Second)
	_, err := client.GetAgentStatus(ctx, "invocation-only")
	require.Error(t, err)
	_, err = client.ShutdownAgent(ctx, "invocation-only", true, 1)
	require.Error(t, err)
	require.Zero(t, hits.Load())
}
