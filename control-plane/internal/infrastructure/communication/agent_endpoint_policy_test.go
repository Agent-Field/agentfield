package communication

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// C5: status (also the health-monitor poll) and shutdown land on fixed paths
// under the base prefix, with no query or fragment.
func TestAgentEndpointStatusShutdownPaths(t *testing.T) {
	paths := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths <- r.URL.Path
		require.Empty(t, r.URL.RawQuery)
		w.Write([]byte(`{"status":"running","node_id":"agent-test"}`))
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
