package packages

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/stretchr/testify/require"
)

// C7/C7b/C10: local readiness/identity/capability/shutdown probes share the
// agent policy while retaining their client-side connection-closing wrapper.
func TestAgentEndpointLifecycleClientPolicy(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path == "/same" {
			http.Redirect(w, r, "/health", 302)
			return
		}
		if r.URL.Path == "/cross" {
			http.Redirect(w, r, "http://other.test/health", 302)
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer server.Close()
	client := NewNodeHTTPClient(time.Second)
	resp, err := client.Get(server.URL + "/same")
	require.NoError(t, err)
	resp.Body.Close()
	require.EqualValues(t, 2, hits.Load())
	_, err = client.Get(server.URL + "/cross")
	require.ErrorContains(t, err, "cross-origin")
	require.EqualValues(t, 3, hits.Load())
	defer agenturl.SetNetworkHooksForTesting(nil, func(context.Context, string, string) (net.Conn, error) {
		t.Fatal("metadata dial was attempted")
		return nil, nil
	})()
	_, err = client.Get("http://169.254.169.254/health")
	require.ErrorContains(t, err, "link-local")
	require.EqualValues(t, 3, hits.Load())
}
