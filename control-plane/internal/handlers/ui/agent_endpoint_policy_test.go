package ui

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// C5/C7/C7b/C9: logs keep the prefix and legitimate query values, while stored
// malformed endpoints, DNS rebinding and cross-origin redirects relay no body.
func TestAgentEndpointLogsPolicy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if strings.HasPrefix(r.URL.Path, "/redirect/") {
			http.Redirect(w, r, "http://other.test/logs", 302)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/rebind/") {
			w.Header().Set("Connection", "close")
			http.Redirect(w, r, "/end", 302)
			return
		}
		require.Equal(t, "/prefix/agentfield/v1/logs", r.URL.Path)
		require.Equal(t, "tail_lines=2", r.URL.RawQuery)
		if _, err := w.Write([]byte("{\"message\":\"safe\"}\n")); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	var lookups atomic.Int32
	defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		if lookups.Add(1) == 1 {
			return []string{"127.0.0.1"}, nil
		}
		return []string{"169.254.169.254"}, nil
	}, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, strings.TrimPrefix(server.URL, "http://"))
	})()
	for _, tc := range []struct {
		base         string
		ok           bool
		expectedHits int32
	}{
		{server.URL + "/prefix/", true, 1},
		{server.URL + "/prefix?hidden", false, 0},
		{server.URL + "/prefix#frag", false, 0},
		{"http://169.254.169.254/prefix", false, 0},
		{server.URL + "/redirect", false, 1},
		{"http://rebound.test/rebind", false, 1},
		{"http://rebound.test/prefix", false, 0},
	} {
		t.Run(tc.base, func(t *testing.T) {
			node := &types.AgentNode{ID: "policy-node", BaseURL: tc.base}
			store := &overrideStorage{getAgentFn: func(context.Context, string) (*types.AgentNode, error) { return node, nil }}
			h := &NodeLogsProxyHandler{Storage: store, Snapshot: func() (config.NodeLogProxyConfig, string) {
				return config.NodeLogProxyConfig{ConnectTimeout: time.Second, MaxTailLines: 10}, ""
			}}
			router := gin.New()
			router.GET("/nodes/:nodeId/logs", h.ProxyNodeLogsHandler)
			before := hits.Load()
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest("GET", "/nodes/policy-node/logs?tail_lines=2", nil))
			if tc.ok {
				require.Equal(t, 200, rec.Code, rec.Body.String())
				require.Contains(t, rec.Body.String(), "safe")
			} else {
				require.GreaterOrEqual(t, rec.Code, 400, rec.Body.String())
				require.NotContains(t, rec.Body.String(), "safe")
			}
			require.Equal(t, before+tc.expectedHits, hits.Load())
		})
	}
}

// C5/C9: lifecycle advertisements retain prefixes; invalid legacy data yields
// no advertised URL and does not break rendering/listing.
func TestAgentEndpointLifecyclePaths(t *testing.T) {
	store := &overrideStorage{getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
		return &types.AgentNode{BaseURL: "http://127.0.0.1:18190/prefix/"}, nil
	}}
	h := NewLifecycleHandler(store, nil)
	got := h.buildEndpoints(context.Background(), "policy-node", 18190)
	require.Equal(t, map[string]string{"health": "http://127.0.0.1:18190/prefix/health", "reasoners": "http://127.0.0.1:18190/prefix/reasoners", "skills": "http://127.0.0.1:18190/prefix/skills"}, got)
	store.getAgentFn = func(context.Context, string) (*types.AgentNode, error) {
		return &types.AgentNode{BaseURL: "http://127.0.0.1:18190/prefix?hidden"}, nil
	}
	require.Empty(t, h.buildEndpoints(context.Background(), "policy-node", 18190))
}

// C5: with no caller query, the prefixed logs endpoint receives an empty query.
func TestAgentEndpointLogsEmptyQuery(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Equal(t, "/prefix/agentfield/v1/logs", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
		if _, err := w.Write([]byte("{}\n")); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	store := &overrideStorage{getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
		return &types.AgentNode{ID: "empty-query", BaseURL: server.URL + "/prefix/"}, nil
	}}
	h := &NodeLogsProxyHandler{Storage: store, Snapshot: func() (config.NodeLogProxyConfig, string) {
		return config.NodeLogProxyConfig{ConnectTimeout: time.Second}, ""
	}}
	router := gin.New()
	router.GET("/nodes/:nodeId/logs", h.ProxyNodeLogsHandler)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest("GET", "/nodes/empty-query/logs", nil))
	require.Equal(t, 200, rec.Code, rec.Body.String())
	require.Equal(t, "{}\n", rec.Body.String())
	require.EqualValues(t, 1, hits.Load())
}

// Review finding 3: invocation-only nodes keep the existing logs 502 and local
// lifecycle fallback; those endpoints must not reuse the invocation URL.
func TestReviewInvocationOnlyAuxiliaryEndpoints(t *testing.T) {
	invocation := "http://127.0.0.1:18190/invoke"
	store := &overrideStorage{getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
		return &types.AgentNode{ID: "invocation-only", InvocationURL: &invocation}, nil
	}}
	h := &NodeLogsProxyHandler{Storage: store, Snapshot: func() (config.NodeLogProxyConfig, string) {
		return config.NodeLogProxyConfig{ConnectTimeout: time.Second}, ""
	}}
	router := gin.New()
	router.GET("/nodes/:nodeId/logs", h.ProxyNodeLogsHandler)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest("GET", "/nodes/invocation-only/logs", nil))
	require.Equal(t, http.StatusBadGateway, rec.Code, rec.Body.String())
	endpoints := NewLifecycleHandler(store, nil).buildEndpoints(context.Background(), "invocation-only", 18190)
	require.Equal(t, "http://localhost:18190/health", endpoints["health"])
}
