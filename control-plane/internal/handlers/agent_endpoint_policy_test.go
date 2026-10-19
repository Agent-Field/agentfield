package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func registerEndpointRequest(t *testing.T, store *nodeRESTStorageStub, base string, invocation *string, discovery *types.CallbackDiscoveryInfo) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(types.AgentNode{ID: "policy-node", BaseURL: base, InvocationURL: invocation, CallbackDiscovery: discovery})
	require.NoError(t, err)
	router := gin.New()
	router.POST("/nodes", RegisterNodeHandler(store, nil, nil, nil, nil, nil))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/nodes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(rec, req)
	return rec
}

// C1-C4/C8: invalid registration/re-registration never writes; default local,
// container, cluster and prefix URLs still work even before DNS is ready.
func TestAgentEndpointRegistrationContract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	defer agenturl.Configure(false, nil)
	defer agenturl.SetNetworkHooksForTesting(func(ctx context.Context, host string) ([]string, error) {
		if host == "pod.cluster.local" {
			return []string{"10.0.0.2"}, nil
		}
		return nil, errors.New("DNS not ready")
	}, nil)()
	forbidden := []string{"http://127.0.0.1:18190/x?", "http://127.0.0.1:18190/x?a=b", "http://127.0.0.1:18190/x#frag", "http://127.0.0.1:18190/x#", "http://169.254.1.2", "http://[fe80::1]", "http://[fd00:ec2::254]", "http://[::ffff:169.254.169.254]", "http://[::169.254.169.254]", "http://[64:ff9b::a9fe:a9fe]", "http://[64:ff9b:1:a9fe:a9:fe00::]", "http://[2002:a9fe:a9fe::]", "http://[2001:0:4136:e378:8000:63bf:5601:5601]", "http://metadata.google.internal", "http://METADATA.GOOG.", "http://metadata"}
	for _, base := range forbidden {
		t.Run(base, func(t *testing.T) {
			store := &nodeRESTStorageStub{}
			rec := registerEndpointRequest(t, store, base, nil, &types.CallbackDiscoveryInfo{Mode: "auto", Preferred: "http://127.0.0.1:18190"})
			require.Equal(t, 400, rec.Code, rec.Body.String())
			require.Nil(t, store.registeredAgent)
			require.Contains(t, rec.Body.String(), "base_url")
		})
	}
	for _, base := range []string{"http://127.0.0.1:18190", "http://localhost:18190", "http://10.0.0.1", "http://172.16.1.2", "http://192.168.0.1", "http://pod.cluster.local", "http://host.docker.internal", "https://example-agent.internal/prefix"} {
		t.Run(base, func(t *testing.T) {
			store := &nodeRESTStorageStub{}
			rec := registerEndpointRequest(t, store, base, nil, nil)
			require.Equal(t, 201, rec.Code, rec.Body.String())
			require.Equal(t, base, store.registeredAgent.BaseURL)
		})
	}
	for _, tc := range []struct {
		base    string
		allowed []string
		ok      bool
	}{
		{"http://127.0.0.1:18190", nil, false}, {"http://10.0.0.1", nil, false},
		{"http://localhost:18190", []string{"LOCALHOST"}, true},
		{"http://pod.cluster.local", []string{"*.cluster.local"}, true},
		{"http://10.0.0.1", []string{"10.0.0.0/8"}, true},
		{"http://169.254.169.254", []string{"169.254.0.0/16"}, false},
	} {
		agenturl.Configure(true, tc.allowed)
		store := &nodeRESTStorageStub{}
		rec := registerEndpointRequest(t, store, tc.base, nil, nil)
		if tc.ok {
			require.Equal(t, 201, rec.Code, rec.Body.String())
		} else {
			require.Equal(t, 400, rec.Code, rec.Body.String())
			require.Nil(t, store.registeredAgent)
		}
	}
	agenturl.Configure(false, nil)
	for _, inv := range []string{"http://169.254.169.254/execute", "http://127.0.0.1/execute?", "http://127.0.0.1/a/../execute"} {
		existing := &types.AgentNode{ID: "policy-node", BaseURL: "http://127.0.0.1:18190"}
		store := &nodeRESTStorageStub{agent: existing}
		rec := registerEndpointRequest(t, store, existing.BaseURL, &inv, nil)
		require.Equal(t, 400, rec.Code, rec.Body.String())
		require.Nil(t, store.registeredAgent)
		require.Nil(t, store.agent.InvocationURL)
	}
	inv := "HTTP://LOCALHOST:18190/prefix/execute/"
	store := &nodeRESTStorageStub{}
	rec := registerEndpointRequest(t, store, "HTTP://LOCALHOST:18190/prefix/", &inv, nil)
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.Equal(t, "http://localhost:18190/prefix", store.registeredAgent.BaseURL)
	require.Equal(t, "http://localhost:18190/prefix/execute", *store.registeredAgent.InvocationURL)
}

// C5-C6: execution and legacy reasoner/skill calls retain the base prefix and
// cannot turn a target name into an alternate path or query.
func TestAgentEndpointExecutionPaths(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Empty(t, r.URL.RawQuery)
		require.True(t, strings.HasPrefix(r.URL.Path, "/prefix/reasoners/") || strings.HasPrefix(r.URL.Path, "/prefix/skills/"), r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"ok":true}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	for _, name := range []string{"echo", "a?b", "a#b", "a/b", "..", "%2f"} {
		t.Run(name, func(t *testing.T) {
			agent := &types.AgentNode{ID: "policy-node", BaseURL: server.URL + "/prefix/", Reasoners: []types.ReasonerDefinition{{ID: name}}}
			controller := newExecutionController(newTestExecutionStorage(agent), nil, nil, time.Second, "")
			plan := testPlan(agent)
			plan.target.TargetName = name
			before := hits.Load()
			resp, err := controller.dispatchAgentRequest(context.Background(), plan)
			if strings.Contains(name, "/") || name == ".." || name == "%2f" {
				require.Error(t, err)
				require.Equal(t, before, hits.Load())
			} else {
				require.NoError(t, err)
				resp.Body.Close()
				require.Equal(t, before+1, hits.Load())
			}
		})
	}
	// Legacy direct handlers use the same builder and policy factory.
	for _, kind := range []string{"reasoner", "skill"} {
		node := &types.AgentNode{ID: "policy-node", BaseURL: server.URL + "/prefix", Reasoners: []types.ReasonerDefinition{{ID: "echo"}}, Skills: []types.SkillDefinition{{ID: "echo"}}}
		store := newReasonerHandlerStorage(node)
		router := gin.New()
		if kind == "reasoner" {
			router.POST("/nodes/:node_id/reasoners/:reasoner_id", ExecuteReasonerHandler(store))
		} else {
			router.POST("/nodes/:node_id/skills/:skill_id", ExecuteSkillHandler(store))
		}
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/nodes/policy-node/"+kind+"s/policy-node.echo", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(rec, req)
		require.Equal(t, 200, rec.Code, rec.Body.String())
	}
}

// C7/C9: real execute responses name the policy/shape failure and never relay
// an agent body. Nodes inserted directly remain listable and can re-register.
func TestAgentEndpointStoredNodeDispatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if _, err := w.Write([]byte(`{"secret":"must-not-relay"}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return []string{"169.254.169.254"}, nil }, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, strings.TrimPrefix(server.URL, "http://"))
	})()
	for _, base := range []string{server.URL + "/prefix?hidden=true", server.URL + "/#frag", "http://169.254.169.254", "http://rebound.test:18190"} {
		t.Run(base, func(t *testing.T) {
			agent := &types.AgentNode{ID: "policy-node", BaseURL: base, Reasoners: []types.ReasonerDefinition{{ID: "echo"}}}
			store := newTestExecutionStorage(agent)
			listStore := &nodeRESTStorageStub{listAgents: []*types.AgentNode{agent}}
			router := gin.New()
			router.GET("/nodes", ListNodesHandler(listStore))
			router.POST("/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
			list := httptest.NewRecorder()
			router.ServeHTTP(list, httptest.NewRequest("GET", "/nodes", nil))
			require.Equal(t, 200, list.Code)
			require.Contains(t, list.Body.String(), "policy-node")
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/execute/policy-node.echo", strings.NewReader(`{"input":{}}`))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(rec, req)
			require.GreaterOrEqual(t, rec.Code, 400, rec.Body.String())
			require.Contains(t, rec.Body.String(), "endpoint")
			require.NotContains(t, rec.Body.String(), "must-not-relay")
			require.Zero(t, hits.Load())
		})
	}
}

// C8: public-looking serverless discovery resolves through a test-only hook,
// retains prefixes, and stores the validated execution endpoint.
func TestAgentEndpointServerlessDiscovery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/prefix/discover", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
		if _, err := w.Write([]byte(`{"node_id":"fn","reasoners":[{"id":"echo"}]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return []string{"127.0.0.1"}, nil }, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, strings.TrimPrefix(server.URL, "http://"))
	})()
	store := &nodeRESTStorageStub{}
	router := gin.New()
	router.POST("/serverless", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"function.example"}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/serverless", strings.NewReader(`{"invocation_url":"http://FUNCTION.EXAMPLE/prefix/"}`))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(rec, req)
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.Equal(t, "http://function.example/prefix", store.registeredAgent.BaseURL)
	require.Equal(t, "http://function.example/prefix/execute", *store.registeredAgent.InvocationURL)
}

// C7/C7b: execution refuses a cross-origin redirect and a same-host redirect
// whose second connection resolves to metadata, without relaying any body.
func TestAgentEndpointExecutionRedirects(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, kind := range []string{"cross", "rebind"} {
		t.Run(kind, func(t *testing.T) {
			var hits, lookups atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Connection", "close")
				destination := "/metadata"
				if kind == "cross" {
					destination = "http://other.test/metadata"
				}
				http.Redirect(w, r, destination, 302)
			}))
			defer server.Close()
			defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
				if lookups.Add(1) == 1 {
					return []string{"127.0.0.1"}, nil
				}
				return []string{"169.254.169.254"}, nil
			}, func(ctx context.Context, network, addr string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, strings.TrimPrefix(server.URL, "http://"))
			})()
			agent := &types.AgentNode{ID: "redirect-node", BaseURL: "http://agent.test/prefix", Reasoners: []types.ReasonerDefinition{{ID: "echo"}}}
			store := newTestExecutionStorage(agent)
			router := gin.New()
			router.POST("/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/execute/redirect-node.echo", strings.NewReader(`{"input":{}}`))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(rec, req)
			require.GreaterOrEqual(t, rec.Code, 400, rec.Body.String())
			require.EqualValues(t, 1, hits.Load())
			if kind == "cross" {
				require.Contains(t, rec.Body.String(), "cross-origin")
			} else {
				require.Contains(t, rec.Body.String(), "link-local")
			}
		})
	}
}

// C9: a real pre-existing database row remains listable, refuses dispatch, and
// becomes executable again when the same node re-registers a valid endpoint.
func TestAgentEndpointReregistrationRepairsLegacyNode(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Equal(t, "/prefix/reasoners/echo", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
		if _, err := w.Write([]byte(`{"ok":true}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	store := newLegacyLocalStorage(t)
	require.NoError(t, store.RegisterAgent(context.Background(), &types.AgentNode{ID: "legacy-policy", BaseURL: server.URL + "/prefix?bad", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, RegisteredAt: time.Now(), LastHeartbeat: time.Now(), Reasoners: []types.ReasonerDefinition{{ID: "echo"}}}))
	router := gin.New()
	router.GET("/nodes", ListNodesHandler(store))
	router.POST("/nodes", RegisterNodeHandler(store, nil, nil, nil, nil, nil))
	router.POST("/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	request := func(method, path, body string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(rec, req)
		return rec
	}
	rec := request("GET", "/nodes?show_all=true", "")
	require.Equal(t, 200, rec.Code)
	require.Contains(t, rec.Body.String(), "legacy-policy")
	rec = request("POST", "/execute/legacy-policy.echo", `{"input":{}}`)
	require.GreaterOrEqual(t, rec.Code, 400)
	require.Contains(t, rec.Body.String(), "query")
	require.Zero(t, hits.Load())
	body, err := json.Marshal(types.AgentNode{ID: "legacy-policy", BaseURL: server.URL + "/prefix/", Reasoners: []types.ReasonerDefinition{{ID: "echo"}}})
	require.NoError(t, err)
	rec = request("POST", "/nodes", string(body))
	require.Equal(t, 201, rec.Code, rec.Body.String())
	rec = request("POST", "/execute/legacy-policy.echo", `{"input":{}}`)
	require.Equal(t, 200, rec.Code, rec.Body.String())
	require.EqualValues(t, 1, hits.Load())
}

// Review finding 2: selection must not wait for unused candidates' DNS.
func TestReviewRegistrationSkipsUnusedDNS(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var lookups atomic.Int32
	defer agenturl.SetNetworkHooksForTesting(func(ctx context.Context, host string) ([]string, error) {
		lookups.Add(1)
		<-ctx.Done()
		return nil, ctx.Err()
	}, nil)()
	store := &nodeRESTStorageStub{}
	started := time.Now()
	rec := registerEndpointRequest(t, store, "http://127.0.0.1:18190", nil, &types.CallbackDiscoveryInfo{
		Mode: "auto", Candidates: []string{"http://slow-one.test", "http://slow-two.test", "http://slow-three.test"},
	})
	require.Less(t, time.Since(started), 3*time.Second)
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.Equal(t, "http://127.0.0.1:18190", store.registeredAgent.BaseURL)
	require.Zero(t, lookups.Load())
}

// Review finding 2: each normalized URL gets at most one DNS check per request.
func TestReviewRegistrationChecksURLOnce(t *testing.T) {
	var lookups atomic.Int32
	defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		lookups.Add(1)
		return []string{"127.0.0.1"}, nil
	}, nil)()
	endpoint := "http://agent.test:18190/prefix"
	rec := registerEndpointRequest(t, &nodeRESTStorageStub{}, endpoint, &endpoint, &types.CallbackDiscoveryInfo{Mode: "auto", Candidates: []string{endpoint + "/"}})
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.EqualValues(t, 1, lookups.Load())
}

// Review finding 2: discovery DNS inherits request cancellation; unavailable
// DNS still permits registration, with the destination checked at dial time.
func TestReviewRegistrationDiscoveryContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer agenturl.SetNetworkHooksForTesting(func(lookupCtx context.Context, host string) ([]string, error) {
		cancel()
		<-lookupCtx.Done()
		return nil, lookupCtx.Err()
	}, nil)()
	router := gin.New()
	store := &nodeRESTStorageStub{}
	router.POST("/nodes", RegisterNodeHandler(store, nil, nil, nil, nil, nil))
	req := httptest.NewRequest("POST", "/nodes", strings.NewReader(`{"id":"context-node","callback_discovery":{"mode":"auto","candidates":["http://slow.test:18190"]}}`)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	started := time.Now()
	router.ServeHTTP(rec, req)
	require.Less(t, time.Since(started), time.Second)
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.Equal(t, "http://slow.test:18190", store.registeredAgent.BaseURL)
}

// Review finding 3: invocation-only registrations and existing database rows
// dispatch through the real execute handler to the validated invocation path.
func TestReviewInvocationOnlyDispatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, direct := range []bool{false, true} {
		t.Run(map[bool]string{false: "registered", true: "stored-row"}[direct], func(t *testing.T) {
			var hits atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				require.Equal(t, "/prefix/invoke", r.URL.Path)
				require.Empty(t, r.URL.RawQuery)
				w.Header().Set("Content-Type", "application/json")
				require.NoError(t, json.NewEncoder(w).Encode(gin.H{"result": "invocation reached"}))
			}))
			defer server.Close()
			invocation := server.URL + "/prefix/invoke"
			node := &types.AgentNode{ID: "invocation-only", DeploymentType: "serverless", InvocationURL: &invocation, Reasoners: []types.ReasonerDefinition{{ID: "echo"}}, RegisteredAt: time.Now(), LastHeartbeat: time.Now()}
			store := newLegacyLocalStorage(t)
			router := gin.New()
			router.POST("/nodes", RegisterNodeHandler(store, nil, nil, nil, nil, nil))
			router.POST("/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
			if direct {
				require.NoError(t, store.RegisterAgent(context.Background(), node))
			} else {
				body, err := json.Marshal(node)
				require.NoError(t, err)
				rec := httptest.NewRecorder()
				req := httptest.NewRequest("POST", "/nodes", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				router.ServeHTTP(rec, req)
				require.Equal(t, 201, rec.Code, rec.Body.String())
			}
			stored, err := store.GetAgent(context.Background(), node.ID)
			require.NoError(t, err)
			require.Empty(t, stored.BaseURL)
			listed, err := store.ListAgents(context.Background(), types.AgentFilters{})
			require.NoError(t, err)
			require.Len(t, listed, 1)
			require.Empty(t, listed[0].BaseURL)
			require.Equal(t, &invocation, listed[0].InvocationURL)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/execute/invocation-only.echo", strings.NewReader(`{"input":{}}`))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(rec, req)
			require.Equal(t, 200, rec.Code, rec.Body.String())
			require.Contains(t, rec.Body.String(), "invocation reached")
			require.EqualValues(t, 1, hits.Load())
		})
	}
}

// Review finding 2: serverless discovery DNS also uses request cancellation.
func TestReviewServerlessRegistrationContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer agenturl.SetNetworkHooksForTesting(func(lookupCtx context.Context, host string) ([]string, error) {
		cancel()
		<-lookupCtx.Done()
		return nil, lookupCtx.Err()
	}, nil)()
	router := gin.New()
	router.POST("/serverless", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"slow.test"}))
	req := httptest.NewRequest("POST", "/serverless", strings.NewReader(`{"invocation_url":"http://slow.test/invoke"}`)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	started := time.Now()
	router.ServeHTTP(rec, req)
	require.Less(t, time.Since(started), time.Second)
	// DNS unavailability is tolerated; the subsequent discovery HTTP request
	// fails promptly because its request context was cancelled.
	require.Equal(t, http.StatusBadGateway, rec.Code, rec.Body.String())
}

// Review finding 4: handler registration honors localhost's resolved CIDRs.
func TestReviewStrictLocalhostRegistration(t *testing.T) {
	agenturl.Configure(true, []string{"127.0.0.0/8", "::1/128"})
	defer agenturl.Configure(false, nil)
	defer agenturl.SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		return []string{"127.0.0.1", "::1"}, nil
	}, nil)()
	store := &nodeRESTStorageStub{}
	rec := registerEndpointRequest(t, store, "http://localhost:18190", nil, nil)
	require.Equal(t, 201, rec.Code, rec.Body.String())
	require.Equal(t, "http://localhost:18190", store.registeredAgent.BaseURL)
}
