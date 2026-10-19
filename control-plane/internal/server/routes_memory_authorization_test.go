package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

type memoryTouch struct{ operation, scope, id string }

// Embedding the interface makes unexpected storage access fail loudly. Every
// memory method is recorded, including previous-value reads and event writes.
type authorizedMemoryStore struct {
	storage.StorageProvider
	touches       []memoryTouch
	agentErr      error
	callerErr     error
	events        []*types.MemoryChangeEvent
	participation []types.ExecutionFilter
	policies      []*types.AccessPolicy
}

func (s *authorizedMemoryStore) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	if id == "agent-beta" && s.callerErr != nil {
		return nil, s.callerErr
	}
	if id != "agent-beta" && s.agentErr != nil {
		return nil, s.agentErr
	}
	if id == "agent-alpha" || id == "agent-beta" {
		return &types.AgentNode{ID: id, ApprovedTags: []string{"reader"}}, nil
	}
	return nil, nil
}
func (s *authorizedMemoryStore) ListAgents(_ context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if s.agentErr != nil {
		return nil, s.agentErr
	}
	return []*types.AgentNode{{ID: "agent-alpha"}, {ID: "agent-beta"}}, nil
}
func (s *authorizedMemoryStore) touch(op, scope, id string) {
	s.touches = append(s.touches, memoryTouch{op, scope, id})
}
func (s *authorizedMemoryStore) GetMemory(_ context.Context, scope, id, key string) (*types.Memory, error) {
	s.touch("get", scope, id)
	return &types.Memory{Scope: scope, ScopeID: id, Key: key, Data: json.RawMessage(`"PRIVATE_MARKER"`)}, nil
}
func (s *authorizedMemoryStore) SetMemory(_ context.Context, m *types.Memory) error {
	s.touch("set", m.Scope, m.ScopeID)
	return nil
}
func (s *authorizedMemoryStore) DeleteMemory(_ context.Context, scope, id, key string) error {
	s.touch("delete", scope, id)
	return nil
}
func (s *authorizedMemoryStore) ListMemory(_ context.Context, scope, id string) ([]*types.Memory, error) {
	s.touch("list", scope, id)
	return []*types.Memory{}, nil
}
func (s *authorizedMemoryStore) StoreEvent(_ context.Context, e *types.MemoryChangeEvent) error {
	s.touch("store event", e.Scope, e.ScopeID)
	return nil
}
func (s *authorizedMemoryStore) PublishMemoryChange(_ context.Context, e types.MemoryChangeEvent) error {
	s.touch("publish", e.Scope, e.ScopeID)
	return nil
}
func (s *authorizedMemoryStore) SetVector(_ context.Context, v *types.VectorRecord) error {
	s.touch("set vector", v.Scope, v.ScopeID)
	return nil
}
func (s *authorizedMemoryStore) GetVector(_ context.Context, scope, id, key string) (*types.VectorRecord, error) {
	s.touch("get vector", scope, id)
	return &types.VectorRecord{Scope: scope, ScopeID: id, Key: key}, nil
}
func (s *authorizedMemoryStore) DeleteVector(_ context.Context, scope, id, key string) error {
	s.touch("delete vector", scope, id)
	return nil
}
func (s *authorizedMemoryStore) DeleteVectorsByPrefix(_ context.Context, scope, id, prefix string) (int, error) {
	s.touch("delete namespace", scope, id)
	return 1, nil
}
func (s *authorizedMemoryStore) SimilaritySearch(_ context.Context, scope, id string, _ []float32, _ int, _ map[string]interface{}) ([]*types.VectorSearchResult, error) {
	s.touch("search", scope, id)
	return []*types.VectorSearchResult{}, nil
}
func (s *authorizedMemoryStore) SubscribeToMemoryChanges(_ context.Context, scope, id string) (<-chan types.MemoryChangeEvent, error) {
	s.touch("subscribe", scope, id)
	ch := make(chan types.MemoryChangeEvent, len(s.events))
	for _, e := range s.events {
		ch <- *e
	}
	close(ch)
	return ch, nil
}
func (s *authorizedMemoryStore) GetEventHistory(_ context.Context, f types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	scope, id := "", ""
	if f.Scope != nil {
		scope = *f.Scope
	}
	if f.ScopeID != nil {
		id = *f.ScopeID
	}
	s.touch("history", scope, id)
	return s.events, nil
}
func (s *authorizedMemoryStore) QueryExecutionRecords(_ context.Context, f types.ExecutionFilter) ([]*types.Execution, error) {
	s.participation = append(s.participation, f)
	if f.AgentNodeID != nil && *f.AgentNodeID == "agent-beta" &&
		((f.ActorID != nil && (*f.ActorID == "user-123" || *f.ActorID == "agent-alpha:notes")) || (f.SessionID != nil && *f.SessionID == "session-1") || (f.RunID != nil && *f.RunID == "run-1")) {
		return []*types.Execution{{AgentNodeID: "agent-beta"}}, nil
	}
	return nil, nil
}

func memoryAuthorizationRouter(t *testing.T, store *authorizedMemoryStore, enabled bool) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)
	s := &AgentFieldServer{config: &config.Config{}, storage: store}
	s.config.Features.DID.Authorization.Enabled = enabled
	s.accessPolicyService = services.NewAccessPolicyService(store)
	require.NoError(t, s.accessPolicyService.Initialize(context.Background()))
	s.didWebService = services.NewDIDWebService("local.test", nil, store)
	r := gin.New()
	r.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
	// Call the real production registrar: this is also the C11 coverage guard.
	s.registerMemoryRoutes(r.Group("/api/v1"))
	t.Cleanup(s.cancelStreams)
	return r
}

type memoryRouteCase struct{ method, route, path, body string }

var memoryDenialCases = []memoryRouteCase{
	{"POST", "/set", "/set", `{"key":"k","data":"new","scope":"SCOPE"}`},
	{"POST", "/get", "/get", `{"key":"k","scope":"SCOPE"}`},
	{"POST", "/delete", "/delete", `{"key":"k","scope":"SCOPE"}`},
	{"GET", "/list", "/list?scope=SCOPE", ``},
	{"POST", "/vector", "/vector", `{"key":"k","embedding":[1,2],"scope":"SCOPE"}`},
	{"POST", "/vector/set", "/vector/set", `{"key":"k","embedding":[1,2],"scope":"SCOPE"}`},
	{"GET", "/vector/:key", "/vector/k?scope=SCOPE", ``},
	{"POST", "/vector/search", "/vector/search", `{"query_embedding":[1,2],"scope":"SCOPE"}`},
	{"DELETE", "/vector/:key", "/vector/k?scope=SCOPE", ``},
	{"POST", "/vector/delete", "/vector/delete?scope=SCOPE", `{"key":"k","scope":"SCOPE"}`},
	{"DELETE", "/vector/namespace", "/vector/namespace", `{"namespace":"ns","scope":"SCOPE"}`},
	{"GET", "/events/ws", "/events/ws?scope=SCOPE&scope_id=ID", ``},
	{"GET", "/events/sse", "/events/sse?scope=SCOPE&scope_id=ID", ``},
	{"GET", "/events/history", "/events/history?scope=SCOPE&scope_id=ID", ``},
}

func memoryRequest(t *testing.T, r *gin.Engine, tc memoryRouteCase, scope, id, caller string, extra map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	path := strings.ReplaceAll(strings.ReplaceAll(tc.path, "SCOPE", url.QueryEscape(scope)), "ID", url.QueryEscape(id))
	body := strings.ReplaceAll(tc.body, "SCOPE", scope)
	req := httptest.NewRequest(tc.method, "/api/v1/memory"+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", "shared-key")
	req.Header.Set("X-Agent-Node-ID", caller)
	switch scope {
	case "actor":
		req.Header.Set("X-Actor-ID", id)
	case "session":
		req.Header.Set("X-Session-ID", id)
	case "workflow":
		req.Header.Set("X-Workflow-ID", id)
	}
	for k, v := range extra {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	return w
}

// C1/C11: every production memory route must have an anonymous denial case.
func TestMemoryRoutesAnonymousDenialCoverage(t *testing.T) {
	store := &authorizedMemoryStore{}
	r := memoryAuthorizationRouter(t, store, true)
	covered := map[string]bool{}
	for _, tc := range memoryDenialCases {
		covered[tc.method+" /api/v1/memory"+tc.route] = true
		for _, scope := range []string{"actor", "session", "workflow"} {
			t.Run(tc.method+tc.route+"/"+scope, func(t *testing.T) {
				store.touches = nil
				w := memoryRequest(t, r, tc, scope, "agent-alpha", "", nil)
				require.Equal(t, 403, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "memory_identity_required")
				require.NotContains(t, w.Body.String(), "PRIVATE_MARKER")
				require.Empty(t, store.touches, "no reads, writes, stored events, publications or subscriptions")
			})
		}
	}
	for _, route := range r.Routes() {
		require.True(t, covered[route.Method+" "+route.Path], "missing C1 case: %s %s", route.Method, route.Path)
	}
	require.Len(t, r.Routes(), len(covered))
}

// C4: misleading headers never authorize alpha's actor scope on any route.
func TestMemoryRoutesAuthorizeActualActorScope(t *testing.T) {
	store := &authorizedMemoryStore{}
	r := memoryAuthorizationRouter(t, store, true)
	variants := []map[string]string{
		{"X-Memory-Scope": "global"}, {"X-Workflow-ID": "run-1"}, {"X-Session-ID": "session-1"},
		{"X-Memory-Scope": "actor", "X-Session-ID": "session-1"},
	}
	// C4 extension: the same protection applies to storage-key aliases, on
	// vector and concrete event routes as well as key/value routes.
	variants = append(variants, map[string]string{"X-Actor-ID": "agent-alpha:notes", "X-Memory-Scope": "global"})
	for _, tc := range memoryDenialCases {
		for i, headers := range variants {
			t.Run(fmt.Sprintf("%s%s/variant%d", tc.method, tc.route, i), func(t *testing.T) {
				store.touches = nil
				id := "agent-alpha"
				if headers["X-Actor-ID"] != "" {
					id = headers["X-Actor-ID"]
				}
				w := memoryRequest(t, r, tc, "actor", id, "agent-beta", headers)
				require.Equal(t, 403, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "scope_ownership_denied")
				require.NotContains(t, w.Body.String(), "PRIVATE_MARKER")
				require.Empty(t, store.touches)
			})
		}
	}
}

// C7 extension: caller lookup failures deny every route/scope before storage,
// including websocket upgrade, SSE headers, and global memory operations.
func TestMemoryRoutesCallerLookupUnavailable(t *testing.T) {
	for _, tc := range memoryDenialCases {
		for _, scope := range []string{"global", "actor", "session", "workflow"} {
			t.Run(tc.method+tc.route+"/"+scope, func(t *testing.T) {
				store := &authorizedMemoryStore{callerErr: fmt.Errorf("database unavailable")}
				r := memoryAuthorizationRouter(t, store, true)
				w := memoryRequest(t, r, tc, scope, "scope-id", "agent-beta", nil)
				require.Equal(t, 503, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "memory_authorization_unavailable")
				require.NotContains(t, w.Body.String(), "PRIVATE_MARKER")
				require.Empty(t, store.touches)
			})
		}
	}
}

// C2/C5/C9: legitimate explicit selectors work, including without middleware.
func TestMemoryRoutesAllowedScopesAndLegacyMode(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		for _, scope := range []string{"global", "actor", "session", "workflow"} {
			caller := "agent-beta"
			if scope == "global" || !enabled {
				caller = ""
			}
			id := map[string]string{"global": "global", "actor": "user-123", "session": "session-1", "workflow": "run-1"}[scope]
			for _, tc := range memoryDenialCases {
				t.Run(fmt.Sprintf("auth=%v/%s/%s%s", enabled, scope, tc.method, tc.route), func(t *testing.T) {
					store := &authorizedMemoryStore{}
					r := memoryAuthorizationRouter(t, store, enabled)
					if tc.route == "/events/ws" {
						assertMemoryWebSocket(t, r, "?scope="+scope+"&scope_id="+id, caller, nil)
						return
					}
					w := memoryRequest(t, r, tc, scope, id, caller, nil)
					want := 200
					if tc.route == "/delete" || (tc.route == "/vector/:key" && tc.method == "DELETE") || tc.route == "/vector/delete" {
						want = 204
					}
					require.Equal(t, want, w.Code, w.Body.String())
					require.NotEmpty(t, store.touches)
					for _, touch := range store.touches {
						require.Equal(t, scope, touch.scope)
						require.Equal(t, id, touch.id)
					}
				})
			}
		}
	}
}

// C3/C4: hierarchical reads skip unauthorized candidates without leaking values.
func TestMemoryHierarchicalAuthorization(t *testing.T) {
	for _, caller := range []string{"", "agent-beta"} {
		store := &authorizedMemoryStore{}
		r := memoryAuthorizationRouter(t, store, true)
		headers := map[string]string{"X-Actor-ID": "agent-alpha"}
		if caller == "" {
			headers["X-Workflow-ID"] = "run-1"
			headers["X-Session-ID"] = "session-1"
		}
		w := memoryRequest(t, r, memoryRouteCase{method: "POST", path: "/get", body: `{"key":"k"}`}, "global", "global", caller, headers)
		require.Equal(t, 200, w.Code)
		require.Equal(t, []memoryTouch{{"get", "global", "global"}}, store.touches)
	}
}

// C4/C2: X-Memory-Scope is irrelevant; the inferred session is what is written.
func TestMemoryIgnoresScopeHeaderForInferredWrites(t *testing.T) {
	store := &authorizedMemoryStore{}
	r := memoryAuthorizationRouter(t, store, true)
	w := memoryRequest(t, r, memoryRouteCase{method: "POST", path: "/set", body: `{"key":"k","data":1}`}, "session", "session-1", "agent-beta", map[string]string{"X-Memory-Scope": "actor", "X-Actor-ID": "agent-alpha"})
	require.Equal(t, 200, w.Code)
	for _, touch := range store.touches {
		require.Equal(t, "session", touch.scope)
		require.Equal(t, "session-1", touch.id)
	}
	store.touches = nil
	w = memoryRequest(t, r, memoryRouteCase{method: "POST", path: "/set", body: `{"key":"k","data":1}`}, "global", "global", "", nil)
	require.Equal(t, 200, w.Code)
	for _, touch := range store.touches {
		require.Equal(t, "global", touch.scope)
	}
}

func assertMemoryWebSocket(t *testing.T, r *gin.Engine, query, caller string, want []string) {
	t.Helper()
	srv := httptest.NewServer(r)
	defer srv.Close()
	headers := http.Header{"X-API-Key": []string{"shared-key"}, "X-Agent-Node-ID": []string{caller}}
	conn, resp, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(srv.URL, "http")+"/api/v1/memory/events/ws"+query, headers)
	require.NoError(t, err)
	require.Equal(t, 101, resp.StatusCode)
	defer conn.Close()
	var keys []string
	for {
		var ev types.MemoryChangeEvent
		if err := conn.ReadJSON(&ev); err != nil {
			break
		}
		keys = append(keys, ev.Key)
	}
	require.Equal(t, want, keys)
}

// C6/C9: streams and history apply participation only to wildcard selectors.
func TestMemoryEventVisibility(t *testing.T) {
	events := []*types.MemoryChangeEvent{}
	// C4 extension: even participation cannot grant access to an actor alias
	// whose prefix belongs to another registered agent.
	for _, pair := range [][2]string{{"global", "global"}, {"actor", "agent-beta"}, {"actor", "agent-alpha"}, {"actor", "user-123"}, {"session", "session-1"}, {"workflow", "run-1"}, {"actor", "user-other"}, {"session", "session-other"}, {"workflow", "run-other"}, {"unknown", "unknown"}, {"actor", "agent-alpha:notes"}} {
		events = append(events, &types.MemoryChangeEvent{Scope: pair[0], ScopeID: pair[1], Key: pair[0] + ":" + pair[1], Data: json.RawMessage(`"PRIVATE_MARKER"`), PreviousData: json.RawMessage(`"PREVIOUS_MARKER"`)})
	}
	// C6: shared empty-ID buckets are never broadcast on wildcard connections.
	for _, scope := range []string{"actor", "session", "workflow"} {
		events = append(events, &types.MemoryChangeEvent{Scope: scope, Key: scope + ":empty"})
	}
	for _, enabled := range []bool{true, false} {
		for _, caller := range []string{"", "agent-beta"} {
			for _, query := range []string{"", "?scope=actor", "?scope_id=*", "?scope=actor&scope_id=*", "?scope=*", "?scope=*&scope_id=*", "?scope=*&scope_id=session-1"} {
				want := []string{events[0].Key}
				if caller != "" {
					want = []string{events[0].Key, events[1].Key, events[3].Key, events[4].Key, events[5].Key}
				}
				if !enabled {
					want = nil
					for _, ev := range events {
						want = append(want, ev.Key)
					}
				}
				for _, transport := range []string{"history", "sse", "ws"} {
					t.Run(fmt.Sprintf("auth=%v/%s/%s%s", enabled, caller, transport, query), func(t *testing.T) {
						store := &authorizedMemoryStore{events: events}
						r := memoryAuthorizationRouter(t, store, enabled)
						if transport == "ws" {
							assertMemoryWebSocket(t, r, query, caller, want)
							return
						}
						w := memoryRequest(t, r, memoryRouteCase{method: "GET", path: "/events/" + transport + query}, "", "", caller, nil)
						require.Equal(t, 200, w.Code)
						var got []string
						if transport == "history" {
							var visible []*types.MemoryChangeEvent
							require.NoError(t, json.Unmarshal(w.Body.Bytes(), &visible))
							for _, ev := range visible {
								got = append(got, ev.Key)
							}
						} else {
							for _, line := range strings.Split(w.Body.String(), "\n") {
								if strings.HasPrefix(line, "data:") {
									var ev types.MemoryChangeEvent
									require.NoError(t, json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &ev))
									got = append(got, ev.Key)
								}
							}
						}
						require.Equal(t, want, got)
					})
				}
			}
		}
	}
	// Concrete user selector grants read visibility even without participation.
	store := &authorizedMemoryStore{events: events[6:7]}
	r := memoryAuthorizationRouter(t, store, true)
	assertMemoryWebSocket(t, r, "?scope=actor&scope_id=user-other", "agent-beta", []string{events[6].Key})
	for _, transport := range []string{"sse", "history"} {
		w := memoryRequest(t, r, memoryRouteCase{method: "GET", path: "/events/" + transport + "?scope=actor&scope_id=user-other"}, "", "", "agent-beta", nil)
		require.Equal(t, 200, w.Code)
		require.Contains(t, w.Body.String(), events[6].Key)
	}
	require.Empty(t, store.participation)
}

// C10: warnings distinguish asserted identity from missing authorization services.
func TestMemoryAuthorizationStartupLogs(t *testing.T) {
	original := logger.Logger
	t.Cleanup(func() { logger.Logger = original })
	for _, tc := range []struct {
		name                          string
		enabled, didAuth, policy, did bool
		level, message                string
	}{
		{"asserted", true, false, true, true, "warn", "any holder of the API key can set"},
		{"signed", true, true, true, true, "info", "unsigned requests still fall back"},
		{"no policy", true, false, false, true, "warn", "not authorization-checked"},
		{"no DID", true, false, true, false, "warn", "not authorization-checked"},
		{"no services", true, false, false, false, "warn", "not authorization-checked"},
		{"disabled", false, false, false, false, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			logger.Logger = zerolog.New(&buf)
			s := &AgentFieldServer{config: &config.Config{}, storage: &authorizedMemoryStore{}}
			s.config.Features.DID.Authorization.Enabled = tc.enabled
			s.config.Features.DID.Authorization.DIDAuthEnabled = tc.didAuth
			if tc.policy {
				s.accessPolicyService = services.NewAccessPolicyService(s.storage)
			}
			if tc.did {
				s.didWebService = services.NewDIDWebService("local.test", nil, s.storage)
			}
			s.registerMemoryRoutes(gin.New().Group("/api/v1"))
			defer s.cancelStreams()
			if tc.message == "" {
				require.Empty(t, buf.String())
				return
			}
			var entry map[string]interface{}
			require.NoError(t, json.Unmarshal(buf.Bytes(), &entry))
			require.Equal(t, tc.level, entry["level"])
			require.Contains(t, entry["message"], tc.message)
			if tc.name == "asserted" {
				require.Contains(t, entry["message"], "global memory")
				require.Contains(t, entry["message"], "enable DID request authentication")
			}
			if strings.HasPrefix(tc.name, "no ") {
				require.Equal(t, tc.policy, entry["access_policy_service_available"])
				require.Equal(t, tc.did, entry["did_web_service_available"])
			}
		})
	}
}

// C5: invalid scopes and actor registry failures fail before storage on every route.
func TestMemoryRoutesInvalidScopesAndLookupFailures(t *testing.T) {
	for _, tc := range memoryDenialCases {
		for _, scenario := range []struct {
			scope, id   string
			lookupError bool
		}{
			{"Actor", "id", false}, {"agent", "id", false}, {" actor", "id", false},
			{"actor", "user-123", true},
		} {
			t.Run(tc.method+tc.route+"/"+scenario.scope, func(t *testing.T) {
				store := &authorizedMemoryStore{}
				if scenario.lookupError {
					store.agentErr = fmt.Errorf("database unavailable")
				}
				r := memoryAuthorizationRouter(t, store, true)
				w := memoryRequest(t, r, tc, scenario.scope, scenario.id, "agent-beta", nil)
				require.Equal(t, 403, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "scope_ownership_denied")
				require.Empty(t, store.touches)
			})
		}
	}
}

// C1/C5: released SDKs use empty session/workflow buckets. Identified callers
// retain access on all explicit data routes; anonymous callers are still denied.
func TestMemoryRoutesEmptySessionWorkflowIDs(t *testing.T) {
	for _, tc := range memoryDenialCases {
		if strings.HasPrefix(tc.route, "/events/") {
			continue // An empty event selector is wildcard, covered by C6 above.
		}
		for _, scope := range []string{"session", "workflow"} {
			for _, caller := range []string{"", "agent-beta"} {
				t.Run(tc.method+tc.route+"/"+scope+"/"+caller, func(t *testing.T) {
					store := &authorizedMemoryStore{}
					r := memoryAuthorizationRouter(t, store, true)
					w := memoryRequest(t, r, tc, scope, "", caller, nil)
					if caller == "" {
						require.Equal(t, 403, w.Code, w.Body.String())
						require.Contains(t, w.Body.String(), "memory_identity_required")
						require.Empty(t, store.touches)
						return
					}
					require.Contains(t, []int{200, 204}, w.Code, w.Body.String())
					require.NotEmpty(t, store.touches)
					for _, touch := range store.touches {
						require.Equal(t, scope, touch.scope)
						require.Empty(t, touch.id)
					}
				})
			}
		}
	}
}

// C4/C9: the legacy delete route historically selects scope from query/headers,
// ignoring body.scope. Authorize exactly that selection and preserve it with auth off.
func TestMemoryLegacyVectorDeleteAuthorizesResolvedScope(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		store := &authorizedMemoryStore{}
		r := memoryAuthorizationRouter(t, store, enabled)
		w := memoryRequest(t, r, memoryRouteCase{method: "POST", path: "/vector/delete", body: `{"key":"k","scope":"actor"}`}, "session", "session-1", "agent-beta", map[string]string{"X-Actor-ID": "agent-alpha", "X-Memory-Scope": "actor"})
		require.Equal(t, 204, w.Code)
		require.Equal(t, []memoryTouch{{"delete vector", "session", "session-1"}}, store.touches)
	}
}

func (s *authorizedMemoryStore) GetAccessPolicies(context.Context) ([]*types.AccessPolicy, error) {
	return s.policies, nil
}

// C7: a matching policy denies every real route and operation, including global.
func TestMemoryRoutesMatchingDenyPolicy(t *testing.T) {
	for _, tc := range memoryDenialCases {
		for _, scope := range []string{"global", "actor", "session", "workflow"} {
			t.Run(tc.method+tc.route+"/"+scope, func(t *testing.T) {
				store := &authorizedMemoryStore{policies: []*types.AccessPolicy{{ID: 1, Name: "deny-memory", Enabled: true, CallerTags: []string{"reader"}, TargetTags: []string{"memory"}, DenyFunctions: []string{"*"}}}}
				r := memoryAuthorizationRouter(t, store, true)
				w := memoryRequest(t, r, tc, scope, "scope-id", "agent-beta", nil)
				require.Equal(t, 403, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "memory_access_denied")
				require.Empty(t, store.touches)
			})
		}
	}
}
