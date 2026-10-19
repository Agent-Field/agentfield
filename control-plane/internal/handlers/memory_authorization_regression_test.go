package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// C1: the reporter's shared API key must not disclose another actor's memory.
func TestReproSharedAPIKeyWithoutCallerIdentityReadsOtherActorMemory(t *testing.T) {
	store := newMemoryStorageStub()
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
	router.Use(middleware.MemoryPermissionMiddleware(nil, &isolationTestAgentResolver{}, &isolationTestDIDResolver{},
		middleware.MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
	router.POST("/api/v1/memory/set", SetMemoryHandler(store))
	router.POST("/api/v1/memory/get", GetMemoryHandler(store))
	do := func(path, body, caller string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-API-Key", "shared-key")
		req.Header.Set("X-Actor-ID", "agent-alpha")
		if caller != "" {
			req.Header.Set("X-Agent-Node-ID", caller)
		}
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w
	}
	require.Equal(t, http.StatusOK, do("/api/v1/memory/set",
		`{"key":"private-note","data":"ALPHA_PRIVATE_MARKER","scope":"actor"}`, "agent-alpha").Code)
	w := do("/api/v1/memory/get", `{"key":"private-note","scope":"actor"}`, "")
	require.Equal(t, http.StatusForbidden, w.Code)
	require.Contains(t, w.Body.String(), "memory_identity_required")
	require.NotContains(t, w.Body.String(), "ALPHA_PRIVATE_MARKER")
	require.Len(t, store.store, 1)
	require.Len(t, store.events, 1)
	require.Len(t, store.published, 1)
}

// C6 / SPEC-6: real storage applies literal filters, so wildcard history must
// omit '*' predicates while retaining a concrete scope_id and event visibility.
func TestMemoryWildcardHistoryWithLocalStorage(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store := storage.NewLocalStorage(storage.LocalStorageConfig{})
	require.NoError(t, store.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: storage.LocalStorageConfig{
		DatabasePath: filepath.Join(dir, "nodes.db"), KVStorePath: filepath.Join(dir, "memory.bolt"),
	}}))
	t.Cleanup(func() { require.NoError(t, store.Close(ctx)) })
	for i, pair := range [][2]string{{"global", "shared"}, {"actor", "beta"}, {"actor", "alpha"}} {
		require.NoError(t, store.RegisterAgent(ctx, &types.AgentNode{ID: pair[1]}))
		require.NoError(t, store.StoreEvent(ctx, &types.MemoryChangeEvent{
			ID: pair[1], Scope: pair[0], ScopeID: pair[1], Key: pair[1], Timestamp: time.Now().Add(time.Duration(i) * time.Second),
		}))
	}
	r := gin.New()
	r.Use(middleware.MemoryPermissionMiddleware(nil, store, nil, middleware.MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
	r.GET("/history", GetEventHistoryHandler(store))
	for _, caller := range []string{"", "beta"} {
		for _, id := range []string{"", "*", "shared", "alpha"} {
			t.Run(caller+"/"+id, func(t *testing.T) {
				req := httptest.NewRequest("GET", "/history?scope=*&scope_id="+id, nil)
				req.Header.Set("X-Agent-Node-ID", caller)
				w := httptest.NewRecorder()
				r.ServeHTTP(w, req)
				require.Equal(t, 200, w.Code, w.Body.String())
				var events []*types.MemoryChangeEvent
				require.NoError(t, json.Unmarshal(w.Body.Bytes(), &events))
				want := 1
				if id == "alpha" {
					want = 0
				} else if caller == "beta" && (id == "" || id == "*") {
					want = 2
				}
				require.Len(t, events, want)
				for _, event := range events {
					require.NotEqual(t, "alpha", event.ScopeID)
				}
			})
		}
	}
}

// C4 extension: exercise the actual SQLite registry, BoltDB keys, cache and
// prefix-based list through API-key auth, memory middleware and real handlers.
func TestMemoryActorScopeAliasesWithLocalStorage(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store := storage.NewLocalStorage(storage.LocalStorageConfig{})
	require.NoError(t, store.Initialize(ctx, storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(dir, "agentfield.db"),
			KVStorePath:  filepath.Join(dir, "agentfield.bolt"),
		},
	}))
	t.Cleanup(func() { require.NoError(t, store.Close(ctx)) })
	for _, id := range []string{"agent-alpha", "agent-beta"} {
		require.NoError(t, store.RegisterAgent(ctx, &types.AgentNode{ID: id}))
	}
	_, err := store.GetAgent(ctx, "agent-alpha:notes")
	require.EqualError(t, err, "agent node with ID 'agent-alpha:notes' not found")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
	r.Use(middleware.MemoryPermissionMiddleware(nil, store, nil,
		middleware.MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
	r.POST("/api/v1/memory/set", SetMemoryHandler(store))
	r.POST("/api/v1/memory/get", GetMemoryHandler(store))
	r.POST("/api/v1/memory/delete", DeleteMemoryHandler(store))
	r.GET("/api/v1/memory/list", ListMemoryHandler(store))
	events := NewMemoryEventsHandler(store)
	r.GET("/api/v1/memory/events/ws", events.WebSocketHandler)
	r.GET("/api/v1/memory/events/sse", events.SSEHandler)
	r.GET("/api/v1/memory/events/history", GetEventHistoryHandler(store))
	do := func(method, path, body, caller, actor string) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, "/api/v1/memory"+path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-API-Key", "shared-key")
		req.Header.Set("X-Agent-Node-ID", caller)
		req.Header.Set("X-Actor-ID", actor)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		return w
	}
	set := do("POST", "/set", `{"key":"notes:private","data":"ALPHA_PRIVATE_MARKER","scope":"actor"}`, "agent-alpha", "agent-alpha")
	require.Equal(t, 200, set.Code, set.Body.String())
	// Pin the storage condition being defended: the alias really resolves to
	// alpha's record, so an in-memory fake with separate tuple keys won't suffice.
	alias, err := store.GetMemory(ctx, "actor", "agent-alpha:notes", "private")
	require.NoError(t, err)
	require.JSONEq(t, `"ALPHA_PRIVATE_MARKER"`, string(alias.Data))
	listed, err := store.ListMemory(ctx, "actor", "agent-alpha:notes")
	require.NoError(t, err)
	require.Len(t, listed, 1)
	assertUnchanged := func(t *testing.T) {
		t.Helper()
		memory, err := store.GetMemory(ctx, "actor", "agent-alpha", "notes:private")
		require.NoError(t, err)
		require.JSONEq(t, `"ALPHA_PRIVATE_MARKER"`, string(memory.Data))
		// ListMemory reads BoltDB directly, independently of the cache.
		memories, err := store.ListMemory(ctx, "actor", "agent-alpha")
		require.NoError(t, err)
		require.Len(t, memories, 1)
		require.JSONEq(t, `"ALPHA_PRIVATE_MARKER"`, string(memories[0].Data))
		history, err := store.GetEventHistory(ctx, types.EventFilter{})
		require.NoError(t, err)
		require.Len(t, history, 1, "denied requests must not persist events")
	}
	for _, tc := range []struct{ method, path, body string }{
		{"POST", "/get", `{"key":"private","scope":"actor"}`},
		{"GET", "/list?scope=actor", ""},
		{"POST", "/set", `{"key":"private","data":"BETA_OVERWRITE","scope":"actor"}`},
		{"POST", "/delete", `{"key":"private","scope":"actor"}`},
		{"GET", "/events/ws?scope=actor&scope_id=agent-alpha:notes", ""},
		{"GET", "/events/sse?scope=actor&scope_id=agent-alpha:notes", ""},
		{"GET", "/events/history?scope=actor&scope_id=agent-alpha:notes", ""},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			w := do(tc.method, tc.path, tc.body, "agent-beta", "agent-alpha:notes")
			require.Equal(t, 403, w.Code, w.Body.String())
			require.Contains(t, w.Body.String(), "scope_ownership_denied")
			require.NotContains(t, w.Body.String(), "ALPHA_PRIVATE_MARKER")
			assertUnchanged(t)
		})
	}
	// A hierarchical read silently skips the denied actor candidate.
	w := do("POST", "/get", `{"key":"private"}`, "agent-beta", "agent-alpha:notes")
	require.Equal(t, 404, w.Code, w.Body.String())
	require.NotContains(t, w.Body.String(), "ALPHA_PRIVATE_MARKER")
	assertUnchanged(t)
	// The owner's alias and user/persona IDs with colons remain valid selectors.
	w = do("POST", "/get", `{"key":"private","scope":"actor"}`, "agent-alpha", "agent-alpha:notes")
	require.Equal(t, 200, w.Code, w.Body.String())
	require.Contains(t, w.Body.String(), "ALPHA_PRIVATE_MARKER")
	for _, actor := range []string{"", "user:123", "tenant:acme:user:7", "agent-beta:anything"} {
		t.Run(actor, func(t *testing.T) {
			w := do("POST", "/set", `{"key":"allowed","data":"PERSONA_MARKER","scope":"actor"}`, "agent-beta", actor)
			require.Equal(t, 200, w.Code, w.Body.String())
			for _, request := range []struct{ method, path, body string }{
				{"POST", "/get", `{"key":"allowed","scope":"actor"}`},
				{"GET", "/list?scope=actor", ""},
			} {
				w = do(request.method, request.path, request.body, "agent-beta", actor)
				require.Equal(t, 200, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "PERSONA_MARKER")
			}
			w = do("POST", "/delete", `{"key":"allowed","scope":"actor"}`, "agent-beta", actor)
			require.Equal(t, 204, w.Code, w.Body.String())
		})
	}
	for _, caller := range []string{"agent-alpha", "agent-beta"} {
		w = do("GET", "/list?scope=actor", "", caller, caller+strings.Repeat(":x", 17))
		require.Equal(t, 403, w.Code, fmt.Sprintf("caller=%s: %s", caller, w.Body.String()))
	}
	// SPEC-6: both alias directions and own-ID selectors against real registry,
	// cache and BoltDB. Seed directly because related owners must also be denied.
	for _, tc := range []struct {
		name, owner, caller, actor, storedKey, aliasKey string
		registerTeam                                    bool
	}{
		{"registered child", "team:alpha", "beta", "team", "secret", "alpha:secret", false},
		{"registered parent", "team", "beta", "team:alpha", "alpha:secret", "secret", true},
		{"own parent ID", "team:alpha", "team", "team", "secret", "alpha:secret", true},
		{"own child ID", "team", "team:alpha", "team:alpha", "alpha:secret", "secret", true},
		// SPEC-7: BoltDB's ':alpha:secret' also aliases an empty actor ID.
		{"empty selector", ":alpha", "beta", "", "secret", "alpha:secret", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			local := storage.NewLocalStorage(storage.LocalStorageConfig{})
			require.NoError(t, local.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: storage.LocalStorageConfig{
				DatabasePath: filepath.Join(dir, "nodes.db"), KVStorePath: filepath.Join(dir, "memory.bolt"),
			}}))
			t.Cleanup(func() { require.NoError(t, local.Close(ctx)) })
			ids := []string{"team:alpha", "beta"}
			if tc.actor == "" {
				ids = append(ids, tc.owner)
			}
			if tc.registerTeam {
				ids = append(ids, "team")
			}
			for _, id := range ids {
				require.NoError(t, local.RegisterAgent(ctx, &types.AgentNode{ID: id,
					HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline}))
			}
			nodes, err := local.ListAgents(ctx, types.AgentFilters{})
			require.NoError(t, err)
			require.Len(t, nodes, len(ids), "unfiltered snapshot includes inactive/offline nodes")
			if !tc.registerTeam {
				_, err := local.GetAgent(ctx, "team")
				require.EqualError(t, err, "agent node with ID 'team' not found")
			}
			require.NoError(t, local.SetMemory(ctx, &types.Memory{Scope: "actor", ScopeID: tc.owner, Key: tc.storedKey, Data: []byte(`"COLON_PRIVATE_MARKER"`)}))
			aliased, err := local.GetMemory(ctx, "actor", tc.actor, tc.aliasKey)
			require.NoError(t, err)
			require.JSONEq(t, `"COLON_PRIVATE_MARKER"`, string(aliased.Data))
			aliasedList, err := local.ListMemory(ctx, "actor", tc.actor)
			require.NoError(t, err)
			require.Len(t, aliasedList, 1)

			r := gin.New()
			r.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "shared-key"}))
			r.Use(middleware.MemoryPermissionMiddleware(nil, local, nil, middleware.MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
			r.POST("/get", GetMemoryHandler(local))
			r.GET("/list", ListMemoryHandler(local))
			r.POST("/set", SetMemoryHandler(local))
			r.POST("/delete", DeleteMemoryHandler(local))
			for _, request := range []struct{ method, path, body string }{
				{"POST", "/get", fmt.Sprintf(`{"key":%q,"scope":"actor"}`, tc.aliasKey)},
				{"GET", "/list?scope=actor", ""},
				{"POST", "/set", fmt.Sprintf(`{"key":%q,"scope":"actor","data":"OVERWRITE"}`, tc.aliasKey)},
				{"POST", "/delete", fmt.Sprintf(`{"key":%q,"scope":"actor"}`, tc.aliasKey)},
			} {
				req := httptest.NewRequest(request.method, request.path, strings.NewReader(request.body))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("X-API-Key", "shared-key")
				req.Header.Set("X-Agent-Node-ID", tc.caller)
				req.Header.Set("X-Actor-ID", tc.actor)
				w := httptest.NewRecorder()
				r.ServeHTTP(w, req)
				require.Equal(t, 403, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "scope_ownership_denied")
				require.NotContains(t, w.Body.String(), "COLON_PRIVATE_MARKER")
				original, err := local.GetMemory(ctx, "actor", tc.owner, tc.storedKey)
				require.NoError(t, err)
				require.JSONEq(t, `"COLON_PRIVATE_MARKER"`, string(original.Data))
				originalList, err := local.ListMemory(ctx, "actor", tc.owner)
				require.NoError(t, err)
				require.Len(t, originalList, 1)
				require.JSONEq(t, `"COLON_PRIVATE_MARKER"`, string(originalList[0].Data))
				history, err := local.GetEventHistory(ctx, types.EventFilter{})
				require.NoError(t, err)
				require.Empty(t, history, "denied writes/deletes must not store events")
			}
		})
	}

}
