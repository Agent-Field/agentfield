package middleware

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

type memoryAgentLookup func(context.Context, string) (*types.AgentNode, error)

func (f memoryAgentLookup) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	return f(ctx, id)
}

type memoryExecutionLookup func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)

func (f memoryExecutionLookup) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return f(ctx, filter)
}

type memoryPolicyCheck func([]string, []string, string, map[string]any) *types.PolicyEvaluationResult

func (f memoryPolicyCheck) EvaluateAccess(caller, target []string, op string, input map[string]any) *types.PolicyEvaluationResult {
	return f(caller, target, op, input)
}

type memoryAgentList func(context.Context, types.AgentFilters) ([]*types.AgentNode, error)

func (f memoryAgentList) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	return f(ctx, filters)
}

func (f memoryAgentList) GetAgent(context.Context, string) (*types.AgentNode, error) {
	return nil, nil
}

func memoryRegistry(ids ...string) *memoryAgentSnapshot {
	return newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
		var nodes []*types.AgentNode
		for _, id := range ids {
			nodes = append(nodes, &types.AgentNode{ID: id})
		}
		return nodes, nil
	}))
}

// MemoryRegistryClockForTest injects a clock into the real middleware for the
// external WS/SSE regressions, without adding a production configuration knob.
func MemoryRegistryClockForTest(now func() time.Time) gin.HandlerFunc {
	return func(c *gin.Context) {
		getMemoryAuthorizer(c).registry.now = now
		c.Next()
	}
}

// C1/C2/C5 / SPEC-6: identity and scope rules with fail-closed snapshots.
func TestMemoryAuthorizerScopeRules(t *testing.T) {
	for _, tc := range []struct {
		scope, id, caller                             string
		skipOwnership, missingRegistry, registryError bool
		code                                          string
	}{
		{scope: "global"},
		{scope: "actor", id: "user-123", code: "memory_identity_required"},
		{scope: "session", id: "s", code: "memory_identity_required"},
		{scope: "workflow", id: "r", code: "memory_identity_required"},
		{scope: "actor", code: "memory_identity_required"},
		{scope: "session", code: "memory_identity_required"},
		{scope: "workflow", code: "memory_identity_required"},
		{scope: "actor", caller: "beta", id: "beta"},
		{scope: "actor", caller: "beta"},
		{scope: "actor", caller: "beta", missingRegistry: true, code: "scope_ownership_denied"},
		{scope: "actor", caller: "beta", registryError: true, code: "scope_ownership_denied"},
		{scope: "actor", caller: "beta", id: "user-123"},
		{scope: "actor", caller: "beta", id: "alpha", code: "scope_ownership_denied"},
		{scope: "actor", caller: "beta", id: "user-123", registryError: true, code: "scope_ownership_denied"},
		{scope: "actor", caller: "beta", id: "beta", registryError: true, code: "scope_ownership_denied"},
		{scope: "actor", caller: "beta", id: "user-123", missingRegistry: true, code: "scope_ownership_denied"},
		{scope: "session", caller: "beta", id: "s"},
		{scope: "workflow", caller: "beta", id: "r"},
		{scope: "session", caller: "beta"},
		{scope: "workflow", caller: "beta"},
		{scope: "actor", caller: "beta", id: "alpha", skipOwnership: true},
		{scope: "session", caller: "beta", skipOwnership: true},
		{scope: "workflow", caller: "beta", skipOwnership: true},
		{scope: "actor", id: "alpha", skipOwnership: true, code: "memory_identity_required"},
		{scope: "Actor", caller: "beta", code: "scope_ownership_denied"},
		{scope: "agent", caller: "beta", code: "scope_ownership_denied"},
		{scope: " actor", caller: "beta", code: "scope_ownership_denied"},
		{scope: "", code: "scope_ownership_denied"},
		{scope: "unknown", skipOwnership: true, code: "scope_ownership_denied"},
	} {
		t.Run(fmt.Sprintf("%s/%s/caller=%s/skip=%v/missing=%v/error=%v", tc.scope, tc.id, tc.caller, tc.skipOwnership, tc.missingRegistry, tc.registryError), func(t *testing.T) {
			a := &memoryAuthorizer{callerID: tc.caller, registry: memoryRegistry("alpha", "beta"), config: MemoryPermissionConfig{EnforceScopeOwnership: !tc.skipOwnership}}
			if tc.missingRegistry {
				a.registry = nil
			}
			if tc.registryError {
				a.registry = newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
					return nil, errors.New("registry unavailable")
				}))
			}
			denial := a.authorize(context.Background(), tc.scope, tc.id)
			if tc.code == "" {
				require.Nil(t, denial)
			} else {
				require.NotNil(t, denial)
				require.Equal(t, tc.code, denial.Error)
				if tc.scope == "Actor" || tc.scope == "agent" || tc.scope == " actor" || tc.scope == "" || tc.scope == "unknown" {
					require.Contains(t, denial.Message, "Unknown")
				}
			}
		})
	}
}

// C4 / SPEC-6: colon-related node IDs in either direction, including own IDs.
func TestMemoryAuthorizerActorPrefixes(t *testing.T) {
	for _, tc := range []struct {
		name, caller, id string
		registered       []string
		denied           bool
	}{
		{name: "foreign prefix", id: "alpha:notes", registered: []string{"alpha"}, denied: true},
		{name: "nested prefix", id: "alpha:notes:private", registered: []string{"alpha"}, denied: true},
		{name: "full registered ID", id: "team:alpha", registered: []string{"team:alpha"}, denied: true},
		{name: "registered child", id: "team", registered: []string{"team:alpha"}, denied: true},
		{name: "own ID with foreign child", caller: "team", id: "team", registered: []string{"team", "team:alpha"}, denied: true},
		{name: "own ID with foreign parent", caller: "team:alpha", id: "team:alpha", registered: []string{"team", "team:alpha"}, denied: true},
		{name: "own colon ID", caller: "team:alpha", id: "team:alpha", registered: []string{"team:alpha"}},
		{name: "own prefix", caller: "beta", id: "beta:anything", registered: []string{"beta"}},
		{name: "persona", id: "user:123"},
		{name: "nested persona", id: "tenant:acme:user:7"},
		{name: "non separator prefix", id: "team", registered: []string{"teammate"}},
		{name: "empty prefix", id: ":user", registered: []string{""}, denied: true},
		// SPEC-7: empty selectors must still check exact and colon relations.
		{name: "empty selector", registered: []string{":user"}, denied: true},
		{name: "empty selector unrelated nodes", registered: []string{"alpha", "beta"}},
		{name: "empty selector registered empty ID", registered: []string{""}, denied: true},
		{name: "empty selector own child", caller: ":alpha", registered: []string{":alpha", ":alpha"}},
		{name: "empty selector foreign child after caller", caller: ":alpha", registered: []string{":beta", ":alpha", ":alpha"}, denied: true},
		{name: "foreign child after caller", caller: "team:alpha", id: "team", registered: []string{"team:beta", "team:alpha", "team:alpha"}, denied: true},
		{name: "only own child", caller: "team:alpha", id: "team", registered: []string{"team:alpha", "team:alpha", "teammate"}},
		{name: "nested own prefix with foreign parent", caller: "team:alpha", id: "team:alpha:user", registered: []string{"team:alpha", "team"}, denied: true},
		{name: "at limit", id: strings.Repeat("user:", 16) + "7"},
		{name: "over limit", id: strings.Repeat("user:", 17) + "7", denied: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caller := tc.caller
			if caller == "" {
				caller = "beta"
			}
			a := &memoryAuthorizer{callerID: caller, registry: memoryRegistry(tc.registered...), config: MemoryPermissionConfig{EnforceScopeOwnership: true}}
			// Oversized selectors need no registry access.
			if tc.name == "over limit" {
				a.registry = nil
			}
			denial := a.authorize(context.Background(), "actor", tc.id)
			if tc.denied {
				require.NotNil(t, denial)
				require.Equal(t, "scope_ownership_denied", denial.Error)
			} else {
				require.Nil(t, denial)
			}
		})
	}
}

// SPEC-7: the shared snapshot deduplicates and sorts once per refresh; an old
// index stays immutable while another request refreshes the registry.
func TestMemoryActorRegistrySnapshotIndex(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	nodes := []*types.AgentNode{{ID: "z"}, nil, {ID: "a"}, {ID: "z"}, {ID: ""}, {ID: ":alpha"}}
	registry := newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
		return nodes, nil
	}))
	registry.now = func() time.Time { return now }
	index, err := registry.agentIDs(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"", ":alpha", "a", "z"}, index.ids)
	require.Len(t, index.set, 4)
	for _, id := range index.ids {
		require.Contains(t, index.set, id)
	}
	nodes = []*types.AgentNode{{ID: "new"}}
	now = now.Add(memoryRegistryTTL)
	fresh, err := registry.agentIDs(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"new"}, fresh.ids)
	require.Equal(t, []string{"", ":alpha", "a", "z"}, index.ids)
	require.Len(t, index.set, 4)
}

// SPEC-7: a concrete subscription retains bearer semantics but drops actor
// events after registration or refresh failure, using an injected clock.
func TestMemoryConcreteEventFilterRechecksRegistry(t *testing.T) {
	gin.SetMode(gin.TestMode)
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	var nodes []*types.AgentNode
	var lookupErr error
	registry := newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
		return nodes, lookupErr
	}))
	registry.now = func() time.Time { return now }
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("GET", "/events?scope=actor&scope_id=alpha", nil)
	c.Set(memoryAuthorizerKey, &memoryAuthorizer{callerID: "beta", registry: registry, config: MemoryPermissionConfig{
		EnforceScopeOwnership: true,
		ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
			t.Fatal("concrete selectors must not query participation")
			return nil, nil
		}),
	}})
	require.True(t, RequireMemoryScope(c, c.Query("scope"), c.Query("scope_id")))
	filter := MemoryEventScopeFilter(c)
	require.NotNil(t, filter)
	ctx := context.Background()
	ev := &types.MemoryChangeEvent{Scope: "actor", ScopeID: "alpha"}
	require.True(t, filter(ctx, ev), "unregistered actor is accessible without participation")
	nodes = []*types.AgentNode{{ID: "alpha"}}
	now = now.Add(memoryRegistryTTL - time.Nanosecond)
	require.True(t, filter(ctx, ev))
	now = now.Add(time.Nanosecond)
	require.False(t, filter(ctx, ev), "the next event after expiration must be dropped")
	nodes = nil
	now = now.Add(memoryRegistryTTL)
	require.True(t, filter(ctx, ev))
	lookupErr = errors.New("registry unavailable")
	now = now.Add(memoryRegistryTTL)
	require.False(t, filter(ctx, ev))
	require.False(t, filter(ctx, nil))
	for _, scope := range []string{"global", "session", "workflow"} {
		require.True(t, filter(ctx, &types.MemoryChangeEvent{Scope: scope, ScopeID: "explicit"}))
	}
	require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "unknown"}))
	c.Set(memoryAuthorizerKey, nil)
	require.Nil(t, MemoryEventScopeFilter(c), "no middleware preserves legacy delivery")
}

// SPEC-6: short TTL, all statuses, error caching/recovery and no stale fallback.
func TestMemoryActorRegistrySnapshotRefresh(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	nodes := []*types.AgentNode{nil, {ID: "beta"}}
	var lookupErr error
	calls := 0
	registry := newMemoryAgentSnapshot(memoryAgentList(func(_ context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
		calls++
		require.Equal(t, types.AgentFilters{}, filters)
		return nodes, lookupErr
	}))
	registry.now = func() time.Time { return now }
	a := &memoryAuthorizer{callerID: "beta", registry: registry, config: MemoryPermissionConfig{EnforceScopeOwnership: true}}
	ctx := context.Background()
	require.Nil(t, a.authorize(ctx, "actor", "team"))
	nodes = append(nodes, &types.AgentNode{ID: "team:alpha", HealthStatus: types.HealthStatusInactive, LifecycleStatus: types.AgentStatusOffline})
	now = now.Add(memoryRegistryTTL - time.Nanosecond)
	require.Nil(t, a.authorize(ctx, "actor", "team"))
	require.Equal(t, 1, calls)
	now = now.Add(time.Nanosecond)
	require.NotNil(t, a.authorize(ctx, "actor", "team"), "inactive/offline nodes remain protected on refresh")
	require.Equal(t, 2, calls)
	lookupErr = errors.New("storage unavailable")
	now = now.Add(memoryRegistryTTL)
	require.NotNil(t, a.authorize(ctx, "actor", "beta"), "own ID cannot bypass refresh errors")
	require.NotNil(t, a.authorize(ctx, "actor", "user"))
	require.Equal(t, 3, calls)
	now = now.Add(time.Minute)
	require.NotNil(t, a.authorize(ctx, "actor", "user"), "never fall back to a stale successful snapshot")
	require.Equal(t, 4, calls)
	lookupErr = nil
	now = now.Add(memoryRegistryTTL)
	require.Nil(t, a.authorize(ctx, "actor", "user"))
	require.Equal(t, 5, calls)
}

// C6 / SPEC-6: cached participation cannot bypass refreshed ownership or a
// registry failure on a long-lived connection.
func TestMemoryActorEventCacheRechecksRegistry(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	var nodes []*types.AgentNode
	var lookupErr error
	registry := newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
		return nodes, lookupErr
	}))
	registry.now = func() time.Time { return now }
	participationCalls := 0
	a := &memoryAuthorizer{callerID: "beta", registry: registry, config: MemoryPermissionConfig{
		EnforceScopeOwnership: true,
		ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
			participationCalls++
			return []*types.Execution{{AgentNodeID: "beta"}}, nil
		}),
	}}
	filter := a.eventFilter(func() time.Time { return now }, false)
	ctx := context.Background()
	ev := &types.MemoryChangeEvent{Scope: "actor", ScopeID: "team"}
	require.True(t, filter(ctx, ev))
	lookupErr = errors.New("registry unavailable")
	now = now.Add(memoryRegistryTTL)
	require.False(t, filter(ctx, ev))
	lookupErr = nil
	nodes = []*types.AgentNode{{ID: "team:alpha"}}
	now = now.Add(memoryRegistryTTL)
	require.False(t, filter(ctx, ev))
	nodes = nil
	now = now.Add(memoryRegistryTTL)
	require.True(t, filter(ctx, ev))
	require.Equal(t, 1, participationCalls, "successful participation remains cached independently")
}

// SPEC-6: a middleware instance shares a single-flight snapshot across callers;
// errors coalesce too, and separate middleware instances have separate snapshots.
func TestMemoryActorRegistryMiddlewareSingleFlight(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			var calls atomic.Int32
			started, release := make(chan struct{}), make(chan struct{})
			reader := memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
				if calls.Add(1) == 1 {
					close(started)
					<-release
				}
				if fail {
					return nil, errors.New("registry unavailable")
				}
				return []*types.AgentNode{{ID: "alpha"}}, nil
			})
			newRouter := func() *gin.Engine {
				r := gin.New()
				r.Use(MemoryPermissionMiddleware(nil, reader, nil, MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
				r.GET("/actor", func(c *gin.Context) {
					if RequireMemoryScope(c, "actor", "user") {
						c.Status(204)
					}
				})
				return r
			}
			r := newRouter()
			var wg sync.WaitGroup
			for i := 0; i < 20; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					req := httptest.NewRequest("GET", "/actor", nil)
					req.Header.Set("X-Agent-Node-ID", fmt.Sprint(i))
					w := httptest.NewRecorder()
					r.ServeHTTP(w, req)
					if fail {
						require.Equal(t, 403, w.Code)
					} else {
						require.Equal(t, 204, w.Code)
					}
				}(i)
			}
			<-started
			close(release)
			wg.Wait()
			require.EqualValues(t, 1, calls.Load())
			other := newRouter()
			req := httptest.NewRequest("GET", "/actor", nil)
			req.Header.Set("X-Agent-Node-ID", "beta")
			other.ServeHTTP(httptest.NewRecorder(), req)
			require.EqualValues(t, 2, calls.Load())
		})
	}
}

// C7: evaluate every request, including global, anonymous and unregistered nodes,
// with the caller's approved tags (or an empty slice), target memory, and operation.
func TestMemoryTagPolicyAllScopesAndIdentities(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, caller := range []string{"", "beta", "unregistered"} {
		for _, scope := range []string{"global", "actor", "session", "workflow"} {
			t.Run(caller+"/"+scope, func(t *testing.T) {
				called := false
				policy := memoryPolicyCheck(func(tags, target []string, op string, input map[string]any) *types.PolicyEvaluationResult {
					called = true
					require.Equal(t, []string{"memory"}, target)
					require.Equal(t, "memory.write", op)
					require.Nil(t, input)
					if caller == "beta" {
						require.Equal(t, []string{"approved"}, tags)
					} else {
						require.Empty(t, tags)
					}
					return &types.PolicyEvaluationResult{Matched: true, Allowed: false, PolicyName: "deny-memory"}
				})
				agents := memoryAgentLookup(func(_ context.Context, id string) (*types.AgentNode, error) {
					if id == "beta" {
						return &types.AgentNode{ID: id, ApprovedTags: []string{"approved"}, ProposedTags: []string{"not-approved"}}, nil
					}
					return nil, nil
				})
				r := gin.New()
				r.Use(MemoryPermissionMiddleware(policy, agents, nil, MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
				r.POST("/api/v1/memory/set", func(*gin.Context) { t.Fatal("denied request reached handler") })
				req := httptest.NewRequest("POST", "/api/v1/memory/set", nil)
				req.Header.Set("X-Agent-Node-ID", caller)
				req.Header.Set("X-Memory-Scope", scope)
				w := httptest.NewRecorder()
				r.ServeHTTP(w, req)
				require.True(t, called)
				require.Equal(t, 403, w.Code)
				require.Contains(t, w.Body.String(), "memory_access_denied")
			})
		}
	}
}

// C7 extension: anonymous/missing callers use empty tags, but registry failures
// abort before evaluating policy or reaching any memory storage handler.
func TestMemoryTagPolicyLookupFailsClosed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct {
		name, caller string
		err          error
		status       int
	}{
		{name: "anonymous", status: 200},
		{name: "nil node", caller: "beta", status: 200},
		{name: "local not found", caller: "beta", err: errors.New("agent node with ID 'beta' not found"), status: 200},
		{name: "SQL not found", caller: "beta", err: fmt.Errorf("lookup: %w", sql.ErrNoRows), status: 200},
		{name: "storage error", caller: "beta", err: errors.New("storage down"), status: 503},
		{name: "other not found", caller: "beta", err: errors.New("database not found"), status: 503},
		{name: "wrong node not found", caller: "beta", err: errors.New("agent node with ID 'alpha' not found"), status: 503},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lookupCalled, policyCalled, handlerCalled := false, false, false
			agents := memoryAgentLookup(func(_ context.Context, id string) (*types.AgentNode, error) {
				lookupCalled = true
				require.Equal(t, tc.caller, id)
				return nil, tc.err
			})
			policy := memoryPolicyCheck(func(tags, target []string, op string, input map[string]any) *types.PolicyEvaluationResult {
				policyCalled = true
				require.Empty(t, tags)
				return &types.PolicyEvaluationResult{}
			})
			r := gin.New()
			r.Use(MemoryPermissionMiddleware(policy, agents, nil, MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
			r.GET("/api/v1/memory/list", func(c *gin.Context) { handlerCalled = true; c.Status(200) })
			req := httptest.NewRequest("GET", "/api/v1/memory/list?scope=global", nil)
			req.Header.Set("X-Agent-Node-ID", tc.caller)
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)
			require.Equal(t, tc.status, w.Code)
			require.Equal(t, tc.caller != "", lookupCalled)
			require.Equal(t, tc.status == 200, policyCalled)
			require.Equal(t, tc.status == 200, handlerCalled)
			if tc.status == 503 {
				require.Contains(t, w.Body.String(), "memory_authorization_unavailable")
				require.NotContains(t, w.Body.String(), tc.err.Error())
			}
		})
	}
}

// C8: DID identity overwrites APIKeyAuth's identity and cannot fall back to headers.
func TestMemoryMiddlewareAuthoritativeIdentity(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct{ did, callerHeader, nodeHeader, want string }{
		{"verified", " header ", " node ", "did-agent"},
		{"unresolved", "header", "node", ""},
		{"", " caller ", " node ", "caller"},
		{"", "  ", " node ", "node"},
		{"", "", "  ", ""},
	} {
		r := gin.New()
		r.Use(func(c *gin.Context) {
			c.Set(string(CallerAgentIDKey), "apikey-identity")
			c.Set(string(VerifiedCallerDIDKey), tc.did)
		})
		r.Use(MemoryPermissionMiddleware(nil, nil, &permissionDIDResolverStub{dids: map[string]string{"verified": "did-agent"}}, MemoryPermissionConfig{Enabled: true, EnforceScopeOwnership: true}))
		r.GET("/", func(c *gin.Context) {
			require.Equal(t, tc.want, GetCallerAgentID(c))
			require.Equal(t, tc.want, GetMemoryPermissionResult(c).CallerID)
			if RequireMemoryScope(c, "session", "session-1") {
				c.Status(200)
			}
		})
		req := httptest.NewRequest("GET", "/", nil)
		req.Header.Set("X-Caller-Agent-ID", tc.callerHeader)
		req.Header.Set("X-Agent-Node-ID", tc.nodeHeader)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if tc.want == "" {
			require.Equal(t, 403, w.Code)
		} else {
			require.Equal(t, 200, w.Code)
		}
	}
}

// C6 / SPEC-5: participation created after a dropped event is visible on the
// very next event; successes are cached, but absence and storage errors are not.
func TestMemoryEventFilterNewParticipation(t *testing.T) {
	for _, scope := range []string{"actor", "session", "workflow"} {
		t.Run(scope, func(t *testing.T) {
			now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
			calls := 0
			participated, lookupErr := false, false
			a := &memoryAuthorizer{callerID: "beta", registry: memoryRegistry(), config: MemoryPermissionConfig{EnforceScopeOwnership: true, ExecutionReader: memoryExecutionLookup(func(_ context.Context, f types.ExecutionFilter) ([]*types.Execution, error) {
				calls++
				require.Equal(t, "beta", *f.AgentNodeID)
				require.Equal(t, 1, f.Limit)
				require.True(t, f.ExcludePayloads)
				caller := "beta"
				want := types.ExecutionFilter{AgentNodeID: &caller, Limit: 1, ExcludePayloads: true}
				id := "scope-1"
				switch scope {
				case "actor":
					want.ActorID = &id
				case "session":
					want.SessionID = &id
				case "workflow":
					want.RunID = &id
				}
				require.Equal(t, want, f)
				if lookupErr {
					return nil, errors.New("storage down")
				}
				if participated {
					return []*types.Execution{{AgentNodeID: "beta"}}, nil
				}
				return nil, nil
			})}}
			filter := a.eventFilter(func() time.Time { return now }, false)
			ctx := context.Background()
			ev := &types.MemoryChangeEvent{Scope: scope, ScopeID: "scope-1"}
			require.False(t, filter(ctx, nil))
			require.False(t, filter(ctx, ev))
			require.Equal(t, 1, calls)
			participated = true
			require.True(t, filter(ctx, ev), "new participation must be visible without advancing the clock")
			require.Equal(t, 2, calls)
			lookupErr = true
			now = now.Add(time.Hour)
			require.True(t, filter(ctx, ev))
			require.Equal(t, 2, calls, "positive cache avoids repeat queries")
			// Connections do not share caches; transient failures are retried too.
			filter = a.eventFilter(func() time.Time { return now }, false)
			require.False(t, filter(ctx, ev))
			require.Equal(t, 3, calls)
			lookupErr = false
			require.True(t, filter(ctx, ev))
			require.Equal(t, 4, calls)
		})
	}
}

// C6 / SPEC-5: the live positive cache remains bounded. Advance the injected
// clock between distinct scopes so the lookup bucket does not mask eviction.
func TestMemoryEventFilterPositiveCacheBound(t *testing.T) {
	calls := 0
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	a := &memoryAuthorizer{callerID: "beta", config: MemoryPermissionConfig{ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
		calls++
		return []*types.Execution{{AgentNodeID: "beta"}}, nil
	})}}
	filter := a.eventFilter(func() time.Time { return now }, false)
	for pass := 0; pass < 2; pass++ {
		for i := 0; i <= memoryEventCacheLimit; i++ {
			now = now.Add(time.Second)
			require.True(t, filter(context.Background(), &types.MemoryChangeEvent{Scope: "session", ScopeID: fmt.Sprint(i)}))
		}
		if pass == 0 {
			require.Equal(t, memoryEventCacheLimit+1, calls)
		}
	}
	require.Greater(t, calls, memoryEventCacheLimit+1, "at least one positive entry must have been evicted")
}

// C6 / SPEC-5: exhaust/refill the per-connection bucket with a fake clock;
// dropped events never query storage, and cached positives need no tokens.
func TestMemoryEventFilterLookupBudget(t *testing.T) {
	original := logger.Logger
	t.Cleanup(func() { logger.Logger = original })
	var logs bytes.Buffer
	logger.Logger = zerolog.New(&logs).Level(zerolog.DebugLevel)
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	calls, participated := 0, false
	a := &memoryAuthorizer{callerID: "beta", config: MemoryPermissionConfig{ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
		calls++
		if participated {
			return []*types.Execution{{AgentNodeID: "beta"}}, nil
		}
		return nil, nil
	})}}
	filter := a.eventFilter(func() time.Time { return now }, false)
	ev := &types.MemoryChangeEvent{Scope: "workflow", ScopeID: "run-1"}
	ctx := context.Background()
	for i := 0; i < memoryEventLookupBurst; i++ {
		require.False(t, filter(ctx, ev))
	}
	require.Equal(t, memoryEventLookupBurst, calls)
	participated = true
	for i := 0; i < 10; i++ {
		require.False(t, filter(ctx, ev))
	}
	require.Equal(t, memoryEventLookupBurst, calls, "empty bucket must deny without a query")
	require.Equal(t, 1, strings.Count(logs.String(), "lookup rate limit reached"))
	require.Contains(t, logs.String(), `"level":"debug"`)
	now = now.Add(19 * time.Millisecond)
	require.False(t, filter(ctx, ev), "fractional tokens cannot fund a lookup")
	require.Equal(t, memoryEventLookupBurst, calls)
	now = now.Add(time.Millisecond)
	require.True(t, filter(ctx, ev), "one token refills after 1/50 second")
	require.Equal(t, memoryEventLookupBurst+1, calls)
	require.True(t, filter(ctx, ev), "a cached positive works with an empty bucket")
	require.Equal(t, memoryEventLookupBurst+1, calls)
	participated = false
	ev = &types.MemoryChangeEvent{Scope: "workflow", ScopeID: "run-2"}
	require.False(t, filter(ctx, ev))
	require.Equal(t, memoryEventLookupBurst+1, calls)
	require.Equal(t, 1, strings.Count(logs.String(), "lookup rate limit reached"), "logs are throttled within a second")

	// Long idle periods refill only up to the burst, and allow another log.
	now = now.Add(time.Hour)
	before := calls
	for i := 0; i < memoryEventLookupBurst+10; i++ {
		require.False(t, filter(ctx, ev))
	}
	require.Equal(t, before+memoryEventLookupBurst, calls)
	require.Equal(t, 2, strings.Count(logs.String(), "lookup rate limit reached"))
	// A second connection has an independent bucket and log throttle.
	other := a.eventFilter(func() time.Time { return now }, false)
	before = calls
	for i := 0; i < memoryEventLookupBurst+1; i++ {
		require.False(t, other(ctx, ev))
	}
	require.Equal(t, before+memoryEventLookupBurst, calls)
	require.Equal(t, 3, strings.Count(logs.String(), "lookup rate limit reached"))
}

// C6 / SPEC-5: decisions without participation lookups leave the budget intact.
// Scope-rule denials must not become cached when the registry changes either.
func TestMemoryEventFilterTrivialDecisions(t *testing.T) {
	calls, clockCalls := 0, 0
	registered := true
	a := &memoryAuthorizer{callerID: "beta", registry: newMemoryAgentSnapshot(memoryAgentList(func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
		if registered {
			return []*types.AgentNode{{ID: "alpha"}}, nil
		}
		return nil, nil
	})), config: MemoryPermissionConfig{EnforceScopeOwnership: true, ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
		calls++
		return nil, nil
	})}}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	filter := a.eventFilter(func() time.Time { clockCalls++; return now }, false)
	ctx := context.Background()
	for i := 0; i < memoryEventLookupBurst+1; i++ {
		require.True(t, filter(ctx, &types.MemoryChangeEvent{Scope: "global", ScopeID: fmt.Sprint(i)}))
		require.True(t, filter(ctx, &types.MemoryChangeEvent{Scope: "actor", ScopeID: "beta"}))
		require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "actor", ScopeID: "alpha"}))
		require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "unknown", ScopeID: "id"}))
		for _, scope := range []string{"actor", "session", "workflow"} {
			require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: scope}))
		}
	}
	require.Zero(t, calls)
	require.Zero(t, clockCalls)
	registered = false
	a.registry.checked = time.Time{} // Force the next registry refresh without sleeping.
	for i := 0; i < memoryEventLookupBurst; i++ {
		require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "actor", ScopeID: "alpha"}))
	}
	require.Equal(t, memoryEventLookupBurst, calls, "scope denials consumed no tokens and were not cached")
	a.config.ExecutionReader = nil
	filter = a.eventFilter(func() time.Time { t.Fatal("no reader means no participation lookup"); return now }, false)
	require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "workflow", ScopeID: "run-1"}))
	a.callerID = ""
	filter = a.eventFilter(func() time.Time { t.Fatal("anonymous events need no participation lookup"); return now }, false)
	require.True(t, filter(ctx, &types.MemoryChangeEvent{Scope: "global"}))
	require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "workflow", ScopeID: "run-1"}))
}

// C6 / SPEC-5: history caches negative decisions for one request, without a
// live-stream rate limit or eviction, and the next request rechecks membership.
func TestMemoryEventSnapshotFilterCache(t *testing.T) {
	calls, participated := 0, false
	a := &memoryAuthorizer{callerID: "beta", config: MemoryPermissionConfig{ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
		calls++
		if participated {
			return []*types.Execution{{AgentNodeID: "beta"}}, nil
		}
		return nil, nil
	})}}
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Set(memoryAuthorizerKey, a)
	filter := MemoryEventSnapshotFilter(c)
	ctx := context.Background()
	for i := 0; i <= memoryEventCacheLimit; i++ {
		require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "workflow", ScopeID: fmt.Sprint(i)}))
	}
	require.Equal(t, memoryEventCacheLimit+1, calls, "snapshots have no token bucket")
	participated = true
	for i := 0; i <= memoryEventCacheLimit; i++ {
		require.False(t, filter(ctx, &types.MemoryChangeEvent{Scope: "workflow", ScopeID: fmt.Sprint(i)}))
	}
	require.Equal(t, memoryEventCacheLimit+1, calls, "snapshot negatives remain cached for the whole request")
	filter = MemoryEventSnapshotFilter(c)
	ev := &types.MemoryChangeEvent{Scope: "workflow", ScopeID: "0"}
	require.True(t, filter(ctx, ev))
	require.True(t, filter(ctx, ev))
	require.Equal(t, memoryEventCacheLimit+2, calls, "new snapshots recheck membership, then cache successes")
}

// C6: empty shared buckets are readable explicitly but have no participation
// identity for wildcard events. Never query executions with an empty selector.
func TestMemoryAuthorizerWildcardEmptyScopeIDs(t *testing.T) {
	for _, caller := range []string{"", "beta"} {
		a := &memoryAuthorizer{callerID: caller, config: MemoryPermissionConfig{
			EnforceScopeOwnership: true,
			ExecutionReader: memoryExecutionLookup(func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
				t.Fatal("empty scopes must not query participation")
				return nil, nil
			}),
		}}
		filter := a.eventFilter(time.Now, false)
		for _, scope := range []string{"actor", "session", "workflow"} {
			require.False(t, filter(context.Background(), &types.MemoryChangeEvent{Scope: scope}))
		}
		require.True(t, filter(context.Background(), &types.MemoryChangeEvent{Scope: "global"}))
	}
}

// C9: neither a missing authorizer nor disabled middleware changes legacy scope access.
func TestMemoryAuthorizationAbsentOrDisabled(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, installed := range []bool{false, true} {
		r := gin.New()
		if installed {
			r.Use(MemoryPermissionMiddleware(nil, nil, nil, MemoryPermissionConfig{Enabled: false}))
		}
		r.GET("/", func(c *gin.Context) {
			require.Nil(t, MemoryEventFilter(c))
			require.Nil(t, MemoryEventSnapshotFilter(c))
			require.Nil(t, AuthorizeMemoryScope(c, "unknown", ""))
			require.True(t, RequireMemoryScope(c, "actor", "alpha"))
			c.Status(200)
		})
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest("GET", "/", nil))
		require.Equal(t, 200, w.Code)
	}
}
