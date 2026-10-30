package middleware

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

const (
	memoryAuthorizerKey  = "memory_scope_authorizer"
	memoryActorMaxColons = 16
	memoryRegistryTTL    = 2 * time.Second
)

// MemoryAgentLister supplies all registered nodes, including inactive/offline
// nodes. It is optional on AgentResolverInterface; without it actor checks fail closed.
type MemoryAgentLister interface {
	ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error)
}

// One snapshot is shared by every request/connection using a middleware instance.
// Holding mu during refresh coalesces concurrent requests, including failures.
// Errors are cached for the short TTL too; stale successful data is never served.
type memoryAgentSnapshot struct {
	mu      sync.Mutex
	reader  MemoryAgentLister
	now     func() time.Time
	loaded  bool
	checked time.Time
	index   *memoryAgentIndex
	err     error
}

// Published indexes are immutable so authorization can use them outside the
// snapshot lock. The set handles parents; the sorted IDs find children.
type memoryAgentIndex struct {
	ids []string
	set map[string]struct{}
}

func (index *memoryAgentIndex) hasForeignRelatedID(selector, caller string) bool {
	foreign := func(id string) bool {
		_, exists := index.set[id]
		return exists && id != caller
	}
	if foreign(selector) {
		return true
	}
	for i, char := range selector {
		if char == ':' && foreign(selector[:i]) {
			return true
		}
	}
	prefix := selector + ":"
	for i := sort.SearchStrings(index.ids, prefix); i < len(index.ids) && strings.HasPrefix(index.ids[i], prefix); i++ {
		if index.ids[i] != caller {
			return true
		}
	}
	return false
}

func newMemoryAgentSnapshot(reader MemoryAgentLister) *memoryAgentSnapshot {
	return &memoryAgentSnapshot{reader: reader, now: time.Now}
}

func (s *memoryAgentSnapshot) agentIDs(ctx context.Context) (*memoryAgentIndex, error) {
	if s == nil || s.reader == nil {
		return nil, errors.New("memory agent registry unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded && s.now().Sub(s.checked) < memoryRegistryTTL {
		return s.index, s.err
	}
	nodes, err := s.reader.ListAgents(ctx, types.AgentFilters{})
	s.index = nil
	if err == nil {
		index := &memoryAgentIndex{set: make(map[string]struct{}, len(nodes))}
		for _, node := range nodes {
			if node != nil {
				index.set[node.ID] = struct{}{}
			}
		}
		for id := range index.set {
			index.ids = append(index.ids, id)
		}
		sort.Strings(index.ids)
		s.index = index
	}
	s.checked, s.loaded, s.err = s.now(), true, err
	return s.index, s.err
}

// MemoryExecutionReader is the minimal storage contract for event participation.
type MemoryExecutionReader interface {
	QueryExecutionRecords(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
}

// MemoryAccessDenial describes an authorization failure without writing a response.
// Hierarchical reads and wildcard events can skip denied scopes quietly.
type MemoryAccessDenial struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// WriteMemoryAccessDenial returns a denial before any storage or stream access.
func WriteMemoryAccessDenial(c *gin.Context, denial *MemoryAccessDenial) {
	c.AbortWithStatusJSON(http.StatusForbidden, denial)
}

// AuthorizeMemoryScope checks the scope actually selected by a handler. Absence
// of middleware intentionally preserves the legacy behavior when auth is off.
func AuthorizeMemoryScope(c *gin.Context, scope, scopeID string) *MemoryAccessDenial {
	if authorizer := getMemoryAuthorizer(c); authorizer != nil {
		return authorizer.authorize(c.Request.Context(), scope, scopeID)
	}
	return nil
}

// RequireMemoryScope writes a 403 and aborts on denial; handlers must return when false.
func RequireMemoryScope(c *gin.Context, scope, scopeID string) bool {
	if denial := AuthorizeMemoryScope(c, scope, scopeID); denial != nil {
		WriteMemoryAccessDenial(c, denial)
		return false
	}
	return true
}

type memoryAuthorizer struct {
	callerID string
	registry *memoryAgentSnapshot
	config   MemoryPermissionConfig
}

func getMemoryAuthorizer(c *gin.Context) *memoryAuthorizer {
	value, _ := c.Get(memoryAuthorizerKey)
	authorizer, _ := value.(*memoryAuthorizer)
	return authorizer
}

func (a *memoryAuthorizer) authorize(ctx context.Context, scope, scopeID string) *MemoryAccessDenial {
	switch scope {
	case "global":
		return nil
	case "actor", "session", "workflow":
	default:
		return &MemoryAccessDenial{"scope_ownership_denied", fmt.Sprintf("Unknown memory scope %q", scope)}
	}
	if a.callerID == "" {
		return &MemoryAccessDenial{"memory_identity_required", fmt.Sprintf(
			"A caller agent identity (verified DID, X-Agent-Node-ID or X-Caller-Agent-ID) is required for %s-scoped memory when memory authorization is enabled", scope)}
	}
	if !a.config.EnforceScopeOwnership {
		return nil
	}
	if scope == "actor" {
		if strings.Count(scopeID, ":") > memoryActorMaxColons {
			return &MemoryAccessDenial{"scope_ownership_denied", "Actor memory scope ID contains too many ':' separators"}
		}
		index, err := a.registry.agentIDs(ctx)
		if err == nil {
			// Node IDs containing ':' are supported. BoltDB and the memory
			// cache concatenate scopeID and key with ':', so protect registered
			// IDs related in either direction, even when selecting our own ID.
			// This includes empty selectors, which alias nodes starting with ':'.
			if index.hasForeignRelatedID(scopeID, a.callerID) {
				return &MemoryAccessDenial{"scope_ownership_denied", "Agent does not own the requested memory scope"}
			}
			return nil
		}
	} else {
		// Released SDKs use the empty session/workflow bucket, including
		// during hierarchical reads. These buckets are shared.
		return nil
	}
	return &MemoryAccessDenial{"scope_ownership_denied", "Agent does not own the requested memory scope"}
}

// LocalStorage.GetAgent formats not-found without wrapping sql.ErrNoRows.
// Match only that exact error, never generic DB errors containing "not found".
// Some storage fakes instead return (nil, nil), handled by their callers.
func isMemoryAgentNotFound(err error, id string) bool {
	return errors.Is(err, sql.ErrNoRows) || (err != nil && err.Error() == fmt.Sprintf("agent node with ID '%s' not found", id))
}

const (
	memoryEventCacheLimit = 256
	// Each live connection can query participation 50 times/second with a
	// burst of 100. Denials stay fresh while sustained lookup load is bounded.
	memoryEventLookupRate  = 50
	memoryEventLookupBurst = 100
)

type memoryScopeKey struct{ scope, id string }

// MemoryEventScopeFilter rechecks explicit stream selectors before each event.
// Explicit selectors retain read/bearer semantics, without participation checks.
func MemoryEventScopeFilter(c *gin.Context) func(context.Context, *types.MemoryChangeEvent) bool {
	if authorizer := getMemoryAuthorizer(c); authorizer != nil {
		return func(ctx context.Context, ev *types.MemoryChangeEvent) bool {
			return ev != nil && authorizer.authorize(ctx, ev.Scope, ev.ScopeID) == nil
		}
	}
	return nil
}

// MemoryEventFilter limits wildcard streams to owned or participated scopes.
// Only positive participation decisions are cached, bounded per connection.
// A nil filter means authorization is not installed and all events are allowed.
func MemoryEventFilter(c *gin.Context) func(context.Context, *types.MemoryChangeEvent) bool {
	if authorizer := getMemoryAuthorizer(c); authorizer != nil {
		return authorizer.eventFilter(time.Now, false)
	}
	return nil
}

// MemoryEventSnapshotFilter caches participation decisions, including denials,
// only for this history request. Its lookups are bounded by distinct scopes in
// the snapshot, so it needs neither a token bucket nor cache expiration.
func MemoryEventSnapshotFilter(c *gin.Context) func(context.Context, *types.MemoryChangeEvent) bool {
	if authorizer := getMemoryAuthorizer(c); authorizer != nil {
		return authorizer.eventFilter(time.Now, true)
	}
	return nil
}

func (a *memoryAuthorizer) eventFilter(now func() time.Time, snapshot bool) func(context.Context, *types.MemoryChangeEvent) bool {
	cache := make(map[memoryScopeKey]bool)
	var bucket *rate.Limiter
	if !snapshot {
		bucket = rate.NewLimiter(memoryEventLookupRate, memoryEventLookupBurst)
	}
	var lastLog time.Time
	logged := false
	var mu sync.Mutex
	return func(ctx context.Context, ev *types.MemoryChangeEvent) bool {
		if ev == nil {
			return false
		}
		key := memoryScopeKey{ev.Scope, ev.ScopeID}
		mu.Lock()
		defer mu.Unlock()
		if a.authorize(ctx, ev.Scope, ev.ScopeID) != nil {
			return false
		}
		if allowed, ok := cache[key]; ok {
			return allowed
		}
		if ev.Scope == "global" || ev.Scope == "actor" && ev.ScopeID == a.callerID {
			return true
		}
		// An empty shared bucket has no participation identity. Wildcard
		// subscriptions must not turn it into a broadcast to every agent.
		if ev.ScopeID == "" || a.config.ExecutionReader == nil {
			return false
		}
		if bucket != nil {
			at := now()
			if !bucket.AllowN(at, 1) {
				if !logged || at.Sub(lastLog) >= time.Second {
					logger.Logger.Debug().Str("caller_agent_id", a.callerID).
						Msg("Memory event participation lookup rate limit reached; dropping event")
					lastLog, logged = at, true
				}
				return false
			}
		}
		filter := types.ExecutionFilter{AgentNodeID: &a.callerID, Limit: 1, ExcludePayloads: true}
		switch ev.Scope {
		case "actor":
			filter.ActorID = &ev.ScopeID
		case "session":
			filter.SessionID = &ev.ScopeID
		case "workflow":
			filter.RunID = &ev.ScopeID
		}
		rows, err := a.config.ExecutionReader.QueryExecutionRecords(ctx, filter)
		allowed := err == nil && len(rows) > 0
		if !snapshot && !allowed {
			return false
		}
		if !snapshot && len(cache) >= memoryEventCacheLimit {
			// Eviction affects performance only; the next event is reauthorized.
			for old := range cache {
				delete(cache, old)
				break
			}
		}
		cache[key] = allowed
		return allowed
	}
}
