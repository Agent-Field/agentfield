package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type relayRoutesRig struct {
	t      *testing.T
	ctx    context.Context
	store  *storage.LocalStorage
	pipe   *nodelink.Pipe
	router *gin.Engine
}

func newRelayRoutesRig(t *testing.T, workerBaseURL string) *relayRoutesRig {
	t.Helper()
	gin.SetMode(gin.TestMode)

	ctx := context.Background()
	dir := t.TempDir()
	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(ctx, storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(dir, "cp.db"),
			KVStorePath:  filepath.Join(dir, "cp.bolt"),
		},
	})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() { _ = ls.Close(ctx) })

	require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{
		ID:            "worker",
		Version:       "1.0.0",
		BaseURL:       workerBaseURL,
		HealthStatus:  types.HealthStatusActive,
		LastHeartbeat: time.Now().UTC(),
	}))

	pipe := &nodelink.Pipe{
		Store:         ls,
		InternalToken: "relay-itok",
		SteerWait:     2 * time.Second,
		Retention:     func() time.Duration { return time.Hour },
	}

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/relay", RelayPutHandler(pipe))
	router.GET("/api/v1/executions/:execution_id/relay", RelayReadHandler(pipe))
	router.POST("/api/v1/executions/:execution_id/relay/ack", RelayAckHandler(pipe))
	router.GET("/api/v1/relay/count", RelayCountHandler(pipe))
	router.POST("/api/v1/executions/:execution_id/steer", SteerExecutionHandler(pipe))

	return &relayRoutesRig{
		t:      t,
		ctx:    ctx,
		store:  ls,
		pipe:   pipe,
		router: router,
	}
}

func (r *relayRoutesRig) seedRunning(runID, execID, owner string) {
	r.t.Helper()
	now := time.Now().UTC()
	require.NoError(r.t, r.store.CreateExecutionRecord(r.ctx, &types.Execution{
		ExecutionID: execID,
		RunID:       runID,
		AgentNodeID: "worker",
		NodeID:      "worker",
		ReasonerID:  "code",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))
	if owner != "" {
		require.NoError(r.t, r.pipe.Own(r.ctx, runID, owner))
	}
}

func (r *relayRoutesRig) doRaw(method, path, owner, rawBody string) (int, map[string]any) {
	r.t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(rawBody))
	req.Header.Set("Content-Type", "application/json")
	if owner != "" {
		req.Header.Set(RunOwnerHeader, owner)
	}
	rec := httptest.NewRecorder()
	r.router.ServeHTTP(rec, req)
	out := map[string]any{}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

// Contract H1: POST /executions/:id/relay stores entries for an owned run and
// answers with stored and through, while refusing a malformed body or an entry
// missing seq or kind with HTTP 400.
func TestRelayRoutes_H1_PutStoresEntriesAndRefusesInvalidPayloads(t *testing.T) {
	r := newRelayRoutesRig(t, "")
	r.seedRunning("run-h1", "exec-h1", "workaf")

	code, out := r.doRaw(http.MethodPost, "/api/v1/executions/exec-h1/relay", "",
		`{"entries":[{"seq":1,"kind":"assistant","data":{"text":"starting"}},{"seq":2,"kind":"tool","data":{"cmd":"go vet"}}]}`)
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(2), out["stored"])
	require.Equal(t, float64(2), out["through"])

	// Replaying seq 2 together with a new seq 3 reports only 1 new row and through=3.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h1/relay", "",
		`{"entries":[{"seq":2,"kind":"tool","data":{"cmd":"go vet"}},{"seq":3,"kind":"done"}]}`)
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(1), out["stored"])
	require.Equal(t, float64(3), out["through"])

	// Malformed JSON, an empty entries list, a missing seq, and a blank kind all
	// fail validation with HTTP 400.
	for _, badBody := range []string{
		`{not-json`,
		`{"entries":[]}`,
		`{"entries":[{"kind":"assistant"}]}`,
		`{"entries":[{"seq":0,"kind":"assistant"}]}`,
		`{"entries":[{"seq":4,"kind":"   "}]}`,
	} {
		code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h1/relay", "", badBody)
		require.Equal(t, http.StatusBadRequest, code, "body %s should be refused with 400: %v", badBody, out)
	}
}

// Contract H2: GET /executions/:id/relay?after=N&limit=M returns held entries
// after N for the run owner named in X-AgentField-Run-Owner, refuses another
// owner with 403, and permits an empty owner header as the API-key holder per
// nodelink.Pipe.Read's scoping contract.
func TestRelayRoutes_H2_ReadScopesByRunOwnerAndAdvancesCursor(t *testing.T) {
	r := newRelayRoutesRig(t, "")
	r.seedRunning("run-h2", "exec-h2", "workaf")

	code, _ := r.doRaw(http.MethodPost, "/api/v1/executions/exec-h2/relay", "",
		`{"entries":[{"seq":1,"kind":"a"},{"seq":2,"kind":"b"},{"seq":3,"kind":"c"},{"seq":4,"kind":"d"}]}`)
	require.Equal(t, http.StatusOK, code)

	// The run's owner reads entries after seq 1 capped at limit 2 and gets next=3.
	code, out := r.doRaw(http.MethodGet, "/api/v1/executions/exec-h2/relay?after=1&limit=2", "workaf", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(3), out["next"])
	entries, ok := out["entries"].([]any)
	require.True(t, ok)
	require.Len(t, entries, 2)
	require.Equal(t, float64(2), entries[0].(map[string]any)["seq"])
	require.Equal(t, float64(3), entries[1].(map[string]any)["seq"])

	// Reading past the highest seq returns an empty slice and preserves next=after.
	code, out = r.doRaw(http.MethodGet, "/api/v1/executions/exec-h2/relay?after=4&limit=10", "workaf", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(4), out["next"])
	require.Empty(t, out["entries"])

	// Another node naming itself in X-AgentField-Run-Owner is refused with 403.
	code, out = r.doRaw(http.MethodGet, "/api/v1/executions/exec-h2/relay?after=0&limit=10", "other-owner", "")
	require.Equal(t, http.StatusForbidden, code, out)

	// No X-AgentField-Run-Owner header speaks as the trusted API-key holder in
	// Pipe.scope and therefore reads the held entries.
	code, out = r.doRaw(http.MethodGet, "/api/v1/executions/exec-h2/relay?after=2&limit=10", "", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(4), out["next"])
	require.Len(t, out["entries"], 2)
}

// Contract H3: POST /executions/:id/relay/ack {through} drops the entries the
// run's owner acknowledged and refuses non-owners or invalid through cursors.
func TestRelayRoutes_H3_AckDropsEntriesForRunOwner(t *testing.T) {
	r := newRelayRoutesRig(t, "")
	r.seedRunning("run-h3", "exec-h3", "workaf")

	code, _ := r.doRaw(http.MethodPost, "/api/v1/executions/exec-h3/relay", "",
		`{"entries":[{"seq":1,"kind":"a"},{"seq":2,"kind":"b"},{"seq":3,"kind":"c"}]}`)
	require.Equal(t, http.StatusOK, code)

	// Another owner cannot drop this run's buffered transcript lines.
	code, out := r.doRaw(http.MethodPost, "/api/v1/executions/exec-h3/relay/ack", "intruder", `{"through":2}`)
	require.Equal(t, http.StatusForbidden, code, out)

	// A missing or zero through cursor is refused with 400.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h3/relay/ack", "workaf", `{"through":0}`)
	require.Equal(t, http.StatusBadRequest, code, out)

	// The owner acks through seq 2, dropping 2 rows and leaving seq 3 held.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h3/relay/ack", "workaf", `{"through":2}`)
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(2), out["dropped"])

	code, out = r.doRaw(http.MethodGet, "/api/v1/executions/exec-h3/relay?after=0", "workaf", "")
	require.Equal(t, http.StatusOK, code, out)
	entries := out["entries"].([]any)
	require.Len(t, entries, 1)
	require.Equal(t, float64(3), entries[0].(map[string]any)["seq"])
}

// Contract H4: GET /relay/count[?execution_id=] reports how many entries the
// pipe currently holds across all executions or for one named execution.
func TestRelayRoutes_H4_CountReportsHeldEntries(t *testing.T) {
	r := newRelayRoutesRig(t, "")
	r.seedRunning("run-h4-a", "exec-h4-a", "workaf")
	r.seedRunning("run-h4-b", "exec-h4-b", "workaf")

	code, out := r.doRaw(http.MethodGet, "/api/v1/relay/count", "", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(0), out["held"])

	code, _ = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h4-a/relay", "",
		`{"entries":[{"seq":1,"kind":"a"},{"seq":2,"kind":"b"}]}`)
	require.Equal(t, http.StatusOK, code)
	code, _ = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h4-b/relay", "",
		`{"entries":[{"seq":1,"kind":"a"},{"seq":2,"kind":"b"},{"seq":3,"kind":"c"}]}`)
	require.Equal(t, http.StatusOK, code)

	code, out = r.doRaw(http.MethodGet, "/api/v1/relay/count", "", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(5), out["held"])

	code, out = r.doRaw(http.MethodGet, "/api/v1/relay/count?execution_id=exec-h4-a", "", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(2), out["held"])

	code, out = r.doRaw(http.MethodGet, "/api/v1/relay/count?execution_id=exec-h4-b", "", "")
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, float64(3), out["held"])

	// A cancelled request context surfaces the store error as HTTP 500 via pipeRefusal.
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/relay/count", nil).WithContext(cancelledCtx)
	rec := httptest.NewRecorder()
	r.router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusInternalServerError, rec.Code)
}

// Contract H5: POST /executions/:id/steer {message} from the run's owner
// reaches the node running the execution, while an empty message or a steer
// from a non-owner is refused with the pipe's HTTP status code.
func TestRelayRoutes_H5_SteerReachesNodeAndRefusesInvalidOrUnauthorized(t *testing.T) {
	type steerHit struct {
		path    string
		auth    string
		execHdr string
		message string
	}
	var (
		mu   sync.Mutex
		hits []steerHit
	)
	nodeSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		var body struct {
			Message string `json:"message"`
		}
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		hits = append(hits, steerHit{
			path:    req.Method + " " + req.URL.Path,
			auth:    req.Header.Get("Authorization"),
			execHdr: req.Header.Get("X-Execution-ID"),
			message: body.Message,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.Copy(w, bytes.NewReader([]byte(`{"accepted":true}`)))
	}))
	defer nodeSrv.Close()

	r := newRelayRoutesRig(t, nodeSrv.URL)
	r.seedRunning("run-h5", "exec-h5", "workaf")

	// Empty or whitespace-only messages are refused by Pipe.Steer with 400.
	code, out := r.doRaw(http.MethodPost, "/api/v1/executions/exec-h5/steer", "workaf", `{"message":"   "}`)
	require.Equal(t, http.StatusBadRequest, code, out)

	// A caller that does not own the run is refused by Pipe.Steer with 403.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h5/steer", "intruder", `{"message":"take over"}`)
	require.Equal(t, http.StatusForbidden, code, out)

	// An unknown execution is refused by Pipe.Steer with 404.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-missing/steer", "workaf", `{"message":"hello"}`)
	require.Equal(t, http.StatusNotFound, code, out)

	mu.Lock()
	require.Empty(t, hits, "refused steers must not contact the worker node")
	mu.Unlock()

	// The run's owner delivers the note to the worker node and gets {ok: true, via: "http"}.
	code, out = r.doRaw(http.MethodPost, "/api/v1/executions/exec-h5/steer", "workaf", `{"message":"use batch size 500"}`)
	require.Equal(t, http.StatusOK, code, out)
	require.Equal(t, true, out["ok"])
	require.Equal(t, "http", out["via"])

	mu.Lock()
	require.Equal(t, []steerHit{{
		path:    "POST /_internal/executions/exec-h5/steer",
		auth:    "Bearer relay-itok",
		execHdr: "exec-h5",
		message: "use batch size 500",
	}}, hits)
	mu.Unlock()
}

type stubRunOwnerStore struct {
	mu      sync.Mutex
	owners  map[string]string
	err     error
	lookups int
}

func (s *stubRunOwnerStore) RunOwner(_ context.Context, runID string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lookups++
	if s.err != nil {
		return "", s.err
	}
	return s.owners[runID], nil
}

// Contract H6: lookupRunWatched returns false for an empty run id, a nil store,
// a store without RunOwner, or a store error, and returns true when the store
// names an owner while caching only positive answers in memory.
func TestRelayRoutes_H6_LookupRunWatchedCachesPositiveAnswersOnly(t *testing.T) {
	ctx := context.Background()
	cache := &sync.Map{}

	store := &stubRunOwnerStore{
		owners: map[string]string{"run-owned": "workaf"},
	}

	// Empty run ID, nil store, and a store without RunOwner all yield false
	// without querying the RunOwner store.
	require.False(t, lookupRunWatched(ctx, store, cache, ""))
	require.False(t, lookupRunWatched(ctx, store, cache, "   "))
	require.False(t, lookupRunWatched(ctx, nil, cache, "run-owned"))
	require.False(t, lookupRunWatched(ctx, struct{}{}, cache, "run-owned"))
	require.Equal(t, 0, store.lookups)

	// A store error yields false and does not populate the positive cache.
	store.err = errors.New("sqlite busy")
	require.False(t, lookupRunWatched(ctx, store, cache, "run-owned"))
	require.Equal(t, 1, store.lookups)

	// Once the store succeeds, lookupRunWatched returns true, caches the run ID,
	// and serves subsequent calls from memory without calling RunOwner again.
	store.err = nil
	require.True(t, lookupRunWatched(ctx, store, cache, "run-owned"))
	require.Equal(t, 2, store.lookups)

	store.err = errors.New("should not be reached after positive cache hit")
	require.True(t, lookupRunWatched(ctx, store, cache, "run-owned"))
	require.Equal(t, 2, store.lookups)

	// An unowned run returns false without caching negative results.
	store.err = nil
	require.False(t, lookupRunWatched(ctx, store, cache, "run-unowned"))
	require.Equal(t, 3, store.lookups)
	require.False(t, lookupRunWatched(ctx, store, cache, "run-unowned"))
	require.Equal(t, 4, store.lookups)
}
