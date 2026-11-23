package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// fakeLink is a link carrier that answers as a linked node would.
type fakeLink struct {
	mu      sync.Mutex
	node    string
	paths   []string
	auth    []string
	watched []string
}

func (f *fakeLink) Live(node string) bool { return node == f.node }

func (f *fakeLink) RoundTripLink(node string, req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	f.paths = append(f.paths, node+" "+req.Method+" "+req.URL.Path)
	f.auth = append(f.auth, req.Header.Get("Authorization"))
	f.watched = append(f.watched, req.Header.Get(RunWatchedHeader))
	f.mu.Unlock()
	body := `{"answer":"over the link"}`
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body)), Request: req}, nil
}

type ownerRecordingStore struct {
	*testExecutionStorage
	mu           sync.Mutex
	owners       map[string]string
	ownerLookups int
}

func (s *ownerRecordingStore) SetRunOwner(_ context.Context, runID, owner string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.owners[runID] == "" {
		s.owners[runID] = owner
	}
	return nil
}

func (s *ownerRecordingStore) RunOwner(_ context.Context, runID string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ownerLookups++
	return s.owners[runID], nil
}

// A DIRECT EXECUTE REACHES A NODE THAT HAS NO base_url DOWN ITS LINK — even
// when its last HTTP health probe said it was inactive, because a probe
// cannot reach a node behind NAT — and the node that started the run is
// recorded as its owner.
func TestExecuteReachesALinkOnlyNodeOverTheLink(t *testing.T) {
	gin.SetMode(gin.TestMode)
	link := &fakeLink{node: "worker"}
	agenturl.SetLinkCarrier(link)
	t.Cleanup(func() { agenturl.SetLinkCarrier(nil) })

	agent := &types.AgentNode{ID: "worker", BaseURL: "", HealthStatus: types.HealthStatusInactive,
		Reasoners: []types.ReasonerDefinition{{ID: "build"}}}
	store := &ownerRecordingStore{testExecutionStorage: newTestExecutionStorage(agent), owners: map[string]string{}}
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 30*time.Second, "itok"))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/worker.build", bytes.NewReader([]byte(`{"input":{"job":62}}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(RunOwnerHeader, "codeaf-factory")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	var envelope ExecuteResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &envelope))
	require.Equal(t, types.ExecutionStatusSucceeded, envelope.Status)
	require.Equal(t, map[string]any{"answer": "over the link"}, envelope.Result)

	link.mu.Lock()
	require.Equal(t, []string{"worker POST /reasoners/build"}, link.paths)
	require.Equal(t, []string{"Bearer itok"}, link.auth)
	link.mu.Unlock()
	require.Equal(t, "codeaf-factory", store.owners[envelope.RunID])
}

// A NODE WITH AN ADDRESS KEEPS HTTP: the link carrier is installed, but this
// node registered a base_url, so the execute dials it exactly as before.
func TestExecuteKeepsHTTPForANodeWithAnAddress(t *testing.T) {
	gin.SetMode(gin.TestMode)
	link := &fakeLink{node: "someone-else"}
	agenturl.SetLinkCarrier(link)
	t.Cleanup(func() { agenturl.SetLinkCarrier(nil) })

	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":"over http"}`))
	}))
	defer srv.Close()
	agent := &types.AgentNode{ID: "node-1", BaseURL: srv.URL, Reasoners: []types.ReasonerDefinition{{ID: "build"}}}
	store := newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 30*time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.build", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	require.Equal(t, 1, hits)
	require.Empty(t, link.paths)
}

// Contract C5: EVERY EXECUTION DELIVERY TO A NODE SETS X-AgentField-Run-Watched: 1
// IFF THE RUN HAS AN OWNER, NEVER FORWARDS X-AgentField-Run-Owner, PROPAGATES TO
// DESCENDANTS OF AN OWNED RUN, AND CACHES ONLY POSITIVE OWNER ANSWERS.
func TestExecuteSetsRunWatchedOnlyWhenRunHasOwner(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type seenHeaders struct {
		watched  []string
		runOwner []string
	}
	var mu sync.Mutex
	var deliveries []seenHeaders
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		deliveries = append(deliveries, seenHeaders{
			watched:  append([]string(nil), r.Header.Values(RunWatchedHeader)...),
			runOwner: append([]string(nil), r.Header.Values(RunOwnerHeader)...),
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":"ok"}`))
	}))
	defer srv.Close()

	link := &fakeLink{node: "link-worker"}
	agenturl.SetLinkCarrier(link)
	t.Cleanup(func() { agenturl.SetLinkCarrier(nil) })

	httpNode := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   srv.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "build"}},
		Skills:    []types.SkillDefinition{{ID: "lint"}},
	}
	linkNode := &types.AgentNode{
		ID:           "link-worker",
		BaseURL:      "",
		HealthStatus: types.HealthStatusInactive,
		Reasoners:    []types.ReasonerDefinition{{ID: "step"}},
	}
	store := &ownerRecordingStore{
		testExecutionStorage: newTestExecutionStorage(httpNode),
		owners:               map[string]string{},
	}
	linkStore := &ownerRecordingStore{
		testExecutionStorage: newTestExecutionStorage(linkNode),
		owners:               map[string]string{},
	}
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 30*time.Second, ""))
	linkRouter := gin.New()
	linkRouter.POST("/api/v1/execute/:target", ExecuteHandler(linkStore, services.NewFilePayloadStore(t.TempDir()), nil, 30*time.Second, ""))

	postExecuteOn := func(r *gin.Engine, target string, hdrs map[string]string) ExecuteResponse {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/"+target, strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		for k, v := range hdrs {
			req.Header.Set(k, v)
		}
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
		var resp ExecuteResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		return resp
	}
	postExecute := func(target string, hdrs map[string]string) ExecuteResponse {
		return postExecuteOn(router, target, hdrs)
	}

	// 1. Root execute WITH X-AgentField-Run-Owner sets X-AgentField-Run-Watched: 1
	//    and never echoes X-AgentField-Run-Owner to the node.
	ownedRoot := postExecute("node-1.build", map[string]string{RunOwnerHeader: "codeaf-factory"})
	mu.Lock()
	require.Len(t, deliveries, 1)
	require.Equal(t, []string{"1"}, deliveries[0].watched)
	require.Empty(t, deliveries[0].runOwner, "X-AgentField-Run-Owner must never be forwarded to nodes")
	lookupsAfterOwnedRoot := store.ownerLookups
	mu.Unlock()
	require.Equal(t, 1, lookupsAfterOwnedRoot)

	// 2. Root execute WITHOUT X-AgentField-Run-Owner carries no watched header.
	unownedRoot := postExecute("node-1.build", nil)
	mu.Lock()
	require.Len(t, deliveries, 2)
	require.Empty(t, deliveries[1].watched, "an unowned run must not receive X-AgentField-Run-Watched")
	require.Empty(t, deliveries[1].runOwner)
	lookupsAfterUnownedRoot := store.ownerLookups
	mu.Unlock()
	require.Equal(t, 2, lookupsAfterUnownedRoot)

	// 3. Nested execute (with parent) inside the owned run carries no
	//    X-AgentField-Run-Owner on the request yet still delivers
	//    X-AgentField-Run-Watched: 1, hitting the positive owner cache.
	_ = postExecute("node-1.build", map[string]string{
		"X-Run-ID":              ownedRoot.RunID,
		"X-Parent-Execution-ID": ownedRoot.ExecutionID,
	})
	mu.Lock()
	require.Len(t, deliveries, 3)
	require.Equal(t, []string{"1"}, deliveries[2].watched)
	require.Empty(t, deliveries[2].runOwner)
	lookupsAfterOwnedChild := store.ownerLookups
	mu.Unlock()
	require.Equal(t, 2, lookupsAfterOwnedChild, "positive RunOwner answers are cached in memory")

	// 4. Nested execute inside the unowned run queries RunOwner again (negative
	//    answers stay uncached), and link deliveries obey the same rule.
	_ = postExecute("node-1.build", map[string]string{
		"X-Run-ID":              unownedRoot.RunID,
		"X-Parent-Execution-ID": unownedRoot.ExecutionID,
	})
	mu.Lock()
	require.Len(t, deliveries, 4)
	require.Empty(t, deliveries[3].watched)
	mu.Unlock()
	store.mu.Lock()
	require.Equal(t, 3, store.ownerLookups, "negative RunOwner answers stay uncached")
	store.mu.Unlock()

	require.NoError(t, linkStore.SetRunOwner(context.Background(), ownedRoot.RunID, "codeaf-factory"))
	_ = postExecuteOn(linkRouter, "link-worker.step", map[string]string{
		"X-Run-ID":              ownedRoot.RunID,
		"X-Parent-Execution-ID": ownedRoot.ExecutionID,
	})
	_ = postExecuteOn(linkRouter, "link-worker.step", map[string]string{
		"X-Run-ID":              unownedRoot.RunID,
		"X-Parent-Execution-ID": unownedRoot.ExecutionID,
	})
	link.mu.Lock()
	require.Equal(t, []string{"1", ""}, link.watched)
	link.mu.Unlock()

	// 5. Legacy /reasoners and /skills routes on a real LocalStorage also set
	//    X-AgentField-Run-Watched: 1 iff the workflow/run has an owner.
	dir := t.TempDir()
	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(context.Background(), storage.StorageConfig{
		Mode:  "local",
		Local: storage.LocalStorageConfig{DatabasePath: dir + "/cp.db", KVStorePath: dir + "/cp.bolt"},
	})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		return
	}
	require.NoError(t, err)
	t.Cleanup(func() { _ = ls.Close(context.Background()) })
	httpNode.LastHeartbeat = time.Now().UTC()
	httpNode.HealthStatus = types.HealthStatusActive
	require.NoError(t, ls.RegisterAgent(context.Background(), httpNode))
	require.NoError(t, ls.SetRunOwner(context.Background(), "wf_owned", "codeaf-factory"))

	legacy := gin.New()
	legacy.POST("/api/v1/reasoners/:reasoner_id", ExecuteReasonerHandler(ls))
	legacy.POST("/api/v1/skills/:skill_id", ExecuteSkillHandler(ls))

	for _, tc := range []struct {
		path      string
		workflow  string
		wantWatch []string
	}{
		{path: "/api/v1/reasoners/node-1.build", workflow: "wf_owned", wantWatch: []string{"1"}},
		{path: "/api/v1/reasoners/node-1.build", workflow: "wf_unowned", wantWatch: nil},
		{path: "/api/v1/skills/node-1.lint", workflow: "wf_owned", wantWatch: []string{"1"}},
		{path: "/api/v1/skills/node-1.lint", workflow: "wf_unowned", wantWatch: nil},
	} {
		mu.Lock()
		before := len(deliveries)
		mu.Unlock()
		req := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewReader([]byte(`{"input":{}}`)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Workflow-ID", tc.workflow)
		rec := httptest.NewRecorder()
		legacy.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
		mu.Lock()
		require.Len(t, deliveries, before+1)
		require.Equal(t, tc.wantWatch, deliveries[before].watched, "%s (%s)", tc.path, tc.workflow)
		require.Empty(t, deliveries[before].runOwner)
		mu.Unlock()
	}
}
