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
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// fakeLink is a link carrier that answers as a linked node would.
type fakeLink struct {
	mu    sync.Mutex
	node  string
	paths []string
	auth  []string
}

func (f *fakeLink) Live(node string) bool { return node == f.node }

func (f *fakeLink) RoundTripLink(node string, req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	f.paths = append(f.paths, node+" "+req.Method+" "+req.URL.Path)
	f.auth = append(f.auth, req.Header.Get("Authorization"))
	f.mu.Unlock()
	body := `{"answer":"over the link"}`
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body)), Request: req}, nil
}

type ownerRecordingStore struct {
	*testExecutionStorage
	mu     sync.Mutex
	owners map[string]string
}

func (s *ownerRecordingStore) SetRunOwner(_ context.Context, runID, owner string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.owners[runID] == "" {
		s.owners[runID] = owner
	}
	return nil
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
