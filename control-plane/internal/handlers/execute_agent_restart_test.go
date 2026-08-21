package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// deadAddress returns a host:port that is guaranteed to refuse connections:
// the listener is bound (so the port is real) and then closed.
func deadAddress(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := ln.Addr().String()
	require.NoError(t, ln.Close())
	return addr
}

// withRestartGrace sets the package-level grace window for one test.
func withRestartGrace(t *testing.T, d time.Duration) {
	t.Helper()
	previous := agentRestartGrace()
	SetAgentRestartGrace(d)
	t.Cleanup(func() { SetAgentRestartGrace(previous) })
}

func testPlan(agent *types.AgentNode) *preparedExecution {
	return &preparedExecution{
		exec: &types.Execution{
			ExecutionID: "exec_1",
			RunID:       "run_1",
		},
		requestBody: []byte(`{"message":"hi"}`),
		agent:       agent,
		target:      &parsedTarget{NodeID: agent.ID, TargetName: "echo", TargetType: "reasoner"},
	}
}

func TestEnsureAgentDispatchable(t *testing.T) {
	t.Run("nil agent is allowed", func(t *testing.T) {
		require.NoError(t, ensureAgentDispatchable(nil))
	})

	t.Run("unknown health is allowed so the first call of a session works", func(t *testing.T) {
		require.NoError(t, ensureAgentDispatchable(&types.AgentNode{
			ID:           "demoproj",
			HealthStatus: types.HealthStatusUnknown,
		}))
	})

	t.Run("active and degraded are allowed", func(t *testing.T) {
		require.NoError(t, ensureAgentDispatchable(&types.AgentNode{ID: "a", HealthStatus: types.HealthStatusActive}))
		require.NoError(t, ensureAgentDispatchable(&types.AgentNode{ID: "a", HealthStatus: types.HealthStatusDegraded}))
	})

	t.Run("inactive health is rejected as node_unavailable", func(t *testing.T) {
		err := ensureAgentDispatchable(&types.AgentNode{ID: "demoproj", HealthStatus: types.HealthStatusInactive})
		var pe *executionPreconditionError
		require.ErrorAs(t, err, &pe)
		assert.Equal(t, http.StatusServiceUnavailable, pe.HTTPStatusCode())
		assert.Equal(t, ErrorCategoryNodeUnavailable, pe.Category())
		assert.Equal(t, "node_unavailable", pe.ErrorCode())
		assert.Contains(t, pe.Error(), "demoproj")
	})

	t.Run("offline lifecycle is rejected as node_unavailable", func(t *testing.T) {
		err := ensureAgentDispatchable(&types.AgentNode{ID: "demoproj", LifecycleStatus: types.AgentStatusOffline})
		var pe *executionPreconditionError
		require.ErrorAs(t, err, &pe)
		assert.Equal(t, http.StatusServiceUnavailable, pe.HTTPStatusCode())
		assert.Equal(t, ErrorCategoryNodeUnavailable, pe.Category())
	})
}

func TestIsDialFailure(t *testing.T) {
	assert.False(t, isDialFailure(nil))
	assert.False(t, isDialFailure(errors.New("boom")))
	assert.True(t, isDialFailure(&net.OpError{Op: "dial", Err: errors.New("connection refused")}))
	// A failure after the connection was established must NOT be retried:
	// the agent may already be running the reasoner.
	assert.False(t, isDialFailure(&net.OpError{Op: "read", Err: errors.New("connection reset by peer")}))
	assert.True(t, isDialFailure(fmt.Errorf("agent call failed: %w", &net.OpError{Op: "dial", Err: errors.New("refused")})))
}

func TestAgentRestartGraceIsConfigurable(t *testing.T) {
	withRestartGrace(t, 42*time.Second)
	assert.Equal(t, 42*time.Second, agentRestartGrace())
}

func TestAgentCameBack(t *testing.T) {
	base := time.Now().Add(-time.Minute)

	t.Run("a new instance id is definitive", func(t *testing.T) {
		assert.True(t, agentCameBack(&types.AgentNode{InstanceID: "new", LastHeartbeat: base}, "old", base))
	})

	t.Run("the same instance id is not a restart", func(t *testing.T) {
		assert.False(t, agentCameBack(&types.AgentNode{InstanceID: "old", LastHeartbeat: base}, "old", base))
	})

	t.Run("a fresher heartbeat is the fallback for SDKs with no instance id", func(t *testing.T) {
		assert.True(t, agentCameBack(&types.AgentNode{LastHeartbeat: base.Add(time.Second)}, "", base))
		assert.False(t, agentCameBack(&types.AgentNode{LastHeartbeat: base}, "", base))
	})
}

func TestAgentMayRestart(t *testing.T) {
	withRestartGrace(t, time.Second)
	longRunning := &types.AgentNode{ID: "a", DeploymentType: "long_running"}

	assert.True(t, (&executionController{}).agentMayRestart(testPlan(longRunning)))
	assert.False(t, (&executionController{}).agentMayRestart(nil))
	assert.False(t, (&executionController{}).agentMayRestart(&preparedExecution{}))
	// Serverless has no resident process to come back.
	assert.False(t, (&executionController{}).agentMayRestart(testPlan(&types.AgentNode{ID: "a", DeploymentType: "serverless"})))

	SetAgentRestartGrace(0)
	assert.False(t, (&executionController{}).agentMayRestart(testPlan(longRunning)))
}

func TestNewAgentRequestCarriesExecutionContext(t *testing.T) {
	parent := "exec_parent"
	session := "sess_1"
	actor := "actor_1"
	controller := newExecutionController(newTestExecutionStorage(nil), nil, nil, 0, "internal-token")

	plan := testPlan(&types.AgentNode{ID: "demoproj", BaseURL: "http://127.0.0.1:8001"})
	plan.exec.ParentExecutionID = &parent
	plan.exec.SessionID = &session
	plan.exec.ActorID = &actor
	plan.callerDID = "did:af:caller"
	plan.targetDID = "did:af:target"
	plan.replaySourceRunID = "run_src"
	plan.replayBeforeExecutionID = "exec_before"
	plan.replayMode = "strict"

	req, err := controller.newAgentRequest(context.Background(), plan, buildAgentURL(plan.agent, plan.target))
	require.NoError(t, err)

	assert.Equal(t, "http://127.0.0.1:8001/reasoners/echo", req.URL.String())
	assert.Equal(t, "application/json", req.Header.Get("Content-Type"))
	assert.Equal(t, "run_1", req.Header.Get("X-Run-ID"))
	assert.Equal(t, "exec_1", req.Header.Get("X-Execution-ID"))
	assert.Equal(t, parent, req.Header.Get("X-Parent-Execution-ID"))
	assert.Equal(t, session, req.Header.Get("X-Session-ID"))
	assert.Equal(t, actor, req.Header.Get("X-Actor-ID"))
	assert.Equal(t, "Bearer internal-token", req.Header.Get("Authorization"))
	assert.Equal(t, "did:af:caller", req.Header.Get("X-Caller-DID"))
	assert.Equal(t, "did:af:target", req.Header.Get("X-Target-DID"))
	assert.Equal(t, "run_src", req.Header.Get("X-AgentField-Replay-Source-Run-ID"))
	assert.Equal(t, "exec_before", req.Header.Get("X-AgentField-Replay-Before-Execution-ID"))
	assert.Equal(t, "strict", req.Header.Get("X-AgentField-Replay-Mode"))

	t.Run("an unbuildable url is reported", func(t *testing.T) {
		_, err := controller.newAgentRequest(context.Background(), plan, "http://\x7f/bad")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "create agent request")
	})
}

// restartingStore serves an agent record that changes on a later read, the way
// storage does once a restarted node re-registers.
type restartingStore struct {
	*testExecutionStorage
	mu      sync.Mutex
	after   *types.AgentNode
	reads   int
	readsAt int // reads to serve before switching to `after`
}

func (s *restartingStore) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reads++
	if s.after != nil && s.reads >= s.readsAt {
		return s.after, nil
	}
	return s.testExecutionStorage.GetAgent(ctx, id)
}

func (s *restartingStore) readCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reads
}

// peekAgent returns the record a read would currently observe, without
// advancing the read counter — for assertions and conditional checks that
// must not perturb the switch-over bookkeeping.
func (s *restartingStore) peekAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.after != nil && s.reads >= s.readsAt {
		return s.after, nil
	}
	return s.testExecutionStorage.GetAgent(ctx, id)
}

// healthMarkingStore records the demotion the dispatcher asks for. It mirrors
// LocalStorage.UpdateAgentHealthAtomic's optimistic locking: when the caller
// passes an expected heartbeat, the write is rejected unless it matches the
// node's current LastHeartbeat, so tests exercise the same conditional
// semantics production storage enforces.
type healthMarkingStore struct {
	*restartingStore
	mu     sync.Mutex
	marked []types.HealthStatus
	err    error
}

func (s *healthMarkingStore) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	if expectedLastHeartbeat != nil {
		current, err := s.restartingStore.peekAgent(ctx, id)
		if err != nil || current == nil {
			return fmt.Errorf("agent %s not found for conditional health update", id)
		}
		if !current.LastHeartbeat.Equal(*expectedLastHeartbeat) {
			return fmt.Errorf("conditional health update rejected: heartbeat advanced")
		}
	}
	s.marked = append(s.marked, status)
	return nil
}

func (s *healthMarkingStore) markedStatuses() []types.HealthStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]types.HealthStatus(nil), s.marked...)
}

func TestDispatchAgentRequestSucceedsWithoutRetry(t *testing.T) {
	withRestartGrace(t, 2*time.Second)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	agent := &types.AgentNode{ID: "demoproj", BaseURL: server.URL, DeploymentType: "long_running"}
	store := &restartingStore{testExecutionStorage: newTestExecutionStorage(agent)}
	controller := newExecutionController(store, nil, nil, 0, "")

	resp, err := controller.dispatchAgentRequest(context.Background(), testPlan(agent))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	// No wait means the node record was never re-read.
	assert.Zero(t, store.readCount())
}

func TestDispatchAgentRequestWaitsOutARestart(t *testing.T) {
	withRestartGrace(t, 5*time.Second)

	// The address the caller dials is dead — this is the agent mid-restart.
	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
		LastHeartbeat:  time.Now().Add(-time.Second),
	}

	// The node comes back on a different port, as it may after a restart.
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		assert.Equal(t, "exec_1", r.Header.Get("X-Execution-ID"))
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	revived := *dead
	revived.BaseURL = server.URL
	revived.InstanceID = "instance-new"

	store := &restartingStore{
		testExecutionStorage: newTestExecutionStorage(dead),
		after:                &revived,
		readsAt:              2, // stay down for one poll, then come back
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	resp, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, int64(1), hits.Load())
}

func TestDispatchAgentRequestGivesUpAndDemotesTheNode(t *testing.T) {
	withRestartGrace(t, 600*time.Millisecond)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
		LastHeartbeat:  time.Now().Add(-time.Second),
	}
	store := &healthMarkingStore{
		restartingStore: &restartingStore{testExecutionStorage: newTestExecutionStorage(dead)},
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	resp, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.Nil(t, resp)
	assert.True(t, isDialFailure(err), "the original dial failure is preserved for classification")
	assert.Equal(t, []types.HealthStatus{types.HealthStatusInactive}, store.markedStatuses())
}

func TestDispatchAgentRequestStopsWhenTheContextEnds(t *testing.T) {
	withRestartGrace(t, 10*time.Second)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
	}
	store := &healthMarkingStore{
		restartingStore: &restartingStore{testExecutionStorage: newTestExecutionStorage(dead)},
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := controller.dispatchAgentRequest(ctx, testPlan(dead))
	require.Error(t, err)
	assert.Less(t, time.Since(start), 5*time.Second, "the wait must abandon with the request, not run the full grace")
	assert.Empty(t, store.markedStatuses(), "an abandoned wait must not demote the node")
}

func TestDispatchAgentRequestDoesNotWaitForServerless(t *testing.T) {
	withRestartGrace(t, 10*time.Second)

	agent := &types.AgentNode{
		ID:             "lambda",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "serverless",
	}
	store := &restartingStore{testExecutionStorage: newTestExecutionStorage(agent)}
	controller := newExecutionController(store, nil, nil, 0, "")

	start := time.Now()
	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(agent))
	require.Error(t, err)
	assert.Less(t, time.Since(start), 2*time.Second)
	assert.Zero(t, store.readCount())
}

func TestDispatchAgentRequestDoesNotRetryNonDialFailures(t *testing.T) {
	withRestartGrace(t, 10*time.Second)

	agent := &types.AgentNode{ID: "demoproj", BaseURL: "ftp://127.0.0.1:1", DeploymentType: "long_running"}
	store := &restartingStore{testExecutionStorage: newTestExecutionStorage(agent)}
	controller := newExecutionController(store, nil, nil, 0, "")

	start := time.Now()
	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(agent))
	require.Error(t, err)
	assert.Less(t, time.Since(start), 2*time.Second)
	assert.Zero(t, store.readCount())
}

func TestMarkAgentUnreachableToleratesStoreWithoutHealthSupport(t *testing.T) {
	agent := &types.AgentNode{ID: "demoproj", BaseURL: "http://127.0.0.1:1"}
	controller := newExecutionController(newTestExecutionStorage(agent), nil, nil, 0, "")
	// testExecutionStorage does not implement agentHealthMarker; this must be
	// a no-op rather than a panic.
	controller.markAgentUnreachable(context.Background(), testPlan(agent))

	t.Run("a failing demotion is swallowed", func(t *testing.T) {
		store := &healthMarkingStore{
			restartingStore: &restartingStore{testExecutionStorage: newTestExecutionStorage(agent)},
			err:             errors.New("write conflict"),
		}
		c := newExecutionController(store, nil, nil, 0, "")
		c.markAgentUnreachable(context.Background(), testPlan(agent))
		assert.Empty(t, store.markedStatuses())
	})

	t.Run("a plan without an agent is ignored", func(t *testing.T) {
		store := &healthMarkingStore{
			restartingStore: &restartingStore{testExecutionStorage: newTestExecutionStorage(agent)},
		}
		c := newExecutionController(store, nil, nil, 0, "")
		c.markAgentUnreachable(context.Background(), &preparedExecution{target: &parsedTarget{NodeID: "demoproj"}})
		assert.Empty(t, store.markedStatuses())
	})
}

// A cancel that lands while the dispatch is waiting out a restart must win:
// the replay is never sent, so the agent does no work the caller already
// disowned.
func TestDispatchAgentRequestHonoursCancelDuringTheWait(t *testing.T) {
	withRestartGrace(t, 5*time.Second)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
		LastHeartbeat:  time.Now().Add(-time.Second),
	}

	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	revived := *dead
	revived.BaseURL = server.URL
	revived.InstanceID = "instance-new"

	store := &restartingStore{
		testExecutionStorage: newTestExecutionStorage(dead),
		after:                &revived,
		readsAt:              1,
	}
	// The execution was cancelled while the node was down.
	store.executionRecords["exec_1"] = &types.Execution{
		ExecutionID: "exec_1",
		RunID:       "run_1",
		Status:      types.ExecutionStatusCancelled,
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	start := time.Now()
	resp, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "cancelled")
	assert.Zero(t, hits.Load(), "a cancelled execution must never be replayed to the agent")
	assert.Less(t, time.Since(start), 5*time.Second, "the cancel ends the wait early")
}

// Once the node's record says the node is definitively down — the health
// checker or another dispatch's expired grace demoted it — waiting longer
// cannot help, and every queued dispatch aimed at it must stop burning its
// full grace on a verdict that is already in.
func TestDispatchAgentRequestStopsWaitingOnceTheNodeIsDemoted(t *testing.T) {
	withRestartGrace(t, 10*time.Second)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
		LastHeartbeat:  time.Now().Add(-time.Second),
	}
	demoted := *dead
	demoted.HealthStatus = types.HealthStatusInactive

	store := &restartingStore{
		testExecutionStorage: newTestExecutionStorage(dead),
		after:                &demoted,
		readsAt:              1,
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	start := time.Now()
	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.True(t, isDialFailure(err), "the original dial failure is preserved for classification")
	assert.Less(t, time.Since(start), 3*time.Second, "a demoted node ends the wait long before the grace expires")
}

// Serverless nodes have no heartbeat loop and are never polled by the health
// monitor, so their recorded health goes inactive as a matter of course. The
// fail-fast gate must not apply to them: a serverless target whose record
// says inactive must still be invoked.
func TestExecuteStillDispatchesServerlessDespiteInactiveHealth(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var hits atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(`{"result":{"ok":true}}`))
	}))
	defer backend.Close()

	agent := &types.AgentNode{
		ID:              "lambda",
		BaseURL:         backend.URL,
		DeploymentType:  "serverless",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		Reasoners:       []types.ReasonerDefinition{{ID: "echo"}},
	}
	store := newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/execute/:target", ExecuteHandler(store, nil, nil, 0, ""))

	req := httptest.NewRequest(http.MethodPost, "/execute/lambda.echo",
		strings.NewReader(`{"input":{"message":"hi"}}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.NotEqual(t, http.StatusServiceUnavailable, rec.Code,
		"serverless must not be rejected by the node-health gate; body: %s", rec.Body.String())
	assert.NotContains(t, rec.Body.String(), "node_unavailable")
	assert.Equal(t, int64(1), hits.Load(), "the serverless backend must be invoked")
}

// A replay request is served from the recorded run without contacting the
// agent, so the target node being down must not reject it. (A replay miss
// dials and fails exactly as it did before the gate existed.)
func TestExecuteReplayRequestBypassesTheNodeHealthGate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:              "demoproj",
		BaseURL:         "http://" + deadAddress(t),
		DeploymentType:  "long_running",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		Reasoners:       []types.ReasonerDefinition{{ID: "echo"}},
	}
	store := newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/execute/:target", ExecuteHandler(store, nil, nil, 0, ""))

	req := httptest.NewRequest(http.MethodPost, "/execute/demoproj.echo",
		strings.NewReader(`{"input":{"message":"hi"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-AgentField-Replay-Source-Run-ID", "run_recorded")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.NotContains(t, rec.Body.String(), "node_unavailable",
		"a replay request must not be rejected by the node-health gate")
}

// The end-to-end contract: a call aimed at a node we already know is down is
// rejected, and leaves no execution record behind to be counted as a failure.
func TestExecuteRejectsKnownDownNodeWithoutRecordingAnExecution(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:              "demoproj",
		BaseURL:         "http://127.0.0.1:8001",
		DeploymentType:  "long_running",
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusOffline,
		Reasoners:       []types.ReasonerDefinition{{ID: "echo"}},
	}
	store := newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/execute/:target", ExecuteHandler(store, nil, nil, 0, ""))

	req := httptest.NewRequest(http.MethodPost, "/execute/demoproj.echo",
		strings.NewReader(`{"input":{"message":"hi"}}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusServiceUnavailable, rec.Code)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	assert.Equal(t, "node_unavailable", body["error"])
	assert.Equal(t, "node_unavailable", body["error_category"])

	store.mu.Lock()
	recorded := len(store.executionRecords)
	store.mu.Unlock()
	assert.Zero(t, recorded, "a request we never dispatched must not be counted as a failed execution")
}

// errAgentReadStore fails every node re-read, the way storage can while the
// control plane is under load. The wait must ride it out rather than crash.
type errAgentReadStore struct {
	*testExecutionStorage
}

func (s *errAgentReadStore) GetAgent(context.Context, string) (*types.AgentNode, error) {
	return nil, errors.New("storage unavailable")
}

func TestDispatchAgentRequestToleratesFailingNodeReads(t *testing.T) {
	withRestartGrace(t, 600*time.Millisecond)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
	}
	controller := newExecutionController(&errAgentReadStore{newTestExecutionStorage(dead)}, nil, nil, 0, "")

	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.True(t, isDialFailure(err))
}

func TestDispatchAgentRequestKeepsWaitingWhenTheNodeIsNotServingYet(t *testing.T) {
	withRestartGrace(t, 900*time.Millisecond)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
	}
	// The node re-registered — new instance id — but its HTTP server is not
	// listening yet, so the retry dials into another refusal.
	announced := *dead
	announced.InstanceID = "instance-new"

	store := &healthMarkingStore{
		restartingStore: &restartingStore{
			testExecutionStorage: newTestExecutionStorage(dead),
			after:                &announced,
			readsAt:              1,
		},
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.True(t, isDialFailure(err))
	assert.Equal(t, []types.HealthStatus{types.HealthStatusInactive}, store.markedStatuses())
	assert.Greater(t, store.readCount(), 1, "the wait keeps polling after a premature re-registration")
}

func TestDispatchAgentRequestSurfacesAnUnbuildableRetryURL(t *testing.T) {
	withRestartGrace(t, 2*time.Second)

	dead := &types.AgentNode{
		ID:             "demoproj",
		BaseURL:        "http://" + deadAddress(t),
		DeploymentType: "long_running",
		InstanceID:     "instance-old",
	}
	revived := *dead
	revived.InstanceID = "instance-new"
	revived.BaseURL = "http://\x7f"

	store := &restartingStore{
		testExecutionStorage: newTestExecutionStorage(dead),
		after:                &revived,
		readsAt:              1,
	}
	controller := newExecutionController(store, nil, nil, 0, "")

	_, err := controller.dispatchAgentRequest(context.Background(), testPlan(dead))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create agent request")
}

func TestClassifyRawErrorNamesMissingTargets(t *testing.T) {
	// The first-run case: the quickstart curl runs before `python main.py`,
	// so the node has never registered. That is the caller's mistake, and
	// reporting it as internal_error made the product look broken.
	assert.Equal(t, ErrorCategoryTargetNotFound, classifyRawError(errors.New("agent 'myagent' not found")))
	assert.Equal(t, ErrorCategoryTargetNotFound, classifyRawError(errors.New("target 'demo_ecoh' not found on agent 'myagent'")))
	// Anything else keeps the internal fallback.
	assert.Equal(t, ErrorCategoryInternal, classifyRawError(errors.New("config file not found")))
	assert.Equal(t, ErrorCategoryInternal, classifyRawError(errors.New("boom")))
}

func TestExecutionRestartHoldIsStampedOnBothRows(t *testing.T) {
	agent := &types.AgentNode{ID: "demoproj", BaseURL: "http://127.0.0.1:8001"}
	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 0, "")
	plan := testPlan(agent)

	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: plan.exec.ExecutionID,
		RunID:       plan.exec.RunID,
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: plan.exec.ExecutionID,
		WorkflowID:  plan.exec.RunID,
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	reasonOf := func(t *testing.T) (string, string) {
		t.Helper()
		exec, err := store.GetExecutionRecord(context.Background(), plan.exec.ExecutionID)
		require.NoError(t, err)
		wf, err := store.GetWorkflowExecution(context.Background(), plan.exec.ExecutionID)
		require.NoError(t, err)
		read := func(p *string) string {
			if p == nil {
				return ""
			}
			return *p
		}
		return read(exec.StatusReason), read(wf.StatusReason)
	}

	controller.markExecutionAwaitingRestart(context.Background(), plan)
	execReason, wfReason := reasonOf(t)
	assert.Equal(t, types.ExecutionReasonAwaitingAgentRestart, execReason)
	assert.Equal(t, types.ExecutionReasonAwaitingAgentRestart, wfReason,
		"the DAG mirror must carry the hold too, or the reaper fails the run the retry saved")

	controller.clearExecutionAwaitingRestart(context.Background(), plan)
	execReason, wfReason = reasonOf(t)
	assert.Empty(t, execReason)
	assert.Empty(t, wfReason)

	t.Run("clearing a hold that was never set leaves the reason alone", func(t *testing.T) {
		other := "agent_error"
		_, err := store.UpdateExecutionRecord(context.Background(), plan.exec.ExecutionID,
			func(current *types.Execution) (*types.Execution, error) {
				current.StatusReason = &other
				return current, nil
			})
		require.NoError(t, err)
		controller.clearExecutionAwaitingRestart(context.Background(), plan)
		execReason, _ := reasonOf(t)
		assert.Equal(t, other, execReason)
	})

	t.Run("a terminal execution is left to whoever finished it", func(t *testing.T) {
		_, err := store.UpdateExecutionRecord(context.Background(), plan.exec.ExecutionID,
			func(current *types.Execution) (*types.Execution, error) {
				current.Status = types.ExecutionStatusSucceeded
				current.StatusReason = nil
				return current, nil
			})
		require.NoError(t, err)
		controller.markExecutionAwaitingRestart(context.Background(), plan)
		execReason, _ := reasonOf(t)
		assert.Empty(t, execReason)
	})

	t.Run("a missing row is logged, not fatal", func(t *testing.T) {
		missing := testPlan(agent)
		missing.exec.ExecutionID = "exec_absent"
		controller.markExecutionAwaitingRestart(context.Background(), missing)
	})

	t.Run("a plan without an execution is ignored", func(t *testing.T) {
		controller.markExecutionAwaitingRestart(context.Background(), nil)
		controller.markExecutionAwaitingRestart(context.Background(), &preparedExecution{})
	})
}
