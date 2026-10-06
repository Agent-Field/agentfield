package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// reasonerHandlerStorage embeds the StorageProvider interface (its methods are
// nil stubs that panic if called) and overrides only the small set of methods
// that ExecuteReasonerHandler actually invokes. testExecutionStorage is held as
// a named field — embedding it would create ambiguous selectors with the
// interface methods.
type reasonerHandlerStorage struct {
	storage.StorageProvider
	exec           *testExecutionStorage
	agent          *types.AgentNode
	getAgentErr    error
	persisted      chan *types.WorkflowExecution
	releasePersist chan struct{}
}

func newReasonerHandlerStorage(agent *types.AgentNode) *reasonerHandlerStorage {
	return &reasonerHandlerStorage{
		exec:  newTestExecutionStorage(agent),
		agent: agent,
	}
}

// Methods used by the tests themselves to inspect persisted state.
func (s *reasonerHandlerStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	return s.exec.QueryWorkflowExecutions(ctx, filters)
}

func (s *reasonerHandlerStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return s.exec.GetWorkflowExecution(ctx, executionID)
}

// Methods invoked by ExecuteReasonerHandler.
func (s *reasonerHandlerStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.getAgentErr != nil {
		return nil, s.getAgentErr
	}
	if s.agent != nil && s.agent.ID == id {
		return s.agent, nil
	}
	return nil, errors.New("agent not found")
}

func (s *reasonerHandlerStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	if execution == nil {
		return s.exec.StoreWorkflowExecution(ctx, execution)
	}

	cloned := *execution
	if s.persisted != nil {
		s.persisted <- &cloned
	}
	if s.releasePersist != nil {
		<-s.releasePersist
	}
	return s.exec.StoreWorkflowExecution(ctx, &cloned)
}

func (s *reasonerHandlerStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return s.exec.CreateExecutionRecord(ctx, execution)
}

func (s *reasonerHandlerStorage) GetExecutionEventBus() *events.ExecutionEventBus {
	return s.exec.GetExecutionEventBus()
}

func (s *reasonerHandlerStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return s.exec.GetWorkflowExecutionEventBus()
}

func (s *reasonerHandlerStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return s.exec.UpdateWorkflowExecution(ctx, executionID, updateFunc)
}

func newReasonerAgent(baseURL string) *types.AgentNode {
	return &types.AgentNode{
		ID:              "node-1",
		BaseURL:         baseURL,
		Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}
}

func TestExecuteReasonerHandler_MalformedReasonerIDReturnsBadRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newReasonerHandlerStorage(nil)
	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/not-valid", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "node_id.reasoner_name")
}

func TestExecuteReasonerHandler_NodeLookupAndAvailabilityErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		store      *reasonerHandlerStorage
		wantCode   int
		wantErrMsg string
	}{
		{
			name: "node not found",
			store: &reasonerHandlerStorage{
				exec:                 newTestExecutionStorage(nil),
				getAgentErr:          errors.New("missing"),
			},
			wantCode:   http.StatusNotFound,
			wantErrMsg: "node 'node-404' not found",
		},
		{
			name: "inactive node",
			store: newReasonerHandlerStorage(&types.AgentNode{
				ID:              "node-404",
				BaseURL:         "http://agent.invalid",
				Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
				HealthStatus:    types.HealthStatusInactive,
				LifecycleStatus: types.AgentStatusReady,
			}),
			wantCode:   http.StatusServiceUnavailable,
			wantErrMsg: "is not healthy",
		},
		{
			name: "offline node",
			store: newReasonerHandlerStorage(&types.AgentNode{
				ID:              "node-404",
				BaseURL:         "http://agent.invalid",
				Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusOffline,
			}),
			wantCode:   http.StatusServiceUnavailable,
			wantErrMsg: "is offline",
		},
		{
			name: "pending approval node",
			store: newReasonerHandlerStorage(&types.AgentNode{
				ID:              "node-404",
				BaseURL:         "http://agent.invalid",
				Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusPendingApproval,
			}),
			wantCode:   http.StatusServiceUnavailable,
			wantErrMsg: "agent_pending_approval",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(tt.store))

			req := httptest.NewRequest(http.MethodPost, "/reasoners/node-404.ping", strings.NewReader(`{"input":{}}`))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			require.Equal(t, tt.wantCode, resp.Code)

			var payload map[string]string
			require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
			require.Contains(t, payload["error"], tt.wantErrMsg)

			records, err := tt.store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
			require.NoError(t, err)
			require.Empty(t, records)
		})
	}
}

func TestExecuteReasonerHandler_PersistsSuccessfulExecutionBeforeResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/reasoners/ping", r.URL.Path)

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		defer r.Body.Close()
		require.JSONEq(t, `{}`, string(body))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newReasonerHandlerStorage(newReasonerAgent(agentServer.URL))
	store.persisted = make(chan *types.WorkflowExecution, 1)
	store.releasePersist = make(chan struct{})

	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		router.ServeHTTP(resp, req)
		close(done)
	}()

	persisted := <-store.persisted
	require.Equal(t, string(types.ExecutionStatusSucceeded), persisted.Status)
	require.Equal(t, 2, persisted.InputSize)
	require.Equal(t, len(`{"ok":true}`), persisted.OutputSize)
	require.NotNil(t, persisted.DurationMS)
	require.GreaterOrEqual(t, *persisted.DurationMS, int64(0))

	select {
	case <-done:
		t.Fatal("handler responded before StoreWorkflowExecution returned")
	default:
	}

	close(store.releasePersist)
	<-done

	require.Equal(t, http.StatusOK, resp.Code)

	var payload ExecuteReasonerResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "node-1", payload.NodeID)
	require.GreaterOrEqual(t, payload.Duration, int64(0))

	stored, err := store.GetWorkflowExecution(context.Background(), persisted.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	require.Equal(t, string(types.ExecutionStatusSucceeded), stored.Status)
	require.JSONEq(t, `{"ok":true}`, string(stored.OutputData))
}

func TestExecuteReasonerHandler_PersistsFailedExecutionBeforeResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newReasonerHandlerStorage(newReasonerAgent("://bad"))
	store.persisted = make(chan *types.WorkflowExecution, 1)
	store.releasePersist = make(chan struct{})

	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		router.ServeHTTP(resp, req)
		close(done)
	}()

	persisted := <-store.persisted
	require.Equal(t, string(types.ExecutionStatusFailed), persisted.Status)
	require.NotNil(t, persisted.ErrorMessage)
	require.Contains(t, *persisted.ErrorMessage, "failed to create agent request")
	require.NotNil(t, persisted.DurationMS)
	require.GreaterOrEqual(t, *persisted.DurationMS, int64(0))

	select {
	case <-done:
		t.Fatal("handler responded before failed execution was persisted")
	default:
	}

	close(store.releasePersist)
	<-done

	require.Equal(t, http.StatusInternalServerError, resp.Code)

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "failed to create agent request")

	stored, err := store.GetWorkflowExecution(context.Background(), persisted.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	require.Equal(t, string(types.ExecutionStatusFailed), stored.Status)
}

func TestExecuteReasonerHandler_ServerlessPayloadAndHeaderPropagation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	observedHeaders := make(chan http.Header, 1)
	observedBody := make(chan map[string]interface{}, 1)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/execute", r.URL.Path)

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		defer r.Body.Close()

		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal(body, &payload))

		observedHeaders <- r.Header.Clone()
		observedBody <- payload

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"serverless":true}`))
	}))
	defer agentServer.Close()

	agent := newReasonerAgent(agentServer.URL)
	agent.DeploymentType = "serverless"

	store := newReasonerHandlerStorage(agent)
	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{"message":"hello"}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-serverless")
	req.Header.Set("X-Session-ID", "session-1")
	req.Header.Set("X-Agent-Node-ID", "caller-node")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	headers := <-observedHeaders
	require.Equal(t, "wf-serverless", headers.Get("X-Workflow-ID"))
	require.Equal(t, "wf-serverless", headers.Get("X-Run-ID"))
	require.Equal(t, "session-1", headers.Get("X-Session-ID"))
	require.NotEmpty(t, headers.Get("X-Execution-ID"))

	body := <-observedBody
	require.Equal(t, "/execute/ping", body["path"])
	require.Equal(t, "ping", body["target"])
	require.Equal(t, "ping", body["reasoner"])
	require.Equal(t, "reasoner", body["type"])

	input, ok := body["input"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, "hello", input["message"])

	execCtx, ok := body["execution_context"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, "wf-serverless", execCtx["run_id"])
	require.Equal(t, "wf-serverless", execCtx["workflow_id"])
	require.Equal(t, "session-1", execCtx["session_id"])
	require.NotEmpty(t, execCtx["execution_id"])
}

func (s *reasonerHandlerStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	return s.exec.GetExecutionRecord(ctx, executionID)
}

func (s *reasonerHandlerStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return s.exec.UpdateExecutionRecord(ctx, executionID, update)
}

type legacyDispatchObservation struct {
	executionID string
	record      *types.Execution
}

// legacyRowAgent returns a stub agent that captures the executions row as it
// exists while the control plane is dispatching to it.
func legacyRowAgent(t *testing.T, store *reasonerHandlerStorage, observed chan<- legacyDispatchObservation) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		executionID := r.Header.Get("X-Execution-ID")
		rec, err := store.GetExecutionRecord(r.Context(), executionID)
		if err != nil {
			rec = nil
		}
		observed <- legacyDispatchObservation{executionID: executionID, record: rec}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
}

func TestExecuteReasonerHandler_CreatesExecutionRowBeforeDispatch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newReasonerHandlerStorage(nil)
	observed := make(chan legacyDispatchObservation, 1)
	agentServer := legacyRowAgent(t, store, observed)
	defer agentServer.Close()
	store.agent = newReasonerAgent(agentServer.URL)

	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{"q":1}}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workflow-ID", "wf-legacy")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	obs := <-observed
	require.NotEmpty(t, obs.executionID)
	require.NotNil(t, obs.record, "executions row must exist while the agent is being called")
	require.Equal(t, string(types.ExecutionStatusRunning), obs.record.Status)
	require.Equal(t, "wf-legacy", obs.record.RunID)
	require.Equal(t, "node-1", obs.record.AgentNodeID)
	require.Equal(t, "ping", obs.record.ReasonerID)
	require.JSONEq(t, `{"q":1}`, string(obs.record.InputPayload))

	final, err := store.GetExecutionRecord(context.Background(), obs.executionID)
	require.NoError(t, err)
	require.NotNil(t, final)
	require.Equal(t, string(types.ExecutionStatusSucceeded), final.Status)
	require.JSONEq(t, `{"ok":true}`, string(final.ResultPayload))
	require.NotNil(t, final.CompletedAt)
}

func TestExecuteSkillHandler_CreatesExecutionRowBeforeDispatch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newReasonerHandlerStorage(nil)
	observed := make(chan legacyDispatchObservation, 1)
	agentServer := legacyRowAgent(t, store, observed)
	defer agentServer.Close()
	agent := newReasonerAgent(agentServer.URL)
	agent.Skills = []types.SkillDefinition{{ID: "ping"}}
	store.agent = agent

	router := gin.New()
	router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/skills/node-1.ping", strings.NewReader(`{"input":{"q":1}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	obs := <-observed
	require.NotEmpty(t, obs.executionID)
	require.NotNil(t, obs.record, "executions row must exist while the agent is being called")
	require.Equal(t, string(types.ExecutionStatusRunning), obs.record.Status)
	require.Equal(t, "ping", obs.record.ReasonerID)

	final, err := store.GetExecutionRecord(context.Background(), obs.executionID)
	require.NoError(t, err)
	require.NotNil(t, final)
	require.Equal(t, string(types.ExecutionStatusSucceeded), final.Status)
}

// TestExecuteReasonerHandler_AgentStatusCallbackFindsExecution reproduces #1086
// end to end on the real local storage: while the legacy endpoint is
// dispatching, the agent posts its status callback to the real
// UpdateExecutionStatusHandler. Without the executions row that callback is a
// 404. The legacy handler must also keep the terminal status the agent
// reported instead of overwriting it with its own result.
func TestExecuteReasonerHandler_AgentStatusCallbackFindsExecution(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ctx := context.Background()

	store := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := store.Initialize(ctx, storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(t.TempDir(), "agentfield.db"),
			KVStorePath:  filepath.Join(t.TempDir(), "agentfield.bolt"),
		},
	})
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close(ctx) })

	router := gin.New()
	router.POST("/api/v1/reasoners/:reasoner_id", ExecuteReasonerHandler(store))
	router.POST("/api/v1/executions/:execution_id/status",
		UpdateExecutionStatusHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 90*time.Second))
	controlPlane := httptest.NewServer(router)
	defer controlPlane.Close()

	executionIDs := make(chan string, 1)
	callbackCodes := make(chan int, 1)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		executionID := r.Header.Get("X-Execution-ID")
		executionIDs <- executionID
		code := 0
		resp, err := http.Post(
			controlPlane.URL+"/api/v1/executions/"+executionID+"/status",
			"application/json",
			strings.NewReader(`{"status":"failed","error":"agent reported failure"}`),
		)
		if err == nil {
			code = resp.StatusCode
			_ = resp.Body.Close()
		}
		callbackCodes <- code
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	require.NoError(t, store.RegisterAgent(ctx, newReasonerAgent(agentServer.URL)))

	resp, err := http.Post(controlPlane.URL+"/api/v1/reasoners/node-1.ping", "application/json", strings.NewReader(`{}`))
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	executionID := <-executionIDs
	require.NotEmpty(t, executionID)
	require.Equal(t, http.StatusOK, <-callbackCodes, "agent status callback for a legacy execution must not 404")

	final, err := store.GetExecutionRecord(ctx, executionID)
	require.NoError(t, err)
	require.NotNil(t, final)
	require.Equal(t, string(types.ExecutionStatusFailed), final.Status)
	require.NotNil(t, final.ErrorMessage)
	require.Equal(t, "agent reported failure", *final.ErrorMessage)
}

type failingCreateRecordStorage struct {
	*reasonerHandlerStorage
}

func (s *failingCreateRecordStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	return errors.New("insert failed")
}

// A failed executions insert is logged and the legacy call still goes through,
// so the endpoint gains no new failure mode.
func TestExecuteReasonerHandler_ExecutionRowInsertFailureDoesNotFailCall(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newReasonerHandlerStorage(nil)
	observed := make(chan legacyDispatchObservation, 1)
	agentServer := legacyRowAgent(t, store, observed)
	defer agentServer.Close()
	store.agent = newReasonerAgent(agentServer.URL)

	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(&failingCreateRecordStorage{store}))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)

	obs := <-observed
	require.Nil(t, obs.record)

	wfExec, err := store.GetWorkflowExecution(context.Background(), obs.executionID)
	require.NoError(t, err)
	require.NotNil(t, wfExec)
	require.Equal(t, string(types.ExecutionStatusSucceeded), string(wfExec.Status))
}
