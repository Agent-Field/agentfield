package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// mcpTestStore adds a multi-agent discovery surface on top of the shared
// execution-store test double so the MCP tools can be exercised end-to-end.
type mcpTestStore struct {
	*testExecutionStorage
	agents []*types.AgentNode
}

type stopPoolOnCreateMCPStore struct {
	*mcpTestStore
	pool      *asyncWorkerPool
	cancel    context.CancelFunc
	updateErr error
	updated   bool
}

type panicOnCreateMCPStore struct {
	*mcpTestStore
}

func (s *panicOnCreateMCPStore) CreateExecutionRecord(context.Context, *types.Execution) error {
	panic("injected MCP execution persistence panic")
}

func (s *stopPoolOnCreateMCPStore) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	s.pool.mu.Lock()
	s.pool.stopped = true
	s.pool.mu.Unlock()
	err := s.mcpTestStore.CreateExecutionRecord(ctx, execution)
	if s.cancel != nil {
		s.cancel()
	}
	return err
}

func (s *stopPoolOnCreateMCPStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	s.updated = true
	if s.updateErr != nil {
		return nil, s.updateErr
	}
	return s.mcpTestStore.UpdateExecutionRecord(ctx, executionID, update)
}

func newMCPTestStore(agents ...*types.AgentNode) *mcpTestStore {
	var primary *types.AgentNode
	if len(agents) > 0 {
		primary = agents[0]
	}
	return &mcpTestStore{
		testExecutionStorage: newTestExecutionStorage(primary),
		agents:               agents,
	}
}

func (s *mcpTestStore) ListAgents(_ context.Context, _ types.AgentFilters) ([]*types.AgentNode, error) {
	return s.agents, nil
}

func (s *mcpTestStore) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	for _, a := range s.agents {
		if a != nil && a.ID == id {
			return a, nil
		}
	}
	return nil, fmt.Errorf("agent %q not found", id)
}

func mcpActiveAgent() *types.AgentNode {
	return &types.AgentNode{
		ID:            "planner",
		BaseURL:       "http://agent.example",
		HealthStatus:  types.HealthStatusActive,
		LastHeartbeat: time.Now().UTC(),
		Reasoners: []types.ReasonerDefinition{
			{
				ID:           "plan",
				Description:  "Make a plan",
				Tags:         []string{"entrypoint", "planning"},
				InputSchema:  json.RawMessage(`{"type":"object","properties":{"goal":{"type":"string"}},"required":["goal"]}`),
				OutputSchema: json.RawMessage(`{"type":"object","properties":{"steps":{"type":"array"}}}`),
			},
		},
	}
}

func mcpDeadAgent() *types.AgentNode {
	return &types.AgentNode{
		ID:            "ghost",
		BaseURL:       "http://ghost.example",
		HealthStatus:  types.HealthStatusInactive,
		LastHeartbeat: time.Now().Add(-time.Hour).UTC(),
		Reasoners: []types.ReasonerDefinition{
			{ID: "haunt", Description: "Boo"},
		},
	}
}

func newMCPTestRouter(t *testing.T, store MCPStore) *gin.Engine {
	return newMCPTestRouterWithAuthorizer(t, store, nil)
}

func newMCPTestRouterWithAuthorizer(t *testing.T, store MCPStore, authorize MCPAuthorizer) *gin.Engine {
	return newMCPTestRouterWithCaller(t, store, authorize, "")
}

func newMCPTestRouterWithCaller(t *testing.T, store MCPStore, authorize MCPAuthorizer, callerDID string) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	if callerDID != "" {
		r.Use(func(c *gin.Context) {
			c.Set("verified_caller_did", callerDID)
			c.Next()
		})
	}
	payloads := services.NewFilePayloadStore(t.TempDir())
	r.POST("/mcp", MCPHandler(store, payloads, nil, 5*time.Second, "", "1.2.3-test", authorize))
	return r
}

func mcpPost(t *testing.T, router *gin.Engine, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	// Stateless mode: harnesses send these headers; they must be tolerated.
	req.Header.Set("Mcp-Session-Id", "ignored-session")
	req.Header.Set("MCP-Protocol-Version", "2025-06-18")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

func mcpDecode(t *testing.T, w *httptest.ResponseRecorder) map[string]interface{} {
	t.Helper()
	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp), "body: %s", w.Body.String())
	return resp
}

// mcpCallTool issues a tools/call and returns the decoded text payload plus the
// isError flag. When the tool's text block is JSON it is parsed; otherwise it is
// surfaced under "text".
func mcpCallTool(t *testing.T, router *gin.Engine, name string, args map[string]interface{}) (map[string]interface{}, bool) {
	t.Helper()
	reqBody, err := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]interface{}{"name": name, "arguments": args},
	})
	require.NoError(t, err)

	w := mcpPost(t, router, string(reqBody))
	require.Equal(t, http.StatusOK, w.Code)
	resp := mcpDecode(t, w)
	require.Nil(t, resp["error"], "unexpected JSON-RPC error: %v", resp["error"])

	result, ok := resp["result"].(map[string]interface{})
	require.True(t, ok, "missing result: %v", resp)
	isError, _ := result["isError"].(bool)

	content, ok := result["content"].([]interface{})
	require.True(t, ok && len(content) > 0, "missing content: %v", result)
	first := content[0].(map[string]interface{})
	require.Equal(t, "text", first["type"])
	text, _ := first["text"].(string)

	payload := map[string]interface{}{}
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		payload = map[string]interface{}{"text": text}
	}
	return payload, isError
}

func TestMCP_InitializeAndToolsList(t *testing.T) {
	router := newMCPTestRouter(t, newMCPTestStore(mcpActiveAgent()))

	t.Run("initialize echoes known protocol version", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}`)
		require.Equal(t, http.StatusOK, w.Code)
		result := mcpDecode(t, w)["result"].(map[string]interface{})
		require.Equal(t, "2025-06-18", result["protocolVersion"])

		caps := result["capabilities"].(map[string]interface{})
		_, hasTools := caps["tools"]
		require.True(t, hasTools, "capabilities.tools must be present")

		info := result["serverInfo"].(map[string]interface{})
		require.Equal(t, "agentfield", info["name"])
		require.Equal(t, "1.2.3-test", info["version"])
	})

	t.Run("initialize falls back to default for unknown version", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}`)
		result := mcpDecode(t, w)["result"].(map[string]interface{})
		require.Equal(t, "2025-06-18", result["protocolVersion"])
	})

	t.Run("tools/list returns exactly the five tools with schemas", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
		result := mcpDecode(t, w)["result"].(map[string]interface{})
		tools := result["tools"].([]interface{})
		require.Len(t, tools, 5)

		names := map[string]bool{}
		for _, tl := range tools {
			m := tl.(map[string]interface{})
			names[m["name"].(string)] = true
			schema, ok := m["inputSchema"].(map[string]interface{})
			require.True(t, ok, "tool %v missing inputSchema", m["name"])
			require.Equal(t, "object", schema["type"])
		}
		for _, want := range []string{"discover_agents", "get_reasoner_schema", "execute_reasoner", "get_run", "wait_run"} {
			require.True(t, names[want], "missing tool %q", want)
		}
	})
}

func TestMCP_ProtocolMiscMethods(t *testing.T) {
	router := newMCPTestRouter(t, newMCPTestStore(mcpActiveAgent()))

	t.Run("notifications/initialized returns 202 with no body", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
		require.Equal(t, http.StatusAccepted, w.Code)
		require.Empty(t, w.Body.String())
	})

	t.Run("ping returns empty result", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":9,"method":"ping"}`)
		resp := mcpDecode(t, w)
		require.Equal(t, map[string]interface{}{}, resp["result"])
	})

	t.Run("unknown method returns -32601", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":3,"method":"does/notexist"}`)
		errObj := mcpDecode(t, w)["error"].(map[string]interface{})
		require.Equal(t, float64(mcpErrMethodNotFound), errObj["code"])
	})

	t.Run("batch requests are rejected", func(t *testing.T) {
		w := mcpPost(t, router, `[{"jsonrpc":"2.0","id":1,"method":"ping"}]`)
		errObj := mcpDecode(t, w)["error"].(map[string]interface{})
		require.Equal(t, float64(mcpErrInvalidRequest), errObj["code"])
	})

	t.Run("unknown tool returns -32602", func(t *testing.T) {
		w := mcpPost(t, router, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"frobnicate","arguments":{}}}`)
		errObj := mcpDecode(t, w)["error"].(map[string]interface{})
		require.Equal(t, float64(mcpErrInvalidParams), errObj["code"])
		require.Contains(t, errObj["message"], "unknown tool")
	})
}

func TestMCP_DiscoverAgents(t *testing.T) {
	router := newMCPTestRouter(t, newMCPTestStore(mcpActiveAgent(), mcpDeadAgent()))

	t.Run("default returns only active agents", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "discover_agents", map[string]interface{}{})
		require.False(t, isErr)
		require.Equal(t, "active", payload["health"])

		agents := payload["agents"].([]interface{})
		require.Len(t, agents, 1)
		a0 := agents[0].(map[string]interface{})
		require.Equal(t, "planner", a0["id"])
		require.Equal(t, "active", a0["health_status"])

		reasoners := a0["reasoners"].([]interface{})
		require.Len(t, reasoners, 1)
		r0 := reasoners[0].(map[string]interface{})
		require.Equal(t, "plan", r0["id"])
		require.Equal(t, "planner.plan", r0["target"])
		require.Equal(t, "Make a plan", r0["description"])
		tags := r0["tags"].([]interface{})
		require.Contains(t, tags, "entrypoint")
	})

	t.Run("health=all includes unhealthy agents", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "discover_agents", map[string]interface{}{"health": "all"})
		require.False(t, isErr)

		agents := payload["agents"].([]interface{})
		require.Len(t, agents, 2)
		byID := map[string]string{}
		for _, a := range agents {
			m := a.(map[string]interface{})
			byID[m["id"].(string)] = m["health_status"].(string)
		}
		require.Equal(t, "active", byID["planner"])
		require.Equal(t, "inactive", byID["ghost"])
	})

	t.Run("invalid health is a tool error", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "discover_agents", map[string]interface{}{"health": "zombie"})
		require.True(t, isErr)
		require.Contains(t, payload["text"], "active")
	})
}

func TestMCP_GetReasonerSchema(t *testing.T) {
	router := newMCPTestRouter(t, newMCPTestStore(mcpActiveAgent()))

	t.Run("returns input and output schema", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "get_reasoner_schema", map[string]interface{}{"node": "planner", "reasoner": "plan"})
		require.False(t, isErr)
		require.Equal(t, "planner", payload["node"])
		require.Equal(t, "plan", payload["reasoner"])

		inSchema := payload["input_schema"].(map[string]interface{})
		require.Equal(t, "object", inSchema["type"])
		props := inSchema["properties"].(map[string]interface{})
		_, hasGoal := props["goal"]
		require.True(t, hasGoal)

		outSchema := payload["output_schema"].(map[string]interface{})
		require.Equal(t, "object", outSchema["type"])
	})

	t.Run("unknown reasoner errors mentioning discover_agents", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "get_reasoner_schema", map[string]interface{}{"node": "planner", "reasoner": "nope"})
		require.True(t, isErr)
		require.Contains(t, payload["text"], "discover_agents")
	})
}

func TestMCP_ExecuteReasoner(t *testing.T) {
	store := newMCPTestStore(mcpActiveAgent())
	router := newMCPTestRouter(t, store)

	t.Run("bad target errors mentioning discover_agents", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "execute_reasoner", map[string]interface{}{
			"target": "planner.missing",
			"input":  map[string]interface{}{},
		})
		require.True(t, isErr)
		require.Contains(t, payload["text"], "discover_agents")
	})

	t.Run("malformed target errors mentioning discover_agents", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "execute_reasoner", map[string]interface{}{"target": "no-separator"})
		require.True(t, isErr)
		require.Contains(t, payload["text"], "discover_agents")
	})

	t.Run("good target returns run_id and creates an execution", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "execute_reasoner", map[string]interface{}{
			"target": "planner.plan",
			"input":  map[string]interface{}{"goal": "ship"},
		})
		require.False(t, isErr)
		require.Equal(t, "accepted", payload["status"])
		runID, _ := payload["run_id"].(string)
		require.NotEmpty(t, runID)

		execs, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{RunID: &runID})
		require.NoError(t, err)
		require.NotEmpty(t, execs)
		require.Equal(t, "plan", execs[0].ReasonerID)
	})
}

func TestMCP_ExecuteReasonerConcurrencyRejectionHasNoPersistence(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
	require.NoError(t, concurrencyLimiter.Acquire("planner"))
	defer func() { concurrencyLimiter = oldLimiter }()
	useAsyncPoolForTest(t, newAsyncWorkerPool(0, 2))
	store := newMCPTestStore(mcpActiveAgent())
	router := newMCPTestRouter(t, store)

	payload, isErr := mcpCallTool(t, router, "execute_reasoner", map[string]interface{}{
		"target": "planner.plan",
		"input":  map[string]interface{}{"goal": "ship"},
	})
	require.True(t, isErr)
	require.Contains(t, payload["text"], "reached max concurrent executions")
	execs, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, execs)
}

func TestMCP_ExecuteReasonerAlreadyStoppedPoolReturnsShutdownWithoutPersistence(t *testing.T) {
	pool := newAsyncWorkerPool(0, 2)
	pool.mu.Lock()
	pool.stopped = true
	pool.mu.Unlock()
	useAsyncPoolForTest(t, pool)
	store := newMCPTestStore(mcpActiveAgent())
	server := &mcpServer{store: store, payloads: services.NewFilePayloadStore(t.TempDir()), timeout: time.Second}

	_, _, err := server.startAsyncRun(context.Background(), "planner.plan", map[string]interface{}{"goal": "ship"}, executionHeaders{}, "", "")
	var admissionErr *executionPreconditionError
	require.ErrorAs(t, err, &admissionErr)
	require.Equal(t, ErrorCategoryControlPlaneShutdown, admissionErr.category)
	require.Equal(t, 1, admissionErr.retryAfter)
	execs, queryErr := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, queryErr)
	require.Empty(t, execs)
}

func TestMCP_ExecuteReasonerPersistencePanicReleasesSlot(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
	t.Cleanup(func() { concurrencyLimiter = oldLimiter })
	useAsyncPoolForTest(t, newAsyncWorkerPool(0, 2))
	store := &panicOnCreateMCPStore{mcpTestStore: newMCPTestStore(mcpActiveAgent())}
	server := &mcpServer{store: store, payloads: services.NewFilePayloadStore(t.TempDir()), timeout: time.Second}

	require.Panics(t, func() {
		_, _, _ = server.startAsyncRun(context.Background(), "planner.plan", map[string]interface{}{"goal": "ship"}, executionHeaders{}, "", "")
	})
	require.Zero(t, concurrencyLimiter.GetRunningCount("planner"))
}

func TestMCP_ExecuteReasonerPoolStoppedTerminatesPersistedRowsWithCancelledRequest(t *testing.T) {
	pool := newAsyncWorkerPool(0, 2)
	useAsyncPoolForTest(t, pool)
	reqCtx, cancel := context.WithCancel(context.Background())
	base := newMCPTestStore(mcpActiveAgent())
	store := &stopPoolOnCreateMCPStore{mcpTestStore: base, pool: pool, cancel: cancel}
	server := &mcpServer{store: store, payloads: services.NewFilePayloadStore(t.TempDir()), timeout: time.Second}

	_, _, err := server.startAsyncRun(reqCtx, "planner.plan", map[string]interface{}{"goal": "ship"}, executionHeaders{}, "", "")
	require.Error(t, err)
	var admissionErr *executionPreconditionError
	require.ErrorAs(t, err, &admissionErr)
	require.Equal(t, http.StatusServiceUnavailable, admissionErr.code)
	require.Equal(t, ErrorCategoryControlPlaneShutdown, admissionErr.category)
	require.Equal(t, 1, admissionErr.retryAfter)

	execs, err := base.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, execs, 1)
	require.Equal(t, types.ExecutionStatusFailed, execs[0].Status)
	require.NotNil(t, execs[0].StatusReason)
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), *execs[0].StatusReason)
	workflows, err := base.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, workflows, 1)
	require.Equal(t, string(types.ExecutionStatusFailed), workflows[0].Status)
	require.NotNil(t, workflows[0].StatusReason)
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), *workflows[0].StatusReason)
	require.Equal(t, string(execs[0].Status), workflows[0].Status)
}

func TestMCP_ExecuteReasonerPoolStoppedReturnsStructuredShutdownContract(t *testing.T) {
	pool := newAsyncWorkerPool(0, 2)
	useAsyncPoolForTest(t, pool)
	base := newMCPTestStore(mcpActiveAgent())
	store := &stopPoolOnCreateMCPStore{mcpTestStore: base, pool: pool}
	router := newMCPTestRouter(t, store)

	reqBody, err := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]interface{}{
			"name": "execute_reasoner",
			"arguments": map[string]interface{}{
				"target": "planner.plan",
				"input":  map[string]interface{}{"goal": "ship"},
			},
		},
	})
	require.NoError(t, err)
	w := mcpPost(t, router, string(reqBody))
	require.Equal(t, "1", w.Header().Get("Retry-After"))
	resp := mcpDecode(t, w)
	result := resp["result"].(map[string]interface{})
	require.Equal(t, true, result["isError"])
	structured := result["structuredContent"].(map[string]interface{})
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), structured["error_category"])
	require.Equal(t, float64(1), structured["retry_after"])
}

func TestMCP_ExecuteReasonerPoolStoppedExercisesPersistenceFailure(t *testing.T) {
	pool := newAsyncWorkerPool(0, 2)
	useAsyncPoolForTest(t, pool)
	base := newMCPTestStore(mcpActiveAgent())
	store := &stopPoolOnCreateMCPStore{mcpTestStore: base, pool: pool, updateErr: errors.New("update failed")}
	server := &mcpServer{store: store, payloads: services.NewFilePayloadStore(t.TempDir()), timeout: time.Second}

	_, _, err := server.startAsyncRun(context.Background(), "planner.plan", map[string]interface{}{"goal": "ship"}, executionHeaders{}, "", "")
	require.Error(t, err)
	var admissionErr *executionPreconditionError
	require.ErrorAs(t, err, &admissionErr)
	require.Equal(t, http.StatusServiceUnavailable, admissionErr.code)
	require.True(t, store.updated)
}

func TestMCP_ExecuteReasonerAuthorizesAndBindsRunToVerifiedCaller(t *testing.T) {
	store := newMCPTestStore(mcpActiveAgent())
	var gotCaller, gotTarget string
	router := newMCPTestRouterWithCaller(t, store, func(_ context.Context, callerDID, target string, input map[string]interface{}) (string, error) {
		gotCaller, gotTarget = callerDID, target
		require.Equal(t, map[string]interface{}{"goal": "ship"}, input)
		return "did:web:example.com:agents:planner", nil
	}, "did:web:example.com:agents:caller")

	payload, isErr := mcpCallTool(t, router, "execute_reasoner", map[string]interface{}{
		"target": "planner.plan",
		"input":  map[string]interface{}{"goal": "ship"},
	})
	require.False(t, isErr)
	require.Equal(t, "did:web:example.com:agents:caller", gotCaller)
	require.Equal(t, "planner.plan", gotTarget)
	runID := payload["run_id"].(string)
	execs, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{RunID: &runID})
	require.NoError(t, err)
	require.NotNil(t, execs[0].ActorID)
	require.Equal(t, "did:web:example.com:agents:caller", *execs[0].ActorID)

	// A different DID cannot turn a known run ID into a read capability.
	router = newMCPTestRouterWithCaller(t, store, nil, "did:web:example.com:agents:other")
	read, readErr := mcpCallTool(t, router, "get_run", map[string]interface{}{"run_id": runID})
	require.True(t, readErr)
	require.Contains(t, read["text"], "not found")
}

func TestMCP_GetRun(t *testing.T) {
	store := newMCPTestStore(mcpActiveAgent())
	router := newMCPTestRouter(t, store)

	runID := "run-seed-1"
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID:   "exec-1",
		RunID:         runID,
		NodeID:        "planner",
		ReasonerID:    "plan",
		Status:        types.ExecutionStatusSucceeded,
		ResultPayload: json.RawMessage(`{"answer":42}`),
	}))

	t.Run("returns status, result and per-execution summaries", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "get_run", map[string]interface{}{"run_id": runID})
		require.False(t, isErr)
		require.Equal(t, "succeeded", payload["status"])

		result := payload["result"].(map[string]interface{})
		require.Equal(t, float64(42), result["answer"])

		execs := payload["executions"].([]interface{})
		require.Len(t, execs, 1)
		e0 := execs[0].(map[string]interface{})
		require.Equal(t, "plan", e0["reasoner_id"])
		require.Equal(t, "succeeded", e0["status"])
	})

	t.Run("unknown run is a tool error", func(t *testing.T) {
		payload, isErr := mcpCallTool(t, router, "get_run", map[string]interface{}{"run_id": "does-not-exist"})
		require.True(t, isErr)
		require.Contains(t, payload["text"], "not found")
	})

	t.Run("failed run surfaces error and failed status", func(t *testing.T) {
		failRun := "run-failed"
		msg := "boom"
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID:  "exec-fail",
			RunID:        failRun,
			NodeID:       "planner",
			ReasonerID:   "plan",
			Status:       types.ExecutionStatusFailed,
			ErrorMessage: &msg,
		}))
		payload, isErr := mcpCallTool(t, router, "get_run", map[string]interface{}{"run_id": failRun})
		require.False(t, isErr)
		require.Equal(t, "failed", payload["status"])
		require.Equal(t, "boom", payload["error"])
	})
}

func TestMCP_WaitRun(t *testing.T) {
	store := newMCPTestStore(mcpActiveAgent())
	router := newMCPTestRouter(t, store)

	// Shrink the poll interval so the timeout path resolves quickly.
	old := mcpWaitPollInterval
	mcpWaitPollInterval = 10 * time.Millisecond
	defer func() { mcpWaitPollInterval = old }()

	t.Run("times out on a never-finishing run", func(t *testing.T) {
		runID := "run-forever"
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-forever",
			RunID:       runID,
			NodeID:      "planner",
			ReasonerID:  "plan",
			Status:      types.ExecutionStatusRunning,
		}))

		payload, isErr := mcpCallTool(t, router, "wait_run", map[string]interface{}{"run_id": runID, "timeout_seconds": 1})
		require.False(t, isErr)
		require.Equal(t, true, payload["timed_out"])
		require.Equal(t, "running", payload["status"])
	})

	t.Run("returns immediately for a terminal run", func(t *testing.T) {
		runID := "run-done"
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID:   "exec-done",
			RunID:         runID,
			NodeID:        "planner",
			ReasonerID:    "plan",
			Status:        types.ExecutionStatusSucceeded,
			ResultPayload: json.RawMessage(`{"ok":true}`),
		}))

		payload, isErr := mcpCallTool(t, router, "wait_run", map[string]interface{}{"run_id": runID, "timeout_seconds": 120})
		require.False(t, isErr)
		require.Equal(t, false, payload["timed_out"])
		require.Equal(t, "succeeded", payload["status"])
	})
}
