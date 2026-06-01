package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type updateRetryStore struct {
	*testExecutionStorage
	failures int
	calls    int
}

func (s *updateRetryStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	s.calls++
	if s.calls <= s.failures {
		return nil, fmt.Errorf("SQLITE_BUSY: temporary lock")
	}
	return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
}

type heartbeatErrorStore struct {
	*nodeRESTStorageStub
	updateHeartbeatErr error
}

func (s *heartbeatErrorStore) UpdateAgentHeartbeat(ctx context.Context, id, version string, ts time.Time) error {
	if s.updateHeartbeatErr != nil {
		return s.updateHeartbeatErr
	}
	return s.nodeRESTStorageStub.UpdateAgentHeartbeat(ctx, id, version, ts)
}

func TestDetermineTargetTypeAdditionalCoverage(t *testing.T) {
	t.Parallel()

	agent := &types.AgentNode{
		ID: "agent-1",
		Reasoners: []types.ReasonerDefinition{
			{ID: "summarize"},
		},
		Skills: []types.SkillDefinition{
			{ID: "web_search"},
		},
	}

	tests := []struct {
		name       string
		targetName string
		wantType   string
		wantErr    string
	}{
		{name: "reasoner match", targetName: "summarize", wantType: "reasoner"},
		{name: "skill match", targetName: "web_search", wantType: "skill"},
		{name: "missing target", targetName: "missing", wantErr: "target 'missing' not found on agent 'agent-1'"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := determineTargetType(agent, tt.targetName)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantType, got)
		})
	}
}

func TestWaitForResume_PreSubscriptionBranches(t *testing.T) {
	t.Parallel()

	t.Run("event bus missing", func(t *testing.T) {
		controller := &executionController{}
		err := controller.waitForResume(context.Background(), "exec-missing-bus")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "event bus not available")
	})

	t.Run("cancelled before subscription", func(t *testing.T) {
		store := newTestExecutionStorage(nil)
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-cancelled",
			Status:      types.ExecutionStatusCancelled,
			CreatedAt:   time.Now().UTC(),
			StartedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}))

		controller := newExecutionController(store, nil, nil, time.Second, "")
		err := controller.waitForResume(context.Background(), "exec-cancelled")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "execution cancelled")
	})

	t.Run("already resumed before subscription", func(t *testing.T) {
		store := newTestExecutionStorage(nil)
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-resumed",
			Status:      types.ExecutionStatusRunning,
			CreatedAt:   time.Now().UTC(),
			StartedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}))

		controller := newExecutionController(store, nil, nil, time.Second, "")
		require.NoError(t, controller.waitForResume(context.Background(), "exec-resumed"))
	})
}

func TestExecutionController_CompletionAndFailureCoverage(t *testing.T) {
	t.Parallel()

	agent := &types.AgentNode{
		ID:              "node-1",
		BaseURL:         "https://example.com",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
	}

	t.Run("complete execution persists payload and marks succeeded", func(t *testing.T) {
		store := newTestExecutionStorage(agent)
		payloads := services.NewFilePayloadStore(t.TempDir())
		controller := newExecutionController(store, payloads, nil, time.Second, "")

		now := time.Now().UTC()
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-success",
			RunID:       "run-success",
			AgentNodeID: agent.ID,
			Status:      types.ExecutionStatusRunning,
			InputPayload: []byte(`{"prompt":"hello"}`),
			CreatedAt:   now,
			StartedAt:   now,
			UpdatedAt:   now,
		}))

		plan := &preparedExecution{
			exec: &types.Execution{
				ExecutionID: "exec-success",
				RunID:       "run-success",
				InputPayload: []byte(`{"prompt":"hello"}`),
			},
			agent:  agent,
			target: &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a"},
		}

		require.NoError(t, controller.completeExecution(context.Background(), plan, []byte(`{"ok":true}`), 125*time.Millisecond))

		record, err := store.GetExecutionRecord(context.Background(), "exec-success")
		require.NoError(t, err)
		require.NotNil(t, record)
		assert.Equal(t, types.ExecutionStatusSucceeded, record.Status)
		assert.JSONEq(t, `{"ok":true}`, string(record.ResultPayload))
		require.NotNil(t, record.ResultURI)
		assert.True(t, strings.HasPrefix(*record.ResultURI, "payload://"))
		require.NotNil(t, record.CompletedAt)
		require.NotNil(t, record.DurationMS)
		assert.Equal(t, int64(125), *record.DurationMS)
	})

	t.Run("fail execution persists payload and marks failed", func(t *testing.T) {
		store := newTestExecutionStorage(agent)
		payloads := services.NewFilePayloadStore(t.TempDir())
		controller := newExecutionController(store, payloads, nil, time.Second, "")

		now := time.Now().UTC()
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-fail",
			RunID:       "run-fail",
			AgentNodeID: agent.ID,
			Status:      types.ExecutionStatusRunning,
			CreatedAt:   now,
			StartedAt:   now,
			UpdatedAt:   now,
		}))

		plan := &preparedExecution{
			exec: &types.Execution{
				ExecutionID: "exec-fail",
				RunID:       "run-fail",
			},
			agent:  agent,
			target: &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a"},
		}

		require.NoError(t, controller.failExecution(context.Background(), plan, errors.New("agent exploded"), 250*time.Millisecond, []byte(`{"detail":"boom"}`)))

		record, err := store.GetExecutionRecord(context.Background(), "exec-fail")
		require.NoError(t, err)
		require.NotNil(t, record)
		assert.Equal(t, types.ExecutionStatusFailed, record.Status)
		require.NotNil(t, record.ErrorMessage)
		assert.Contains(t, *record.ErrorMessage, "agent exploded")
		require.NotNil(t, record.StatusReason)
		assert.Equal(t, string(ErrorCategoryInternal), *record.StatusReason)
		require.NotNil(t, record.ResultURI)
		assert.True(t, strings.HasPrefix(*record.ResultURI, "payload://"))
	})
}

func TestPrepareExecution_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("versioned serverless agent registers webhook and preserves headers", func(t *testing.T) {
		versionedAgent := &types.AgentNode{
			ID:              "node-v",
			BaseURL:         "https://agent.example.com/base/",
			Version:         "v2",
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{"serverless": "true"},
			},
		}
		store := &versionedExecutionStore{
			testExecutionStorage: newTestExecutionStorage(nil),
			versions:             []*types.AgentNode{versionedAgent},
			getAgentErr:          errors.New("missing default agent"),
		}
		controller := newExecutionController(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, "")

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Params = gin.Params{{Key: "target", Value: "node-v.reasoner-a"}}
		ctx.Request = httptest.NewRequest(http.MethodPost, "/execute/node-v.reasoner-a", strings.NewReader(`{
			"input":{"prompt":"hello"},
			"context":{"llm_endpoint":"openai"},
			"webhook":{"url":"https://example.com/webhook","secret":"top-secret"}
		}`))
		ctx.Request.Header.Set("Content-Type", "application/json")
		ctx.Request.Header.Set("X-Run-ID", "run-header")
		ctx.Request.Header.Set("X-Parent-Execution-ID", "parent-1")
		ctx.Request.Header.Set("X-Session-ID", "session-1")
		ctx.Request.Header.Set("X-Actor-ID", "actor-1")

		plan, err := controller.prepareExecution(context.Background(), ctx)
		require.NoError(t, err)
		require.NotNil(t, plan)
		assert.Equal(t, "v2", plan.routedVersion)
		assert.Equal(t, "reasoner", plan.targetType)
		assert.Equal(t, "openai", plan.llmEndpoint)
		assert.True(t, plan.webhookRegistered)
		assert.Equal(t, "run-header", plan.exec.RunID)
		assert.Equal(t, "session-1", *plan.exec.SessionID)
		assert.Equal(t, "actor-1", *plan.exec.ActorID)
		require.NotNil(t, plan.exec.InputURI)
		assert.Equal(t, "serverless", plan.agent.DeploymentType)
		require.NotNil(t, plan.agent.InvocationURL)
		assert.Equal(t, "https://agent.example.com/base/execute", *plan.agent.InvocationURL)
		assert.Contains(t, string(plan.requestBody), `"target":"reasoner-a"`)
		assert.Contains(t, string(plan.requestBody), `"type":"reasoner"`)
		assert.Contains(t, string(plan.requestBody), `"parent_execution_id":"parent-1"`)
		assert.Contains(t, string(plan.requestBody), `"session_id":"session-1"`)
		assert.Contains(t, string(plan.requestBody), `"actor_id":"actor-1"`)
	})

	t.Run("blocks calls to pending_approval agent with 503 (TC-034)", func(t *testing.T) {
		pendingAgent := &types.AgentNode{
			ID:              "node-revoked",
			BaseURL:         "https://agent.example.com",
			Version:         "v1",
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusPendingApproval,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
		}
		store := newTestExecutionStorage(pendingAgent)
		controller := newExecutionController(store, nil, nil, time.Second, "")

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Params = gin.Params{{Key: "target", Value: "node-revoked.reasoner-a"}}
		ctx.Request = httptest.NewRequest(http.MethodPost, "/execute/node-revoked.reasoner-a", strings.NewReader(`{"input":{}}`))
		ctx.Request.Header.Set("Content-Type", "application/json")

		_, err := controller.prepareExecution(context.Background(), ctx)
		require.Error(t, err)

		var pe *executionPreconditionError
		require.ErrorAs(t, err, &pe)
		assert.Equal(t, http.StatusServiceUnavailable, pe.HTTPStatusCode())
		assert.Equal(t, ErrorCategoryAgentError, pe.Category())
		assert.Equal(t, "agent_pending_approval", pe.ErrorCode())
		assert.Contains(t, pe.Error(), "node-revoked")
		assert.Contains(t, pe.Error(), "awaiting tag approval")

		// Verify the wire-level response contract matches the sibling handlers
		// (reasoners/skills/permission middleware): stable code in `error`,
		// human text in `message`, 503 Service Unavailable.
		writeExecutionError(ctx, err)
		assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
		var body map[string]interface{}
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		assert.Equal(t, "agent_pending_approval", body["error"])
		assert.Equal(t, "agent_error", body["error_category"])
		require.Contains(t, body, "message")
		assert.Contains(t, body["message"], "awaiting tag approval")

		// No execution record should have been persisted before the guard fired.
		store.mu.Lock()
		defer store.mu.Unlock()
		assert.Empty(t, store.executionRecords, "no execution record should be created for a blocked call")
	})

	t.Run("returns error for invalid target and invalid body", func(t *testing.T) {
		store := newTestExecutionStorage(&types.AgentNode{
			ID:        "node-1",
			BaseURL:   "https://example.com",
			Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
		})
		controller := newExecutionController(store, nil, nil, time.Second, "")

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Params = gin.Params{{Key: "target", Value: "bad-target"}}
		ctx.Request = httptest.NewRequest(http.MethodPost, "/execute/bad-target", strings.NewReader(`{"input":{}}`))
		ctx.Request.Header.Set("Content-Type", "application/json")
		_, err := controller.prepareExecution(context.Background(), ctx)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid target")

		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Params = gin.Params{{Key: "target", Value: "node-1.reasoner-a"}}
		ctx.Request = httptest.NewRequest(http.MethodPost, "/execute/node-1.reasoner-a", strings.NewReader(`{`))
		ctx.Request.Header.Set("Content-Type", "application/json")
		_, err = controller.prepareExecution(context.Background(), ctx)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid request body")
	})
}

func TestAsyncExecutionJob_ProcessFallbackCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	previousLimiter := concurrencyLimiter
	previousQueue := completionQueue
	t.Setenv("AGENTFIELD_EXEC_COMPLETION_QUEUE", os.Getenv("AGENTFIELD_EXEC_COMPLETION_QUEUE"))
	t.Cleanup(func() {
		concurrencyLimiter = previousLimiter
		completionQueue = previousQueue
		completionOnce = sync.Once{}
	})

	tests := []struct {
		name           string
		statusCode     int
		responseBody   string
		wantStatus     types.ExecutionStatus
		wantErrorMatch string
	}{
		{
			name:         "fallback completion when queue is full",
			statusCode:   http.StatusOK,
			responseBody: `{"result":"ok"}`,
			wantStatus:   types.ExecutionStatusSucceeded,
		},
		{
			name:           "fallback failure when queue is full",
			statusCode:     http.StatusInternalServerError,
			responseBody:   `{"error":"broken"}`,
			wantStatus:     types.ExecutionStatusFailed,
			wantErrorMatch: "agent error (500)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save and restore per-subtest to avoid ordering dependencies.
			prevLimiter := concurrencyLimiter
			prevQueue := completionQueue
			t.Cleanup(func() {
				concurrencyLimiter = prevLimiter
				completionQueue = prevQueue
				completionOnce = sync.Once{}
			})

			concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
			require.NoError(t, concurrencyLimiter.Acquire("node-queue"))

			completionQueue = make(chan completionJob)
			completionOnce = sync.Once{}
			completionOnce.Do(func() {})

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(tt.responseBody))
			}))
			defer server.Close()

			agent := &types.AgentNode{
				ID:              "node-queue",
				BaseURL:         server.URL,
				Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
				HealthStatus:    types.HealthStatusActive,
				LifecycleStatus: types.AgentStatusReady,
			}

			store := newTestExecutionStorage(agent)
			now := time.Now().UTC()
			require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
				ExecutionID: "exec-queue",
				RunID:       "run-queue",
				AgentNodeID: agent.ID,
				Status:      types.ExecutionStatusRunning,
				CreatedAt:   now,
				StartedAt:   now,
				UpdatedAt:   now,
			}))

			job := asyncExecutionJob{
				controller: newExecutionController(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""),
				plan: preparedExecution{
					exec:        &types.Execution{ExecutionID: "exec-queue", RunID: "run-queue"},
					requestBody: []byte(`{"prompt":"hello"}`),
					agent:       agent,
					target:      &parsedTarget{NodeID: agent.ID, TargetName: "reasoner-a", TargetType: "reasoner"},
				},
			}

			job.process()

			record, err := store.GetExecutionRecord(context.Background(), "exec-queue")
			require.NoError(t, err)
			require.NotNil(t, record)
			assert.Equal(t, tt.wantStatus, record.Status)
			if tt.wantErrorMatch != "" {
				require.NotNil(t, record.ErrorMessage)
				assert.Contains(t, *record.ErrorMessage, tt.wantErrorMatch)
			}
			assert.Equal(t, int64(0), concurrencyLimiter.GetRunningCount("node-queue"))
		})
	}
}

func TestExecuteHelpers_AdditionalCoverage(t *testing.T) {
	t.Parallel()

	t.Run("buildAgentURL variants", func(t *testing.T) {
		invocationURL := "https://invoke.example.com/run"
		agent := &types.AgentNode{BaseURL: "https://agent.example.com/base/", InvocationURL: &invocationURL}
		assert.Equal(t, "", buildAgentURL(nil, &parsedTarget{TargetName: "reasoner-a"}))
		assert.Equal(t, invocationURL, buildAgentURL(agent, &parsedTarget{TargetName: "reasoner-a"}))

		agent.InvocationURL = nil
		agent.DeploymentType = "serverless"
		assert.Equal(t, "https://agent.example.com/base/execute", buildAgentURL(agent, &parsedTarget{TargetName: "reasoner-a"}))

		agent.DeploymentType = ""
		assert.Equal(t, "https://agent.example.com/base/skills/skill-a", buildAgentURL(agent, &parsedTarget{TargetName: "skill-a", TargetType: "skill"}))
		assert.Equal(t, "https://agent.example.com/base/reasoners/reasoner-a", buildAgentURL(agent, &parsedTarget{TargetName: "reasoner-a", TargetType: "reasoner"}))
	})

	t.Run("truncateForLog and resolveIntFromEnv", func(t *testing.T) {
		assert.Equal(t, "short", truncateForLog([]byte("short")))
		assert.True(t, strings.HasSuffix(truncateForLog([]byte(strings.Repeat("a", 1025))), "..."))

		// Use os.Setenv instead of t.Setenv because the parent test uses t.Parallel.
		key := "AGENTFIELD_TEST_INT"
		orig, hadOrig := os.LookupEnv(key)
		t.Cleanup(func() {
			if hadOrig {
				os.Setenv(key, orig)
			} else {
				os.Unsetenv(key)
			}
		})
		os.Unsetenv(key)
		assert.Equal(t, 7, resolveIntFromEnv(key, 7))
		os.Setenv(key, "invalid")
		assert.Equal(t, 9, resolveIntFromEnv(key, 9))
		os.Setenv(key, "42")
		assert.Equal(t, 42, resolveIntFromEnv(key, 1))
	})

	t.Run("savePayload and triggerWebhook", func(t *testing.T) {
		controller := &executionController{}
		assert.Nil(t, controller.savePayload(context.Background(), nil))
		assert.Nil(t, controller.savePayload(context.Background(), []byte{}))

		controller.payloads = services.NewFilePayloadStore(t.TempDir())
		uri := controller.savePayload(context.Background(), []byte(`{"ok":true}`))
		require.NotNil(t, uri)
		assert.True(t, strings.HasPrefix(*uri, "payload://"))

		controller.triggerWebhook("")
		controller.webhooks = &mockWebhookDispatcher{
			notifyFunc: func(ctx context.Context, executionID string) error {
				if executionID == "exec-ok" {
					return nil
				}
				return errors.New("notify failed")
			},
		}
		controller.triggerWebhook("exec-err")
		controller.triggerWebhook("exec-ok")
	})
}

func TestExecutionController_BatchStatusAndRetryCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("batch status rejects invalid json", func(t *testing.T) {
		router := gin.New()
		router.POST("/executions/batch", BatchExecutionStatusHandler(newTestExecutionStorage(nil)))

		req := httptest.NewRequest(http.MethodPost, "/executions/batch", strings.NewReader(`{`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "error")
	})

	t.Run("complete execution retries retryable update errors", func(t *testing.T) {
		agent := &types.AgentNode{ID: "node-retry"}
		base := newTestExecutionStorage(agent)
		store := &updateRetryStore{testExecutionStorage: base, failures: 1}
		controller := newExecutionController(store, nil, nil, time.Second, "")

		now := time.Now().UTC()
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-retry",
			RunID:       "run-retry",
			Status:      types.ExecutionStatusRunning,
			CreatedAt:   now,
			StartedAt:   now,
			UpdatedAt:   now,
		}))

		plan := &preparedExecution{
			exec:   &types.Execution{ExecutionID: "exec-retry", RunID: "run-retry"},
			agent:  agent,
			target: &parsedTarget{NodeID: "node-retry", TargetName: "reasoner-a"},
		}

		require.NoError(t, controller.completeExecution(context.Background(), plan, []byte(`{"ok":true}`), 10*time.Millisecond))
		assert.Equal(t, 2, store.calls)

		record, err := store.GetExecutionRecord(context.Background(), "exec-retry")
		require.NoError(t, err)
		require.NotNil(t, record)
		assert.Equal(t, types.ExecutionStatusSucceeded, record.Status)
	})
}

func TestDiscoveryAndNodeHandlers_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("discovery storage error returns 500", func(t *testing.T) {
		InvalidateDiscoveryCache()
		lister := &stubAgentLister{err: errors.New("storage unavailable")}
		router := gin.New()
		router.GET("/api/v1/discovery/capabilities", DiscoveryCapabilitiesHandler(lister))

		req := httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		assert.Contains(t, rec.Body.String(), "internal_error")
	})

	t.Run("get node uses version lookup and rejects missing node id", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			versionedAgent: &types.AgentNode{ID: "node-v", Version: "v2"},
		}
		router := gin.New()
		router.GET("/nodes/:node_id", GetNodeHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes/node-v?version=v2", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"version":"v2"`)

		rec2 := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec2)
		c.Request = httptest.NewRequest(http.MethodGet, "/", nil)
		GetNodeHandler(store)(c)
		require.Equal(t, http.StatusBadRequest, rec2.Code)
		assert.Contains(t, rec2.Body.String(), "node_id is required")
	})

	t.Run("direct helpers cover parseBool and normalized serverless url errors", func(t *testing.T) {
		trueValue, err := parseBool(" YeS ")
		require.NoError(t, err)
		assert.True(t, trueValue)

		falseValue, err := parseBool("0")
		require.NoError(t, err)
		assert.False(t, falseValue)

		_, err = parseBool("maybe")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid boolean")

		_, err = normalizeServerlessDiscoveryURL("ftp://example.com", []string{"example.com"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "http or https")

		_, err = normalizeServerlessDiscoveryURL("https://user@example.com", []string{"example.com"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "must not include user info")
	})

	t.Run("register node covers explicit callback defaults and reregistration approval carry-over", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/health" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"status":"ok"}`))
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		t.Run("new registration", func(t *testing.T) {
			presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})
			store := &nodeRESTStorageStub{}
			router := gin.New()
			router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, presence, nil, nil))

			req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(fmt.Sprintf(`{
				"id":"node-new",
				"base_url":"%s",
				"reasoners":[{"id":"reasoner-a","proposed_tags":["alpha","beta"]}],
				"skills":[{"id":"skill-a","tags":["gamma"]}]
			}`, server.URL)))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusCreated, rec.Code)
			require.NotNil(t, store.registeredAgent)
			assert.Equal(t, "node-new", store.registeredAgent.GroupID)
			assert.Equal(t, types.AgentStatusStarting, store.registeredAgent.LifecycleStatus)
			assert.Equal(t, types.HealthStatusUnknown, store.registeredAgent.HealthStatus)
			require.NotNil(t, store.registeredAgent.CallbackDiscovery)
			assert.Equal(t, "auto", store.registeredAgent.CallbackDiscovery.Mode)
			assert.Equal(t, server.URL, store.registeredAgent.CallbackDiscovery.Preferred)
			assert.Equal(t, server.URL, store.registeredAgent.CallbackDiscovery.Resolved)
			assert.Equal(t, []string{"alpha", "beta"}, store.registeredAgent.Reasoners[0].Tags)
			assert.Equal(t, []string{"alpha", "beta"}, store.registeredAgent.Reasoners[0].ProposedTags)
			assert.Equal(t, []string{"gamma"}, store.registeredAgent.Skills[0].Tags)
			assert.Equal(t, []string{"gamma"}, store.registeredAgent.Skills[0].ProposedTags)
			assert.True(t, presence.HasLease("node-new"))
		})

		t.Run("re-registration preserves approved tags and resets offline lifecycle", func(t *testing.T) {
			existing := &types.AgentNode{
				ID:              "node-old",
				BaseURL:         server.URL,
				Version:         "v1",
				ApprovedTags:    []string{"alpha", "gamma"},
				LifecycleStatus: types.AgentStatusOffline,
			}
			store := &nodeRESTStorageStub{agent: existing}
			router := gin.New()
			router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

			req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(fmt.Sprintf(`{
				"id":"node-old",
				"base_url":"%s",
				"reasoners":[{"id":"reasoner-a","proposed_tags":["alpha","beta"]}],
				"skills":[{"id":"skill-a","proposed_tags":["gamma","delta"]}]
			}`, server.URL)))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusCreated, rec.Code)
			require.NotNil(t, store.registeredAgent)
			assert.Equal(t, types.AgentStatusStarting, store.registeredAgent.LifecycleStatus)
			assert.Equal(t, []string{"alpha", "gamma"}, store.registeredAgent.ApprovedTags)
			assert.Equal(t, []string{"alpha"}, store.registeredAgent.Reasoners[0].ApprovedTags)
			assert.Equal(t, []string{"gamma"}, store.registeredAgent.Skills[0].ApprovedTags)
		})
	})

	t.Run("heartbeat covers version fallback and async update errors", func(t *testing.T) {
		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

		versionStore := &nodeRESTStorageStub{
			versionedAgent: &types.AgentNode{
				ID:              "node-versioned",
				Version:         "v2",
				BaseURL:         "https://example.com",
				LifecycleStatus: types.AgentStatusStarting,
			},
		}

		router := gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(versionStore, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-versioned/heartbeat", strings.NewReader(`{"version":"v2","status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Eventually(t, func() bool { return len(versionStore.heartbeats) == 1 }, 5*time.Second, 50*time.Millisecond)
		assert.Equal(t, "v2", versionStore.lastVersion)
		require.NotNil(t, versionStore.updatedLifecycle)
		assert.Equal(t, types.AgentStatusReady, *versionStore.updatedLifecycle)

		errStore := &heartbeatErrorStore{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{ID: "node-error", Version: "v1"},
			},
			updateHeartbeatErr: errors.New("write failed"),
		}

		processHeartbeatAsync(errStore, nil, "node-error", "v1", &CachedNodeData{LastDBUpdate: time.Now().UTC()})
		// Assert that the async goroutine does not record a heartbeat on error.
		require.Never(t, func() bool { return len(errStore.heartbeats) > 0 }, 200*time.Millisecond, 10*time.Millisecond)
	})
}
