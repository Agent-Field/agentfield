package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestNewExecutionControllerPreservesAgentCallTimeout(t *testing.T) {
	store := newTestExecutionStorage(nil)

	for _, timeout := range []time.Duration{-time.Second, 0, 30 * time.Second} {
		t.Run(timeout.String(), func(t *testing.T) {
			controller := newExecutionController(store, nil, nil, timeout, "")
			require.Equal(t, timeout, controller.timeout)
			require.Equal(t, timeout, controller.httpClient.Timeout)
		})
	}
}

func TestExecuteHandlerAgentCallTimeoutCanBeDisabled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(300 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":42}`))
	}))
	defer agentServer.Close()

	for _, tc := range []struct {
		name     string
		timeout  time.Duration
		wantCode int
	}{
		{name: "short timeout", timeout: 100 * time.Millisecond, wantCode: http.StatusGatewayTimeout},
		{name: "disabled timeout", timeout: 0, wantCode: http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			agent := &types.AgentNode{
				ID:        "node-1",
				BaseURL:   agentServer.URL,
				Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
			}
			store := newTestExecutionStorage(agent)
			router := gin.New()
			router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, tc.timeout, ""))

			req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tc.wantCode, resp.Code, resp.Body.String())
		})
	}
}

func TestExecuteHandlerDisabledTimeoutWaitsForAsyncCompletion(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var store *testExecutionStorage
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		executionID := r.Header.Get("X-Execution-ID")
		require.NotEmpty(t, executionID)
		go func() {
			time.Sleep(50 * time.Millisecond)
			_, err := store.UpdateExecutionRecord(context.Background(), executionID, func(current *types.Execution) (*types.Execution, error) {
				now := time.Now().UTC()
				current.Status = types.ExecutionStatusSucceeded
				current.ResultPayload = []byte(`{"answer":42}`)
				current.CompletedAt = &now
				return current, nil
			})
			require.NoError(t, err)
			store.GetExecutionEventBus().Publish(events.ExecutionEvent{
				Type: events.ExecutionCompleted, ExecutionID: executionID,
			})
		}()
		w.WriteHeader(http.StatusAccepted)
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{ID: "node-1", BaseURL: agentServer.URL, Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store = newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, 0, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
	require.Contains(t, resp.Body.String(), `"answer":42`)
}
