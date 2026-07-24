package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func setupExecutionLogsStorage(t *testing.T) (*storage.LocalStorage, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping execution log handler tests")
		}
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

func TestExecutionLogsUtilityHelpers(t *testing.T) {
	require.Nil(t, parseCSVQuery(" "))
	require.Equal(t, []string{"info", "error"}, parseCSVQuery(" info, , error "))

	timer := time.NewTimer(time.Hour)
	t.Cleanup(func() {
		timer.Stop()
	})
	resetTimer(timer, 0)
	resetTimer(timer, 5*time.Millisecond)
	select {
	case <-timer.C:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timer did not reset")
	}
}

func TestExecutionLogsHandlerValidationAndSuccess(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("returns configuration and parameter errors", func(t *testing.T) {
		handler := NewExecutionLogsHandler(nil, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{MaxTailEntries: 2}
		})

		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/logs", nil)
		ctx.Params = gin.Params{{Key: "execution_id", Value: "exec-1"}}
		handler.GetExecutionLogsHandler(ctx)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)

		store, _ := setupExecutionLogsStorage(t)
		handler = NewExecutionLogsHandler(store, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{MaxTailEntries: 2}
		})

		for name, target := range map[string]string{
			"missing id":       "/api/ui/v1/executions//logs",
			"invalid tail":     "/api/ui/v1/executions/exec-1/logs?tail=abc",
			"tail too large":   "/api/ui/v1/executions/exec-1/logs?tail=9",
			"invalid after seq": "/api/ui/v1/executions/exec-1/logs?after_seq=-1",
		} {
			t.Run(name, func(t *testing.T) {
				recorder := httptest.NewRecorder()
				ctx, _ := gin.CreateTestContext(recorder)
				ctx.Request = httptest.NewRequest(http.MethodGet, target, nil)
				if name != "missing id" {
					ctx.Params = gin.Params{{Key: "execution_id", Value: "exec-1"}}
				}
				handler.GetExecutionLogsHandler(ctx)
				require.Equal(t, http.StatusBadRequest, recorder.Code)
			})
		}
	})

	t.Run("lists structured execution logs with filters", func(t *testing.T) {
		store, ctx := setupExecutionLogsStorage(t)
		runID := "run-1"
		reasoner := "planner"
		now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
		entries := []*types.ExecutionLogEntry{
			{
				ExecutionID: "exec-1",
				WorkflowID:  "wf-1",
				RunID:       &runID,
				Sequence:    1,
				AgentNodeID: "node-1",
				ReasonerID:  &reasoner,
				Level:       "info",
				Source:      "sdk",
				Message:     "booted",
				EmittedAt:   now,
			},
			{
				ExecutionID: "exec-1",
				WorkflowID:  "wf-1",
				RunID:       &runID,
				Sequence:    2,
				AgentNodeID: "node-2",
				ReasonerID:  &reasoner,
				Level:       "error",
				Source:      "agent",
				Message:     "failed validation",
				EmittedAt:   now.Add(time.Second),
			},
		}
		for _, entry := range entries {
			require.NoError(t, store.StoreExecutionLogEntry(ctx, entry))
		}

		handler := NewExecutionLogsHandler(store, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{MaxTailEntries: 5}
		})

		recorder := httptest.NewRecorder()
		ctxGin, _ := gin.CreateTestContext(recorder)
		ctxGin.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/logs?tail=2&after_seq=0&levels=error&node_ids=node-2&sources=agent&q=validation", nil)
		ctxGin.Params = gin.Params{{Key: "execution_id", Value: "exec-1"}}

		handler.GetExecutionLogsHandler(ctxGin)
		require.Equal(t, http.StatusOK, recorder.Code)

		var response executionLogsResponse
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
		require.Len(t, response.Entries, 1)
		require.Equal(t, int64(2), response.Entries[0].Sequence)
		require.Equal(t, "failed validation", response.Entries[0].Message)
	})
}

func TestExecutionLogsAuxHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := NewExecutionLogsHandler(nil, nil, func() config.ExecutionLogsConfig {
		return config.ExecutionLogsConfig{}
	})

	t.Run("reports queue status without limiter", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/queue", nil)
		handler.GetExecutionQueueStatusHandler(ctx)
		require.Equal(t, http.StatusOK, recorder.Code)
		require.Contains(t, recorder.Body.String(), `"enabled":false`)
	})

	t.Run("reports llm health as disabled when no monitor is configured", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/llm/health", nil)
		handler.GetLLMHealthHandler(ctx)
		require.Equal(t, http.StatusOK, recorder.Code)
		require.Contains(t, recorder.Body.String(), `"enabled":false`)
	})
}
