package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionLogger_AdditionalBranches(t *testing.T) {
	t.Run("nil agent and nil logger paths are safe", func(t *testing.T) {
		var a *Agent
		logger := a.executionLogger(context.Background(), " sdk.user ")
		require.NotNil(t, logger)
		assert.Equal(t, " sdk.user ", logger.source)

		var nilLogger *ExecutionLogger
		nilLogger.Emit("info", "event", "message", nil, false)
		assert.Nil(t, nilLogger.WithSource("other"))

		a.logExecution(context.Background(), "info", "event", "message", nil, true)
	})

	t.Run("entry normalizes defaults and source overrides", func(t *testing.T) {
		a := &Agent{cfg: Config{NodeID: "node-logs"}}
		ctx := contextWithExecution(context.Background(), ExecutionContext{
			RunID:             "run-1",
			ParentExecutionID: "parent-1",
			ReasonerName:      "demo",
			SessionID:         "session-1",
			ActorID:           "actor-1",
			Depth:             2,
		})

		logger := a.ExecutionLogger(ctx).WithSource(" custom.source ")
		entry := logger.entry("", "", "", nil, true)
		assert.Equal(t, "info", entry.Level)
		assert.Equal(t, "log", entry.EventType)
		assert.Equal(t, "log", entry.Message)
		assert.Equal(t, "custom.source", entry.Source)
		assert.Equal(t, "run-1", entry.WorkflowID)
		assert.Equal(t, "run-1", entry.RootWorkflowID)
		assert.Equal(t, "node-logs", entry.AgentNodeID)
		assert.Equal(t, "demo", entry.ReasonerID)
		assert.Equal(t, "session-1", entry.SessionID)
		assert.Equal(t, "actor-1", entry.ActorID)
		assert.Equal(t, 2, entry.Depth)
		assert.Equal(t, map[string]any{}, entry.Attributes)

		stdout, _, err := captureOutput(t, func() error {
			logger.Emit("WARN", "evt", "hello", map[string]any{"ok": true}, false)
			a.logExecutionInfo(ctx, "runtime.info", "", nil)
			a.logExecutionWarn(ctx, "runtime.warn", "", nil)
			a.logExecutionError(ctx, "runtime.error", "", map[string]any{"code": 1})
			return nil
		})
		require.NoError(t, err)
		assert.Contains(t, stdout, `"source":"custom.source"`)
		assert.Contains(t, stdout, `"event_type":"runtime.info"`)
		assert.Contains(t, stdout, `"event_type":"runtime.warn"`)
		assert.Contains(t, stdout, `"event_type":"runtime.error"`)
		assert.True(t, strings.Contains(stdout, `"level":"warn"`) || strings.Contains(stdout, `"level":"error"`))
	})
}
