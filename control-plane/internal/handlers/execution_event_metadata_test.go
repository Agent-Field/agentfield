package handlers

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestEnrichExecutionLifecycleData(t *testing.T) {
	duration := int64(125)
	parentID := "parent-exec"
	orphanReason := "agent_restart_orphaned: node restarted"

	tests := []struct {
		name           string
		exec           *types.Execution
		status         string
		wantRoot       bool
		wantOutcome    string
		wantCategory   string
		wantDurationMS interface{}
	}{
		{
			name: "successful root",
			exec: &types.Execution{
				DurationMS: &duration,
			},
			status:         types.ExecutionStatusSucceeded,
			wantRoot:       true,
			wantOutcome:    "succeeded",
			wantDurationMS: duration,
		},
		{
			name: "failed child with canonical reason prefix",
			exec: &types.Execution{
				ParentExecutionID: &parentID,
				StatusReason:      &orphanReason,
			},
			status:       types.ExecutionStatusFailed,
			wantRoot:     false,
			wantOutcome:  "failed",
			wantCategory: "agent_restart_orphaned",
		},
		{
			name:         "timeout",
			exec:         &types.Execution{},
			status:       types.ExecutionStatusTimeout,
			wantRoot:     true,
			wantOutcome:  "timeout",
			wantCategory: "timeout",
		},
		{
			name:         "cancelled",
			exec:         &types.Execution{},
			status:       types.ExecutionStatusCancelled,
			wantRoot:     true,
			wantOutcome:  "cancelled",
			wantCategory: "cancelled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := map[string]interface{}{}
			enrichExecutionLifecycleData(data, tt.exec, tt.status)

			require.Equal(t, tt.wantRoot, data["is_root_execution"])
			require.Equal(t, tt.wantOutcome, data["outcome"])
			if tt.wantCategory != "" {
				require.Equal(t, tt.wantCategory, data["failure_category"])
			}
			if tt.wantDurationMS != nil {
				require.Equal(t, tt.wantDurationMS, data["duration_ms"])
			}
		})
	}
}

func TestPublishExecutionStartedEventCarriesProductDimensions(t *testing.T) {
	bus := events.NewExecutionEventBus()
	ch := bus.Subscribe("metadata-test")
	defer bus.Unsubscribe("metadata-test")

	now := time.Now().UTC()
	store := newTestExecutionStorage(nil)
	controller := &executionController{eventBus: bus, store: store}
	controller.publishExecutionStartedEvent(&preparedExecution{
		exec: &types.Execution{
			ExecutionID: "exec-1",
			RunID:       "run-1",
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner-1",
			StartedAt:   now,
		},
		target:        &parsedTarget{TargetName: "reasoner-1"},
		targetType:    "reasoner",
		executionMode: "async",
	})

	select {
	case event := <-ch:
		require.Equal(t, events.ExecutionStarted, event.Type)
		data, ok := event.Data.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, "reasoner", data["target_type"])
		require.Equal(t, "async", data["execution_mode"])
		require.Equal(t, "execution_controller", data["transition_source"])
		require.Equal(t, true, data["is_root_execution"])
		require.Equal(t, 0, data["workflow_depth"])
	default:
		t.Fatal("expected execution started event")
	}
}
