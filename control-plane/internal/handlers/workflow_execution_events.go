package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// WorkflowExecutionEventRequest describes the payload emitted by agents when a reasoner
// starts, completes, or fails inside an existing workflow run.
type WorkflowExecutionEventRequest struct {
	ExecutionID       string                 `json:"execution_id" binding:"required"`
	WorkflowID        string                 `json:"workflow_id"`
	RunID             string                 `json:"run_id"`
	ReasonerID        string                 `json:"reasoner_id"`
	Type              string                 `json:"type"`
	AgentNodeID       string                 `json:"agent_node_id"`
	Status            string                 `json:"status"`
	ParentExecutionID *string                `json:"parent_execution_id"`
	ParentWorkflowID  *string                `json:"parent_workflow_id"`
	InputData         map[string]interface{} `json:"input_data"`
	Result            interface{}            `json:"result"`
	Error             string                 `json:"error"`
	DurationMS        *int64                 `json:"duration_ms"`
}

// WorkflowExecutionEventHandler ingests workflow step events emitted by agents and mirrors
// them into the executions table so the UI can render full DAGs, even when steps are executed
// locally within the same process.
func WorkflowExecutionEventHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req WorkflowExecutionEventRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid payload: %v", err)})
			return
		}

		ctx := c.Request.Context()
		now := time.Now().UTC()

		existing, err := store.GetExecutionRecord(ctx, req.ExecutionID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to load execution: %v", err)})
			return
		}

		if existing == nil {
			if err := store.CreateExecutionRecord(ctx, buildExecutionRecordFromEvent(&req, now)); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to create execution: %v", err)})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "created": true})
			return
		}

		_, err = store.UpdateExecutionRecord(ctx, req.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return buildExecutionRecordFromEvent(&req, now), nil
			}
			applyEventToExecution(current, &req, now)
			return current, nil
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to update execution: %v", err)})
			return
		}

		c.JSON(http.StatusOK, gin.H{"success": true, "updated": true})
	}
}

func buildExecutionRecordFromEvent(req *WorkflowExecutionEventRequest, now time.Time) *types.Execution {
	runID := firstNonEmpty(req.RunID, req.WorkflowID, req.ExecutionID)
	agentNodeID := req.AgentNodeID
	if agentNodeID == "" {
		agentNodeID = req.Type
	}

	reasonerID := firstNonEmpty(req.ReasonerID, req.Type, "reasoner")
	status := types.NormalizeExecutionStatus(req.Status)
	inputPayload := marshalJSON(req.InputData)
	resultPayload := marshalJSON(req.Result)

	exec := &types.Execution{
		ExecutionID:       req.ExecutionID,
		RunID:             runID,
		ParentExecutionID: req.ParentExecutionID,
		AgentNodeID:       agentNodeID,
		NodeID:            agentNodeID,
		ReasonerID:        reasonerID,
		Status:            status,
		InputPayload:      inputPayload,
		ResultPayload:     resultPayload,
		StartedAt:         now,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	if req.DurationMS != nil {
		exec.DurationMS = req.DurationMS
	}

	if req.Error != "" {
		errCopy := req.Error
		exec.ErrorMessage = &errCopy
	}

	if types.IsTerminalExecutionStatus(status) {
		completed := now
		exec.CompletedAt = &completed
	}

	return exec
}

func applyEventToExecution(current *types.Execution, req *WorkflowExecutionEventRequest, now time.Time) {
	current.Status = types.NormalizeExecutionStatus(req.Status)
	current.UpdatedAt = now

	if current.StartedAt.IsZero() {
		current.StartedAt = now
	}

	if req.ParentExecutionID != nil {
		current.ParentExecutionID = req.ParentExecutionID
	}

	if current.AgentNodeID == "" {
		current.AgentNodeID = firstNonEmpty(req.AgentNodeID, req.Type)
		current.NodeID = current.AgentNodeID
	}
	if current.ReasonerID == "" {
		current.ReasonerID = firstNonEmpty(req.ReasonerID, req.Type, "reasoner")
	}
	if req.RunID != "" || req.WorkflowID != "" {
		current.RunID = firstNonEmpty(req.RunID, req.WorkflowID, current.RunID)
	}

	if payload := marshalJSON(req.InputData); payload != nil && len(payload) > 0 {
		current.InputPayload = payload
	}
	if result := marshalJSON(req.Result); result != nil && len(result) > 0 {
		current.ResultPayload = result
	}

	if req.DurationMS != nil {
		current.DurationMS = req.DurationMS
	}

	if req.Error != "" {
		errCopy := req.Error
		current.ErrorMessage = &errCopy
	} else if types.NormalizeExecutionStatus(req.Status) == string(types.ExecutionStatusSucceeded) {
		current.ErrorMessage = nil
	}

	if types.IsTerminalExecutionStatus(req.Status) {
		completed := now
		current.CompletedAt = &completed
	}
}

func marshalJSON(value interface{}) json.RawMessage {
	if value == nil {
		return nil
	}
	bytes, err := json.Marshal(value)
	if err != nil || len(bytes) == 0 || string(bytes) == "null" {
		return nil
	}
	return json.RawMessage(bytes)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if trimmed := strings.TrimSpace(v); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
