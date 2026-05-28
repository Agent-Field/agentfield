package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

type executionLogEnvelope struct {
	Entries []types.ExecutionLogEntry `json:"entries"`
}

// ExecutionLogStore defines the storage needed for structured execution log ingestion.
type ExecutionLogStore interface {
	StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error
	PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error
}

type executionLogBatchStore interface {
	StoreExecutionLogEntries(ctx context.Context, executionID string, entries []*types.ExecutionLogEntry) error
}

// StructuredExecutionLogsHandler ingests structured execution logs emitted by SDK runtimes.
// POST /api/v1/executions/:execution_id/logs
func StructuredExecutionLogsHandler(store ExecutionLogStore, snapshot func() config.ExecutionLogsConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		executionID := strings.TrimSpace(c.Param("execution_id"))
		if executionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
			return
		}

		const maxBodyBytes int64 = 10 << 20 // 10 MiB
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBodyBytes)
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body too large", "max_bytes": maxBodyBytes})
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read body"})
			}
			return
		}
		if len(body) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "request body is required"})
			return
		}

		entries, err := decodeExecutionLogEntries(body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid payload: %v", err)})
			return
		}
		if len(entries) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "at least one execution log entry is required"})
			return
		}

		cfg := config.EffectiveExecutionLogs(snapshot())
		ctx := c.Request.Context()
		pruneBefore := time.Time{}
		if cfg.RetentionPeriod > 0 {
			pruneBefore = time.Now().UTC().Add(-cfg.RetentionPeriod)
		}

		prepared := make([]types.ExecutionLogEntry, 0, len(entries))
		for i := range entries {
			entry := entries[i]
			entry.ExecutionID = strings.TrimSpace(entry.ExecutionID)
			if entry.ExecutionID == "" {
				entry.ExecutionID = executionID
			}
			if entry.ExecutionID != executionID {
				c.JSON(http.StatusBadRequest, gin.H{
					"error":          "execution_id_mismatch",
					"path_execution": executionID,
					"body_execution": entry.ExecutionID,
				})
				return
			}
			if strings.TrimSpace(entry.Message) == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "message is required"})
				return
			}
			if strings.TrimSpace(entry.AgentNodeID) == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "agent_node_id is required"})
				return
			}
			if strings.TrimSpace(entry.WorkflowID) == "" {
				entry.WorkflowID = executionID
			}
			if entry.RootWorkflowID == nil || strings.TrimSpace(*entry.RootWorkflowID) == "" {
				rootWorkflowID := entry.WorkflowID
				entry.RootWorkflowID = &rootWorkflowID
			}
			if len(entry.Attributes) == 0 {
				entry.Attributes = json.RawMessage("{}")
			}

			prepared = append(prepared, entry)
		}

		if batchStore, ok := store.(executionLogBatchStore); ok {
			ptrs := make([]*types.ExecutionLogEntry, 0, len(prepared))
			for i := range prepared {
				ptrs = append(ptrs, &prepared[i])
			}
			if err := batchStore.StoreExecutionLogEntries(ctx, executionID, ptrs); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to store execution log entry batch: %v", err)})
				return
			}
		} else {
			for i := range prepared {
				if err := store.StoreExecutionLogEntry(ctx, &prepared[i]); err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to store execution log entry: %v", err)})
					return
				}
			}
		}

		if err := store.PruneExecutionLogEntries(ctx, executionID, cfg.MaxEntriesPerExecution, pruneBefore); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to prune execution logs: %v", err)})
			return
		}

		c.JSON(http.StatusAccepted, gin.H{
			"success":      true,
			"execution_id": executionID,
			"accepted":     len(prepared),
		})
	}
}

func decodeExecutionLogEntries(body []byte) ([]types.ExecutionLogEntry, error) {
	var envelope executionLogEnvelope
	if err := json.Unmarshal(body, &envelope); err == nil && len(envelope.Entries) > 0 {
		return envelope.Entries, nil
	}

	var single types.ExecutionLogEntry
	if err := json.Unmarshal(body, &single); err != nil {
		return nil, err
	}
	return []types.ExecutionLogEntry{single}, nil
}
