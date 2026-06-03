package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// ExecutionLogsHandler handles live and recent structured execution logs.
type ExecutionLogsHandler struct {
	storage          storage.StorageProvider
	llmHealthMonitor *services.LLMHealthMonitor
	Snapshot         func() config.ExecutionLogsConfig
}

// NewExecutionLogsHandler creates a new ExecutionLogsHandler.
func NewExecutionLogsHandler(store storage.StorageProvider, llmHealthMonitor *services.LLMHealthMonitor, snapshot func() config.ExecutionLogsConfig) *ExecutionLogsHandler {
	return &ExecutionLogsHandler{
		storage:          store,
		llmHealthMonitor: llmHealthMonitor,
		Snapshot:         snapshot,
	}
}

type executionLogsResponse struct {
	Entries []types.ExecutionLogEntry `json:"entries"`
}

func parseCSVQuery(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	raw := strings.Split(value, ",")
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if trimmed := strings.TrimSpace(item); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

// GetExecutionLogsHandler returns a recent or filtered tail of structured execution logs.
// GET /api/ui/v1/executions/:execution_id/logs
func (h *ExecutionLogsHandler) GetExecutionLogsHandler(c *gin.Context) {
	if h.storage == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "not_configured"})
		return
	}
	executionID := strings.TrimSpace(c.Param("execution_id"))
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}
	maxTail := config.EffectiveExecutionLogs(h.Snapshot()).MaxTailEntries
	limit := maxTail
	if raw := strings.TrimSpace(c.Query("tail")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_tail"})
			return
		}
		if n > maxTail {
			c.JSON(http.StatusBadRequest, gin.H{"error": "tail_too_large", "max_allowed": maxTail})
			return
		}
		limit = n
	}
	var afterSeq *int64
	if raw := strings.TrimSpace(c.Query("after_seq")); raw != "" {
		n, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || n < 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_after_seq"})
			return
		}
		afterSeq = &n
	}
	entries, err := h.storage.ListExecutionLogEntries(
		c.Request.Context(),
		executionID,
		afterSeq,
		limit,
		parseCSVQuery(c.Query("levels")),
		parseCSVQuery(c.Query("node_ids")),
		parseCSVQuery(c.Query("sources")),
		c.Query("q"),
	)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to list execution logs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed_to_list_execution_logs"})
		return
	}
	response := executionLogsResponse{Entries: make([]types.ExecutionLogEntry, 0, len(entries))}
	for _, entry := range entries {
		if entry != nil {
			response.Entries = append(response.Entries, *entry)
		}
	}
	c.JSON(http.StatusOK, response)
}

// StreamExecutionLogsHandler streams structured execution logs for a specific execution via SSE.
// GET /api/ui/v1/executions/:executionId/logs/stream
func (h *ExecutionLogsHandler) StreamExecutionLogsHandler(c *gin.Context) {
	if h.storage == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "not_configured"})
		return
	}
	executionID := strings.TrimSpace(c.Param("execution_id"))
	if executionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "executionId is required"})
		return
	}

	cfg := config.EffectiveExecutionLogs(h.Snapshot())
	maxTail := cfg.MaxTailEntries
	tail := 0
	if raw := strings.TrimSpace(c.Query("tail")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_tail"})
			return
		}
		if n > maxTail {
			c.JSON(http.StatusBadRequest, gin.H{"error": "tail_too_large", "max_allowed": maxTail})
			return
		}
		tail = n
	}
	var sinceSeq *int64
	if raw := strings.TrimSpace(c.Query("since_seq")); raw != "" {
		n, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || n < 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_since_seq"})
			return
		}
		sinceSeq = &n
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	subscriberID := fmt.Sprintf("exec_logs_%s_%d_%s", executionID, time.Now().UnixNano(), c.ClientIP())
	eventChan := h.storage.GetExecutionLogEventBus().Subscribe(subscriberID)
	defer h.storage.GetExecutionLogEventBus().Unsubscribe(subscriberID)

	if initial, err := h.storage.ListExecutionLogEntries(
		c.Request.Context(),
		executionID,
		sinceSeq,
		tail,
		parseCSVQuery(c.Query("levels")),
		parseCSVQuery(c.Query("node_ids")),
		parseCSVQuery(c.Query("sources")),
		c.Query("q"),
	); err == nil {
		for _, entry := range initial {
			if entry == nil {
				continue
			}
			payload, marshalErr := json.Marshal(entry)
			if marshalErr != nil {
				continue
			}
			if !writeSSE(c, payload) {
				return
			}
			if sinceSeq == nil || entry.Sequence > *sinceSeq {
				next := entry.Sequence
				sinceSeq = &next
			}
		}
	}

	initialEvent := map[string]interface{}{
		"type":         "connected",
		"execution_id": executionID,
		"message":      "Execution log stream connected",
		"timestamp":    time.Now().Format(time.RFC3339),
	}
	if eventJSON, err := json.Marshal(initialEvent); err == nil {
		if !writeSSE(c, eventJSON) {
			return
		}
	}

	streamCtx := c.Request.Context()
	if cfg.MaxStreamDuration > 0 {
		var cancel context.CancelFunc
		streamCtx, cancel = context.WithTimeout(c.Request.Context(), cfg.MaxStreamDuration)
		defer cancel()
	}

	heartbeatTicker := time.NewTicker(30 * time.Second)
	defer heartbeatTicker.Stop()
	idleTimer := time.NewTimer(cfg.StreamIdleTimeout)
	defer idleTimer.Stop()

	// Check if execution is already in a terminal state before streaming.
	if exec, err := h.storage.GetWorkflowExecution(c.Request.Context(), executionID); err == nil && exec != nil {
		if types.IsTerminalExecutionStatus(exec.Status) {
			closeEvt, _ := json.Marshal(map[string]interface{}{
				"type": "stream_end", "reason": "terminal_status", "status": exec.Status,
			})
			writeSSE(c, closeEvt)
			return
		}
	}

	for {
		select {
		case entry, ok := <-eventChan:
			if !ok {
				return
			}
			if entry == nil || entry.ExecutionID != executionID {
				continue
			}
			if sinceSeq != nil && entry.Sequence <= *sinceSeq {
				continue
			}
			payload, err := json.Marshal(entry)
			if err != nil {
				logger.Logger.Error().Err(err).Msg("error marshalling execution log entry")
				continue
			}
			if !writeSSE(c, payload) {
				return
			}
			next := entry.Sequence
			sinceSeq = &next
			resetTimer(idleTimer, cfg.StreamIdleTimeout)
		case <-heartbeatTicker.C:
			// Check if execution reached terminal state; close stream if so.
			if exec, err := h.storage.GetWorkflowExecution(streamCtx, executionID); err == nil && exec != nil {
				if types.IsTerminalExecutionStatus(exec.Status) {
					closeEvt, _ := json.Marshal(map[string]interface{}{
						"type": "stream_end", "reason": "terminal_status", "status": exec.Status,
					})
					writeSSE(c, closeEvt)
					return
				}
			}
			heartbeat := map[string]interface{}{
				"type":      "heartbeat",
				"timestamp": time.Now().Format(time.RFC3339),
			}
			if heartbeatJSON, err := json.Marshal(heartbeat); err == nil {
				if !writeSSE(c, heartbeatJSON) {
					return
				}
			}
			// Don't reset idle timer on heartbeats — only real log entries should.
		case <-idleTimer.C:
			return
		case <-streamCtx.Done():
			if !errors.Is(streamCtx.Err(), http.ErrAbortHandler) {
				logger.Logger.Debug().Str("execution_id", executionID).Msg("execution log SSE client disconnected")
			}
			return
		}
	}
}

func resetTimer(timer *time.Timer, d time.Duration) {
	if d <= 0 {
		return
	}
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
	timer.Reset(d)
}

// GetExecutionQueueStatusHandler returns concurrency slot usage per agent and overall queue health.
// GET /api/ui/v1/executions/queue
func (h *ExecutionLogsHandler) GetExecutionQueueStatusHandler(c *gin.Context) {
	limiter := handlers.GetConcurrencyLimiter()
	maxPerAgent := 0
	counts := map[string]int64{}
	if limiter != nil {
		maxPerAgent = limiter.MaxPerAgent()
		counts = limiter.GetAllCounts()
	}

	type agentSlot struct {
		AgentNodeID string `json:"agent_node_id"`
		Running     int64  `json:"running"`
		Max         int    `json:"max"`
		Available   int    `json:"available"`
	}

	agents := make([]agentSlot, 0, len(counts))
	totalRunning := int64(0)
	for agentID, running := range counts {
		avail := maxPerAgent - int(running)
		if avail < 0 {
			avail = 0
		}
		agents = append(agents, agentSlot{
			AgentNodeID: agentID,
			Running:     running,
			Max:         maxPerAgent,
			Available:   avail,
		})
		totalRunning += running
	}

	c.JSON(http.StatusOK, gin.H{
		"enabled":       maxPerAgent > 0,
		"max_per_agent": maxPerAgent,
		"total_running": totalRunning,
		"agents":        agents,
		"checked_at":    time.Now().Format(time.RFC3339),
	})
}

// GetLLMHealthHandler returns the health status of all configured LLM endpoints.
// GET /api/ui/v1/llm/health
func (h *ExecutionLogsHandler) GetLLMHealthHandler(c *gin.Context) {
	if h.llmHealthMonitor == nil {
		c.JSON(http.StatusOK, gin.H{
			"enabled":   false,
			"endpoints": []interface{}{},
		})
		return
	}

	statuses := h.llmHealthMonitor.GetAllStatuses()
	anyHealthy := h.llmHealthMonitor.IsAnyEndpointHealthy()

	c.JSON(http.StatusOK, gin.H{
		"enabled":    true,
		"healthy":    anyHealthy,
		"endpoints":  statuses,
		"checked_at": time.Now().Format(time.RFC3339),
	})
}
