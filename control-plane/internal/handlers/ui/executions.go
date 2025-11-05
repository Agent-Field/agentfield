package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/services"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type executionRecordStore interface {
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
}

// ExecutionHandler provides handlers for agent execution history operations.
type ExecutionHandler struct {
	store    executionRecordStore
	payloads services.PayloadStore
	storage  storage.StorageProvider
	webhooks services.WebhookDispatcher
}

func writeSSE(c *gin.Context, payload []byte) bool {
	if _, err := c.Writer.WriteString("data: " + string(payload) + "\n\n"); err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to write SSE payload")
		return false
	}
	c.Writer.Flush()
	return true
}

// NewExecutionHandler creates a new ExecutionHandler.
func NewExecutionHandler(store storage.StorageProvider, payloadStore services.PayloadStore, webhooks services.WebhookDispatcher) *ExecutionHandler {
	return &ExecutionHandler{
		store:    store,
		payloads: payloadStore,
		storage:  store,
		webhooks: webhooks,
	}
}

// StreamWorkflowNodeNotesHandler handles SSE connections for workflow node notes.
// GET /api/ui/v1/workflows/:workflowId/notes/events
func (h *ExecutionHandler) StreamWorkflowNodeNotesHandler(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	workflowID := c.Param("workflowId")
	if workflowID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "workflowId is required"})
		return
	}

	subscriberID := fmt.Sprintf("sse_notes_%d_%s", time.Now().UnixNano(), workflowID)
	eventBus := h.storage.GetExecutionEventBus()
	eventChan := eventBus.Subscribe(subscriberID)
	defer eventBus.Unsubscribe(subscriberID)

	initialEvent := map[string]interface{}{
		"type":        "connected",
		"workflow_id": workflowID,
		"message":     "Workflow node notes stream connected",
		"timestamp":   time.Now().Format(time.RFC3339),
	}
	if payload, err := json.Marshal(initialEvent); err == nil {
		if !writeSSE(c, payload) {
			return
		}
	}

	ctx := c.Request.Context()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			heartbeat := map[string]interface{}{
				"type":      "heartbeat",
				"timestamp": time.Now().Format(time.RFC3339),
			}
			if payload, err := json.Marshal(heartbeat); err == nil {
				if !writeSSE(c, payload) {
					return
				}
			}
		case event, ok := <-eventChan:
			if !ok {
				return
			}
			if payload, err := json.Marshal(event); err == nil {
				if !writeSSE(c, payload) {
					return
				}
			}
		}
	}
}

// ExecutionListResponse represents the response for listing executions.
type ExecutionListResponse struct {
	Executions []ExecutionSummary `json:"executions"`
	Total      int                `json:"total"`
	Page       int                `json:"page"`
	PageSize   int                `json:"page_size"`
	TotalPages int                `json:"total_pages"`
}

// ExecutionSummary represents execution summary information in the list.
type ExecutionSummary struct {
	ID           int64                `json:"id"`
	ExecutionID  string               `json:"execution_id"`
	WorkflowID   string               `json:"workflow_id"`
	SessionID    *string              `json:"session_id,omitempty"`
	AgentNodeID  string               `json:"agent_node_id"`
	ReasonerID   string               `json:"reasoner_id"`
	Status       string               `json:"status"`
	DurationMS   int                  `json:"duration_ms"`
	InputSize    int                  `json:"input_size"`
	OutputSize   int                  `json:"output_size"`
	ErrorMessage *string              `json:"error_message,omitempty"`
	CreatedAt    time.Time            `json:"created_at"`
	NotesCount   int                  `json:"notes_count"`
	LatestNote   *types.ExecutionNote `json:"latest_note,omitempty"`
}

// ExecutionStatsResponse represents execution statistics.
type ExecutionStatsResponse struct {
	TotalExecutions    int            `json:"total_executions"`
	SuccessfulCount    int            `json:"successful_count"`
	FailedCount        int            `json:"failed_count"`
	RunningCount       int            `json:"running_count"`
	AverageDurationMS  float64        `json:"average_duration_ms"`
	ExecutionsByStatus map[string]int `json:"executions_by_status"`
	ExecutionsByAgent  map[string]int `json:"executions_by_agent"`
}

// ExecutionDetailsResponse represents detailed execution information.
type ExecutionDetailsResponse struct {
	ID                  int64                          `json:"id"`
	ExecutionID         string                         `json:"execution_id"`
	WorkflowID          string                         `json:"workflow_id"`
	AgentFieldRequestID *string                        `json:"agentfield_request_id,omitempty"`
	SessionID           *string                        `json:"session_id,omitempty"`
	ActorID             *string                        `json:"actor_id,omitempty"`
	AgentNodeID         string                         `json:"agent_node_id"`
	ParentWorkflowID    *string                        `json:"parent_workflow_id,omitempty"`
	RootWorkflowID      *string                        `json:"root_workflow_id,omitempty"`
	WorkflowDepth       *int                           `json:"workflow_depth,omitempty"`
	ReasonerID          string                         `json:"reasoner_id"`
	InputData           interface{}                    `json:"input_data"`
	OutputData          interface{}                    `json:"output_data"`
	InputSize           int                            `json:"input_size"`
	OutputSize          int                            `json:"output_size"`
	WorkflowName        *string                        `json:"workflow_name,omitempty"`
	WorkflowTags        []string                       `json:"workflow_tags"`
	Status              string                         `json:"status"`
	StartedAt           *string                        `json:"started_at,omitempty"`
	CompletedAt         *string                        `json:"completed_at,omitempty"`
	DurationMS          *int                           `json:"duration_ms,omitempty"`
	ErrorMessage        *string                        `json:"error_message,omitempty"`
	RetryCount          int                            `json:"retry_count"`
	CreatedAt           string                         `json:"created_at"`
	UpdatedAt           *string                        `json:"updated_at,omitempty"`
	Notes               []types.ExecutionNote          `json:"notes"`
	NotesCount          int                            `json:"notes_count"`
	LatestNote          *types.ExecutionNote           `json:"latest_note,omitempty"`
	WebhookRegistered   bool                           `json:"webhook_registered"`
	WebhookEvents       []*types.ExecutionWebhookEvent `json:"webhook_events,omitempty"`
}

type EnhancedExecution struct {
	ExecutionID     string  `json:"execution_id"`
	WorkflowID      string  `json:"workflow_id"`
	Status          string  `json:"status"`
	TaskName        string  `json:"task_name"`
	WorkflowName    string  `json:"workflow_name"`
	AgentName       string  `json:"agent_name"`
	RelativeTime    string  `json:"relative_time"`
	DurationDisplay string  `json:"duration_display"`
	WorkflowContext *string `json:"workflow_context,omitempty"`
	StartedAt       string  `json:"started_at"`
	CompletedAt     *string `json:"completed_at,omitempty"`
	DurationMS      *int64  `json:"duration_ms,omitempty"`
	SessionID       *string `json:"session_id,omitempty"`
	ActorID         *string `json:"actor_id,omitempty"`
}

type EnhancedExecutionsResponse struct {
	Executions []EnhancedExecution `json:"executions"`
	TotalCount int                 `json:"total_count"`
	Page       int                 `json:"page"`
	PageSize   int                 `json:"page_size"`
	TotalPages int                 `json:"total_pages"`
	HasMore    bool                `json:"has_more"`
}

// ListExecutionsHandler handles requests for listing agent executions.
// GET /api/ui/v1/agents/:agentId/executions
func (h *ExecutionHandler) ListExecutionsHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := strings.TrimSpace(c.Param("agentId"))
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	page := parsePositiveIntOrDefault(c.Query("page"), 1)
	pageSize := parseBoundedIntOrDefault(c.Query("pageSize"), 10, 1, 100)
	status := strings.TrimSpace(c.Query("status"))
	runID := strings.TrimSpace(c.Query("workflowId"))
	sortField := sanitizeExecutionSortField(c.DefaultQuery("sortBy", "started_at"))
	sortDesc := strings.ToLower(c.DefaultQuery("sortOrder", "desc")) != "asc"

	filter := types.ExecutionFilter{
		AgentNodeID:    &agentID,
		Limit:          pageSize,
		Offset:         (page - 1) * pageSize,
		SortBy:         sortField,
		SortDescending: sortDesc,
	}
	if status != "" {
		filter.Status = &status
	}
	if runID != "" {
		filter.RunID = &runID
	}

	execs, err := h.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to query executions: " + err.Error()})
		return
	}

	summaries := make([]ExecutionSummary, 0, len(execs))
	for _, exec := range execs {
		summaries = append(summaries, h.toExecutionSummary(exec))
	}

	totalPages := page
	if len(execs) == pageSize {
		totalPages = page + 1
	}

	response := ExecutionListResponse{
		Executions: summaries,
		Total:      len(summaries),
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(http.StatusOK, response)
}

// GetExecutionDetailsHandler handles requests for getting detailed execution information for a given agent.
// GET /api/ui/v1/agents/:agentId/executions/:executionId
func (h *ExecutionHandler) GetExecutionDetailsHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := strings.TrimSpace(c.Param("agentId"))
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	executionID := strings.TrimSpace(c.Param("executionId"))
	if executionID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "executionId is required"})
		return
	}

	exec, err := h.store.GetExecutionRecord(ctx, executionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load execution: " + err.Error()})
		return
	}
	if exec == nil || exec.AgentNodeID != agentID {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "execution not found for this agent"})
		return
	}

	c.JSON(http.StatusOK, h.toExecutionDetails(ctx, exec))
}

// GetExecutionsSummaryHandler handles global execution summary requests.
// GET /api/ui/v1/executions/summary
func (h *ExecutionHandler) GetExecutionsSummaryHandler(c *gin.Context) {
	ctx := c.Request.Context()
	page := parsePositiveIntOrDefault(c.Query("page"), 1)
	pageSize := parseBoundedIntOrDefault(c.Query("page_size"), 20, 1, 100)
	status := strings.TrimSpace(c.Query("status"))
	runID := strings.TrimSpace(c.Query("workflow_id"))
	agentID := strings.TrimSpace(c.Query("agent_node_id"))
	sessionID := strings.TrimSpace(c.Query("session_id"))
	groupBy := strings.TrimSpace(c.Query("group_by"))
	startTime, err := parseTimePtrValue(c.Query("start_time"))
	if err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid start_time format, expected RFC3339"})
		return
	}
	endTime, err := parseTimePtrValue(c.Query("end_time"))
	if err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid end_time format, expected RFC3339"})
		return
	}

	filter := types.ExecutionFilter{
		Limit:          pageSize,
		Offset:         (page - 1) * pageSize,
		SortBy:         "started_at",
		SortDescending: true,
		StartTime:      startTime,
		EndTime:        endTime,
	}
	if status != "" {
		filter.Status = &status
	}
	if runID != "" {
		filter.RunID = &runID
	}
	if agentID != "" {
		filter.AgentNodeID = &agentID
	}
	if sessionID != "" {
		filter.SessionID = &sessionID
	}

	execs, queryErr := h.store.QueryExecutionRecords(ctx, filter)
	if queryErr != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to query executions: " + queryErr.Error()})
		return
	}

	summaries := make([]ExecutionSummary, 0, len(execs))
	for _, exec := range execs {
		summaries = append(summaries, h.toExecutionSummary(exec))
	}

	if groupBy != "" && groupBy != "none" {
		c.JSON(http.StatusOK, gin.H{
			"grouped":   h.groupExecutionSummaries(summaries, groupBy),
			"total":     len(summaries),
			"page":      page,
			"page_size": pageSize,
		})
		return
	}

	response := ExecutionsSummaryResponse{
		Executions: summaries,
		Total:      len(summaries),
		Page:       page,
		PageSize:   pageSize,
		TotalPages: computeTotalPages(len(summaries), pageSize),
	}

	c.JSON(http.StatusOK, response)
}

// ExecutionsSummaryResponse represents the summary response body.
type ExecutionsSummaryResponse struct {
	Executions []ExecutionSummary `json:"executions"`
	Total      int                `json:"total"`
	Page       int                `json:"page"`
	PageSize   int                `json:"page_size"`
	TotalPages int                `json:"total_pages"`
}

// GetExecutionStatsHandler handles execution statistics requests.
// GET /api/ui/v1/executions/stats
func (h *ExecutionHandler) GetExecutionStatsHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := strings.TrimSpace(c.Query("agent_node_id"))
	sessionID := strings.TrimSpace(c.Query("session_id"))
	runID := strings.TrimSpace(c.Query("workflow_id"))

	filter := types.ExecutionFilter{
		Limit:          1000,
		SortBy:         "started_at",
		SortDescending: true,
	}
	if agentID != "" {
		filter.AgentNodeID = &agentID
	}
	if sessionID != "" {
		filter.SessionID = &sessionID
	}
	if runID != "" {
		filter.RunID = &runID
	}

	execs, err := h.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to query executions: " + err.Error()})
		return
	}

	stats := ExecutionStatsResponse{
		TotalExecutions:    len(execs),
		ExecutionsByStatus: make(map[string]int),
		ExecutionsByAgent:  make(map[string]int),
	}

	var totalDuration int64
	for _, exec := range execs {
		status := types.NormalizeExecutionStatus(exec.Status)
		stats.ExecutionsByStatus[status]++
		stats.ExecutionsByAgent[exec.AgentNodeID]++

		switch status {
		case string(types.ExecutionStatusSucceeded):
			stats.SuccessfulCount++
		case string(types.ExecutionStatusFailed):
			stats.FailedCount++
		case string(types.ExecutionStatusRunning), string(types.ExecutionStatusPending), string(types.ExecutionStatusQueued):
			stats.RunningCount++
		}

		if exec.DurationMS != nil {
			totalDuration += *exec.DurationMS
		}
	}

	if stats.TotalExecutions > 0 {
		stats.AverageDurationMS = float64(totalDuration) / float64(stats.TotalExecutions)
	}

	c.JSON(http.StatusOK, stats)
}

// GetEnhancedExecutionsHandler provides the flattened execution list used by the enhanced executions view.
// GET /api/ui/v1/executions/enhanced
func (h *ExecutionHandler) GetEnhancedExecutionsHandler(c *gin.Context) {
	ctx := c.Request.Context()

	page := parsePositiveIntOrDefault(c.Query("page"), 1)
	limit := parseBoundedIntOrDefault(c.Query("limit"), 50, 1, 200)
	offset := (page - 1) * limit

	filter := types.ExecutionFilter{
		Limit:          limit,
		Offset:         offset,
		SortBy:         sanitizeExecutionSortField(c.DefaultQuery("sort_by", "started_at")),
		SortDescending: strings.ToLower(c.DefaultQuery("sort_order", "desc")) != "asc",
	}

	if status := strings.TrimSpace(c.Query("status")); status != "" {
		normalized := types.NormalizeExecutionStatus(status)
		filter.Status = &normalized
	}
	if agentID := strings.TrimSpace(c.Query("agent_id")); agentID != "" {
		filter.AgentNodeID = &agentID
	}
	if workflowID := strings.TrimSpace(c.Query("workflow_id")); workflowID != "" {
		filter.RunID = &workflowID
	}
	if sessionID := strings.TrimSpace(c.Query("session_id")); sessionID != "" {
		filter.SessionID = &sessionID
	}
	if actorID := strings.TrimSpace(c.Query("actor_id")); actorID != "" {
		filter.ActorID = &actorID
	}
	if since := strings.TrimSpace(c.Query("since")); since != "" {
		if ts, err := time.Parse(time.RFC3339, since); err == nil {
			filter.StartTime = &ts
		}
	}

	executions, err := h.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to query executions: " + err.Error()})
		return
	}

	now := time.Now().UTC()
	items := make([]EnhancedExecution, 0, len(executions))
	for _, exec := range executions {
		if exec == nil {
			continue
		}

		startedAt := exec.StartedAt.UTC()
		var completedAt *string
		if exec.CompletedAt != nil {
			formatted := exec.CompletedAt.UTC().Format(time.RFC3339)
			completedAt = &formatted
		}

		items = append(items, EnhancedExecution{
			ExecutionID:     exec.ExecutionID,
			WorkflowID:      exec.RunID,
			Status:          types.NormalizeExecutionStatus(exec.Status),
			TaskName:        exec.ReasonerID,
			WorkflowName:    exec.RunID,
			AgentName:       exec.AgentNodeID,
			RelativeTime:    formatRelativeTimeString(now, startedAt),
			DurationDisplay: formatDurationDisplay(exec.DurationMS),
			StartedAt:       startedAt.Format(time.RFC3339),
			CompletedAt:     completedAt,
			DurationMS:      exec.DurationMS,
			SessionID:       exec.SessionID,
			ActorID:         exec.ActorID,
		})
	}

	hasMore := len(executions) == limit
	totalCount := offset + len(executions)
	totalPages := computeTotalPages(totalCount, limit)

	response := EnhancedExecutionsResponse{
		Executions: items,
		TotalCount: totalCount,
		Page:       page,
		PageSize:   limit,
		TotalPages: totalPages,
		HasMore:    hasMore,
	}

	c.JSON(http.StatusOK, response)
}

// GetExecutionDetailsGlobalHandler handles requests for a single execution (global view).
// GET /api/ui/v1/executions/:execution_id/details
func (h *ExecutionHandler) GetExecutionDetailsGlobalHandler(c *gin.Context) {
	ctx := c.Request.Context()
	executionID := strings.TrimSpace(c.Param("execution_id"))
	if executionID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "execution_id is required"})
		return
	}

	exec, err := h.store.GetExecutionRecord(ctx, executionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load execution: " + err.Error()})
		return
	}
	if exec == nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "execution not found"})
		return
	}

	c.JSON(http.StatusOK, h.toExecutionDetails(ctx, exec))
}

// RetryExecutionWebhookHandler re-enqueues webhook delivery attempts for an execution.
func (h *ExecutionHandler) RetryExecutionWebhookHandler(c *gin.Context) {
	if h.webhooks == nil {
		c.JSON(http.StatusServiceUnavailable, ErrorResponse{Error: "webhook dispatcher unavailable"})
		return
	}

	executionID := strings.TrimSpace(c.Param("execution_id"))
	if executionID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "execution_id is required"})
		return
	}

	ctx := c.Request.Context()
	exec, err := h.store.GetExecutionRecord(ctx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to load execution for webhook retry")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load execution: " + err.Error()})
		return
	}
	if exec == nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "execution not found"})
		return
	}

	hasWebhook, err := h.storage.HasExecutionWebhook(ctx, executionID)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", executionID).Msg("failed to check webhook registration")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to check webhook registration"})
		return
	}
	if !hasWebhook {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "no webhook registered for this execution"})
		return
	}

	if err := h.webhooks.Notify(ctx, executionID); err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to enqueue webhook retry")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to enqueue webhook retry"})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{"status": "queued"})
}

// StreamExecutionEventsHandler streams execution events for the UI dashboard.
// GET /api/ui/v1/executions/events
func (h *ExecutionHandler) StreamExecutionEventsHandler(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	subscriberID := fmt.Sprintf("ui_exec_events_%d", time.Now().UnixNano())
	eventBus := h.storage.GetExecutionEventBus()
	eventChan := eventBus.Subscribe(subscriberID)
	defer eventBus.Unsubscribe(subscriberID)

	ctx := c.Request.Context()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			heartbeat := map[string]interface{}{
				"type":      "heartbeat",
				"timestamp": time.Now().Format(time.RFC3339),
			}
			if payload, err := json.Marshal(heartbeat); err == nil {
				if !writeSSE(c, payload) {
					return
				}
			}
		case event, ok := <-eventChan:
			if !ok {
				return
			}
			if payload, err := json.Marshal(event); err == nil {
				if !writeSSE(c, payload) {
					return
				}
			}
		}
	}
}

// Helper utilities ---------------------------------------------------------

func (h *ExecutionHandler) toExecutionSummary(exec *types.Execution) ExecutionSummary {
	duration := 0
	if exec.DurationMS != nil {
		duration = int(*exec.DurationMS)
	}

	return ExecutionSummary{
		ID:           0,
		ExecutionID:  exec.ExecutionID,
		WorkflowID:   exec.RunID,
		SessionID:    exec.SessionID,
		AgentNodeID:  exec.AgentNodeID,
		ReasonerID:   exec.ReasonerID,
		Status:       types.NormalizeExecutionStatus(exec.Status),
		DurationMS:   duration,
		InputSize:    len(exec.InputPayload),
		OutputSize:   len(exec.ResultPayload),
		ErrorMessage: exec.ErrorMessage,
		CreatedAt:    exec.StartedAt,
		NotesCount:   0,
		LatestNote:   nil,
	}
}

func (h *ExecutionHandler) toExecutionDetails(ctx context.Context, exec *types.Execution) ExecutionDetailsResponse {
	inputData, inputSize := h.resolveExecutionData(ctx, exec.InputPayload, exec.InputURI)
	outputData, outputSize := h.resolveExecutionData(ctx, exec.ResultPayload, exec.ResultURI)

	var startedAt *string
	if !exec.StartedAt.IsZero() {
		started := exec.StartedAt.Format(time.RFC3339)
		startedAt = &started
	}

	var completedAt *string
	if exec.CompletedAt != nil {
		formatted := exec.CompletedAt.Format(time.RFC3339)
		completedAt = &formatted
	}

	var durationPtr *int
	if exec.DurationMS != nil {
		duration := int(*exec.DurationMS)
		durationPtr = &duration
	}

	updated := exec.UpdatedAt.Format(time.RFC3339)

	webhookRegistered := exec.WebhookRegistered
	webhookEvents := exec.WebhookEvents

	return ExecutionDetailsResponse{
		ID:                  0,
		ExecutionID:         exec.ExecutionID,
		WorkflowID:          exec.RunID,
		AgentFieldRequestID: nil,
		SessionID:           exec.SessionID,
		ActorID:             exec.ActorID,
		AgentNodeID:         exec.AgentNodeID,
		ParentWorkflowID:    exec.ParentExecutionID,
		RootWorkflowID:      nil,
		WorkflowDepth:       nil,
		ReasonerID:          exec.ReasonerID,
		InputData:           inputData,
		OutputData:          outputData,
		InputSize:           inputSize,
		OutputSize:          outputSize,
		WorkflowName:        nil,
		WorkflowTags:        nil,
		Status:              types.NormalizeExecutionStatus(exec.Status),
		StartedAt:           startedAt,
		CompletedAt:         completedAt,
		DurationMS:          durationPtr,
		ErrorMessage:        exec.ErrorMessage,
		RetryCount:          0,
		CreatedAt:           exec.StartedAt.Format(time.RFC3339),
		UpdatedAt:           &updated,
		Notes:               nil,
		NotesCount:          0,
		LatestNote:          nil,
		WebhookRegistered:   webhookRegistered,
		WebhookEvents:       webhookEvents,
	}
}

func (h *ExecutionHandler) resolveExecutionData(ctx context.Context, raw []byte, uri *string) (interface{}, int) {
	data := decodePayload(raw)
	size := len(raw)

	if hasMeaningfulData(data) {
		return data, size
	}

	if uri == nil || h.payloads == nil {
		return data, size
	}

	trimmed := strings.TrimSpace(*uri)
	if trimmed == "" {
		return data, size
	}

	payload, payloadSize, err := h.loadPayloadData(ctx, trimmed)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("uri", trimmed).Msg("failed to load payload for execution data")
		return data, size
	}
	return payload, payloadSize
}

func (h *ExecutionHandler) loadPayloadData(ctx context.Context, uri string) (interface{}, int, error) {
	if h.payloads == nil {
		return nil, 0, fmt.Errorf("payload store unavailable")
	}

	reader, err := h.payloads.Open(ctx, uri)
	if err != nil {
		return nil, 0, err
	}
	defer reader.Close()

	payloadBytes, err := io.ReadAll(reader)
	if err != nil {
		return nil, 0, err
	}

	if len(payloadBytes) > largePayloadWarningThreshold {
		logger.Logger.Warn().Str("uri", uri).Int("bytes", len(payloadBytes)).Msg("large payload loaded for execution IO display")
	}

	return decodePayload(payloadBytes), len(payloadBytes), nil
}

const (
	largePayloadWarningThreshold = 5 * 1024 * 1024 // 5 MiB
	corruptedJSONSentinel        = "corrupted_json_data"
)

func decodePayload(raw []byte) interface{} {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil
	}
	var data interface{}
	if err := json.Unmarshal(trimmed, &data); err == nil {
		return data
	}
	return string(trimmed)
}

func hasMeaningfulData(data interface{}) bool {
	switch v := data.(type) {
	case nil:
		return false
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return false
		}
		return !strings.EqualFold(trimmed, corruptedJSONSentinel)
	case []byte:
		return len(v) > 0
	case []interface{}:
		return len(v) > 0
	case map[string]interface{}:
		if len(v) == 0 {
			return false
		}
		if errValue, ok := v["error"]; ok {
			if errStr, ok := errValue.(string); ok && strings.EqualFold(errStr, corruptedJSONSentinel) {
				return false
			}
		}
		return true
	default:
		return true
	}
}

func parsePositiveIntOrDefault(value string, fallback int) int {
	if value == "" {
		return fallback
	}
	if v, err := strconv.Atoi(value); err == nil && v > 0 {
		return v
	}
	return fallback
}

func parseBoundedIntOrDefault(value string, fallback, min, max int) int {
	v := parsePositiveIntOrDefault(value, fallback)
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func parseTimePtrValue(value string) (*time.Time, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	parsed, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return nil, err
	}
	return &parsed, nil
}

func sanitizeExecutionSortField(field string) string {
	switch strings.ToLower(strings.TrimSpace(field)) {
	case "status":
		return "status"
	case "task_name", "reasoner", "reasoner_id":
		return "reasoner_id"
	case "duration_ms":
		return "duration_ms"
	case "duration":
		return "duration_ms"
	case "agent_node_id":
		return "agent_node_id"
	case "execution_id":
		return "execution_id"
	case "run_id", "workflow_id":
		return "run_id"
	case "when", "started", "started_at", "created_at":
		return "started_at"
	default:
		return "started_at"
	}
}

func computeTotalPages(total, pageSize int) int {
	if pageSize <= 0 {
		return 1
	}
	pages := total / pageSize
	if total%pageSize != 0 {
		pages++
	}
	if pages == 0 {
		pages = 1
	}
	return pages
}

func (h *ExecutionHandler) groupExecutionSummaries(summaries []ExecutionSummary, groupBy string) map[string][]ExecutionSummary {
	grouped := make(map[string][]ExecutionSummary)
	key := strings.ToLower(groupBy)
	for _, summary := range summaries {
		var bucket string
		switch key {
		case "status":
			bucket = summary.Status
		case "agent", "agent_node_id":
			bucket = summary.AgentNodeID
		case "reasoner", "reasoner_id":
			bucket = summary.ReasonerID
		default:
			bucket = "ungrouped"
		}
		grouped[bucket] = append(grouped[bucket], summary)
	}
	return grouped
}

func formatRelativeTimeString(now, started time.Time) string {
	if started.IsZero() {
		return ""
	}

	diff := now.Sub(started)
	if diff < time.Minute {
		return "just now"
	}
	if diff < time.Hour {
		return fmt.Sprintf("%dm ago", int(diff.Minutes()))
	}
	if diff < 24*time.Hour {
		return fmt.Sprintf("%dh ago", int(diff.Hours()))
	}
	days := int(diff.Hours()) / 24
	return fmt.Sprintf("%dd ago", days)
}

func formatDurationDisplay(durationMS *int64) string {
	if durationMS == nil || *durationMS <= 0 {
		return "—"
	}

	duration := time.Duration(*durationMS) * time.Millisecond
	if duration < time.Second {
		return fmt.Sprintf("%dms", duration.Milliseconds())
	}
	if duration < time.Minute {
		return fmt.Sprintf("%.1fs", duration.Seconds())
	}
	if duration < time.Hour {
		minutes := int(duration.Minutes())
		seconds := int(duration.Seconds()) % 60
		if seconds == 0 {
			return fmt.Sprintf("%dm", minutes)
		}
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	}

	hours := int(duration.Hours())
	minutes := int(duration.Minutes()) % 60
	if minutes == 0 {
		return fmt.Sprintf("%dh", hours)
	}
	return fmt.Sprintf("%dh %dm", hours, minutes)
}
