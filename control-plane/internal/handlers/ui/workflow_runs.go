package ui

import (
	"context"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type WorkflowRunHandler struct {
	storage storage.StorageProvider
}

func NewWorkflowRunHandler(storage storage.StorageProvider) *WorkflowRunHandler {
	return &WorkflowRunHandler{storage: storage}
}

type WorkflowRunSummary struct {
	WorkflowID       string         `json:"workflow_id"`
	RunID            string         `json:"run_id"`
	RootExecutionID  string         `json:"root_execution_id"`
	// RootExecutionStatus is the status of the root execution row, which is
	// the unit the user actually controls via Pause/Resume/Cancel. The
	// aggregate Status field above can drift from this when in-flight
	// children are still running after the user pauses or cancels the
	// root — see execute.go's dispatch-time guard for the full story.
	RootExecutionStatus string     `json:"root_execution_status,omitempty"`
	Status           string         `json:"status"`
	DisplayName      string         `json:"display_name"`
	CurrentTask      string         `json:"current_task"`
	RootReasoner     string         `json:"root_reasoner"`
	AgentID          *string        `json:"agent_id,omitempty"`
	SessionID        *string        `json:"session_id,omitempty"`
	ActorID          *string        `json:"actor_id,omitempty"`
	TotalExecutions  int            `json:"total_executions"`
	MaxDepth         int            `json:"max_depth"`
	ActiveExecutions int            `json:"active_executions"`
	StatusCounts     map[string]int `json:"status_counts"`
	StartedAt        time.Time      `json:"started_at"`
	UpdatedAt        time.Time      `json:"updated_at"`
	CompletedAt      *time.Time     `json:"completed_at,omitempty"`
	DurationMs       *int64         `json:"duration_ms,omitempty"`
	LatestActivity   time.Time      `json:"latest_activity"`
	Terminal         bool           `json:"terminal"`
}

type WorkflowRunListResponse struct {
	Runs       []WorkflowRunSummary `json:"runs"`
	TotalCount int                  `json:"total_count"`
	Page       int                  `json:"page"`
	PageSize   int                  `json:"page_size"`
	HasMore    bool                 `json:"has_more"`
}

type WorkflowRunDetailResponse struct {
	Run struct {
		RunID           string         `json:"run_id"`
		RootWorkflowID  string         `json:"root_workflow_id"`
		RootExecutionID string         `json:"root_execution_id,omitempty"`
		Status          string         `json:"status"`
		TotalSteps      int            `json:"total_steps"`
		CompletedSteps  int            `json:"completed_steps"`
		FailedSteps     int            `json:"failed_steps"`
		ReturnedSteps   int            `json:"returned_steps"`
		StatusCounts    map[string]int `json:"status_counts,omitempty"`
		CreatedAt       string         `json:"created_at"`
		UpdatedAt       string         `json:"updated_at"`
		CompletedAt     *string        `json:"completed_at,omitempty"`
	} `json:"run"`
	Executions []apiWorkflowExecution `json:"executions"`
}

type apiWorkflowExecution struct {
	WorkflowID        string  `json:"workflow_id"`
	ExecutionID       string  `json:"execution_id"`
	ParentExecutionID *string `json:"parent_execution_id,omitempty"`
	ParentWorkflowID  *string `json:"parent_workflow_id,omitempty"`
	AgentNodeID       string  `json:"agent_node_id"`
	ReasonerID        string  `json:"reasoner_id"`
	Status            string  `json:"status"`
	StatusReason      *string `json:"status_reason,omitempty"`
	StartedAt         string  `json:"started_at"`
	CompletedAt       *string `json:"completed_at,omitempty"`
	WorkflowDepth     int     `json:"workflow_depth"`
	ActiveChildren    int     `json:"active_children"`
	PendingChildren   int     `json:"pending_children"`
	LastUpdatedAt     *string `json:"last_updated_at,omitempty"`
	// Approval fields (populated when execution has an approval request)
	ApprovalRequestID  *string `json:"approval_request_id,omitempty"`
	ApprovalRequestURL *string `json:"approval_request_url,omitempty"`
	ApprovalStatus     *string `json:"approval_status,omitempty"`
}

func (h *WorkflowRunHandler) ListWorkflowRunsHandler(c *gin.Context) {
	ctx := c.Request.Context()

	page := parsePositiveInt(c.DefaultQuery("page", "1"), 1)
	pageSize := parsePositiveIntWithin(c.DefaultQuery("page_size", "20"), 20, 1, 200)
	offset := (page - 1) * pageSize

	// Build filter for run aggregation query
	filter := types.ExecutionFilter{
		Limit:          pageSize,
		Offset:         offset,
		SortBy:         sanitizeRunSortField(c.DefaultQuery("sort_by", "updated_at")),
		SortDescending: strings.ToLower(c.DefaultQuery("sort_order", "desc")) != "asc",
	}

	if runID := strings.TrimSpace(c.Query("run_id")); runID != "" {
		filter.RunID = &runID
	}
	if status := strings.TrimSpace(c.Query("status")); status != "" {
		filter.Status = &status
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
	if search := strings.TrimSpace(c.Query("search")); search != "" {
		filter.Search = &search
	}

	// Use the efficient aggregation method that scales to millions of nodes
	runAggregations, totalRuns, err := h.storage.QueryRunSummaries(ctx, filter)
	if err != nil {
		// Log the actual error for debugging
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to query run summaries",
			"details": err.Error(),
		})
		return
	}

	// Convert aggregations to API response format
	summaries := make([]WorkflowRunSummary, 0, len(runAggregations))
	for _, agg := range runAggregations {
		summaries = append(summaries, convertAggregationToSummary(agg))
	}

	totalCount := totalRuns
	if totalCount == 0 {
		totalCount = len(runAggregations)
	}
	hasMore := page*pageSize < totalCount

	response := WorkflowRunListResponse{
		Runs:       summaries,
		TotalCount: totalCount,
		Page:       page,
		PageSize:   pageSize,
		HasMore:    hasMore,
	}

	c.JSON(http.StatusOK, response)
}

// convertAggregationToSummary converts a storage.RunSummaryAggregation to WorkflowRunSummary
func convertAggregationToSummary(agg *storage.RunSummaryAggregation) WorkflowRunSummary {
	summary := WorkflowRunSummary{
		WorkflowID:       agg.RunID,
		RunID:            agg.RunID,
		StatusCounts:     agg.StatusCounts,
		TotalExecutions:  agg.TotalExecutions,
		MaxDepth:         agg.MaxDepth,
		ActiveExecutions: agg.ActiveExecutions,
		StartedAt:        agg.EarliestStarted,
		UpdatedAt:        agg.LatestStarted,
		LatestActivity:   agg.LatestStarted,
	}

	// Set root execution ID + status. The root status is what the lifecycle
	// controls in the UI key off so they can reflect what the user actually
	// controls, not the children-aggregated value.
	if agg.RootExecutionID != nil {
		summary.RootExecutionID = *agg.RootExecutionID
	}
	if agg.RootStatus != nil {
		summary.RootExecutionStatus = *agg.RootStatus
	}

	// Set display name from root reasoner or run ID
	if agg.RootReasonerID != nil && *agg.RootReasonerID != "" {
		summary.DisplayName = *agg.RootReasonerID
		summary.RootReasoner = *agg.RootReasonerID
		summary.CurrentTask = *agg.RootReasonerID
	} else {
		summary.DisplayName = agg.RunID
		summary.RootReasoner = agg.RunID
		summary.CurrentTask = agg.RunID
	}

	// Set agent ID
	if agg.RootAgentNodeID != nil && *agg.RootAgentNodeID != "" {
		summary.AgentID = agg.RootAgentNodeID
	}

	// Set session and actor IDs
	summary.SessionID = agg.SessionID
	summary.ActorID = agg.ActorID

	// Determine overall status
	summary.Status = deriveStatusFromCounts(agg.StatusCounts, agg.ActiveExecutions)

	// Check if terminal
	summary.Terminal = summary.Status == string(types.ExecutionStatusSucceeded) ||
		summary.Status == string(types.ExecutionStatusFailed) ||
		summary.Status == string(types.ExecutionStatusTimeout) ||
		summary.Status == string(types.ExecutionStatusCancelled)

	// Calculate duration if completed
	if summary.Terminal {
		completedAt := agg.LatestStarted
		summary.CompletedAt = &completedAt
		duration := completedAt.Sub(agg.EarliestStarted).Milliseconds()
		summary.DurationMs = &duration
	}

	return summary
}

// deriveStatusFromCounts determines overall workflow status from status counts.
// Priority: active (running/waiting/pending/queued) > failed > timeout > cancelled > succeeded.
func deriveStatusFromCounts(statusCounts map[string]int, activeExecutions int) string {
	// If there are active executions (running, waiting, pending, queued), the workflow is running
	if activeExecutions > 0 {
		return string(types.ExecutionStatusRunning)
	}

	// If there are any failed executions, the workflow is failed
	if statusCounts[string(types.ExecutionStatusFailed)] > 0 {
		return string(types.ExecutionStatusFailed)
	}

	// If there are any timed-out executions, the workflow timed out
	if statusCounts[string(types.ExecutionStatusTimeout)] > 0 {
		return string(types.ExecutionStatusTimeout)
	}

	// If there are any cancelled executions, the workflow is cancelled
	if statusCounts[string(types.ExecutionStatusCancelled)] > 0 {
		return string(types.ExecutionStatusCancelled)
	}

	// If there are any paused executions (and no active ones), the workflow is paused
	if statusCounts[string(types.ExecutionStatusPaused)] > 0 {
		return string(types.ExecutionStatusPaused)
	}

	// All executions are in terminal non-error states (succeeded) or no executions exist
	return string(types.ExecutionStatusSucceeded)
}

func (h *WorkflowRunHandler) GetWorkflowRunDetailHandler(c *gin.Context) {
	ctx := c.Request.Context()
	runID := strings.TrimSpace(c.Param("run_id"))
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "run_id is required"})
		return
	}

	filter := types.ExecutionFilter{
		RunID:           &runID,
		SortBy:          "started_at",
		SortDescending:  false,
		Limit:           10000,
		ExcludePayloads: true,
	}

	executions, err := h.storage.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to query executions"})
		return
	}
	if len(executions) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "workflow run not found"})
		return
	}

	dag, timeline, status, name, _, _, _ := handlers.BuildWorkflowDAG(executions)
	apiExecutions := buildAPIExecutions(timeline)

	completed, failed := countOutcomeSteps(executions)

	detail := WorkflowRunDetailResponse{}
	detail.Run.RunID = runID
	detail.Run.RootWorkflowID = runID
	detail.Run.RootExecutionID = dag.ExecutionID
	detail.Run.Status = status
	detail.Run.TotalSteps = len(executions)
	detail.Run.CompletedSteps = completed
	detail.Run.FailedSteps = failed
	detail.Run.ReturnedSteps = len(executions)
	detail.Run.CreatedAt = executions[0].StartedAt.Format(time.RFC3339)
	detail.Run.UpdatedAt = executions[len(executions)-1].StartedAt.Format(time.RFC3339)
	if dag.CompletedAt != nil && *dag.CompletedAt != "" {
		detail.Run.CompletedAt = dag.CompletedAt
	}
	_ = name

	if agg := h.loadRunSummary(ctx, runID); agg != nil {
		detail.Run.TotalSteps = agg.TotalExecutions
		detail.Run.CompletedSteps = agg.StatusCounts[string(types.ExecutionStatusSucceeded)]
		detail.Run.FailedSteps =
			agg.StatusCounts[string(types.ExecutionStatusFailed)] +
				agg.StatusCounts[string(types.ExecutionStatusCancelled)] +
				agg.StatusCounts[string(types.ExecutionStatusTimeout)]
		detail.Run.Status = deriveStatusFromCounts(agg.StatusCounts, agg.ActiveExecutions)
		detail.Run.StatusCounts = cloneStatusCounts(agg.StatusCounts)

		if agg.RootExecutionID != nil && detail.Run.RootExecutionID == "" {
			detail.Run.RootExecutionID = *agg.RootExecutionID
		}
	} else {
		fallbackSummary := summarizeRun(runID, executions)
		detail.Run.TotalSteps = fallbackSummary.TotalExecutions
		detail.Run.CompletedSteps = fallbackSummary.StatusCounts[string(types.ExecutionStatusSucceeded)]
		detail.Run.FailedSteps =
			fallbackSummary.StatusCounts[string(types.ExecutionStatusFailed)] +
				fallbackSummary.StatusCounts[string(types.ExecutionStatusCancelled)] +
				fallbackSummary.StatusCounts[string(types.ExecutionStatusTimeout)]
		detail.Run.Status = fallbackSummary.Status
		detail.Run.StatusCounts = cloneStatusCounts(fallbackSummary.StatusCounts)

		if detail.Run.RootExecutionID == "" {
			detail.Run.RootExecutionID = fallbackSummary.RootExecutionID
		}
	}

	// Enrich executions in waiting status with approval data from workflow executions
	h.enrichApprovalData(ctx, apiExecutions)

	detail.Executions = apiExecutions

	c.JSON(http.StatusOK, detail)
}

func summarizeRun(runID string, executions []*types.Execution) WorkflowRunSummary {
	summary := WorkflowRunSummary{
		WorkflowID:      runID,
		RunID:           runID,
		StatusCounts:    make(map[string]int),
		TotalExecutions: len(executions),
	}
	if len(executions) == 0 {
		return summary
	}

	sortedExecutions := make([]*types.Execution, len(executions))
	copy(sortedExecutions, executions)
	sort.Slice(sortedExecutions, func(i, j int) bool {
		return sortedExecutions[i].StartedAt.Before(sortedExecutions[j].StartedAt)
	})

	dag, _, status, name, sessionID, actorID, maxDepth := handlers.BuildWorkflowDAG(sortedExecutions)

	summary.RootExecutionID = dag.ExecutionID
	if name != "" {
		summary.DisplayName = name
	} else if dag.ReasonerID != "" {
		summary.DisplayName = dag.ReasonerID
	} else {
		summary.DisplayName = runID
	}
	summary.RootReasoner = dag.ReasonerID
	if dag.AgentNodeID != "" {
		summary.AgentID = &dag.AgentNodeID
	}
	summary.SessionID = sessionID
	summary.ActorID = actorID
	summary.StartedAt = sortedExecutions[0].StartedAt
	summary.UpdatedAt = sortedExecutions[len(sortedExecutions)-1].StartedAt
	summary.Status = status
	summary.MaxDepth = maxDepth
	if len(sortedExecutions) > 0 {
		lastExec := sortedExecutions[len(sortedExecutions)-1]
		if lastExec != nil && lastExec.ReasonerID != "" {
			summary.CurrentTask = lastExec.ReasonerID
		}
	}
	if summary.CurrentTask == "" {
		summary.CurrentTask = dag.ReasonerID
	}
	if summary.CurrentTask == "" {
		summary.CurrentTask = summary.DisplayName
	}

	active := 0
	for _, exec := range sortedExecutions {
		normalized := types.NormalizeExecutionStatus(exec.Status)
		summary.StatusCounts[normalized]++
		if normalized == string(types.ExecutionStatusRunning) ||
			normalized == string(types.ExecutionStatusWaiting) ||
			normalized == string(types.ExecutionStatusPending) ||
			normalized == string(types.ExecutionStatusQueued) {
			active++
		}
		if exec.CompletedAt != nil {
			if summary.CompletedAt == nil || exec.CompletedAt.After(*summary.CompletedAt) {
				summary.CompletedAt = exec.CompletedAt
			}
		}
		if exec.StartedAt.After(summary.UpdatedAt) {
			summary.UpdatedAt = exec.StartedAt
		}
	}
	summary.ActiveExecutions = active
	summary.LatestActivity = summary.UpdatedAt
	summary.Terminal = status == string(types.ExecutionStatusSucceeded) || status == string(types.ExecutionStatusFailed)

	if summary.CompletedAt != nil {
		duration := summary.CompletedAt.Sub(summary.StartedAt).Milliseconds()
		summary.DurationMs = &duration
	}

	return summary
}

func (h *WorkflowRunHandler) loadRunSummary(ctx context.Context, runID string) *storage.RunSummaryAggregation {
	filter := types.ExecutionFilter{
		RunID:  &runID,
		Limit:  1,
		Offset: 0,
	}

	summaries, _, err := h.storage.QueryRunSummaries(ctx, filter)
	if err != nil {
		logger.Logger.Warn().
			Str("run_id", runID).
			Err(err).
			Msg("failed to load run summary aggregation")
		return nil
	}
	if len(summaries) == 0 {
		return nil
	}
	return summaries[0]
}

func cloneStatusCounts(input map[string]int) map[string]int {
	if input == nil {
		return nil
	}

	result := make(map[string]int, len(input))
	for key, value := range input {
		result[key] = value
	}
	return result
}

func countOutcomeSteps(executions []*types.Execution) (int, int) {
	completed := 0
	failed := 0
	for _, exec := range executions {
		switch types.NormalizeExecutionStatus(exec.Status) {
		case string(types.ExecutionStatusSucceeded):
			completed++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			failed++
		}
	}
	return completed, failed
}

func buildAPIExecutions(nodes []handlers.WorkflowDAGNode) []apiWorkflowExecution {
	childMap := make(map[string][]handlers.WorkflowDAGNode, len(nodes))
	for _, node := range nodes {
		if node.ParentExecutionID != nil && *node.ParentExecutionID != "" {
			childMap[*node.ParentExecutionID] = append(childMap[*node.ParentExecutionID], node)
		}
	}

	apiNodes := make([]apiWorkflowExecution, 0, len(nodes))
	for _, node := range nodes {
		children := childMap[node.ExecutionID]
		activeChildren := 0
		pendingChildren := 0
		for _, child := range children {
			switch types.NormalizeExecutionStatus(child.Status) {
			case string(types.ExecutionStatusRunning), string(types.ExecutionStatusWaiting):
				activeChildren++
			case string(types.ExecutionStatusPending), string(types.ExecutionStatusQueued):
				pendingChildren++
			}
		}

		apiNode := apiWorkflowExecution{
			WorkflowID:        node.WorkflowID,
			ExecutionID:       node.ExecutionID,
			ParentExecutionID: node.ParentExecutionID,
			ParentWorkflowID: func() *string {
				if node.ParentExecutionID != nil && *node.ParentExecutionID != "" {
					workflowID := node.WorkflowID
					return &workflowID
				}
				return nil
			}(),
			AgentNodeID:     node.AgentNodeID,
			ReasonerID:      node.ReasonerID,
			Status:          node.Status,
			StatusReason:    node.StatusReason,
			StartedAt:       node.StartedAt,
			CompletedAt:     node.CompletedAt,
			WorkflowDepth:   node.WorkflowDepth,
			ActiveChildren:  activeChildren,
			PendingChildren: pendingChildren,
		}
		apiNodes = append(apiNodes, apiNode)
	}
	return apiNodes
}

// enrichApprovalData looks up workflow executions for any api nodes in waiting status
// and populates their approval fields.
func (h *WorkflowRunHandler) enrichApprovalData(ctx context.Context, executions []apiWorkflowExecution) {
	for i := range executions {
		// Only look up approval data for executions that have a waiting-related status
		normalized := types.NormalizeExecutionStatus(executions[i].Status)
		if normalized != types.ExecutionStatusWaiting {
			continue
		}
		wfExec, err := h.storage.GetWorkflowExecution(ctx, executions[i].ExecutionID)
		if err != nil || wfExec == nil {
			continue
		}
		executions[i].ApprovalRequestID = wfExec.ApprovalRequestID
		executions[i].ApprovalRequestURL = wfExec.ApprovalRequestURL
		executions[i].ApprovalStatus = wfExec.ApprovalStatus
	}
}

func parsePositiveInt(value string, fallback int) int {
	v, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func parsePositiveIntWithin(value string, fallback, min, max int) int {
	v := parsePositiveInt(value, fallback)
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// sanitizeRunSortField maps friendly sort keys from the UI to backend field names used for ordering.
func sanitizeRunSortField(field string) string {
	switch strings.ToLower(strings.TrimSpace(field)) {
	case "started_at", "started", "created_at":
		return "started_at"
	case "status":
		return "status"
	case "total_steps", "total_executions", "nodes":
		return "total_steps"
	case "failed_steps", "failed":
		return "failed_steps"
	case "active_executions", "active":
		return "active_executions"
	case "updated_at", "latest_activity", "latest":
		return "updated_at"
	default:
		return "updated_at"
	}
}
