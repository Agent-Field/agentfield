package ui

import (
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
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
		RunID           string  `json:"run_id"`
		RootWorkflowID  string  `json:"root_workflow_id"`
		RootExecutionID string  `json:"root_execution_id,omitempty"`
		Status          string  `json:"status"`
		TotalSteps      int     `json:"total_steps"`
		CompletedSteps  int     `json:"completed_steps"`
		FailedSteps     int     `json:"failed_steps"`
		CreatedAt       string  `json:"created_at"`
		UpdatedAt       string  `json:"updated_at"`
		CompletedAt     *string `json:"completed_at,omitempty"`
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
	StartedAt         string  `json:"started_at"`
	CompletedAt       *string `json:"completed_at,omitempty"`
	WorkflowDepth     int     `json:"workflow_depth"`
	ActiveChildren    int     `json:"active_children"`
	PendingChildren   int     `json:"pending_children"`
	LastUpdatedAt     *string `json:"last_updated_at,omitempty"`
}

func (h *WorkflowRunHandler) ListWorkflowRunsHandler(c *gin.Context) {
	ctx := c.Request.Context()

	page := parsePositiveInt(c.DefaultQuery("page", "1"), 1)
	pageSize := parsePositiveIntWithin(c.DefaultQuery("page_size", "20"), 20, 1, 200)
	offset := (page - 1) * pageSize

	filter := types.ExecutionFilter{
		Limit:          pageSize,
		Offset:         offset,
		SortBy:         sanitizeExecutionSortField(c.DefaultQuery("sort_by", "started_at")),
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

	executions, err := h.storage.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to query executions"})
		return
	}

	grouped := types.GroupExecutionsByRun(executions)
	summaries := make([]WorkflowRunSummary, 0, len(grouped))
	for runID, execs := range grouped {
		summaries = append(summaries, summarizeRun(runID, execs))
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].StartedAt.After(summaries[j].StartedAt)
	})

	hasMore := len(executions) == pageSize

	response := WorkflowRunListResponse{
		Runs:       summaries,
		TotalCount: len(summaries),
		Page:       page,
		PageSize:   pageSize,
		HasMore:    hasMore,
	}

	c.JSON(http.StatusOK, response)
}

func (h *WorkflowRunHandler) GetWorkflowRunDetailHandler(c *gin.Context) {
	ctx := c.Request.Context()
	runID := strings.TrimSpace(c.Param("run_id"))
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "run_id is required"})
		return
	}

	filter := types.ExecutionFilter{
		RunID:          &runID,
		SortBy:         "started_at",
		SortDescending: false,
		Limit:          10000,
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
	detail.Run.CreatedAt = executions[0].StartedAt.Format(time.RFC3339)
	detail.Run.UpdatedAt = executions[len(executions)-1].StartedAt.Format(time.RFC3339)
	if dag.CompletedAt != nil && *dag.CompletedAt != "" {
		detail.Run.CompletedAt = dag.CompletedAt
	}
	_ = name

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

	sort.Slice(executions, func(i, j int) bool {
		return executions[i].StartedAt.Before(executions[j].StartedAt)
	})

	dag, _, status, name, sessionID, actorID, maxDepth := handlers.BuildWorkflowDAG(executions)

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
	summary.StartedAt = executions[0].StartedAt
	summary.UpdatedAt = executions[len(executions)-1].StartedAt
	summary.Status = status
	summary.MaxDepth = maxDepth
	if len(executions) > 0 {
		lastExec := executions[len(executions)-1]
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
	for _, exec := range executions {
		normalized := types.NormalizeExecutionStatus(exec.Status)
		summary.StatusCounts[normalized]++
		if normalized == string(types.ExecutionStatusRunning) ||
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
			case string(types.ExecutionStatusRunning):
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
