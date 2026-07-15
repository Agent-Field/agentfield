package handlers

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// RunSummaryStore is the narrow storage surface the active-executions
// endpoint needs.
type RunSummaryStore interface {
	QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error)
}

// ActiveRun is one in-flight workflow run as seen by external callers.
type ActiveRun struct {
	RunID           string `json:"run_id"`
	RootExecutionID string `json:"root_execution_id,omitempty"`
	// Target is "<agent>.<reasoner>" of the root execution — the value the
	// caller originally POSTed to /api/v1/execute/async/<target>.
	Target     string `json:"target,omitempty"`
	AgentID    string `json:"agent_id,omitempty"`
	ReasonerID string `json:"reasoner_id,omitempty"`
	// RootStatus is the status of the root execution — the unit
	// cancel/pause/resume act on.
	RootStatus       string         `json:"root_status,omitempty"`
	ActiveExecutions int            `json:"active_executions"`
	TotalExecutions  int            `json:"total_executions"`
	StatusCounts     map[string]int `json:"status_counts"`
	SessionID        string         `json:"session_id,omitempty"`
	StartedAt        time.Time      `json:"started_at"`
	// LatestActivity is the newest updated_at across the run's executions —
	// a run whose latest_activity is many minutes old while active_executions
	// is still > 0 is likely wedged, not working.
	LatestActivity time.Time `json:"latest_activity"`
}

// ActiveExecutionsResponse answers "what is in flight right now?".
type ActiveExecutionsResponse struct {
	// Count is the total number of active runs (not just the returned page).
	Count int         `json:"count"`
	Runs  []ActiveRun `json:"runs"`
}

// activeExecutionCount sums every non-terminal status in a run's counts —
// the same definition ActiveOnly's HAVING clause uses (running, pending,
// queued, waiting, paused, unknown), and deliberately NOT the aggregation's
// narrower pre-existing ActiveExecutions column, which excludes paused and
// which the UI's status derivation still depends on.
func activeExecutionCount(statusCounts map[string]int) int {
	active := 0
	for status, count := range statusCounts {
		if !types.IsTerminalExecutionStatus(status) {
			active += count
		}
	}
	return active
}

// ActiveExecutionsHandler serves GET /api/v1/executions/active: every run
// with at least one non-terminal execution (running/pending/queued/waiting/
// paused/unknown), with complete per-run status counts. This is the
// documented caller surface for in-flight visibility — callers no longer
// need to track execution IDs themselves (batch-status) or scrape
// UI-internal endpoints.
func ActiveExecutionsHandler(store RunSummaryStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		limit := 100
		if raw := strings.TrimSpace(c.Query("limit")); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 {
				c.JSON(http.StatusBadRequest, gin.H{"error": "limit must be a positive integer"})
				return
			}
			if parsed > 200 {
				parsed = 200
			}
			limit = parsed
		}

		filter := types.ExecutionFilter{
			ActiveOnly:     true,
			Limit:          limit,
			SortBy:         "updated_at",
			SortDescending: true,
		}
		if sessionID := strings.TrimSpace(c.Query("session_id")); sessionID != "" {
			filter.SessionID = &sessionID
		}
		if agentID := strings.TrimSpace(c.Query("agent_id")); agentID != "" {
			filter.AgentNodeID = &agentID
		}

		summaries, total, err := store.QueryRunSummaries(c.Request.Context(), filter)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to query active runs", "details": err.Error()})
			return
		}

		runs := make([]ActiveRun, 0, len(summaries))
		for _, agg := range summaries {
			run := ActiveRun{
				RunID:            agg.RunID,
				RootExecutionID:  derefOrEmpty(agg.RootExecutionID),
				AgentID:          derefOrEmpty(agg.RootAgentNodeID),
				ReasonerID:       derefOrEmpty(agg.RootReasonerID),
				RootStatus:       derefOrEmpty(agg.RootStatus),
				ActiveExecutions: activeExecutionCount(agg.StatusCounts),
				TotalExecutions:  agg.TotalExecutions,
				StatusCounts:     agg.StatusCounts,
				SessionID:        derefOrEmpty(agg.SessionID),
				StartedAt:        agg.EarliestStarted,
				LatestActivity:   agg.LatestStarted,
			}
			if run.AgentID != "" && run.ReasonerID != "" {
				run.Target = run.AgentID + "." + run.ReasonerID
			}
			runs = append(runs, run)
		}

		c.JSON(http.StatusOK, ActiveExecutionsResponse{Count: total, Runs: runs})
	}
}
