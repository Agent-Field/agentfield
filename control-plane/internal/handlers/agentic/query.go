package agentic

import (
	"context"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// QueryRequest is the body for POST /agentic/query.
type QueryRequest struct {
	Resource string       `json:"resource" binding:"required"` // "runs", "executions", "agents", "workflows", "sessions"
	Filters  QueryFilters `json:"filters"`
	Include  []string     `json:"include,omitempty"` // related resources to include
	Limit    int          `json:"limit"`
	Offset   int          `json:"offset"`
}

// QueryFilters contains optional filter criteria.
type QueryFilters struct {
	Status    string `json:"status,omitempty"`
	AgentID   string `json:"agent_id,omitempty"`
	RunID     string `json:"run_id,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	ActorID   string `json:"actor_id,omitempty"`
	Since     string `json:"since,omitempty"` // RFC3339
	Until     string `json:"until,omitempty"` // RFC3339
}

// QueryHandler handles unified resource queries.
func QueryHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req QueryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		if req.Limit <= 0 || req.Limit > 100 {
			req.Limit = 20
		}

		ctx := c.Request.Context()

		switch req.Resource {
		case "runs":
			queryRuns(c, ctx, store, req)
		case "executions":
			queryExecutions(c, ctx, store, req)
		case "agents":
			queryAgents(c, ctx, store, req)
		case "workflows":
			queryWorkflows(c, ctx, store, req)
		case "sessions":
			querySessions(c, ctx, store, req)
		default:
			respondError(c, http.StatusBadRequest, "invalid_resource",
				"resource must be one of: runs, executions, agents, workflows, sessions")
		}
	}
}

func queryRuns(c *gin.Context, ctx context.Context, store storage.StorageProvider, req QueryRequest) {
	filter := types.ExecutionFilter{}
	if req.Filters.Status != "" {
		filter.Status = &req.Filters.Status
	}
	if req.Filters.AgentID != "" {
		filter.AgentNodeID = &req.Filters.AgentID
	}
	if req.Filters.Since != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Since); err == nil {
			filter.StartTime = &t
		}
	}

	runs, total, err := store.QueryRunSummaries(ctx, filter)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
		return
	}

	respondOK(c, gin.H{
		"resource": "runs",
		"results":  runs,
		"total":    total,
		"limit":    req.Limit,
		"offset":   req.Offset,
	})
}

func queryExecutions(c *gin.Context, ctx context.Context, store storage.StorageProvider, req QueryRequest) {
	filter := types.ExecutionFilter{}
	if req.Filters.Status != "" {
		filter.Status = &req.Filters.Status
	}
	if req.Filters.AgentID != "" {
		filter.AgentNodeID = &req.Filters.AgentID
	}
	if req.Filters.RunID != "" {
		filter.RunID = &req.Filters.RunID
	}
	if req.Filters.SessionID != "" {
		filter.SessionID = &req.Filters.SessionID
	}
	if req.Filters.ActorID != "" {
		filter.ActorID = &req.Filters.ActorID
	}
	if req.Filters.Since != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Since); err == nil {
			filter.StartTime = &t
		}
	}
	if req.Filters.Until != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Until); err == nil {
			filter.EndTime = &t
		}
	}

	execs, err := store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
		return
	}

	total := len(execs)

	// Apply offset and limit
	if req.Offset > 0 && req.Offset < len(execs) {
		execs = execs[req.Offset:]
	}
	if len(execs) > req.Limit {
		execs = execs[:req.Limit]
	}

	respondOK(c, gin.H{
		"resource": "executions",
		"results":  execs,
		"total":    total,
		"limit":    req.Limit,
		"offset":   req.Offset,
	})
}

func queryAgents(c *gin.Context, ctx context.Context, store storage.StorageProvider, req QueryRequest) {
	filters := types.AgentFilters{}

	agents, err := store.ListAgents(ctx, filters)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
		return
	}

	total := len(agents)
	if req.Offset > 0 && req.Offset < len(agents) {
		agents = agents[req.Offset:]
	}
	if len(agents) > req.Limit {
		agents = agents[:req.Limit]
	}

	respondOK(c, gin.H{
		"resource": "agents",
		"results":  agents,
		"total":    total,
		"limit":    req.Limit,
		"offset":   req.Offset,
	})
}

func queryWorkflows(c *gin.Context, ctx context.Context, store storage.StorageProvider, req QueryRequest) {
	filters := types.WorkflowFilters{}
	if req.Filters.Status != "" {
		filters.Status = &req.Filters.Status
	}
	if req.Filters.SessionID != "" {
		filters.SessionID = &req.Filters.SessionID
	}
	if req.Filters.ActorID != "" {
		filters.ActorID = &req.Filters.ActorID
	}
	if req.Filters.Since != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Since); err == nil {
			filters.StartTime = &t
		}
	}
	if req.Filters.Until != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Until); err == nil {
			filters.EndTime = &t
		}
	}

	workflows, err := store.QueryWorkflows(ctx, filters)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
		return
	}

	total := len(workflows)
	if req.Offset > 0 && req.Offset < len(workflows) {
		workflows = workflows[req.Offset:]
	}
	if len(workflows) > req.Limit {
		workflows = workflows[:req.Limit]
	}

	respondOK(c, gin.H{
		"resource": "workflows",
		"results":  workflows,
		"total":    total,
		"limit":    req.Limit,
		"offset":   req.Offset,
	})
}

func querySessions(c *gin.Context, ctx context.Context, store storage.StorageProvider, req QueryRequest) {
	filters := types.SessionFilters{}
	if req.Filters.ActorID != "" {
		filters.ActorID = &req.Filters.ActorID
	}
	if req.Filters.Since != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Since); err == nil {
			filters.StartTime = &t
		}
	}
	if req.Filters.Until != "" {
		if t, err := time.Parse(time.RFC3339, req.Filters.Until); err == nil {
			filters.EndTime = &t
		}
	}

	sessions, err := store.QuerySessions(ctx, filters)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "query_failed", err.Error())
		return
	}

	total := len(sessions)
	if req.Offset > 0 && req.Offset < len(sessions) {
		sessions = sessions[req.Offset:]
	}
	if len(sessions) > req.Limit {
		sessions = sessions[:req.Limit]
	}

	respondOK(c, gin.H{
		"resource": "sessions",
		"results":  sessions,
		"total":    total,
		"limit":    req.Limit,
		"offset":   req.Offset,
	})
}
