package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// ReasonerCatalogStore captures storage calls needed by the human CLI trigger surface.
type ReasonerCatalogStore interface {
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
}

type ReasonerCatalogRow struct {
	Node        string   `json:"node"`
	Reasoner    string   `json:"reasoner"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	LastRunAt   *string  `json:"last_run_at,omitempty"`
	Status      string   `json:"status"`
}

type ReasonerCatalogResponse struct {
	Reasoners []ReasonerCatalogRow `json:"reasoners"`
	Shown     int                  `json:"shown"`
	Total     int                  `json:"total"`
}

// ListReasonersHandler returns a recency-first reasoner catalog for `af ls`.
func ListReasonersHandler(store ReasonerCatalogStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		agents, err := store.ListAgents(ctx, types.AgentFilters{})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("list agents: %v", err)})
			return
		}

		recentExecutions, err := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
			Limit:           1000,
			SortBy:          "started_at",
			SortDescending:  true,
			ExcludePayloads: true,
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("query executions: %v", err)})
			return
		}

		lastRunByReasoner := make(map[string]time.Time)
		for _, exec := range recentExecutions {
			if exec == nil || exec.AgentNodeID == "" || exec.ReasonerID == "" || exec.StartedAt.IsZero() {
				continue
			}
			key := exec.AgentNodeID + "." + exec.ReasonerID
			if _, exists := lastRunByReasoner[key]; !exists {
				lastRunByReasoner[key] = exec.StartedAt.UTC()
			}
		}

		query := strings.TrimSpace(c.Query("query"))
		nodeFilter := strings.TrimSpace(c.Query("node"))
		liveOnly := parseQueryBool(c.Query("live"))
		showAll := parseQueryBool(c.Query("all"))
		entrypointsOnly := parseQueryBool(c.Query("entrypoints"))

		rows := make([]ReasonerCatalogRow, 0)
		for _, agent := range agents {
			if agent == nil {
				continue
			}
			if nodeFilter != "" && agent.ID != nodeFilter {
				continue
			}
			status := reasonerNodeStatus(agent)
			if liveOnly && status != "live" {
				continue
			}
			for _, reasoner := range agent.Reasoners {
				if entrypointsOnly && !hasTag(reasoner.Tags, types.TagEntrypoint) {
					continue
				}
				fullName := agent.ID + "." + reasoner.ID
				if query != "" && !fuzzyReasonerMatch(fullName, query) {
					continue
				}
				var lastRun *string
				if ts, ok := lastRunByReasoner[fullName]; ok {
					formatted := ts.Format(time.RFC3339)
					lastRun = &formatted
				}
				description := strings.TrimSpace(reasoner.Description)
				if description == "" {
					if fromMeta := extractDescription(agent.Metadata, reasoner.ID); fromMeta != nil {
						description = strings.TrimSpace(*fromMeta)
					}
				}
				rows = append(rows, ReasonerCatalogRow{
					Node:        agent.ID,
					Reasoner:    reasoner.ID,
					Description: description,
					Tags:        reasoner.Tags,
					LastRunAt:   lastRun,
					Status:      status,
				})
			}
		}

		sort.SliceStable(rows, func(i, j int) bool {
			left, leftOK := parseCatalogTime(rows[i].LastRunAt)
			right, rightOK := parseCatalogTime(rows[j].LastRunAt)
			if leftOK != rightOK {
				return leftOK
			}
			if leftOK && !left.Equal(right) {
				return left.After(right)
			}
			// Among never-run (or same-timestamp) rows, surface declared entry
			// points first so a fresh registry lists callable surfaces on top.
			leftEntry := hasTag(rows[i].Tags, types.TagEntrypoint)
			rightEntry := hasTag(rows[j].Tags, types.TagEntrypoint)
			if leftEntry != rightEntry {
				return leftEntry
			}
			return rows[i].Node+"."+rows[i].Reasoner < rows[j].Node+"."+rows[j].Reasoner
		})

		total := len(rows)
		if !showAll && len(rows) > 20 {
			rows = rows[:20]
		}
		c.JSON(http.StatusOK, ReasonerCatalogResponse{
			Reasoners: rows,
			Shown:     len(rows),
			Total:     total,
		})
	}
}

// StreamExecutionEventsHandler streams live execution events for `af tail`.
func StreamExecutionEventsHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := strings.TrimSpace(c.Param("execution_id"))
		if id == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
			return
		}
		bus := store.GetExecutionEventBus()
		if bus == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "execution event stream unavailable"})
			return
		}

		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")

		from := parseFromStep(c.Query("from"))
		sequence := 0
		writeEvent := func(event events.ExecutionEvent) bool {
			if event.ExecutionID != id && event.WorkflowID != id {
				return true
			}
			sequence++
			if sequence < from {
				return true
			}
			payload, err := json.Marshal(event)
			if err != nil {
				return true
			}
			fmt.Fprintf(c.Writer, "id: %d\nevent: execution\ndata: %s\n\n", sequence, payload)
			c.Writer.Flush()
			return !types.IsTerminalExecutionStatus(event.Status)
		}

		if snapshot, ok := loadExecutionSnapshot(c.Request.Context(), store, id); ok {
			if !writeEvent(snapshot) {
				return
			}
		}

		subscriberID := fmt.Sprintf("cli_tail_%d", time.Now().UnixNano())
		ch := bus.Subscribe(subscriberID)
		defer bus.Unsubscribe(subscriberID)
		heartbeat := time.NewTicker(15 * time.Second)
		defer heartbeat.Stop()

		for {
			select {
			case <-c.Request.Context().Done():
				return
			case <-heartbeat.C:
				fmt.Fprint(c.Writer, ": heartbeat\n\n")
				c.Writer.Flush()
			case event, ok := <-ch:
				if !ok {
					return
				}
				if !writeEvent(event) {
					return
				}
			}
		}
	}
}

func loadExecutionSnapshot(ctx context.Context, store ExecutionStore, id string) (events.ExecutionEvent, bool) {
	exec, err := store.GetExecutionRecord(ctx, id)
	if err != nil || exec == nil {
		runID := id
		execs, queryErr := store.QueryExecutionRecords(ctx, types.ExecutionFilter{
			RunID:           &runID,
			Limit:           1,
			SortBy:          "updated_at",
			SortDescending:  true,
			ExcludePayloads: false,
		})
		if queryErr != nil || len(execs) == 0 || execs[0] == nil {
			return events.ExecutionEvent{}, false
		}
		exec = execs[0]
	}
	data := map[string]interface{}{}
	if exec.ResultPayload != nil {
		var result interface{}
		if err := json.Unmarshal(exec.ResultPayload, &result); err == nil {
			data["result"] = result
		}
	}
	if exec.ErrorMessage != nil {
		data["error"] = *exec.ErrorMessage
	}
	return events.ExecutionEvent{
		Type:        eventTypeForStatus(exec.Status),
		ExecutionID: exec.ExecutionID,
		WorkflowID:  exec.RunID,
		AgentNodeID: exec.AgentNodeID,
		Status:      exec.Status,
		Timestamp:   exec.UpdatedAt,
		Data:        data,
	}, true
}

func eventTypeForStatus(status string) events.ExecutionEventType {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case string(types.ExecutionStatusSucceeded):
		return events.ExecutionCompleted
	case string(types.ExecutionStatusFailed):
		return events.ExecutionFailed
	case string(types.ExecutionStatusCancelled):
		return events.ExecutionCancelledEvent
	default:
		return events.ExecutionUpdated
	}
}

func reasonerNodeStatus(agent *types.AgentNode) string {
	switch agent.HealthStatus {
	case types.HealthStatusActive:
		return "live"
	case types.HealthStatusInactive:
		return "down"
	default:
		return "stale"
	}
}

func hasTag(tags []string, want string) bool {
	for _, tag := range tags {
		if strings.EqualFold(strings.TrimSpace(tag), want) {
			return true
		}
	}
	return false
}

func fuzzyReasonerMatch(value, query string) bool {
	value = strings.ToLower(value)
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" || strings.Contains(value, query) {
		return true
	}
	pos := 0
	for _, r := range query {
		idx := strings.IndexRune(value[pos:], r)
		if idx < 0 {
			return false
		}
		pos += idx + 1
	}
	return true
}

func parseQueryBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "y":
		return true
	default:
		return false
	}
}

func parseCatalogTime(value *string) (time.Time, bool) {
	if value == nil || strings.TrimSpace(*value) == "" {
		return time.Time{}, false
	}
	ts, err := time.Parse(time.RFC3339, strings.TrimSpace(*value))
	return ts, err == nil
}

func parseFromStep(value string) int {
	if strings.TrimSpace(value) == "" {
		return 0
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}
