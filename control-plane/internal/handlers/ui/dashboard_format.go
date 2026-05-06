package ui

import "time"

// TimeRangeInfo describes the time range used for the dashboard query
type TimeRangeInfo struct {
	StartTime time.Time       `json:"start_time"`
	EndTime   time.Time       `json:"end_time"`
	Preset    TimeRangePreset `json:"preset,omitempty"`
}

// ComparisonData contains delta information comparing current to previous period
type ComparisonData struct {
	PreviousPeriod TimeRangeInfo         `json:"previous_period"`
	OverviewDelta  EnhancedOverviewDelta `json:"overview_delta"`
}

// EnhancedOverviewDelta contains changes compared to the previous period
type EnhancedOverviewDelta struct {
	ExecutionsDelta     int     `json:"executions_delta"`
	ExecutionsDeltaPct  float64 `json:"executions_delta_pct"`
	SuccessRateDelta    float64 `json:"success_rate_delta"`
	AvgDurationDeltaMs  float64 `json:"avg_duration_delta_ms"`
	AvgDurationDeltaPct float64 `json:"avg_duration_delta_pct"`
}

// HotspotSummary contains top error contributors by reasoner
type HotspotSummary struct {
	TopFailingReasoners []HotspotItem `json:"top_failing_reasoners"`
}

// HotspotItem represents a single reasoner's failure statistics
type HotspotItem struct {
	ReasonerID       string       `json:"reasoner_id"`
	TotalExecutions  int          `json:"total_executions"`
	FailedExecutions int          `json:"failed_executions"`
	ErrorRate        float64      `json:"error_rate"`
	ContributionPct  float64      `json:"contribution_pct"`
	TopErrors        []ErrorCount `json:"top_errors"`
}

// ErrorCount tracks error message frequency
type ErrorCount struct {
	Message string `json:"message"`
	Count   int    `json:"count"`
}

// ActivityPatterns contains temporal patterns for failures and usage
type ActivityPatterns struct {
	// HourlyHeatmap is a 7x24 matrix [dayOfWeek][hourOfDay]
	HourlyHeatmap [][]HeatmapCell `json:"hourly_heatmap"`
}

// HeatmapCell contains execution statistics for a specific day/hour combination
type HeatmapCell struct {
	Total     int     `json:"total"`
	Failed    int     `json:"failed"`
	ErrorRate float64 `json:"error_rate"`
}

// Enhanced dashboard response structures
type EnhancedDashboardResponse struct {
	GeneratedAt      time.Time          `json:"generated_at"`
	TimeRange        TimeRangeInfo      `json:"time_range"`
	Overview         EnhancedOverview   `json:"overview"`
	ExecutionTrends  ExecutionTrends    `json:"execution_trends"`
	AgentHealth      AgentHealthSummary `json:"agent_health"`
	Workflows        WorkflowInsights   `json:"workflows"`
	Incidents        []IncidentItem     `json:"incidents"`
	Comparison       *ComparisonData    `json:"comparison,omitempty"`
	Hotspots         HotspotSummary     `json:"hotspots"`
	ActivityPatterns ActivityPatterns   `json:"activity_patterns"`
}

type EnhancedOverview struct {
	TotalAgents          int     `json:"total_agents"`
	ActiveAgents         int     `json:"active_agents"`
	DegradedAgents       int     `json:"degraded_agents"`
	OfflineAgents        int     `json:"offline_agents"`
	TotalReasoners       int     `json:"total_reasoners"`
	TotalSkills          int     `json:"total_skills"`
	ExecutionsLast24h    int     `json:"executions_last_24h"`
	ExecutionsLast7d     int     `json:"executions_last_7d"`
	SuccessRate24h       float64 `json:"success_rate_24h"`
	AverageDurationMs24h float64 `json:"average_duration_ms_24h"`
	MedianDurationMs24h  float64 `json:"median_duration_ms_24h"`
}

type ExecutionTrends struct {
	Last24h   ExecutionWindowMetrics `json:"last_24h"`
	Last7Days []ExecutionTrendPoint  `json:"last_7_days"`
}

type ExecutionWindowMetrics struct {
	Total             int     `json:"total"`
	Succeeded         int     `json:"succeeded"`
	Failed            int     `json:"failed"`
	SuccessRate       float64 `json:"success_rate"`
	AverageDurationMs float64 `json:"average_duration_ms"`
	ThroughputPerHour float64 `json:"throughput_per_hour"`
}

type ExecutionTrendPoint struct {
	Date      string `json:"date"`
	Total     int    `json:"total"`
	Succeeded int    `json:"succeeded"`
	Failed    int    `json:"failed"`
}

type AgentHealthSummary struct {
	Total    int               `json:"total"`
	Active   int               `json:"active"`
	Degraded int               `json:"degraded"`
	Offline  int               `json:"offline"`
	Agents   []AgentHealthItem `json:"agents"`
}

type AgentHealthItem struct {
	ID            string    `json:"id"`
	TeamID        string    `json:"team_id"`
	Version       string    `json:"version"`
	Status        string    `json:"status"`
	Health        string    `json:"health"`
	Lifecycle     string    `json:"lifecycle"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	Reasoners     int       `json:"reasoners"`
	Skills        int       `json:"skills"`
	Uptime        string    `json:"uptime,omitempty"`
}

type WorkflowInsights struct {
	TopWorkflows      []WorkflowStat           `json:"top_workflows"`
	ActiveRuns        []ActiveWorkflowRun      `json:"active_runs"`
	LongestExecutions []CompletedExecutionStat `json:"longest_executions"`
}

type WorkflowStat struct {
	WorkflowID       string    `json:"workflow_id"`
	Name             string    `json:"name,omitempty"`
	TotalExecutions  int       `json:"total_executions"`
	SuccessRate      float64   `json:"success_rate"`
	FailedExecutions int       `json:"failed_executions"`
	AverageDuration  float64   `json:"average_duration_ms"`
	LastActivity     time.Time `json:"last_activity"`
}

type ActiveWorkflowRun struct {
	ExecutionID string    `json:"execution_id"`
	WorkflowID  string    `json:"workflow_id"`
	Name        string    `json:"name,omitempty"`
	StartedAt   time.Time `json:"started_at"`
	ElapsedMs   int64     `json:"elapsed_ms"`
	AgentNodeID string    `json:"agent_node_id"`
	ReasonerID  string    `json:"reasoner_id"`
	Status      string    `json:"status"`
}

type CompletedExecutionStat struct {
	ExecutionID string     `json:"execution_id"`
	WorkflowID  string     `json:"workflow_id"`
	Name        string     `json:"name,omitempty"`
	DurationMs  int64      `json:"duration_ms"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	Status      string     `json:"status"`
}

type IncidentItem struct {
	ExecutionID string     `json:"execution_id"`
	WorkflowID  string     `json:"workflow_id"`
	Name        string     `json:"name,omitempty"`
	Status      string     `json:"status"`
	StartedAt   time.Time  `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	AgentNodeID string     `json:"agent_node_id"`
	ReasonerID  string     `json:"reasoner_id"`
	Error       string     `json:"error,omitempty"`
}
