package types

import "time"

// ExecutionUsage is a single persisted token/cost usage row tied to an
// execution. One execution may produce many rows (one per LLM/harness entry
// reported by the agent SDK in the result envelope's "usage" object).
type ExecutionUsage struct {
	ID                  int64
	ExecutionID         string
	WorkflowID          string
	AgentNodeID         string
	Reasoner            string
	Source              string // "llm" | "harness"
	Provider            string // e.g. "anthropic", "openrouter" (may be empty)
	Model               string
	Harness             string // e.g. "claude_code" (may be empty)
	InputTokens         int64
	OutputTokens        int64
	CacheReadTokens     int64
	CacheCreationTokens int64
	TotalTokens         int64
	CostUSD             *float64 // nil when the agent reported no cost
	CostSource          string   // "provider" | "litellm" | ""
	CreatedAt           time.Time
}

// UsageStatsTotals holds the aggregate token/cost totals over a time window.
type UsageStatsTotals struct {
	CostUSD             *float64
	InputTokens         int64
	OutputTokens        int64
	CacheReadTokens     int64
	CacheCreationTokens int64
	TotalTokens         int64
	ExecutionsWithUsage int64
}

// UsageStatsGroup is one grouped bucket (by model, provider, agent, or harness).
type UsageStatsGroup struct {
	Key          string
	Provider     string // only populated for the by_model grouping
	CostUSD      *float64
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	Entries      int64
}

// UsageStatsAggregation is the full aggregation result for a usage-stats query.
type UsageStatsAggregation struct {
	Totals      UsageStatsTotals
	ByModel     []UsageStatsGroup
	ByProvider  []UsageStatsGroup
	ByAgent     []UsageStatsGroup
	ByHarness   []UsageStatsGroup
	LastUpdated *time.Time
}

// UsageTimeseriesPoint is one zero-filled time bucket in a usage series.
// Start is the bucket's start time (UTC). CostUSD is nil when the bucket has no
// rows, or has rows but all of them report a null cost.
type UsageTimeseriesPoint struct {
	Start       time.Time
	TotalTokens int64
	CostUSD     *float64
}

// UsageTimeseries is a bucketed token/cost series over a window. Points always
// has exactly the requested number of buckets, ascending by Start, zero-filled.
type UsageTimeseries struct {
	BucketSeconds int64
	Points        []UsageTimeseriesPoint
}

// UsageModelTimeseriesPoint is one zero-filled token bucket for a per-model
// series. Tokens only (no cost). Start is the bucket's start time (UTC).
type UsageModelTimeseriesPoint struct {
	Start       time.Time
	TotalTokens int64
}

// UsageModelSeries is one per-model token series over a bucket grid. Key is the
// model name, or "other" for the rolled-up remainder beyond the top models.
// Points always has exactly the requested number of buckets, ascending by
// Start, zero-filled.
type UsageModelSeries struct {
	Key    string
	Points []UsageModelTimeseriesPoint
}
