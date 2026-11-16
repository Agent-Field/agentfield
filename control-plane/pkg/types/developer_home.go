package types

import "time"

// ExecutionStatusCounts is an exact count of execution records whose
// started_at falls inside a window, split by outcome. It is computed with a
// COUNT query, so it never stops at a row limit.
type ExecutionStatusCounts struct {
	Total     int64 `json:"total"`
	Succeeded int64 `json:"succeeded"`
	// Failed counts failed and timed-out executions.
	Failed    int64 `json:"failed"`
	Cancelled int64 `json:"cancelled"`
	// Terminal counts every execution that finished: succeeded, failed,
	// timed out or cancelled.
	Terminal int64 `json:"terminal"`
}

// DeveloperHomeQuery selects the window the Home aggregates cover.
type DeveloperHomeQuery struct {
	Start time.Time
	End   time.Time
	// BucketSeconds splits the window into equal buckets for the sparklines.
	BucketSeconds int64
	// BaselineStart is the beginning of the comparison period, which always
	// ends where the window starts. A zero value skips the baseline.
	BaselineStart time.Time
	// Limit caps every list in the result (problems, endpoints, approvals).
	Limit int
}

// DeveloperHomeData is everything the Home page reads from the execution
// tables for one window. Every number is a COUNT, SUM or rank over recorded
// rows; nothing is estimated.
type DeveloperHomeData struct {
	Totals    DeveloperHomeWindowStats    `json:"totals"`
	Baseline  *DeveloperHomeBaseline      `json:"baseline,omitempty"`
	Series    []DeveloperHomeBucket       `json:"series"`
	Problems  []DeveloperHomeProblem      `json:"problems"`
	Edges     []DeveloperHomeEdge         `json:"edges"`
	Entries   []DeveloperHomeEntryEdge    `json:"entries"`
	Endpoints []DeveloperHomeEndpoint     `json:"endpoints"`
	Approvals DeveloperHomeApprovals      `json:"approvals"`
	Spend     *DeveloperHomeSpend         `json:"spend,omitempty"`
	Webhooks  *DeveloperHomeWebhooks      `json:"webhooks,omitempty"`
	Triggers  *DeveloperHomeTriggers      `json:"triggers,omitempty"`
	Forwarder *DeveloperHomeForwarderDrop `json:"forwarder_dead_letters,omitempty"`
}

// DeveloperHomeWindowStats is the window's totals.
type DeveloperHomeWindowStats struct {
	Counts ExecutionStatusCounts `json:"counts"`
	// Held counts executions in the window that are waiting (an approval or
	// another person-side pause).
	Held       int64  `json:"held"`
	RunningNow int64  `json:"running_now"`
	P50Ms      *int64 `json:"p50_ms,omitempty"`
	P95Ms      *int64 `json:"p95_ms,omitempty"`
}

// DeveloperHomeBaseline is the same totals over the comparison period. Span
// is the part of that period that actually has recorded executions in it, so
// a three-day-old install is compared with three days, not seven.
type DeveloperHomeBaseline struct {
	Start  time.Time             `json:"start"`
	End    time.Time             `json:"end"`
	Span   float64               `json:"span_seconds"`
	Counts ExecutionStatusCounts `json:"counts"`
	P95Ms  *int64                `json:"p95_ms,omitempty"`
	// BusyMs is the summed duration of finished executions in the period.
	// BusyMs divided by the span is the average number in flight.
	BusyMs int64 `json:"busy_ms"`
}

// DeveloperHomeBucket is one sparkline point.
type DeveloperHomeBucket struct {
	Start  time.Time `json:"start"`
	Calls  int64     `json:"calls"`
	Failed int64     `json:"failed"`
	// Terminal is the number of calls in the bucket that finished.
	Terminal int64  `json:"terminal"`
	P95Ms    *int64 `json:"p95_ms,omitempty"`
}

// DeveloperHomeProblem is one failure cause: the failing execution's own
// agent and reasoner plus its normalized error message.
type DeveloperHomeProblem struct {
	AgentNodeID        string    `json:"agent_node_id"`
	ReasonerID         string    `json:"reasoner_id"`
	Status             string    `json:"status"`
	Message            string    `json:"message"`
	Count              int64     `json:"count"`
	FirstSeen          time.Time `json:"first_seen"`
	LastSeen           time.Time `json:"last_seen"`
	ExampleExecutionID string    `json:"example_execution_id"`
	ExampleRunID       string    `json:"example_run_id"`
	// Buckets holds the count per window bucket, aligned with Series.
	Buckets []int64 `json:"buckets"`
}

// DeveloperHomeEdge is caller agent to target agent, built from each child
// execution's parent_execution_id.
type DeveloperHomeEdge struct {
	Caller string `json:"caller"`
	Target string `json:"target"`
	Calls  int64  `json:"calls"`
	Failed int64  `json:"failed"`
	Held   int64  `json:"held"`
}

// DeveloperHomeEntryEdge is a call that started a run (no parent execution),
// either sent directly to the control plane or dispatched by a trigger.
type DeveloperHomeEntryEdge struct {
	// Source is the trigger's source name, or empty for a direct call.
	Source string `json:"source"`
	Target string `json:"target"`
	Calls  int64  `json:"calls"`
	Failed int64  `json:"failed"`
	Held   int64  `json:"held"`
}

// DeveloperHomeEndpoint is one agent.reasoner as an API.
type DeveloperHomeEndpoint struct {
	AgentNodeID string `json:"agent_node_id"`
	ReasonerID  string `json:"reasoner_id"`
	Calls       int64  `json:"calls"`
	Failed      int64  `json:"failed"`
	Terminal    int64  `json:"terminal"`
	P50Ms       *int64 `json:"p50_ms,omitempty"`
	P95Ms       *int64 `json:"p95_ms,omitempty"`
	// CallerAgents lists the distinct agents whose executions called this
	// endpoint, by call count.
	CallerAgents []string `json:"caller_agents"`
	DirectCalls  int64    `json:"direct_calls"`
	TriggerCalls int64    `json:"trigger_calls"`
}

// DeveloperHomeApproval is one execution waiting for a person's decision.
type DeveloperHomeApproval struct {
	ExecutionID string     `json:"execution_id"`
	RunID       string     `json:"run_id"`
	AgentNodeID string     `json:"agent_node_id"`
	ReasonerID  string     `json:"reasoner_id"`
	RequestURL  string     `json:"request_url,omitempty"`
	RequestedAt *time.Time `json:"requested_at,omitempty"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
}

// DeveloperHomeApprovals is every pending approval, with the oldest first.
type DeveloperHomeApprovals struct {
	Total int64                   `json:"total"`
	Items []DeveloperHomeApproval `json:"items"`
}

// DeveloperHomeSpend is recorded token cost. It is present only when usage
// rows exist in the window or the baseline.
type DeveloperHomeSpend struct {
	WindowCostUSD   *float64 `json:"window_cost_usd,omitempty"`
	WindowTokens    int64    `json:"window_tokens"`
	WindowRows      int64    `json:"window_rows"`
	BaselineCostUSD *float64 `json:"baseline_cost_usd,omitempty"`
	BaselineRows    int64    `json:"baseline_rows"`
}

// DeveloperHomeWebhook is one result webhook that ran out of attempts.
type DeveloperHomeWebhook struct {
	ExecutionID   string     `json:"execution_id"`
	URL           string     `json:"url"`
	AttemptCount  int        `json:"attempt_count"`
	LastError     string     `json:"last_error,omitempty"`
	LastAttemptAt *time.Time `json:"last_attempt_at,omitempty"`
}

// DeveloperHomeWebhooks counts result webhooks registered in the window. It
// is present only when at least one was registered.
type DeveloperHomeWebhooks struct {
	Registered int64                  `json:"registered"`
	Delivered  int64                  `json:"delivered"`
	Failed     int64                  `json:"failed"`
	Pending    int64                  `json:"pending"`
	Recent     []DeveloperHomeWebhook `json:"recent_failures"`
}

// DeveloperHomeTriggers counts inbound events in the window. It is present
// only when at least one trigger is configured.
type DeveloperHomeTriggers struct {
	Configured  int64 `json:"configured"`
	Enabled     int64 `json:"enabled"`
	Events      int64 `json:"events"`
	Dispatched  int64 `json:"dispatched"`
	Failed      int64 `json:"failed"`
	DeadLetters int64 `json:"dead_letters"`
}

// DeveloperHomeForwarderDrop is the observability forwarder's dead-letter
// queue depth.
type DeveloperHomeForwarderDrop struct {
	DeadLetters int64 `json:"dead_letters"`
}

// DeveloperHomeEntryPoint is an entry reasoner (a run's root execution) and
// how many of its runs in the window passed through a given agent.
type DeveloperHomeEntryPoint struct {
	AgentNodeID string `json:"agent_node_id"`
	ReasonerID  string `json:"reasoner_id"`
	Runs        int64  `json:"runs"`
	// FailedRuns counts the runs whose execution on the agent failed.
	FailedRuns int64 `json:"failed_runs"`
}
