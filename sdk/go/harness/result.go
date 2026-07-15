// Package harness dispatches tasks to external coding agents (CLI subprocesses)
// and extracts structured results with schema validation and retry logic.
package harness

// FailureType classifies how a harness invocation failed so the runner
// can decide on a retry strategy.
type FailureType string

const (
	FailureNone     FailureType = "none"
	FailureCrash    FailureType = "crash"
	FailureTimeout  FailureType = "timeout"
	FailureAPIError FailureType = "api_error"
	FailureNoOutput FailureType = "no_output"
	FailureSchema   FailureType = "schema"
)

// Metrics captured from a single provider execution.
type Metrics struct {
	DurationMS    int
	DurationAPIMS int
	NumTurns      int
	SessionID     string

	// CostUSD is the cost in USD reported by the provider for this single
	// execution, or nil when the provider does not report a cost. Mirrors the
	// Python SDK's Metrics.total_cost_usd (Optional[float]): nil means
	// "unknown", not "free". Only providers that surface a native cost (e.g.
	// Claude Code's result JSON) populate this; providers whose Python
	// counterparts derive cost via litellm token estimation leave it nil in Go
	// because there is no Go equivalent of litellm's pricing database.
	CostUSD *float64
}

// RawResult is the output from a single provider execution before schema
// parsing.
type RawResult struct {
	Result       string
	Messages     []map[string]any
	Metrics      Metrics
	IsError      bool
	ErrorMessage string
	FailureType  FailureType
	ReturnCode   int
}

// Result is the final harness output after schema validation, retries, and
// metrics accumulation.
type Result struct {
	// Result is the raw text output from the provider.
	Result string

	// Parsed holds the validated and deserialized structured output.
	// The caller passes a pointer to a struct; on success it is populated.
	Parsed any

	IsError      bool
	ErrorMessage string
	FailureType  FailureType

	// CostUSD is the total cost in USD accumulated across every provider
	// execution that contributed to this result (including failed retry
	// attempts), or nil when no execution reported a cost. Mirrors the Python
	// SDK's HarnessResult.cost_usd (Optional[float]).
	CostUSD *float64

	NumTurns   int
	DurationMS int
	SessionID  string
	Messages   []map[string]any
}

// Text returns the result text, or empty string if nil.
func (r *Result) Text() string {
	return r.Result
}
