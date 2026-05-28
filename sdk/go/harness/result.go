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

	NumTurns   int
	DurationMS int
	SessionID  string
	Messages   []map[string]any
}

// Text returns the result text, or empty string if nil.
func (r *Result) Text() string {
	return r.Result
}
