package agent

import (
	"context"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// harness.go integrates the harness package with the Agent struct,
// providing lazy initialization and a convenience Harness() method.
// HarnessConfig configures the default harness runner for the agent.
type HarnessConfig struct {
	// Provider is the default provider: "claude-code", "codex", "gemini", or "opencode".
	Provider string

	// Model is the default model identifier.
	Model string

	// MaxTurns is the default max agent iterations.
	MaxTurns int

	// PermissionMode is the default permission mode ("auto", "plan").
	PermissionMode string

	// Env is additional environment variables for the subprocess.
	Env map[string]string

	// BinPath overrides the provider binary path.
	BinPath string

	// Timeout in seconds for the subprocess. Default 600.
	Timeout int

	// MaxRetries for transient errors. Default 3.
	MaxRetries int

	// SchemaMaxRetries for schema validation failures. Default 2.
	SchemaMaxRetries int
}

// HarnessRunner returns the agent's lazily-initialized harness runner.
func (a *Agent) HarnessRunner() *harness.Runner {
	a.initMu.Lock()
	defer a.initMu.Unlock()
	if a.harnessRunner == nil {
		opts := harness.Options{}
		if a.cfg.HarnessConfig != nil {
			hc := a.cfg.HarnessConfig
			opts.Provider = hc.Provider
			opts.Model = hc.Model
			opts.MaxTurns = hc.MaxTurns
			opts.PermissionMode = hc.PermissionMode
			opts.Env = hc.Env
			opts.BinPath = hc.BinPath
			opts.Timeout = hc.Timeout
			opts.MaxRetries = hc.MaxRetries
			opts.SchemaMaxRetries = hc.SchemaMaxRetries
		}
		a.harnessRunner = harness.NewRunner(opts)
	}
	return a.harnessRunner
}

// Harness dispatches a prompt to an external coding agent and returns
// structured results. It is the Go equivalent of Python SDK's .harness().
//
// Parameters:
//   - prompt: Task description for the coding agent.
//   - schema: JSON Schema as map[string]any (nil for unstructured output).
//   - dest: Pointer to struct for schema validation (nil if schema is nil).
//   - opts: Per-call option overrides (zero values use runner defaults).
//
// Example:
//
//	type ReviewResult struct {
//	    Findings []string `json:"findings"`
//	    Severity string   `json:"severity"`
//	}
//	var result ReviewResult
//	schema, _ := harness.StructToJSONSchema(result)
//	hr, err := agent.Harness(ctx, "Review this code...", schema, &result, harness.Options{
//	    Model: "sonnet",
//	})
func (a *Agent) Harness(ctx context.Context, prompt string, schema map[string]any, dest any, opts harness.Options) (*harness.Result, error) {
	return a.HarnessRunner().Run(ctx, prompt, schema, dest, opts)
}
