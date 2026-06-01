package harness

import "context"

const (
	// ProviderOpenCode is the provider name for OpenCode CLI.
	ProviderOpenCode = "opencode"
	// ProviderClaudeCode is the provider name for Claude Code CLI.
	ProviderClaudeCode = "claude-code"
	// ProviderCodex is the provider name for Codex CLI.
	ProviderCodex = "codex"
	// ProviderGemini is the provider name for Gemini CLI.
	ProviderGemini = "gemini"
)

// Provider is the interface that CLI-based harness providers implement.
// Each provider knows how to invoke a specific coding agent (opencode,
// claude-code, etc.) and return a RawResult.
type Provider interface {
	Execute(ctx context.Context, prompt string, options Options) (*RawResult, error)
}

// Options control a single harness invocation. Fields are optional;
// zero values mean "use default".
type Options struct {
	// Provider name: "opencode", "claude-code".
	Provider string

	// Model identifier passed to the coding agent.
	Model string

	// MaxTurns limits the number of agent iterations.
	MaxTurns int

	// PermissionMode controls tool permissions ("auto", "plan").
	PermissionMode string

	// SystemPrompt is prepended to the user prompt.
	SystemPrompt string

	// Env is additional environment variables for the subprocess.
	// Empty string values unset variables in the subprocess environment.
	Env map[string]string

	// Cwd is the working directory for the subprocess.
	Cwd string

	// ProjectDir is the project root the coding agent explores.
	// When set, the output file is placed inside this directory so the
	// coding agent's Write tool can reach it.
	ProjectDir string

	// Tools is a list of allowed tools for the agent.
	Tools []string

	// MaxBudgetUSD is a cost cap (provider-dependent).
	MaxBudgetUSD float64

	// ResumeSessionID resumes a previous session (provider-dependent).
	ResumeSessionID string

	// BinPath overrides the provider binary path.
	BinPath string

	// Timeout in seconds for the subprocess. 0 means provider default.
	Timeout int

	// MaxRetries for transient errors. Default 3.
	MaxRetries int

	// InitialDelay in seconds for retry backoff. Default 1.0.
	InitialDelay float64

	// MaxDelay in seconds for retry backoff. Default 30.0.
	MaxDelay float64

	// BackoffFactor for exponential backoff. Default 2.0.
	BackoffFactor float64

	// SchemaMaxRetries controls how many times to retry when schema
	// validation fails. Default 2.
	SchemaMaxRetries int
}

func (o Options) maxRetries() int {
	if o.MaxRetries > 0 {
		return o.MaxRetries
	}
	return 3
}

func (o Options) initialDelay() float64 {
	if o.InitialDelay > 0 {
		return o.InitialDelay
	}
	return 1.0
}

func (o Options) maxDelay() float64 {
	if o.MaxDelay > 0 {
		return o.MaxDelay
	}
	return 30.0
}

func (o Options) backoffFactor() float64 {
	if o.BackoffFactor > 0 {
		return o.BackoffFactor
	}
	return 2.0
}

func (o Options) schemaMaxRetries() int {
	if o.SchemaMaxRetries > 0 {
		return o.SchemaMaxRetries
	}
	return 2
}

func (o Options) timeout() int {
	if o.Timeout > 0 {
		return o.Timeout
	}
	return 600
}
