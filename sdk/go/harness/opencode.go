package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	openCodeSemaphore chan struct{}
	semOnce           sync.Once
)

const defaultMaxConcurrent = 4

// OpenCodeProvider invokes the opencode CLI as a subprocess.
type OpenCodeProvider struct {
	BinPath   string
	ServerURL string
	runCLI    func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error)
}

func getSemaphore() chan struct{} {
	semOnce.Do(func() {
		max := defaultMaxConcurrent
		if val := os.Getenv("OPENCODE_MAX_CONCURRENT"); val != "" {
			if i, err := strconv.Atoi(val); err == nil && i > 0 {
				max = i
			}
		}
		openCodeSemaphore = make(chan struct{}, max)
	})
	return openCodeSemaphore
}

// NewOpenCodeProvider creates an OpenCode provider. If binPath is empty,
// it defaults to "opencode".
func NewOpenCodeProvider(binPath, serverURL string) *OpenCodeProvider {
	if binPath == "" {
		binPath = "opencode"
	}
	if serverURL == "" {
		serverURL = os.Getenv("OPENCODE_SERVER")
	}
	return &OpenCodeProvider{BinPath: binPath, ServerURL: serverURL, runCLI: RunCLI}
}

func (p *OpenCodeProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	// opencode 1.14+ moved non-interactive execution to the `run` subcommand.
	// The legacy top-level `-c <dir> -q -p <prompt>` surface was rebound:
	//   -c → --continue (resume previous session)
	//   -p → --password (provider password)
	// so the old invocation made the binary print help and exit without
	// running, leaving callers with empty trajectories. See issue #517.
	cmd := []string{p.BinPath, "run"}

	// OpenCode uses --dir for the project directory the agent operates on.
	// ProjectDir is the canonical caller-facing field; fall back to Cwd if
	// only that is set so we still honour the caller's explicit working
	// directory.
	dir := options.ProjectDir
	if dir == "" {
		dir = options.Cwd
	}
	if dir != "" {
		cmd = append(cmd, "--dir", dir)
	}

	// Pass model via -m on the run subcommand when supplied.
	if options.Model != "" {
		cmd = append(cmd, "-m", options.Model)
	}

	// opencode v1.14 does not accept --dangerously-skip-permissions on the
	// `run` subcommand — passing it makes yargs print the run-help screen
	// to stdout and exit 0, which the SDK then captures as the LLM
	// response. opencode in non-TTY mode proceeds without permission
	// prompting, so no flag is needed. See agentfield#582.

	// Prepend system prompt if provided. OpenCode has no native
	// --system-prompt flag, so inline it ahead of the user prompt.
	effectivePrompt := prompt
	if options.SystemPrompt != "" {
		effectivePrompt = fmt.Sprintf(
			"SYSTEM INSTRUCTIONS:\n%s\n\n---\n\nUSER REQUEST:\n%s",
			strings.TrimSpace(options.SystemPrompt), prompt,
		)
	}

	// Prompt is positional on `opencode run` (replaces deprecated -p).
	cmd = append(cmd, effectivePrompt)

	// Build environment
	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(options.Model)), "openrouter/") {
		if _, callerSet := env["OPENCODE_CONFIG_CONTENT"]; !callerSet && os.Getenv("OPENCODE_CONFIG_CONTENT") == "" {
			attributionEnv := mergedProcessEnv(env)
			headers := openRouterAttributionHeaders(attributionEnv)
			modelSlug := strings.TrimPrefix(options.Model, "openrouter/")
			if modelSlug != "" && len(headers) > 0 {
				content := map[string]any{
					"provider": map[string]any{
						"openrouter": map[string]any{
							"models": map[string]any{
								modelSlug: map[string]any{"headers": headers},
							},
						},
					},
				}
				if encoded, err := json.Marshal(content); err == nil {
					env["OPENCODE_CONFIG_CONTENT"] = string(encoded)
				}
			}
		}
	}

	sem := getSemaphore()
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Use a temp data dir to isolate opencode state.
	tempDataDir, err := os.MkdirTemp("", ".agentfield-opencode-data-")
	if err != nil {
		return nil, fmt.Errorf("creating temp data dir: %w", err)
	}
	defer os.RemoveAll(tempDataDir)
	env["XDG_DATA_HOME"] = tempDataDir

	startAPI := time.Now()

	cliResult, err := p.runCLI(ctx, cmd, env, options.Cwd, options.timeout())
	apiMS := int(time.Since(startAPI).Milliseconds())

	if err != nil {
		// Check if it's a "not found" error
		if isExecNotFound(err) {
			return &RawResult{
				IsError: true,
				ErrorMessage: fmt.Sprintf(
					"OpenCode binary not found at '%s'. Install OpenCode: https://opencode.ai",
					p.BinPath,
				),
				FailureType: FailureCrash,
				Metrics:     Metrics{},
			}, nil
		}
		// Timeout
		if strings.Contains(err.Error(), "timed out") {
			return &RawResult{
				IsError:      true,
				ErrorMessage: err.Error(),
				FailureType:  FailureTimeout,
				Metrics:      Metrics{DurationAPIMS: apiMS},
			}, nil
		}
		return nil, err
	}

	resultText := strings.TrimSpace(cliResult.Stdout)
	cleanStderr := StripANSI(strings.TrimSpace(cliResult.Stderr))

	raw := &RawResult{
		Result:   resultText,
		Messages: nil,
		Metrics: Metrics{
			DurationAPIMS: apiMS,
			NumTurns:      1,
			SessionID:     "",
		},
		ReturnCode: cliResult.ReturnCode,
	}

	if cliResult.ReturnCode < 0 {
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d. stderr: %.500s",
				-cliResult.ReturnCode, cleanStderr)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d.", -cliResult.ReturnCode)
		}
	} else if cliResult.ReturnCode != 0 && resultText == "" {
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = truncate(cleanStderr, 1000)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d and produced no output.", cliResult.ReturnCode)
		}
	}

	if resultText == "" {
		raw.Metrics.NumTurns = 0
	}

	return raw, nil
}

func mergedProcessEnv(overrides map[string]string) map[string]string {
	merged := make(map[string]string)
	for _, entry := range os.Environ() {
		key, value, found := strings.Cut(entry, "=")
		if found {
			merged[key] = value
		}
	}
	for k, v := range overrides {
		if v == "" {
			delete(merged, k)
		} else {
			merged[k] = v
		}
	}
	return merged
}
