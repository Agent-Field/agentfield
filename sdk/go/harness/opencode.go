package harness

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var runCLI = RunCLI
var (
	openCodeSemaphore chan struct{}
	semOnce           sync.Once
)

const defaultMaxConcurrent = 4

// OpenCodeProvider invokes the opencode CLI as a subprocess.
type OpenCodeProvider struct {
	BinPath   string
	ServerURL string
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
	return &OpenCodeProvider{BinPath: binPath, ServerURL: serverURL}
}

func (p *OpenCodeProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	cmd := []string{p.BinPath}

	// OpenCode uses -c for working directory.
	if options.ProjectDir != "" {
		cmd = append(cmd, "-c", options.ProjectDir)
	}

	// Quiet mode suppresses spinner (avoids TTY errors in subprocess).
	cmd = append(cmd, "-q")

	// Prepend system prompt if provided.
	effectivePrompt := prompt
	if options.SystemPrompt != "" {
		effectivePrompt = fmt.Sprintf(
			"SYSTEM INSTRUCTIONS:\n%s\n\n---\n\nUSER REQUEST:\n%s",
			strings.TrimSpace(options.SystemPrompt), prompt,
		)
	}

	// Non-interactive prompt mode with -p flag.
	cmd = append(cmd, "-p", effectivePrompt)

	// Build environment
	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
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

	cliResult, err := runCLI(ctx, cmd, env, options.Cwd, options.timeout())
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
