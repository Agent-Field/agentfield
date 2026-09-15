package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// CursorProvider invokes the Cursor CLI (`agent`) as a subprocess.
// It uses `agent -p ... --output-format json`, which emits a single JSON
// object on completion carrying the final result text and a session id.
type CursorProvider struct {
	BinPath string

	// runCLI is the subprocess runner, injectable for tests. It defaults to
	// RunCLI. A nil value falls back to RunCLI at call time.
	runCLI func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error)
}

// NewCursorProvider creates a Cursor provider. If binPath is empty, it
// defaults to "agent" (the Cursor CLI binary name).
func NewCursorProvider(binPath string) *CursorProvider {
	if binPath == "" {
		binPath = "agent"
	}
	return &CursorProvider{BinPath: binPath, runCLI: RunCLI}
}

func (p *CursorProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	// -p runs headless; --force/--trust bypass interactive confirmations so the
	// agent can act non-interactively; --output-format json yields one JSON
	// object on completion instead of streamed text.
	cmd := []string{p.BinPath, "-p", "--force", "--trust", "--output-format", "json"}

	// Cursor operates on a workspace directory. ProjectDir is the canonical
	// caller-facing field; fall back to Cwd so an explicit working directory is
	// still honoured.
	dir := options.ProjectDir
	if dir == "" {
		dir = options.Cwd
	}
	if dir != "" {
		cmd = append(cmd, "--workspace", dir)
	}

	// Cursor has no reasoning-effort flag; strip any "#variant" suffix so the
	// CLI still receives a valid model id, and drop the variant.
	modelValue, _ := options.resolveModelAndVariant()
	if modelValue != "" {
		cmd = append(cmd, "--model", modelValue)
	}

	if options.ResumeSessionID != "" {
		cmd = append(cmd, "--resume", options.ResumeSessionID)
	}

	if options.PermissionMode == "plan" {
		cmd = append(cmd, "--mode", "plan")
	}

	// Prompt is positional, last.
	cmd = append(cmd, prompt)

	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	// Cwd for the subprocess: prefer an explicit Cwd, else the project dir.
	cwd := options.Cwd
	if cwd == "" {
		cwd = options.ProjectDir
	}

	runCLI := p.runCLI
	if runCLI == nil {
		runCLI = RunCLI
	}

	startAPI := time.Now()

	cliResult, err := runCLI(ctx, cmd, env, cwd, options.timeout())
	apiMS := int(time.Since(startAPI).Milliseconds())

	if err != nil {
		if isExecNotFound(err) {
			return &RawResult{
				IsError: true,
				ErrorMessage: fmt.Sprintf(
					"Cursor binary not found at '%s'. Install the Cursor CLI: https://docs.cursor.com/en/cli/overview",
					p.BinPath,
				),
				FailureType: FailureCrash,
				Metrics:     Metrics{},
			}, nil
		}
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

	raw := &RawResult{
		Metrics: Metrics{
			DurationAPIMS: apiMS,
		},
		ReturnCode: cliResult.ReturnCode,
	}

	stdout := strings.TrimSpace(cliResult.Stdout)
	cleanStderr := StripANSI(strings.TrimSpace(cliResult.Stderr))

	if stdout != "" {
		p.parseJSONOutput(stdout, raw)
	}

	switch {
	case cliResult.ReturnCode < 0:
		raw.IsError = true
		raw.FailureType = FailureCrash
		if cleanStderr != "" {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d. stderr: %.500s",
				-cliResult.ReturnCode, cleanStderr)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d.", -cliResult.ReturnCode)
		}
	case cliResult.ReturnCode != 0 && raw.Result == "":
		raw.IsError = true
		raw.FailureType = FailureCrash
		if cleanStderr != "" {
			raw.ErrorMessage = truncate(cleanStderr, 1000)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d and produced no output.",
				cliResult.ReturnCode)
		}
	case cliResult.ReturnCode != 0:
		raw.IsError = true
		raw.ErrorMessage = fmt.Sprintf("Process exited with code %d", cliResult.ReturnCode)
	}

	return raw, nil
}

// parseJSONOutput extracts the final result text and session id from Cursor's
// single-object JSON output:
//
//	{"type":"result","subtype":"success","result":"<text>",
//	 "session_id":"<uuid>","duration_ms":1234}
//
// When stdout is not valid JSON (older CLI, or a plain-text failure), the raw
// trimmed stdout is surfaced as the result so output is not silently dropped.
func (p *CursorProvider) parseJSONOutput(stdout string, raw *RawResult) {
	var event map[string]any
	if err := json.Unmarshal([]byte(stdout), &event); err != nil {
		raw.Result = stdout
		raw.Metrics.NumTurns = 1
		return
	}

	raw.Messages = []map[string]any{event}

	if r, ok := event["result"].(string); ok {
		raw.Result = r
	}
	if sid, ok := event["session_id"].(string); ok {
		raw.Metrics.SessionID = sid
	}

	// Cursor reports the agent's own wall time; prefer it over our measured
	// API duration when present.
	if ms, ok := event["duration_ms"].(float64); ok {
		raw.Metrics.DurationAPIMS = int(ms)
	}

	if raw.Result != "" {
		raw.Metrics.NumTurns = 1
	}
}
