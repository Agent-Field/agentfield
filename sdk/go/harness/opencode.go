package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// openCodePromptViaStdin reports whether to hand the prompt to opencode over
// stdin instead of argv. On Windows the CLI on PATH is usually an npm .cmd
// shim that runs via cmd.exe, whose ~8k command-line cap real prompts blow
// straight through ("The command line is too long.") — pr-af style prompts
// that embed diff context die there, and schema retries (which append more
// text) can never recover. opencode reads the prompt from stdin when the
// positional arg is absent, so feed it that way there. POSIX keeps the
// battle-tested positional-arg path. Mirrors the Python SDK's
// _prompt_via_stdin (harness/providers/opencode.py). Package var so tests can
// exercise both paths on any OS.
var openCodePromptViaStdin = runtime.GOOS == "windows"

var (
	openCodeSemaphore chan struct{}
	semOnce           sync.Once
)

const defaultMaxConcurrent = 4

// OpenCodeProvider invokes the opencode CLI as a subprocess.
type OpenCodeProvider struct {
	BinPath   string
	ServerURL string
	runCLI    func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, stdin []byte) (*CLIResult, error)
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
	return &OpenCodeProvider{BinPath: binPath, ServerURL: serverURL, runCLI: RunCLIWithStdin}
}

func (p *OpenCodeProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	// opencode 1.14+ moved non-interactive execution to the `run` subcommand.
	// The legacy top-level `-c <dir> -q -p <prompt>` surface was rebound:
	//   -c → --continue (resume previous session)
	//   -p → --password (provider password)
	// so the old invocation made the binary print help and exit without
	// running, leaving callers with empty trajectories. See issue #517.
	//
	// --format json emits a JSONL event stream (step_start / text / step_finish
	// / tool_use / error) instead of plain text, which lets us recover the final
	// message, per-step cost, and turn count, and surface in-band error events.
	cmd := []string{p.BinPath, "run", "--format", "json"}

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

	// Prompt is positional on `opencode run` (replaces deprecated -p) on
	// POSIX; on Windows it goes over stdin instead (see openCodePromptViaStdin).
	var stdinPrompt []byte
	if openCodePromptViaStdin {
		stdinPrompt = []byte(effectivePrompt)
	} else {
		cmd = append(cmd, effectivePrompt)
	}

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

	cliResult, err := p.runCLI(ctx, cmd, env, options.Cwd, options.timeout(), stdinPrompt)
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

	cleanStderr := StripANSI(strings.TrimSpace(cliResult.Stderr))

	// Parse the JSON event stream. When opencode emitted no parseable events
	// (older versions, or a hard failure before any output), fall back to the
	// trimmed raw stdout so plain-text output is still surfaced.
	events := parseOpenCodeEvents(cliResult.Stdout)
	var resultText string
	if len(events) > 0 {
		resultText = extractOpenCodeFinalText(events)
	} else {
		resultText = strings.TrimSpace(cliResult.Stdout)
	}
	eventError := extractOpenCodeEventError(events)

	raw := &RawResult{
		Result:   resultText,
		Messages: events,
		Metrics: Metrics{
			DurationAPIMS: apiMS,
			SessionID:     "",
		},
		ReturnCode: cliResult.ReturnCode,
	}

	switch {
	case cliResult.ReturnCode < 0:
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d. stderr: %.500s",
				-cliResult.ReturnCode, cleanStderr)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d.", -cliResult.ReturnCode)
		}
	case cliResult.ReturnCode != 0 && resultText == "":
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = extractOpenCodeError(cleanStderr)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d and produced no output.", cliResult.ReturnCode)
		}
	case eventError != "" && resultText == "":
		raw.FailureType = FailureCrash
		raw.IsError = true
		raw.ErrorMessage = eventError
	case resultText == "" && cleanStderr != "" && matchesOpenCodeError(cleanStderr):
		// opencode sometimes exits 0 even on hard failures like "Model not
		// found" or auth errors — surface the real error from stderr instead
		// of silently returning empty output that downstream callers would
		// interpret as "the agent produced no valid result".
		raw.FailureType = FailureCrash
		raw.IsError = true
		raw.ErrorMessage = extractOpenCodeError(cleanStderr)
	}

	// Turn count: prefer the event-derived count, else 1 when a result exists.
	numTurns := countTurnsFromEvents(events)
	if numTurns == 0 && resultText != "" {
		numTurns = 1
	}
	raw.Metrics.NumTurns = numTurns
	raw.Metrics.CostUSD = costFromEvents(events)

	return raw, nil
}

// opencode CLI sometimes prints a hard error to stderr but exits 0 (notably
// "Model not found", auth errors, schema-validation failures). These patterns
// mark stderr as carrying a real failure rather than noise like the one-time
// SQLite migration prelude. Ported from providers/opencode.py:29-35.
var openCodeStderrErrorPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?m)^Error:`),
	regexp.MustCompile(`\bModel not found\b`),
	regexp.MustCompile(`\bAuthenticationError\b`),
	regexp.MustCompile(`\bUnauthorized\b`),
	regexp.MustCompile(`\bAPIError\b`),
}

func matchesOpenCodeError(stderr string) bool {
	for _, pat := range openCodeStderrErrorPatterns {
		if pat.MatchString(stderr) {
			return true
		}
	}
	return false
}

// parseOpenCodeEvents parses opencode's JSONL event stream, skipping any line
// that is not valid JSON (e.g. interleaved plain-text log lines).
func parseOpenCodeEvents(stdout string) []map[string]any {
	var events []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(stdout), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		events = append(events, event)
	}
	return events
}

// extractOpenCodeFinalText reconstructs the final assistant text from the event
// stream. Ported from _cli.extract_final_text (the branches opencode emits):
// step_start resets the accumulated text; "text" events append part/text
// content; result/message/assistant/turn.completed/item.completed carry a final
// message directly.
func extractOpenCodeFinalText(events []map[string]any) string {
	var resultText string
	var currentParts []string

	for _, event := range events {
		eventType, _ := event["type"].(string)
		switch eventType {
		case "step_start":
			currentParts = nil
		case "item.completed":
			if item, ok := event["item"].(map[string]any); ok {
				if it, _ := item["type"].(string); it == "agent_message" {
					if text, ok := item["text"].(string); ok && text != "" {
						resultText = text
					}
				}
			}
		case "result":
			if r, ok := event["result"].(string); ok {
				resultText = r
			} else if r, ok := event["text"].(string); ok {
				resultText = r
			}
		case "turn.completed":
			if text, ok := event["text"].(string); ok && text != "" {
				resultText = text
			}
		case "message", "assistant":
			if content, ok := event["content"].(string); ok && content != "" {
				resultText = content
			} else if content, ok := event["text"].(string); ok && content != "" {
				resultText = content
			}
		case "text":
			content := stringField(event, "text")
			if content == "" {
				content = stringField(event, "content")
			}
			if content == "" {
				if part, ok := event["part"].(map[string]any); ok {
					content = stringField(part, "text")
				}
			}
			if content != "" {
				currentParts = append(currentParts, content)
				resultText = strings.Join(currentParts, "")
			}
		}
	}
	return resultText
}

// stringField returns m[key] when it is a non-empty string, else "".
func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

// countTurnsFromEvents counts opencode turns: one per step_start event, or —
// when the stream has no step markers — one per tool_use event. Ported from
// providers/opencode.py:_count_turns_from_events.
func countTurnsFromEvents(events []map[string]any) int {
	stepStarts := 0
	toolUses := 0
	for _, event := range events {
		switch t, _ := event["type"].(string); t {
		case "step_start":
			stepStarts++
		case "tool_use":
			toolUses++
		}
	}
	if stepStarts > 0 {
		return stepStarts
	}
	return toolUses
}

// costFromEvents sums opencode per-step costs from step_finish events. Returns
// nil when no step carried a cost, so callers distinguish "unknown" from
// "$0.00". Ported from providers/opencode.py:_cost_from_events.
func costFromEvents(events []map[string]any) *float64 {
	total := 0.0
	found := false
	for _, event := range events {
		if t, _ := event["type"].(string); t != "step_finish" {
			continue
		}
		part, ok := event["part"].(map[string]any)
		if !ok {
			continue
		}
		// JSON numbers decode to float64; bool cost values never match here,
		// matching the Python guard against isinstance(cost, bool).
		if cost, ok := part["cost"].(float64); ok {
			total += cost
			found = true
		}
	}
	if !found {
		return nil
	}
	return &total
}

// extractOpenCodeEventError pulls a meaningful failure message from an in-band
// JSON "error" event. Ported from providers/opencode.py:_extract_opencode_event_error.
func extractOpenCodeEventError(events []map[string]any) string {
	for _, event := range events {
		if t, _ := event["type"].(string); t != "error" {
			continue
		}
		for _, key := range []string{"message", "error", "text"} {
			if v := strings.TrimSpace(stringField(event, key)); v != "" {
				return truncate(v, 1000)
			}
		}
		if part, ok := event["part"].(map[string]any); ok {
			for _, key := range []string{"message", "error", "text"} {
				if v := strings.TrimSpace(stringField(part, key)); v != "" {
					return truncate(v, 1000)
				}
			}
		}
		if b, err := json.Marshal(event); err == nil {
			return truncate(string(b), 1000)
		}
		return ""
	}
	return ""
}

// extractOpenCodeError pulls the meaningful failure line(s) out of opencode
// stderr. opencode's stderr typically opens with the SQLite migration prelude
// followed by the real error, so prefer the line carrying an error marker plus
// a small window of context. Ported from
// providers/opencode.py:_extract_opencode_error.
func extractOpenCodeError(stderr string) string {
	lines := strings.Split(stderr, "\n")
	for i, line := range lines {
		for _, pat := range openCodeStderrErrorPatterns {
			if pat.MatchString(line) {
				start := i - 1
				if start < 0 {
					start = 0
				}
				end := i + 5
				if end > len(lines) {
					end = len(lines)
				}
				window := strings.Join(lines[start:end], "\n")
				return truncate(strings.TrimSpace(window), 1000)
			}
		}
	}
	return truncate(stderr, 1000)
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
