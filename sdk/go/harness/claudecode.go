package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ClaudeCodeProvider invokes the Claude Code CLI as a subprocess.
// It uses the `claude` CLI with `--print` mode for non-interactive output.
type ClaudeCodeProvider struct {
	BinPath string

	// runCLI is the subprocess runner, injectable for tests. It defaults to
	// RunCLIWithStdin so the prompt can be delivered via stdin.
	runCLI func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, stdin []byte) (*CLIResult, error)
}

// NewClaudeCodeProvider creates a Claude Code provider. If binPath is empty,
// it defaults to "claude".
func NewClaudeCodeProvider(binPath string) *ClaudeCodeProvider {
	if binPath == "" {
		binPath = "claude"
	}
	return &ClaudeCodeProvider{BinPath: binPath, runCLI: RunCLIWithStdin}
}

// permissionMap translates common permission mode names to Claude Code flags.
var permissionMap = map[string]string{
	"auto": "bypassPermissions",
	"plan": "plan",
}

func (p *ClaudeCodeProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	// stream-json (rather than plain json) keeps RunCLI's idle watchdog fed:
	// with --output-format json the CLI is COMPLETELY SILENT until the run
	// finishes, so any turn quieter than AGENTFIELD_HARNESS_IDLE_SECONDS
	// (default 120s) was SIGKILLed mid-run. stream-json emits per-message
	// JSONL events (init, thinking deltas, assistant messages, final result)
	// as they happen — verified against claude CLI 2.1.191 — and the final
	// "result" event carries the same fields as the json format (result,
	// session_id, num_turns, total_cost_usd), so parseJSONOutput's contract
	// is unchanged. In --print mode the CLI hard-requires --verbose alongside
	// stream-json ("--output-format=stream-json requires --verbose").
	cmd := []string{p.BinPath, "--print", "--output-format", "stream-json", "--verbose"}

	if options.Model != "" {
		cmd = append(cmd, "--model", options.Model)
	}

	if options.MaxTurns > 0 {
		cmd = append(cmd, "--max-turns", fmt.Sprintf("%d", options.MaxTurns))
	}

	if options.PermissionMode != "" {
		mode := options.PermissionMode
		if mapped, ok := permissionMap[mode]; ok {
			mode = mapped
		}
		cmd = append(cmd, "--permission-mode", mode)
	}

	if options.SystemPrompt != "" {
		cmd = append(cmd, "--system-prompt", options.SystemPrompt)
	}

	if options.ResumeSessionID != "" {
		cmd = append(cmd, "--resume", options.ResumeSessionID)
	}

	if options.MaxBudgetUSD > 0 {
		cmd = append(cmd, "--max-budget-usd", fmt.Sprintf("%.4f", options.MaxBudgetUSD))
	}

	for _, tool := range options.Tools {
		cmd = append(cmd, "--allowedTools", tool)
	}

	// The prompt is delivered on stdin, NOT as a trailing positional argument.
	// `--allowedTools` is variadic in the claude CLI (verified against 2.1.191):
	// a positional prompt immediately following it is greedily absorbed into the
	// tool list, so `--print` sees no prompt and exits non-zero with
	// "Input must be provided ... when using --print". `claude --print` reads
	// the prompt from stdin when no positional is present.

	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	// Unset CLAUDECODE to allow spawning Claude Code from within a Claude
	// Code session (the CLI refuses to start when this var is present).
	env["CLAUDECODE"] = ""

	cwd := options.Cwd
	if cwd == "" {
		cwd = options.ProjectDir
	}

	runCLI := p.runCLI
	if runCLI == nil {
		runCLI = RunCLIWithStdin
	}

	startAPI := time.Now()

	cliResult, err := runCLI(ctx, cmd, env, cwd, options.timeout(), []byte(prompt))
	apiMS := int(time.Since(startAPI).Milliseconds())

	if err != nil {
		if isExecNotFound(err) {
			return &RawResult{
				IsError: true,
				ErrorMessage: fmt.Sprintf(
					"Claude Code binary not found at '%s'. Install: npm install -g @anthropic-ai/claude-code",
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

	// Parse the JSONL event stream from Claude Code's --output-format stream-json
	raw := &RawResult{
		Metrics: Metrics{
			DurationAPIMS: apiMS,
		},
		ReturnCode: cliResult.ReturnCode,
	}

	stdout := strings.TrimSpace(cliResult.Stdout)
	cleanStderr := StripANSI(strings.TrimSpace(cliResult.Stderr))

	if stdout != "" {
		raw.Result = stdout
		// Try to parse the JSON output for structured fields
		p.parseJSONOutput(stdout, raw)
	}

	if cliResult.ReturnCode != 0 && raw.Result == "" {
		raw.IsError = true
		raw.FailureType = FailureCrash
		if cleanStderr != "" {
			raw.ErrorMessage = truncate(cleanStderr, 1000)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d and produced no output.",
				cliResult.ReturnCode)
		}
	} else if cliResult.ReturnCode != 0 {
		// Non-zero exit but we got output — note the error but don't mark as fatal
		raw.IsError = true
		raw.ErrorMessage = fmt.Sprintf("Process exited with code %d", cliResult.ReturnCode)
	}

	return raw, nil
}

// parseJSONOutput extracts structured data from Claude Code's JSONL output.
// It consumes both the stream-json event stream (system/init, thinking
// deltas, assistant messages, terminal "result" event) and the legacy
// single-line json format: every line is parsed as an event, all events are
// collected into Messages (mirroring the Python provider, which appends every
// SDK stream message), and the LAST "result" event supplies Result,
// SessionID, CostUSD and NumTurns — its fields are identical across both
// output formats. Assistant-message text is only a fallback when no result
// event ever arrives (e.g. a stream truncated by a crash).
func (p *ClaudeCodeProvider) parseJSONOutput(stdout string, raw *RawResult) {
	var messages []map[string]any
	var resultText string
	var sessionID string
	var cost *float64
	numTurns := 0

	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var msg map[string]any
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			continue
		}
		messages = append(messages, msg)

		msgType, _ := msg["type"].(string)
		switch msgType {
		case "result":
			if r, ok := msg["result"].(string); ok {
				resultText = r
			} else if r, ok := msg["text"].(string); ok {
				resultText = r
			}
			if sid, ok := msg["session_id"].(string); ok {
				sessionID = sid
			}
			if c := extractCost(msg); c != nil {
				cost = c
			}
			if turns, ok := msg["num_turns"].(float64); ok {
				numTurns = int(turns)
			}
		case "assistant":
			if resultText == "" {
				resultText = extractAssistantText(msg)
			}
		}
	}

	if resultText != "" {
		raw.Result = resultText
	}
	raw.Messages = messages
	raw.Metrics.SessionID = sessionID
	raw.Metrics.CostUSD = cost
	raw.Metrics.NumTurns = numTurns
	if numTurns == 0 && len(messages) > 0 {
		raw.Metrics.NumTurns = len(messages)
	}
}

// extractCost pulls the per-call cost from a Claude Code result message.
//
// Mirrors the Python provider's semantics exactly
// (agentfield/harness/providers/claude.py):
//
//	cost_info = msg.get("cost_usd") or msg.get("total_cost_usd")
//	if cost_info is not None:
//	    total_cost = float(cost_info)
//
// The Python `or` treats a zero/absent "cost_usd" as falsy and falls through
// to "total_cost_usd". Returns nil when neither yields a usable number, so the
// caller can distinguish "unknown cost" (nil) from "$0.00".
func extractCost(msg map[string]any) *float64 {
	if v, ok := msg["cost_usd"].(float64); ok && v != 0 {
		c := v
		return &c
	}
	if v, ok := msg["total_cost_usd"].(float64); ok {
		c := v
		return &c
	}
	return nil
}

// extractAssistantText pulls text content from an assistant message.
func extractAssistantText(msg map[string]any) string {
	// Direct content field
	if content, ok := msg["content"].(string); ok && content != "" {
		return content
	}

	// Nested message.content
	if message, ok := msg["message"].(map[string]any); ok {
		if content, ok := message["content"].(string); ok && content != "" {
			return content
		}
	}

	// Content as array of blocks
	var contentBlocks []any
	if blocks, ok := msg["content"].([]any); ok {
		contentBlocks = blocks
	} else if message, ok := msg["message"].(map[string]any); ok {
		if blocks, ok := message["content"].([]any); ok {
			contentBlocks = blocks
		}
	}

	for _, block := range contentBlocks {
		if b, ok := block.(map[string]any); ok {
			if bType, _ := b["type"].(string); bType == "text" {
				if text, ok := b["text"].(string); ok && text != "" {
					return text
				}
			}
		}
	}

	return ""
}
