package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// CodexProvider invokes the Codex CLI as a subprocess.
// It uses `codex exec --json` for structured JSONL output.
type CodexProvider struct {
	BinPath string
}

// NewCodexProvider creates a Codex provider. If binPath is empty,
// it defaults to "codex".
func NewCodexProvider(binPath string) *CodexProvider {
	if binPath == "" {
		binPath = "codex"
	}
	return &CodexProvider{BinPath: binPath}
}

func (p *CodexProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	cmd := []string{p.BinPath, "exec", "--json"}

	if options.Cwd != "" {
		cmd = append(cmd, "-C", options.Cwd)
	} else if options.ProjectDir != "" {
		cmd = append(cmd, "-C", options.ProjectDir)
	}

	if options.PermissionMode == "auto" {
		cmd = append(cmd, "--full-auto")
	}

	// Prompt is the final positional argument.
	cmd = append(cmd, prompt)

	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	cwd := options.Cwd
	if cwd == "" {
		cwd = options.ProjectDir
	}

	startAPI := time.Now()

	cliResult, err := RunCLI(ctx, cmd, env, cwd, options.timeout())
	apiMS := int(time.Since(startAPI).Milliseconds())

	if err != nil {
		if isExecNotFound(err) {
			return &RawResult{
				IsError: true,
				ErrorMessage: fmt.Sprintf(
					"Codex binary not found at '%s'. Install: https://github.com/openai/codex",
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
		raw.Result = stdout
		p.parseJSONLOutput(stdout, raw)
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
		raw.IsError = true
		raw.ErrorMessage = fmt.Sprintf("Process exited with code %d", cliResult.ReturnCode)
	}

	return raw, nil
}

// parseJSONLOutput extracts structured data from Codex's JSONL event stream.
func (p *CodexProvider) parseJSONLOutput(stdout string, raw *RawResult) {
	var messages []map[string]any
	var resultText string
	var sessionID string
	numTurns := 0

	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		messages = append(messages, event)

		eventType, _ := event["type"].(string)
		switch eventType {
		case "turn.completed":
			numTurns++
		case "thread.started":
			// Codex uses "thread_id" in thread.started events.
			if sid, ok := event["thread_id"].(string); ok {
				sessionID = sid
			}
			if sid, ok := event["session_id"].(string); ok {
				sessionID = sid
			}
		case "item.completed":
			// Extract agent message content from completed items.
			if item, ok := event["item"].(map[string]any); ok {
				if itemType, _ := item["type"].(string); itemType == "agent_message" {
					// Codex uses "text" for agent message content.
					if text, ok := item["text"].(string); ok && text != "" {
						resultText = text
					}
					if content, ok := item["content"].(string); ok && content != "" {
						resultText = content
					}
				}
			}
		case "result":
			if r, ok := event["result"].(string); ok {
				resultText = r
			}
			if sid, ok := event["session_id"].(string); ok {
				sessionID = sid
			}
			if turns, ok := event["num_turns"].(float64); ok {
				numTurns = int(turns)
			}
		}
	}

	if resultText != "" {
		raw.Result = resultText
	}
	raw.Messages = messages
	raw.Metrics.SessionID = sessionID
	raw.Metrics.NumTurns = numTurns
	if numTurns == 0 && len(messages) > 0 {
		raw.Metrics.NumTurns = len(messages)
	}
}
