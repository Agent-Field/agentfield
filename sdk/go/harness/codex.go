package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// CodexProvider invokes the Codex CLI as a subprocess.
// It uses `codex exec --json` for structured JSONL output.
type CodexProvider struct {
	BinPath string

	// runCLI is the subprocess runner, injectable for tests. It defaults to
	// RunCLIWithStdin so the prompt can be delivered via stdin (mirroring the
	// claude provider). A nil value falls back to RunCLIWithStdin at call time.
	runCLI func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int, stdin []byte) (*CLIResult, error)

	// schemaPath / outputPath are set by the runner (via SetSchema) when a JSON
	// schema is in effect. codex consumes the schema natively through
	// --output-schema and persists its final message to --output-last-message,
	// which the runner then reads — instead of relying on the Write-tool file
	// protocol used by the claude/opencode providers.
	schemaPath string
	outputPath string
}

// NewCodexProvider creates a Codex provider. If binPath is empty,
// it defaults to "codex".
func NewCodexProvider(binPath string) *CodexProvider {
	if binPath == "" {
		binPath = "codex"
	}
	return &CodexProvider{BinPath: binPath, runCLI: RunCLIWithStdin}
}

// SetSchema tells the codex provider that a JSON schema is in effect, giving it
// the deterministic paths where the runner wrote the strict schema and expects
// codex to persist its final JSON answer. The runner owns strict-schema
// computation and file writing; the provider only needs the paths so it can
// point codex's native --output-schema / --output-last-message flags at them.
//
// An empty schemaPath with a non-empty outputPath means "last-message only":
// the runner determined the schema cannot be expressed in OpenAI strict mode
// (codexSchemaStrictExpressible, schema.go), so --output-schema must not be
// sent — the server would reject it with invalid_json_schema — but codex still
// persists its final JSON answer to outputPath for local validation.
//
// This implements the schemaAware interface the runner detects (see runner.go).
func (p *CodexProvider) SetSchema(schemaPath, outputPath string) {
	p.schemaPath = schemaPath
	p.outputPath = outputPath
}

// isOutputSchemaRejection reports whether CLI output carries the server-side
// strict-schema validator's 400 for --output-schema. The error event embeds
// `"code": "invalid_json_schema"` and `Invalid schema for response_format
// 'codex_output_schema'` (both observed live on codex-cli 0.144.1).
func isOutputSchemaRejection(output string) bool {
	lower := strings.ToLower(output)
	return strings.Contains(lower, "invalid_json_schema") ||
		strings.Contains(lower, "invalid schema for response_format")
}

// withoutFlagValue returns cmd with one `flag value` pair removed.
func withoutFlagValue(cmd []string, flag string) []string {
	out := make([]string, 0, len(cmd))
	for i := 0; i < len(cmd); i++ {
		if cmd[i] == flag && i+1 < len(cmd) {
			i++
			continue
		}
		out = append(out, cmd[i])
	}
	return out
}

func (p *CodexProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	// --skip-git-repo-check lets the harness run in arbitrary working dirs
	// (temp dirs, non-repo project roots); codex exec otherwise refuses to
	// start outside a git repo.
	cmd := []string{p.BinPath, "exec", "--json", "--skip-git-repo-check"}

	cwd := options.Cwd
	if cwd == "" {
		cwd = options.ProjectDir
	}
	if cwd != "" {
		cmd = append(cmd, "-C", cwd)
	}

	// Pass the model through with -m. SWE-AF resolves gpt-5.5 vs gpt-5.3-codex by
	// auth mode and relies on the value reaching the CLI; the old provider
	// ignored options.Model entirely. Reasoning effort has no dedicated flag —
	// it's the model_reasoning_effort config key, fed by a "#variant" suffix
	// on the model (or an explicit Options.Variant), e.g. "gpt-5.3-codex#high".
	modelValue, variantValue := options.resolveModelAndVariant()
	if modelValue != "" {
		cmd = append(cmd, "-m", modelValue)
	}
	if variantValue != "" {
		cmd = append(cmd, "-c", "model_reasoning_effort="+variantValue)
	}

	// permission_mode → sandbox policy (port of codex_harness_patch.py:165-170).
	// codex exec never prompts (approval policy is always Never); the sandbox
	// controls what it may write. --full-auto is deprecated in favour of the
	// bypass flag / --sandbox.
	switch options.PermissionMode {
	case "auto":
		cmd = append(cmd, "--dangerously-bypass-approvals-and-sandbox")
	case "read-only", "workspace-write", "danger-full-access":
		cmd = append(cmd, "--sandbox", options.PermissionMode)
	default:
		cmd = append(cmd, "--sandbox", "workspace-write")
	}

	// Native structured output: when the runner has set a schema, point codex at
	// the strict schema file (patch lines 176-178). Kept independent of the
	// last-message flag below: the runner passes an empty schemaPath when the
	// schema is not strict-expressible (see SetSchema), and the reactive
	// fallback after execution needs the answer file even when the server
	// rejects the schema flag.
	usedOutputSchema := false
	if p.schemaPath != "" && fileExists(p.schemaPath) {
		cmd = append(cmd, "--output-schema", p.schemaPath)
		usedOutputSchema = true
	}
	// codex writes its final message to the last-message file, which the
	// runner reads back.
	if p.outputPath != "" {
		cmd = append(cmd, "--output-last-message", p.outputPath)
	}

	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	runCLI := p.runCLI
	if runCLI == nil {
		runCLI = RunCLIWithStdin
	}

	startAPI := time.Now()

	// The prompt is delivered on stdin, NOT as a trailing positional argument
	// (patch lines 84-102, mirroring the claude provider). codex exec reads the
	// prompt from stdin, and delivering it there keeps large prompts off the
	// argv and out of process listings.
	cliResult, err := runCLI(ctx, cmd, env, cwd, options.timeout(), []byte(prompt))

	// Reactive fallback: if the server's strict-schema validator refused the
	// schema we sent (invalid_json_schema 400 — the validator's rules can
	// tighten upstream at any time), rerun once WITHOUT --output-schema. The
	// prompt suffix still pins the JSON contract and --output-last-message
	// still captures the final answer, so the runner's local validation takes
	// over exactly as in the not-strict-expressible path.
	if err == nil && usedOutputSchema && cliResult.ReturnCode != 0 &&
		isOutputSchemaRejection(cliResult.Stdout+cliResult.Stderr) {
		retryCmd := withoutFlagValue(cmd, "--output-schema")
		if retryResult, retryErr := runCLI(ctx, retryCmd, env, cwd, options.timeout(), []byte(prompt)); retryErr == nil {
			cliResult = retryResult
		}
	}

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
		// parseJSONLOutput sets raw.Result to the extracted final message only
		// when one is present, so an empty Result below reliably signals "no
		// parseable final text" and triggers the last-message fallback.
		p.parseJSONLOutput(stdout, raw)
	}

	// Native last-message fallback: when stdout parsing yielded no usable final
	// text but codex persisted its answer to the --output-last-message file,
	// read it (patch lines 236-243).
	if raw.Result == "" && p.outputPath != "" && fileExists(p.outputPath) {
		if data, readErr := os.ReadFile(p.outputPath); readErr == nil {
			raw.Result = strings.TrimSpace(string(data))
		}
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
	tokens := extractTokenUsage(messages)
	raw.Metrics.InputTokens = tokens.inputTokens
	raw.Metrics.OutputTokens = tokens.outputTokens
	raw.Metrics.CacheReadTokens = tokens.cacheReadTokens
	raw.Metrics.CacheCreationTokens = tokens.cacheCreationTokens
}
