package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type piFlavor string

const (
	piFlavorPi  piFlavor = "pi"
	piFlavorOMP piFlavor = "omp"
)

var piReadOnlyTools = map[string]bool{
	"read": true, "grep": true, "find": true, "glob": true, "ls": true, "lsp": true,
}

type piFamilyProvider struct {
	BinPath string
	flavor  piFlavor
	runCLI  func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error)
}

// PiProvider invokes the Pi coding-agent CLI as a subprocess.
type PiProvider struct{ *piFamilyProvider }

// OMPProvider invokes the Oh My Pi (OMP) coding-agent CLI as a subprocess.
type OMPProvider struct{ *piFamilyProvider }

// NewPiProvider creates a Pi provider. An empty binPath defaults to "pi".
func NewPiProvider(binPath string) *PiProvider {
	if binPath == "" {
		binPath = "pi"
	}
	return &PiProvider{&piFamilyProvider{
		BinPath: binPath,
		flavor:  piFlavorPi,
		runCLI:  RunCLIWithStdin,
	}}
}

// NewOMPProvider creates an OMP provider. An empty binPath defaults to "omp".
func NewOMPProvider(binPath string) *OMPProvider {
	if binPath == "" {
		binPath = "omp"
	}
	return &OMPProvider{&piFamilyProvider{
		BinPath: binPath,
		flavor:  piFlavorOMP,
		runCLI:  RunCLIWithStdin,
	}}
}

func (p *PiProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	return p.piFamilyProvider.execute(ctx, prompt, options)
}

func (p *OMPProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	return p.piFamilyProvider.execute(ctx, prompt, options)
}

func (p *piFamilyProvider) execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	cmd := []string{p.BinPath, "--print", "--mode", "json"}

	root := options.ProjectDir
	if root == "" {
		root = options.Cwd
	}
	if p.flavor == piFlavorOMP && root != "" {
		cmd = append(cmd, "--cwd", root)
	}

	model, variant := options.resolveModelAndVariant()
	if model != "" {
		cmd = append(cmd, "--model", model)
	}
	if variant != "" {
		cmd = append(cmd, "--thinking", variant)
	}
	if strings.TrimSpace(options.SystemPrompt) != "" {
		cmd = append(cmd, "--system-prompt", strings.TrimSpace(options.SystemPrompt))
	}
	if options.ResumeSessionID != "" {
		resumeFlag := "--session"
		if p.flavor == piFlavorOMP {
			resumeFlag = "--resume"
		}
		cmd = append(cmd, resumeFlag, options.ResumeSessionID)
	}
	// --tools is the enforced, vendor-documented read-only allowlist. Pi has no
	// approval flag (unknown options fail); OMP auto-approves read-only tiers even
	// under always-ask.
	if options.PermissionMode == "auto" {
		if p.flavor == piFlavorOMP {
			cmd = append(cmd, "--auto-approve")
		}
	}

	tools := normalizePiTools(options.Tools, p.flavor)
	if options.PermissionMode == "plan" {
		readOnly := make([]string, 0, len(tools))
		for _, tool := range tools {
			if piReadOnlyTools[tool] {
				readOnly = append(readOnly, tool)
			}
		}
		tools = readOnly
		if len(tools) == 0 {
			globTool := "find"
			if p.flavor == piFlavorOMP {
				globTool = "glob"
			}
			tools = []string{"read", "grep", globTool}
		}
	}
	if options.Tools != nil || options.PermissionMode == "plan" {
		if len(tools) == 0 {
			cmd = append(cmd, "--no-tools")
		} else {
			cmd = append(cmd, "--tools", strings.Join(tools, ","))
		}
	}

	env := make(map[string]string, len(options.Env))
	for key, value := range options.Env {
		env[key] = value
	}

	start := time.Now()
	runCLI := p.runCLI
	if runCLI == nil {
		runCLI = RunCLIWithStdin
	}
	cliResult, err := runCLI(ctx, cmd, env, root, options.timeout(), []byte(prompt))
	apiMS := int(time.Since(start).Milliseconds())
	if err != nil {
		if isExecNotFound(err) {
			install := "npm install -g --ignore-scripts @earendil-works/pi-coding-agent"
			if p.flavor == piFlavorOMP {
				install = "curl -fsSL https://omp.sh/install | sh"
			}
			return &RawResult{
				IsError:      true,
				ErrorMessage: fmt.Sprintf("%s binary not found at '%s'. Install: %s", strings.ToUpper(string(p.flavor)), p.BinPath, install),
				FailureType:  FailureCrash,
			}, nil
		}
		if strings.Contains(err.Error(), "timed out") || strings.Contains(err.Error(), "no progress") {
			return &RawResult{
				IsError:      true,
				ErrorMessage: err.Error(),
				FailureType:  FailureTimeout,
				Metrics:      Metrics{DurationAPIMS: apiMS},
			}, nil
		}
		return nil, err
	}

	raw := parsePiJSONL(cliResult.Stdout)
	if model != "" {
		raw.Metrics.Model = model
	}
	raw.Metrics.DurationAPIMS = apiMS
	raw.ReturnCode = cliResult.ReturnCode
	stderr := StripANSI(strings.TrimSpace(cliResult.Stderr))
	if cliResult.ReturnCode < 0 {
		raw.IsError = true
		raw.FailureType = FailureCrash
		raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d.", -cliResult.ReturnCode)
	} else if cliResult.ReturnCode != 0 {
		raw.IsError = true
		raw.FailureType = FailureCrash
		if stderr != "" {
			raw.ErrorMessage = truncate(stderr, 1000)
		} else if raw.ErrorMessage == "" {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d.", cliResult.ReturnCode)
		}
	} else if raw.ErrorMessage != "" {
		raw.IsError = true
		raw.FailureType = FailureAPIError
	} else if raw.Result == "" {
		raw.IsError = true
		raw.FailureType = FailureNoOutput
		raw.ErrorMessage = stderr
		if raw.ErrorMessage == "" {
			raw.ErrorMessage = fmt.Sprintf("%s exited successfully without an assistant response.", p.flavor)
		}
	}
	return raw, nil
}

func normalizePiTools(tools []string, flavor piFlavor) []string {
	normalized := make([]string, 0, len(tools))
	seen := make(map[string]bool, len(tools))
	for _, tool := range tools {
		name := strings.ToLower(strings.TrimSpace(tool))
		if name == "glob" && flavor == piFlavorPi {
			name = "find"
		}
		if name != "" && !seen[name] {
			seen[name] = true
			normalized = append(normalized, name)
		}
	}
	return normalized
}

func parsePiJSONL(stdout string) *RawResult {
	raw := &RawResult{FailureType: FailureNone}
	var totalCost float64
	hasCost := false

	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		raw.Messages = append(raw.Messages, event)

		eventType, _ := event["type"].(string)
		if eventType == "session" {
			if id, ok := event["id"].(string); ok {
				raw.Metrics.SessionID = id
			}
		}
		if eventType == "turn_end" {
			raw.Metrics.NumTurns++
		}
		if eventType != "message_end" {
			continue
		}
		message, ok := event["message"].(map[string]any)
		if !ok || message["role"] != "assistant" {
			continue
		}
		if model, ok := message["model"].(string); ok {
			raw.Metrics.Model = model
		}
		if text := piMessageText(message); text != "" {
			raw.Result = text
		}
		if usage, ok := message["usage"].(map[string]any); ok {
			raw.Metrics.InputTokens += intValue(usage["input"])
			raw.Metrics.OutputTokens += intValue(usage["output"])
			raw.Metrics.CacheReadTokens += intValue(usage["cacheRead"])
			raw.Metrics.CacheCreationTokens += intValue(usage["cacheWrite"])
			if cost, ok := usage["cost"].(map[string]any); ok {
				if value, ok := floatValue(cost["total"]); ok {
					totalCost += value
					hasCost = true
				}
			}
		}
		if reason, _ := message["stopReason"].(string); reason == "error" || reason == "aborted" {
			raw.ErrorMessage = fmt.Sprintf("Pi stopped with reason %q.", reason)
			if detail, ok := message["errorMessage"].(string); ok && detail != "" {
				raw.ErrorMessage = detail
			}
		} else {
			// Only the final message_end's stop reason decides: a turn that
			// errored and then recovered must not be reported as a failure.
			raw.ErrorMessage = ""
		}
	}

	if raw.Metrics.NumTurns == 0 && raw.Result != "" {
		raw.Metrics.NumTurns = 1
	}
	if hasCost {
		raw.Metrics.CostUSD = &totalCost
	}
	return raw
}

func piMessageText(message map[string]any) string {
	if content, ok := message["content"].(string); ok {
		return content
	}
	content, ok := message["content"].([]any)
	if !ok {
		return ""
	}
	var text strings.Builder
	for _, item := range content {
		part, ok := item.(map[string]any)
		if !ok || part["type"] != "text" {
			continue
		}
		if value, ok := part["text"].(string); ok {
			text.WriteString(value)
		}
	}
	return text.String()
}

func intValue(value any) int {
	if number, ok := floatValue(value); ok {
		return int(number)
	}
	return 0
}

func floatValue(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}
