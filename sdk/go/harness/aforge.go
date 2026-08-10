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

const (
	defaultAforgeMaxConcurrent = 8
	defaultAforgeTimeout       = 1800
	aforgeLandingWindow        = 5
)

var (
	aforgeSemaphore chan struct{}
	aforgeSemOnce   sync.Once
)

// AforgeProvider invokes aforge's one-shot machine-readable harness mode.
type AforgeProvider struct {
	BinPath string
	runCLI  func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idleSeconds int, stdin []byte) (*CLIResult, error)
}

// NewAforgeProvider creates an Aforge provider. If binPath is empty, it
// defaults to "aforge".
func NewAforgeProvider(binPath string) *AforgeProvider {
	if binPath == "" {
		binPath = strings.TrimSpace(os.Getenv("AFORGE_BIN"))
		if binPath == "" {
			binPath = "aforge"
		}
	}
	return &AforgeProvider{BinPath: binPath, runCLI: runCLIWithStdinIdle}
}

func getAforgeSemaphore() chan struct{} {
	aforgeSemOnce.Do(func() {
		maxConcurrent := defaultAforgeMaxConcurrent
		if raw := strings.TrimSpace(os.Getenv("AFORGE_MAX_CONCURRENT")); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
				maxConcurrent = parsed
			}
		}
		aforgeSemaphore = make(chan struct{}, maxConcurrent)
	})
	return aforgeSemaphore
}

func aforgeTimeout(options Options) int {
	if options.Timeout > 0 {
		return options.Timeout
	}
	if raw := strings.TrimSpace(os.Getenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultAforgeTimeout
}

func aforgeInnerTimeout(outer int) int {
	if outer > aforgeLandingWindow {
		return outer - aforgeLandingWindow
	}
	return 1
}

func stripOpenRouterPrefix(model string) string {
	return strings.TrimPrefix(model, "openrouter/")
}

func supportedAforgeVariant(variant string) (string, bool) {
	normalized := strings.ToLower(strings.TrimSpace(variant))
	switch normalized {
	case "off", "low", "medium", "high":
		return normalized, true
	default:
		return "", false
	}
}

func aforgeTaskInput(prompt, systemPrompt string) string {
	if systemPrompt = strings.TrimSpace(systemPrompt); systemPrompt != "" {
		return systemPrompt + "\n\nTask:\n" + prompt
	}
	return prompt
}

func parseAforgeEnvelope(stdout string) map[string]any {
	lines := strings.Split(stdout, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		var envelope map[string]any
		if err := json.Unmarshal([]byte(line), &envelope); err != nil {
			continue
		}
		if _, ok := envelope["deliverable"]; ok {
			return envelope
		}
	}
	return nil
}

func aforgeNumber(value any) (float64, bool) {
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

func aforgeCrashMessage(returnCode int, blockedOn, deliverable, stderr string) string {
	cleanStderr := StripANSI(strings.TrimSpace(stderr))
	exitContext := fmt.Sprintf("aforge exit code %d", returnCode)
	message := exitContext
	if returnCode < 0 {
		message = fmt.Sprintf("Process killed by signal %d. %s", -returnCode, exitContext)
	}
	switch {
	case cleanStderr != "":
		message += ". stderr: " + truncate(cleanStderr, 1000)
	case blockedOn != "":
		message += ". blocked_on: " + truncate(blockedOn, 1000)
	case deliverable != "":
		message += ". partial: " + truncate(deliverable, 1000)
	}
	return message
}

func (p *AforgeProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	sem := getAforgeSemaphore()
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	root := options.ProjectDir
	if root == "" {
		root = options.Cwd
	}
	if root == "" {
		root = "."
	}
	outerTimeout := aforgeTimeout(options)
	cmd := []string{
		p.BinPath, "do", "--json", "--yes-spend", "-w", root,
		"--timeout", strconv.Itoa(aforgeInnerTimeout(outerTimeout)),
	}

	model, variant := options.resolveModelAndVariant()
	env := make(map[string]string)
	if model != "" {
		env["AFORGE_MODEL"] = stripOpenRouterPrefix(model)
	}
	if normalized, ok := supportedAforgeVariant(variant); ok {
		env["AFORGE_EXEC_REASONING"] = normalized
	}
	// Caller-supplied environment wins over values derived from model/variant.
	for key, value := range options.Env {
		env[key] = value
	}

	started := time.Now()
	cliResult, err := p.runCLI(ctx, cmd, env, "", outerTimeout, 0, []byte(aforgeTaskInput(prompt, options.SystemPrompt)))
	apiMS := int(time.Since(started).Milliseconds())
	if err != nil {
		if isExecNotFound(err) {
			return &RawResult{
				IsError:      true,
				ErrorMessage: fmt.Sprintf("Aforge binary not found at '%s'. Build it from https://github.com/Agent-Field/aforge-v2", p.BinPath),
				FailureType:  FailureCrash,
				Metrics:      Metrics{DurationAPIMS: apiMS},
			}, nil
		}
		if strings.Contains(strings.ToLower(err.Error()), "timed out") || strings.Contains(strings.ToLower(err.Error()), "deadline exceeded") {
			return &RawResult{
				IsError:      true,
				ErrorMessage: err.Error(),
				FailureType:  FailureTimeout,
				Metrics:      Metrics{DurationAPIMS: apiMS},
			}, nil
		}
		return nil, err
	}

	envelope := parseAforgeEnvelope(cliResult.Stdout)
	resultText := ""
	blockedOn := ""
	usage := map[string]any{}
	if envelope != nil {
		if text, ok := envelope["deliverable"].(string); ok {
			resultText = strings.TrimSpace(text)
		}
		if value, ok := envelope["blocked_on"].(string); ok {
			blockedOn = strings.TrimSpace(value)
		}
		if value, ok := envelope["usage"].(map[string]any); ok {
			usage = value
		}
	}

	isError := cliResult.ReturnCode != 0 || resultText == "" || blockedOn != ""
	metrics := Metrics{DurationAPIMS: apiMS}
	if value, ok := aforgeNumber(usage["calls"]); ok {
		metrics.NumTurns = int(value)
	}
	if value, ok := aforgeNumber(usage["prompt_tokens"]); ok {
		metrics.InputTokens = int(value)
	}
	if value, ok := aforgeNumber(usage["completion_tokens"]); ok {
		metrics.OutputTokens = int(value)
	}
	if value, ok := aforgeNumber(usage["cached_tokens"]); ok {
		metrics.CacheReadTokens = int(value)
	}
	if value, ok := aforgeNumber(usage["cost"]); ok && value > 0 {
		cost := value
		metrics.CostUSD = &cost
	}

	messages := []map[string]any(nil)
	if envelope != nil {
		messages = []map[string]any{envelope}
	}
	raw := &RawResult{
		Result:      resultText,
		Messages:    messages,
		Metrics:     metrics,
		IsError:     isError,
		FailureType: FailureNone,
		ReturnCode:  cliResult.ReturnCode,
	}
	if isError {
		if cliResult.ReturnCode == 2 {
			raw.FailureType = FailureTimeout
		} else {
			raw.FailureType = FailureCrash
		}
		raw.ErrorMessage = aforgeCrashMessage(cliResult.ReturnCode, blockedOn, resultText, cliResult.Stderr)
	}
	return raw, nil
}
