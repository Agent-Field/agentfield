package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// ProviderCodeAF is the provider name for the CodeAF CLI.
	ProviderCodeAF = "codeaf"
	// DefaultCodeAFModel pins the worker instead of allowing CodeAF's router to choose it.
	DefaultCodeAFModel         = "~deepseek/deepseek-v4-flash-latest"
	defaultCodeAFMaxConcurrent = 8
)

var (
	codeAFSemaphore chan struct{}
	codeAFSemOnce   sync.Once
)

// CodeAFProvider invokes CodeAF's canonical headless exec or do protocol.
// Binary discovery and managed installation happen lazily on first execution.
// When CodeAF is unavailable it can delegate to the unchanged legacy adapter.
type CodeAFProvider struct {
	BinPath       string
	alias         bool
	runCLI        func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idleSeconds int, stdin []byte) (*CLIResult, error)
	resolveBinary func(context.Context, string) (codeAFBinary, error)
}

// NewCodeAFProvider creates a CodeAF provider without resolving or downloading a binary.
func NewCodeAFProvider(binPath string) *CodeAFProvider {
	return &CodeAFProvider{BinPath: binPath, runCLI: runCLIWithStdinIdle, resolveBinary: resolveCodeAF}
}

func codeAFMaxConcurrent() int {
	raw, set := os.LookupEnv("CODEAF_MAX_CONCURRENT")
	if !set {
		raw = os.Getenv("AFORGE_MAX_CONCURRENT")
	}
	if parsed, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && parsed > 0 {
		return parsed
	}
	return defaultCodeAFMaxConcurrent
}

func getCodeAFSemaphore() chan struct{} {
	codeAFSemOnce.Do(func() {
		codeAFSemaphore = make(chan struct{}, codeAFMaxConcurrent())
	})
	return codeAFSemaphore
}

func codeAFCommand() string {
	command := strings.TrimSpace(os.Getenv("AGENTFIELD_CODEAF_COMMAND"))
	if command == "do" || command == "exec" {
		return command
	}
	if command != "" {
		codeAFCommandOnce.Do(func() {
			log.Printf("WARNING AGENTFIELD_CODEAF_COMMAND must be exec or do; using exec")
		})
	}
	return "exec"
}

func codeAFEnvironment(options Options, variant string) (map[string]string, error) {
	env := map[string]string{"CODEAF_MODELS": "", "AFORGE_MODELS": ""}
	present := func(key string) bool {
		if _, ok := options.Env[key]; ok {
			return true
		}
		_, ok := os.LookupEnv(key)
		return ok
	}
	value := func(key string) string {
		if v, ok := options.Env[key]; ok {
			return v
		}
		return os.Getenv(key)
	}
	if !present("CODEAF_HOME") {
		home := value("AGENTFIELD_HOME")
		if home == "" {
			var err error
			home, err = agentFieldHome()
			if err != nil {
				return nil, err
			}
		}
		env["CODEAF_HOME"] = filepath.Join(home, "codeaf")
	}
	if !present("CODEAF_NO_UPDATE_CHECK") {
		env["CODEAF_NO_UPDATE_CHECK"] = "1"
	}
	telemetryDisabled := false
	switch strings.ToLower(strings.TrimSpace(value("AGENTFIELD_TELEMETRY_ENABLED"))) {
	case "0", "false", "no", "off":
		telemetryDisabled = true
	}
	if !present("CODEAF_TELEMETRY") && (codeAFTruthy(value("CI")) || codeAFTruthy(value("DO_NOT_TRACK")) || telemetryDisabled) {
		env["CODEAF_TELEMETRY"] = "off"
	}
	if normalized, ok := supportedAforgeVariant(variant); ok {
		env["CODEAF_EXEC_REASONING"] = normalized
	}
	for key, v := range options.Env {
		env[key] = v
	}
	home := env["CODEAF_HOME"]
	if home == "" {
		home = value("CODEAF_HOME")
	}
	if home != "" {
		if err := os.MkdirAll(home, 0o755); err != nil {
			return nil, fmt.Errorf("create CODEAF_HOME: %w", err)
		}
	}
	return env, nil
}

func parseCodeAFEnvelope(stdout string) map[string]any {
	parse := func(text string) map[string]any {
		var envelope map[string]any
		if json.Unmarshal([]byte(strings.TrimSpace(text)), &envelope) == nil {
			_, hasOK := envelope["ok"]
			_, hasStop := envelope["stop"]
			if hasOK && hasStop {
				return envelope
			}
		}
		return nil
	}
	if envelope := parse(stdout); envelope != nil {
		return envelope
	}
	lines := strings.Split(stdout, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if envelope := parse(lines[i]); envelope != nil {
			return envelope
		}
	}
	return nil
}

func codeAFErrorMessage(code int, stop, detail, blockedOn, stderr string) string {
	if detail == "" {
		detail = strings.TrimSpace(StripANSI(stderr))
		if len(detail) > 1000 {
			detail = detail[len(detail)-1000:]
		}
	}
	if blockedOn != "" {
		blocked := "; blocked_on: " + truncate(blockedOn, 400)
		detail = truncate(detail, 1000-len(blocked)) + blocked
	}
	return fmt.Sprintf("codeaf exit code %d (stop=%s): %s", code, stop, truncate(detail, 1000))
}

func (p *CodeAFProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	override := p.BinPath
	if options.BinPath != "" {
		override = options.BinPath
	}
	binary, err := p.resolveBinary(ctx, override)
	if err != nil {
		return nil, err
	}
	if binary.legacy {
		codeAFLegacyOnce.Do(func() {
			log.Printf("WARNING CodeAF is not installed; falling back to the legacy aforge binary at %s. Run `af codeaf ensure`.", binary.path)
		})
		legacy := NewAforgeProvider(binary.path)
		legacy.runCLI = p.runCLI
		raw, err := legacy.Execute(ctx, prompt, options)
		if raw != nil {
			raw.Provider = ProviderAforge
		}
		return raw, err
	}
	if p.alias {
		codeAFAliasOnce.Do(func() {
			log.Printf("INFO provider 'aforge' now runs CodeAF; set AFORGE_BIN to pin the legacy binary")
		})
	}
	sem := getCodeAFSemaphore()
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	root := firstNonEmpty(options.ProjectDir, options.Cwd, ".")
	outerTimeout := aforgeTimeout(options)
	model, variant := options.resolveModelAndVariant()
	model = stripOpenRouterPrefix(model)
	if model == "" {
		model = DefaultCodeAFModel
	}
	command := codeAFCommand()
	cmd := []string{binary.path, command, "--json", "--dir", root}
	if command == "do" {
		cmd = append(cmd, "--yes-spend", "--model", model, "--plan-model", model, "--check-model", model)
	} else {
		cmd = append(cmd, "--model", model)
	}
	cmd = append(cmd, "--timeout", strconv.Itoa(aforgeInnerTimeout(outerTimeout))+"s")
	input := []byte(prompt)
	if command == "exec" {
		if options.MaxTurns > 0 {
			cmd = append(cmd, "--max-turns", strconv.Itoa(options.MaxTurns))
		}
		if system := strings.TrimSpace(options.SystemPrompt); system != "" {
			cmd = append(cmd, "--system", system)
		}
	} else {
		cmd = append(cmd, "-")
		input = []byte(aforgeTaskInput(prompt, options.SystemPrompt))
	}
	env, err := codeAFEnvironment(options, variant)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(outerTimeout)*time.Second)
	defer cancel()
	cli, err := p.runCLI(runCtx, cmd, env, "", outerTimeout, 0, input)
	elapsed := int(time.Since(started).Milliseconds())
	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return &RawResult{Provider: ProviderCodeAF, IsError: true, FailureType: FailureTimeout,
			ErrorMessage: fmt.Sprintf("CodeAF timed out after %ds", outerTimeout),
			Metrics:      Metrics{DurationMS: elapsed, DurationAPIMS: elapsed, Model: model}}, nil
	}
	if err != nil {
		if isExecNotFound(err) {
			return nil, newProviderUnavailable(ProviderCodeAF, binary.path)
		}
		if errors.Is(err, context.DeadlineExceeded) || strings.Contains(strings.ToLower(err.Error()), "timed out") || strings.Contains(strings.ToLower(err.Error()), "deadline exceeded") {
			return &RawResult{Provider: ProviderCodeAF, IsError: true, FailureType: FailureTimeout,
				ErrorMessage: err.Error(), Metrics: Metrics{DurationMS: elapsed, DurationAPIMS: elapsed, Model: model}}, nil
		}
		return nil, err
	}
	return codeAFResult(cli, model, elapsed), nil
}

func codeAFResult(cli *CLIResult, model string, elapsed int) *RawResult {
	envelope := parseCodeAFEnvelope(cli.Stdout)
	text := func(key string) string {
		value, _ := envelope[key].(string)
		return strings.TrimSpace(value)
	}
	answer, stop := text("answer"), text("stop")
	metrics := Metrics{DurationMS: elapsed, DurationAPIMS: elapsed, Model: firstNonEmpty(text("model"), model)}
	if steps, ok := aforgeNumber(envelope["steps"]); ok {
		metrics.NumTurns = int(steps)
	}
	if tokens, ok := envelope["tokens"].(map[string]any); ok {
		in, _ := aforgeNumber(tokens["in"])
		out, _ := aforgeNumber(tokens["out"])
		metrics.InputTokens, metrics.OutputTokens = int(in), int(out)
	}
	if spend, ok := aforgeNumber(envelope["spend_usd"]); ok {
		metrics.CostUSD = &spend
	}
	raw := &RawResult{Provider: ProviderCodeAF, Metrics: metrics, ReturnCode: cli.ReturnCode, FailureType: FailureNone}
	if envelope != nil {
		raw.Messages = []map[string]any{envelope}
	}
	ok, _ := envelope["ok"].(bool)
	switch {
	case cli.ReturnCode < 0:
		raw.FailureType = FailureCrash
	case envelope == nil:
		if cli.ReturnCode == 0 && strings.TrimSpace(cli.Stdout) == "" {
			raw.FailureType = FailureNoOutput
		} else {
			raw.FailureType = FailureCrash
		}
	case stop == "deadline":
		raw.FailureType = FailureTimeout
	case stop == "done" && ok:
		raw.Result = answer
	case (stop == "unchecked" || stop == "budget" || stop == "turn-cap") && answer != "":
		log.Printf("WARNING CodeAF returned a partial or unchecked answer (stop=%s)", stop)
		raw.Result = answer
	case stop == "budget" || stop == "turn-cap":
		raw.FailureType = FailureNoOutput
	case stop == "question" || stop == "incomplete" || stop == "error" || stop == "price" || stop == "done" || stop == "unchecked":
		raw.FailureType = FailureCrash
	case cli.ReturnCode == 0 && answer != "":
		raw.Result = answer
	default:
		raw.FailureType = FailureCrash
	}
	raw.IsError = raw.FailureType != FailureNone
	if raw.IsError {
		blockedOn := ""
		if stop == "question" {
			blockedOn = text("blocked_on")
		}
		raw.ErrorMessage = codeAFErrorMessage(cli.ReturnCode, stop, firstNonEmpty(text("error"), answer), blockedOn, cli.Stderr)
	}
	return raw
}
