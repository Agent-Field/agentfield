package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
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

const (
	openCodeConfigSchema          = "https://opencode.ai/config.json"
	openCodeAgentName             = "agentfield-harness"
	openCodeDefaultSteps          = 500
	openCodeInlineSystemPromptEnv = "AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT"
	openCodeStepsEnv              = "AGENTFIELD_OPENCODE_STEPS"
	agentFieldWorkerInstruction   = "You are an AgentField-launched worker. Complete the assigned prompt directly. " +
		"Do not invoke AgentField orchestration, the `af` CLI, `swe-planner.plan`, " +
		"or delegate work back to AgentField."
)

type orderedJSONObject struct {
	keys   []string
	values map[string]any
}

func (o orderedJSONObject) MarshalJSON() ([]byte, error) {
	encoded := []byte{'{'}
	for index, key := range o.keys {
		if index > 0 {
			encoded = append(encoded, ',')
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		encodedValue, err := json.Marshal(o.values[key])
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, encodedKey...)
		encoded = append(encoded, ':')
		encoded = append(encoded, encodedValue...)
	}
	return append(encoded, '}'), nil
}

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

func openCodePermissions() map[string]any {
	// Harness tools and permission mode intentionally do not change this
	// baseline. With the wildcard allow present, translating a tool allowlist
	// would only add redundant allow-on-top-of-allow rules.
	return map[string]any{
		"*":        "allow",
		"skill":    map[string]any{"agentfield*": "deny"},
		"question": "deny",
		"task":     "deny",
	}
}

func orderedOpenCodePermissions(permission map[string]any) orderedJSONObject {
	keys := make([]string, 0, len(permission))
	values := make(map[string]any, len(permission))
	if value, ok := permission["*"]; ok {
		keys = append(keys, "*")
		values["*"] = value
	}

	otherKeys := make([]string, 0, len(permission))
	for key := range permission {
		switch key {
		case "*", "skill", "question", "task":
			continue
		default:
			otherKeys = append(otherKeys, key)
		}
	}
	sort.Strings(otherKeys)
	for _, key := range otherKeys {
		keys = append(keys, key)
		values[key] = permission[key]
	}

	if skill, ok := permission["skill"]; ok {
		keys = append(keys, "skill")
		if skillObject, ok := skill.(map[string]any); ok {
			skillKeys := make([]string, 0, len(skillObject))
			skillValues := make(map[string]any, len(skillObject))
			for key, value := range skillObject {
				if key != "agentfield*" {
					skillKeys = append(skillKeys, key)
					skillValues[key] = value
				}
			}
			sort.Strings(skillKeys)
			if value, ok := skillObject["agentfield*"]; ok {
				skillKeys = append(skillKeys, "agentfield*")
				skillValues["agentfield*"] = value
			}
			values["skill"] = orderedJSONObject{keys: skillKeys, values: skillValues}
		} else {
			values["skill"] = skill
		}
	}
	for _, key := range []string{"question", "task"} {
		if value, ok := permission[key]; ok {
			keys = append(keys, key)
			values[key] = value
		}
	}

	return orderedJSONObject{keys: keys, values: values}
}

func agentSystemPrompt(options Options) string {
	callerPrompt := strings.TrimSpace(options.SystemPrompt)
	if callerPrompt == "" {
		return agentFieldWorkerInstruction
	}
	return callerPrompt + "\n\n" + agentFieldWorkerInstruction
}

func inlineSystemPromptEnabled(options Options) bool {
	value, ok := options.Env[openCodeInlineSystemPromptEnv]
	if !ok {
		value = os.Getenv(openCodeInlineSystemPromptEnv)
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func openCodeSteps(options Options) int {
	value, ok := options.Env[openCodeStepsEnv]
	if !ok {
		value = os.Getenv(openCodeStepsEnv)
	}
	if steps, err := strconv.Atoi(strings.TrimSpace(value)); err == nil && steps > 0 {
		return steps
	}
	return openCodeDefaultSteps
}

func deepMergeOpenCodeConfig(base, overlay map[string]any) map[string]any {
	// Sized from the base alone: overlay keys mostly land on existing ones, and
	// summing both lengths is what CodeQL's allocation-size-overflow rule flags.
	merged := make(map[string]any, len(base))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range overlay {
		baseObject, baseOK := merged[key].(map[string]any)
		overlayObject, overlayOK := value.(map[string]any)
		if baseOK && overlayOK {
			merged[key] = deepMergeOpenCodeConfig(baseObject, overlayObject)
			continue
		}
		merged[key] = value
	}
	return merged
}

func normalizeAgentFieldHarnessConfig(config map[string]any, stripSystemPrompt bool) {
	agents, ok := config["agent"].(map[string]any)
	if !ok {
		return
	}
	selectedAgent, ok := agents[openCodeAgentName].(map[string]any)
	if !ok {
		return
	}
	if stripSystemPrompt {
		delete(selectedAgent, "prompt")
	}
	permission, ok := selectedAgent["permission"].(map[string]any)
	if !ok {
		return
	}
	selectedAgent["permission"] = orderedOpenCodePermissions(permission)
}

func mergeOpenCodeConfigContent(existingContent, overlayContent string, stripSystemPrompt bool) (string, error) {
	if strings.TrimSpace(existingContent) == "" {
		return overlayContent, nil
	}

	var existingValue any
	if err := json.Unmarshal([]byte(existingContent), &existingValue); err != nil {
		return "", fmt.Errorf("OPENCODE_CONFIG_CONTENT must be valid JSON when supplied to the AgentField OpenCode provider: %w", err)
	}
	existing, ok := existingValue.(map[string]any)
	if !ok {
		return "", fmt.Errorf("OPENCODE_CONFIG_CONTENT must contain a JSON object when supplied to the AgentField OpenCode provider")
	}

	var overlayValue any
	if err := json.Unmarshal([]byte(overlayContent), &overlayValue); err != nil {
		return "", fmt.Errorf("AgentField OpenCode overlay must contain a JSON object")
	}
	overlay, ok := overlayValue.(map[string]any)
	if !ok {
		return "", fmt.Errorf("AgentField OpenCode overlay must contain a JSON object")
	}

	merged := deepMergeOpenCodeConfig(existing, overlay)
	normalizeAgentFieldHarnessConfig(merged, stripSystemPrompt)
	encoded, err := json.Marshal(merged)
	if err != nil {
		return "", fmt.Errorf("serializing OPENCODE_CONFIG_CONTENT: %w", err)
	}
	return string(encoded), nil
}

func inlineOpenCodePrompt(prompt string, options Options) string {
	return fmt.Sprintf(
		"SYSTEM INSTRUCTIONS:\n%s\n\n---\n\nUSER REQUEST:\n%s",
		agentSystemPrompt(options), prompt,
	)
}

func buildOpenCodeConfigContent(options Options, modelValue, variantValue string, includeSystemPrompt bool) (string, error) {
	agent := map[string]any{
		"mode":       "primary",
		"steps":      openCodeSteps(options),
		"permission": orderedOpenCodePermissions(openCodePermissions()),
	}
	if includeSystemPrompt {
		agent["prompt"] = agentSystemPrompt(options)
	}
	if modelValue != "" {
		agent["model"] = modelValue
	}
	if variantValue != "" {
		agent["reasoningEffort"] = variantValue
	}

	content := map[string]any{
		"$schema":       openCodeConfigSchema,
		"default_agent": openCodeAgentName,
		"agent":         map[string]any{openCodeAgentName: agent},
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("serializing AgentField OpenCode overlay: %w", err)
	}
	return string(encoded), nil
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
	cmd := []string{p.BinPath, "run", "--format", "json", "--agent", openCodeAgentName}
	inlineSystemPrompt := inlineSystemPromptEnabled(options)

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

	// Pass model via -m on the run subcommand when supplied. A "#variant"
	// suffix on the model (or an explicit Options.Variant) maps to
	// --variant — opencode's provider-specific reasoning effort (e.g. high,
	// max, minimal).
	modelValue, variantValue := options.resolveModelAndVariant()
	if modelValue != "" {
		cmd = append(cmd, "-m", modelValue)
	}
	if variantValue != "" {
		cmd = append(cmd, "--variant", variantValue)
	}

	// opencode v1.14 does not accept --dangerously-skip-permissions on the
	// `run` subcommand — passing it makes yargs print the run-help screen
	// to stdout and exit 0, which the SDK then captures as the LLM
	// response. opencode in non-TTY mode proceeds without permission
	// prompting, so no flag is needed. See agentfield#582.

	effectivePrompt := prompt
	if inlineSystemPrompt {
		effectivePrompt = inlineOpenCodePrompt(prompt, options)
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

	// The attribution check keys off the BASE model — a "#variant" suffix
	// must not defeat the openrouter/ prefix match nor leak into the
	// per-model config overlay key.
	if strings.HasPrefix(strings.ToLower(modelValue), "openrouter/") {
		if _, callerSet := env["OPENCODE_CONFIG_CONTENT"]; !callerSet && os.Getenv("OPENCODE_CONFIG_CONTENT") == "" {
			attributionEnv := mergedProcessEnv(env)
			headers := openRouterAttributionHeaders(attributionEnv)
			modelSlug := strings.TrimPrefix(modelValue, "openrouter/")
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

	existingConfig, callerSet := env["OPENCODE_CONFIG_CONTENT"]
	if !callerSet {
		existingConfig = os.Getenv("OPENCODE_CONFIG_CONTENT")
	}
	overlayContent, err := buildOpenCodeConfigContent(
		options,
		modelValue,
		variantValue,
		!inlineSystemPrompt,
	)
	if err != nil {
		return nil, err
	}
	mergedConfig, err := mergeOpenCodeConfigContent(existingConfig, overlayContent, inlineSystemPrompt)
	if err != nil {
		return nil, err
	}
	env["OPENCODE_CONFIG_CONTENT"] = mergedConfig

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
	tokens, foundTokens := tokenUsageFromOpenCodeEvents(events)
	if !foundTokens {
		tokens = extractTokenUsage(events)
	}
	raw.Metrics.InputTokens = tokens.inputTokens
	raw.Metrics.OutputTokens = tokens.outputTokens
	raw.Metrics.CacheReadTokens = tokens.cacheReadTokens
	raw.Metrics.CacheCreationTokens = tokens.cacheCreationTokens

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

// tokenUsageFromOpenCodeEvents sums token counts from opencode's per-step
// step_finish.part.tokens objects. The boolean reports whether any step carried
// a tokens object, allowing callers to fall back to generic event usage shapes
// when opencode did not report part-level tokens at all.
func tokenUsageFromOpenCodeEvents(events []map[string]any) (tokenUsage, bool) {
	var total tokenUsage
	found := false
	for _, event := range events {
		if t, _ := event["type"].(string); t != "step_finish" {
			continue
		}
		part, ok := event["part"].(map[string]any)
		if !ok {
			continue
		}
		tokens, ok := part["tokens"].(map[string]any)
		if !ok {
			continue
		}
		found = true
		total.inputTokens += intField(tokens, "input")
		total.outputTokens += intField(tokens, "output") + intField(tokens, "reasoning")
		if cache, ok := tokens["cache"].(map[string]any); ok {
			total.cacheReadTokens += intField(cache, "read")
			total.cacheCreationTokens += intField(cache, "write")
		}
	}
	return total, found
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
