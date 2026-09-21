package harness

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

type capturedOpenCodeInvocation struct {
	cmd   []string
	env   map[string]string
	stdin []byte
}

func captureOpenCodeInvocation(t *testing.T, prompt string, options Options) capturedOpenCodeInvocation {
	t.Helper()
	originalPromptViaStdin := openCodePromptViaStdin
	openCodePromptViaStdin = false
	t.Cleanup(func() { openCodePromptViaStdin = originalPromptViaStdin })

	var captured capturedOpenCodeInvocation
	provider := NewOpenCodeProvider("opencode", "")
	provider.runCLI = func(_ context.Context, cmd []string, env map[string]string, _ string, _ int, stdin []byte) (*CLIResult, error) {
		captured.cmd = append([]string(nil), cmd...)
		captured.env = make(map[string]string, len(env))
		for key, value := range env {
			captured.env[key] = value
		}
		captured.stdin = append([]byte(nil), stdin...)
		return &CLIResult{Stdout: "ok\n", ReturnCode: 0}, nil
	}

	raw, err := provider.Execute(context.Background(), prompt, options)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if raw.IsError {
		t.Fatalf("Execute returned provider error: %s", raw.ErrorMessage)
	}
	return captured
}

func decodeCapturedOpenCodeConfig(t *testing.T, captured capturedOpenCodeInvocation) map[string]any {
	t.Helper()
	content := captured.env["OPENCODE_CONFIG_CONTENT"]
	if content == "" {
		t.Fatal("OPENCODE_CONFIG_CONTENT was not passed to the child")
	}
	var config map[string]any
	if err := json.Unmarshal([]byte(content), &config); err != nil {
		t.Fatalf("decode OPENCODE_CONFIG_CONTENT: %v", err)
	}
	return config
}

func harnessAgentFromConfig(t *testing.T, config map[string]any) map[string]any {
	t.Helper()
	agents, ok := config["agent"].(map[string]any)
	if !ok {
		t.Fatalf("agent config is %T, want object", config["agent"])
	}
	agent, ok := agents[openCodeAgentName].(map[string]any)
	if !ok {
		t.Fatalf("agent.%s is %T, want object", openCodeAgentName, agents[openCodeAgentName])
	}
	return agent
}

func TestOpenCodeHarnessOverlayAndPromptTransport(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "")
	t.Setenv(openCodeStepsEnv, "")

	const task = "inspect the repository byte-for-byte"
	captured := captureOpenCodeInvocation(t, task, Options{
		Model:          "openai/gpt-5#low",
		Variant:        "max",
		MaxTurns:       3,
		PermissionMode: "plan",
		SystemPrompt:   "  Work autonomously.  ",
		Tools:          []string{"Read", "Write", "Bash"},
		Env:            map[string]string{"CUSTOM": "kept"},
	})

	wantCmd := []string{
		"opencode", "run", "--format", "json",
		"--agent", openCodeAgentName,
		"-m", "openai/gpt-5", "--variant", "max", task,
	}
	if !reflect.DeepEqual(captured.cmd, wantCmd) {
		t.Fatalf("argv mismatch:\n got  %q\n want %q", captured.cmd, wantCmd)
	}
	if captured.env["CUSTOM"] != "kept" {
		t.Fatalf("custom env was not preserved: %#v", captured.env)
	}

	config := decodeCapturedOpenCodeConfig(t, captured)
	if config["$schema"] != openCodeConfigSchema {
		t.Fatalf("$schema = %#v", config["$schema"])
	}
	if config["default_agent"] != openCodeAgentName {
		t.Fatalf("default_agent = %#v", config["default_agent"])
	}
	agent := harnessAgentFromConfig(t, config)
	if agent["mode"] != "primary" || agent["steps"] != float64(openCodeDefaultSteps) {
		t.Fatalf("mode/steps = %#v/%#v", agent["mode"], agent["steps"])
	}
	wantPrompt := "Work autonomously.\n\n" + agentFieldWorkerInstruction
	if agent["prompt"] != wantPrompt {
		t.Fatalf("agent prompt = %q, want %q", agent["prompt"], wantPrompt)
	}
	if agent["model"] != "openai/gpt-5" || agent["reasoningEffort"] != "max" {
		t.Fatalf("model/reasoningEffort = %#v/%#v", agent["model"], agent["reasoningEffort"])
	}
	wantPermissions := map[string]any{
		"*":        "allow",
		"skill":    map[string]any{"agentfield*": "deny"},
		"question": "deny",
		"task":     "deny",
	}
	if !reflect.DeepEqual(agent["permission"], wantPermissions) {
		t.Fatalf("permission = %#v, want %#v", agent["permission"], wantPermissions)
	}
	if _, ok := agent["max_turns"]; ok {
		t.Fatalf("max_turns must not be serialized: %#v", agent)
	}

	content := captured.env["OPENCODE_CONFIG_CONTENT"]
	permissionStart := strings.Index(content, `"permission":{`)
	if permissionStart < 0 {
		t.Fatalf("serialized permission object missing from %s", content)
	}
	permissionJSON := content[permissionStart:]
	wildcardIndex := strings.Index(permissionJSON, `"*"`)
	for _, denial := range []string{"question", "skill", "task"} {
		denialIndex := strings.Index(permissionJSON, `"`+denial+`"`)
		if wildcardIndex < 0 || denialIndex < 0 || wildcardIndex >= denialIndex {
			t.Fatalf("wildcard must be serialized before %q: %s", denial, permissionJSON)
		}
	}
}

func TestOpenCodeHarnessOverlayOmitsUnresolvedModelAndUsesWorkerInstruction(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "")
	t.Setenv(openCodeStepsEnv, "")

	agent := harnessAgentFromConfig(t, decodeCapturedOpenCodeConfig(
		t,
		captureOpenCodeInvocation(t, "task only", Options{}),
	))
	if agent["prompt"] != agentFieldWorkerInstruction {
		t.Fatalf("prompt = %#v", agent["prompt"])
	}
	if _, ok := agent["model"]; ok {
		t.Fatalf("unresolved model must be omitted: %#v", agent)
	}
	if _, ok := agent["reasoningEffort"]; ok {
		t.Fatalf("unresolved reasoningEffort must be omitted: %#v", agent)
	}
}

func TestOpenCodeHarnessOverlayMergesCallerAndAmbientConfig(t *testing.T) {
	t.Setenv(openCodeInlineSystemPromptEnv, "")
	t.Setenv(openCodeStepsEnv, "")
	ambient := `{"provider":{"ambient":{"model":"ambient"}},"plugin":["ambient"]}`
	t.Setenv("OPENCODE_CONFIG_CONTENT", ambient)
	caller := `{
		"provider":{"keep-me":{"npm":"custom/provider"}},
		"plugin":["caller"],
		"mcp":{"local":{"type":"local","command":["tool"]}},
		"agent":{
			"custom-agent":{"prompt":"preserve this agent"},
			"agentfield-harness":{
				"mode":"secondary",
				"temperature":0.2,
				"permission":{"read":"deny","skill":{"other*":"allow"}}
			}
		}
	}`

	config := decodeCapturedOpenCodeConfig(t, captureOpenCodeInvocation(t, "hello", Options{
		Model: "openai/gpt-5",
		Env:   map[string]string{"OPENCODE_CONFIG_CONTENT": caller},
	}))
	provider := config["provider"].(map[string]any)
	if _, ok := provider["ambient"]; ok {
		t.Fatalf("ambient config must lose to per-call config: %#v", provider)
	}
	if !reflect.DeepEqual(provider["keep-me"], map[string]any{"npm": "custom/provider"}) {
		t.Fatalf("caller provider was not preserved: %#v", provider)
	}
	if !reflect.DeepEqual(config["plugin"], []any{"caller"}) {
		t.Fatalf("caller array was not preserved: %#v", config["plugin"])
	}
	if config["mcp"].(map[string]any)["local"].(map[string]any)["type"] != "local" {
		t.Fatalf("caller MCP config was not preserved: %#v", config["mcp"])
	}
	agents := config["agent"].(map[string]any)
	if agents["custom-agent"].(map[string]any)["prompt"] != "preserve this agent" {
		t.Fatalf("unrelated agent was not preserved: %#v", agents)
	}
	agent := harnessAgentFromConfig(t, config)
	if agent["mode"] != "primary" || agent["temperature"] != 0.2 || agent["model"] != "openai/gpt-5" {
		t.Fatalf("generated fields did not win while caller fields survived: %#v", agent)
	}
	permission := agent["permission"].(map[string]any)
	if permission["read"] != "deny" || permission["*"] != "allow" || permission["task"] != "deny" {
		t.Fatalf("permission deep merge failed: %#v", permission)
	}
	skill := permission["skill"].(map[string]any)
	if skill["other*"] != "allow" || skill["agentfield*"] != "deny" {
		t.Fatalf("skill permission deep merge failed: %#v", skill)
	}

	t.Setenv("OPENCODE_CONFIG_CONTENT", `{"provider":{"ambient":{"model":"kept"}}}`)
	ambientConfig := decodeCapturedOpenCodeConfig(t, captureOpenCodeInvocation(t, "hello", Options{}))
	if ambientConfig["provider"].(map[string]any)["ambient"].(map[string]any)["model"] != "kept" {
		t.Fatalf("ambient config was not preserved: %#v", ambientConfig)
	}
}

func TestOpenCodeHarnessOverlayCoexistsWithAttribution(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "")
	t.Setenv(openCodeStepsEnv, "")

	config := decodeCapturedOpenCodeConfig(t, captureOpenCodeInvocation(t, "hello", Options{
		Model: "openrouter/openai/gpt-4o",
	}))
	models := config["provider"].(map[string]any)["openrouter"].(map[string]any)["models"].(map[string]any)
	headers := models["openai/gpt-4o"].(map[string]any)["headers"].(map[string]any)
	if headers["HTTP-Referer"] != defaultOpenRouterSiteURL {
		t.Fatalf("attribution headers missing: %#v", headers)
	}
	_ = harnessAgentFromConfig(t, config)
}

func TestOpenCodeInlineSystemPromptRollback(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "off")
	t.Setenv(openCodeStepsEnv, "")

	callerConfig := `{"provider":{"custom":{"model":"keep"}},"agent":{"agentfield-harness":{"prompt":"remove me","temperature":0.2}}}`
	options := Options{
		Model:          "openai/gpt-5#low",
		Variant:        "max",
		SystemPrompt:   "  caller instructions  ",
		Tools:          []string{"Read"},
		PermissionMode: "auto",
		Env: map[string]string{
			openCodeInlineSystemPromptEnv: "  TrUe  ",
			"OPENCODE_CONFIG_CONTENT":     callerConfig,
		},
	}
	captured := captureOpenCodeInvocation(t, "complete the task", options)
	wantPrompt := "SYSTEM INSTRUCTIONS:\ncaller instructions\n\n" + agentFieldWorkerInstruction +
		"\n\n---\n\nUSER REQUEST:\ncomplete the task"
	if captured.cmd[len(captured.cmd)-1] != wantPrompt {
		t.Fatalf("inline prompt = %q, want %q", captured.cmd[len(captured.cmd)-1], wantPrompt)
	}
	if !reflect.DeepEqual(captured.cmd[:len(captured.cmd)-1], []string{
		"opencode", "run", "--format", "json", "--agent", openCodeAgentName,
		"-m", "openai/gpt-5", "--variant", "max",
	}) {
		t.Fatalf("inline argv flags changed: %q", captured.cmd)
	}
	agent := harnessAgentFromConfig(t, decodeCapturedOpenCodeConfig(t, captured))
	if _, ok := agent["prompt"]; ok {
		t.Fatalf("inline mode must strip caller and generated agent prompts: %#v", agent)
	}
	if agent["temperature"] != 0.2 || !reflect.DeepEqual(agent["permission"], map[string]any{
		"*":        "allow",
		"skill":    map[string]any{"agentfield*": "deny"},
		"question": "deny",
		"task":     "deny",
	}) {
		t.Fatalf("inline mode changed merged agent config: %#v", agent)
	}

	t.Setenv(openCodeInlineSystemPromptEnv, " YES ")
	ambientCaptured := captureOpenCodeInvocation(t, "ambient task", Options{})
	if !strings.Contains(ambientCaptured.cmd[len(ambientCaptured.cmd)-1], agentFieldWorkerInstruction) {
		t.Fatalf("ambient inline opt-out did not include worker instruction: %q", ambientCaptured.cmd)
	}
}

func TestOpenCodeStepsOverride(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "")

	cases := []struct {
		name      string
		ambient   string
		perCall   *string
		wantSteps float64
		maxTurns  int
	}{
		{name: "default is independent of max turns", wantSteps: 500, maxTurns: 3},
		{name: "ambient positive integer", ambient: "42", wantSteps: 42},
		{name: "per-call wins", ambient: "42", perCall: stringPointer("17"), wantSteps: 17},
		{name: "per-call invalid ignores ambient", ambient: "42", perCall: stringPointer("not-a-number"), wantSteps: 500},
		{name: "zero is ignored", perCall: stringPointer("0"), wantSteps: 500},
		{name: "negative is ignored", perCall: stringPointer("-2"), wantSteps: 500},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv(openCodeStepsEnv, test.ambient)
			options := Options{MaxTurns: test.maxTurns}
			if test.perCall != nil {
				options.Env = map[string]string{openCodeStepsEnv: *test.perCall}
			}
			agent := harnessAgentFromConfig(t, decodeCapturedOpenCodeConfig(
				t,
				captureOpenCodeInvocation(t, "hello", options),
			))
			if agent["steps"] != test.wantSteps {
				t.Fatalf("steps = %#v, want %v", agent["steps"], test.wantSteps)
			}
		})
	}
}

func stringPointer(value string) *string {
	return &value
}

func TestOpenCodeRejectsMalformedCallerConfigBeforeLaunching(t *testing.T) {
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	t.Setenv(openCodeInlineSystemPromptEnv, "")
	t.Setenv(openCodeStepsEnv, "")

	for _, content := range []string{"{not-json", `[]`, `null`, `"string"`} {
		t.Run(content, func(t *testing.T) {
			launched := false
			provider := NewOpenCodeProvider("opencode", "")
			provider.runCLI = func(context.Context, []string, map[string]string, string, int, []byte) (*CLIResult, error) {
				launched = true
				return &CLIResult{}, nil
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			result, err := provider.Execute(ctx, "hello", Options{
				Env: map[string]string{"OPENCODE_CONFIG_CONTENT": content},
			})
			if err == nil || !strings.Contains(err.Error(), "OPENCODE_CONFIG_CONTENT") {
				t.Fatalf("error = %v, want OPENCODE_CONFIG_CONTENT validation error", err)
			}
			if result != nil {
				t.Fatalf("result = %#v, want nil", result)
			}
			if launched {
				t.Fatal("subprocess launched for malformed caller config")
			}
		})
	}
}
