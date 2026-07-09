package packages

import (
	"strings"
	"testing"
)

// fakePrompter returns canned answers and records what it was asked. choices
// feeds PromptLine (the require_one_of menu selection) in order; once exhausted
// it returns "" (i.e. the user pressed Enter to skip).
type fakePrompter struct {
	interactive bool
	answers     map[string]string
	asked       []string
	choices     []string
	ci          int
}

func (f *fakePrompter) Interactive() bool { return f.interactive }
func (f *fakePrompter) Prompt(v UserEnvironmentVar) (string, error) {
	f.asked = append(f.asked, v.Name)
	return f.answers[v.Name], nil
}
func (f *fakePrompter) PromptLine(string) (string, error) {
	if f.ci >= len(f.choices) {
		return "", nil
	}
	v := f.choices[f.ci]
	f.ci++
	return v, nil
}

func newResolver(t *testing.T, node string, p Prompter) *EnvResolver {
	t.Helper()
	store, err := NewSecretStore(t.TempDir())
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	return &EnvResolver{Store: store, NodeName: node, Prompter: p}
}

// Contract: a value already in the process environment is used and NOT persisted.
func TestResolve_ProcessEnvWinsAndIsNotPersisted(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "from-env")
	p := &fakePrompter{interactive: true}
	r := newResolver(t, "pr-af", p)

	got, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "OPENROUTER_API_KEY", Type: "secret"}},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "from-env" {
		t.Fatalf("got %q, want from-env", got["OPENROUTER_API_KEY"])
	}
	if len(p.asked) != 0 {
		t.Fatalf("should not prompt when env is set, asked=%v", p.asked)
	}
	if _, ok, _ := r.Store.Get("pr-af", "OPENROUTER_API_KEY"); ok {
		t.Fatalf("process-env value must not be persisted to the store")
	}
}

// Contract: a missing required secret is prompted once, persisted encrypted to
// the global scope, and reused on the next resolve without prompting again.
func TestResolve_PromptsThenPersistsAndReuses(t *testing.T) {
	p := &fakePrompter{interactive: true, answers: map[string]string{"OPENROUTER_API_KEY": "sk-prompted"}}
	r := newResolver(t, "pr-af", p)
	cfg := UserEnvironmentConfig{Required: []UserEnvironmentVar{{Name: "OPENROUTER_API_KEY", Type: "secret"}}}

	got, err := r.Resolve(cfg)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "sk-prompted" {
		t.Fatalf("got %q, want sk-prompted", got["OPENROUTER_API_KEY"])
	}
	if v, ok, _ := r.Store.Get("pr-af", "OPENROUTER_API_KEY"); !ok || v != "sk-prompted" {
		t.Fatalf("secret not persisted to global scope")
	}

	// Second resolve with a non-interactive prompter must not be asked again.
	r2 := &EnvResolver{Store: r.Store, NodeName: "pr-af", Prompter: &fakePrompter{interactive: false}}
	got2, err := r2.Resolve(cfg)
	if err != nil {
		t.Fatalf("second Resolve: %v", err)
	}
	if got2["OPENROUTER_API_KEY"] != "sk-prompted" {
		t.Fatalf("stored secret not reused: %q", got2["OPENROUTER_API_KEY"])
	}
}

// Contract: a shared (global-scope) secret entered for one node is reused by
// another node without re-prompting.
func TestResolve_GlobalSecretSharedAcrossNodes(t *testing.T) {
	store, _ := NewSecretStore(t.TempDir())
	cfg := UserEnvironmentConfig{Required: []UserEnvironmentVar{{Name: "OPENROUTER_API_KEY", Type: "secret"}}}

	r1 := &EnvResolver{Store: store, NodeName: "pr-af",
		Prompter: &fakePrompter{interactive: true, answers: map[string]string{"OPENROUTER_API_KEY": "shared"}}}
	if _, err := r1.Resolve(cfg); err != nil {
		t.Fatalf("r1: %v", err)
	}

	// Different node, non-interactive — must resolve from the shared global value.
	r2 := &EnvResolver{Store: store, NodeName: "sec-af", Prompter: &fakePrompter{interactive: false}}
	got, err := r2.Resolve(cfg)
	if err != nil {
		t.Fatalf("r2: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "shared" {
		t.Fatalf("sec-af did not reuse shared key: %q", got["OPENROUTER_API_KEY"])
	}
}

// Contract: a node-scoped variable persists to the node scope, not global.
func TestResolve_NodeScopedPersistsToNode(t *testing.T) {
	p := &fakePrompter{interactive: true, answers: map[string]string{"NODE_TOKEN": "nv"}}
	r := newResolver(t, "pr-af", p)
	if _, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "NODE_TOKEN", Type: "secret", Scope: "node"}},
	}); err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if keys, _ := r.Store.List("pr-af"); len(keys) != 1 || keys[0] != "NODE_TOKEN" {
		t.Fatalf("NODE_TOKEN not in node scope: %v", keys)
	}
	if keys, _ := r.Store.List("global"); len(keys) != 0 {
		t.Fatalf("node-scoped secret leaked into global: %v", keys)
	}
}

// Contract: in a non-interactive session, a missing required var is an error
// naming the missing variable.
func TestResolve_NonInteractiveMissingErrors(t *testing.T) {
	r := newResolver(t, "pr-af", &fakePrompter{interactive: false})
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "MUST_HAVE", Type: "secret"}},
	})
	if err == nil {
		t.Fatalf("expected error for missing required var")
	}
}

// Contract: optional vars fall back to their default and are never prompted.
func TestResolve_OptionalUsesDefaultNoPrompt(t *testing.T) {
	p := &fakePrompter{interactive: true}
	r := newResolver(t, "pr-af", p)
	got, err := r.Resolve(UserEnvironmentConfig{
		Optional: []UserEnvironmentVar{{Name: "REGION", Default: "us-east-1"}},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["REGION"] != "us-east-1" {
		t.Fatalf("REGION = %q, want default us-east-1", got["REGION"])
	}
	if len(p.asked) != 0 {
		t.Fatalf("optional vars must not prompt, asked=%v", p.asked)
	}
}

// Contract: a value failing the validation regex is rejected; a later valid
// answer is accepted.
func TestResolve_ValidationRegexRejectsThenAccepts(t *testing.T) {
	// retryPrompter returns a bad value first, then a good one.
	rp := &retryPrompter{values: []string{"bad value", "GOOD123"}}
	r := newResolver(t, "pr-af", rp)
	got, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "CODE", Validation: "^[A-Z0-9]+$"}},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["CODE"] != "GOOD123" {
		t.Fatalf("CODE = %q, want GOOD123", got["CODE"])
	}
}

// llmGroup is the motivating require_one_of case: an Anthropic key OR an
// OpenRouter key, at least one required.
func llmGroup() UserEnvironmentConfig {
	return UserEnvironmentConfig{
		RequireOneOf: []RequireOneOfGroup{{
			ID:          "llm_provider",
			Description: "an LLM provider key",
			Options: []UserEnvironmentVar{
				{Name: "ANTHROPIC_API_KEY", Type: "secret"},
				{Name: "OPENROUTER_API_KEY", Type: "secret"},
			},
		}},
	}
}

// Contract: a require_one_of group is satisfied when one option is in the
// environment; the other option is not required and not prompted.
func TestResolve_OneOfSatisfiedByEnv(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "sk-or-env")
	p := &fakePrompter{interactive: true}
	r := newResolver(t, "swe-planner", p)

	got, err := r.Resolve(llmGroup())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "sk-or-env" {
		t.Fatalf("group option not injected: %v", got)
	}
	if _, ok := got["ANTHROPIC_API_KEY"]; ok {
		t.Fatalf("the other option must not be set")
	}
	if len(p.asked) != 0 {
		t.Fatalf("a satisfied group must not prompt, asked=%v", p.asked)
	}
}

// Contract: a satisfied group's stored option is reused non-interactively.
func TestResolve_OneOfSatisfiedByStore(t *testing.T) {
	store, _ := NewSecretStore(t.TempDir())
	if err := store.Set("global", "ANTHROPIC_API_KEY", "sk-ant-stored"); err != nil {
		t.Fatal(err)
	}
	r := &EnvResolver{Store: store, NodeName: "swe-planner", Prompter: &fakePrompter{interactive: false}}
	got, err := r.Resolve(llmGroup())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["ANTHROPIC_API_KEY"] != "sk-ant-stored" {
		t.Fatalf("stored group option not reused: %v", got)
	}
}

// Contract: an unsatisfied group in a non-interactive session errors, naming
// every option so the caller knows the alternatives.
func TestResolve_OneOfNonInteractiveErrorsNamingOptions(t *testing.T) {
	r := newResolver(t, "swe-planner", &fakePrompter{interactive: false})
	_, err := r.Resolve(llmGroup())
	if err == nil {
		t.Fatal("expected error for unsatisfied require_one_of group")
	}
	msg := err.Error()
	if !strings.Contains(msg, "ANTHROPIC_API_KEY") || !strings.Contains(msg, "OPENROUTER_API_KEY") {
		t.Fatalf("error should name both options: %q", msg)
	}
}

// Contract: interactively, choosing one option from the menu satisfies the
// group and persists just that option encrypted.
func TestResolve_OneOfPromptFillsOneAndPersists(t *testing.T) {
	// User picks OpenRouter (option 2) and provides its key.
	p := &fakePrompter{
		interactive: true,
		choices:     []string{"2"},
		answers:     map[string]string{"OPENROUTER_API_KEY": "sk-or-prompted"},
	}
	r := newResolver(t, "swe-planner", p)

	got, err := r.Resolve(llmGroup())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "sk-or-prompted" {
		t.Fatalf("chosen option not resolved: %v", got)
	}
	if _, ok := got["ANTHROPIC_API_KEY"]; ok {
		t.Fatalf("skipped option must not be set")
	}
	if v, ok, _ := r.Store.Get("swe-planner", "OPENROUTER_API_KEY"); !ok || v != "sk-or-prompted" {
		t.Fatalf("chosen option not persisted encrypted")
	}
	if _, ok, _ := r.Store.Get("swe-planner", "ANTHROPIC_API_KEY"); ok {
		t.Fatalf("skipped option must not be persisted")
	}
}

// Contract: if the user skips every option, the group stays unsatisfied and
// Resolve errors.
func TestResolve_OneOfPromptAllSkippedErrors(t *testing.T) {
	p := &fakePrompter{interactive: true, answers: map[string]string{}} // all answers empty
	r := newResolver(t, "swe-planner", p)
	if _, err := r.Resolve(llmGroup()); err == nil {
		t.Fatal("expected error when every group option is skipped")
	}
}

// Contract: required vars and groups coexist — a set required var plus a
// satisfied group both resolve.
func TestResolve_RequiredAndGroupTogether(t *testing.T) {
	t.Setenv("GH_TOKEN", "ghp_x")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant")
	cfg := llmGroup()
	cfg.Required = []UserEnvironmentVar{{Name: "GH_TOKEN", Type: "secret"}}
	r := newResolver(t, "swe-planner", &fakePrompter{interactive: false})
	got, err := r.Resolve(cfg)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got["GH_TOKEN"] != "ghp_x" || got["ANTHROPIC_API_KEY"] != "sk-ant" {
		t.Fatalf("required+group not both resolved: %v", got)
	}
}

type retryPrompter struct {
	values []string
	i      int
}

func (r *retryPrompter) Interactive() bool { return true }
func (r *retryPrompter) Prompt(v UserEnvironmentVar) (string, error) {
	if r.i >= len(r.values) {
		return "", nil
	}
	val := r.values[r.i]
	r.i++
	return val, nil
}
func (r *retryPrompter) PromptLine(string) (string, error) { return "", nil }
