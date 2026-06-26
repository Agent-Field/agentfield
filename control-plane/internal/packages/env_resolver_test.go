package packages

import (
	"testing"
)

// fakePrompter returns canned answers and records what it was asked.
type fakePrompter struct {
	interactive bool
	answers     map[string]string
	asked       []string
}

func (f *fakePrompter) Interactive() bool { return f.interactive }
func (f *fakePrompter) Prompt(v UserEnvironmentVar) (string, error) {
	f.asked = append(f.asked, v.Name)
	return f.answers[v.Name], nil
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
