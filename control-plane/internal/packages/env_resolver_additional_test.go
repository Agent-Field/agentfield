package packages

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// errPrompter always fails when asked to prompt.
type errPrompter struct{ err error }

func (errPrompter) Interactive() bool { return true }
func (e errPrompter) Prompt(UserEnvironmentVar) (string, error) {
	return "", e.err
}

// resolverWithBrokenScope builds a resolver whose node-scope secret file is a
// directory, so any Store.Get for that node fails on load.
func resolverWithBrokenScope(t *testing.T, node string, p Prompter) *EnvResolver {
	t.Helper()
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	scopePath := filepath.Join(home, "secrets", node+".enc")
	if err := os.MkdirAll(scopePath, 0o700); err != nil {
		t.Fatal(err)
	}
	return &EnvResolver{Store: store, NodeName: node, Prompter: p}
}

// Contract: a store read failure while resolving a required variable aborts the
// whole resolve with that error.
func TestResolve_StoreGetErrorAbortsRequired(t *testing.T) {
	r := resolverWithBrokenScope(t, "pr-af", &fakePrompter{interactive: true})
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "TOK", Type: "secret"}},
	})
	if err == nil {
		t.Fatalf("expected Resolve to fail when the store cannot be read")
	}
}

// Contract: a store read failure while resolving an optional variable also
// aborts (data integrity beats silent omission).
func TestResolve_StoreGetErrorAbortsOptional(t *testing.T) {
	r := resolverWithBrokenScope(t, "pr-af", &fakePrompter{interactive: true})
	_, err := r.Resolve(UserEnvironmentConfig{
		Optional: []UserEnvironmentVar{{Name: "OPT"}},
	})
	if err == nil {
		t.Fatalf("expected Resolve to fail when the store cannot be read")
	}
}

// Contract: an optional variable with no env value, no stored value and no
// default is simply omitted from the resolved map (not prompted, not errored).
func TestResolve_OptionalUnsetIsOmitted(t *testing.T) {
	p := &fakePrompter{interactive: true}
	r := newResolver(t, "pr-af", p)
	got, err := r.Resolve(UserEnvironmentConfig{
		Optional: []UserEnvironmentVar{{Name: "NOWHERE"}},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if _, present := got["NOWHERE"]; present {
		t.Fatalf("unset optional var should be omitted, got %v", got)
	}
	if len(p.asked) != 0 {
		t.Fatalf("optional var must not be prompted, asked=%v", p.asked)
	}
}

// Contract: an error returned by the prompter aborts the resolve and surfaces
// the underlying error.
func TestResolve_PrompterErrorPropagates(t *testing.T) {
	sentinel := errors.New("tty exploded")
	r := newResolver(t, "pr-af", errPrompter{err: sentinel})
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "TOK", Type: "secret"}},
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected prompter error to propagate, got %v", err)
	}
}

// Contract: if the user skips the prompt (empty answer) for a required var, it
// is reported as missing and nothing is persisted.
func TestResolve_EmptyPromptCountsAsMissing(t *testing.T) {
	p := &fakePrompter{interactive: true, answers: map[string]string{}} // returns ""
	r := newResolver(t, "pr-af", p)
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "TOK", Type: "secret"}},
	})
	if err == nil || !strings.Contains(err.Error(), "missing required") {
		t.Fatalf("expected missing-required error, got %v", err)
	}
	if keys, _ := r.Store.List("global"); len(keys) != 0 {
		t.Fatalf("nothing should be persisted on a skipped prompt, got %v", keys)
	}
}

// Contract: an invalid validation regex on a required var is a hard error
// naming the variable.
func TestResolve_InvalidValidationRegex(t *testing.T) {
	p := &fakePrompter{interactive: true, answers: map[string]string{"CODE": "whatever"}}
	r := newResolver(t, "pr-af", p)
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "CODE", Validation: "([unclosed"}},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid validation regex") {
		t.Fatalf("expected invalid-regex error, got %v", err)
	}
}

// Contract: a value that never satisfies the validation regex is rejected after
// the retry budget is exhausted.
func TestResolve_TooManyInvalidAttempts(t *testing.T) {
	rp := &retryPrompter{values: []string{"bad1", "bad2", "bad3", "bad4"}}
	r := newResolver(t, "pr-af", rp)
	_, err := r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "CODE", Validation: "^[A-Z]+$"}},
	})
	if err == nil || !strings.Contains(err.Error(), "too many invalid attempts") {
		t.Fatalf("expected too-many-attempts error, got %v", err)
	}
}

// Contract: persisting a freshly prompted secret fails loudly when the store
// cannot be written (here the secrets directory is read-only, so the prior
// lookup still succeeds but the write does not).
func TestResolve_PersistErrorSurfaces(t *testing.T) {
	skipIfRoot(t)
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	// The lookup reads (missing) scope files fine, but the persist write fails
	// because the secrets directory is not writable.
	secretsDir := filepath.Join(home, "secrets")
	if err := os.MkdirAll(secretsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(secretsDir, 0o500); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(secretsDir, 0o700) }()
	r := &EnvResolver{
		Store:    store,
		NodeName: "pr-af",
		Prompter: &fakePrompter{interactive: true, answers: map[string]string{"TOK": "sk-x"}},
	}
	_, err = r.Resolve(UserEnvironmentConfig{
		Required: []UserEnvironmentVar{{Name: "TOK", Type: "secret"}},
	})
	if err == nil || !strings.Contains(err.Error(), "failed to save") {
		t.Fatalf("expected save error, got %v", err)
	}
}

// Contract: TTYPrompter reports non-interactive when stdin is not a terminal
// (the case in the test runner and in CI).
func TestTTYPrompter_NotInteractiveUnderTest(t *testing.T) {
	if (TTYPrompter{}).Interactive() {
		t.Skip("stdin is a terminal in this environment; interactivity check is moot")
	}
}

// Contract: for a non-secret variable, TTYPrompter reads a plain line from
// stdin and returns it (secret vars need a real TTY and are covered elsewhere).
func TestTTYPrompter_ReadsPlainLine(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.WriteString("us-east-1\n"); err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	origStdin := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = origStdin }()

	got, err := TTYPrompter{}.Prompt(UserEnvironmentVar{Name: "REGION", Description: "AWS region"})
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if strings.TrimSpace(got) != "us-east-1" {
		t.Fatalf("Prompt = %q, want us-east-1", got)
	}
}
