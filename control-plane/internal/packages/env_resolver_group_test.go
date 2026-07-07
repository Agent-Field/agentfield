package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func llmGroupCfg() UserEnvironmentConfig {
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

// Contract: a store read failure while resolving a require_one_of group aborts
// the whole resolve — same as for required/optional variables.
func TestResolve_OneOfGroupStoreReadErrorAborts(t *testing.T) {
	r := resolverWithBrokenScope(t, "swe-planner", &fakePrompter{interactive: true})
	if _, err := r.Resolve(llmGroupCfg()); err == nil {
		t.Fatal("expected Resolve to fail when the store cannot be read for a group option")
	}
}

// Contract: when the chosen group option cannot be persisted, Resolve surfaces
// the save error.
func TestResolve_OneOfGroupPersistErrorSurfaces(t *testing.T) {
	home := t.TempDir()
	store, err := NewSecretStoreWithProvider(home, fixedProvider{pass: "test-pass-phrase"})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	secretsDir := filepath.Join(home, "secrets")
	if err := os.MkdirAll(secretsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(secretsDir, 0o500); err != nil { // read-only: writes fail
		t.Fatal(err)
	}
	defer func() { _ = os.Chmod(secretsDir, 0o700) }()

	r := &EnvResolver{
		Store:    store,
		NodeName: "swe-planner",
		Prompter: &fakePrompter{interactive: true, answers: map[string]string{"OPENROUTER_API_KEY": "sk-or"}},
	}
	_, err = r.Resolve(llmGroupCfg())
	if err == nil || !strings.Contains(err.Error(), "failed to save") {
		t.Fatalf("expected save error, got %v", err)
	}
}

// Contract: a non-interactive session missing both a required var and a group
// reports both in one error.
func TestResolve_MissingRequiredAndGroupCombined(t *testing.T) {
	cfg := llmGroupCfg()
	cfg.Required = []UserEnvironmentVar{{Name: "GH_TOKEN", Type: "secret"}}
	r := newResolver(t, "swe-planner", &fakePrompter{interactive: false})
	_, err := r.Resolve(cfg)
	if err == nil {
		t.Fatal("expected error for missing required var and group")
	}
	msg := err.Error()
	if !strings.Contains(msg, "GH_TOKEN") || !strings.Contains(msg, "ANTHROPIC_API_KEY") {
		t.Fatalf("error should mention both the missing required var and the group: %q", msg)
	}
}

// Contract: the install-time warning lists an unsatisfied require_one_of group
// and skips a satisfied one, without touching required-only output paths.
func TestCheckEnvironmentVariables_ShowsGroups(t *testing.T) {
	t.Setenv("SET_PROVIDER", "yes") // satisfies the second group
	pi := &PackageInstaller{}
	pi.checkEnvironmentVariables(&PackageMetadata{
		Name: "llm-node",
		UserEnvironment: UserEnvironmentConfig{
			Required: []UserEnvironmentVar{{Name: "UNSET_REQUIRED"}},
			RequireOneOf: []RequireOneOfGroup{
				{ID: "g1", Options: []UserEnvironmentVar{{Name: "A1"}, {Name: "A2"}}}, // unsatisfied, empty desc
				{ID: "g2", Description: "second", Options: []UserEnvironmentVar{{Name: "SET_PROVIDER"}}}, // satisfied
			},
			Optional: []UserEnvironmentVar{{Name: "OPT", Default: "d"}},
		},
	})
}

func TestEnvGroupSatisfied(t *testing.T) {
	g := RequireOneOfGroup{Options: []UserEnvironmentVar{{Name: "GK1"}, {Name: "GK2"}}}
	if envGroupSatisfied(g) {
		t.Fatal("group with no options set should be unsatisfied")
	}
	t.Setenv("GK2", "v")
	if !envGroupSatisfied(g) {
		t.Fatal("group with one option set should be satisfied")
	}
}
