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
		Prompter: &fakePrompter{
			interactive: true,
			choices:     []string{"2"}, // pick OPENROUTER_API_KEY from the menu
			answers:     map[string]string{"OPENROUTER_API_KEY": "sk-or"},
		},
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

// Contract: the menu prompts only for the option the user picks, resolves and
// persists just that one, and leaves the alternatives untouched.
func TestResolve_OneOfGroupMenuSelectsChosenOption(t *testing.T) {
	p := &fakePrompter{
		interactive: true,
		choices:     []string{"1"}, // pick ANTHROPIC_API_KEY
		answers:     map[string]string{"ANTHROPIC_API_KEY": "sk-ant"},
	}
	r := newResolver(t, "swe-planner", p)
	got, err := r.Resolve(llmGroupCfg())
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got["ANTHROPIC_API_KEY"] != "sk-ant" {
		t.Fatalf("chosen option not resolved: %v", got)
	}
	if _, ok := got["OPENROUTER_API_KEY"]; ok {
		t.Fatal("non-selected option must not be resolved")
	}
	if len(p.asked) != 1 || p.asked[0] != "ANTHROPIC_API_KEY" {
		t.Fatalf("only the chosen option should be prompted, asked=%v", p.asked)
	}
	if v, ok, _ := r.Store.Get("swe-planner", "ANTHROPIC_API_KEY"); !ok || v != "sk-ant" {
		t.Fatalf("chosen option not persisted: %q ok=%v", v, ok)
	}
}

// Contract: selecting the second menu entry resolves that alternative.
func TestResolve_OneOfGroupMenuSelectsSecondOption(t *testing.T) {
	p := &fakePrompter{
		interactive: true,
		choices:     []string{"2"},
		answers:     map[string]string{"OPENROUTER_API_KEY": "sk-or"},
	}
	r := newResolver(t, "swe-planner", p)
	got, err := r.Resolve(llmGroupCfg())
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got["OPENROUTER_API_KEY"] != "sk-or" {
		t.Fatalf("second option not resolved: %v", got)
	}
	if _, ok := got["ANTHROPIC_API_KEY"]; ok {
		t.Fatal("unselected option resolved")
	}
	if p.asked[0] != "OPENROUTER_API_KEY" {
		t.Fatalf("wrong option prompted: %v", p.asked)
	}
}

// Contract: an out-of-range / non-numeric selection re-prompts rather than
// picking anything, and a later valid choice still works.
func TestResolve_OneOfGroupMenuRetriesOnInvalidChoice(t *testing.T) {
	p := &fakePrompter{
		interactive: true,
		choices:     []string{"9", "abc", "1"}, // two bad, then valid
		answers:     map[string]string{"ANTHROPIC_API_KEY": "sk-ant"},
	}
	r := newResolver(t, "swe-planner", p)
	got, err := r.Resolve(llmGroupCfg())
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got["ANTHROPIC_API_KEY"] != "sk-ant" {
		t.Fatalf("retry should have selected option 1: %v", got)
	}
}

// Contract: pressing Enter at the menu skips the whole group — no option is
// prompted — and the group is reported missing.
func TestResolve_OneOfGroupMenuSkipLeavesGroupMissing(t *testing.T) {
	p := &fakePrompter{interactive: true} // no choices -> PromptLine returns "" (Enter)
	r := newResolver(t, "swe-planner", p)
	if _, err := r.Resolve(llmGroupCfg()); err == nil {
		t.Fatal("skipping the group should surface a missing-env error")
	}
	if len(p.asked) != 0 {
		t.Fatalf("no option should be prompted when the menu is skipped, asked=%v", p.asked)
	}
}

// Contract: exhausting the invalid-choice retries leaves the group unsatisfied
// without ever prompting for a secret.
func TestResolve_OneOfGroupMenuExhaustsInvalidChoices(t *testing.T) {
	p := &fakePrompter{interactive: true, choices: []string{"5", "6", "7"}}
	r := newResolver(t, "swe-planner", p)
	if _, err := r.Resolve(llmGroupCfg()); err == nil {
		t.Fatal("expected missing-env error after exhausting invalid choices")
	}
	if len(p.asked) != 0 {
		t.Fatalf("no secret should be prompted, asked=%v", p.asked)
	}
}

// Contract: picking an option but leaving its value blank leaves the group
// unsatisfied (nothing persisted).
func TestResolve_OneOfGroupSelectedButBlankLeavesGroupMissing(t *testing.T) {
	p := &fakePrompter{interactive: true, choices: []string{"1"}, answers: map[string]string{}}
	r := newResolver(t, "swe-planner", p)
	if _, err := r.Resolve(llmGroupCfg()); err == nil {
		t.Fatal("blank value after selection should leave the group unsatisfied")
	}
	if len(p.asked) != 1 || p.asked[0] != "ANTHROPIC_API_KEY" {
		t.Fatalf("selected option should have been prompted once: %v", p.asked)
	}
}

// Contract: a single-option group prompts for it directly — no numbered menu.
func TestResolve_OneOfGroupSingleOptionPromptsDirectly(t *testing.T) {
	cfg := UserEnvironmentConfig{RequireOneOf: []RequireOneOfGroup{{
		ID:          "solo",
		Description: "a token",
		Options:     []UserEnvironmentVar{{Name: "ONLY_KEY", Type: "secret"}},
	}}}
	p := &fakePrompter{interactive: true, answers: map[string]string{"ONLY_KEY": "v"}}
	r := newResolver(t, "swe-planner", p)
	got, err := r.Resolve(cfg)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got["ONLY_KEY"] != "v" {
		t.Fatalf("single-option group not resolved: %v", got)
	}
	if p.ci != 0 {
		t.Fatal("single-option group must not consult the selection menu")
	}
}

// Contract: user-facing group enumeration joins option names with "or", not "|".
func TestMissingEnvError_JoinsGroupOptionsWithOr(t *testing.T) {
	err := missingEnvError(nil, []RequireOneOfGroup{{
		Options: []UserEnvironmentVar{{Name: "A"}, {Name: "B"}},
	}})
	if !strings.Contains(err.Error(), "A or B") {
		t.Fatalf("group options should be joined with 'or': %q", err)
	}
	if strings.Contains(err.Error(), "|") {
		t.Fatalf("pipe separator should be gone: %q", err)
	}
}

func TestOrJoin(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want string
	}{
		{"none", nil, ""},
		{"one", []string{"1"}, "1"},
		{"two", []string{"1", "2"}, "1 or 2"},
		{"three", []string{"1", "2", "3"}, "1, 2, or 3"},
	}
	for _, c := range cases {
		if got := orJoin(c.in); got != c.want {
			t.Errorf("%s: orJoin(%v)=%q want %q", c.name, c.in, got, c.want)
		}
	}
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
