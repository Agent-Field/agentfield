package skillkit

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const legacyBuilder = "agentfield-multi-reasoner-builder"

func reconciliationState(targets map[string]InstalledTarget) *State {
	return &State{Version: stateFileVersion, Skills: map[string]InstalledSkill{
		"agentfield":      {CurrentVersion: "0.5.0", Targets: map[string]InstalledTarget{}},
		legacyBuilder:     {CurrentVersion: "0.5.0", Targets: targets},
		"removed-unknown": {CurrentVersion: "1.0.0", Targets: map[string]InstalledTarget{}},
	}}
}

func setupReconciliation(t *testing.T, state *State) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("USERPROFILE", home)
	if err := SaveState(state); err != nil {
		t.Fatal(err)
	}
	return home
}

func TestAliasOrphanNamesSelectsOnlyCurrentExactAliases(t *testing.T) {
	state := reconciliationState(nil)
	state.Skills["agentfield-use"] = InstalledSkill{}
	state.Skills["AGENTFIELD-MULTI-REASONER-BUILDER"] = InstalledSkill{}
	if got, want := aliasOrphanNames(state), []string{legacyBuilder}; !reflect.DeepEqual(got, want) {
		t.Fatalf("aliasOrphanNames() = %v, want %v", got, want)
	}
}

func TestReconcileAliasOrphansRemovesOnlyRecordedIntegrations(t *testing.T) {
	home := t.TempDir()
	claudeAlias := filepath.Join(home, "claude-alias")
	claudeCanonical := filepath.Join(home, "claude-canonical")
	if err := os.MkdirAll(claudeAlias, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(claudeCanonical, 0o755); err != nil {
		t.Fatal(err)
	}
	marker := func(name string) string {
		return "<!-- agentfield-skill:" + name + " v0.5.0 -->\nread: /tmp/" + name + "\n<!-- /agentfield-skill:" + name + " -->\n"
	}
	for _, file := range []string{"codex.md", "gemini.md", "opencode.md"} {
		if err := os.WriteFile(filepath.Join(home, file), []byte("prefix\n"+marker(legacyBuilder)+"\n"+marker("agentfield")+"suffix\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	unknownDir := filepath.Join(home, "unknown")
	if err := os.MkdirAll(unknownDir, 0o755); err != nil {
		t.Fatal(err)
	}
	state := reconciliationState(map[string]InstalledTarget{
		"claude-code": {Method: "symlink", Path: claudeAlias},
		"codex":       {Method: "marker-block", Path: filepath.Join(home, "codex.md")},
		"gemini":      {Method: "marker-block", Path: filepath.Join(home, "gemini.md")},
		"opencode":    {Method: "marker-block", Path: filepath.Join(home, "opencode.md")},
	})
	state.Skills["removed-unknown"] = InstalledSkill{Targets: map[string]InstalledTarget{"manual": {Method: "symlink", Path: unknownDir}}}
	root := setupReconciliation(t, state)
	if err := os.MkdirAll(filepath.Join(root, "skills", legacyBuilder), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := reconcileAliasOrphans(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(claudeAlias); !os.IsNotExist(err) {
		t.Fatalf("obsolete recorded path remains: %v", err)
	}
	if _, err := os.Lstat(claudeCanonical); err != nil {
		t.Fatalf("canonical path was affected: %v", err)
	}
	for _, file := range []string{"codex.md", "gemini.md", "opencode.md"} {
		data, err := os.ReadFile(filepath.Join(home, file))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), legacyBuilder) || !strings.Contains(string(data), "agentfield-skill:agentfield") || !strings.Contains(string(data), "prefix") || !strings.Contains(string(data), "suffix") {
			t.Fatalf("marker cleanup lost content in %s: %q", file, data)
		}
	}
	got, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got.Skills[legacyBuilder]; ok {
		t.Fatal("orphan state entry remains")
	}
	if _, ok := got.Skills["agentfield"]; !ok {
		t.Fatal("canonical state entry removed")
	}
	if _, ok := got.Skills["removed-unknown"]; !ok {
		t.Fatal("unknown state entry removed")
	}
	if err := reconcileAliasOrphans(); err != nil {
		t.Fatalf("second reconciliation: %v", err)
	}
}

func TestReconcileAliasOrphansFailureRetainsState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orphan")
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{"broken": {Method: "manual", Path: path}}))
	if err := reconcileAliasOrphans(); err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "broken") {
		t.Fatalf("error = %v", err)
	}
	state, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := state.Skills[legacyBuilder]; !ok {
		t.Fatal("state was not retained")
	}
}

func TestInstallStopsBeforeCanonicalMutationWhenRecordedCleanupFails(t *testing.T) {
	home := t.TempDir()
	path := filepath.Join(home, "alias")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{"claude": {Method: "symlink", Path: path}}))
	oldRemove := reconcileRemove
	reconcileRemove = func(string) error { return errors.New("forced remove failure") }
	t.Cleanup(func() { reconcileRemove = oldRemove })
	_, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}, Force: true})
	if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "claude") {
		t.Fatalf("Install error = %v", err)
	}
	root, rootErr := CanonicalRoot()
	if rootErr != nil {
		t.Fatal(rootErr)
	}
	if _, err := os.Lstat(filepath.Join(root, "agentfield")); !os.IsNotExist(err) {
		t.Fatalf("canonical mutation occurred before failed cleanup: %v", err)
	}
	state, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := state.Skills[legacyBuilder]; !ok {
		t.Fatal("failed cleanup removed retryable orphan state")
	}
}

func TestRemoveRecordedPathHandlesFilesAndDirectories(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"file", "directory"} {
		path := filepath.Join(root, name)
		var err error
		if name == "file" {
			err = os.WriteFile(path, []byte("x"), 0o644)
		} else {
			err = os.MkdirAll(filepath.Join(path, "child"), 0o755)
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := removeRecordedPath(path); err != nil {
			t.Fatalf("remove %s: %v", name, err)
		}
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("%s remains: %v", name, err)
		}
	}
}

func TestReconcileAliasOrphansSaveFailureLeavesSerializedState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orphan")
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{"claude": {Method: "symlink", Path: path}}))
	oldSave := reconcileSaveState
	reconcileSaveState = func(*State) error { return errors.New("forced save failure") }
	t.Cleanup(func() { reconcileSaveState = oldSave })
	if err := reconcileAliasOrphans(); err == nil || !strings.Contains(err.Error(), "save state") {
		t.Fatalf("error = %v", err)
	}
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatalf("filesystem cleanup did not occur: %v", err)
	}
	state, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := state.Skills[legacyBuilder]; !ok {
		t.Fatal("on-disk state changed after save failure")
	}
}

func TestPublicOperationsReconcileAndDryRunsDoNot(t *testing.T) {
	for _, op := range []struct {
		name string
		run  func() error
	}{
		{"install", func() error {
			_, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}, Force: true})
			return err
		}},
		{"install-all", func() error {
			reports, err := InstallAll(InstallOptions{Targets: []string{"codex"}, Force: true})
			if err == nil && len(reports) != len(Catalog) {
				return errors.New("wrong report count")
			}
			return err
		}},
		{"update", func() error { _, err := Update("agentfield"); return err }},
	} {
		t.Run(op.name, func(t *testing.T) {
			home := t.TempDir()
			aliasPath := filepath.Join(home, "alias")
			if err := os.MkdirAll(aliasPath, 0o755); err != nil {
				t.Fatal(err)
			}
			state := reconciliationState(map[string]InstalledTarget{"old": {Method: "symlink", Path: aliasPath, InstalledAt: time.Now()}})
			if op.name == "update" {
				state.Skills["agentfield"] = InstalledSkill{CurrentVersion: "0.1.0", Targets: map[string]InstalledTarget{"codex": {Method: "marker-block", Version: "0.1.0"}}}
			}
			setupReconciliation(t, state)
			if err := op.run(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Lstat(aliasPath); !os.IsNotExist(err) {
				t.Fatalf("orphan was not reconciled: %v", err)
			}
		})
	}
	for _, run := range []func() error{
		func() error { _, err := Install(InstallOptions{SkillName: "agentfield", DryRun: true}); return err },
		func() error { _, err := InstallAll(InstallOptions{DryRun: true}); return err },
	} {
		home := t.TempDir()
		path := filepath.Join(home, "alias")
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
		setupReconciliation(t, reconciliationState(map[string]InstalledTarget{"old": {Method: "symlink", Path: path}}))
		if err := run(); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("dry run changed orphan path: %v", err)
		}
		state, err := LoadState()
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := state.Skills[legacyBuilder]; !ok {
			t.Fatal("dry run changed state")
		}
	}
}
