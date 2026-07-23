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

func TestAliasOrphanNamesHandlesEmptyState(t *testing.T) {
	if got := aliasOrphanNames(&State{}); len(got) != 0 {
		t.Fatalf("aliasOrphanNames(empty state) = %v, want no aliases", got)
	}
}

func TestReconcileAliasOrphansRemovesEmptyTargetStateAtomically(t *testing.T) {
	home := setupReconciliation(t, reconciliationState(map[string]InstalledTarget{}))
	root := filepath.Join(home, "skills")
	orphanDir := filepath.Join(root, legacyBuilder)
	if err := os.MkdirAll(orphanDir, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := reconcileAliasOrphans(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(orphanDir); !os.IsNotExist(err) {
		t.Fatalf("empty-target orphan directory remains: %v", err)
	}
	statePath := filepath.Join(root, ".state.json")
	if _, err := os.Lstat(statePath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("atomic state temporary remains: %v", err)
	}
	data, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"agentfield"`) || strings.Contains(string(data), legacyBuilder) {
		t.Fatalf("serialized state did not atomically retain only current entries: %s", data)
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
	if _, err := os.Lstat(unknownDir); err != nil {
		t.Fatalf("unknown removed-skill integration was affected: %v", err)
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

func TestReconcileAliasOrphansSortsTargetsBeforeCleanup(t *testing.T) {
	latePath := filepath.Join(t.TempDir(), "late-recorded-path")
	if err := os.WriteFile(latePath, []byte("legacy integration"), 0o644); err != nil {
		t.Fatal(err)
	}
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
		"z-last":  {Method: "symlink", Path: latePath},
		"a-first": {Method: "manual", Path: filepath.Join(t.TempDir(), "manual")},
	}))

	err := reconcileAliasOrphans()
	if err == nil || !strings.Contains(err.Error(), "a-first") {
		t.Fatalf("error = %v, want first sorted target", err)
	}
	if _, err := os.Lstat(latePath); err != nil {
		t.Fatalf("later target was cleaned before sorted failure: %v", err)
	}
	state, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := state.Skills[legacyBuilder]; !ok {
		t.Fatal("sorted target failure removed retryable state")
	}
}

func TestReconcileAliasOrphansMissingPathsAndMarkerBlocksAreIdempotent(t *testing.T) {
	home := t.TempDir()
	markerPath := filepath.Join(home, "codex.md")
	canonicalBlock := "<!-- agentfield-skill:agentfield v0.6.0 -->\nread: /tmp/agentfield\n<!-- /agentfield-skill:agentfield -->\n"
	if err := os.WriteFile(markerPath, []byte("prefix\n"+canonicalBlock+"suffix\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	missingPath := filepath.Join(home, "does-not-exist")
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
		"codex":  {Method: "marker-block", Path: markerPath, Version: "0.1.0"},
		"claude": {Method: "symlink", Path: missingPath},
	}))

	for attempt := 1; attempt <= 2; attempt++ {
		if err := reconcileAliasOrphans(); err != nil {
			t.Fatalf("attempt %d: %v", attempt, err)
		}
	}
	data, err := os.ReadFile(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(data), "prefix\n"+canonicalBlock+"suffix\n"; got != want {
		t.Fatalf("unrelated marker content changed:\n got %q\nwant %q", got, want)
	}
}

func TestReconcileAliasOrphansMarkerReadFailureRetainsStateAndNamesTarget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "codex.md")
	setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
		"codex": {Method: "marker-block", Path: path},
	}))
	oldRead := reconcileReadFile
	reconcileReadFile = func(string) ([]byte, error) { return nil, errors.New("forced read failure") }
	t.Cleanup(func() { reconcileReadFile = oldRead })

	err := reconcileAliasOrphans()
	if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "codex") {
		t.Fatalf("error = %v", err)
	}
	state, loadErr := LoadState()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if _, ok := state.Skills[legacyBuilder]; !ok {
		t.Fatal("marker read failure removed retryable orphan state")
	}
}

func TestReconcileAliasOrphansMarkerRewriteFailuresRetainStateAndNameTarget(t *testing.T) {
	for _, failure := range []struct {
		name string
		set  func() func()
	}{
		{
			name: "write",
			set: func() func() {
				old := reconcileWriteFile
				reconcileWriteFile = func(string, []byte, os.FileMode) error {
					return errors.New("forced write failure")
				}
				return func() { reconcileWriteFile = old }
			},
		},
		{
			name: "rename",
			set: func() func() {
				old := reconcileRename
				reconcileRename = func(string, string) error { return errors.New("forced rename failure") }
				return func() { reconcileRename = old }
			},
		},
	} {
		t.Run(failure.name, func(t *testing.T) {
			home := t.TempDir()
			path := filepath.Join(home, "codex.md")
			data := []byte("before\n<!-- agentfield-skill:" + legacyBuilder + " v0.5.0 -->\nold\n<!-- /agentfield-skill:" + legacyBuilder + " -->\nafter\n")
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatal(err)
			}
			setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
				"codex": {Method: "marker-block", Path: path},
			}))
			restore := failure.set()
			t.Cleanup(restore)

			err := reconcileAliasOrphans()
			if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "codex") {
				t.Fatalf("error = %v, want orphan and target", err)
			}
			state, loadErr := LoadState()
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			if _, ok := state.Skills[legacyBuilder]; !ok {
				t.Fatal("marker rewrite failure removed retryable orphan state")
			}
		})
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

func TestInstallStopsBeforeCanonicalMutationWhenRecordedPathCleanupFails(t *testing.T) {
	for _, failure := range []struct {
		name      string
		makePath  func(t *testing.T) string
		intercept func() func()
	}{
		{
			name:     "lstat",
			makePath: func(t *testing.T) string { return filepath.Join(t.TempDir(), "alias") },
			intercept: func() func() {
				old := reconcileLstat
				reconcileLstat = func(string) (os.FileInfo, error) { return nil, errors.New("forced lstat failure") }
				return func() { reconcileLstat = old }
			},
		},
		{
			name: "remove",
			makePath: func(t *testing.T) string {
				path := filepath.Join(t.TempDir(), "alias")
				if err := os.WriteFile(path, []byte("legacy"), 0o644); err != nil {
					t.Fatal(err)
				}
				return path
			},
			intercept: func() func() {
				old := reconcileRemove
				reconcileRemove = func(string) error { return errors.New("forced remove failure") }
				return func() { reconcileRemove = old }
			},
		},
		{
			name: "remove-all",
			makePath: func(t *testing.T) string {
				path := filepath.Join(t.TempDir(), "alias")
				if err := os.MkdirAll(path, 0o755); err != nil {
					t.Fatal(err)
				}
				return path
			},
			intercept: func() func() {
				old := reconcileRemoveAll
				reconcileRemoveAll = func(string) error { return errors.New("forced remove-all failure") }
				return func() { reconcileRemoveAll = old }
			},
		},
	} {
		t.Run(failure.name, func(t *testing.T) {
			path := failure.makePath(t)
			setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
				"claude": {Method: "symlink", Path: path},
			}))
			restore := failure.intercept()
			t.Cleanup(restore)

			_, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}, Force: true})
			if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "claude") {
				t.Fatalf("Install error = %v, want orphan and target", err)
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
		})
	}
}

func TestPublicOperationsStopBeforeCanonicalMutationWhenFilesystemCleanupFails(t *testing.T) {
	for _, op := range []struct {
		name string
		run  func() error
	}{
		{"install", func() error {
			_, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}, Force: true})
			return err
		}},
		{"install-all", func() error {
			_, err := InstallAll(InstallOptions{Targets: []string{"codex"}, Force: true})
			return err
		}},
		{"update", func() error {
			_, err := Update("agentfield")
			return err
		}},
	} {
		t.Run(op.name, func(t *testing.T) {
			aliasPath := filepath.Join(t.TempDir(), "legacy-integration")
			if err := os.WriteFile(aliasPath, []byte("legacy"), 0o644); err != nil {
				t.Fatal(err)
			}
			state := reconciliationState(map[string]InstalledTarget{
				"claude-code": {Method: "symlink", Path: aliasPath},
			})
			if op.name == "update" {
				state.Skills["agentfield"] = InstalledSkill{Targets: map[string]InstalledTarget{
					"codex": {Method: "marker-block", Path: filepath.Join(t.TempDir(), "codex.md")},
				}}
			}
			home := setupReconciliation(t, state)
			oldRemove := reconcileRemove
			reconcileRemove = func(string) error { return errors.New("forced filesystem remove failure") }
			t.Cleanup(func() { reconcileRemove = oldRemove })

			err := op.run()
			if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "claude-code") {
				t.Fatalf("%s error = %v, want orphan and target", op.name, err)
			}
			if _, err := os.Lstat(filepath.Join(home, "skills", "agentfield")); !os.IsNotExist(err) {
				t.Fatalf("%s mutated canonical skill before filesystem cleanup failure: %v", op.name, err)
			}
			got, err := LoadState()
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := got.Skills[legacyBuilder]; !ok {
				t.Fatalf("%s removed retryable orphan state after filesystem cleanup failure", op.name)
			}
		})
	}
}

func TestPublicOperationsStopBeforeCanonicalMutationOnUnsupportedOrphan(t *testing.T) {
	for _, op := range []struct {
		name string
		run  func() error
	}{
		{"install", func() error {
			_, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}, Force: true})
			return err
		}},
		{"install-all", func() error {
			_, err := InstallAll(InstallOptions{Targets: []string{"codex"}, Force: true})
			return err
		}},
		{"update", func() error {
			_, err := Update("agentfield")
			return err
		}},
	} {
		t.Run(op.name, func(t *testing.T) {
			state := reconciliationState(map[string]InstalledTarget{
				"manual-target": {Method: "manual", Path: filepath.Join(t.TempDir(), "legacy")},
			})
			if op.name == "update" {
				state.Skills["agentfield"] = InstalledSkill{Targets: map[string]InstalledTarget{
					"codex": {Method: "marker-block", Path: filepath.Join(t.TempDir(), "codex.md")},
				}}
			}
			home := setupReconciliation(t, state)
			statePath := filepath.Join(home, "skills", ".state.json")
			before, err := os.ReadFile(statePath)
			if err != nil {
				t.Fatal(err)
			}

			err = op.run()
			if err == nil || !strings.Contains(err.Error(), legacyBuilder) || !strings.Contains(err.Error(), "manual-target") {
				t.Fatalf("error = %v, want orphan and target", err)
			}
			if _, err := os.Lstat(filepath.Join(home, "skills", "agentfield")); !os.IsNotExist(err) {
				t.Fatalf("%s mutated canonical skill before cleanup failure: %v", op.name, err)
			}
			after, err := os.ReadFile(statePath)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(after, before) {
				t.Fatalf("%s changed retryable serialized state on cleanup failure", op.name)
			}
		})
	}
}

func TestUpdateReconcilesAliasOnlyLegacyInstallation(t *testing.T) {
	home := t.TempDir()
	aliasPath := filepath.Join(home, "legacy-claude")
	if err := os.MkdirAll(aliasPath, 0o755); err != nil {
		t.Fatal(err)
	}
	state := &State{Version: stateFileVersion, Skills: map[string]InstalledSkill{
		legacyBuilder: {Targets: map[string]InstalledTarget{
			"claude-code": {Method: "symlink", Path: aliasPath},
		}},
	}}
	setupReconciliation(t, state)

	_, err := Update("agentfield")
	if err == nil || !strings.Contains(err.Error(), "agentfield") || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("Update error = %v, want canonical not-installed error after cleanup", err)
	}
	if _, err := os.Lstat(aliasPath); !os.IsNotExist(err) {
		t.Fatalf("legacy alias integration remains: %v", err)
	}
	got, err := LoadState()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got.Skills[legacyBuilder]; ok {
		t.Fatal("legacy alias state remains after Update reconciliation")
	}
}

func TestRemoveRecordedPathHandlesFilesDirectoriesAndSymlinks(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"file", "directory", "symlink"} {
		path := filepath.Join(root, name)
		var err error
		switch name {
		case "file":
			err = os.WriteFile(path, []byte("x"), 0o644)
		case "directory":
			err = os.MkdirAll(filepath.Join(path, "child"), 0o755)
		case "symlink":
			target := filepath.Join(root, "symlink-target")
			if err = os.MkdirAll(target, 0o755); err == nil {
				err = os.Symlink(target, path)
			}
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
	home := setupReconciliation(t, reconciliationState(map[string]InstalledTarget{"claude": {Method: "symlink", Path: path}}))
	statePath := filepath.Join(home, "skills", ".state.json")
	before, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
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
	after, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatal("serialized state changed after save failure")
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
			for i, report := range reports {
				if err == nil && report.Skill.Name != Catalog[i].Name {
					return errors.New("reports are not in catalog order")
				}
			}
			return err
		}},
		{"update", func() error { _, err := Update("agentfield"); return err }},
	} {
		t.Run(op.name, func(t *testing.T) {
			reconcileSaves := 0
			oldSave := reconcileSaveState
			reconcileSaveState = func(state *State) error {
				reconcileSaves++
				return oldSave(state)
			}
			t.Cleanup(func() { reconcileSaveState = oldSave })
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
			if reconcileSaves != 1 {
				t.Fatalf("%s reconciled %d times, want once", op.name, reconcileSaves)
			}
		})
	}
	for _, op := range []struct {
		name string
		run  func() error
	}{
		{"install", func() error { _, err := Install(InstallOptions{SkillName: "agentfield", DryRun: true}); return err }},
		{"install-all", func() error { _, err := InstallAll(InstallOptions{DryRun: true}); return err }},
	} {
		t.Run("dry-run-"+op.name, func(t *testing.T) {
			home := t.TempDir()
			path := filepath.Join(home, "alias")
			markerPath := filepath.Join(home, "codex.md")
			if err := os.MkdirAll(path, 0o755); err != nil {
				t.Fatal(err)
			}
			markerData := []byte("before\n<!-- agentfield-skill:" + legacyBuilder + " v0.5.0 -->\nold\n<!-- /agentfield-skill:" + legacyBuilder + " -->\nafter\n")
			if err := os.WriteFile(markerPath, markerData, 0o644); err != nil {
				t.Fatal(err)
			}
			home = setupReconciliation(t, reconciliationState(map[string]InstalledTarget{
				"old":   {Method: "symlink", Path: path},
				"codex": {Method: "marker-block", Path: markerPath},
			}))
			root := filepath.Join(home, "skills")
			orphanDir := filepath.Join(root, legacyBuilder)
			if err := os.MkdirAll(orphanDir, 0o755); err != nil {
				t.Fatal(err)
			}
			statePath := filepath.Join(root, ".state.json")
			stateBefore, err := os.ReadFile(statePath)
			if err != nil {
				t.Fatal(err)
			}

			if err := op.run(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Lstat(path); err != nil {
				t.Fatalf("dry run changed orphan integration: %v", err)
			}
			if _, err := os.Lstat(orphanDir); err != nil {
				t.Fatalf("dry run changed orphan canonical directory: %v", err)
			}
			if got, err := os.ReadFile(markerPath); err != nil || !reflect.DeepEqual(got, markerData) {
				t.Fatalf("dry run changed marker file: %q, %v", got, err)
			}
			if got, err := os.ReadFile(statePath); err != nil || !reflect.DeepEqual(got, stateBefore) {
				t.Fatalf("dry run changed serialized state: %q, %v", got, err)
			}
		})
	}
}
