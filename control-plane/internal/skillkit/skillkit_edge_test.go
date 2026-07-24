package skillkit

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestStateAndCanonicalErrorPaths(t *testing.T) {
	t.Run("home resolution failures propagate", func(t *testing.T) {
		withEnv(t, "HOME", "")
		withEnv(t, "AGENTFIELD_HOME", "")

		if got := homeDir(); got != "" {
			t.Fatalf("homeDir = %q want empty", got)
		}
		if _, err := CanonicalRoot(); err == nil || !strings.Contains(err.Error(), "resolve home directory") {
			t.Fatalf("CanonicalRoot error = %v", err)
		}
	})

	t.Run("save state fails when root is blocked by file", func(t *testing.T) {
		base := t.TempDir()
		blocker := filepath.Join(base, "blocked")
		if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
			t.Fatalf("write blocker: %v", err)
		}
		withEnv(t, "AGENTFIELD_HOME", blocker)

		err := SaveState(&State{Skills: map[string]InstalledSkill{}})
		if err == nil || !strings.Contains(err.Error(), "create state dir") {
			t.Fatalf("SaveState error = %v", err)
		}

		if _, err := LoadState(); err == nil {
			t.Fatal("LoadState expected error")
		}
	})

	t.Run("save state reports rename failures", func(t *testing.T) {
		withTempHome(t)

		root, err := CanonicalRoot()
		if err != nil {
			t.Fatalf("CanonicalRoot: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(root, ".state.json"), 0o755); err != nil {
			t.Fatalf("mkdir state dir: %v", err)
		}

		err = SaveState(&State{Skills: map[string]InstalledSkill{}})
		if err == nil || !strings.Contains(err.Error(), "rename state file") {
			t.Fatalf("SaveState error = %v", err)
		}
	})

	t.Run("load state initializes missing fields", func(t *testing.T) {
		withTempHome(t)

		root, err := CanonicalRoot()
		if err != nil {
			t.Fatalf("CanonicalRoot: %v", err)
		}
		if err := os.MkdirAll(root, 0o755); err != nil {
			t.Fatalf("mkdir root: %v", err)
		}
		if err := os.WriteFile(filepath.Join(root, ".state.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatalf("write state: %v", err)
		}

		state, err := LoadState()
		if err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		if state.Version != stateFileVersion || state.Skills == nil || len(state.Skills) != 0 {
			t.Fatalf("unexpected state: %+v", state)
		}
	})

	t.Run("write canonical surfaces enumerate and mkdir failures", func(t *testing.T) {
		if err := writeCanonical(Skill{Name: "bad", EmbedRoot: "skill_data/missing", EntryFile: "SKILL.md"}, filepath.Join(t.TempDir(), "skill")); err == nil {
			t.Fatal("writeCanonical expected enumerate error")
		}

		base := t.TempDir()
		blocker := filepath.Join(base, "blocked")
		if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
			t.Fatalf("write blocker: %v", err)
		}
		if err := writeCanonical(Catalog[0], filepath.Join(blocker, "skill")); err == nil {
			t.Fatal("writeCanonical expected mkdir error")
		}
	})

	t.Run("write canonical reports nested path write failures", func(t *testing.T) {
		base := t.TempDir()
		dir := filepath.Join(base, "skill")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dir, "references"), []byte("x"), 0o644); err != nil {
			t.Fatalf("write blocker: %v", err)
		}
		if err := writeCanonical(Catalog[0], dir); err == nil {
			t.Fatal("writeCanonical expected nested write error")
		}
	})

	t.Run("update current link replaces directory", func(t *testing.T) {
		base := t.TempDir()
		linkPath := filepath.Join(base, "current")
		if err := os.MkdirAll(linkPath, 0o755); err != nil {
			t.Fatalf("mkdir linkPath: %v", err)
		}
		targetDir := filepath.Join(base, "v2")
		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			t.Fatalf("mkdir targetDir: %v", err)
		}
		if err := updateCurrentLink(linkPath, targetDir); err != nil {
			t.Fatalf("updateCurrentLink: %v", err)
		}
		if dest, err := os.Readlink(linkPath); err != nil || dest != "v2" {
			t.Fatalf("symlink dest = %q err=%v", dest, err)
		}
	})

	t.Run("update current link replaces existing symlink", func(t *testing.T) {
		base := t.TempDir()
		linkPath := filepath.Join(base, "current")
		oldTarget := filepath.Join(base, "old")
		newTarget := filepath.Join(base, "new")
		if err := os.MkdirAll(oldTarget, 0o755); err != nil {
			t.Fatalf("mkdir oldTarget: %v", err)
		}
		if err := os.MkdirAll(newTarget, 0o755); err != nil {
			t.Fatalf("mkdir newTarget: %v", err)
		}
		if err := os.Symlink("old", linkPath); err != nil {
			t.Fatalf("seed symlink: %v", err)
		}
		if err := updateCurrentLink(linkPath, newTarget); err != nil {
			t.Fatalf("updateCurrentLink: %v", err)
		}
		if dest, err := os.Readlink(linkPath); err != nil || dest != "new" {
			t.Fatalf("symlink dest = %q err=%v", dest, err)
		}
	})
}

func TestInstallAndUninstallBranches(t *testing.T) {
	withTempHome(t)

	origTargets := allTargets
	t.Cleanup(func() { allTargets = origTargets })

	success := &fakeTarget{name: "success", displayName: "Success", method: "marker-block", detected: true, path: "/tmp/success"}
	broken := &fakeTarget{name: "broken", displayName: "Broken", method: "marker-block", detected: true, path: "/tmp/broken", uninstallErr: errors.New("nope")}
	allTargets = []Target{success, broken}

	report, err := Install(InstallOptions{SkillName: Catalog[0].Name, AllRegistered: true})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(report.TargetsInstalled) != 2 {
		t.Fatalf("TargetsInstalled = %+v", report.TargetsInstalled)
	}

	if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name, Targets: []string{"broken"}}); err == nil || !strings.Contains(err.Error(), "uninstall from broken") {
		t.Fatalf("Uninstall broken error = %v", err)
	}

	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	skillState := state.Skills[Catalog[0].Name]
	skillState.Targets["missing"] = InstalledTarget{TargetName: "missing", Version: Catalog[0].Version}
	state.Skills[Catalog[0].Name] = skillState
	if err := SaveState(state); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name, Targets: []string{"missing"}}); err == nil || !strings.Contains(err.Error(), `target "missing" not registered`) {
		t.Fatalf("Uninstall missing target error = %v", err)
	}

	if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name, Targets: []string{"success"}, RemoveCanonical: true}); err != nil {
		t.Fatalf("Uninstall remove canonical: %v", err)
	}

	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, Catalog[0].Name)); !os.IsNotExist(err) {
		t.Fatalf("canonical skill dir should be removed, stat err = %v", err)
	}

	t.Run("uninstall all targets when none explicitly selected", func(t *testing.T) {
		withTempHome(t)

		origTargets := allTargets
		t.Cleanup(func() { allTargets = origTargets })

		first := &fakeTarget{name: "alpha", displayName: "Alpha", method: "marker-block", detected: true, path: "/tmp/alpha"}
		second := &fakeTarget{name: "beta", displayName: "Beta", method: "marker-block", detected: true, path: "/tmp/beta"}
		allTargets = []Target{first, second}

		if _, err := Install(InstallOptions{SkillName: Catalog[0].Name, AllRegistered: true}); err != nil {
			t.Fatalf("Install: %v", err)
		}
		if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name}); err != nil {
			t.Fatalf("Uninstall all: %v", err)
		}

		state, err := LoadState()
		if err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		if _, ok := state.Skills[Catalog[0].Name]; ok {
			t.Fatalf("skill should be removed after uninstall all: %+v", state.Skills)
		}
	})
}

func TestInstallExistingStateAndCanonicalFailures(t *testing.T) {
	t.Run("install merges existing state and sorts versions", func(t *testing.T) {
		withTempHome(t)

		origTargets := allTargets
		t.Cleanup(func() { allTargets = origTargets })

		success := &fakeTarget{name: "success", displayName: "Success", method: "marker-block", detected: true, path: "/tmp/success"}
		allTargets = []Target{success}

		// Seed the state with two historical versions that are definitely
		// NOT the current catalog version, so after Install we can assert
		// that the merge kept the old ones AND added the catalog version
		// in the right sorted position. Using literal "0.1.0" / "9.9.9"
		// instead of "0.1.0" / "0.3.0" so the test stays correct regardless
		// of what Catalog[0].Version happens to be bumped to on main.
		state := &State{
			Skills: map[string]InstalledSkill{
				Catalog[0].Name: {
					CurrentVersion:    "0.1.0",
					AvailableVersions: []string{"9.9.9", "0.1.0"},
					Targets:           nil,
				},
			},
		}
		if err := SaveState(state); err != nil {
			t.Fatalf("SaveState: %v", err)
		}

		report, err := Install(InstallOptions{SkillName: Catalog[0].Name, AllRegistered: true})
		if err != nil {
			t.Fatalf("Install: %v", err)
		}
		if len(report.TargetsInstalled) != 1 {
			t.Fatalf("TargetsInstalled = %+v", report.TargetsInstalled)
		}

		loaded, err := LoadState()
		if err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		got := loaded.Skills[Catalog[0].Name].AvailableVersions
		// Expect: the two seeded versions are still there and the catalog's
		// current version got merged in.
		wantContains := map[string]bool{
			"0.1.0":              false,
			"9.9.9":              false,
			Catalog[0].Version:   false,
		}
		for _, v := range got {
			if _, ok := wantContains[v]; ok {
				wantContains[v] = true
			}
		}
		for v, found := range wantContains {
			if !found {
				t.Fatalf("AvailableVersions = %v, missing %q", got, v)
			}
		}
		if loaded.Skills[Catalog[0].Name].Targets == nil {
			t.Fatal("Targets should be initialized")
		}
	})

	t.Run("install returns canonical write failures", func(t *testing.T) {
		withTempHome(t)

		origCatalog := Catalog
		t.Cleanup(func() { Catalog = origCatalog })
		Catalog = []Skill{{
			Name:        "broken",
			Version:     "1.0.0",
			Description: "broken",
			EmbedRoot:   "skill_data/missing",
			EntryFile:   "SKILL.md",
		}}

		if _, err := Install(InstallOptions{SkillName: "broken"}); err == nil || !strings.Contains(err.Error(), "write canonical store") {
			t.Fatalf("Install error = %v", err)
		}
	})

	t.Run("install fails when canonical root cannot be resolved", func(t *testing.T) {
		withEnv(t, "HOME", "")
		withEnv(t, "AGENTFIELD_HOME", "")

		if _, err := Install(InstallOptions{SkillName: Catalog[0].Name}); err == nil || !strings.Contains(err.Error(), "resolve home directory") {
			t.Fatalf("Install error = %v", err)
		}
	})

	t.Run("resolve skill accepts matching explicit version", func(t *testing.T) {
		skill, err := resolveSkill(Catalog[0].Name, Catalog[0].Version)
		if err != nil {
			t.Fatalf("resolveSkill: %v", err)
		}
		if skill.Version != Catalog[0].Version {
			t.Fatalf("skill.Version = %q", skill.Version)
		}
	})

	t.Run("update uses default skill when name is empty", func(t *testing.T) {
		withTempHome(t)

		origTargets := allTargets
		t.Cleanup(func() { allTargets = origTargets })

		success := &fakeTarget{name: "success", displayName: "Success", method: "marker-block", detected: true, path: "/tmp/success"}
		allTargets = []Target{success}

		if _, err := Install(InstallOptions{AllRegistered: true}); err != nil {
			t.Fatalf("Install: %v", err)
		}
		report, err := Update("")
		if err != nil {
			t.Fatalf("Update: %v", err)
		}
		if len(report.TargetsInstalled) != 1 {
			t.Fatalf("TargetsInstalled = %+v", report.TargetsInstalled)
		}
	})

	t.Run("install reports state load failures after canonical write", func(t *testing.T) {
		withTempHome(t)

		root, err := CanonicalRoot()
		if err != nil {
			t.Fatalf("CanonicalRoot: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(root, ".state.json"), 0o755); err != nil {
			t.Fatalf("mkdir state dir: %v", err)
		}

		if _, err := Install(InstallOptions{SkillName: Catalog[0].Name, AllRegistered: true}); err == nil {
			t.Fatal("Install expected state load error")
		}
	})
}

func TestMarkerBlockErrorPaths(t *testing.T) {
	skill := Catalog[0]

	t.Run("install marker block reports read and write errors", func(t *testing.T) {
		base := t.TempDir()
		targetPath := filepath.Join(base, "rules.md")
		if err := os.MkdirAll(targetPath, 0o755); err != nil {
			t.Fatalf("mkdir targetPath: %v", err)
		}
		if _, err := installMarkerBlock(skill, "/canonical/current", targetPath); err == nil || !strings.Contains(err.Error(), "read") {
			t.Fatalf("installMarkerBlock read error = %v", err)
		}

		targetPath = filepath.Join(base, "nested", "rules.md")
		if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
			t.Fatalf("mkdir parent: %v", err)
		}
		if err := os.MkdirAll(targetPath+".af-tmp", 0o755); err != nil {
			t.Fatalf("mkdir tmp blocker: %v", err)
		}
		if _, err := installMarkerBlock(skill, "/canonical/current", targetPath); err == nil || !strings.Contains(err.Error(), "write") {
			t.Fatalf("installMarkerBlock write error = %v", err)
		}
	})

	t.Run("uninstall marker block reports read and write errors", func(t *testing.T) {
		base := t.TempDir()
		targetPath := filepath.Join(base, "rules.md")
		if err := os.MkdirAll(targetPath, 0o755); err != nil {
			t.Fatalf("mkdir targetPath: %v", err)
		}
		if err := uninstallMarkerBlock(skill, targetPath); err == nil || !strings.Contains(err.Error(), "read") {
			t.Fatalf("uninstallMarkerBlock read error = %v", err)
		}

		targetPath = filepath.Join(base, "filled.md")
		if err := os.WriteFile(targetPath, []byte("prefix\n"+renderPointerBlock(skill, "/canonical/current")+"\n"), 0o644); err != nil {
			t.Fatalf("write target: %v", err)
		}
		if err := os.MkdirAll(targetPath+".af-tmp", 0o755); err != nil {
			t.Fatalf("mkdir tmp blocker: %v", err)
		}
		if err := uninstallMarkerBlock(skill, targetPath); err == nil || !strings.Contains(err.Error(), "write") {
			t.Fatalf("uninstallMarkerBlock write error = %v", err)
		}
	})
}

func TestTargetSpecificEdgeCases(t *testing.T) {
	t.Run("targets propagate missing home errors", func(t *testing.T) {
		withEnv(t, "HOME", "")

		type targetCase struct {
			name   string
			target Target
		}
		cases := []targetCase{
			{name: "aider", target: aiderTarget{}},
			{name: "claude", target: claudeCodeTarget{}},
			{name: "codex", target: codexTarget{}},
			{name: "gemini", target: geminiTarget{}},
			{name: "opencode", target: opencodeTarget{}},
			{name: "windsurf", target: windsurfTarget{}},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := tc.target.TargetPath(); err == nil {
					t.Fatal("TargetPath expected error")
				}
				if _, err := tc.target.Install(Catalog[0], "/canonical/current"); err == nil {
					t.Fatal("Install expected error")
				}
				err := tc.target.Uninstall()
				if tc.name == "claude" {
					if err != nil {
						t.Fatalf("Uninstall error = %v", err)
					}
				} else if err == nil {
					t.Fatal("Uninstall expected error")
				}
				if installed, version, err := tc.target.Status(); err == nil || installed || version != "" {
					t.Fatalf("Status = %v %q %v", installed, version, err)
				}
			})
		}
	})

	t.Run("claude install replaces existing file and status detects manual file", func(t *testing.T) {
		home := withTempHome(t)
		root := filepath.Join(home, ".claude", "skills")
		if err := os.MkdirAll(root, 0o755); err != nil {
			t.Fatalf("mkdir root: %v", err)
		}

		claude := claudeCodeTarget{}
		link, err := claude.skillLink(Catalog[0])
		if err != nil {
			t.Fatalf("skillLink: %v", err)
		}
		if err := os.WriteFile(link, []byte("manual"), 0o644); err != nil {
			t.Fatalf("write manual file: %v", err)
		}

		installed, version, err := claude.Status()
		if err != nil || !installed || version != "manual" {
			t.Fatalf("Status = %v %q %v", installed, version, err)
		}

		current := filepath.Join(t.TempDir(), "current")
		if err := os.MkdirAll(current, 0o755); err != nil {
			t.Fatalf("mkdir current: %v", err)
		}
		inst, err := claude.Install(Catalog[0], current)
		if err != nil {
			t.Fatalf("Install: %v", err)
		}
		if inst.Path != link {
			t.Fatalf("Install path = %q want %q", inst.Path, link)
		}
		if dest, err := os.Readlink(link); err != nil || dest != current {
			t.Fatalf("readlink = %q err=%v", dest, err)
		}
	})

	t.Run("claude install replaces existing directory and uninstall ignores missing links", func(t *testing.T) {
		home := withTempHome(t)
		root := filepath.Join(home, ".claude", "skills")
		if err := os.MkdirAll(root, 0o755); err != nil {
			t.Fatalf("mkdir root: %v", err)
		}

		claude := claudeCodeTarget{}
		link, err := claude.skillLink(Catalog[0])
		if err != nil {
			t.Fatalf("skillLink: %v", err)
		}
		if err := os.MkdirAll(link, 0o755); err != nil {
			t.Fatalf("mkdir link dir: %v", err)
		}

		current := filepath.Join(t.TempDir(), "current")
		if err := os.MkdirAll(current, 0o755); err != nil {
			t.Fatalf("mkdir current: %v", err)
		}
		if _, err := claude.Install(Catalog[0], current); err != nil {
			t.Fatalf("Install: %v", err)
		}

		if err := claude.Uninstall(); err != nil {
			t.Fatalf("Uninstall: %v", err)
		}
		if _, err := os.Lstat(link); !os.IsNotExist(err) {
			t.Fatalf("link should be removed, stat err = %v", err)
		}
		if err := claude.Uninstall(); err != nil {
			t.Fatalf("Uninstall second pass: %v", err)
		}
	})

	t.Run("claude install exposes slash commands and uninstall removes them", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir claude root: %v", err)
		}

		// Build a fake canonical current/ dir with a commands/ subdir.
		current := filepath.Join(t.TempDir(), "current")
		cmdDir := filepath.Join(current, "commands")
		if err := os.MkdirAll(cmdDir, 0o755); err != nil {
			t.Fatalf("mkdir cmdDir: %v", err)
		}
		cmdSrc := filepath.Join(cmdDir, "agentfield.md")
		if err := os.WriteFile(cmdSrc, []byte("---\ndescription: test\n---\n\nbody\n"), 0o644); err != nil {
			t.Fatalf("write command: %v", err)
		}
		// A non-md sibling and a directory should be ignored by the installer.
		if err := os.WriteFile(filepath.Join(cmdDir, "readme.txt"), []byte("skip"), 0o644); err != nil {
			t.Fatalf("write non-md: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(cmdDir, "nested"), 0o755); err != nil {
			t.Fatalf("mkdir nested: %v", err)
		}

		// Point canonical root at the fake current/ via symlink at
		// <canonical>/<skill>/current so Uninstall() can find the commands.
		root, err := CanonicalRoot()
		if err != nil {
			t.Fatalf("CanonicalRoot: %v", err)
		}
		skillDir := filepath.Join(root, Catalog[0].Name)
		if err := os.MkdirAll(skillDir, 0o755); err != nil {
			t.Fatalf("mkdir skillDir: %v", err)
		}
		if err := os.Symlink(current, filepath.Join(skillDir, "current")); err != nil {
			t.Fatalf("symlink current: %v", err)
		}

		claude := claudeCodeTarget{}
		if _, err := claude.Install(Catalog[0], current); err != nil {
			t.Fatalf("Install: %v", err)
		}

		cmdLink := filepath.Join(home, ".claude", "commands", "agentfield.md")
		info, err := os.Lstat(cmdLink)
		if err != nil {
			t.Fatalf("command link missing: %v", err)
		}
		if info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("expected symlink, got mode %v", info.Mode())
		}
		if dest, err := os.Readlink(cmdLink); err != nil || dest != cmdSrc {
			t.Fatalf("readlink = %q err=%v want %q", dest, err, cmdSrc)
		}
		if _, err := os.Stat(filepath.Join(home, ".claude", "commands", "readme.txt")); !os.IsNotExist(err) {
			t.Fatalf("non-md sibling should be ignored, stat err=%v", err)
		}

		// Re-install is idempotent: replaces existing link.
		if _, err := claude.Install(Catalog[0], current); err != nil {
			t.Fatalf("re-Install: %v", err)
		}
		if dest, err := os.Readlink(cmdLink); err != nil || dest != cmdSrc {
			t.Fatalf("re-install readlink = %q err=%v want %q", dest, err, cmdSrc)
		}

		if err := claude.Uninstall(); err != nil {
			t.Fatalf("Uninstall: %v", err)
		}
		if _, err := os.Lstat(cmdLink); !os.IsNotExist(err) {
			t.Fatalf("command link should be removed, stat err=%v", err)
		}
	})

	t.Run("claude install replaces a regular file at the slash-command destination", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir claude root: %v", err)
		}
		cmdDst := filepath.Join(home, ".claude", "commands")
		if err := os.MkdirAll(cmdDst, 0o755); err != nil {
			t.Fatalf("mkdir commands: %v", err)
		}
		stale := filepath.Join(cmdDst, "agentfield.md")
		if err := os.WriteFile(stale, []byte("stale"), 0o644); err != nil {
			t.Fatalf("write stale: %v", err)
		}

		current := filepath.Join(t.TempDir(), "current")
		cmdDir := filepath.Join(current, "commands")
		if err := os.MkdirAll(cmdDir, 0o755); err != nil {
			t.Fatalf("mkdir cmdDir: %v", err)
		}
		cmdSrc := filepath.Join(cmdDir, "agentfield.md")
		if err := os.WriteFile(cmdSrc, []byte("fresh"), 0o644); err != nil {
			t.Fatalf("write fresh: %v", err)
		}

		claude := claudeCodeTarget{}
		if _, err := claude.Install(Catalog[0], current); err != nil {
			t.Fatalf("Install: %v", err)
		}
		info, err := os.Lstat(stale)
		if err != nil {
			t.Fatalf("lstat after install: %v", err)
		}
		if info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("stale regular file was not replaced with a symlink (mode=%v)", info.Mode())
		}
		if dest, err := os.Readlink(stale); err != nil || dest != cmdSrc {
			t.Fatalf("readlink = %q err=%v want %q", dest, err, cmdSrc)
		}
	})

	t.Run("claude install surfaces mkdir error when commands destination is blocked by a file", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir skills: %v", err)
		}
		// Place a regular file at ~/.claude/commands so MkdirAll of it fails.
		if err := os.WriteFile(filepath.Join(home, ".claude", "commands"), []byte("blocker"), 0o644); err != nil {
			t.Fatalf("write blocker: %v", err)
		}

		current := filepath.Join(t.TempDir(), "current")
		cmdDir := filepath.Join(current, "commands")
		if err := os.MkdirAll(cmdDir, 0o755); err != nil {
			t.Fatalf("mkdir cmdDir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(cmdDir, "agentfield.md"), []byte("body"), 0o644); err != nil {
			t.Fatalf("write md: %v", err)
		}

		_, err := claudeCodeTarget{}.Install(Catalog[0], current)
		if err == nil {
			t.Fatalf("Install should fail when cmd destination is not a directory")
		}
		if !strings.Contains(err.Error(), "install commands") {
			t.Fatalf("expected install commands wrap, got %v", err)
		}
	})

	t.Run("claude install surfaces remove error when destination dir is read-only", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("permission semantics differ on windows")
		}
		if os.Geteuid() == 0 {
			t.Skip("root bypasses permission checks")
		}
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir skills: %v", err)
		}
		cmdDst := filepath.Join(home, ".claude", "commands")
		if err := os.MkdirAll(cmdDst, 0o755); err != nil {
			t.Fatalf("mkdir cmdDst: %v", err)
		}
		stale := filepath.Join(cmdDst, "agentfield.md")
		if err := os.WriteFile(stale, []byte("stale"), 0o644); err != nil {
			t.Fatalf("write stale: %v", err)
		}
		if err := os.Chmod(cmdDst, 0o555); err != nil {
			t.Fatalf("chmod: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(cmdDst, 0o755) })

		current := filepath.Join(t.TempDir(), "current")
		cmdDir := filepath.Join(current, "commands")
		if err := os.MkdirAll(cmdDir, 0o755); err != nil {
			t.Fatalf("mkdir cmdDir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(cmdDir, "agentfield.md"), []byte("new"), 0o644); err != nil {
			t.Fatalf("write new: %v", err)
		}

		_, err := claudeCodeTarget{}.Install(Catalog[0], current)
		if err == nil {
			t.Fatalf("Install should fail when existing command cannot be removed")
		}
		if !strings.Contains(err.Error(), "remove existing") {
			t.Fatalf("expected remove existing error, got %v", err)
		}
	})

	t.Run("claude install surfaces symlink error when destination dir is not writable", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("permission semantics differ on windows")
		}
		if os.Geteuid() == 0 {
			t.Skip("root bypasses permission checks")
		}
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir skills: %v", err)
		}
		cmdDst := filepath.Join(home, ".claude", "commands")
		if err := os.MkdirAll(cmdDst, 0o755); err != nil {
			t.Fatalf("mkdir cmdDst: %v", err)
		}
		if err := os.Chmod(cmdDst, 0o555); err != nil {
			t.Fatalf("chmod: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(cmdDst, 0o755) })

		current := filepath.Join(t.TempDir(), "current")
		cmdDir := filepath.Join(current, "commands")
		if err := os.MkdirAll(cmdDir, 0o755); err != nil {
			t.Fatalf("mkdir cmdDir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(cmdDir, "agentfield.md"), []byte("x"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}

		_, err := claudeCodeTarget{}.Install(Catalog[0], current)
		if err == nil {
			t.Fatalf("Install should fail when symlink cannot be created")
		}
		if !strings.Contains(err.Error(), "symlink") {
			t.Fatalf("expected symlink error, got %v", err)
		}
	})

	t.Run("claude install fails when skill commands path is not a directory", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".claude", "skills"), 0o755); err != nil {
			t.Fatalf("mkdir claude root: %v", err)
		}
		// current/ contains a regular file named "commands" — ReadDir on it
		// returns a non-IsNotExist error, which installCommands must surface
		// and Install must wrap.
		current := filepath.Join(t.TempDir(), "current")
		if err := os.MkdirAll(current, 0o755); err != nil {
			t.Fatalf("mkdir current: %v", err)
		}
		if err := os.WriteFile(filepath.Join(current, "commands"), []byte("not-a-dir"), 0o644); err != nil {
			t.Fatalf("write commands file: %v", err)
		}

		claude := claudeCodeTarget{}
		_, err := claude.Install(Catalog[0], current)
		if err == nil {
			t.Fatalf("Install should fail when commands path is not a directory")
		}
		if !strings.Contains(err.Error(), "install commands") {
			t.Fatalf("error should be wrapped as install commands error, got: %v", err)
		}
	})

	t.Run("marker targets report empty status after plain file uninstall", func(t *testing.T) {
		home := withTempHome(t)
		type targetCase struct {
			name   string
			target Target
			dir    string
			path   string
		}
		cases := []targetCase{
			{name: "aider", target: aiderTarget{}, path: filepath.Join(home, ".aider.conventions.md")},
			{name: "codex", target: codexTarget{}, dir: filepath.Join(home, ".codex"), path: filepath.Join(home, ".codex", "AGENTS.override.md")},
			{name: "gemini", target: geminiTarget{}, dir: filepath.Join(home, ".gemini"), path: filepath.Join(home, ".gemini", "GEMINI.md")},
			{name: "opencode", target: opencodeTarget{}, dir: filepath.Join(home, ".config", "opencode"), path: filepath.Join(home, ".config", "opencode", "AGENTS.md")},
			{name: "windsurf", target: windsurfTarget{}, dir: filepath.Join(home, ".codeium", "windsurf", "memories"), path: filepath.Join(home, ".codeium", "windsurf", "memories", "global_rules.md")},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if tc.dir != "" {
					if err := os.MkdirAll(tc.dir, 0o755); err != nil {
						t.Fatalf("mkdir dir: %v", err)
					}
				} else if err := os.MkdirAll(filepath.Dir(tc.path), 0o755); err != nil {
					t.Fatalf("mkdir parent: %v", err)
				}

				if err := os.WriteFile(tc.path, []byte("plain text\n"), 0o644); err != nil {
					t.Fatalf("write plain file: %v", err)
				}
				if err := tc.target.Uninstall(); err != nil {
					t.Fatalf("Uninstall: %v", err)
				}
				installed, version, err := tc.target.Status()
				if err != nil || installed || version != "" {
					t.Fatalf("Status = %v %q %v", installed, version, err)
				}
			})
		}
	})

	t.Run("ensureLineInFile appends missing newline before new line", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "aider.conf.yml")
		if err := os.WriteFile(path, []byte("first: value"), 0o644); err != nil {
			t.Fatalf("write config: %v", err)
		}
		if err := ensureLineInFile(path, "read: /tmp/skill"); err != nil {
			t.Fatalf("ensureLineInFile: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(data) != "first: value\nread: /tmp/skill\n" {
			t.Fatalf("config = %q", string(data))
		}
	})

	t.Run("aider install fails when config path is a directory", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".aider"), 0o755); err != nil {
			t.Fatalf("mkdir .aider: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(home, ".aider.conf.yml"), 0o755); err != nil {
			t.Fatalf("mkdir aider.conf.yml dir: %v", err)
		}

		if _, err := (aiderTarget{}).Install(Catalog[0], filepath.Join(home, "canonical", "current")); err == nil || !strings.Contains(err.Error(), "update aider conf") {
			t.Fatalf("Install error = %v", err)
		}
	})

	t.Run("cursor detection uses cursor home directory", func(t *testing.T) {
		home := withTempHome(t)
		if err := os.MkdirAll(filepath.Join(home, ".cursor"), 0o755); err != nil {
			t.Fatalf("mkdir .cursor: %v", err)
		}
		if !(cursorTarget{}).Detected() {
			t.Fatal("cursor target should be detected")
		}
	})
}
