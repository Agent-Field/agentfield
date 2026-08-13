package skillkit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// stubTarget is a Target that exists only to report a Method() for the
// validity check.
type stubTarget struct {
	name   string
	method string
}

func (s stubTarget) Name() string                { return s.name }
func (s stubTarget) DisplayName() string         { return s.name }
func (s stubTarget) Detected() bool              { return true }
func (s stubTarget) Method() string              { return s.method }
func (s stubTarget) TargetPath() (string, error) { return "", nil }
func (s stubTarget) Install(Skill, string) (InstalledTarget, error) {
	return InstalledTarget{}, nil
}
func (s stubTarget) Uninstall() error              { return nil }
func (s stubTarget) Status() (bool, string, error) { return false, "", nil }

func TestInstalledArtifactValidSymlinkCases(t *testing.T) {
	home := withTempHome(t)
	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	skill := Catalog[0]
	version := filepath.Join(root, skill.Name, skill.Version)
	if err := os.MkdirAll(version, 0o755); err != nil {
		t.Fatalf("mkdir version: %v", err)
	}
	current := filepath.Join(root, skill.Name, "current")
	if err := os.Symlink(version, current); err != nil {
		t.Fatalf("symlink current: %v", err)
	}
	links := filepath.Join(home, "links")
	if err := os.MkdirAll(links, 0o755); err != nil {
		t.Fatalf("mkdir links: %v", err)
	}
	link := func(name, dest string) string {
		path := filepath.Join(links, name)
		if err := os.Symlink(dest, path); err != nil {
			t.Fatalf("symlink %s: %v", name, err)
		}
		return path
	}

	otherSkill := filepath.Join(root, "some-other-skill", "current")
	if err := os.MkdirAll(otherSkill, 0o755); err != nil {
		t.Fatalf("mkdir other skill: %v", err)
	}
	regular := filepath.Join(links, "regular-dir")
	if err := os.MkdirAll(regular, 0o755); err != nil {
		t.Fatalf("mkdir regular: %v", err)
	}

	target := stubTarget{name: "stub", method: "symlink"}
	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "link into the skill's store", path: link("good", current), want: true},
		{name: "empty path", path: "", want: false},
		{name: "missing path", path: filepath.Join(links, "nope"), want: false},
		{name: "regular directory instead of a link", path: regular, want: false},
		{name: "dangling link inside the store", path: link("dangling", filepath.Join(root, skill.Name, "9.9.9")), want: false},
		{name: "link to another skill's store", path: link("foreign", otherSkill), want: false},
		{name: "link outside the canonical store", path: link("escaped", home), want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorded := InstalledTarget{Method: "symlink", Path: tc.path, Version: skill.Version}
			if got := installedArtifactValid(skill, target, recorded); got != tc.want {
				t.Fatalf("installedArtifactValid(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

func TestInstalledArtifactValidMarkerBlockCases(t *testing.T) {
	base := t.TempDir()
	skill := Catalog[0]
	target := stubTarget{name: "stub", method: "marker-block"}

	livePointer := filepath.Join(base, "canonical", "current")
	if err := os.MkdirAll(livePointer, 0o755); err != nil {
		t.Fatalf("mkdir pointer dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(livePointer, skill.EntryFile), []byte("# skill"), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}

	write := func(name, content string) string {
		path := filepath.Join(base, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
		return path
	}

	valid := write("valid.md", "notes\n"+renderPointerBlock(skill, livePointer)+"\n")
	deletedPointer := write("deleted.md", "notes\n"+renderPointerBlock(skill, filepath.Join(base, "gone", "current"))+"\n")
	noBlock := write("plain.md", "just the user's own rules\n")
	otherSkillBlock := write("other.md", renderPointerBlock(Catalog[len(Catalog)-1], livePointer)+"\n")
	truncated := write("truncated.md", strings.Split(renderPointerBlock(skill, livePointer), markerEnd(skill))[0])

	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "block pointing at a live SKILL.md", path: valid, want: true},
		{name: "block pointing at a deleted SKILL.md", path: deletedPointer, want: false},
		{name: "rules file without our block", path: noBlock, want: false},
		{name: "only another skill's block", path: otherSkillBlock, want: false},
		{name: "block missing its closing marker", path: truncated, want: false},
		{name: "missing rules file", path: filepath.Join(base, "absent.md"), want: false},
		{name: "empty path", path: "", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorded := InstalledTarget{Method: "marker-block", Path: tc.path, Version: skill.Version}
			if got := installedArtifactValid(skill, target, recorded); got != tc.want {
				t.Fatalf("installedArtifactValid(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

// Contract: what the artifact was installed WITH decides how it is checked,
// and a method this binary no longer uses for that target is never valid —
// that is what migrates legacy Codex marker-block records onto the native
// skills symlink instead of skipping them forever at a matching version.
func TestInstalledArtifactValidMethodChanges(t *testing.T) {
	base := t.TempDir()
	skill := Catalog[0]
	rules := filepath.Join(base, "AGENTS.override.md")
	if err := os.MkdirAll(filepath.Join(base, "current"), 0o755); err != nil {
		t.Fatalf("mkdir current: %v", err)
	}
	if err := os.WriteFile(filepath.Join(base, "current", skill.EntryFile), []byte("# skill"), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}
	if err := os.WriteFile(rules, []byte(renderPointerBlock(skill, filepath.Join(base, "current"))), 0o644); err != nil {
		t.Fatalf("write rules: %v", err)
	}

	legacy := InstalledTarget{Method: "marker-block", Path: rules, Version: skill.Version}
	if installedArtifactValid(skill, codexTarget{}, legacy) {
		t.Fatal("a legacy codex marker-block record must be invalid now that codex installs by symlink")
	}
	if !installedArtifactValid(skill, stubTarget{name: "legacy", method: "marker-block"}, legacy) {
		t.Fatal("a marker-block record must stay valid for a target that still installs marker blocks")
	}

	manual := InstalledTarget{Method: "manual", Path: "Cursor Settings → Rules for AI (manual)", Version: skill.Version}
	if !installedArtifactValid(skill, cursorTarget{}, manual) {
		t.Fatal("manual targets write nothing to disk; there is no artifact to invalidate")
	}
	unknown := InstalledTarget{Method: "carrier-pigeon", Path: rules, Version: skill.Version}
	if installedArtifactValid(skill, stubTarget{name: "stub", method: "carrier-pigeon"}, unknown) {
		t.Fatal("an unrecognized install method must be treated as needing repair")
	}
}

// Contract: this is the self-healing property the desktop's launch-time
// `af skill install` relies on. State says the target is current; the artifact
// is gone. The install must repair it instead of skipping it, without --force.
func TestInstallRepairsRecordedTargetWhoseArtifactVanished(t *testing.T) {
	home := codexHome(t)

	first, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}})
	if err != nil {
		t.Fatalf("first Install: %v", err)
	}
	if len(first.TargetsInstalled) != 1 {
		t.Fatalf("first install did not install codex: %+v", first)
	}
	link := filepath.Join(home, ".codex", "skills", "agentfield")

	// A second run with the artifact intact is a no-op skip.
	second, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}})
	if err != nil {
		t.Fatalf("second Install: %v", err)
	}
	if len(second.TargetsInstalled) != 0 || len(second.TargetsSkipped) != 1 ||
		!strings.Contains(second.TargetsSkipped[0].Reason, "already installed") {
		t.Fatalf("intact artifact should be skipped: %+v", second)
	}

	// Now the artifact disappears (an uninstall of the agent, a cleaned home,
	// a link into a deleted temp dir) while state still says v<current>.
	if err := os.RemoveAll(link); err != nil {
		t.Fatalf("remove link: %v", err)
	}
	third, err := Install(InstallOptions{SkillName: "agentfield", Targets: []string{"codex"}})
	if err != nil {
		t.Fatalf("third Install: %v", err)
	}
	if len(third.TargetsInstalled) != 1 || third.TargetsInstalled[0].TargetName != "codex" {
		t.Fatalf("missing artifact was not repaired: %+v", third)
	}
	if _, err := os.Stat(link); err != nil {
		t.Fatalf("repaired link missing: %v", err)
	}
}

// Contract: the machines already in the wild. State records codex at the
// current version through the old marker-block method, and the block sits in
// a file Codex never reads. A plain `af skill install` must repair that.
func TestInstallMigratesLegacyCodexMarkerBlockInstallation(t *testing.T) {
	home := codexHome(t)
	skill := Catalog[0]
	legacyPath := filepath.Join(home, ".codex", "AGENTS.override.md")
	if err := os.WriteFile(legacyPath, []byte("keep me\n\n"+renderPointerBlock(skill, "/deleted/canonical/current")+"\n"), 0o644); err != nil {
		t.Fatalf("seed legacy rules file: %v", err)
	}
	if err := SaveState(&State{Skills: map[string]InstalledSkill{
		skill.Name: {
			CurrentVersion:    skill.Version,
			InstalledAt:       time.Now().UTC(),
			AvailableVersions: []string{skill.Version},
			Targets: map[string]InstalledTarget{
				"codex": {
					TargetName:  "codex",
					Method:      "marker-block",
					Path:        legacyPath,
					Version:     skill.Version,
					InstalledAt: time.Now().UTC(),
				},
			},
		},
	}}); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	report, err := Install(InstallOptions{SkillName: skill.Name, Targets: []string{"codex"}})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(report.TargetsInstalled) != 1 || report.TargetsInstalled[0].Method != "symlink" {
		t.Fatalf("legacy marker-block install was not migrated: %+v", report)
	}
	link := filepath.Join(home, ".codex", "skills", skill.Name)
	if _, err := os.Stat(filepath.Join(link, skill.EntryFile)); err != nil {
		t.Fatalf("SKILL.md not readable through the native skills dir: %v", err)
	}
	data, err := os.ReadFile(legacyPath)
	if err != nil {
		t.Fatalf("read legacy rules file: %v", err)
	}
	if strings.Contains(string(data), markerStartPattern(skill)) {
		t.Fatalf("legacy block survived the migration:\n%s", data)
	}
	if !strings.Contains(string(data), "keep me") {
		t.Fatalf("migration destroyed user content:\n%s", data)
	}

	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if got := state.Skills[skill.Name].Targets["codex"]; got.Method != "symlink" || got.Path != link {
		t.Fatalf("state still records the legacy integration: %+v", got)
	}
}
