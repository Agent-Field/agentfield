package skillkit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// codexHome prepares an isolated home with a ~/.codex directory, the shape a
// machine with Codex installed has.
func codexHome(t *testing.T) string {
	t.Helper()
	home := withTempHome(t)
	if err := os.MkdirAll(filepath.Join(home, ".codex"), 0o755); err != nil {
		t.Fatalf("mkdir .codex: %v", err)
	}
	return home
}

// Contract: Codex loads personal skills from ~/.codex/skills/<name>/SKILL.md.
// Installing must put a symlink to the canonical store there — the old
// marker block in ~/.codex/AGENTS.override.md went into a file Codex never
// reads, so the skill was never actually installed.
func TestCodexInstallLinksNativeSkillsDirectory(t *testing.T) {
	home := codexHome(t)
	current := filepath.Join(t.TempDir(), "current")
	if err := os.MkdirAll(current, 0o755); err != nil {
		t.Fatalf("mkdir current: %v", err)
	}
	if err := os.WriteFile(filepath.Join(current, "SKILL.md"), []byte("# skill\n"), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}

	inst, err := codexTarget{}.Install(Catalog[0], current)
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	link := filepath.Join(home, ".codex", "skills", Catalog[0].Name)
	if inst.Path != link || inst.Method != "symlink" {
		t.Fatalf("install result = %+v, want symlink at %q", inst, link)
	}
	if dest, err := os.Readlink(link); err != nil || dest != current {
		t.Fatalf("readlink = %q err=%v, want %q", dest, err, current)
	}
	// The skill's own SKILL.md must be reachable through the link — that is
	// the whole contract with Codex.
	if _, err := os.Stat(filepath.Join(link, "SKILL.md")); err != nil {
		t.Fatalf("SKILL.md not reachable through the link: %v", err)
	}
	// Codex has no ~/.claude/commands analogue; nothing else may be created.
	entries, err := os.ReadDir(filepath.Join(home, ".codex"))
	if err != nil {
		t.Fatalf("read .codex: %v", err)
	}
	for _, entry := range entries {
		if entry.Name() != "skills" {
			t.Fatalf("install created unexpected entry in ~/.codex: %q", entry.Name())
		}
	}
}

// Contract: installing replaces whatever occupies the skill directory —
// including a real directory a user (or an older tool) copied in.
func TestCodexInstallReplacesExistingDirectoryAndStaleLink(t *testing.T) {
	home := codexHome(t)
	link := filepath.Join(home, ".codex", "skills", Catalog[0].Name)
	if err := os.MkdirAll(link, 0o755); err != nil {
		t.Fatalf("seed directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(link, "SKILL.md"), []byte("stale copy"), 0o644); err != nil {
		t.Fatalf("seed stale copy: %v", err)
	}

	first := filepath.Join(t.TempDir(), "v1")
	second := filepath.Join(t.TempDir(), "v2")
	for _, dir := range []string{first, second} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}

	target := codexTarget{}
	if _, err := target.Install(Catalog[0], first); err != nil {
		t.Fatalf("Install over directory: %v", err)
	}
	if dest, err := os.Readlink(link); err != nil || dest != first {
		t.Fatalf("readlink = %q err=%v, want %q", dest, err, first)
	}
	if _, err := target.Install(Catalog[0], second); err != nil {
		t.Fatalf("Install over stale link: %v", err)
	}
	if dest, err := os.Readlink(link); err != nil || dest != second {
		t.Fatalf("readlink = %q err=%v, want %q", dest, err, second)
	}
}

// Contract: the migration off the old integration. Every install/update/
// uninstall strips this skill's marker block from ~/.codex/AGENTS.override.md
// while leaving the user's own prose and other tools' blocks alone.
func TestCodexInstallStripsLegacyMarkerBlockAndKeepsForeignContent(t *testing.T) {
	home := codexHome(t)
	legacy := filepath.Join(home, ".codex", "AGENTS.override.md")
	foreign := "<!-- plandb-skill:plandb v1 -->\nplandb rules\n<!-- /plandb-skill:plandb -->"
	content := "# my own notes\n\n" +
		renderPointerBlock(Catalog[0], "/gone/canonical/current") + "\n\n" + foreign + "\n"
	if err := os.WriteFile(legacy, []byte(content), 0o644); err != nil {
		t.Fatalf("seed legacy rules file: %v", err)
	}

	current := filepath.Join(t.TempDir(), "current")
	if err := os.MkdirAll(current, 0o755); err != nil {
		t.Fatalf("mkdir current: %v", err)
	}
	if _, err := (codexTarget{}).Install(Catalog[0], current); err != nil {
		t.Fatalf("Install: %v", err)
	}

	data, err := os.ReadFile(legacy)
	if err != nil {
		t.Fatalf("read legacy rules file: %v", err)
	}
	got := string(data)
	if strings.Contains(got, markerStartPattern(Catalog[0])) {
		t.Fatalf("legacy marker block survived the install:\n%s", got)
	}
	if !strings.Contains(got, "# my own notes") || !strings.Contains(got, foreign) {
		t.Fatalf("migration destroyed content it does not own:\n%s", got)
	}
}

// Contract: a rules file that held nothing but our block is deleted rather
// than left behind as an empty (or whitespace-only) file Codex would keep
// listing forever.
func TestCodexInstallDeletesLegacyRulesFileItOwnedAlone(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
	}{
		{name: "block only", content: renderPointerBlock(Catalog[0], "/gone/current") + "\n"},
		{name: "block and whitespace", content: "\n  \n" + renderPointerBlock(Catalog[0], "/gone/current") + "\n \n\t\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := codexHome(t)
			legacy := filepath.Join(home, ".codex", "AGENTS.override.md")
			if err := os.WriteFile(legacy, []byte(tc.content), 0o644); err != nil {
				t.Fatalf("seed legacy rules file: %v", err)
			}
			current := filepath.Join(t.TempDir(), "current")
			if err := os.MkdirAll(current, 0o755); err != nil {
				t.Fatalf("mkdir current: %v", err)
			}
			if _, err := (codexTarget{}).Install(Catalog[0], current); err != nil {
				t.Fatalf("Install: %v", err)
			}
			if _, err := os.Lstat(legacy); !os.IsNotExist(err) {
				data, _ := os.ReadFile(legacy)
				t.Fatalf("legacy rules file should be removed, lstat err=%v content=%q", err, data)
			}
		})
	}
}

// Contract: uninstall removes every catalog skill's link and finishes the
// legacy migration for machines that never ran an install in between.
func TestCodexUninstallRemovesLinksAndLegacyBlocks(t *testing.T) {
	home := codexHome(t)
	legacy := filepath.Join(home, ".codex", "AGENTS.override.md")
	var content strings.Builder
	content.WriteString("user prose\n\n")
	for _, s := range Catalog {
		content.WriteString(renderPointerBlock(s, "/gone/current"))
		content.WriteString("\n\n")
	}
	if err := os.WriteFile(legacy, []byte(content.String()), 0o644); err != nil {
		t.Fatalf("seed legacy rules file: %v", err)
	}

	current := filepath.Join(t.TempDir(), "current")
	if err := os.MkdirAll(current, 0o755); err != nil {
		t.Fatalf("mkdir current: %v", err)
	}
	target := codexTarget{}
	for _, s := range Catalog {
		if _, err := target.Install(s, current); err != nil {
			t.Fatalf("Install(%s): %v", s.Name, err)
		}
	}

	if err := target.Uninstall(); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	for _, s := range Catalog {
		if _, err := os.Lstat(filepath.Join(home, ".codex", "skills", s.Name)); !os.IsNotExist(err) {
			t.Fatalf("link for %s remains: %v", s.Name, err)
		}
	}
	data, err := os.ReadFile(legacy)
	if err != nil {
		t.Fatalf("read legacy rules file: %v", err)
	}
	if strings.Contains(string(data), "agentfield-skill:") {
		t.Fatalf("legacy blocks remain after uninstall:\n%s", data)
	}
	if !strings.Contains(string(data), "user prose") {
		t.Fatalf("uninstall destroyed user content:\n%s", data)
	}
	// Uninstalling twice is a no-op, not an error.
	if err := target.Uninstall(); err != nil {
		t.Fatalf("second Uninstall: %v", err)
	}
}
