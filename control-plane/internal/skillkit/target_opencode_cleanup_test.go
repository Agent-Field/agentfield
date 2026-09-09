package skillkit

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// opencodeLegacyHome prepares an isolated home with ~/.config/opencode present
// and returns the home plus the legacy rules file path inside it.
func opencodeLegacyHome(t *testing.T) (string, string) {
	t.Helper()
	home := withTempHome(t)
	dir := filepath.Join(home, ".config", "opencode")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir .config/opencode: %v", err)
	}
	return home, filepath.Join(dir, "AGENTS.md")
}

// seedCurrentDir returns a canonical current/ directory an install can link at.
func seedCurrentDir(t *testing.T) string {
	t.Helper()
	current := filepath.Join(t.TempDir(), "current")
	if err := os.MkdirAll(current, 0o755); err != nil {
		t.Fatalf("mkdir current: %v", err)
	}
	return current
}

// foreignBlock is another tool's marker block: same file, different owner.
const foreignBlock = "<!-- plandb-skill:plandb v1 -->\nplandb rules\n<!-- /plandb-skill:plandb -->"

// Contract (a) + (e): installing the native skill strips this skill's legacy
// marker block from ~/.config/opencode/AGENTS.md while leaving the user's prose
// on both sides of it — and another tool's block — untouched.
func TestOpenCodeInstallStripsLegacyMarkerBlockAndKeepsForeignContent(t *testing.T) {
	_, legacy := opencodeLegacyHome(t)
	content := "# my own opencode notes\n\n" +
		renderPointerBlock(Catalog[0], "/gone/canonical/current") + "\n\n" +
		foreignBlock + "\n\nnotes that come after the block\n"
	if err := os.WriteFile(legacy, []byte(content), 0o644); err != nil {
		t.Fatalf("seed legacy rules file: %v", err)
	}

	if _, err := (opencodeTarget{}).Install(Catalog[0], seedCurrentDir(t)); err != nil {
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
	for _, keep := range []string{"# my own opencode notes", foreignBlock, "notes that come after the block"} {
		if !strings.Contains(got, keep) {
			t.Fatalf("migration destroyed content it does not own (%q missing):\n%s", keep, got)
		}
	}
}

// Contract (c): a rules file that held nothing but our block was ours alone, so
// it is deleted rather than left behind as an empty file OpenCode keeps reading.
func TestOpenCodeInstallDeletesLegacyRulesFileItOwnedAlone(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
	}{
		{name: "block only", content: renderPointerBlock(Catalog[0], "/gone/current") + "\n"},
		{name: "block and whitespace", content: "\n  \n" + renderPointerBlock(Catalog[0], "/gone/current") + "\n \n\t\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, legacy := opencodeLegacyHome(t)
			if err := os.WriteFile(legacy, []byte(tc.content), 0o644); err != nil {
				t.Fatalf("seed legacy rules file: %v", err)
			}
			if _, err := (opencodeTarget{}).Install(Catalog[0], seedCurrentDir(t)); err != nil {
				t.Fatalf("Install: %v", err)
			}
			if _, err := os.Lstat(legacy); !os.IsNotExist(err) {
				data, _ := os.ReadFile(legacy)
				t.Fatalf("legacy rules file should be removed, lstat err=%v content=%q", err, data)
			}
		})
	}
}

// Contract (d) + (e): ~/.config/opencode/AGENTS.md is the user's own file. When
// it carries no block of ours, neither install nor uninstall may touch it —
// same bytes, same modification time, whitespace-only content included.
func TestOpenCodeLeavesARulesFileWithoutOurBlockUntouched(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
	}{
		{name: "user prose", content: "# my rules\n\nbe concise\n"},
		{name: "foreign block only", content: foreignBlock + "\n"},
		{name: "whitespace only", content: "\n  \n\t\n"},
		{name: "empty", content: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, legacy := opencodeLegacyHome(t)
			if err := os.WriteFile(legacy, []byte(tc.content), 0o644); err != nil {
				t.Fatalf("seed legacy rules file: %v", err)
			}
			// Backdate so a rewrite is visible even at coarse mtime resolution.
			stamp := time.Date(2020, time.March, 4, 5, 6, 7, 0, time.UTC)
			if err := os.Chtimes(legacy, stamp, stamp); err != nil {
				t.Fatalf("chtimes: %v", err)
			}

			target := opencodeTarget{}
			if _, err := target.Install(Catalog[0], seedCurrentDir(t)); err != nil {
				t.Fatalf("Install: %v", err)
			}
			assertRulesFileUnchanged(t, legacy, tc.content, stamp, "install")

			if err := target.Uninstall(); err != nil {
				t.Fatalf("Uninstall: %v", err)
			}
			assertRulesFileUnchanged(t, legacy, tc.content, stamp, "uninstall")
		})
	}
}

func assertRulesFileUnchanged(t *testing.T, path, want string, stamp time.Time, stage string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s removed the user's rules file: %v", stage, err)
	}
	if string(data) != want {
		t.Fatalf("%s rewrote the user's rules file:\ngot:  %q\nwant: %q", stage, data, want)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat after %s: %v", stage, err)
	}
	if !info.ModTime().Equal(stamp) {
		t.Fatalf("%s opened the user's rules file for writing: mtime %s, want %s",
			stage, info.ModTime().UTC(), stamp)
	}
}

// Contract (b) + (e): uninstall removes every catalog skill's link and finishes
// the migration for machines that never ran an install in between, keeping the
// user's prose and other tools' blocks.
func TestOpenCodeUninstallRemovesLinksAndLegacyBlocks(t *testing.T) {
	home, legacy := opencodeLegacyHome(t)
	var content strings.Builder
	content.WriteString("user prose\n\n")
	for _, s := range Catalog {
		content.WriteString(renderPointerBlock(s, "/gone/current"))
		content.WriteString("\n\n")
	}
	content.WriteString(foreignBlock + "\n")
	if err := os.WriteFile(legacy, []byte(content.String()), 0o644); err != nil {
		t.Fatalf("seed legacy rules file: %v", err)
	}

	target := opencodeTarget{}
	root, err := target.TargetPath()
	if err != nil {
		t.Fatalf("TargetPath: %v", err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatalf("mkdir skills root: %v", err)
	}
	for _, s := range Catalog {
		if err := os.Symlink(filepath.Join(home, "gone"), filepath.Join(root, s.Name)); err != nil {
			t.Fatalf("seed link for %s: %v", s.Name, err)
		}
	}

	if err := target.Uninstall(); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	for _, s := range Catalog {
		if _, err := os.Lstat(filepath.Join(root, s.Name)); !os.IsNotExist(err) {
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
	if !strings.Contains(string(data), "user prose") || !strings.Contains(string(data), foreignBlock) {
		t.Fatalf("uninstall destroyed content it does not own:\n%s", data)
	}
	// Uninstalling twice is a no-op, not an error.
	if err := target.Uninstall(); err != nil {
		t.Fatalf("second Uninstall: %v", err)
	}
}

// Contract (f): with no legacy rules file on disk, install and uninstall both
// succeed and neither conjures the file into existence.
func TestOpenCodeCleanupIsANoOpWithoutALegacyRulesFile(t *testing.T) {
	_, legacy := opencodeLegacyHome(t)
	target := opencodeTarget{}

	if _, err := target.Install(Catalog[0], seedCurrentDir(t)); err != nil {
		t.Fatalf("Install: %v", err)
	}
	if _, err := os.Lstat(legacy); !os.IsNotExist(err) {
		t.Fatalf("install created a legacy rules file: %v", err)
	}
	if err := target.Uninstall(); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	if _, err := os.Lstat(legacy); !os.IsNotExist(err) {
		t.Fatalf("uninstall created a legacy rules file: %v", err)
	}
}

// Contract (g): uninstall reports a legacy rules file it cannot read, rather
// than silently leaving the block behind — there, stripping the block is the
// entire point of the call.
func TestOpenCodeUninstallReportsAnUnreadableLegacyRulesFile(t *testing.T) {
	_, legacy := opencodeLegacyHome(t)
	// A directory where the rules file belongs: readable path, unreadable file.
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatalf("mkdir over legacy rules path: %v", err)
	}

	err := (opencodeTarget{}).Uninstall()
	if err == nil {
		t.Fatal("Uninstall should report a legacy rules file it cannot read")
	}
	if !strings.Contains(err.Error(), "AGENTS.md") {
		t.Fatalf("error should name the file it could not read: %v", err)
	}
}

// Contract (h): the cleanup is a migration courtesy, not part of making the
// skill work. A legacy rules file af cannot clean up must not fail an install
// whose symlink is already on disk: the caller records nothing for a failed
// target, so failing here would report OpenCode as not installed while it is
// live, and every later `af skill install` would exit non-zero over a stale
// block OpenCode never reads.
func TestOpenCodeInstallSurvivesALegacyRulesFileItCannotClean(t *testing.T) {
	home := withTempHome(t)
	legacy := filepath.Join(home, ".config", "opencode", "AGENTS.md")
	// A directory where the rules file belongs: readable path, unreadable file.
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatalf("mkdir over legacy rules path: %v", err)
	}

	report, err := Install(InstallOptions{SkillName: Catalog[0].Name, Targets: []string{"opencode"}})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(report.TargetsFailed) != 0 {
		t.Fatalf("an uncleanable legacy rules file failed the install: %+v", report.TargetsFailed)
	}
	if len(report.TargetsInstalled) != 1 || report.TargetsInstalled[0].TargetName != "opencode" {
		t.Fatalf("opencode was not reported as installed: %+v", report)
	}

	link := filepath.Join(home, ".config", "opencode", "skills", Catalog[0].Name)
	if _, err := os.Lstat(link); err != nil {
		t.Fatalf("native skill link missing: %v", err)
	}
	// The link is on disk, so state has to agree — otherwise `af skill list`
	// and the next install both disagree with reality.
	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	recorded, ok := state.Skills[Catalog[0].Name].Targets["opencode"]
	if !ok {
		t.Fatal("a live OpenCode install was not recorded in state")
	}
	if recorded.Path != link {
		t.Fatalf("recorded path = %q, want the link on disk %q", recorded.Path, link)
	}
}

// Contract (i): every write the cleanup performs is reported when it fails.
// These branches are unreachable through real filesystem permissions on some
// platforms, so they are driven through the package's reconcile* seams — the
// same way the reconciler's own rewrite failures are covered.
func TestOpenCodeUninstallReportsLegacyRewriteFailures(t *testing.T) {
	ourBlock := renderPointerBlock(Catalog[0], "/gone/current")
	for _, tc := range []struct {
		name    string
		content string
		inject  func(t *testing.T)
	}{
		{
			name:    "write",
			content: "user prose\n\n" + ourBlock + "\n",
			inject: func(t *testing.T) {
				old := reconcileWriteFile
				reconcileWriteFile = func(string, []byte, os.FileMode) error {
					return errors.New("forced write failure")
				}
				t.Cleanup(func() { reconcileWriteFile = old })
			},
		},
		{
			name:    "rename",
			content: "user prose\n\n" + ourBlock + "\n",
			inject: func(t *testing.T) {
				old := reconcileRename
				reconcileRename = func(string, string) error { return errors.New("forced rename failure") }
				t.Cleanup(func() { reconcileRename = old })
			},
		},
		{
			name:    "remove",
			content: ourBlock + "\n",
			inject: func(t *testing.T) {
				old := reconcileRemove
				reconcileRemove = func(string) error { return errors.New("forced remove failure") }
				t.Cleanup(func() { reconcileRemove = old })
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, legacy := opencodeLegacyHome(t)
			if err := os.WriteFile(legacy, []byte(tc.content), 0o644); err != nil {
				t.Fatalf("seed legacy rules file: %v", err)
			}
			tc.inject(t)

			err := (opencodeTarget{}).Uninstall()
			if err == nil {
				t.Fatalf("Uninstall should report a failed %s of the legacy rules file", tc.name)
			}
			if !strings.Contains(err.Error(), "AGENTS.md") ||
				!strings.Contains(err.Error(), "forced "+tc.name+" failure") {
				t.Fatalf("error should name the file and the cause: %v", err)
			}
		})
	}
}
