package skillkit

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// realHomeBeforeIsolation is the user's actual home directory, captured once
// before TestMain redirects every home lookup into a temp tree. Tests use it
// only to assert that nothing was written there.
var realHomeBeforeIsolation string

// isolatedHome is the package-wide fake home every test inherits.
var isolatedHome string

// TestMain redirects the home directory for the whole package before any test
// runs.
//
// The canonical store honors AGENTFIELD_HOME, so tests that set it were
// isolated for state — but every target resolves its install path through
// os.UserHomeDir(), which AGENTFIELD_HOME does not touch. Tests that installed
// into real targets therefore wrote into the developer's actual home: a real
// ~/.codex/AGENTS.override.md was found in the wild holding marker blocks
// written by the reconcile suite, pointing at /var/folders temp paths that had
// been deleted with the test run. Setting HOME (and USERPROFILE, which
// os.UserHomeDir uses on Windows) here makes that structurally impossible —
// individual tests can still narrow it further with t.Setenv, and the value is
// restored to this temp tree after each one.
func TestMain(m *testing.M) {
	if home, err := os.UserHomeDir(); err == nil {
		realHomeBeforeIsolation = home
	}

	dir, err := os.MkdirTemp("", "skillkit-home-")
	if err != nil {
		fmt.Fprintf(os.Stderr, "skillkit tests: create isolated home: %v\n", err)
		os.Exit(1)
	}
	// Resolve symlinks (macOS hands out /var/... for /private/var/...) so the
	// path a test compares against is the one the OS reports back.
	if resolved, err := filepath.EvalSymlinks(dir); err == nil {
		dir = resolved
	}
	isolatedHome = dir

	for key, value := range map[string]string{
		"HOME":            dir,
		"USERPROFILE":     dir,
		"AGENTFIELD_HOME": filepath.Join(dir, ".agentfield"),
	} {
		if err := os.Setenv(key, value); err != nil {
			fmt.Fprintf(os.Stderr, "skillkit tests: set %s: %v\n", key, err)
			os.Exit(1)
		}
	}

	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

// TestPackageHomeIsIsolated is the guard for the guard: if TestMain ever stops
// redirecting the home directory, this fails before any test can write into
// the developer's real one.
func TestPackageHomeIsIsolated(t *testing.T) {
	if isolatedHome == "" {
		t.Fatal("TestMain did not create an isolated home")
	}
	if got := homeDir(); got != isolatedHome {
		t.Fatalf("homeDir() = %q, want the isolated temp home %q", got, isolatedHome)
	}
	if realHomeBeforeIsolation != "" && homeDir() == realHomeBeforeIsolation {
		t.Fatalf("homeDir() still resolves to the real home %q", realHomeBeforeIsolation)
	}
	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	if !pathWithin(isolatedHome, root) {
		t.Fatalf("CanonicalRoot() = %q, want inside the isolated home %q", root, isolatedHome)
	}
}

// TestInstallAllWritesNothingIntoTheRealHome is the regression test for the
// pollution that motivated TestMain: a full catalog install into every
// registered target must land entirely inside the fake home and leave the
// user's real home byte-for-byte untouched.
func TestInstallAllWritesNothingIntoTheRealHome(t *testing.T) {
	if realHomeBeforeIsolation == "" {
		t.Skip("real home directory could not be resolved")
	}
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	home := withTempHome(t)

	before := realHomeSnapshot(t)

	reports, err := InstallAll(InstallOptions{AllRegistered: true, Force: true})
	if err != nil {
		t.Fatalf("InstallAll: %v", err)
	}
	if len(reports) != len(Catalog) {
		t.Fatalf("got %d reports, want one per catalog skill (%d)", len(reports), len(Catalog))
	}

	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	installedSomewhere := false
	for _, report := range reports {
		if !pathWithin(root, report.CanonicalDir) {
			t.Fatalf("canonical dir %q escaped the fake store %q", report.CanonicalDir, root)
		}
		for _, installed := range report.TargetsInstalled {
			if installed.Path == "" || installed.Method == "manual" {
				continue // cursor prints instructions; it writes nothing
			}
			installedSomewhere = true
			if !pathWithin(home, installed.Path) && !pathWithin(root, installed.Path) {
				t.Fatalf("target %q wrote to %q, outside the fake home %q",
					installed.TargetName, installed.Path, home)
			}
			if pathWithin(realHomeBeforeIsolation, installed.Path) {
				t.Fatalf("target %q wrote into the real home: %q", installed.TargetName, installed.Path)
			}
		}
	}
	if !installedSomewhere {
		t.Fatal("no target reported an installed path; the assertion above proved nothing")
	}

	if after := realHomeSnapshot(t); after != before {
		t.Fatalf("InstallAll changed the real home:\nbefore: %s\nafter:  %s", before, after)
	}
}

// realHomeSnapshot fingerprints every path in the real home that a skill
// install would touch, so the test can prove none of them moved.
func realHomeSnapshot(t *testing.T) string {
	t.Helper()
	paths := []string{
		filepath.Join(realHomeBeforeIsolation, ".agentfield", "skills"),
		filepath.Join(realHomeBeforeIsolation, ".claude", "skills"),
		filepath.Join(realHomeBeforeIsolation, ".claude", "commands"),
		filepath.Join(realHomeBeforeIsolation, ".codex", "skills"),
		filepath.Join(realHomeBeforeIsolation, ".codex", "AGENTS.override.md"),
		filepath.Join(realHomeBeforeIsolation, ".gemini", "GEMINI.md"),
		filepath.Join(realHomeBeforeIsolation, ".config", "opencode", "skills"),
		filepath.Join(realHomeBeforeIsolation, ".config", "opencode", "AGENTS.md"),
		filepath.Join(realHomeBeforeIsolation, ".aider.conventions.md"),
		filepath.Join(realHomeBeforeIsolation, ".aider.conf.yml"),
		filepath.Join(realHomeBeforeIsolation, ".codeium", "windsurf", "memories", "global_rules.md"),
	}
	snapshot := ""
	for _, path := range paths {
		info, err := os.Lstat(path)
		if err != nil {
			snapshot += fmt.Sprintf("%s=absent\n", path)
			continue
		}
		snapshot += fmt.Sprintf("%s=%d/%d/%s\n", path, info.Size(), info.Mode(), info.ModTime().UTC())
	}
	return snapshot
}
