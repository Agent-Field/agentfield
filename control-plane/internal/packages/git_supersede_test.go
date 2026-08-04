package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These cover the shape a real repo uses to retire a node: the root manifest
// declares itself superseded by a package in a subdirectory of the same repo,
// so `af install <repo>` lands on the successor.

const supersededRoot = "name: dual-node\nversion: 1.0.0\n" +
	"superseded_by: https://gitlab.com/acme/dual//go\n"

func seedInstalled(t *testing.T, home, name string) string {
	t.Helper()
	pkgDir := filepath.Join(home, "packages", name)
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	pu := &PackageUninstaller{AgentFieldHome: home}
	registry, err := pu.loadRegistry()
	if err != nil {
		t.Fatal(err)
	}
	registry.Installed[name] = InstalledPackage{Name: name, Path: pkgDir, Status: "stopped"}
	if err := pu.saveRegistry(registry); err != nil {
		t.Fatal(err)
	}
	return pkgDir
}

// Contract: installing a superseded package installs its successor instead,
// and the superseded name never reaches the registry.
func TestInstallFromGit_SupersededRedirectsToSuccessor(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	if err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if _, ok := registry.Installed["dual-node-go"]; !ok {
		t.Fatalf("successor missing from registry, got %v", registry.Installed)
	}
	if _, ok := registry.Installed["dual-node"]; ok {
		t.Fatal("the superseded package must not be installed")
	}
	if _, err := os.Stat(filepath.Join(home, "packages", "dual-node-go", "agentfield-package.yaml")); err != nil {
		t.Fatalf("successor not on disk: %v", err)
	}
}

// Contract: when the superseded package is already installed it is replaced —
// the successor lands first, then the old package is stopped and removed.
func TestInstallFromGit_SupersededReplacesExistingInstall(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	oldDir := seedInstalled(t, home, "dual-node")

	if err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if _, ok := registry.Installed["dual-node-go"]; !ok {
		t.Fatalf("successor missing, got %v", registry.Installed)
	}
	if _, ok := registry.Installed["dual-node"]; ok {
		t.Fatal("superseded package should have been retired from the registry")
	}
	if _, err := os.Stat(oldDir); !os.IsNotExist(err) {
		t.Fatalf("superseded package dir should be gone, stat err = %v", err)
	}
}

// Contract: node-scoped secrets follow the user across the swap, because
// uninstalling the old package deletes that scope outright. A value already
// set on the successor wins, and global secrets are untouched.
func TestInstallFromGit_SupersededMigratesNodeScopedSecrets(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	seedInstalled(t, home, "dual-node")
	store, err := NewSecretStore(home)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Set("dual-node", "CARRIED", "from-old"); err != nil {
		t.Fatal(err)
	}
	if err := store.Set("dual-node", "KEPT", "old-value"); err != nil {
		t.Fatal(err)
	}
	if err := store.Set("dual-node-go", "KEPT", "new-value"); err != nil {
		t.Fatal(err)
	}
	if err := store.Set("global", "SHARED", "shared-value"); err != nil {
		t.Fatal(err)
	}

	if err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	after, err := NewSecretStore(home)
	if err != nil {
		t.Fatal(err)
	}
	values, err := after.load("dual-node-go")
	if err != nil {
		t.Fatal(err)
	}
	if values["CARRIED"] != "from-old" {
		t.Fatalf("secret did not follow the swap: %v", values)
	}
	if values["KEPT"] != "new-value" {
		t.Fatalf("successor's own value must win, got %q", values["KEPT"])
	}
	globals, err := after.load("global")
	if err != nil {
		t.Fatal(err)
	}
	if globals["SHARED"] != "shared-value" {
		t.Fatal("global secrets must survive the swap")
	}
}

// Contract: with nothing to replace, the redirect is a plain install — no
// error, and no attempt to retire a package that was never there.
func TestInstallFromGit_SupersededWithoutPriorInstall(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	if err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}
	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if len(registry.Installed) != 1 {
		t.Fatalf("expected exactly the successor installed, got %v", registry.Installed)
	}
}

// Contract: two manifests pointing at each other fail loudly instead of
// redirecting forever.
func TestInstallFromGit_SupersededCycleIsBounded(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	// The successor points straight back at the root: A → B → A → …
	if err := os.MkdirAll(filepath.Join(repo, "go"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: dual-node-go\nversion: 1.0.0\n" +
		"entrypoint:\n  start: python -m dual-node-go\n" +
		"superseded_by: https://gitlab.com/acme/dual\n"
	if err := os.WriteFile(
		filepath.Join(repo, "go", "agentfield-package.yaml"), []byte(manifest), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	setupFakeGit(t, "copy", repo, false)

	err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/dual", false)
	if err == nil || !strings.Contains(err.Error(), "superseded_by chain longer than") {
		t.Fatalf("expected a bounded-chain error, got %v", err)
	}
	// Nothing was installed, so the registry was never even created.
	if _, statErr := os.Stat(filepath.Join(home, "installed.yaml")); !os.IsNotExist(statErr) {
		registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
		if len(registry.Installed) != 0 {
			t.Fatalf("a cycle must install nothing, got %v", registry.Installed)
		}
	}
}

// Contract: a source recorded for a --path install round-trips through
// ParseGitURL back to the same repo AND subdir, so the next update resolves
// the package that is actually installed rather than the repo root.
func TestAppendSubdirSelectorRoundTrips(t *testing.T) {
	cases := []struct {
		url, subdir, want, wantRef string
	}{
		{"https://github.com/acme/repo", "go", "https://github.com/acme/repo//go", ""},
		{"https://github.com/acme/repo@main", "go", "https://github.com/acme/repo//go@main", "main"},
		{"https://github.com/acme/repo", "nested/dir", "https://github.com/acme/repo//nested/dir", ""},
		{"https://github.com/acme/repo", "", "https://github.com/acme/repo", ""},
	}
	for _, c := range cases {
		got := appendSubdirSelector(c.url, c.subdir)
		if got != c.want {
			t.Errorf("appendSubdirSelector(%q, %q) = %q, want %q", c.url, c.subdir, got, c.want)
			continue
		}
		info, err := ParseGitURL(got)
		if err != nil {
			t.Errorf("ParseGitURL(%q): %v", got, err)
			continue
		}
		wantSubdir := strings.Trim(c.subdir, "/")
		if info.Subdir != wantSubdir || info.Ref != c.wantRef {
			t.Errorf("round-trip of %q = subdir %q ref %q, want %q %q",
				got, info.Subdir, info.Ref, wantSubdir, c.wantRef)
		}
	}
}

// Contract: the registry records the subdirectory even when it arrived by the
// --path flag rather than the URL selector. Without this the stored source
// resolves to the repo root and the next update installs a different package.
func TestInstallFromGit_PathFlagRecordsSubdirInSource(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, "name: dual-node\nversion: 1.0.0\n")
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	gi := &GitInstaller{AgentFieldHome: home, Subdir: "go"}
	if err := gi.InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	pkg, ok := registry.Installed["dual-node-go"]
	if !ok {
		t.Fatalf("expected dual-node-go installed, got %v", registry.Installed)
	}
	if pkg.SourcePath != "https://gitlab.com/acme/dual//go" {
		t.Fatalf("source path = %q, want the //go selector recorded", pkg.SourcePath)
	}
	info, err := ParseGitURL(pkg.SourcePath)
	if err != nil || info.Subdir != "go" {
		t.Fatalf("recorded source must resolve back to the subdir: %v / %+v", err, info)
	}
}
