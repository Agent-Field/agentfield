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

	installer := &GitInstaller{AgentFieldHome: home}
	if err := installer.InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}
	if installer.InstalledName() != "dual-node-go" {
		t.Fatalf("installed name = %q, want successor", installer.InstalledName())
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

// writeMarkedSubdirPackage writes a subdirectory package that shares the root
// manifest's name — the shape a node takes when it renames itself in place —
// carrying a marker file so a test can tell whose files ended up installed.
func writeMarkedSubdirPackage(t *testing.T, dir, name, marker string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: " + name + "\nversion: 2.0.0\nentrypoint:\n  start: bin/" + name + "\n"
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, marker), []byte("successor\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Contract: a successor that carries the SAME name as the package it retires
// replaces it in place. The redirect already warned the user, so it must not
// fail with "already installed (use --force)" — the case a node hits when it
// renames itself to the name its predecessor held.
func TestInstallFromGit_SupersededSameNameReplacesInPlace(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeMarkedSubdirPackage(t, filepath.Join(repo, "go"), "dual-node", "successor.txt")
	setupFakeGit(t, "copy", repo, false)

	oldDir := seedInstalled(t, home, "dual-node")
	if err := os.WriteFile(filepath.Join(oldDir, "predecessor.txt"), []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	installer := &GitInstaller{AgentFieldHome: home}
	if err := installer.InstallFromGit("https://gitlab.com/acme/dual", false); err != nil {
		t.Fatalf("same-name supersede must not need --force: %v", err)
	}
	if installer.InstalledName() != "dual-node" {
		t.Fatalf("installed name = %q, want shared name", installer.InstalledName())
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if len(registry.Installed) != 1 {
		t.Fatalf("in-place replacement must leave one registry entry, got %v", registry.Installed)
	}
	pkg, ok := registry.Installed["dual-node"]
	if !ok {
		t.Fatalf("the shared name must still be installed, got %v", registry.Installed)
	}
	if pkg.Version != "2.0.0" {
		t.Fatalf("registry still describes the predecessor: version %q", pkg.Version)
	}
	if pkg.SourcePath != "https://gitlab.com/acme/dual//go" {
		t.Fatalf("source path = %q, want successor source", pkg.SourcePath)
	}
	if _, err := os.Stat(filepath.Join(oldDir, "successor.txt")); err != nil {
		t.Fatalf("successor's files are not installed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(oldDir, "predecessor.txt")); !os.IsNotExist(err) {
		t.Fatalf("predecessor's files must not survive the replace, stat err = %v", err)
	}
}

// Contract: a plain install reports the manifest name without a redirect.
func TestInstallFromGit_PlainInstallReportsManifestName(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, "name: plain-node\nversion: 1.0.0\n")
	setupFakeGit(t, "copy", repo, false)

	installer := &GitInstaller{AgentFieldHome: home}
	if err := installer.InstallFromGit("https://gitlab.com/acme/plain", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}
	if installer.InstalledName() != "plain-node" {
		t.Fatalf("installed name = %q, want plain-node", installer.InstalledName())
	}
}

// Contract: node-scoped secrets survive a same-name replace. They never move —
// the scope name is unchanged — so the risk is the retire path deleting them.
func TestInstallFromGit_SupersededSameNameKeepsNodeScopedSecrets(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, supersededRoot)
	writeMarkedSubdirPackage(t, filepath.Join(repo, "go"), "dual-node", "successor.txt")
	setupFakeGit(t, "copy", repo, false)

	seedInstalled(t, home, "dual-node")
	store, err := NewSecretStore(home)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Set("dual-node", "KEPT", "node-value"); err != nil {
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
	values, err := after.load("dual-node")
	if err != nil {
		t.Fatal(err)
	}
	if values["KEPT"] != "node-value" {
		t.Fatalf("node-scoped secret lost in an in-place replace: %v", values)
	}
}

// Contract: an install that fails after the destination has been cleared puts
// the previously installed package back. Without this a replace that dies in
// the dependency step — a missing toolchain is enough — leaves the user with
// neither their old node nor a working new one.
func TestInstallFromGit_FailedReinstallRestoresPreviousPackage(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	// language: typescript with no package.json fails in installDependencies,
	// which runs after the destination has already been cleared and copied.
	writeTestPackage(t, repo, "name: solo-node\nversion: 2.0.0\nlanguage: typescript\n")
	setupFakeGit(t, "copy", repo, false)

	oldDir := seedInstalled(t, home, "solo-node")
	if err := os.WriteFile(filepath.Join(oldDir, "predecessor.txt"), []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&GitInstaller{AgentFieldHome: home}).
		InstallFromGit("https://gitlab.com/acme/solo", true)
	if err == nil {
		t.Fatal("expected the install to fail in the dependency step")
	}

	if _, statErr := os.Stat(filepath.Join(oldDir, "predecessor.txt")); statErr != nil {
		t.Fatalf("previously installed package was not restored: %v", statErr)
	}
	entries, readErr := os.ReadDir(filepath.Join(home, "packages"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			t.Fatalf("a backup was left behind in packages/: %s", e.Name())
		}
	}
}

// Contract: stashing is a no-op when there is nothing installed yet, and a
// discarded stash leaves no residue — the first-install path must not be
// burdened with cleanup that does not apply to it.
func TestStashExistingPackage(t *testing.T) {
	home := t.TempDir()
	missing := filepath.Join(home, "packages", "never-installed")
	backup, err := stashExistingPackage(missing)
	if err != nil {
		t.Fatalf("stashing a missing package must succeed: %v", err)
	}
	backup.restore() // must not recreate anything
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Fatalf("restoring a no-op stash created something, stat err = %v", err)
	}

	present := filepath.Join(home, "packages", "installed")
	if err := os.MkdirAll(present, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(present, "keep.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	backup, err = stashExistingPackage(present)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(present); !os.IsNotExist(err) {
		t.Fatalf("stashing must move the directory aside, stat err = %v", err)
	}
	backup.restore()
	if _, err := os.Stat(filepath.Join(present, "keep.txt")); err != nil {
		t.Fatalf("restore did not put the package back: %v", err)
	}

	backup, err = stashExistingPackage(present)
	if err != nil {
		t.Fatal(err)
	}
	backup.discard()
	entries, err := os.ReadDir(filepath.Join(home, "packages"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("discard left residue in packages/: %v", entries)
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
