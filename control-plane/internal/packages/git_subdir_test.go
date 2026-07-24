package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Contract: the //subdir selector splits cleanly off git URLs in all the
// spellings users reach for, and composes with @ref.
func TestParseGitURL_SubdirSelector(t *testing.T) {
	cases := []struct {
		url      string
		cloneURL string
		subdir   string
		ref      string
	}{
		{"https://github.com/Agent-Field/pr-af//go", "https://github.com/Agent-Field/pr-af", "go", ""},
		{"https://github.com/Agent-Field/pr-af//go@main", "https://github.com/Agent-Field/pr-af", "go", "main"},
		{"https://github.com/Agent-Field/pr-af//nested/dir", "https://github.com/Agent-Field/pr-af", "nested/dir", ""},
		{"https://github.com/Agent-Field/pr-af@main", "https://github.com/Agent-Field/pr-af", "", "main"},
		{"https://github.com/Agent-Field/pr-af", "https://github.com/Agent-Field/pr-af", "", ""},
		{"git@github.com:Agent-Field/pr-af//go", "git@github.com:Agent-Field/pr-af", "go", ""},
	}
	for _, c := range cases {
		info, err := ParseGitURL(c.url)
		if err != nil {
			t.Fatalf("ParseGitURL(%q): %v", c.url, err)
		}
		if info.CloneURL != c.cloneURL || info.Subdir != c.subdir || info.Ref != c.ref {
			t.Errorf(
				"ParseGitURL(%q) = clone %q subdir %q ref %q, want %q %q %q",
				c.url, info.CloneURL, info.Subdir, info.Ref, c.cloneURL, c.subdir, c.ref,
			)
		}
		if info.URL != c.url {
			t.Errorf("ParseGitURL(%q) should keep the original URL, got %q", c.url, info.URL)
		}
	}
}

func writeSubdirManifest(t *testing.T, dir, name string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: " + name + "\nversion: 0.1.0\nentrypoint:\n  start: python -m " + name + "\n"
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Contract: a URL //subdir selector resolves exactly there (no walking),
// rejects escapes, and errors helpfully when the subdir carries no manifest —
// via the same resolver the --path flag uses.
func TestResolvePackageRoot_UrlSelector(t *testing.T) {
	clone := t.TempDir()
	writeSubdirManifest(t, clone, "root-node")
	writeSubdirManifest(t, filepath.Join(clone, "go"), "root-node-go")

	got, err := (&GitInstaller{Subdir: "go"}).resolvePackageRoot(clone)
	if err != nil {
		t.Fatalf("resolvePackageRoot: %v", err)
	}
	if want := filepath.Join(clone, "go"); got != want {
		t.Fatalf("resolvePackageRoot = %q, want %q", got, want)
	}

	if _, err := (&GitInstaller{Subdir: "missing"}).resolvePackageRoot(clone); err == nil {
		t.Fatal("expected error for a subdir without a manifest")
	}
	if _, err := (&GitInstaller{Subdir: "../outside"}).resolvePackageRoot(clone); err == nil {
		t.Fatal("expected error for a subdir escaping the repository")
	}
}

// Contract: uninstalling a node removes its node-scoped secrets file while
// leaving the shared global scope untouched.
func TestUninstallRemovesNodeScopedSecrets(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "doomed-node")
	writeSubdirManifest(t, pkgDir, "doomed-node")

	store, err := NewSecretStore(home)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Set("doomed-node", "NODE_KEY", "v"); err != nil {
		t.Fatal(err)
	}
	if err := store.Set("global", "SHARED_KEY", "v"); err != nil {
		t.Fatal(err)
	}

	registry := &InstallationRegistry{Installed: map[string]InstalledPackage{
		"doomed-node": {Name: "doomed-node", Path: pkgDir, Status: "stopped"},
	}}
	pu := &PackageUninstaller{AgentFieldHome: home}
	if err := pu.saveRegistry(registry); err != nil {
		t.Fatal(err)
	}

	if err := pu.UninstallPackage("doomed-node"); err != nil {
		t.Fatalf("UninstallPackage: %v", err)
	}

	if _, err := os.Stat(filepath.Join(home, "secrets", "doomed-node.enc")); !os.IsNotExist(err) {
		t.Fatalf("node-scoped secrets file should be gone, stat err = %v", err)
	}
	if _, err := os.Stat(filepath.Join(home, "secrets", "global.enc")); err != nil {
		t.Fatalf("global secrets must survive uninstall: %v", err)
	}
	if _, err := os.Stat(pkgDir); !os.IsNotExist(err) {
		t.Fatalf("package dir should be gone, stat err = %v", err)
	}
}

// Contract: `af install <repo>//<subdir>` installs the subdirectory's package
// (registry keyed by ITS manifest name), not the repo-root one — the two can
// then coexist side by side.
func TestInstallFromGit_SubdirSelector(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, "name: dual-node\nversion: 1.0.0\n")
	writeSubdirManifest(t, filepath.Join(repo, "go"), "dual-node-go")
	setupFakeGit(t, "copy", repo, false)

	gi := &GitInstaller{AgentFieldHome: home}
	if err := gi.InstallFromGit("https://gitlab.com/acme/dual//go", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	pkg, ok := registry.Installed["dual-node-go"]
	if !ok {
		t.Fatalf("expected dual-node-go in registry, got %v", registry.Installed)
	}
	if _, exists := registry.Installed["dual-node"]; exists {
		t.Fatal("root package must not be installed by a //go install")
	}
	if pkg.SourcePath != "https://gitlab.com/acme/dual//go" {
		t.Fatalf("source path = %q", pkg.SourcePath)
	}
	if _, err := os.Stat(filepath.Join(home, "packages", "dual-node-go", "agentfield-package.yaml")); err != nil {
		t.Fatalf("installed subdir package missing manifest: %v", err)
	}

	// A subdir without a manifest fails with a pointed error. Fresh installer:
	// the service constructs one per install, and InstallFromGit folds the URL
	// selector into gi.Subdir.
	err := (&GitInstaller{AgentFieldHome: home}).InstallFromGit(
		"https://gitlab.com/acme/dual//nope", false,
	)
	if err == nil || !strings.Contains(err.Error(), "no agentfield-package.yaml found") {
		t.Fatalf("expected missing-manifest subdir error, got %v", err)
	}
}
