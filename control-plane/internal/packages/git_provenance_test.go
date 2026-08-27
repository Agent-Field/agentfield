package packages

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func boolPointer(value bool) *bool { return &value }

// Contract: a git install records the exact commit resolved by the shallow
// clone and the explicit source ref independently of the source string.
func TestGitInstallRecordsResolvedCommitAndRef(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, "name: provenance-demo\nversion: 1.0.0\n")
	setupFakeGit(t, "copy", repo, false)
	const commit = "89abcdef0123456789abcdef0123456789abcdef"
	t.Setenv("FAKE_GIT_COMMIT", commit)

	installer := &GitInstaller{AgentFieldHome: home}
	if err := installer.InstallFromGit("https://gitlab.com/acme/provenance@v1.2.3", false); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}

	entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["provenance-demo"]
	if entry.Commit != commit {
		t.Fatalf("commit = %q, want %q", entry.Commit, commit)
	}
	if entry.Ref != "v1.2.3" {
		t.Fatalf("ref = %q, want v1.2.3", entry.Ref)
	}
	if !entry.AutoUpdateEnabled() {
		t.Fatal("new git installs must default auto_update to true")
	}
	if entry.UpdatedAt == "" {
		t.Fatal("updated_at must be recorded")
	}
}

// Contract: reinstalling updates provenance without silently unpausing a
// package for which auto_update: false was explicitly persisted.
func TestGitReinstallPreservesDisabledAutoUpdate(t *testing.T) {
	home := t.TempDir()
	port := 8123
	registry := InstallationRegistry{Installed: map[string]InstalledPackage{
		"demo": {
			Name:         "demo",
			AutoUpdate:   boolPointer(false),
			Status:       "stopped",
			DesiredState: DesiredStateRunning,
			Runtime:      RuntimeInfo{Port: &port},
		},
	}}
	data, err := yaml.Marshal(&registry)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o600); err != nil {
		t.Fatal(err)
	}

	info := &GitPackageInfo{
		URL:      "https://github.com/acme/demo@main",
		CloneURL: "https://github.com/acme/demo",
		Ref:      "main",
		Commit:   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	if err := (&GitInstaller{AgentFieldHome: home}).updateRegistryWithGit(
		&PackageMetadata{Name: "demo", Version: "2.0.0"}, info, t.TempDir(), filepath.Join(home, "packages", "demo")); err != nil {
		t.Fatalf("updateRegistryWithGit: %v", err)
	}

	entry := readRegistryFile(t, filepath.Join(home, "installed.yaml")).Installed["demo"]
	if entry.AutoUpdate == nil || *entry.AutoUpdate {
		t.Fatalf("auto_update = %v, want explicit false", entry.AutoUpdate)
	}
	if entry.DesiredState != DesiredStateRunning || entry.Runtime.Port == nil || *entry.Runtime.Port != port {
		t.Fatalf("reinstall lost running intent or restore port: %+v", entry)
	}
}
