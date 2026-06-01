package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func setupFakeGit(t *testing.T, mode, sourceDir string, brokenVersion bool) string {
	t.Helper()

	binDir := filepath.Join(t.TempDir(), "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}

	versionBlock := "echo 'git version 2.42.0'\nexit 0\n"
	if brokenVersion {
		versionBlock = "exit 1\n"
	}

	script := `#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "--version" ]; then
` + versionBlock + `fi
if [ "${1:-}" = "clone" ]; then
  case "${FAKE_GIT_MODE:-copy}" in
    copy)
      for last; do :; done
      dest="$last"
      mkdir -p "$dest"
      cp -R "$FAKE_GIT_SOURCE"/. "$dest"/
      exit 0
      ;;
    auth)
      echo "Authentication failed" >&2
      exit 1
      ;;
    notfound)
      echo "Repository not found" >&2
      exit 1
      ;;
    branch)
      echo "Remote branch ${FAKE_GIT_REF:-missing} not found" >&2
      exit 1
      ;;
    host)
      echo "Could not resolve host" >&2
      exit 1
      ;;
    *)
      echo "generic failure" >&2
      exit 1
      ;;
  esac
fi
echo "unexpected git args: $*" >&2
exit 1
`

	writeExecutable(t, filepath.Join(binDir, "git"), script)
	t.Setenv("FAKE_GIT_MODE", mode)
	if sourceDir != "" {
		t.Setenv("FAKE_GIT_SOURCE", sourceDir)
	}
	prependPath(t, binDir)
	return binDir
}

func TestCheckGitAvailableBrokenInstallation(t *testing.T) {
	setupFakeGit(t, "copy", "", true)

	err := checkGitAvailable()
	if err == nil || !strings.Contains(err.Error(), "git installation appears to be broken") {
		t.Fatalf("checkGitAvailable error = %v", err)
	}
}

func TestGitInstallerCloneRepositoryErrorMapping(t *testing.T) {
	tests := []struct {
		name    string
		mode    string
		ref     string
		wantErr string
	}{
		{name: "authentication failure", mode: "auth", wantErr: "authentication failed"},
		{name: "repository not found", mode: "notfound", wantErr: "repository not found"},
		{name: "branch not found", mode: "branch", ref: "feature", wantErr: "branch/tag 'feature' not found"},
		{name: "host resolution failure", mode: "host", wantErr: "could not resolve host"},
		{name: "generic failure", mode: "generic", wantErr: "git clone failed"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			setupFakeGit(t, tc.mode, "", false)
			if tc.ref != "" {
				t.Setenv("FAKE_GIT_REF", tc.ref)
			}

			gi := &GitInstaller{}
			_, err := gi.cloneRepository(&GitPackageInfo{
				CloneURL: "https://example.com/acme/repo",
				Ref:      tc.ref,
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("cloneRepository error = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestGitInstallerInstallFromGitAdditionalCoverage(t *testing.T) {
	t.Run("success and duplicate install", func(t *testing.T) {
		home := t.TempDir()
		repo := filepath.Join(t.TempDir(), "repo")
		writeTestPackage(t, repo, "name: git-installed\nversion: 1.0.0\ndescription: demo\n")
		setupFakeGit(t, "copy", repo, false)

		gi := &GitInstaller{AgentFieldHome: home}
		if err := gi.InstallFromGit("https://gitlab.com/acme/repo@feature", false); err != nil {
			t.Fatalf("InstallFromGit: %v", err)
		}

		registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
		pkg, ok := registry.Installed["git-installed"]
		if !ok {
			t.Fatalf("expected git-installed in registry")
		}
		if pkg.Source != "gitlab" {
			t.Fatalf("source = %q", pkg.Source)
		}
		if pkg.SourcePath != "https://gitlab.com/acme/repo@feature@feature" {
			t.Fatalf("source path = %q", pkg.SourcePath)
		}
		if _, err := os.Stat(filepath.Join(home, "packages", "git-installed", "main.py")); err != nil {
			t.Fatalf("installed package missing main.py: %v", err)
		}

		if err := gi.InstallFromGit("https://gitlab.com/acme/repo@feature", false); err == nil || !strings.Contains(err.Error(), "already installed") {
			t.Fatalf("expected duplicate install error, got %v", err)
		}
	})

	t.Run("invalid package structure", func(t *testing.T) {
		home := t.TempDir()
		repo := t.TempDir()
		if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("demo\n"), 0644); err != nil {
			t.Fatalf("write README: %v", err)
		}
		setupFakeGit(t, "copy", repo, false)

		gi := &GitInstaller{AgentFieldHome: home}
		err := gi.InstallFromGit("https://gitlab.com/acme/repo", true)
		if err == nil || !strings.Contains(err.Error(), "invalid package structure") {
			t.Fatalf("InstallFromGit error = %v", err)
		}
	})

	t.Run("invalid metadata", func(t *testing.T) {
		home := t.TempDir()
		repo := filepath.Join(t.TempDir(), "repo")
		writeTestPackage(t, repo, "name: bad-package\n")
		setupFakeGit(t, "copy", repo, false)

		gi := &GitInstaller{AgentFieldHome: home}
		err := gi.InstallFromGit("https://gitlab.com/acme/repo", true)
		if err == nil || !strings.Contains(err.Error(), "failed to parse package metadata") {
			t.Fatalf("InstallFromGit error = %v", err)
		}
	})
}

func TestUpdateRegistryWithGitInvalidRegistry(t *testing.T) {
	home := t.TempDir()
	registryPath := filepath.Join(home, "installed.yaml")
	if err := os.WriteFile(registryPath, []byte("{not: valid"), 0644); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	gi := &GitInstaller{AgentFieldHome: home}
	err := gi.updateRegistryWithGit(&PackageMetadata{Name: "demo", Version: "1.0.0"}, &GitPackageInfo{
		URL: "https://github.com/acme/repo",
		Ref: "main",
	}, "/tmp/src", "/tmp/dest")
	if err == nil || !strings.Contains(err.Error(), "failed to parse registry") {
		t.Fatalf("updateRegistryWithGit error = %v", err)
	}

	valid := InstallationRegistry{Installed: map[string]InstalledPackage{}}
	data, err := yaml.Marshal(valid)
	if err != nil {
		t.Fatalf("marshal registry: %v", err)
	}
	if err := os.WriteFile(registryPath, data, 0644); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	if err := gi.updateRegistryWithGit(&PackageMetadata{Name: "demo", Version: "1.0.0"}, &GitPackageInfo{
		URL: "https://bitbucket.org/acme/repo",
		Ref: "release",
	}, "/tmp/src", "/tmp/dest"); err != nil {
		t.Fatalf("updateRegistryWithGit: %v", err)
	}

	updated := readRegistryFile(t, registryPath)
	if updated.Installed["demo"].Source != "bitbucket" {
		t.Fatalf("source = %q", updated.Installed["demo"].Source)
	}
}
