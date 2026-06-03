package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGitURLHelpers(t *testing.T) {
	t.Parallel()

	cases := []struct {
		url     string
		isGit   bool
		ref     string
		clone   string
		wantErr bool
	}{
		{"https://github.com/acme/repo", true, "", "https://github.com/acme/repo", false},
		{"https://github.com/acme/repo@feature", true, "feature", "https://github.com/acme/repo", false},
		{"git@github.com:acme/repo.git", true, "", "git@github.com:acme/repo.git", false},
		{"https://token@github.com/acme/repo", true, "", "https://token@github.com/acme/repo", false},
		{"not-a-url", false, "", "not-a-url", false},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if got := IsGitURL(tc.url); got != tc.isGit {
				t.Fatalf("IsGitURL(%q)=%v want %v", tc.url, got, tc.isGit)
			}
			info, err := ParseGitURL(tc.url)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseGitURL: %v", err)
			}
			if info.Ref != tc.ref || info.CloneURL != tc.clone {
				t.Fatalf("unexpected parse result: %+v", info)
			}
		})
	}

	if !isHTTPSGitURL("https://example.com/repo") {
		t.Fatalf("expected HTTPS git-like URL")
	}
	if isHTTPSGitURL("https://example.com/repo/") {
		t.Fatalf("trailing slash should not be considered git-like")
	}
}

func TestGitInstallerFindPackageRootAndRegistry(t *testing.T) {
	t.Parallel()

	home := t.TempDir()
	gi := &GitInstaller{AgentFieldHome: home}

	repo := filepath.Join(t.TempDir(), "repo", "nested")
	writeTestPackage(t, repo, "name: git-demo\nversion: 1.0.0\n")

	root, err := gi.findPackageRoot(filepath.Dir(filepath.Dir(repo)))
	if err != nil {
		t.Fatalf("findPackageRoot: %v", err)
	}
	if root != repo {
		t.Fatalf("root=%q want %q", root, repo)
	}

	missingMain := filepath.Join(t.TempDir(), "missing-main")
	if err := os.MkdirAll(missingMain, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(missingMain, "agentfield-package.yaml"), []byte("name: bad\nversion: 1.0.0\n"), 0644); err != nil {
		t.Fatalf("write yaml: %v", err)
	}
	if _, err := gi.findPackageRoot(missingMain); err == nil || !strings.Contains(err.Error(), "main.py not found") {
		t.Fatalf("expected main.py error, got %v", err)
	}

	emptyRepo := t.TempDir()
	if _, err := gi.findPackageRoot(emptyRepo); err == nil || !strings.Contains(err.Error(), "agentfield-package.yaml not found") {
		t.Fatalf("expected package yaml error, got %v", err)
	}

	metadata, err := gi.parsePackageMetadata(repo)
	if err != nil {
		t.Fatalf("parsePackageMetadata: %v", err)
	}
	info := &GitPackageInfo{URL: "https://gitlab.com/acme/repo", Ref: "main"}
	dest := filepath.Join(home, "packages", "git-demo")
	if err := gi.updateRegistryWithGit(metadata, info, repo, dest); err != nil {
		t.Fatalf("updateRegistryWithGit: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	pkg := registry.Installed["git-demo"]
	if pkg.Source != "gitlab" {
		t.Fatalf("source=%q want gitlab", pkg.Source)
	}
	if pkg.SourcePath != "https://gitlab.com/acme/repo@main" {
		t.Fatalf("source path=%q", pkg.SourcePath)
	}
}

func TestGitInstallerErrorsWithoutGit(t *testing.T) {
	t.Setenv("PATH", "")

	if err := checkGitAvailable(); err == nil || !strings.Contains(err.Error(), "git is required") {
		t.Fatalf("expected missing git error, got %v", err)
	}

	gi := &GitInstaller{AgentFieldHome: t.TempDir()}
	if err := gi.InstallFromGit("https://github.com/acme/repo", false); err == nil || !strings.Contains(err.Error(), "git is required") {
		t.Fatalf("expected InstallFromGit missing git error, got %v", err)
	}

	_, err := gi.cloneRepository(&GitPackageInfo{CloneURL: "https://github.com/acme/repo"})
	if err == nil || !strings.Contains(err.Error(), "git clone failed") {
		t.Fatalf("expected cloneRepository failure, got %v", err)
	}
}
