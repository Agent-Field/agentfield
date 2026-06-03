package packages

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func createZipFile(t *testing.T, path string, files map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create zip: %v", err)
	}
	defer f.Close()

	w := zip.NewWriter(f)
	for name, body := range files {
		entry, err := w.Create(name)
		if err != nil {
			t.Fatalf("create entry: %v", err)
		}
		if _, err := entry.Write([]byte(body)); err != nil {
			t.Fatalf("write entry: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
}

func TestGitHubURLHelpers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		url         string
		ok          bool
		owner       string
		repo        string
		ref         string
		archivePath string
	}{
		{"github.com/acme/repo", true, "acme", "repo", "main", "/archive/refs/heads/main.zip"},
		{"acme/repo@feature", true, "acme", "repo", "feature", "/archive/refs/heads/feature.zip"},
		{"https://github.com/acme/repo@v1.2.3", true, "acme", "repo", "v1.2.3", "/archive/v1.2.3.zip"},
		{"https://github.com/acme/repo@abcdef1", true, "acme", "repo", "abcdef1", "/archive/abcdef1.zip"},
		{"bad input", false, "", "", "", ""},
	}

	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			if got := IsGitHubURL(tc.url); got != tc.ok {
				t.Fatalf("IsGitHubURL(%q)=%v want %v", tc.url, got, tc.ok)
			}
			info, err := ParseGitHubURL(tc.url)
			if !tc.ok {
				if err == nil {
					t.Fatalf("expected parse error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseGitHubURL: %v", err)
			}
			if info.Owner != tc.owner || info.Repo != tc.repo || info.Ref != tc.ref {
				t.Fatalf("unexpected info: %+v", info)
			}
			if !strings.Contains(info.ArchiveURL, tc.archivePath) {
				t.Fatalf("archive URL %q missing %q", info.ArchiveURL, tc.archivePath)
			}
		})
	}

	if !isCommitHash("abcdef1") || !isCommitHash("ABCDEF1234") {
		t.Fatalf("expected commit hashes to be detected")
	}
	if isCommitHash("xyz") || isCommitHash(strings.Repeat("a", 41)) {
		t.Fatalf("invalid hashes should not be detected")
	}
}

func TestGitHubInstallerZipDownloadAndRegistry(t *testing.T) {
	t.Parallel()

	home := t.TempDir()
	gi := &GitHubInstaller{AgentFieldHome: home}

	sourceZip := filepath.Join(t.TempDir(), "archive.zip")
	createZipFile(t, sourceZip, map[string]string{
		"repo-main/agentfield-package.yaml": "name: gh-demo\nversion: 2.0.0\n",
		"repo-main/main.py":                 "print('ok')\n",
		"repo-main/nested/readme.txt":       "demo\n",
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, sourceZip)
	}))
	defer server.Close()

	info := &GitHubPackageInfo{
		Owner:      "acme",
		Repo:       "repo",
		Ref:        "main",
		ArchiveURL: server.URL + "/archive.zip",
	}

	extracted, err := gi.downloadAndExtract(info)
	if err != nil {
		t.Fatalf("downloadAndExtract: %v", err)
	}

	root, err := gi.findPackageRoot(extracted)
	if err != nil {
		t.Fatalf("findPackageRoot: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "main.py")); err != nil {
		t.Fatalf("expected extracted main.py: %v", err)
	}

	metadata, err := gi.parsePackageMetadata(root)
	if err != nil {
		t.Fatalf("parsePackageMetadata: %v", err)
	}
	dest := filepath.Join(home, "packages", "gh-demo")
	if err := gi.updateRegistryWithGitHub(metadata, info, root, dest); err != nil {
		t.Fatalf("updateRegistryWithGitHub: %v", err)
	}
	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if got := registry.Installed["gh-demo"].SourcePath; got != "acme/repo@main" {
		t.Fatalf("source path=%q", got)
	}

	notFoundInfo := &GitHubPackageInfo{Owner: "acme", Repo: "repo", Ref: "missing", ArchiveURL: server.URL + "/missing"}
	notFoundServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer notFoundServer.Close()
	notFoundInfo.ArchiveURL = notFoundServer.URL
	if _, err := gi.downloadAndExtract(notFoundInfo); err == nil || !strings.Contains(err.Error(), "repository or reference not found") {
		t.Fatalf("expected not found error, got %v", err)
	}
}

func TestGitHubInstallerExtractZipRejectsTraversal(t *testing.T) {
	t.Parallel()

	gi := &GitHubInstaller{}
	zipPath := filepath.Join(t.TempDir(), "bad.zip")
	createZipFile(t, zipPath, map[string]string{
		"../escape.txt": "bad",
	})

	err := gi.extractZip(zipPath, filepath.Join(t.TempDir(), "dest"))
	if err == nil || !strings.Contains(err.Error(), "invalid file path") {
		t.Fatalf("expected traversal error, got %v", err)
	}

	empty := t.TempDir()
	if _, err := gi.findPackageRoot(empty); err == nil || !strings.Contains(err.Error(), "agentfield-package.yaml not found") {
		t.Fatalf("expected missing package root error, got %v", err)
	}

	missingMainZip := filepath.Join(t.TempDir(), "missing-main.zip")
	createZipFile(t, missingMainZip, map[string]string{
		"repo/agentfield-package.yaml": "name: gh-demo\nversion: 2.0.0\n",
	})
	dest := filepath.Join(t.TempDir(), "out")
	if err := gi.extractZip(missingMainZip, dest); err != nil {
		t.Fatalf("extractZip: %v", err)
	}
	if _, err := gi.findPackageRoot(dest); err == nil || !strings.Contains(err.Error(), "main.py not found") {
		t.Fatalf("expected main.py error, got %v", err)
	}

	badZipServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintln(w, "not a zip")
	}))
	defer badZipServer.Close()
	if _, err := gi.downloadAndExtract(&GitHubPackageInfo{Owner: "a", Repo: "b", Ref: "c", ArchiveURL: badZipServer.URL}); err == nil || !strings.Contains(err.Error(), "failed to extract archive") {
		t.Fatalf("expected extract error, got %v", err)
	}
}

func TestGitHubInstallerInvalidURL(t *testing.T) {
	gi := &GitHubInstaller{AgentFieldHome: t.TempDir()}
	if err := gi.InstallFromGitHub("not github", false); err == nil || !strings.Contains(err.Error(), "failed to parse GitHub URL") {
		t.Fatalf("expected parse error, got %v", err)
	}
}

func TestGitHubInstallerInstallFromGitHubSuccess(t *testing.T) {
	home := t.TempDir()

	sourceZip := filepath.Join(t.TempDir(), "install.zip")
	createZipFile(t, sourceZip, map[string]string{
		"repo-main/agentfield-package.yaml": "name: installed-gh\nversion: 1.0.0\n",
		"repo-main/main.py":                 "print('ok')\n",
	})
	zipBytes, err := os.ReadFile(sourceZip)
	if err != nil {
		t.Fatalf("read zip: %v", err)
	}

	gi := &GitHubInstaller{
		AgentFieldHome: home,
		HTTPClient: &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if !strings.Contains(r.URL.Host, "github.com") {
					return nil, fmt.Errorf("unexpected host: %s", r.URL.Host)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader(zipBytes)),
				}, nil
			}),
		},
	}

	if err := gi.InstallFromGitHub("acme/repo@main", true); err != nil {
		t.Fatalf("InstallFromGitHub: %v", err)
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if _, ok := registry.Installed["installed-gh"]; !ok {
		t.Fatalf("expected installed-gh in registry")
	}
}
