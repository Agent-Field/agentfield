
package packages

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGitHubInstallerErrorCasesFinal(t *testing.T) {
	t.Parallel()

	// --- updateRegistryWithGitHub errors ---
	t.Run("updateRegistry-ReadOnly", func(t *testing.T) {
		home := t.TempDir()
		gi := &GitHubInstaller{AgentFieldHome: home}
		info := &GitHubPackageInfo{Owner: "a", Repo: "b", Ref: "c"}
		metadata := &PackageMetadata{Name: "demo"}
		registryPath := filepath.Join(home, "installed.yaml")

		if err := os.WriteFile(registryPath, []byte{}, 0644); err != nil {
			t.Fatalf("write file: %v", err)
		}
		if err := os.Chmod(registryPath, 0444); err != nil {
			t.Fatalf("chmod: %v", err)
		}

		err := gi.updateRegistryWithGitHub(metadata, info, "", "")
		if err == nil || !(strings.Contains(err.Error(), "permission denied") || strings.Contains(err.Error(), "read-only file system")) {
			t.Fatalf("expected permission error, got %v", err)
		}
		_ = os.Chmod(registryPath, 0644) // cleanup
	})

	t.Run("updateRegistry-BadYAML", func(t *testing.T) {
		home := t.TempDir()
		gi := &GitHubInstaller{AgentFieldHome: home}
		info := &GitHubPackageInfo{Owner: "a", Repo: "b", Ref: "c"}
		metadata := &PackageMetadata{Name: "demo"}
		registryPath := filepath.Join(home, "installed.yaml")
		if err := os.WriteFile(registryPath, []byte("bad: yaml: here"), 0644); err != nil {
			t.Fatalf("write bad yaml: %v", err)
		}
		err := gi.updateRegistryWithGitHub(metadata, info, "", "")
		if err == nil || !strings.Contains(err.Error(), "failed to parse registry") {
			t.Fatalf("expected parse error, got %v", err)
		}
	})

	// --- extractZip errors ---
	t.Run("extractZip-SrcNotExist", func(t *testing.T) {
		gi := &GitHubInstaller{AgentFieldHome: t.TempDir()}
		err := gi.extractZip("nonexistent.zip", t.TempDir())
		if err == nil {
			t.Fatalf("expected error for non-existent zip")
		}
	})

	t.Run("extractZip-DestNotCreatable", func(t *testing.T) {
		gi := &GitHubInstaller{AgentFieldHome: t.TempDir()}
		zipPath := filepath.Join(t.TempDir(), "test.zip")
		createZipFile(t, zipPath, map[string]string{"file.txt": "content"})
		destDir := filepath.Join(t.TempDir(), "dest")
		if err := os.WriteFile(destDir, []byte{}, 0644); err != nil {
			t.Fatalf("create file: %v", err)
		}
		err := gi.extractZip(zipPath, destDir)
		if err == nil {
			t.Fatalf("expected error for non-creatable destination")
		}
	})

	// --- downloadAndExtract errors ---
	t.Run("downloadAndExtract-Server500", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "server error", http.StatusInternalServerError)
		}))
		defer server.Close()
		gi := &GitHubInstaller{}
		info := &GitHubPackageInfo{ArchiveURL: server.URL}
		_, err := gi.downloadAndExtract(info)
		if err == nil || !strings.Contains(err.Error(), "HTTP 500") {
			t.Fatalf("expected 500 error, got %v", err)
		}
	})

	t.Run("downloadAndExtract-CopyFails", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("short body"))
		}))
		defer server.Close()
		
		gi := &GitHubInstaller{}
		info := &GitHubPackageInfo{ArchiveURL: server.URL}

		_, err := gi.downloadAndExtract(info)
		if err == nil || !strings.Contains(err.Error(), "unexpected EOF") {
			t.Fatalf("expected save archive error due to unexpected EOF, got %v", err)
		}
	})

	t.Run("findPackageRoot-WalkError", func(t *testing.T) {
		gi := &GitHubInstaller{}
		dir := t.TempDir()
		// Create a file that we can't read to cause filepath.Walk to error
		unreadable := filepath.Join(dir, "unreadable")
		if err := os.Mkdir(unreadable, 0000); err != nil {
			t.Fatalf("mkdir failed: %v", err)
		}
		_, err := gi.findPackageRoot(dir)
		if err == nil {
			t.Fatalf("expected error from findPackageRoot, got nil")
		}
	})

	t.Run("ParseGithubURL-Invalid", func(t *testing.T) {
		_, err := ParseGitHubURL("a/b/c/d")
		if err == nil {
			t.Fatal("expected error parsing invalid url")
		}
	})
}
