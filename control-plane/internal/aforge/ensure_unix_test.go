//go:build unix

package aforge

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnsureFailsToCreateBinDirectory(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "bin"), []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home})
	if err == nil || !strings.Contains(err.Error(), "create aforge bin directory") {
		t.Fatalf("error = %v", err)
	}
}

func TestEnsureFailsToLockInstallation(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root can write through directory permissions")
	}
	home := t.TempDir()
	binDir := filepath.Join(home, "bin")
	if err := os.Mkdir(binDir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(binDir, 0o700); err != nil {
			t.Errorf("restore bin directory permissions: %v", err)
		}
	})
	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home})
	if err == nil || !strings.Contains(err.Error(), "lock aforge installation") {
		t.Fatalf("error = %v", err)
	}
}

func TestLockAforgeRejectsDirectoryLockPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := lockAforge(path); err == nil {
		t.Fatal("lockAforge error = nil")
	}
}

func TestEnsureReportsRenameAndMarkerFailures(t *testing.T) {
	t.Run("rename", func(t *testing.T) {
		home := t.TempDir()
		if err := os.MkdirAll(filepath.Join(home, "bin", "aforge", "child"), 0o755); err != nil {
			t.Fatal(err)
		}
		server, _ := releaseServer(t, []byte("fixture"), "")
		err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
		if err == nil || !strings.Contains(err.Error(), "install aforge") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("marker", func(t *testing.T) {
		home := t.TempDir()
		if err := os.MkdirAll(filepath.Join(home, "bin", versionMarker), 0o755); err != nil {
			t.Fatal(err)
		}
		server, _ := releaseServer(t, []byte("fixture"), "")
		err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home, BaseURL: server.URL})
		if err == nil || !strings.Contains(err.Error(), "record aforge version") {
			t.Fatalf("error = %v", err)
		}
	})
}
