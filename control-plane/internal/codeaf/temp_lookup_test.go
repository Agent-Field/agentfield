//go:build unix

package codeaf

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
)

func TestLookupTrustedPerUserTemp(t *testing.T) {
	for _, scenario := range []string{"owned", "public modes", "base symlink", "version symlink", "base file", "version file", "foreign owner"} {
		t.Run(scenario, func(t *testing.T) {
			t.Setenv("AGENTFIELD_HOME", t.TempDir())
			t.Setenv("CODEAF_BIN", "")
			t.Setenv("PATH", t.TempDir())
			tmp := t.TempDir()
			t.Setenv("TMPDIR", tmp)
			base := filepath.Join(tmp, fmt.Sprintf("agentfield-codeaf-%d", os.Getuid()))
			dir := filepath.Join(base, Version)
			if err := os.MkdirAll(dir, 0755); err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(dir, BinaryName("linux"))
			if err := os.WriteFile(binary, []byte("fixture"), 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, versionMarker), []byte(Version), 0644); err != nil {
				t.Fatal(err)
			}
			reject := false
			switch scenario {
			case "public modes":
				_ = os.Chmod(base, 0777)
				_ = os.Chmod(dir, 0777)
			case "base symlink", "version symlink":
				path := base
				if scenario == "version symlink" {
					path = dir
				}
				target := filepath.Join(tmp, "target")
				if err := os.Rename(path, target); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
				reject = true
			case "base file", "version file":
				path := base
				if scenario == "version file" {
					path = dir
				}
				_ = os.RemoveAll(path)
				if err := os.WriteFile(path, nil, 0600); err != nil {
					t.Fatal(err)
				}
				reject = true
			case "foreign owner":
				if os.Getuid() != 0 {
					t.Skip("changing directory owner requires root")
				}
				if err := os.Chown(base, 65534, 65534); err != nil {
					t.Fatal(err)
				}
				reject = true
			}
			got, err := Lookup("")
			if reject {
				if err == nil {
					t.Fatalf("untrusted fallback accepted: %s", got)
				}
				return
			}
			if err != nil || got != binary {
				t.Fatalf("lookup=%q, %v; want %q", got, err, binary)
			}
			for _, path := range []string{base, dir} {
				info, err := os.Lstat(path)
				if err != nil || info.Mode().Perm() != 0700 {
					t.Fatalf("fallback must be private: %s: %v, %v", path, info, err)
				}
			}
		})
	}
}

func TestLookupSkipsUntrustedTempAndWarnsOnce(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("CODEAF_BIN", "")
	tmp, pathDir := t.TempDir(), t.TempDir()
	t.Setenv("TMPDIR", tmp)
	t.Setenv("PATH", pathDir)
	base := filepath.Join(tmp, fmt.Sprintf("agentfield-codeaf-%d", os.Getuid()))
	target := t.TempDir()
	if err := os.Symlink(target, base); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(pathDir, "codeaf")
	if err := os.WriteFile(binary, []byte("fixture"), 0755); err != nil {
		t.Fatal(err)
	}
	var warnings bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&warnings)
	tempWarning = sync.Once{}
	t.Cleanup(func() {
		log.SetOutput(previous)
		tempWarning = sync.Once{}
	})
	for i := 0; i < 2; i++ {
		if got, err := Lookup(""); err != nil || got != binary {
			t.Fatalf("PATH fallback=%q, %v", got, err)
		}
	}
	if strings.Count(warnings.String(), "warning:") != 1 {
		t.Fatalf("warnings=%s", warnings.String())
	}
	entries, err := os.ReadDir(target)
	if err != nil || len(entries) != 0 {
		t.Fatalf("untrusted target was accessed: entries=%v err=%v", entries, err)
	}
}

type foreignDirectoryInfo struct{ os.FileInfo }

func (info foreignDirectoryInfo) Sys() any {
	return &syscall.Stat_t{Uid: uint32(os.Getuid()) + 1}
}

func TestTempDirectoryRejectsForeignOwnerBeforeChmod(t *testing.T) {
	path := t.TempDir()
	if err := os.Chmod(path, 0755); err != nil {
		t.Fatal(err)
	}
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := trustTempDirectory(path, foreignDirectoryInfo{info}); err == nil {
		t.Fatal("foreign-owned fallback accepted")
	}
	info, err = os.Lstat(path)
	if err != nil || info.Mode().Perm() != 0755 {
		t.Fatalf("changed foreign-owned directory: %v, %v", info, err)
	}
}
