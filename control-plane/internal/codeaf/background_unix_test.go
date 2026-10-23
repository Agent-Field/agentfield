//go:build unix

package codeaf

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Finding 3: exercise a real parent exit, independent process group, inherited
// home, and redirected output. No CodeAF executable or network is involved.
func TestDetachedInstallerProcess(t *testing.T) {
	root := os.Getenv("CODEAF_BACKGROUND_ROOT")
	switch os.Getenv("CODEAF_BACKGROUND_ROLE") {
	case "parent":
		if err := startBackground(filepath.Join(root, "fake-af"), root); err != nil {
			t.Fatal(err)
		}
		return
	case "worker":
		pgid, err := syscall.Getpgid(0)
		if err != nil {
			t.Fatal(err)
		}
		if pgid != os.Getpid() {
			t.Fatalf("child shares parent's process group: %d", pgid)
		}
		args := strings.Join(os.Args, " ")
		if !strings.Contains(args, "codeaf ensure --lock-wait 30s") {
			t.Fatalf("argv: %s", args)
		}
		if os.Getenv("AGENTFIELD_HOME") != root {
			t.Fatal("managed home not inherited")
		}
		fmt.Println("background stdout")
		fmt.Fprintln(os.Stderr, "background stderr")
		if err := os.WriteFile(filepath.Join(root, "ready"), []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			t.Fatal(err)
		}
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			if _, err := os.Stat(filepath.Join(root, "release")); err == nil {
				if err := os.WriteFile(filepath.Join(root, "finished"), []byte("survived"), 0600); err != nil {
					t.Fatal(err)
				}
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
		t.Fatal("child release was never written")
	}
	root = t.TempDir()
	script := "#!/bin/sh\nCODEAF_BACKGROUND_ROLE=worker exec \"$CODEAF_TEST_BINARY\" -test.run=^TestDetachedInstallerProcess$ -- \"$@\"\n"
	if err := os.WriteFile(filepath.Join(root, "fake-af"), []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	parent := exec.Command(os.Args[0], "-test.run=^TestDetachedInstallerProcess$")
	parent.Env = append(os.Environ(), "CODEAF_BACKGROUND_ROLE=parent", "CODEAF_BACKGROUND_ROOT="+root, "AGENTFIELD_HOME="+root, "CODEAF_TEST_BINARY="+os.Args[0], "GORACE=atexit_sleep_ms=0")
	started := time.Now()
	out, err := parent.CombinedOutput()
	if err != nil {
		t.Fatalf("parent: %v %s", err, out)
	}
	if time.Since(started) > 2*time.Second {
		t.Fatal("parent waited for installer")
	}
	defer os.WriteFile(filepath.Join(root, "release"), nil, 0600)
	waitFile(t, filepath.Join(root, "ready"))
	// The parent has exited; now allow its detached installer to finish.
	if err := os.WriteFile(filepath.Join(root, "release"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	waitFile(t, filepath.Join(root, "finished"))
	log, err := os.ReadFile(filepath.Join(root, "logs", "codeaf-ensure.log"))
	if err != nil || !strings.Contains(string(log), "background stdout") || !strings.Contains(string(log), "background stderr") {
		t.Fatalf("log=%s err=%v", log, err)
	}
}

func waitFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("file never created: %s", path)
}

func TestBackgroundSkipInstalledAndErrors(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("PATH", t.TempDir())
	t.Setenv("TMPDIR", t.TempDir())
	t.Setenv("CODEAF_BIN", "")
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "1")
	if err := EnsureBackground(); err != nil {
		t.Fatal(err)
	}
	noSkip(t)
	dir := filepath.Join(home, "bin")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "codeaf"), []byte("installed"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, versionMarker), []byte(Version), 0644); err != nil {
		t.Fatal(err)
	}
	if err := EnsureBackground(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, "logs")); !os.IsNotExist(err) {
		t.Fatal("installed/skip spawned an installer")
	}
	if err := startBackground(filepath.Join(home, "missing-af"), home); err == nil {
		t.Fatal("missing executable accepted")
	}
	if err := os.RemoveAll(filepath.Join(home, "logs")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "logs"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if err := startBackground("irrelevant", home); err == nil {
		t.Fatal("invalid log directory accepted")
	}
}

func TestBackgroundMissingStartsInstaller(t *testing.T) {
	noSkip(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("CODEAF_BIN", "")
	t.Setenv("PATH", t.TempDir())
	t.Setenv("TMPDIR", t.TempDir())
	failure := errors.New("self unavailable")
	if err := ensureBackground(func() (string, error) { return "", failure }); !errors.Is(err, failure) {
		t.Fatalf("self error: %v", err)
	}
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")
	if err := ensureBackground(func() (string, error) { return "unused", nil }); err == nil {
		t.Fatal("missing home accepted")
	}
	t.Setenv("AGENTFIELD_HOME", home)
	self := filepath.Join(home, "fake-af")
	if err := os.WriteFile(self, []byte("#!/bin/sh\nprintf 'installer launched\\n'\n"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := ensureBackground(func() (string, error) { return self, nil }); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		log, err := os.ReadFile(filepath.Join(home, "logs", "codeaf-ensure.log"))
		if err == nil && strings.Contains(string(log), "installer launched") {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("background installer did not run")
}
