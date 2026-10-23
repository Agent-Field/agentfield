//go:build unix

package codeaf

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func backgroundFixture(t *testing.T, exit int) (string, string) {
	t.Helper()
	noSkip(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("PATH", t.TempDir())
	t.Setenv("TMPDIR", t.TempDir())
	t.Setenv("CODEAF_BIN", "")
	self := filepath.Join(home, "fake-af")
	script := fmt.Sprintf("#!/bin/sh\nprintf x >> \"$AGENTFIELD_HOME/starts\"\nprintf '%%s' $$ > \"$AGENTFIELD_HOME/pid\"\nwhile [ ! -f \"$AGENTFIELD_HOME/release\" ]; do /bin/sleep 0.01; done\nexit %d\n", exit)
	if err := os.WriteFile(self, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.WriteFile(filepath.Join(home, "release"), nil, 0600)
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			if _, err := os.Stat(filepath.Join(home, "bin", backgroundLease)); os.IsNotExist(err) {
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	})
	return home, self
}

func TestBackgroundFailureRecordsCooldownAndReapsChild(t *testing.T) {
	home, self := backgroundFixture(t, 1)
	if err := ensureBackground(func() (string, error) { return self, nil }); err != nil {
		t.Fatal(err)
	}
	waitFile(t, filepath.Join(home, "pid"))
	if err := os.WriteFile(filepath.Join(home, "release"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	waitFile(t, filepath.Join(home, "bin", backgroundFailure))
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(home, "bin", backgroundLease)); os.IsNotExist(err) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	pidBytes, err := os.ReadFile(filepath.Join(home, "pid"))
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(string(pidBytes))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := syscall.Wait4(pid, nil, syscall.WNOHANG, nil); err != syscall.ECHILD {
		t.Fatalf("background child was not reaped: %v", err)
	}
	for i := 0; i < 6; i++ {
		if err := ensureBackground(func() (string, error) { return self, nil }); err != nil {
			t.Fatal(err)
		}
	}
	starts, _ := os.ReadFile(filepath.Join(home, "starts"))
	if string(starts) != "x" {
		t.Fatalf("failed installer ignored cooldown: %q", starts)
	}
}

func TestBackgroundRapidCallsShareLease(t *testing.T) {
	home, self := backgroundFixture(t, 0)
	for i := 0; i < 6; i++ {
		if err := ensureBackground(func() (string, error) { return self, nil }); err != nil {
			t.Fatal(err)
		}
	}
	waitFile(t, filepath.Join(home, "starts"))
	// Let every incorrectly spawned child reach the fixture while the first is held.
	time.Sleep(100 * time.Millisecond)
	starts, err := os.ReadFile(filepath.Join(home, "starts"))
	if err != nil || string(starts) != "x" {
		t.Fatalf("six rapid launches spawned %d installers: %v", len(starts), err)
	}
	lease, err := os.ReadFile(filepath.Join(home, "bin", ".codeaf.background"))
	if err != nil || !strings.Contains(string(lease), " ") {
		t.Fatalf("missing pid + start-time lease: %q, %v", lease, err)
	}
}

func TestBackgroundFailureCooldownAndStaleLease(t *testing.T) {
	for _, scenario := range []string{"cooldown", "expired failure", "fresh lease", "stale lease"} {
		t.Run(scenario, func(t *testing.T) {
			home, self := backgroundFixture(t, 1)
			bin := filepath.Join(home, "bin")
			if err := os.MkdirAll(bin, 0755); err != nil {
				t.Fatal(err)
			}
			path, when := ".codeaf.lastfail", time.Now()
			wantStart := false
			switch scenario {
			case "expired failure":
				when, wantStart = when.Add(-11*time.Minute), true
			case "fresh lease":
				path = ".codeaf.background"
			case "stale lease":
				path, when, wantStart = ".codeaf.background", when.Add(-16*time.Minute), true
			}
			data := when.Format(time.RFC3339Nano)
			if path == ".codeaf.background" {
				data = fmt.Sprintf("%d %s\n", os.Getpid(), data)
			}
			if err := os.WriteFile(filepath.Join(bin, path), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
			// Concurrent stale takeovers must also converge on a single installer.
			var wg sync.WaitGroup
			for i := 0; i < 6; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					if err := ensureBackground(func() (string, error) { return self, nil }); err != nil {
						t.Error(err)
					}
				}()
			}
			wg.Wait()
			if wantStart {
				waitFile(t, filepath.Join(home, "starts"))
			}
			time.Sleep(100 * time.Millisecond)
			starts, _ := os.ReadFile(filepath.Join(home, "starts"))
			want := 0
			if wantStart {
				want = 1
			}
			if len(starts) != want {
				t.Fatalf("installers=%d, want %d", len(starts), want)
			}
		})
	}
}

func TestBackgroundLeaseGuardRecoveryAndOldChild(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	bin := filepath.Join(home, "bin")
	if err := os.MkdirAll(bin, 0755); err != nil {
		t.Fatal(err)
	}
	guard := filepath.Join(bin, backgroundLease+".guard")
	if err := os.WriteFile(guard, nil, 0600); err != nil {
		t.Fatal(err)
	}
	if lease, err := acquireBackgroundLease(home); err != nil || lease != "" {
		t.Fatalf("overlapped active lease acquisition: %q, %v", lease, err)
	}
	old := time.Now().Add(-16 * time.Minute)
	if err := os.Chtimes(guard, old, old); err != nil {
		t.Fatal(err)
	}
	lease, err := acquireBackgroundLease(home)
	if err != nil || lease == "" {
		t.Fatalf("could not recover interrupted acquisition: %q, %v", lease, err)
	}
	CompleteBackground("older owner", fmt.Errorf("old failure"))
	data, err := os.ReadFile(filepath.Join(bin, backgroundLease))
	if err != nil || string(data) != lease {
		t.Fatalf("old child removed replacement lease: %q, %v", data, err)
	}
	if _, err := os.Stat(filepath.Join(bin, backgroundFailure)); !os.IsNotExist(err) {
		t.Fatal("old child applied cooldown to replacement")
	}
	CompleteBackground("", nil)
	CompleteBackground(lease, nil)
	if _, err := os.Stat(filepath.Join(bin, backgroundLease)); !os.IsNotExist(err) {
		t.Fatal("completed attempt left a lease")
	}
}

func TestBackgroundHookAcrossProcesses(t *testing.T) {
	switch os.Getenv("CODEAF_BACKGROUND_PROCESS_ROLE") {
	case "caller":
		if err := ensureBackground(func() (string, error) { return os.Getenv("CODEAF_BACKGROUND_SELF"), nil }); err != nil {
			t.Fatal(err)
		}
		return
	case "installer":
		lease := ""
		for i, arg := range os.Args {
			if arg == "--background-lease" && i+1 < len(os.Args) {
				lease = os.Args[i+1]
			}
		}
		if lease == "" {
			t.Fatal("background lease was not passed to child")
		}
		err := Ensure(Options{LockWait: 30 * time.Second})
		CompleteBackground(lease, err)
		return
	}
	noSkip(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("TMPDIR", t.TempDir())
	t.Setenv("PATH", t.TempDir())
	t.Setenv("CODEAF_BIN", "")
	self := filepath.Join(home, "fake-af")
	t.Setenv("CODEAF_BACKGROUND_SELF", self)
	t.Setenv("CODEAF_BACKGROUND_TEST_BINARY", os.Args[0])
	script := "#!/bin/sh\nCODEAF_BACKGROUND_PROCESS_ROLE=installer exec \"$CODEAF_BACKGROUND_TEST_BINARY\" -test.run=^TestBackgroundHookAcrossProcesses$ -- \"$@\"\n"
	if err := os.WriteFile(self, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		<-release
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	var once sync.Once
	defer once.Do(func() { close(release) })
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	launch := func() {
		t.Helper()
		caller := exec.Command(os.Args[0], "-test.run=^TestBackgroundHookAcrossProcesses$")
		caller.Env = append(os.Environ(), "CODEAF_BACKGROUND_PROCESS_ROLE=caller", "GORACE=atexit_sleep_ms=0")
		if out, err := caller.CombinedOutput(); err != nil {
			t.Fatalf("launch parent: %v, %s", err, out)
		}
	}
	for i := 0; i < 6; i++ {
		launch()
	}
	deadline := time.Now().Add(2 * time.Second)
	for requests.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if requests.Load() != 1 {
		t.Fatalf("six processes issued %d downloads", requests.Load())
	}
	once.Do(func() { close(release) })
	waitFile(t, filepath.Join(home, "bin", backgroundFailure))
	for i := 0; i < 6; i++ {
		launch()
	}
	if requests.Load() != 1 {
		t.Fatalf("cross-process failure cooldown retried: %d downloads", requests.Load())
	}
	if _, err := os.Stat(filepath.Join(home, "bin", backgroundLease)); !os.IsNotExist(err) {
		t.Fatal("child did not clear the lease after all launch parents exited")
	}
}
