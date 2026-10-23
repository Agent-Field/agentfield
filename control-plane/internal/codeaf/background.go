package codeaf

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	backgroundLease    = ".codeaf.background"
	backgroundFailure  = ".codeaf.lastfail"
	backgroundLeaseTTL = 15 * time.Minute
	backgroundCooldown = 10 * time.Minute
)

// EnsureBackground starts a detached installer so an optional harness never
// delays node launch. Its output remains available after the launching af exits.
func EnsureBackground() error {
	return ensureBackground(os.Executable)
}

func ensureBackground(executable func() (string, error)) error {
	if SkipRequested() {
		return nil
	}
	if _, err := Lookup(""); err == nil {
		return nil
	}
	self, err := executable()
	if err != nil {
		return err
	}
	home, err := agentfieldHome("")
	if err != nil {
		return err
	}
	return startBackground(self, home)
}

func startBackground(self, home string) error {
	lease, err := acquireBackgroundLease(home)
	if err != nil || lease == "" {
		return err
	}
	started := false
	defer func() {
		if !started {
			completeBackground(home, lease, fmt.Errorf("background installer did not start"))
		}
	}()
	logDir := filepath.Join(home, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(logDir, "codeaf-ensure.log"), os.O_CREATE|os.O_RDWR|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	defer func() { _ = log.Close() }()
	cmd := exec.Command(self, "codeaf", "ensure", "--lock-wait", "30s", "--background-lease", lease)
	cmd.Env = append(os.Environ(), "AGENTFIELD_HOME="+home)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = log, log, log
	detach(cmd)
	if err := cmd.Start(); err != nil {
		return err
	}
	started = true
	go func() {
		// The child records its own result even if this parent has exited. Reap
		// it here and also clean up after a child that fails before reaching CLI code.
		completeBackground(home, lease, cmd.Wait())
	}()
	return nil
}

func acquireBackgroundLease(home string) (string, error) {
	bin := filepath.Join(home, "bin")
	if err := os.MkdirAll(bin, 0755); err != nil {
		return "", err
	}
	// Serialize the short stale-lease replacement across processes, including
	// on Windows. An interrupted guard expires just like an interrupted lease.
	guardPath := filepath.Join(bin, backgroundLease+".guard")
	guard, err := os.OpenFile(guardPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if os.IsExist(err) {
		if info, statErr := os.Lstat(guardPath); statErr == nil && time.Since(info.ModTime()) > backgroundLeaseTTL {
			if err := os.Remove(guardPath); err != nil && !os.IsNotExist(err) {
				return "", err
			}
			guard, err = os.OpenFile(guardPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		}
	}
	if os.IsExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	_ = guard.Close()
	defer func() { _ = os.Remove(guardPath) }()
	if data, err := os.ReadFile(filepath.Join(bin, backgroundFailure)); err == nil {
		if failed, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(data))); err == nil && time.Since(failed) < backgroundCooldown {
			return "", nil
		}
	}
	path := filepath.Join(bin, backgroundLease)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if os.IsExist(err) {
		info, statErr := os.Lstat(path)
		if statErr != nil {
			return "", statErr
		}
		started := info.ModTime()
		if data, readErr := os.ReadFile(path); readErr == nil {
			fields := strings.Fields(string(data))
			if len(fields) == 2 {
				if stamp, parseErr := time.Parse(time.RFC3339Nano, fields[1]); parseErr == nil {
					started = stamp
				}
			}
		}
		if time.Since(started) <= backgroundLeaseTTL {
			return "", nil
		}
		if err := os.Remove(path); err != nil {
			return "", err
		}
		file, err = os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	}
	if err != nil {
		return "", err
	}
	lease := fmt.Sprintf("%d %s\n", os.Getpid(), time.Now().Format(time.RFC3339Nano))
	_, err = file.WriteString(lease)
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return lease, nil
}

// CompleteBackground records a detached ensure's result independently of the
// lifetime of its launching af run. Explicit ensures pass an empty lease.
func CompleteBackground(lease string, installErr error) {
	if lease == "" {
		return
	}
	if home, err := agentfieldHome(""); err == nil {
		completeBackground(home, lease, installErr)
	}
}

func completeBackground(home, lease string, installErr error) {
	bin := filepath.Join(home, "bin")
	path := filepath.Join(bin, backgroundLease)
	data, err := os.ReadFile(path)
	if err != nil || string(data) != lease {
		return // A stale child must not remove a replacement's lease.
	}
	if installErr != nil {
		_ = os.WriteFile(filepath.Join(bin, backgroundFailure), []byte(time.Now().Format(time.RFC3339Nano)+"\n"), 0600)
	} else {
		_ = os.Remove(filepath.Join(bin, backgroundFailure))
	}
	_ = os.Remove(path)
}
