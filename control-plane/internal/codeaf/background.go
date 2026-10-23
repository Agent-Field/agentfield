package codeaf

import (
	"os"
	"os/exec"
	"path/filepath"
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
	logDir := filepath.Join(home, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(logDir, "codeaf-ensure.log"), os.O_CREATE|os.O_RDWR|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	defer func() { _ = log.Close() }()
	cmd := exec.Command(self, "codeaf", "ensure", "--lock-wait", "30s")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = log, log, log
	detach(cmd)
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}
