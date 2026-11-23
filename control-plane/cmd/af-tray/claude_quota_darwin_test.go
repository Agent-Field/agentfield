//go:build darwin

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Contract: the tray reads the Claude Code login from the Keychain only after
// the user has turned "Show Claude usage limits" on. Before that no `security`
// process is started at all, because an unprompted Keychain read by a background
// helper is what endpoint-security tools treat as credential theft.
func TestKeychainIsNotReadUntilTheUserOptsIn(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	bin := filepath.Join(root, "bin")
	marker := filepath.Join(root, "security-ran")
	mustWriteTrayFixture(t, filepath.Join(bin, "security"), "#!/bin/sh\necho \"$@\" >> \""+marker+"\"\nexit 1\n")
	t.Setenv("PATH", bin+":/usr/bin:/bin")

	ran := func() bool {
		_, err := os.Stat(marker)
		return err == nil
	}

	if q := fetchClaudeQuotaNow(); q.OK || ran() {
		t.Fatalf("default: quota=%+v, security ran=%v; want no Keychain read", q, ran())
	}
	if err := setClaudeUsageEnabled(true); err != nil {
		t.Fatal(err)
	}
	fetchClaudeQuotaNow()
	if !ran() {
		t.Fatal("opted in: the Keychain was never consulted")
	}
	if err := setClaudeUsageEnabled(false); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(marker); err != nil {
		t.Fatal(err)
	}
	if q := fetchClaudeQuotaNow(); q.OK || ran() {
		t.Fatalf("opted back out: quota=%+v, security ran=%v; want no Keychain read", q, ran())
	}
}
