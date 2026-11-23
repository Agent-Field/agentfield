package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These tests pin the behaviours endpoint-security tools react to. Each one is a
// promise about what the install, start and update path never does, and each runs
// on every platform because the surfaces it reads are plain text.

const repoRootFromTray = "../../.."

// sourceLines returns the non-comment lines of a file, so a sentence that
// explains why we do NOT call a tool does not trip the test that forbids calling it.
func sourceLines(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, line := range strings.Split(string(data), "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "", strings.HasPrefix(trimmed, "#"), strings.HasPrefix(trimmed, "//"),
			strings.HasPrefix(trimmed, "*"), strings.HasPrefix(trimmed, "/*"):
			continue
		}
		out = append(out, line)
	}
	return out
}

func filesUnder(t *testing.T, patterns ...string) []string {
	t.Helper()
	var files []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(filepath.Join(repoRootFromTray, pattern))
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range matches {
			if strings.HasSuffix(m, "_test.go") || strings.HasSuffix(m, ".test.ts") {
				continue
			}
			files = append(files, m)
		}
	}
	if len(files) == 0 {
		t.Fatalf("no files matched %v: the layout moved, update this test", patterns)
	}
	return files
}

// Contract: nothing in the install, start or update path strips the quarantine
// attribute. curl does not set it, a signed and notarized binary does not need
// it removed, and `xattr -d com.apple.quarantine` is the Gatekeeper evasion that
// malware performs.
func TestInstallPathNeverStripsQuarantine(t *testing.T) {
	for _, file := range filesUnder(t,
		"scripts/install.sh", "scripts/uninstall.sh",
		"desktop/src/main/*.ts",
		"control-plane/cmd/af-tray/*.go", "control-plane/internal/launchdsvc/*.go",
		"control-plane/internal/cli/service*.go",
	) {
		for _, line := range sourceLines(t, file) {
			if strings.Contains(line, "xattr") || strings.Contains(line, "com.apple.quarantine") {
				t.Errorf("%s runs or names xattr/quarantine: %s", file, strings.TrimSpace(line))
			}
		}
	}
}

// Contract: the tray never shells out to osascript. A hidden-answer AppleScript
// dialog is how credential-phishing malware asks for a password; the tray asks
// through a native alert from its own signed process instead.
func TestTrayNeverRunsOsascript(t *testing.T) {
	for _, file := range filesUnder(t, "control-plane/cmd/af-tray/*.go", "control-plane/cmd/af-tray/*.m") {
		for _, line := range sourceLines(t, file) {
			if strings.Contains(line, `"osascript"`) || strings.Contains(line, "NSAppleScript") {
				t.Errorf("%s runs AppleScript: %s", file, strings.TrimSpace(line))
			}
		}
	}
}

// Contract: the one place the tray names the Keychain item is the opt-in-gated
// reader, so a second, ungated caller cannot appear unnoticed.
func TestOnlyTheGatedReaderTouchesTheKeychain(t *testing.T) {
	for _, file := range filesUnder(t, "control-plane/cmd/af-tray/*.go") {
		if filepath.Base(file) == "claude_quota_darwin.go" {
			continue
		}
		for _, line := range sourceLines(t, file) {
			if strings.Contains(line, "find-generic-password") || strings.Contains(line, `"security"`) {
				t.Errorf("%s reads the Keychain outside the opt-in gate: %s", file, strings.TrimSpace(line))
			}
		}
	}
}
