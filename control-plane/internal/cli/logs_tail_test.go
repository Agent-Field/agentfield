package cli

import (
	"runtime"
	"strings"
	"testing"
)

// Contract: on Unix, `af logs` shells out to tail(1) with the requested line
// count, adding -f when following. (The Windows branch builds a PowerShell
// Get-Content command instead; it is compile-verified via the windows
// cross-build and exercised only on a real Windows machine.)
func TestTailCommandUnix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix tail arguments are not used on windows")
	}

	cmd := tailCommand("/var/log/agent.log", 7, false)
	want := []string{"tail", "-n", "7", "/var/log/agent.log"}
	if got := strings.Join(cmd.Args, " "); got != strings.Join(want, " ") {
		t.Fatalf("tailCommand(no follow) args = %q; want %q", got, strings.Join(want, " "))
	}

	follow := tailCommand("/var/log/agent.log", 10, true)
	wantFollow := []string{"tail", "-n", "10", "-f", "/var/log/agent.log"}
	if got := strings.Join(follow.Args, " "); got != strings.Join(wantFollow, " ") {
		t.Fatalf("tailCommand(follow) args = %q; want %q", got, strings.Join(wantFollow, " "))
	}
}

// Contract: psSingleQuote produces a PowerShell single-quoted literal where
// embedded single quotes are doubled — the only escape that quoting form has.
func TestPSSingleQuote(t *testing.T) {
	cases := map[string]string{
		`C:\logs\agent.log`:  `'C:\logs\agent.log'`,
		`C:\it's here\a.log`: `'C:\it''s here\a.log'`,
		``:                   `''`,
	}
	for in, want := range cases {
		if got := psSingleQuote(in); got != want {
			t.Errorf("psSingleQuote(%q) = %s; want %s", in, got, want)
		}
	}
}
