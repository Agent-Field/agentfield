package cli

import (
	"reflect"
	"runtime"
	"testing"
)

// Contract: `af logs` tails via tail(1) on Unix and via PowerShell
// Get-Content on Windows (which has no tail), preserving the requested line
// count, the follow flag, and safe quoting of the log path. The helper is
// parameterized by GOOS so both platform branches are exercised regardless of
// the host running the tests.
func TestTailCommandArgs(t *testing.T) {
	cases := []struct {
		name     string
		goos     string
		file     string
		n        int
		follow   bool
		wantProg string
		wantArgs []string
	}{
		{
			name: "unix no follow", goos: "linux",
			file: "/var/log/agent.log", n: 7,
			wantProg: "tail",
			wantArgs: []string{"-n", "7", "/var/log/agent.log"},
		},
		{
			name: "unix follow", goos: "darwin",
			file: "/var/log/agent.log", n: 10, follow: true,
			wantProg: "tail",
			wantArgs: []string{"-n", "10", "-f", "/var/log/agent.log"},
		},
		{
			name: "windows no follow", goos: "windows",
			file: `C:\logs\agent.log`, n: 7,
			wantProg: "powershell",
			wantArgs: []string{
				"-NoProfile", "-Command",
				`Get-Content -LiteralPath 'C:\logs\agent.log' -Tail 7`,
			},
		},
		{
			name: "windows follow with quote escaping", goos: "windows",
			file: `C:\it's here\agent.log`, n: 10, follow: true,
			wantProg: "powershell",
			wantArgs: []string{
				"-NoProfile", "-Command",
				`Get-Content -LiteralPath 'C:\it''s here\agent.log' -Tail 10 -Wait`,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prog, args := tailCommandArgs(tc.goos, tc.file, tc.n, tc.follow)
			if prog != tc.wantProg {
				t.Fatalf("program = %q; want %q", prog, tc.wantProg)
			}
			if !reflect.DeepEqual(args, tc.wantArgs) {
				t.Fatalf("args = %q; want %q", args, tc.wantArgs)
			}
		})
	}
}

// Contract: tailCommand builds an exec.Cmd from the host platform's
// tailCommandArgs — the first Args entry is the program itself.
func TestTailCommandUsesHostGOOS(t *testing.T) {
	cmd := tailCommand("/var/log/agent.log", 5, false)
	prog, args := tailCommandArgs(runtime.GOOS, "/var/log/agent.log", 5, false)
	want := append([]string{prog}, args...)
	if !reflect.DeepEqual(cmd.Args, want) {
		t.Fatalf("cmd.Args = %q; want %q", cmd.Args, want)
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
