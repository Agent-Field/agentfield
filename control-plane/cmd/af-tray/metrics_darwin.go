//go:build darwin

package main

import (
	"os/exec"
	"strconv"
	"strings"
)

// serverMemoryMB returns the resident set size of the running control-plane
// process in megabytes, or 0 if it can't be determined. The Prometheus
// /metrics endpoint only exports Go runtime memory (a few MB of heap), which
// understates the real footprint, so we read the OS's RSS for the process
// directly. Best-effort: any failure just hides the memory figure.
func serverMemoryMB() int {
	pid := serverPID()
	if pid == "" {
		return 0
	}
	// `ps -o rss=` prints resident size in kilobytes with no header.
	out, err := exec.Command("ps", "-o", "rss=", "-p", pid).Output()
	if err != nil {
		return 0
	}
	kb, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil || kb <= 0 {
		return 0
	}
	return kb / 1024
}

// serverPID finds the control-plane server process. It matches the `server`
// subcommand of the installed binary so it doesn't accidentally match the tray
// or an `af` CLI invocation.
func serverPID() string {
	out, err := exec.Command("pgrep", "-f", "agentfield server").Output()
	if err != nil {
		return ""
	}
	// pgrep may return multiple pids (one per line); take the first.
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if p := strings.TrimSpace(line); p != "" {
			return p
		}
	}
	return ""
}
