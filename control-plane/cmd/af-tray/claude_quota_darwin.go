//go:build darwin

package main

import (
	"fmt"
	"os"
	"os/exec"
)

// This file holds the macOS-only glue for the optional Claude subscription
// quota rows: reading the user's existing Claude Code OAuth token out of the
// login Keychain. The pure parsing/formatting and the HTTP call live in
// shared.go so they can be unit-tested on CI.
//
// STRICT contract (see shared.go): read-only. We never write to the Keychain,
// and the token is only ever handed to fetchClaudeQuota, which sends it to
// api.anthropic.com and nowhere else. It is never logged or persisted.

// readClaudeCodeToken reads the OAuth access token that Claude Code stores in
// the macOS login Keychain under the "Claude Code-credentials" generic-password
// item. It is entirely best-effort: if the `security` tool, the item, or the
// expected JSON shape is missing, it returns "" and the caller hides the rows.
// A single debug line (to stderr, which launchd routes to the tray log) records
// only *that* it failed, never the token itself.
func readClaudeCodeToken() string {
	// -w prints only the password (the stored JSON) to stdout.
	out, err := exec.Command("security", "find-generic-password",
		"-s", "Claude Code-credentials", "-w").Output()
	if err != nil {
		// Absent item / no Keychain access — expected on machines without
		// Claude Code. Stay quiet beyond debug.
		if os.Getenv("AF_TRAY_DEBUG") != "" {
			fmt.Fprintln(os.Stderr, "af-tray: claude quota: no keychain credentials")
		}
		return ""
	}
	token, err := parseClaudeCodeToken(out)
	if err != nil {
		if os.Getenv("AF_TRAY_DEBUG") != "" {
			fmt.Fprintln(os.Stderr, "af-tray: claude quota: unexpected keychain payload shape")
		}
		return ""
	}
	return token
}

// fetchClaudeQuotaNow reads the token and queries the OAuth usage endpoint once.
// It returns a zero (OK=false) claudeQuota whenever anything is unavailable, so
// the caller can render nothing without special-casing.
func fetchClaudeQuotaNow() claudeQuota {
	token := readClaudeCodeToken()
	if token == "" {
		return claudeQuota{}
	}
	return fetchClaudeQuota(claudeUsageURL, token)
}
