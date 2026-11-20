//go:build darwin

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
)

// ---- Server lifecycle (driven from the tray menu) --------------------------

func startServer() error {
	return launchServer(false)
}

// stopServer sends SIGTERM for a graceful shutdown of the managed server.
func stopServer() error {
	return launchdsvc.SignalAgent(serverLabel, "SIGTERM")
}

func restartServer() error {
	return launchServer(true)
}

func launchServer(restart bool) error {
	if !agentLoaded(serverLabel) {
		if launchdsvc.ServerHealthy(serverPort()) {
			return fmt.Errorf("the local server is already running outside launchd")
		}
		plist := serverPlistPath()
		if !serverAutostartEnabled() {
			plist = launchdsvc.ServerSessionPlistPath(home())
			if err := writeServerPlist(plist); err != nil {
				return err
			}
		}
		if err := bootstrapAgent(plist); err != nil {
			return err
		}
		// RunAtLoad starts the newly bootstrapped job.
		return nil
	}
	return kickstartAgent(serverLabel, restart)
}

func serverAutostartEnabled() bool {
	_, err := os.Stat(serverPlistPath())
	return err == nil
}

func writeServerPlist(path string) error {
	if err := os.MkdirAll(logsDir(), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(path, []byte(serverPlist()), 0o644)
}

// Change future logins independently of the current process. An existing
// launchd job keeps running from its in-memory configuration.
func setServerAutostart(enable bool) error {
	if enable {
		if serverAutostartEnabled() {
			return nil
		}
		if err := os.MkdirAll(launchAgentsDir(), 0o755); err != nil {
			return err
		}
		return writeServerPlist(serverPlistPath())
	}
	if err := os.Remove(serverPlistPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// ---- launchctl wrappers ----------------------------------------------------
//
// The implementations live in internal/launchdsvc so `af service` and the tray
// drive launchd through exactly one code path. These thin aliases keep the
// tray's call sites unchanged.

func bootstrapAgent(plistPath string) error { return launchdsvc.Bootstrap(plistPath) }
func bootoutAgent(label string) error       { return launchdsvc.Bootout(label) }
func kickstartAgent(label string, kill bool) error {
	return launchdsvc.Kickstart(label, kill)
}
func agentLoaded(label string) bool { return launchdsvc.AgentLoaded(label) }
