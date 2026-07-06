package main

// This file holds the platform-neutral logic behind the tray: health polling,
// path resolution, launchd plist / Info.plist generation, launchctl argument
// construction, and atomic file writes. It has NO GUI (systray/CGO) dependency
// and compiles on every platform, so it can be unit-tested directly in CI
// (which runs on Linux). The OS-specific glue — the systray event loop and the
// exec.Command("launchctl", …) calls — lives in the _darwin files and calls
// into these helpers.

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

const (
	trayLabel   = "ai.agentfield.tray"
	serverLabel = "ai.agentfield.server"
)

// ---- Paths -----------------------------------------------------------------

func home() string {
	h, _ := os.UserHomeDir()
	return h
}

func agentfieldDir() string   { return filepath.Join(home(), ".agentfield") }
func binDir() string          { return filepath.Join(agentfieldDir(), "bin") }
func logsDir() string         { return filepath.Join(agentfieldDir(), "logs") }
func launchAgentsDir() string { return filepath.Join(home(), "Library", "LaunchAgents") }
func appBundleDir() string    { return filepath.Join(home(), "Applications", "AgentField.app") }
func serverLogPath() string   { return filepath.Join(logsDir(), "control-plane.log") }
func trayLogPath() string     { return filepath.Join(logsDir(), "tray.log") }
func trayPlistPath() string   { return filepath.Join(launchAgentsDir(), trayLabel+".plist") }
func serverPlistPath() string { return filepath.Join(launchAgentsDir(), serverLabel+".plist") }

func trayBundleBinaryPath() string {
	return filepath.Join(appBundleDir(), "Contents", "MacOS", "af-tray")
}

// serverBinaryPath finds the control-plane binary the launchd agent should run.
// It prefers the installed copy, then falls back to whatever is on PATH.
func serverBinaryPath() string {
	cand := filepath.Join(binDir(), "agentfield")
	if isExecutable(cand) {
		return cand
	}
	if p, err := exec.LookPath("af"); err == nil {
		return p
	}
	if p, err := exec.LookPath("agentfield"); err == nil {
		return p
	}
	return cand // best effort; may not exist yet.
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Mode()&0o111 != 0
}

// ---- Health / URLs ---------------------------------------------------------

// serverPort returns the port the control plane is expected to listen on.
func serverPort() int {
	if v := os.Getenv("AGENTFIELD_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			return p
		}
	}
	return 8080
}

func healthURL() string    { return fmt.Sprintf("http://localhost:%d/health", serverPort()) }
func dashboardURL() string { return fmt.Sprintf("http://localhost:%d", serverPort()) }

// checkHealth reports whether the given URL answers HTTP 200 within a short
// timeout. The control plane's /health endpoint returns 200 when healthy and
// 503 when not, so only a 200 counts as "running".
func checkHealth(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

func serverHealthy() bool { return checkHealth(healthURL()) }

// ---- launchctl argument construction ---------------------------------------

func guiDomain() string         { return fmt.Sprintf("gui/%d", os.Getuid()) }
func svcTarget(l string) string { return guiDomain() + "/" + l }

// kickstartArgs builds the argv for `launchctl kickstart`. The -k flag forces a
// restart of an already-running service (kill then relaunch); without it,
// kickstart only starts a loaded-but-idle service.
func kickstartArgs(label string, kill bool) []string {
	args := []string{"kickstart"}
	if kill {
		args = append(args, "-k")
	}
	return append(args, svcTarget(label))
}

// ---- Files -----------------------------------------------------------------

// writeFileAtomic writes data to a temp file in the destination directory and
// renames it into place, so a reader (or a running binary being replaced) never
// sees a half-written file.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".af-tray-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// ---- plist / Info.plist templates ------------------------------------------

func infoPlist() string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>AgentField</string>
  <key>CFBundleDisplayName</key><string>AgentField</string>
  <key>CFBundleIdentifier</key><string>%s</string>
  <key>CFBundleVersion</key><string>%s</string>
  <key>CFBundleShortVersionString</key><string>%s</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>af-tray</string>
  <key>CFBundleIconFile</key><string>appicon</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>10.15</string>
</dict>
</plist>
`, trayLabel, version, version)
}

// serverPlist is the control-plane launchd agent.
//   - RunAtLoad starts it at login.
//   - KeepAlive={SuccessfulExit: false} restarts it only on a crash, so a
//     graceful "Stop" (SIGTERM → clean exit) actually stays stopped.
//   - --open=false stops it opening a browser every time it starts under launchd.
func serverPlist() string {
	log := serverLogPath()
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>server</string>
    <string>--open=false</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>WorkingDirectory</key><string>%s</string>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
`, serverLabel, serverBinaryPath(), agentfieldDir(), log, log)
}

// trayPlist is the menu-bar tray launchd agent. KeepAlive={Crashed: true} means
// a genuine crash relaunches it, but a clean exit (the "Quit" menu item, or the
// no-GUI-session early exit) does not — so it never crash-loops.
func trayPlist() string {
	log := trayLogPath()
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>run</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>Crashed</key><true/></dict>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
`, trayLabel, trayBundleBinaryPath(), log, log)
}
