//go:build darwin

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

func TestTrayUpgradeAndHideKeepServerRunning(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("AF_TEST_ACTIVE_TRAY", filepath.Join(root, "active-tray"))
	t.Setenv("AF_TEST_LAUNCH_LOG", filepath.Join(root, "launchctl.log"))
	server := exec.Command("/bin/sleep", "30")
	if err := server.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = server.Process.Kill()
		_ = server.Wait()
	})
	t.Setenv("AF_TEST_SERVER_PID", strconv.Itoa(server.Process.Pid))

	bin := filepath.Join(root, "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+":/usr/bin:/bin")
	mustWriteTrayFixture(t, filepath.Join(bin, "launchctl"), `#!/bin/sh
printf '%s\n' "$*" >> "$AF_TEST_LAUNCH_LOG"
case "$*" in
  *ai.agentfield.server*) kill "$AF_TEST_SERVER_PID"; exit 1 ;;
esac
case "$1" in
  print)
    [ -f "$AF_TEST_ACTIVE_TRAY" ] || exit 1
    printf '    path = %s\n' "$(cat "$AF_TEST_ACTIVE_TRAY")"
    ;;
  bootstrap) printf '%s\n' "$3" > "$AF_TEST_ACTIVE_TRAY" ;;
  bootout) rm -f "$AF_TEST_ACTIVE_TRAY" ;;
  *) exit 2 ;;
esac
`)
	program := filepath.Join(bin, "af-tray")
	mustWriteTrayFixture(t, program, "packaged tray executable")
	mustWriteTrayFixture(t, os.Getenv("AF_TEST_ACTIVE_TRAY"), trayPlistPath())
	mustWriteTrayFixture(t, trayPlistPath(), legacyTrayPlistFor(trayBundleBinaryPath()))
	mustWriteTrayFixture(t, serverPlistPath(), serverPlist())
	wantServer, err := os.ReadFile(serverPlistPath())
	if err != nil {
		t.Fatal(err)
	}
	// Emulate a full Desktop app whose Info.plist was overwritten by an old
	// tray installer. The old wrapper ID alone cannot prove ownership.
	desktopInfo := infoPlist()
	desktopInfoPath := filepath.Join(appBundleDir(), "Contents", "Info.plist")
	mustWriteTrayFixture(t, desktopInfoPath, desktopInfo)
	mustWriteTrayFixture(t, filepath.Join(appBundleDir(), "Contents", "MacOS", "AgentField"), "Electron")

	// The test binary is outside an app bundle, exercising the launchd fallback.
	if err := withTrayLock(func() error { return syncPackagedTray(program, true, "auto") }); err != nil {
		t.Fatal("upgrade:", err)
	}
	job, err := readTrayAgent(trayPlistPath())
	if err != nil || !strings.Contains(string(job), program) {
		t.Fatalf("legacy opt-in was not preserved by upgrade: %s (%v)", job, err)
	}
	if err := withTrayLock(func() error { return syncPackagedTray(program, false, "login") }); err != nil {
		t.Fatal("hide:", err)
	}
	if _, err := os.Stat(trayPlistPath()); !os.IsNotExist(err) {
		t.Fatalf("tray job still installed: %v", err)
	}
	// A user who chooses to show the icon with login disabled gets a job
	// for this session only.
	if err := withTrayLock(func() error { return syncPackagedTray(program, true, "session") }); err != nil {
		t.Fatal("show for this session:", err)
	}
	if _, err := os.Stat(trayPlistPath()); !os.IsNotExist(err) {
		t.Fatalf("session tray was registered for future logins: %v", err)
	}
	if job, err := readTrayAgent(traySessionPath()); err != nil || len(job) == 0 {
		t.Fatalf("session tray was not registered: %v", err)
	}
	if err := server.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatal("server stopped during tray changes:", err)
	}
	gotServer, err := os.ReadFile(serverPlistPath())
	if err != nil || string(gotServer) != string(wantServer) {
		t.Fatalf("server plist changed: %v", err)
	}
	calls, err := os.ReadFile(os.Getenv("AF_TEST_LAUNCH_LOG"))
	if err != nil || strings.Contains(string(calls), serverLabel) {
		t.Fatalf("tray changes touched the server job: %s (%v)", calls, err)
	}
	gotDesktop, err := os.ReadFile(desktopInfoPath)
	if err != nil || string(gotDesktop) != desktopInfo {
		t.Fatalf("tray changes modified the Desktop app: %v", err)
	}
}

func mustWriteTrayFixture(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestPackagedTrayDiscoversNativeLoginAgent(t *testing.T) {
	if os.Getenv("AF_TEST_PACKAGED_TRAY") == "1" {
		if desktopLoginStatus() == desktopLoginUnavailable {
			t.Fatal("ServiceManagement did not discover the packaged login plist")
		}
		return
	}
	version, err := exec.Command("/usr/bin/sw_vers", "-productVersion").Output()
	if err != nil {
		t.Fatal(err)
	}
	major, _, _ := strings.Cut(strings.TrimSpace(string(version)), ".")
	number, err := strconv.Atoi(major)
	if err != nil || number < 13 {
		t.Skip("ServiceManagement login agents require macOS 13")
	}

	bundle := filepath.Join(t.TempDir(), "AgentField.app", "Contents")
	mustWriteTrayFixture(t, filepath.Join(bundle, "Info.plist"), strings.NewReplacer(
		trayLabel, "ai.agentfield.desktop.traytest",
		"<string>af-tray</string>", "<string>AgentField</string>",
	).Replace(infoPlist()))
	plist, err := os.ReadFile(filepath.Join("..", "..", "..", "desktop", "resources", "macos", desktopLoginLabel+".plist"))
	if err != nil {
		t.Fatal(err)
	}
	mustWriteTrayFixture(t, filepath.Join(bundle, "Library", "LaunchAgents", desktopLoginLabel+".plist"), string(plist))
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	program := filepath.Join(bundle, "MacOS", "af-tray")
	if err := os.MkdirAll(filepath.Dir(program), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("/bin/cp", self, program).CombinedOutput(); err != nil {
		t.Fatalf("copy test helper: %s (%v)", out, err)
	}
	cmd := exec.Command(program, "-test.run=^TestPackagedTrayDiscoversNativeLoginAgent$")
	cmd.Env = append(os.Environ(), "AF_TEST_PACKAGED_TRAY=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("check packaged helper: %s (%v)", out, err)
	}
}
