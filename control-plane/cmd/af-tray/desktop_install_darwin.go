//go:build darwin

package main

import (
	"bytes"
	"errors"
	"fmt"
	"html"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
)

const desktopLoginLabel = "ai.agentfield.tray.login"

func desktopMarkerPath() string { return filepath.Join(agentfieldDir(), "desktop-tray") }
func traySessionPath() string   { return filepath.Join(agentfieldDir(), "tray-session.plist") }

// Both installers manage the same per-user tray. Serialize the transition across
// processes, including a CLI update that overlaps with Desktop startup.
func withTrayLock(apply func() error) error {
	if err := os.MkdirAll(logsDir(), 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(filepath.Join(agentfieldDir(), "tray.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	return apply()
}

// stagedTrayPath is the copy of the helper that older Desktop builds and the
// installer left in ~/.agentfield/bin. A tray that runs from the signed Desktop
// app has no use for it, and an unreferenced executable in a hidden folder is
// exactly what endpoint-security tools quarantine.
func stagedTrayPath() string { return filepath.Join(binDir(), "af-tray") }

// removeStagedTray deletes the staged copy once no tray job names it. It never
// follows a link, never deletes the program that is running, and leaves the copy
// alone while a job still points at it (the CLI-installed tray runs from it).
func removeStagedTray(program string) error {
	staged := stagedTrayPath()
	if staged == program {
		return nil
	}
	info, err := os.Lstat(staged)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil || !info.Mode().IsRegular() {
		return err
	}
	for _, job := range []string{trayPlistPath(), traySessionPath()} {
		data, err := os.ReadFile(job)
		if err != nil {
			continue
		}
		if html.UnescapeString(launchdsvc.ParsePlistOwner(data).Program) == staged {
			return nil
		}
	}
	return removeIfPresent(staged)
}

func desktopOwnsTray() bool {
	data, err := os.ReadFile(desktopMarkerPath())
	if err != nil {
		return false
	}
	program, _, _ := strings.Cut(string(data), "\n")
	info, err := os.Stat(program)
	return err == nil && info.Mode().IsRegular()
}

// A native login job can start while a session tray is still shutting down.
func lockTrayProcess() (*os.File, error) {
	if err := os.MkdirAll(agentfieldDir(), 0o755); err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(filepath.Join(agentfieldDir(), "tray-running.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, nil
		}
		return nil, err
	}
	return lock, nil
}

func installDesktop() error {
	return withTrayLock(func() error {
		if desktopOwnsTray() {
			fmt.Println("The tray is managed by AgentField Desktop.")
			return nil
		}
		program, err := os.Executable()
		if err != nil {
			return err
		}
		if err := installTrayAgent(trayPlistPath(), program); err != nil {
			return err
		}
		if err := launchdsvc.RemoveLegacyTrayBundle(home()); err != nil {
			return err
		}
		fmt.Println("AgentField tray installed. Look for the icon in your menu bar.")
		return nil
	})
}

func uninstallDesktop() error {
	return withTrayLock(func() error {
		if desktopOwnsTray() {
			return fmt.Errorf("hide the tray in AgentField Desktop settings")
		}
		if err := removeTrayAgent(trayPlistPath()); err != nil {
			return err
		}
		return launchdsvc.RemoveLegacyTrayBundle(home())
	})
}

func syncDesktopTray(visible bool, login string) error {
	program, err := os.Executable()
	if err != nil {
		return err
	}
	return withTrayLock(func() error { return syncPackagedTray(program, visible, login) })
}

func syncPackagedTray(program string, visible bool, login string) error {
	native := desktopLoginStatus()
	legacy, err := readTrayAgent(trayPlistPath())
	if err != nil {
		return err
	}
	atLogin := login == "login" || login == "auto" && (native.registered() || len(legacy) > 0 && legacyTrayAtLogin())
	digest, ok := launchdsvc.FileSHA256(program)
	if !ok {
		return fmt.Errorf("could not fingerprint the bundled tray at %s", program)
	}
	previous, _ := os.ReadFile(desktopMarkerPath())
	marker := []byte(program + "\n" + digest + "\n")
	if !visible {
		if err := setDesktopLogin(native, false); err != nil {
			return err
		}
		if err := removeTrayAgent(traySessionPath()); err != nil {
			return err
		}
		if err := removeTrayAgent(trayPlistPath()); err != nil {
			return err
		}
		if err := launchdsvc.RemoveLegacyTrayBundle(home()); err != nil {
			return err
		}
		// Best effort: a leftover copy is clutter, not a reason to fail the sync.
		_ = removeStagedTray(program)
		if err := writeFileAtomic(desktopMarkerPath(), marker, 0o600); err != nil {
			return err
		}
		fmt.Printf("tray-at-login=%t\n", atLogin)
		return nil
	}
	// Older macOS versions use the same launchd format as CLI-only installs,
	// pointing to the signed, packaged helper.
	if atLogin && native == desktopLoginUnavailable {
		if err := removeTrayAgent(traySessionPath()); err != nil {
			return err
		}
		if err := installTrayAgent(trayPlistPath(), program); err != nil {
			return err
		}
	} else {
		// macOS requires registration again when a bundled LaunchAgent changes.
		if atLogin && native.registered() && !bytes.Equal(previous, marker) {
			if err := setDesktopLogin(native, false); err != nil {
				return err
			}
			native = desktopLoginStatus()
		}
		// Remove the legacy job before registering the replacement, which
		// macOS can launch immediately. Roll back if registration fails.
		if err := removeTrayAgent(trayPlistPath()); err != nil {
			return err
		}
		if err := setDesktopLogin(native, atLogin); err != nil {
			if len(legacy) != 0 {
				if restoreErr := writeFileAtomic(trayPlistPath(), legacy, 0o644); restoreErr != nil {
					return errors.Join(err, restoreErr)
				}
				if atLogin {
					return errors.Join(err, bootstrapAgent(trayPlistPath()))
				}
			}
			return err
		}
		if atLogin && desktopLoginStatus() == desktopLoginEnabled {
			if err := removeTrayAgent(traySessionPath()); err != nil {
				return err
			}
			if err := kickstartAgent(desktopLoginLabel, false); err != nil {
				return err
			}
		} else if err := installTrayAgent(traySessionPath(), program); err != nil {
			return err
		}
	}
	if err := launchdsvc.RemoveLegacyTrayBundle(home()); err != nil {
		return err
	}
	_ = removeStagedTray(program)
	if err := writeFileAtomic(desktopMarkerPath(), marker, 0o600); err != nil {
		return err
	}
	fmt.Printf("tray-at-login=%t\n", atLogin)
	return nil
}

func readTrayAgent(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("refusing to modify non-regular tray job at %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	owner := launchdsvc.ParsePlistOwner(data)
	owner.Program = html.UnescapeString(owner.Program)
	template, _, _ := bytes.Cut(data, []byte("<!-- af-tray-sha256:"))
	if !filepath.IsAbs(owner.Program) ||
		(!bytes.Equal(template, []byte(trayPlistFor(owner.Program))) &&
			!bytes.Equal(template, []byte(legacyTrayPlistFor(owner.Program)))) {
		return nil, fmt.Errorf("refusing to modify unrecognized tray job at %s", path)
	}
	return data, nil
}

func installTrayAgent(path, program string) error {
	current, err := readTrayAgent(path)
	if err != nil {
		return err
	}
	digest, ok := launchdsvc.FileSHA256(program)
	if !ok {
		return fmt.Errorf("could not fingerprint tray executable at %s", program)
	}
	want := []byte(trayPlistFor(program) + "<!-- af-tray-sha256:" + digest + " -->\n")
	active, err := activeTrayPlist()
	if err != nil {
		return err
	}
	if active != "" && (active != path || len(current) == 0) {
		return fmt.Errorf("the tray is already running from a different registration")
	}
	if bytes.Equal(current, want) && active == path {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if active == path {
		if err := bootoutAgent(trayLabel); err != nil {
			return err
		}
	}
	if err := writeFileAtomic(path, want, 0o644); err != nil {
		return err
	}
	// launchd can take a moment to release a label after bootout.
	for attempt := 0; attempt < 10; attempt++ {
		if err = bootstrapAgent(path); err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return err
}

func activeTrayPlist() (string, error) {
	out, err := exec.Command("launchctl", "print", launchdsvc.SvcTarget(trayLabel)).Output()
	if err != nil {
		return "", nil
	}
	for _, line := range strings.Split(string(out), "\n") {
		if path, ok := strings.CutPrefix(strings.TrimSpace(line), "path = "); ok {
			return path, nil
		}
	}
	return "", fmt.Errorf("could not identify the active tray registration")
}

func removeTrayAgent(path string) error {
	current, err := readTrayAgent(path)
	if err != nil || len(current) == 0 {
		return err
	}
	active, err := activeTrayPlist()
	if err != nil {
		return err
	}
	if active == path {
		if err := bootoutAgent(trayLabel); err != nil {
			return err
		}
	}
	return removeIfPresent(path)
}

func removeIfPresent(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func legacyTrayAtLogin() bool {
	out, err := exec.Command("launchctl", "print-disabled", launchdsvc.GUIDomain()).Output()
	if err != nil {
		return true
	}
	for _, line := range strings.Split(string(out), "\n") {
		name, status, ok := strings.Cut(strings.TrimSpace(line), " => ")
		if ok && strings.Trim(name, "\"") == trayLabel {
			return status != "disabled" && status != "true"
		}
	}
	return true
}
