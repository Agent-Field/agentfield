//go:build darwin

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"fyne.io/systray"
)

// runTray runs the menu-bar event loop. It first makes sure there is a real GUI
// (Aqua) session — if there isn't (e.g. the binary was somehow launched over an
// SSH-only session or in a headless context), it logs one line and exits 0 so
// the launchd agent's KeepAlive={Crashed: true} does not crash-loop it.
func runTray() error {
	if !hasGUISession() {
		fmt.Fprintln(os.Stderr, "af-tray: no GUI session detected, tray unavailable — exiting")
		return nil
	}
	// systray.Run blocks until systray.Quit() is called.
	systray.Run(onReady, func() {})
	return nil
}

// hasGUISession reports whether we appear to be inside a GUI login session.
// It is deliberately permissive: it only returns false when launchctl gives a
// definitive non-GUI manager name, so a false negative can never prevent the
// tray from showing on a normal desktop.
func hasGUISession() bool {
	out, err := exec.Command("launchctl", "managername").Output()
	if err != nil {
		return true // can't tell — let systray try.
	}
	name := strings.TrimSpace(string(out))
	// "Aqua" is a full GUI login session. "Background"/"System"/"StandardIO"
	// indicate a headless/daemon context.
	return name == "" || name == "Aqua"
}

func onReady() {
	systray.SetIcon(iconInactive)
	systray.SetTooltip("AgentField")

	mStatus := systray.AddMenuItem("AgentField — checking…", "")
	mStatus.Disable()

	systray.AddSeparator()
	mOpen := systray.AddMenuItem("Open Dashboard", "Open the AgentField dashboard in your browser")

	systray.AddSeparator()
	mStart := systray.AddMenuItem("Start control-plane", "Start the AgentField control plane")
	mStop := systray.AddMenuItem("Stop control-plane", "Stop the AgentField control plane")
	mRestart := systray.AddMenuItem("Restart control-plane", "Restart the AgentField control plane")
	mLogin := systray.AddMenuItemCheckbox("Start at login", "Launch the control plane automatically when you log in", serverAutostartEnabled())

	systray.AddSeparator()
	mLogs := systray.AddMenuItem("View logs", "Open the control-plane log file")

	systray.AddSeparator()
	mQuit := systray.AddMenuItem("Quit", "Quit the AgentField tray")

	refresh := func() {
		if serverHealthy() {
			systray.SetIcon(iconActive)
			mStatus.SetTitle(fmt.Sprintf("AgentField — running (:%d)", serverPort()))
			mStart.Disable()
			mStop.Enable()
		} else {
			systray.SetIcon(iconInactive)
			mStatus.SetTitle("AgentField — stopped")
			mStart.Enable()
			mStop.Disable()
		}
	}
	refresh()

	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				refresh()
			case <-mOpen.ClickedCh:
				openDashboard()
			case <-mStart.ClickedCh:
				_ = startServer()
				time.Sleep(800 * time.Millisecond)
				refresh()
			case <-mStop.ClickedCh:
				_ = stopServer()
				time.Sleep(500 * time.Millisecond)
				refresh()
			case <-mRestart.ClickedCh:
				_ = restartServer()
				time.Sleep(800 * time.Millisecond)
				refresh()
			case <-mLogin.ClickedCh:
				if mLogin.Checked() {
					if err := setServerAutostart(false); err == nil {
						mLogin.Uncheck()
					}
				} else {
					if err := setServerAutostart(true); err == nil {
						mLogin.Check()
					}
				}
				refresh()
			case <-mLogs.ClickedCh:
				openLogs()
			case <-mQuit.ClickedCh:
				systray.Quit()
				return
			}
		}
	}()
}

func openDashboard() {
	_ = exec.Command("open", dashboardURL()).Start()
}

func openLogs() {
	_ = exec.Command("open", serverLogPath()).Start()
}
