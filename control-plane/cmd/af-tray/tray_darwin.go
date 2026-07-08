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

// maxAgentSlots bounds how many agent rows the Agents submenu shows. systray
// can't grow/shrink its menu after build, so we pre-allocate a fixed pool of
// child rows and show/hide/relabel them on each refresh; any overflow collapses
// into an "…and N more" row that opens the dashboard.
const maxAgentSlots = 12

func onReady() {
	systray.SetIcon(iconInactive)
	systray.SetTooltip("AgentField")

	// --- Status header ---
	mStatus := systray.AddMenuItem("AgentField — checking…", "")
	mStatus.Disable()

	systray.AddSeparator()

	// --- Agents submenu (keeps the top level uncluttered) ---
	// The parent title carries the headline count; the per-agent rows live one
	// level down so the main menu stays short and scannable.
	mAgentsParent := systray.AddMenuItem("👥  Agents", "Registered agents")
	mAgents := make([]*systray.MenuItem, maxAgentSlots)
	for i := range mAgents {
		it := mAgentsParent.AddSubMenuItem("", "Open the AgentField dashboard")
		it.Hide()
		mAgents[i] = it
	}
	mMore := mAgentsParent.AddSubMenuItem("", "Open the AgentField dashboard to see all agents")
	mMore.Hide()
	mAgentsParent.AddSeparator()
	mAgentsOpen := mAgentsParent.AddSubMenuItem("Open Dashboard →", "Open the AgentField dashboard")

	// --- Metric rows (compact, emoji-led, disabled/non-interactive) ---
	mSuccess := systray.AddMenuItem("", "Execution success rate")
	mSuccess.Disable()
	mSuccess.Hide()
	mPerf := systray.AddMenuItem("", "Average latency and server memory")
	mPerf.Disable()
	mPerf.Hide()

	// Shown only when the API demands a key we don't have (or ours was rejected).
	mEnterKey := systray.AddMenuItem(enterKeyTitle(false), "Provide the API key this control plane requires")
	mEnterKey.Hide()

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

	// Each agent row opens the dashboard when clicked. Rows are reused across
	// refreshes, so the action is intentionally generic.
	for _, slot := range mAgents {
		go func(ch <-chan struct{}) {
			for range ch {
				openDashboard()
			}
		}(slot.ClickedCh)
	}

	hideAgents := func() {
		for _, slot := range mAgents {
			slot.Hide()
		}
		mMore.Hide()
	}

	renderAgents := func(agents []agentInfo) {
		sorted := sortAgents(agents)
		for i, slot := range mAgents {
			if i < len(sorted) {
				slot.SetTitle(agentLine(sorted[i]))
				slot.Show()
			} else {
				slot.Hide()
			}
		}
		if len(sorted) > maxAgentSlots {
			mMore.SetTitle(fmt.Sprintf("⋯  and %d more", len(sorted)-maxAgentSlots))
			mMore.Show()
		} else {
			mMore.Hide()
		}
	}

	// showMetrics/hideMetrics toggle everything below the status header that only
	// makes sense while the server is up and reachable.
	hideMetrics := func() {
		mAgentsParent.Hide()
		hideAgents()
		mSuccess.Hide()
		mPerf.Hide()
	}

	refresh := func() {
		if !serverHealthy() {
			systray.SetIcon(iconInactive)
			mStatus.SetTitle("AgentField — stopped")
			mStart.Enable()
			mStop.Disable()
			hideMetrics()
			mEnterKey.Hide()
			return
		}

		systray.SetIcon(iconActive)
		mStatus.SetTitle(fmt.Sprintf("AgentField — running (:%d)", serverPort()))
		mStart.Disable()
		mStop.Enable()

		key := effectiveAPIKey()
		fleet := fetchFleet(key)

		if fleet.Status == fleetAuthRequired {
			// One clear call to action; everything data-bearing is hidden until
			// we have a working key.
			hideMetrics()
			mEnterKey.SetTitle(enterKeyTitle(key != ""))
			mEnterKey.Show()
			return
		}
		mEnterKey.Hide()

		// Agents.
		mAgentsParent.SetTitle(agentsHeadline(fleet))
		mAgentsParent.Show()
		if fleet.Status == fleetOK {
			renderAgents(fleet.Agents)
		} else {
			hideAgents()
		}

		// Metrics (best-effort; each row hides itself when it has nothing to say).
		stats := fetchExecStats(key)
		mSuccess.SetTitle(successLine(stats))
		mSuccess.Show()
		if line := perfLine(stats, serverMemoryMB()); line != "" {
			mPerf.SetTitle(line)
			mPerf.Show()
		} else {
			mPerf.Hide()
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
			case <-mMore.ClickedCh:
				openDashboard()
			case <-mAgentsOpen.ClickedCh:
				openDashboard()
			case <-mEnterKey.ClickedCh:
				handleEnterAPIKey()
				refresh()
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

// handleEnterAPIKey prompts for an API key with a native macOS dialog, validates
// it against the local API, and persists it only if it is accepted. A rejected
// key surfaces an error and leaves any previously stored key untouched, so the
// next refresh keeps showing the "API key required" prompt.
func handleEnterAPIKey() {
	invalid := effectiveAPIKey() != "" // we already have a key, so it must be wrong/expired
	key, ok := promptForAPIKey(invalid)
	if !ok || key == "" {
		return
	}
	if fetchFleet(key).Status == fleetAuthRequired {
		notify("API key rejected", "That key was not accepted. Please check it and try again.")
		return
	}
	if err := saveAPIKey(key); err != nil {
		notify("Could not save API key", err.Error())
	}
}

// promptForAPIKey shows a native password-style dialog. It returns ok=false when
// the user cancels (osascript exits non-zero) or on any error.
func promptForAPIKey(invalid bool) (string, bool) {
	msg := "Enter the API key for this AgentField control plane:"
	if invalid {
		msg = "This API key was rejected (invalid or expired). Enter a new one:"
	}
	script := fmt.Sprintf(
		`display dialog %q with title "AgentField" default answer "" `+
			`buttons {"Cancel","Save"} default button "Save" with hidden answer`,
		msg,
	)
	out, err := exec.Command("osascript", "-e", script, "-e", "text returned of result").Output()
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(string(out)), true
}

// notify shows a small informational dialog (used for errors the user should see
// right after acting; menu-bar apps have no other affordance for this).
func notify(title, body string) {
	script := fmt.Sprintf(`display dialog %q with title %q buttons {"OK"} default button "OK" with icon caution`, body, title)
	_ = exec.Command("osascript", "-e", script).Start()
}

func openDashboard() {
	_ = exec.Command("open", dashboardURL()).Start()
}

func openLogs() {
	_ = exec.Command("open", serverLogPath()).Start()
}
