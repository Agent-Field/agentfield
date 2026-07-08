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

	// --- Header: brand line + status line ---
	mBrand := systray.AddMenuItem("AgentField", "")
	mBrand.Disable()
	mStatus := systray.AddMenuItem(statusLine(false, serverPort()), "")
	mStatus.SetIcon(iconDotRed) // colored status dot; recolored on refresh
	mStatus.Disable()

	systray.AddSeparator()

	// --- Agents submenu: the headline count on the parent, the roster below ---
	mAgentsParent := systray.AddMenuItem("Agents", "Registered agents")
	mAgentsParent.SetTemplateIcon(iconBot, iconBot)
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

	// --- Metric rows: one fact per row, each led by a monochrome icon. They are
	// live links (not dim, non-interactive labels): clicking opens the dashboard
	// view the stat summarizes, so full-contrast text is honest. ---
	mSuccess := systray.AddMenuItem("", "View executions in the dashboard")
	mSuccess.SetTemplateIcon(iconSuccess, iconSuccess)
	mSuccess.Hide()
	mResponse := systray.AddMenuItem("", "View executions in the dashboard")
	mResponse.SetTemplateIcon(iconGauge, iconGauge)
	mResponse.Hide()
	mMemory := systray.AddMenuItem("", "Open the dashboard")
	mMemory.SetTemplateIcon(iconCPU, iconCPU)
	mMemory.Hide()

	// Shown only when the API demands a key we don't have (or ours was rejected).
	mEnterKey := systray.AddMenuItem(enterKeyTitle(false), "Provide the API key this control plane requires")
	mEnterKey.SetTemplateIcon(iconKey, iconKey)
	mEnterKey.Hide()

	systray.AddSeparator()
	mOpen := systray.AddMenuItem("Open Dashboard", "Open the AgentField dashboard in your browser")
	mOpen.SetTemplateIcon(iconDashboard, iconDashboard)

	// --- Server controls tucked into a submenu to keep the surface calm ---
	mServer := systray.AddMenuItem("Control plane", "Start, stop, or restart the control plane")
	mServer.SetTemplateIcon(iconServer, iconServer)
	mStart := mServer.AddSubMenuItem("Start", "Start the AgentField control plane")
	mStop := mServer.AddSubMenuItem("Stop", "Stop the AgentField control plane")
	mRestart := mServer.AddSubMenuItem("Restart", "Restart the AgentField control plane")
	mServer.AddSeparator()
	mLogin := mServer.AddSubMenuItemCheckbox("Start at login", "Launch the control plane automatically when you log in", serverAutostartEnabled())
	mLogs := systray.AddMenuItem("View logs", "Open the control-plane log file")
	mLogs.SetTemplateIcon(iconLogs, iconLogs)

	systray.AddSeparator()
	mQuit := systray.AddMenuItem("Quit", "Quit the AgentField tray")
	mQuit.SetTemplateIcon(iconPower, iconPower)

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
				if sorted[i].Online {
					slot.SetIcon(iconDotGreen)
				} else {
					slot.SetIcon(iconDotGray)
				}
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

	// setRow sets a row's title and shows it, or hides it when the value is empty.
	setRow := func(it *systray.MenuItem, title string) {
		if title == "" {
			it.Hide()
			return
		}
		it.SetTitle(title)
		it.Show()
	}

	// applyLevelIcon tints a metric row by its traffic-light rating: green/yellow/
	// red colored icons for good/warn/bad, and the monochrome template icon when
	// there's no data to rate.
	applyLevelIcon := func(it *systray.MenuItem, lvl metricLevel, mono, green, yellow, red []byte) {
		switch lvl {
		case levelGood:
			it.SetIcon(green)
		case levelWarn:
			it.SetIcon(yellow)
		case levelBad:
			it.SetIcon(red)
		default:
			it.SetTemplateIcon(mono, mono)
		}
	}

	// hideData hides everything below the header that only makes sense while the
	// server is up, reachable, and authorized.
	hideData := func() {
		mAgentsParent.Hide()
		hideAgents()
		mSuccess.Hide()
		mResponse.Hide()
		mMemory.Hide()
	}

	refresh := func() {
		healthy := serverHealthy()
		mStatus.SetTitle(statusLine(healthy, serverPort()))
		if !healthy {
			systray.SetIcon(iconInactive)
			mStatus.SetIcon(iconDotRed)
			mStart.Enable()
			mStop.Disable()
			hideData()
			mEnterKey.Hide()
			return
		}

		systray.SetIcon(iconActive)
		mStatus.SetIcon(iconDotGreen)
		mStart.Disable()
		mStop.Enable()

		key := effectiveAPIKey()
		fleet := fetchFleet(key)

		if fleet.Status == fleetAuthRequired {
			// One clear call to action; data rows stay hidden until a key works.
			hideData()
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

		// Metrics — one fact per row, each hiding itself when it has nothing to
		// say and tinted green/yellow/red by its threshold.
		stats := fetchExecStats(key)
		setRow(mSuccess, metricSuccess(stats))
		applyLevelIcon(mSuccess, successLevel(stats), iconSuccess, iconSuccessGreen, iconSuccessYellow, iconSuccessRed)
		setRow(mResponse, metricResponse(stats))
		applyLevelIcon(mResponse, responseLevel(stats), iconGauge, iconGaugeGreen, iconGaugeYellow, iconGaugeRed)
		mem := serverMemoryMB()
		setRow(mMemory, metricMemory(mem))
		applyLevelIcon(mMemory, memoryLevel(mem), iconCPU, iconCPUGreen, iconCPUYellow, iconCPURed)
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
				openURL(uiPageURL("agents"))
			case <-mAgentsOpen.ClickedCh:
				openURL(uiPageURL("agents"))
			case <-mSuccess.ClickedCh:
				openURL(uiPageURL("executions"))
			case <-mResponse.ClickedCh:
				openURL(uiPageURL("executions"))
			case <-mMemory.ClickedCh:
				openURL(uiPageURL("dashboard"))
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

// openURL opens an arbitrary URL in the default browser (used for dashboard
// deep-links from the metric rows).
func openURL(u string) {
	_ = exec.Command("open", u).Start()
}

func openLogs() {
	_ = exec.Command("open", serverLogPath()).Start()
}
