//go:build darwin

package main

import (
	"fmt"
	"image/color"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
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

// Usage submenu row pools. Like the agent roster, systray can't grow a menu
// after build, so the model / provider / harness lists are fixed pools of rows
// that are relabeled and shown/hidden on each refresh.
// maxUsageModelSlots caps the per-model rows: a glance UI shows only the top
// talkers; the dashboard has the full breakdown.
const maxUsageModelSlots = 3

// usageRefreshInterval is how often the usage stats are re-fetched. Usage moves
// slower than health, so it is polled less often than the 5s menu refresh.
const usageRefreshInterval = 30 * time.Second

// claudeQuotaInterval is how often the optional Claude subscription quota is
// polled on its own goroutine, well off the main refresh path.
const claudeQuotaInterval = 5 * time.Minute

// Usage submenu geometry (in points) is defined in chart_render.go — a
// cross-platform file — so the pure renderers and their tests share it with the
// darwin tray. PNGs are rendered at 2x these (retina) and applied via the
// vendored systray fork's SetImage.

// retinaScale is the PNG-pixel multiplier over point size (2x for retina).
const retinaScale = 2

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

	// --- Usage submenu: token/cost headline on the parent, a per-model
	// breakdown with proportional bars, provider/harness rollups, the optional
	// Claude subscription quota, and a "last updated" footer below. Every child
	// is a fixed pre-allocated row (systray can't grow a menu after build). ---
	mUsageParent := systray.AddMenuItem("Usage", "Token and cost usage")
	mUsageParent.SetTemplateIcon(iconUsage, iconUsage)
	mUsageParent.Hide()

	// The transparent spacer that reserves the uniform leading slot on every row
	// without a graphic, so all titles start at the same x as the bar/gauge rows.
	usageSpacer := spacerImagePNG(usageSlotWidthPt*retinaScale, usageSlotHeightPt*retinaScale)
	// spaced attaches the uniform-slot spacer to an informational row.
	spaced := func(it *systray.MenuItem) *systray.MenuItem {
		if usageSpacer != nil {
			it.SetImage(usageSpacer, usageSlotWidthPt, usageSlotHeightPt)
		}
		return it
	}

	// 24h token histogram: a wide, text-free image at the very top of the submenu
	// (disabled, no title). Shown only when the server returns a series; applied
	// via the vendored systray fork's SetImage, which shows it at full width in
	// color instead of the stock 16x16 template clamp. Its left edge aligns with
	// the uniform slot, so it sits flush with the rows below.
	mUsageChart := mUsageParent.AddSubMenuItem("", "24-hour token usage by model")
	mUsageChart.Disable()
	mUsageChart.Hide()

	// Per-model rows carry the proportional bars; they open the dashboard.
	mUsageModels := make([]*systray.MenuItem, maxUsageModelSlots)
	for i := range mUsageModels {
		it := mUsageParent.AddSubMenuItem("", "View usage in the dashboard")
		it.Hide()
		mUsageModels[i] = it
	}

	// Optional Claude subscription quota (5h / 7d rate-limit windows), fetched
	// off the main refresh path on its own goroutine. The rows are
	// self-explanatory ("Claude 5h — 3% …"), so they carry no section header.
	mClaude5h := mUsageParent.AddSubMenuItem("", "")
	mClaude5h.Disable()
	mClaude5h.Hide()
	mClaude7d := mUsageParent.AddSubMenuItem("", "")
	mClaude7d.Disable()
	mClaude7d.Hide()

	// Footer: "Last updated: N minutes ago" (disabled), then a dashboard link.
	mUsageFooter := spaced(mUsageParent.AddSubMenuItem("", ""))
	mUsageFooter.Disable()
	mUsageFooter.Hide()
	mUsageParent.AddSeparator()
	mUsageOpen := mUsageParent.AddSubMenuItem("Open Dashboard →", "Open the AgentField dashboard")

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

	// Usage model rows are reused across refreshes, so their click action is
	// generic: open the dashboard where the full usage breakdown lives.
	for _, slot := range mUsageModels {
		go func(ch <-chan struct{}) {
			for range ch {
				openPage("dashboard")
			}
		}(slot.ClickedCh)
	}

	hideAgents := func() {
		for _, slot := range mAgents {
			slot.Hide()
		}
		mMore.Hide()
	}

	// Usage state, refreshed on its own (slower) cadence than health. The Claude
	// subscription quota is fetched on a separate goroutine and read here under a
	// mutex, so it never blocks or slows the main refresh.
	var (
		usage24h       usageStats
		lastUsageFetch time.Time
		claudeMu       sync.Mutex
		claudeVal      claudeQuota
		successHist    []float64 // success-rate samples feeding the Success sparkline
		lastVisualKey  string    // change-key for the menu-bar status visual
	)
	var spinPhase atomic.Int32 // starting-arc animation phase
	var lastState atomic.Int32 // last rendered serverState, read by the spinner ticker

	// renderClaudeQuota fills the two Claude rate-limit rows from the latest
	// polled quota, hiding the whole block when it is unavailable.
	renderClaudeQuota := func(now time.Time) bool {
		claudeMu.Lock()
		q := claudeVal
		claudeMu.Unlock()
		if !q.OK {
			mClaude5h.Hide()
			mClaude7d.Hide()
			return false
		}
		setQuotaRow := func(it *systray.MenuItem, label string, pct float64, reset *time.Time) {
			// Native text title, e.g. "Claude 5h — 24% (resets 3:00 PM)", paired
			// with a small utilization-colored gauge bar image on the left.
			it.SetTitle(claudeQuotaTitle(label, pct, reset, now))
			if img := slotBarPNG(
				pct/100, quotaBarColor(pct),
				usageSlotWidthPt*retinaScale, usageSlotHeightPt*retinaScale,
				usageBarWidthPt*retinaScale, usageBarHeightPt*retinaScale,
			); img != nil {
				it.SetImage(img, usageSlotWidthPt, usageSlotHeightPt)
			}
			it.Show()
		}
		setQuotaRow(mClaude5h, "Claude 5h", q.FiveHourPct, q.FiveHourReset)
		setQuotaRow(mClaude7d, "Claude 7d", q.SevenDayPct, q.SevenDayReset)
		return true
	}

	hideUsageChildren := func() {
		mUsageChart.Hide()
		for _, slot := range mUsageModels {
			slot.Hide()
		}
		mClaude5h.Hide()
		mClaude7d.Hide()
		mUsageFooter.Hide()
	}

	// applyStatusVisual owns the menu-bar status item: the brand badge with a
	// state glyph beside it — green dot (running), rotating orange arc (starting
	// up), gray ring (stopped) — rendered via the vendored fork's SetStatusImage.
	// No text. It re-renders only when the state (or the arc's animation phase)
	// changes, so it never churns the status item on the 5s refresh tick.
	// statusMu serializes applyStatusVisual between the main select loop and the
	// starting-arc animation ticker.
	var statusMu sync.Mutex
	applyStatusVisual := func(healthy bool) {
		statusMu.Lock()
		defer statusMu.Unlock()
		state := deriveServerState(healthy, serverMemoryMB() > 0)
		lastState.Store(int32(state))
		phase := 0
		if state == serverStarting {
			phase = int(spinPhase.Load()) % statusArcSteps
		}
		key := fmt.Sprintf("%d|%d", state, phase)
		if key == lastVisualKey {
			return
		}
		lastVisualKey = key
		systray.SetTitle("")
		if img := statusBadgePNG(iconActive, state, phase); img != nil {
			systray.SetStatusImage(img, statusBadgeWidthPt, statusBadgeHeightPt)
		} else if healthy {
			systray.SetIcon(iconActive)
		} else {
			systray.SetIcon(iconInactive)
		}
	}

	renderUsage := func() {
		now := time.Now()

		// The Claude quota is independent of AgentField usage, so the section is
		// shown when either has something to say.
		claudeShown := renderClaudeQuota(now)
		if !usage24h.hasData() && !claudeShown {
			mUsageParent.Hide()
			hideUsageChildren()
			applyStatusVisual(true)
			return
		}

		if usage24h.hasData() {
			mUsageParent.SetTitle(usageHeadline(usage24h))
		} else {
			mUsageParent.SetTitle("Usage")
		}
		mUsageParent.Show()

		// 24h token histogram row: a bucket histogram stacked by model when the
		// server sent a per-model breakdown, otherwise a single-accent histogram of
		// the plain token series.
		chartShown := false
		var (
			layers [][]float64
			colors []color.NRGBA
		)
		if showStackedChart(usage24h) {
			layers, colors = stackedChartData(usage24h)
		} else if showUsageChart(usage24h) {
			layers = [][]float64{seriesTokenValues(usage24h.Series)}
			colors = []color.NRGBA{modelBarColor(0)}
		}
		if len(layers) > 0 {
			if img := histogramChartPNG(
				layers, colors,
				usageChartWidthPt*retinaScale, usageChartHeightPt*retinaScale,
			); img != nil {
				mUsageChart.SetImage(img, usageChartWidthPt, usageChartHeightPt)
				mUsageChart.Show()
				chartShown = true
			}
		}
		if !chartShown {
			mUsageChart.Hide()
		}

		// Per-model breakdown: a native text title, e.g. "opus-4-8 — 1.2M · $0.90",
		// paired with a rank-hued proportional bar image on the left.
		models := usage24h.ByModel
		modelShown := 0
		for i, slot := range mUsageModels {
			if usage24h.hasData() && i < len(models) {
				slot.SetTitle(usageModelTitle(models[i]))
				if img := slotBarPNG(
					modelShare(models[i], usage24h.TotalTokens), modelBarColor(i),
					usageSlotWidthPt*retinaScale, usageSlotHeightPt*retinaScale,
					usageBarWidthPt*retinaScale, usageBarHeightPt*retinaScale,
				); img != nil {
					slot.SetImage(img, usageSlotWidthPt, usageSlotHeightPt)
				}
				slot.Show()
				modelShown++
			} else {
				slot.Hide()
			}
		}
		// Footer.
		if usage24h.hasData() {
			if foot := usageFooter(usage24h.LastUpdated, now); foot != "" {
				mUsageFooter.SetTitle(foot)
				mUsageFooter.Show()
			} else {
				mUsageFooter.Hide()
			}
		} else {
			mUsageFooter.Hide()
		}

		// Reflect today's usage in the menu-bar status widget (or the plain icon).
		applyStatusVisual(true)
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
		mUsageParent.Hide()
		hideUsageChildren()
		// The menu-bar status visual (plain icon vs widget) is applied separately
		// by applyStatusVisual at each caller of hideData.
	}

	refresh := func() {
		healthy := serverHealthy()
		mStatus.SetTitle(statusLine(healthy, serverPort()))
		if !healthy {
			mStatus.SetIcon(iconDotRed)
			mStart.Enable()
			mStop.Disable()
			hideData()
			mEnterKey.Hide()
			// Server down → revert the menu-bar to the plain inactive icon.
			applyStatusVisual(false)
			return
		}

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
			// Healthy but unauthorized → plain active icon, no widget.
			applyStatusVisual(true)
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
		// The Success row gets a small sparkline of the rate over recent
		// refreshes instead of a static glyph, tinted by the same traffic-light
		// rating (flat baseline until history accumulates).
		if stats.OK && stats.Total > 0 {
			successHist = pushSparkSample(successHist, 100*float64(stats.Successful)/float64(stats.Total))
			if icon := sparklineIconPNG(successHist, levelSparkColor(successLevel(stats))); icon != nil {
				mSuccess.SetIcon(icon)
			}
		} else {
			applyLevelIcon(mSuccess, successLevel(stats), iconSuccess, iconSuccessGreen, iconSuccessYellow, iconSuccessRed)
		}
		setRow(mResponse, metricResponse(stats))
		applyLevelIcon(mResponse, responseLevel(stats), iconGauge, iconGaugeGreen, iconGaugeYellow, iconGaugeRed)
		mem := serverMemoryMB()
		setRow(mMemory, metricMemory(mem))
		applyLevelIcon(mMemory, memoryLevel(mem), iconCPU, iconCPUGreen, iconCPUYellow, iconCPURed)

		// Usage — re-fetched on a slower cadence than health (it moves slowly),
		// then rendered every refresh so the "last updated" footer stays fresh
		// and the Claude quota rows pick up the latest background poll. A 404
		// (older server without the endpoint) hides the whole section.
		if lastUsageFetch.IsZero() || time.Since(lastUsageFetch) >= usageRefreshInterval {
			usage24h = fetchUsageStats("24h", key)
			lastUsageFetch = time.Now()
		}
		renderUsage()
	}
	refresh()

	// Claude subscription quota lives on its own goroutine and cadence so a slow
	// or hanging call to api.anthropic.com can never block the menu refresh. It
	// writes the latest reading under a mutex; renderUsage reads it each refresh.
	go func() {
		poll := func() {
			q := fetchClaudeQuotaNow()
			claudeMu.Lock()
			claudeVal = q
			claudeMu.Unlock()
		}
		poll()
		ticker := time.NewTicker(claudeQuotaInterval)
		defer ticker.Stop()
		for range ticker.C {
			poll()
		}
	}()

	// Starting-arc animation: while the server is in the starting state, advance
	// the arc's rotation on a short ticker. Outside that state this loop is a
	// cheap no-op; state transitions themselves are discovered by the 5s refresh
	// (and the explicit refresh after the Start/Restart menu actions).
	go func() {
		ticker := time.NewTicker(400 * time.Millisecond)
		defer ticker.Stop()
		for range ticker.C {
			if serverState(lastState.Load()) == serverStarting {
				spinPhase.Add(1)
				applyStatusVisual(false)
			}
		}
	}()

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
				openPage("agents")
			case <-mAgentsOpen.ClickedCh:
				openPage("agents")
			case <-mSuccess.ClickedCh:
				openPage("executions")
			case <-mResponse.ClickedCh:
				openPage("executions")
			case <-mMemory.ClickedCh:
				openPage("dashboard")
			case <-mUsageParent.ClickedCh:
				openPage("dashboard")
			case <-mUsageOpen.ClickedCh:
				openPage("dashboard")
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

func openDashboard() { openPage("") }

// openPage opens a view, preferring the AgentField desktop app over the
// browser: `open agentfield://…` succeeds only when something has registered
// the scheme (the desktop app does on install) and fails fast when nothing
// has — in which case the same page opens as web UI in the default browser.
// Run (not Start) on the deep link because the fallback needs its exit code.
func openPage(page string) {
	if exec.Command("open", deepLinkForPage(page)).Run() == nil {
		return
	}
	_ = exec.Command("open", browserURLForPage(page)).Start()
}

func openLogs() {
	_ = exec.Command("open", serverLogPath()).Start()
}
