import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import { type View, isView } from '../../shared/deeplink'
import type { AgentFieldSnapshot } from '../../shared/types'
import { Sidebar } from './components/Sidebar'
import { DashboardView } from './components/DashboardView'
import { AgentsPanel, LocalControlPlaneRestartBanner } from './components/AgentsPanel'
import { ActivityPanel } from './components/ActivityPanel'
import { InstallPanel } from './components/InstallPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { CloudPanel } from './components/CloudPanel'
import { KeysBanner } from './components/KeysBanner'
import { StarBanner } from './components/StarBanner'
import { UpdateBanner } from './components/UpdateBanner'
import { CloudUpdateBanner } from './components/CloudUpdateBanner'

const POLL_INTERVAL_MS = 5000

export type CpTone = 'green' | 'yellow' | 'red' | 'gray'

export function controlPlaneStatus(snapshot: AgentFieldSnapshot | null): {
  tone: CpTone
  label: string
  detail?: string
} {
  const cp = snapshot?.controlPlane
  if (!cp) return { tone: 'gray', label: 'Checking…' }
  if (cp.healthy) return { tone: 'green', label: 'Running' }
  if (cp.reachable && cp.recognized) {
    return { tone: 'yellow', label: 'Unhealthy', detail: cp.error }
  }
  if (cp.reachable) {
    return { tone: 'yellow', label: 'Port in use', detail: cp.error }
  }
  return {
    tone: 'red',
    label: 'Not running',
    detail: 'AgentField server is not running.'
  }
}

// `install` maps to the Agents view with add-mode open (DESIGN.md §2.1) —
// the deep link stays valid, but there is no separate Install place anymore.
const VIEW_TITLES: Record<View, string> = {
  home: 'Home',
  install: 'Agents',
  agents: 'Agents',
  activity: 'Activity',
  settings: 'Settings',
  cloud: 'Remote'
}

/**
 * Cold-launch landing view. Bundled nodes still provisioning win: their rows
 * live in the Agents library and watching them arrive is the first thing a
 * brand-new user should see — dropping them into the marketplace instead would
 * ask them to install what the app is already installing. Otherwise an empty
 * library opens add-mode (the `install` view, DESIGN.md §4.11) and a stocked
 * one opens Home.
 */
export function defaultView(bundledCount: number, agentCount: number): View {
  if (bundledCount > 0) return 'agents'
  if (agentCount === 0) return 'install'
  return 'home'
}

/**
 * Whether a snapshot carries enough to decide the cold-launch route. The
 * registry is read through the control plane, so the first poll after a cold
 * autostart sees "no registry" while the server is still coming up — routing
 * on that would send a user with a stocked library to the marketplace every
 * time. Wait for a readable registry (or provisioning rows, which only exist
 * once the control plane answered); until then the initial Home view and its
 * control-plane status callout are the right thing to show.
 */
export function canDecideDefaultRoute(args: {
  registryExists: boolean
  registryError: string | null | undefined
  bundledCount: number
}): boolean {
  return (args.registryExists && !args.registryError) || args.bundledCount > 0
}

export function shouldRerouteToBundled(args: {
  view: View
  bundledCount: number
  deepLinkHandled: boolean
  userNavigated: boolean
  alreadyRerouted: boolean
}): boolean {
  return (
    args.view === 'install' &&
    args.bundledCount > 0 &&
    !args.deepLinkHandled &&
    !args.userNavigated &&
    !args.alreadyRerouted
  )
}

// ⌘1–⌘5 (Ctrl on Win/Linux) in nav order (DESIGN.md §4.17).
const SHORTCUT_VIEWS: View[] = ['home', 'agents', 'activity', 'settings', 'cloud']

/** True when the keystroke belongs to a text control, not the app. */
function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  )
}

export default function App() {
  const platform = window.agentfield.platform
  const reducedMotion = useReducedMotion()
  const [snapshot, setSnapshot] = useState<AgentFieldSnapshot | null>(null)
  const [ipcError, setIpcError] = useState<string | null>(null)
  const [view, setView] = useState<View>('home')
  const [startingCp, setStartingCp] = useState(false)
  /** Agents add-mode opened via the "+ Add agent" header action. */
  const [addAgentOpen, setAddAgentOpen] = useState(false)
  /**
   * The keys banner is on screen. Only App sees the whole banner stack, so it
   * carries the signal from the banner that computes it to the one that has to
   * yield — the star prompt must not ask for a favour while installed agents
   * cannot run.
   */
  const [keysBannerShowing, setKeysBannerShowing] = useState(false)
  const defaultRouteApplied = useRef(false)
  const deepLinkHandled = useRef(false)
  const userNavigated = useRef(false)
  const bundledRerouted = useRef(false)

  useEffect(() => {
    // Lets styles.css inset window chrome for macOS traffic lights vs the
    // Windows caption-button overlay.
    document.body.dataset.platform = platform
  }, [platform])

  useEffect(() => {
    // agentfield://<view> deep links land here via the main process. Deep
    // links from before this listener existed (a link that cold-started the
    // app) are collected by announceReady once the subscription is live.
    const unsubscribe = window.agentfield.onNavigate((v) => {
      if (isView(v)) {
        deepLinkHandled.current = true
        setAddAgentOpen(false)
        setView(v)
      }
    })
    void window.agentfield.announceReady().then((v) => {
      if (v !== null && isView(v)) {
        deepLinkHandled.current = true
        setAddAgentOpen(false)
        setView(v)
      }
    })
    return unsubscribe
  }, [])

  const refresh = useCallback(async () => {
    try {
      const next = await window.agentfield.getSnapshot()
      setSnapshot(next)
      setIpcError(null)
    } catch (err) {
      setIpcError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  // Bundled nodes still being provisioned this launch (shared/bundled.ts).
  // Derived before the routing effect because the cold-launch view and the
  // add-mode decision both hang off it.
  const bundled = snapshot?.bundled ?? []

  // Cold-launch default (see defaultView). Deep links win; do not re-apply on
  // later polls or remember the last view.
  useEffect(() => {
    if (!snapshot || defaultRouteApplied.current) return
    if (
      !canDecideDefaultRoute({
        registryExists: snapshot.registry.exists,
        registryError: snapshot.registry.error,
        bundledCount: bundled.length
      })
    ) {
      return
    }
    defaultRouteApplied.current = true
    if (deepLinkHandled.current) return
    setView(defaultView(bundled.length, snapshot.registry.agents.length))
  }, [snapshot, bundled.length])

  useEffect(() => {
    // The first snapshot normally arrives before main has seeded provisioning
    // rows, so the cold-launch default may already have selected add-mode.
    if (
      shouldRerouteToBundled({
        view,
        bundledCount: bundled.length,
        deepLinkHandled: deepLinkHandled.current,
        userNavigated: userNavigated.current,
        alreadyRerouted: bundledRerouted.current
      })
    ) {
      bundledRerouted.current = true
      setView('agents')
    }
  }, [view, bundled.length])

  const handleStartControlPlane = useCallback(async () => {
    setStartingCp(true)
    setIpcError(null)
    try {
      const result = await window.agentfield.startControlPlane()
      if (!result.ok) setIpcError(result.message)
      await refresh()
    } catch (err) {
      setIpcError(err instanceof Error ? err.message : String(err))
    } finally {
      setStartingCp(false)
    }
  }, [refresh])

  const cp = controlPlaneStatus(snapshot)
  const agents = snapshot?.registry.agents ?? []
  const provisioningNames = bundled
    .filter((node) => node.phase === 'pending' || node.phase === 'installing')
    .map((node) => node.name)

  // Agents view, two modes (DESIGN.md §4.11). Add-mode when: the install
  // deep link addressed it, "+ Add agent" was clicked, or the library is
  // empty (the marketplace IS the empty state). A launch with bundled nodes
  // still arriving is not empty — flipping it into add-mode would hide the
  // very rows the app is filling in.
  const agentsSelected = view === 'agents' || view === 'install'
  const libraryEmpty =
    snapshot !== null &&
    !snapshot.registry.error &&
    agents.length === 0 &&
    bundled.length === 0
  const agentsAddMode = agentsSelected && (view === 'install' || addAgentOpen || libraryEmpty)

  // Navigation from the sidebar or in-view CTAs closes add-mode so the
  // Agents view comes back in library mode next time.
  const navigate = useCallback((v: View) => {
    userNavigated.current = true
    setAddAgentOpen(false)
    setView(v)
  }, [])

  const closeAddMode = useCallback(() => {
    userNavigated.current = true
    setAddAgentOpen(false)
    setView('agents')
  }, [])

  // Keyboard ergonomics (DESIGN.md §4.17): ⌘/Ctrl+1–4 switch views, ⌘/Ctrl+R
  // refreshes the snapshot (preventDefault so Electron doesn't reload the
  // window), Esc closes Agents add-mode back to the library when non-empty.
  const agentCount = agents.length
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (isEditableTarget(event.target)) return
      const mod = event.metaKey || event.ctrlKey
      if (mod && !event.shiftKey && !event.altKey) {
        const index = Number.parseInt(event.key, 10) - 1
        if (index >= 0 && index < SHORTCUT_VIEWS.length) {
          event.preventDefault()
          navigate(SHORTCUT_VIEWS[index])
          return
        }
        if (event.key === 'r' || event.key === 'R') {
          event.preventDefault()
          void refresh()
          return
        }
      }
      if (event.key === 'Escape' && agentsAddMode && agentCount > 0) {
        closeAddMode()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [navigate, refresh, closeAddMode, agentsAddMode, agentCount])

  // Sidebar highlight: `install` is Agents territory.
  const navView: View = view === 'install' ? 'agents' : view

  return (
    <div className="app">
      <Sidebar
        view={navView}
        onSelect={navigate}
        cpTone={cp.tone}
        cpLabel={cp.label}
        onStartControlPlane={() => void handleStartControlPlane()}
        startingControlPlane={startingCp}
      />

      <div className="main">
        <header
          className={`view-header ${platform !== 'darwin' ? 'window-controls-safe' : ''}`}
        >
          <h1>{VIEW_TITLES[view]}</h1>
          {agentsSelected && !agentsAddMode && (
            <div className="view-header-action">
              <button
                type="button"
                className="action-button primary"
                onClick={() => setAddAgentOpen(true)}
              >
                + Add agent
              </button>
            </div>
          )}
        </header>
        <LocalControlPlaneRestartBanner
          status={snapshot?.localControlPlaneRestart ?? null}
        />
        <UpdateBanner />
        <CloudUpdateBanner />
        {/* Blocked-agents warning sits above the star ask: one reports the
            product cannot work, the other is a favour. */}
        <KeysBanner
          snapshot={snapshot}
          view={navView}
          onNavigate={navigate}
          onShowingChange={setKeysBannerShowing}
        />
        <StarBanner snapshot={snapshot} keysBannerShowing={keysBannerShowing} />
        <div className="view-body">
          {ipcError && <div className="callout error">{ipcError}</div>}
          {cp.tone === 'red' ? (
            <div className="callout">
              {cp.detail}
              <div className="callout-actions">
                <button
                  type="button"
                  className="action-button primary"
                  disabled={startingCp}
                  onClick={() => void handleStartControlPlane()}
                >
                  {startingCp ? 'Starting…' : 'Start AgentField server'}
                </button>
              </div>
            </div>
          ) : (
            cp.detail && <div className="callout">{cp.detail}</div>
          )}

          {/* View change (DESIGN.md §5.2): 160ms opacity crossfade + 4px
              rise on enter, exit-then-enter. `initial={false}` keeps the
              first paint settled. */}
          <AnimatePresence mode="wait" initial={false}>
            <m.div
              className="view-content"
              key={navView}
              initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            >
              {view === 'home' && (
                <DashboardView snapshot={snapshot} onNavigate={navigate} />
              )}
              {agentsSelected &&
                (agentsAddMode ? (
                  <InstallPanel
                    installedAgents={agents}
                    provisioningNames={provisioningNames}
                    onInstalled={() => void refresh()}
                    libraryCount={agents.length}
                    onBackToLibrary={agents.length > 0 ? closeAddMode : undefined}
                  />
                ) : (
                  <AgentsPanel
                    registry={snapshot?.registry ?? null}
                    bundled={bundled}
                    onChanged={() => void refresh()}
                  />
                ))}
              {view === 'activity' && (
                <ActivityPanel
                  executions={snapshot?.executions ?? null}
                  controlPlaneUp={snapshot?.controlPlane.recognized ?? false}
                />
              )}
              {view === 'settings' && <SettingsPanel agents={snapshot?.registry.agents ?? []} />}
              {view === 'cloud' && <CloudPanel />}
            </m.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
