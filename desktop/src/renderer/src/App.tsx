import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import { type View, isView } from '../../shared/deeplink'
import type { AgentFieldSnapshot } from '../../shared/types'
import { Sidebar } from './components/Sidebar'
import { DashboardView } from './components/DashboardView'
import { AgentsPanel } from './components/AgentsPanel'
import { ActivityPanel } from './components/ActivityPanel'
import { InstallPanel } from './components/InstallPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { StarBanner } from './components/StarBanner'
import { UpdateBanner } from './components/UpdateBanner'

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
  settings: 'Settings'
}

// ⌘1–⌘4 (Ctrl on Win/Linux) in nav order (DESIGN.md §4.17).
const SHORTCUT_VIEWS: View[] = ['home', 'agents', 'activity', 'settings']

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
  const defaultRouteApplied = useRef(false)
  const deepLinkHandled = useRef(false)

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

  // Cold-launch default: Agents add-mode (via the `install` view) when the
  // library is empty, otherwise Home. Deep links win; do not re-apply on
  // later polls or remember the last view.
  useEffect(() => {
    if (!snapshot || defaultRouteApplied.current) return
    defaultRouteApplied.current = true
    if (deepLinkHandled.current) return
    setView(snapshot.registry.agents.length === 0 ? 'install' : 'home')
  }, [snapshot])

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
  const installedNames = agents.map((a) => a.name)

  // Agents view, two modes (DESIGN.md §4.11). Add-mode when: the install
  // deep link addressed it, "+ Add agent" was clicked, or the library is
  // empty (the marketplace IS the empty state).
  const agentsSelected = view === 'agents' || view === 'install'
  const libraryEmpty =
    snapshot !== null && !snapshot.registry.error && agents.length === 0
  const agentsAddMode = agentsSelected && (view === 'install' || addAgentOpen || libraryEmpty)

  // Navigation from the sidebar or in-view CTAs closes add-mode so the
  // Agents view comes back in library mode next time.
  const navigate = useCallback((v: View) => {
    setAddAgentOpen(false)
    setView(v)
  }, [])

  const closeAddMode = useCallback(() => {
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
        <UpdateBanner />
        <StarBanner snapshot={snapshot} />
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
                    installedNames={installedNames}
                    onInstalled={() => void refresh()}
                    libraryCount={agents.length}
                    onBackToLibrary={agents.length > 0 ? closeAddMode : undefined}
                  />
                ) : (
                  <AgentsPanel
                    registry={snapshot?.registry ?? null}
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
            </m.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
