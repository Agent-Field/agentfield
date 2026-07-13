import { useCallback, useEffect, useState } from 'react'
import type { AgentFieldSnapshot } from '../../shared/types'
import { Sidebar, type View } from './components/Sidebar'
import { DashboardView } from './components/DashboardView'
import { AgentsPanel } from './components/AgentsPanel'
import { ActivityPanel } from './components/ActivityPanel'
import { InstallPanel } from './components/InstallPanel'

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
    detail: 'Start the control plane with `af server`, then this app picks it up automatically.'
  }
}

const VIEW_TITLES: Record<View, string> = {
  dashboard: 'Dashboard',
  agents: 'Agents',
  activity: 'Activity',
  install: 'Install'
}

export default function App() {
  const [snapshot, setSnapshot] = useState<AgentFieldSnapshot | null>(null)
  const [ipcError, setIpcError] = useState<string | null>(null)
  const [view, setView] = useState<View>('dashboard')

  useEffect(() => {
    // Lets styles.css inset window chrome for macOS traffic lights vs the
    // Windows caption-button overlay.
    document.body.dataset.platform = window.agentfield.platform
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

  const cp = controlPlaneStatus(snapshot)
  const installedNames = snapshot?.registry.agents.map((a) => a.name) ?? []

  return (
    <div className="app">
      <Sidebar view={view} onSelect={setView} cpTone={cp.tone} cpLabel={cp.label} />

      <div className="main">
        <header className="view-header">
          <h1>{VIEW_TITLES[view]}</h1>
        </header>
        <div className="view-body">
          {ipcError && <div className="callout error">{ipcError}</div>}
          {cp.detail && <div className="callout">{cp.detail}</div>}

          {view === 'dashboard' && (
            <DashboardView snapshot={snapshot} onNavigate={setView} />
          )}
          {view === 'agents' && <AgentsPanel registry={snapshot?.registry ?? null} />}
          {view === 'activity' && (
            <ActivityPanel
              executions={snapshot?.executions ?? null}
              controlPlaneUp={snapshot?.controlPlane.recognized ?? false}
            />
          )}
          {view === 'install' && (
            <InstallPanel installedNames={installedNames} onInstalled={() => void refresh()} />
          )}
        </div>
      </div>
    </div>
  )
}
