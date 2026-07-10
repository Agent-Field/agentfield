import { useCallback, useEffect, useState } from 'react'
import type { AgentFieldSnapshot } from '../../shared/types'
import { AgentsList } from './components/AgentsList'
import { ControlPlaneCard } from './components/ControlPlaneCard'

const POLL_INTERVAL_MS = 5000

export default function App() {
  const [snapshot, setSnapshot] = useState<AgentFieldSnapshot | null>(null)
  const [ipcError, setIpcError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const refresh = useCallback(async () => {
    setRefreshing(true)
    try {
      const next = await window.agentfield.getSnapshot()
      setSnapshot(next)
      setIpcError(null)
    } catch (err) {
      setIpcError(err instanceof Error ? err.message : String(err))
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  const lastUpdated = snapshot
    ? new Date(snapshot.fetchedAt).toLocaleTimeString()
    : null

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>AgentField Desktop</h1>
          <p className="subtitle">Read-only dashboard · polls every {POLL_INTERVAL_MS / 1000}s</p>
        </div>
        <div className="header-actions">
          {lastUpdated && <span className="muted">Last updated {lastUpdated}</span>}
          <button onClick={() => void refresh()} disabled={refreshing}>
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </header>

      {ipcError && (
        <div className="banner error">Failed to fetch snapshot: {ipcError}</div>
      )}

      <main className="app-main">
        <ControlPlaneCard controlPlane={snapshot?.controlPlane ?? null} />
        <AgentsList registry={snapshot?.registry ?? null} />
      </main>
    </div>
  )
}
