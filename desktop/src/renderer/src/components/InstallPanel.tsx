import { useEffect, useState } from 'react'
import type { CatalogEntry } from '../../../shared/types'

interface InstallPanelProps {
  installedNames: string[]
  onInstalled: () => void
}

type InstallPhase =
  | { state: 'idle' }
  | { state: 'installing'; name: string; progress: string }
  | { state: 'done'; name: string; ok: boolean; message: string }

export function InstallPanel({ installedNames, onInstalled }: InstallPanelProps) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [phase, setPhase] = useState<InstallPhase>({ state: 'idle' })
  /** Catalog entry with the uninstall confirm step open. */
  const [confirming, setConfirming] = useState<string | null>(null)

  useEffect(() => {
    void window.agentfield.getCatalog().then(setCatalog)
  }, [])

  useEffect(() => {
    return window.agentfield.onInstallProgress((line) => {
      setPhase((prev) =>
        prev.state === 'installing' ? { ...prev, progress: line } : prev
      )
    })
  }, [])

  const install = async (name: string) => {
    setPhase({ state: 'installing', name, progress: 'Starting…' })
    const result = await window.agentfield.install(name)
    setPhase({ state: 'done', name, ok: result.ok, message: result.message })
    if (result.ok) onInstalled()
  }

  const uninstall = async (name: string) => {
    setConfirming(null)
    setPhase({ state: 'installing', name, progress: 'Uninstalling…' })
    const result = await window.agentfield.uninstall(name)
    setPhase({ state: 'done', name, ok: result.ok, message: result.message })
    onInstalled()
  }

  const installing = phase.state === 'installing'

  return (
    <>
      <p className="view-lede">
        Curated agent nodes, installed with one click. More arrive as the catalog grows.
      </p>
      <div className="panel">
        {catalog.length === 0 && <div className="empty secondary">Loading catalog…</div>}
        <ul className="row-list">
          {catalog.map((entry) => {
            const isInstalled = installedNames.includes(entry.name)
            const busy = installing && phase.name === entry.name
            return (
              <li key={entry.name} className="row">
                <div className="row-main">
                  <span className="row-title">{entry.name}</span>
                  <span className="row-sub">{entry.description}</span>
                  {busy && phase.state === 'installing' && (
                    <span className="row-progress">{phase.progress}</span>
                  )}
                  {phase.state === 'done' && phase.name === entry.name && !phase.ok && (
                    <span className="row-progress error-text">{phase.message}</span>
                  )}
                  {confirming === entry.name && (
                    <span className="row-progress warn-text">
                      Stops the agent and removes its files, registry entry, and
                      agent-scoped secrets. Shared keys stay.
                    </span>
                  )}
                </div>
                <div className="row-side">
                  {isInstalled ? (
                    confirming === entry.name ? (
                      <div className="row-actions">
                        <button
                          className="action-button danger"
                          disabled={installing}
                          onClick={() => void uninstall(entry.name)}
                        >
                          Uninstall
                        </button>
                        <button
                          className="action-button"
                          disabled={installing}
                          onClick={() => setConfirming(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div className="row-actions">
                        <span className="installed-check">Installed ✓</span>
                        <button
                          className="action-button"
                          disabled={installing}
                          onClick={() => setConfirming(entry.name)}
                        >
                          Uninstall
                        </button>
                      </div>
                    )
                  ) : (
                    <button
                      className="install-button"
                      disabled={installing}
                      onClick={() => void install(entry.name)}
                    >
                      {busy ? 'Installing…' : 'Install'}
                    </button>
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      </div>
    </>
  )
}
