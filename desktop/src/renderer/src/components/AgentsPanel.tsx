import { useState } from 'react'
import type { ReactElement } from 'react'
import type { AgentFieldSnapshot } from '../../../shared/types'

type AgentAction = 'start' | 'stop' | 'restart'

interface AgentsPanelProps {
  registry: AgentFieldSnapshot['registry'] | null
  /** Called after a lifecycle action so the snapshot refreshes promptly. */
  onChanged: () => void
}

const BADGE_LABEL: Record<string, string> = {
  running: 'Running',
  stopped: 'Stopped',
  unknown: 'Unknown'
}

const BUSY_LABEL: Record<AgentAction, string> = {
  start: 'Starting…',
  stop: 'Stopping…',
  restart: 'Restarting…'
}

export function AgentsPanel({ registry, onChanged }: AgentsPanelProps): ReactElement {
  return (
    <div className="panel">
      <AgentsBody registry={registry} onChanged={onChanged} />
    </div>
  )
}

function AgentsBody({ registry, onChanged }: AgentsPanelProps) {
  const [busy, setBusy] = useState<{ name: string; action: AgentAction } | null>(null)
  const [failure, setFailure] = useState<{ name: string; message: string } | null>(null)

  if (!registry) {
    return <div className="empty">Loading…</div>
  }
  if (registry.error) {
    return <div className="callout error">{registry.error}</div>
  }
  if (!registry.exists || registry.agents.length === 0) {
    return (
      <div className="empty">
        <p>No agents installed yet.</p>
        <p className="secondary">Head to Install to add your first one.</p>
      </div>
    )
  }

  const run = async (action: AgentAction, name: string) => {
    setBusy({ name, action })
    setFailure(null)
    const result = await window.agentfield.agentAction(action, name)
    setBusy(null)
    if (!result.ok) setFailure({ name, message: result.message })
    onChanged()
  }

  return (
    <ul className="row-list">
      {registry.agents.map((agent) => {
        const isBusy = busy?.name === agent.name
        const running = agent.badge === 'running'
        return (
          <li key={agent.name} className="row">
            <span className={`row-dot ${agent.badge}`} aria-hidden="true" />
            <div className="row-main">
              <span className="row-title">{agent.name}</span>
              {agent.description && <span className="row-sub">{agent.description}</span>}
              {isBusy && busy && (
                <span className="row-progress">{BUSY_LABEL[busy.action]}</span>
              )}
              {failure?.name === agent.name && !isBusy && (
                <span className="row-progress error-text">{failure.message}</span>
              )}
            </div>
            <div className="row-side">
              {running && agent.port !== null && <span className="row-meta">:{agent.port}</span>}
              <span className={`badge ${agent.badge}`}>
                {BADGE_LABEL[agent.badge] ?? agent.badge}
              </span>
              <div className="row-actions">
                {running ? (
                  <>
                    <button
                      className="action-button"
                      disabled={busy !== null}
                      onClick={() => void run('restart', agent.name)}
                    >
                      Restart
                    </button>
                    <button
                      className="action-button"
                      disabled={busy !== null}
                      onClick={() => void run('stop', agent.name)}
                    >
                      Stop
                    </button>
                  </>
                ) : (
                  <button
                    className="action-button primary"
                    disabled={busy !== null}
                    onClick={() => void run('start', agent.name)}
                  >
                    Start
                  </button>
                )}
              </div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
