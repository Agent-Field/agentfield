import type { ReactElement } from 'react'
import type { AgentFieldSnapshot } from '../../../shared/types'

interface AgentsPanelProps {
  registry: AgentFieldSnapshot['registry'] | null
}

const BADGE_LABEL: Record<string, string> = {
  running: 'Running',
  stopped: 'Stopped',
  unknown: 'Unknown'
}

export function AgentsPanel({ registry }: AgentsPanelProps): ReactElement {
  return (
    <div className="panel">
      <AgentsBody registry={registry} />
    </div>
  )
}

function AgentsBody({ registry }: AgentsPanelProps) {
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
  return (
    <ul className="row-list">
      {registry.agents.map((agent) => (
        <li key={agent.name} className="row">
          <span className={`row-dot ${agent.badge}`} aria-hidden="true" />
          <div className="row-main">
            <span className="row-title">{agent.name}</span>
            {agent.description && <span className="row-sub">{agent.description}</span>}
          </div>
          <div className="row-side">
            {agent.badge === 'running' && agent.port !== null && (
              <span className="row-meta">:{agent.port}</span>
            )}
            <span className={`badge ${agent.badge}`}>
              {BADGE_LABEL[agent.badge] ?? agent.badge}
            </span>
          </div>
        </li>
      ))}
    </ul>
  )
}
