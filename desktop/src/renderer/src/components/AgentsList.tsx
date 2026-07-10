import type { AgentFieldSnapshot } from '../../../shared/types'

interface AgentsListProps {
  registry: AgentFieldSnapshot['registry'] | null
}

export function AgentsList({ registry }: AgentsListProps) {
  return (
    <section className="card">
      <header className="card-header">
        <h2>Installed agents</h2>
        {registry && registry.exists && (
          <span className="muted">{registry.agents.length} installed</span>
        )}
      </header>
      <AgentsListBody registry={registry} />
    </section>
  )
}

function AgentsListBody({ registry }: AgentsListProps) {
  if (!registry) {
    return <p className="muted">Loading…</p>
  }

  if (registry.error) {
    return <div className="banner error">{registry.error}</div>
  }

  if (!registry.exists) {
    return (
      <div className="empty-state">
        <p>No AgentField installation found (~/.agentfield missing).</p>
        <p className="muted">
          Install an agent with <code>af install &lt;source&gt;</code> to get started.
        </p>
      </div>
    )
  }

  if (registry.agents.length === 0) {
    return (
      <div className="empty-state">
        <p>No agents installed yet.</p>
      </div>
    )
  }

  return (
    <table className="agents-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Version</th>
          <th>Language</th>
          <th>Port</th>
          <th>PID</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {registry.agents.map((agent) => (
          <tr key={agent.name}>
            <td>
              <span className="agent-name">{agent.name}</span>
              {agent.description && <span className="muted small"> — {agent.description}</span>}
            </td>
            <td>{agent.version || '—'}</td>
            <td>{agent.language ?? '—'}</td>
            <td>{agent.port ?? '—'}</td>
            <td>{agent.pid ?? '—'}</td>
            <td>
              <span className={`badge ${agent.badge}`}>{agent.badge}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
