import type { AgentFieldSnapshot } from '../../../shared/types'

interface ControlPlaneCardProps {
  controlPlane: AgentFieldSnapshot['controlPlane'] | null
}

export function ControlPlaneCard({ controlPlane }: ControlPlaneCardProps) {
  let dotClass = 'gray'
  let label = 'Checking…'
  if (controlPlane) {
    if (controlPlane.healthy) {
      dotClass = 'green'
      label = 'Running'
    } else if (controlPlane.reachable && controlPlane.recognized) {
      dotClass = 'yellow'
      label = 'Reachable (unhealthy)'
    } else if (controlPlane.reachable) {
      dotClass = 'yellow'
      label = 'Another service is on this port'
    } else {
      dotClass = 'red'
      label = 'Not reachable'
    }
  }

  return (
    <section className="card">
      <header className="card-header">
        <h2>Control plane</h2>
        <code className="muted">{controlPlane?.baseUrl ?? 'http://localhost:8080'}</code>
      </header>
      <div className="status-line">
        <span className={`status-dot ${dotClass}`} aria-hidden="true" />
        <span className="status-label">{label}</span>
      </div>
      {controlPlane?.error && <p className="muted small">{controlPlane.error}</p>}
    </section>
  )
}
