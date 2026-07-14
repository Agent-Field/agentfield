import type { ReactElement } from 'react'
import type { ExecutionsResult, ExecutionSummary } from '../../../shared/types'

interface ActivityPanelProps {
  executions: ExecutionsResult | null
  controlPlaneUp: boolean
}

function formatDuration(ms: number | null): string {
  if (ms === null) return ''
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms / 60_000)}m`
}

function formatStarted(iso: string): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  return new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

const STATUS_GLYPH: Record<string, string> = {
  succeeded: '✓',
  failed: '✕',
  cancelled: '⊘',
  timeout: '⏱'
}

export function ActivityPanel({ executions, controlPlaneUp }: ActivityPanelProps): ReactElement {
  return (
    <div className="panel">
      <ActivityBody executions={executions} controlPlaneUp={controlPlaneUp} />
    </div>
  )
}

function ActivityBody({ executions, controlPlaneUp }: ActivityPanelProps) {
  if (!controlPlaneUp || executions === null) {
    return (
      <div className="empty secondary">
        Activity appears here once the control plane is running.
      </div>
    )
  }
  if (executions.running.length === 0 && executions.recent.length === 0) {
    return <div className="empty secondary">No executions yet.</div>
  }
  return (
    <ul className="row-list">
      {executions.running.map((run) => (
        <ExecutionRow key={run.runId} run={run} live />
      ))}
      {executions.recent.map((run) => (
        <ExecutionRow key={run.runId} run={run} />
      ))}
    </ul>
  )
}

export function ExecutionRow({ run, live = false }: { run: ExecutionSummary; live?: boolean }) {
  const failed = !live && (run.status === 'failed' || run.status === 'timeout')
  return (
    <li className={`row ${live ? '' : 'row-past'}`}>
      {live ? (
        <span className="row-dot running pulse" aria-hidden="true" />
      ) : (
        <span className={`run-glyph ${run.status}`} aria-hidden="true">
          {STATUS_GLYPH[run.status] ?? '·'}
        </span>
      )}
      <div className="row-main">
        <span className="row-title">{run.displayName}</span>
        <span className="row-sub">
          {run.agentId}
          {live ? ` · started ${formatStarted(run.startedAt)}` : ''}
        </span>
        {failed && run.errorMessage && (
          <span className="row-sub error-text" title={run.errorMessage}>
            {run.errorMessage}
          </span>
        )}
      </div>
      <div className="row-side">
        {live ? (
          <span className="spinner" role="status" aria-label="running" />
        ) : (
          <span className="row-meta">{formatDuration(run.durationMs) || run.status}</span>
        )}
        <button
          className="action-button run-open"
          title="Open this run in the control-plane web UI"
          onClick={() => void window.agentfield.openWebUI(`/ui/runs/${encodeURIComponent(run.runId)}`)}
        >
          ↗
        </button>
      </div>
    </li>
  )
}
