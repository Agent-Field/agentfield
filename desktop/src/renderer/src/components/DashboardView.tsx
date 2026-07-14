import type { ReactElement } from 'react'
import type { AgentFieldSnapshot } from '../../../shared/types'
import type { View } from './Sidebar'
import { ExecutionRow } from './ActivityPanel'

interface DashboardViewProps {
  snapshot: AgentFieldSnapshot | null
  onNavigate: (view: View) => void
}

function Tile({
  label,
  value,
  context,
  tone
}: {
  label: string
  value: string
  context?: string
  tone?: 'good' | 'warn' | 'bad'
}) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className={tone ? `tile-value ${tone}` : 'tile-value'}>{value}</span>
      {context && <span className="tile-context">{context}</span>}
    </div>
  )
}

function todayDelta(today: number, yesterday: number): string {
  if (yesterday === 0) return today > 0 ? 'none yesterday' : ''
  const diff = today - yesterday
  if (diff === 0) return 'same as yesterday'
  return `${diff > 0 ? '+' : ''}${diff} vs yesterday`
}

// Color reinforces the number (never replaces it): ≥90% healthy, ≥60% needs
// attention, below that something is systematically failing.
function successTone(rate: number): 'good' | 'warn' | 'bad' {
  if (rate >= 90) return 'good'
  if (rate >= 60) return 'warn'
  return 'bad'
}

export function DashboardView({ snapshot, onNavigate }: DashboardViewProps): ReactElement {
  const metrics = snapshot?.metrics ?? null
  const executions = snapshot?.executions ?? null
  const runningNow = executions?.running.length ?? 0
  const off = metrics === null

  return (
    <>
      <div className="tile-grid">
        <Tile
          label="Agents running"
          value={off ? '—' : `${metrics.agentsRunning}`}
          context={off ? undefined : `of ${metrics.agentsTotal} installed`}
        />
        <Tile
          label="Executing now"
          value={off ? '—' : `${runningNow}`}
          context={off || runningNow === 0 ? undefined : 'in flight'}
        />
        <Tile
          label="Runs today"
          value={off ? '—' : `${metrics.executionsToday}`}
          context={off ? undefined : todayDelta(metrics.executionsToday, metrics.executionsYesterday)}
        />
        <Tile
          label="Success rate"
          value={
            off || metrics.successRate === null
              ? '—'
              : `${Math.round(metrics.successRate)}%`
          }
          tone={
            off || metrics.successRate === null
              ? undefined
              : successTone(metrics.successRate)
          }
          context={off ? undefined : 'last 24 hours'}
        />
      </div>

      <section>
        <div className="subhead">
          <h2 className="section-title">Recent activity</h2>
          <button className="link-button" onClick={() => onNavigate('activity')}>
            See all
          </button>
        </div>
        <div className="panel">
          {executions === null ? (
            <div className="empty secondary">
              Activity appears here once the control plane is running.
            </div>
          ) : executions.running.length === 0 && executions.recent.length === 0 ? (
            <div className="empty secondary">No executions yet.</div>
          ) : (
            <ul className="row-list">
              {executions.running.map((run) => (
                <ExecutionRow key={run.runId} run={run} live />
              ))}
              {executions.recent.slice(0, 3).map((run) => (
                <ExecutionRow key={run.runId} run={run} />
              ))}
            </ul>
          )}
        </div>
      </section>
    </>
  )
}
