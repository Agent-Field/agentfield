import { useEffect, useState, type ReactElement } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import type {
  AgentEnvReport,
  AgentFieldSnapshot,
  ExecutionSummary,
  UsageGroup
} from '../../../shared/types'
import type { View } from './Sidebar'
import { ExecutionRow } from './ActivityPanel'
import { SkeletonRows } from './Skeleton'
import { COMMUNITY_LINKS } from './communityLinks'

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
}): ReactElement {
  const reducedMotion = useReducedMotion()
  // Metric ticker (DESIGN.md §5.2): when the value string changes after
  // first paint, the old number rolls out and the new one rolls in
  // (~300ms crossfade). `initial={false}` on AnimatePresence skips the
  // initial mount; tabular nums mean zero layout shift.
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className={tone ? `tile-value ${tone}` : 'tile-value'}>
        <AnimatePresence mode="popLayout" initial={false}>
          <m.span
            key={value}
            className="tile-value-inner"
            initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reducedMotion ? { opacity: 0 } : { opacity: 0, y: -8 }}
            transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
          >
            {value}
          </m.span>
        </AnimatePresence>
      </span>
      {context ? <span className="tile-context">{context}</span> : null}
    </div>
  )
}

function successTone(rate: number): 'good' | 'warn' | 'bad' {
  if (rate >= 90) return 'good'
  if (rate >= 60) return 'warn'
  return 'bad'
}

function formatUsd(n: number): string {
  return `$${n.toFixed(2)}`
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function formatHarnessSpend(entry: UsageGroup): string {
  if (entry.costUsd !== null) return `${entry.key} ${formatUsd(entry.costUsd)}`
  return `${entry.key} ${formatTokens(entry.totalTokens)} tok`
}

function pickRecentRows(
  executions: AgentFieldSnapshot['executions']
): Array<{ run: ExecutionSummary; live: boolean }> {
  if (!executions) return []
  const rows: Array<{ run: ExecutionSummary; live: boolean }> = []
  for (const run of executions.running) {
    if (rows.length >= 3) break
    rows.push({ run, live: true })
  }
  for (const run of executions.recent) {
    if (rows.length >= 3) break
    rows.push({ run, live: false })
  }
  return rows
}

export function DashboardView({ snapshot, onNavigate }: DashboardViewProps): ReactElement {
  const [installSkills, setInstallSkills] = useState<boolean | null>(null)
  const [copied, setCopied] = useState(false)
  const [envReports, setEnvReports] = useState<AgentEnvReport[] | null>(null)

  const metrics = snapshot?.metrics ?? null
  const executions = snapshot?.executions ?? null
  const agents = snapshot?.registry.agents ?? []
  const agentNamesKey = agents.map((a) => a.name).join('\0')
  const cp = snapshot?.controlPlane
  const usage = snapshot?.usage ?? null
  const runningNow = executions?.running.length ?? 0
  const off = metrics === null
  const cpHealthy = cp?.healthy ?? false
  const baseUrl = cp?.baseUrl ?? 'http://localhost:8080'

  useEffect(() => {
    let cancelled = false
    void window.agentfield.getSettings().then((settings) => {
      if (!cancelled) setInstallSkills(settings.installSkills)
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!agentNamesKey) {
      setEnvReports(null)
      return
    }
    let cancelled = false
    void window.agentfield.getEnvReports().then((reports) => {
      if (!cancelled) setEnvReports(reports)
    })
    return () => {
      cancelled = true
    }
  }, [agentNamesKey])

  const unknownAgents = agents.filter((a) => a.badge === 'unknown')
  const needingKeys =
    envReports?.filter((r) => !r.satisfied).map((r) => r.agent) ?? []

  const hasCost = usage?.costUsd != null
  const hasTokens = (usage?.totalTokens ?? 0) > 0
  const showTokensTile = !hasCost && hasTokens

  const spendValue = (() => {
    if (usage === null) return '—'
    if (usage.costUsd != null) return formatUsd(usage.costUsd)
    return '—'
  })()
  const spendContext = (() => {
    if (usage === null) return 'No usage yet'
    if (usage.costUsd != null) return 'today'
    if (usage.totalTokens > 0) return 'cost unavailable'
    return 'No usage yet'
  })()

  const callersLine =
    usage?.byHarness && usage.byHarness.length > 0
      ? usage.byHarness.map(formatHarnessSpend).join(' · ')
      : null

  const recentRows = pickRecentRows(executions)

  const attentionChips: Array<{ key: string; label: string; view: View }> = []
  if (cp && !cp.healthy) {
    attentionChips.push({
      key: 'cp',
      label: cp.reachable && !cp.recognized ? 'Port in use' : 'Server not running',
      view: 'settings'
    })
  }
  if (unknownAgents.length > 0) {
    attentionChips.push({
      key: 'unknown',
      label:
        unknownAgents.length === 1
          ? `${unknownAgents[0].name} status unknown`
          : `${unknownAgents.length} agents status unknown`,
      view: 'agents'
    })
  }
  if (needingKeys.length > 0) {
    attentionChips.push({
      key: 'keys',
      label:
        needingKeys.length === 1
          ? `${needingKeys[0]} needs keys`
          : `${needingKeys.length} agents need keys`,
      view: 'agents'
    })
  }

  async function copyEndpoint(): Promise<void> {
    try {
      await navigator.clipboard.writeText(baseUrl)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard can fail in locked-down environments; stay quiet.
    }
  }

  if (snapshot !== null && agents.length === 0) {
    return (
      <>
        {cp && !cp.healthy ? (
          <div className="callout">AgentField server isn’t running.</div>
        ) : null}
        <div className="panel">
          <div className="empty">
            <p>
              Install your first agent node from GitHub. Coding agents on this machine can then call
              it through AgentField.
            </p>
            <div className="empty-actions">
              <button
                type="button"
                className="action-button primary"
                onClick={() => onNavigate('agents')}
              >
                Browse agents
              </button>
            </div>
          </div>
        </div>
      </>
    )
  }

  return (
    <>
      {cp && !cp.healthy ? (
        <div className="callout">AgentField server isn’t running.</div>
      ) : null}

      <div className="tile-grid">
        <Tile
          label="Agents running"
          value={off ? '—' : `${metrics.agentsRunning}`}
          context={off ? undefined : `${metrics.agentsRunning} of ${metrics.agentsTotal}`}
        />
        <Tile
          label="Executing now"
          value={off ? '—' : `${runningNow}`}
          context={off || runningNow === 0 ? undefined : 'in flight'}
        />
        <Tile label="Spend today" value={spendValue} context={spendContext} />
        {showTokensTile ? (
          <Tile
            label="Tokens today"
            value={formatTokens(usage!.totalTokens)}
            context="cost unavailable"
          />
        ) : (
          <Tile
            label="Success rate"
            value={
              off || metrics.successRate === null ? '—' : `${Math.round(metrics.successRate)}%`
            }
            tone={
              off || metrics.successRate === null ? undefined : successTone(metrics.successRate)
            }
            context={off ? undefined : 'last 24 hours'}
          />
        )}
      </div>

      {attentionChips.length > 0 ? (
        <div className="panel">
          <div className="attention-strip" aria-label="Needs attention">
            {attentionChips.map((chip) => (
              <button
                key={chip.key}
                type="button"
                className="attention-chip"
                onClick={() => onNavigate(chip.view)}
              >
                {chip.label}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <section>
        <div className="subhead">
          <h2 className="section-title">Recent activity</h2>
          <button type="button" className="link-button" onClick={() => onNavigate('activity')}>
            See all
          </button>
        </div>
        <div className="panel">
          {snapshot === null ? (
            // First load only — poll refreshes never flash skeletons (§4.15).
            <SkeletonRows count={2} />
          ) : !cpHealthy || executions === null || recentRows.length === 0 ? (
            <div className="empty secondary">
              Runs appear here when an agent is called — from Claude Code, Codex, or any client.
            </div>
          ) : (
            <ul className="row-list">
              {recentRows.map(({ run, live }) => (
                <ExecutionRow key={run.runId} run={run} live={live} />
              ))}
            </ul>
          )}
        </div>
      </section>

      <section>
        <h2 className="section-title">Connect clients</h2>
        <div className="panel">
          <div className="connect-card">
            <div className="connect-row">
              <span className="connect-label">Endpoint</span>
              <span className="connect-value mono" title={baseUrl}>
                {baseUrl}
              </span>
              <button type="button" className="link-button" onClick={() => void copyEndpoint()}>
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <div className="connect-row">
              <span className="connect-label">Skills</span>
              {installSkills === false ? (
                <button
                  type="button"
                  className="link-button"
                  onClick={() => onNavigate('settings')}
                >
                  Off — enable in Settings
                </button>
              ) : (
                <span className="connect-value">
                  {installSkills === null ? '…' : 'Installed for coding agents'}
                </span>
              )}
            </div>
            <div className="connect-row">
              <span className="connect-label">Try it</span>
              <span className="connect-value mono">af agent discover</span>
            </div>
          </div>
          {callersLine ? <div className="callers-line">{callersLine}</div> : null}
          <div className="connect-docs">
            <a
              className="link-button"
              href={COMMUNITY_LINKS.docs}
              target="_blank"
              rel="noreferrer"
            >
              How clients connect
            </a>
          </div>
        </div>
      </section>
    </>
  )
}

/** Alias once App migrates Dashboard → Home. */
export const HomeView = DashboardView
