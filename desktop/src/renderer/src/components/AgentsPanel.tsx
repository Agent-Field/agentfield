import { useCallback, useEffect, useState } from 'react'
import type { ReactElement } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import type { AgentEnvReport, AgentFieldSnapshot, SnapshotAgent } from '../../../shared/types'
import { EnvEditor } from './EnvEditor'
import { SkeletonRows } from './Skeleton'
import { EmptyState } from './EmptyMark'

type AgentAction = 'start' | 'stop' | 'restart' | 'uninstall'

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

const UNKNOWN_TITLE =
  "Registry says running, but the control plane doesn’t see this node. Try Restart."

const BUSY_LABEL: Record<AgentAction, string> = {
  start: 'Starting…',
  stop: 'Stopping…',
  restart: 'Restarting…',
  uninstall: 'Uninstalling…'
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
  const [envReports, setEnvReports] = useState<Record<string, AgentEnvReport>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [confirmUninstall, setConfirmUninstall] = useState<string | null>(null)
  const [openMenu, setOpenMenu] = useState<string | null>(null)

  // Env/secret statuses come from the af CLI + manifests — refreshed on
  // mount and after any change, not on the snapshot poll (each refresh
  // shells out to `af secrets ls`).
  const loadEnv = useCallback(() => {
    window.agentfield
      .getEnvReports()
      .then((reports) => {
        const byAgent: Record<string, AgentEnvReport> = {}
        for (const report of reports) byAgent[report.agent] = report
        setEnvReports(byAgent)
      })
      .catch(() => {})
  }, [])
  useEffect(loadEnv, [loadEnv])

  useEffect(() => {
    if (openMenu === null) return
    const close = () => setOpenMenu(null)
    // Defer so the opening click doesn't immediately close the menu.
    const timer = window.setTimeout(() => {
      window.addEventListener('click', close)
    }, 0)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('click', close)
    }
  }, [openMenu])

  if (!registry) {
    // First load only — layout-matched skeletons, not "Loading…" (§4.15).
    return <SkeletonRows count={3} />
  }
  if (registry.error) {
    return <div className="callout error">{registry.error}</div>
  }
  // Rarely rendered: App shows the Agents add-mode (marketplace) whenever the
  // library is empty, so this only covers odd registry states mid-refresh.
  if (!registry.exists || registry.agents.length === 0) {
    return (
      <EmptyState
        variant="orbit"
        title="No agents installed"
        description="Install your first agent node from GitHub to make it available to coding agents on this machine."
      />
    )
  }

  const run = async (action: AgentAction, name: string) => {
    // Starting an agent with unresolved required keys is a guaranteed
    // "missing required environment variables" failure — open the editor
    // instead of letting it happen.
    const report = envReports[name]
    if (action !== 'stop' && action !== 'uninstall' && report && !report.satisfied) {
      setExpanded(name)
      setFailure({ name, message: 'This agent needs keys before it can start — add them below.' })
      return
    }
    setBusy({ name, action })
    setFailure(null)
    setConfirmUninstall(null)
    setOpenMenu(null)
    const result =
      action === 'uninstall'
        ? await window.agentfield.uninstall(name)
        : await window.agentfield.agentAction(action, name)
    setBusy(null)
    if (!result.ok) setFailure({ name, message: result.message })
    onChanged()
    loadEnv()
  }

  const onEnvChanged = () => {
    loadEnv()
    setFailure(null)
  }

  return (
    <ul className="row-list">
      {registry.agents.map((agent) => (
        <AgentRow
          key={agent.name}
          agent={agent}
          report={envReports[agent.name]}
          busy={busy?.name === agent.name ? busy.action : null}
          failure={failure?.name === agent.name ? failure.message : null}
          isExpanded={expanded === agent.name}
          confirmingUninstall={confirmUninstall === agent.name}
          menuOpen={openMenu === agent.name}
          onToggleKeys={() => setExpanded(expanded === agent.name ? null : agent.name)}
          onToggleMenu={() => setOpenMenu(openMenu === agent.name ? null : agent.name)}
          onConfirmUninstall={() => {
            setOpenMenu(null)
            setConfirmUninstall(agent.name)
          }}
          onCancelUninstall={() => setConfirmUninstall(null)}
          onAction={(action) => void run(action, agent.name)}
          onEnvChanged={onEnvChanged}
        />
      ))}
    </ul>
  )
}

function AgentRow({
  agent,
  report,
  busy,
  failure,
  isExpanded,
  confirmingUninstall,
  menuOpen,
  onToggleKeys,
  onToggleMenu,
  onConfirmUninstall,
  onCancelUninstall,
  onAction,
  onEnvChanged
}: {
  agent: SnapshotAgent
  report: AgentEnvReport | undefined
  busy: AgentAction | null
  failure: string | null
  isExpanded: boolean
  confirmingUninstall: boolean
  menuOpen: boolean
  onToggleKeys: () => void
  onToggleMenu: () => void
  onConfirmUninstall: () => void
  onCancelUninstall: () => void
  onAction: (action: AgentAction) => void
  onEnvChanged: () => void
}) {
  const reducedMotion = useReducedMotion()
  const running = agent.badge === 'running'
  const rowBusy = busy !== null

  const descParts = [
    agent.description || null,
    running && agent.port !== null ? `:${agent.port}` : null
  ].filter(Boolean)

  return (
    <li className="row-item">
      <div className="row">
        <div className="row-main">
          <div className="env-row-head">
            <StatusBadge badge={agent.badge} />
            <span className="row-title">{agent.name}</span>
            {report && !report.satisfied && (
              <span className="badge warn">
                <span className="badge-dot" aria-hidden="true" />
                Needs keys
              </span>
            )}
          </div>
          {descParts.length > 0 && (
            <span className="row-sub">{descParts.join(' · ')}</span>
          )}
          {rowBusy && busy && (
            <span className="row-progress">{BUSY_LABEL[busy]}</span>
          )}
          {failure && !rowBusy && (
            <span className="row-progress error-text">{failure}</span>
          )}
          {confirmingUninstall && !rowBusy && (
            <span className="row-progress warn-text">
              Stops the agent and removes its files, registry entry, and agent-scoped keys.
              Shared keys stay.
            </span>
          )}
        </div>
        <div className="row-side">
          {confirmingUninstall ? (
            <div className="row-actions">
              <button
                className="action-button danger"
                disabled={rowBusy}
                onClick={() => onAction('uninstall')}
              >
                {busy === 'uninstall' ? 'Uninstalling…' : 'Uninstall'}
              </button>
              <button
                className="action-button ghost"
                disabled={rowBusy}
                onClick={onCancelUninstall}
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="row-actions">
              {running ? (
                <button
                  className="action-button"
                  disabled={rowBusy}
                  onClick={() => onAction('stop')}
                >
                  {busy === 'stop' ? 'Stopping…' : 'Stop'}
                </button>
              ) : (
                <button
                  className="action-button primary"
                  disabled={rowBusy}
                  onClick={() => onAction('start')}
                >
                  {busy === 'start' ? 'Starting…' : 'Start'}
                </button>
              )}
              {report && (
                <button className="action-button" onClick={onToggleKeys}>
                  Keys
                </button>
              )}
              <div className="menu-anchor">
                <button
                  className="action-button icon"
                  aria-label="More actions"
                  aria-expanded={menuOpen}
                  disabled={rowBusy}
                  onClick={(event) => {
                    event.stopPropagation()
                    onToggleMenu()
                  }}
                >
                  ⋯
                </button>
                {menuOpen && (
                  <div className="menu-popover" role="menu">
                    {(running || agent.badge === 'unknown') && (
                      <button
                        className="menu-item"
                        role="menuitem"
                        onClick={() => onAction('restart')}
                      >
                        Restart
                      </button>
                    )}
                    <button
                      className="menu-item"
                      role="menuitem"
                      onClick={() => {
                        void window.agentfield.openWebUI('/ui/')
                      }}
                    >
                      Open in Web UI
                    </button>
                    <button
                      className="menu-item danger"
                      role="menuitem"
                      onClick={onConfirmUninstall}
                    >
                      Uninstall
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      {/* Keys expander (DESIGN.md §5.2): real height spring via motion,
          replacing the CSS max-height hack. Reduced motion → instant. */}
      <AnimatePresence initial={false}>
        {isExpanded && report && (
          <m.div
            key="env-editor"
            style={{ overflow: 'hidden' }}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={
              reducedMotion
                ? { duration: 0 }
                : { type: 'spring', stiffness: 500, damping: 40 }
            }
          >
            <EnvEditor report={report} onChanged={onEnvChanged} />
          </m.div>
        )}
      </AnimatePresence>
    </li>
  )
}

function StatusBadge({ badge }: { badge: SnapshotAgent['badge'] }) {
  const label = BADGE_LABEL[badge] ?? badge
  return (
    <span
      className={`badge ${badge}`}
      title={badge === 'unknown' ? UNKNOWN_TITLE : undefined}
    >
      <span className="badge-dot" aria-hidden="true" />
      {label}
    </span>
  )
}
