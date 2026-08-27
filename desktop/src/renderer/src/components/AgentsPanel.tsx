import { useCallback, useEffect, useState } from 'react'
import type { ReactElement } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import type {
  AgentEnvReport,
  AgentFieldSnapshot,
  BundledPhase,
  BundledStatus,
  LocalControlPlaneRestartStatus,
  SnapshotAgent,
  InstallResult
} from '../../../shared/types'
import { EnvEditor } from './EnvEditor'
import { MenuPopover } from './MenuPopover'
import { SkeletonRows } from './Skeleton'
import { EmptyState } from './EmptyMark'

type AgentAction = 'start' | 'stop' | 'restart' | 'update' | 'pause' | 'resume' | 'uninstall'

interface AgentsPanelProps {
  registry: AgentFieldSnapshot['registry'] | null
  /**
   * Bundled nodes the app is provisioning for this launch (shared/bundled.ts).
   * They are not in the registry yet, so they ride above the installed rows as
   * read-only progress until the install lands and the real row replaces them.
   */
  bundled: BundledStatus[]
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
  update: 'Updating from the recorded source…',
  pause: 'Pausing automatic updates…',
  resume: 'Resuming automatic updates…',
  uninstall: 'Uninstalling…'
}

// First-launch provisioning reads as calm progress, never as a broken agent:
// the badge says what the app is doing, not that something is wrong.
const BUNDLED_LABEL: Record<BundledPhase, string> = {
  pending: 'Queued',
  installing: 'Installing…',
  installed: 'Installed',
  failed: 'Install failed'
}

// A failed bundled node is not marked provisioned, so the next launch tries
// again — say so instead of leaving a dead-looking row.
const BUNDLED_FAILED_TITLE = 'This node is retried automatically on the next launch.'

export function rosterKey(names: readonly string[]): string {
  return [...new Set(names)].sort().join('\0')
}

export function visibleBundledRows(
  bundled: BundledStatus[],
  registryNames: readonly string[]
): BundledStatus[] {
  const installed = new Set(registryNames)
  return bundled.filter((node) => !installed.has(node.name))
}

export function agentUpdateChip(agent: SnapshotAgent): string | null {
  if (agent.update?.status === 'failed') return 'Update failed'
  if (agent.update?.status === 'error') return 'Update check failed'
  if (agent.autoUpdate === false) return 'Paused'
  if (agent.update?.status === 'available') return 'Update available'
  // The pass found an update but the node was busy; it retries on its own.
  if (agent.update?.status === 'deferred') return 'Update waiting for the node to be idle'
  if (agent.update?.status === 'pinned') return 'Pinned'
  return null
}

export function agentUpdateChipTitle(agent: SnapshotAgent): string | undefined {
  const status = agent.update?.status
  return status === 'failed' || status === 'error' || status === 'deferred' ? agent.update?.message : undefined
}

export function agentManualUpdateActionVisible(_agent: SnapshotAgent): boolean {
  return true
}

export function agentAutoUpdateActionVisible(agent: SnapshotAgent): boolean {
  return agent.autoUpdate !== undefined
}

export function localControlPlaneRestartVisible(
  status: LocalControlPlaneRestartStatus | null
): boolean {
  return status?.status === 'restart_required'
}

export function activeExecutionsConfirmation(count: number, agent: string): string {
  return `${count} ${count === 1 ? 'run' : 'runs'} in progress on ${agent}. Updating will stop them. Update anyway?`
}

/** Retry exactly once with force after the user acknowledges active runs. */
export async function updateWithExecutionConfirmation(
  agent: string,
  request: (force: boolean) => Promise<InstallResult>,
  confirm: (message: string) => boolean
): Promise<InstallResult> {
  const initial = await request(false)
  if (initial.activeExecutions === undefined || initial.activeExecutions <= 0) return initial
  if (!confirm(activeExecutionsConfirmation(initial.activeExecutions, agent))) {
    return { ok: true, message: 'Update cancelled.' }
  }
  return request(true)
}

export function LocalControlPlaneRestartBanner({
  status
}: {
  status: LocalControlPlaneRestartStatus | null
}): ReactElement | null {
  if (!localControlPlaneRestartVisible(status)) return null
  return (
    <div className="callout warning" role="status">
      {status!.message}
    </div>
  )
}

export function AgentsPanel({ registry, bundled, onChanged }: AgentsPanelProps): ReactElement {
  return (
    <div className="panel">
      <AgentsBody registry={registry} bundled={bundled} onChanged={onChanged} />
    </div>
  )
}

function AgentsBody({ registry, bundled, onChanged }: AgentsPanelProps) {
  const [busy, setBusy] = useState<{ name: string; action: AgentAction } | null>(null)
  const [failure, setFailure] = useState<{ name: string; message: string } | null>(null)
  const [envReports, setEnvReports] = useState<Record<string, AgentEnvReport>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [confirmUninstall, setConfirmUninstall] = useState<string | null>(null)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const registryRosterKey = rosterKey(registry?.agents.map((agent) => agent.name) ?? [])
  const visibleBundled = visibleBundledRows(
    bundled,
    registry?.agents.map((agent) => agent.name) ?? []
  )

  // Env/secret statuses come from the af CLI + manifests — refreshed on
  // mount, when the registry roster changes, and after any action. The stable
  // set key avoids shelling out to `af secrets ls` on ordinary snapshot polls.
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
  useEffect(loadEnv, [loadEnv, registryRosterKey])

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
  // A launch that is still provisioning bundled nodes is not empty — it is
  // not finished — so the provisioning rows suppress this state.
  if ((!registry.exists || registry.agents.length === 0) && visibleBundled.length === 0) {
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
    if ((action === 'start' || action === 'restart') && report && !report.satisfied) {
      setExpanded(name)
      setFailure({ name, message: 'This agent needs keys before it can start — add them below.' })
      return
    }
    setBusy({ name, action })
    setFailure(null)
    setConfirmUninstall(null)
    setOpenMenu(null)
    try {
      const result = action === 'uninstall'
        ? await window.agentfield.uninstall(name)
        : action === 'update'
          ? await updateWithExecutionConfirmation(
              name,
              (force) => window.agentfield.update(name, force ? { force: true } : undefined),
              (message) => window.confirm(message)
            )
          : action === 'pause' || action === 'resume'
            ? await window.agentfield.setPackageAutoUpdate(name, action === 'resume')
            : await window.agentfield.agentAction(action, name)
      if (!result.ok) setFailure({ name, message: result.message })
    } catch (error) {
      setFailure({
        name,
        message: `${error instanceof Error ? error.message : String(error)} Try again after checking the control-plane connection.`
      })
    }
    setBusy(null)
    onChanged()
    loadEnv()
  }

  const onEnvChanged = () => {
    loadEnv()
    setFailure(null)
  }

  return (
    <ul className="row-list">
      {/* Bundled nodes first: on a first launch these two rows are the whole
          view, so the user watches the install stream instead of an empty
          panel. They leave the list once the registry carries the real row. */}
      <AnimatePresence initial={false}>
        {visibleBundled.map((node) => (
          <BundledRow key={node.name} node={node} />
        ))}
      </AnimatePresence>
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

/**
 * A bundled node mid-provisioning. Deliberately inert: there is nothing to
 * start, stop, or configure until the install finishes, and offering buttons
 * that cannot work would read as a broken agent.
 */
function BundledRow({ node }: { node: BundledStatus }) {
  const reducedMotion = useReducedMotion()
  const failed = node.phase === 'failed'

  return (
    <m.li
      className="row-item"
      // Row arrival (DESIGN.md §5.2): fade + 4px settle, exit reversed when
      // the installed row takes over.
      initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="row">
        <div className="row-main">
          <div className="env-row-head">
            <BundledBadge phase={node.phase} />
            <span className="row-title">{node.name}</span>
            {node.language ? <span className="chip lang">{node.language}</span> : null}
          </div>
          {node.description && <span className="row-sub">{node.description}</span>}
          {node.message && (
            <span
              className={`row-progress${failed ? ' error-text' : ''}`}
              aria-live="polite"
            >
              {node.message}
            </span>
          )}
        </div>
      </div>
    </m.li>
  )
}

function BundledBadge({ phase }: { phase: BundledPhase }) {
  return (
    <span
      className={`badge provisioning ${phase}`}
      title={phase === 'failed' ? BUNDLED_FAILED_TITLE : undefined}
    >
      <span className="badge-dot" aria-hidden="true" />
      {BUNDLED_LABEL[phase]}
    </span>
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
  const updateChip = agentUpdateChip(agent)

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
            {updateChip && (
              <span
                className={`chip ${agent.update?.status === 'available' || agent.update?.status === 'failed' ? 'warn' : ''}`}
                title={agentUpdateChipTitle(agent)}
              >
                {updateChip}
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
              <MenuPopover
                open={menuOpen}
                onToggle={onToggleMenu}
                disabled={rowBusy}
                ariaLabel="More actions"
              >
                {(running || agent.badge === 'unknown') && (
                  <button
                    className="menu-item"
                    role="menuitem"
                    onClick={() => onAction('restart')}
                  >
                    Restart
                  </button>
                )}
                {agentManualUpdateActionVisible(agent) && (
                  <button
                    className="menu-item"
                    role="menuitem"
                    onClick={() => onAction('update')}
                  >
                    Update from recorded source
                  </button>
                )}
                {agentAutoUpdateActionVisible(agent) && (
                  <button
                    className="menu-item"
                    role="menuitem"
                    onClick={() => onAction(agent.autoUpdate === false ? 'resume' : 'pause')}
                  >
                    {agent.autoUpdate === false
                      ? 'Resume automatic updates'
                      : 'Pause automatic updates'}
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
              </MenuPopover>
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
