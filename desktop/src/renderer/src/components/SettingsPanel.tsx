import { useEffect, useState } from 'react'
import type {
  AppUpdateStatus,
  CliStatus,
  DesktopSettings,
  InstalledAgent,
  PackageMaintenanceStatus
} from '../../../shared/types'
import { COMMUNITY_LINKS } from './communityLinks'
import { SecretsSection } from './SecretsPanel'
import { updateActionLabel } from './UpdateBanner'
import { EmptyState } from './EmptyMark'

interface SettingsPanelProps {
  agents: InstalledAgent[]
}

/**
 * The "set it and forget it" surface: launch at login, keep the AgentField
 * server up, and pick which agents come up with it — so everything is already
 * answering by the time Claude (or anything else) queries it.
 */
export function SettingsPanel({ agents }: SettingsPanelProps) {
  const [settings, setSettings] = useState<DesktopSettings | null>(null)

  useEffect(() => {
    void window.agentfield.getSettings().then(setSettings)
  }, [])

  const update = (patch: Partial<DesktopSettings>) => {
    // Optimistic: flip the control immediately, reconcile with what main
    // actually persisted (it normalizes and applies login-item effects).
    setSettings((prev) => (prev ? { ...prev, ...patch } : prev))
    void window.agentfield.setSettings(patch).then(setSettings)
  }

  if (!settings) {
    return (
      <div className="panel">
        <div className="empty secondary">Loading…</div>
      </div>
    )
  }

  const toggleAgent = (name: string, on: boolean) => {
    const next = on
      ? [...settings.autostartAgents, name]
      : settings.autostartAgents.filter((n) => n !== name)
    update({ autostartAgents: next })
  }

  return (
    <>
      <p className="view-lede">
        Set everything up once — the app keeps your agents ready for whatever queries them.
      </p>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">General</h2>
        </div>
        <div className="panel">
          <ul className="row-list">
            <ToggleRow
              title="Open at login"
              sub="Launch AgentField when you sign in. It starts quietly in the tray."
              checked={settings.openAtLogin}
              onChange={(on) => update({ openAtLogin: on })}
            />
            <AppearanceRow
              value={settings.appearance}
              onChange={(appearance) => update({ appearance })}
            />
            <ToggleRow
              title="Start the AgentField server automatically"
              sub="When nothing is listening yet, launch the AgentField server on app start."
              checked={settings.autostartControlPlane}
              onChange={(on) => update({ autostartControlPlane: on })}
            />
            <PortRow
              value={settings.controlPlanePort}
              onCommit={(port) => update({ controlPlanePort: port })}
            />
            <LocalApiKeyRow
              value={settings.localApiKey}
              onCommit={(key) => update({ localApiKey: key })}
            />
            <ToggleRow
              title="Keep coding-agent skills installed"
              sub="Teach Claude Code, Codex, and friends how to use AgentField (via `af skill install`)."
              checked={settings.installSkills}
              onChange={(on) => update({ installSkills: on })}
            />
            {window.agentfield.platform === 'darwin' && (
              <ToggleRow
                title="Show the menu bar icon"
                sub="Install the AgentField menu-bar companion (af-tray) for at-a-glance status and quick controls."
                checked={settings.trayCompanion}
                onChange={(on) => update({ trayCompanion: on })}
              />
            )}
          </ul>
        </div>
      </section>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Agents on startup</h2>
        </div>
        <div className="panel">
          {agents.length === 0 ? (
            <EmptyState
              variant="orbit"
              title="No startup agents"
              description="Installed agents will appear here so you can choose which ones start with the app."
            />
          ) : (
            <ul className="row-list">
              {agents.map((agent) => (
                <ToggleRow
                  key={agent.name}
                  title={agent.name}
                  sub={agent.description}
                  checked={settings.autostartAgents.includes(agent.name)}
                  onChange={(on) => toggleAgent(agent.name, on)}
                />
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Updates</h2>
        </div>
        <div className="panel">
          <ul className="row-list">
            <AppUpdateRow />
            <CliRow />
            <PackageUpdatesRow agents={agents} />
          </ul>
        </div>
      </section>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">All keys</h2>
        </div>
        <SecretsSection />
      </section>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">About</h2>
        </div>
        <div className="panel">
          <ul className="row-list">
            <AboutRows />
          </ul>
        </div>
      </section>
    </>
  )
}

export function packageUpdateTimestamps(
  status: PackageMaintenanceStatus | null,
  checkedAt: string | null
): { lastCheck: string | null; lastMaintenance: string | null } {
  return {
    lastCheck: checkedAt,
    lastMaintenance: status?.last_run?.finished_at ?? null
  }
}

export function latestPackageUpdateCheckedAt(agents: InstalledAgent[]): string | null {
  let latest: string | null = null
  for (const agent of agents) {
    const checkedAt = agent.update?.checkedAt
    if (checkedAt && (latest === null || Date.parse(checkedAt) > Date.parse(latest))) {
      latest = checkedAt
    }
  }
  return latest
}

function PackageUpdatesRow({ agents }: { agents: InstalledAgent[] }) {
  const [status, setStatus] = useState<PackageMaintenanceStatus | null>(null)
  const snapshotCheckedAt = latestPackageUpdateCheckedAt(agents)
  const [checkedAt, setCheckedAt] = useState<string | null>(() => snapshotCheckedAt)
  const [busy, setBusy] = useState<'check' | 'run' | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const refresh = () => window.agentfield.getMaintenanceStatus().then(setStatus)

  useEffect(() => {
    void refresh().catch((error) => {
      setMessage(`${error instanceof Error ? error.message : String(error)} Update the control plane, then try again.`)
    })
  }, [])

  useEffect(() => {
    if (!snapshotCheckedAt) return
    setCheckedAt((current) =>
      !current || Date.parse(snapshotCheckedAt) > Date.parse(current)
        ? snapshotCheckedAt
        : current
    )
  }, [snapshotCheckedAt])

  const check = async () => {
    setBusy('check')
    setMessage(null)
    try {
      const result = await window.agentfield.checkPackageUpdates()
      setCheckedAt(result.checked_at)
      setMessage(`Checked ${result.packages.length} installed agent${result.packages.length === 1 ? '' : 's'}.`)
      await refresh()
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : String(error)} Check the server connection and try again.`)
    } finally {
      setBusy(null)
    }
  }

  const run = async () => {
    setBusy('run')
    setMessage(null)
    try {
      await window.agentfield.runPackageMaintenance()
      setMessage('Maintenance started. Eligible agents will update one at a time and return on their previous ports.')
      await refresh()
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : String(error)} Wait for any active maintenance pass, then try again.`)
    } finally {
      setBusy(null)
    }
  }

  const timestamps = packageUpdateTimestamps(status, checkedAt)
  const lastCheck = timestamps.lastCheck
    ? new Date(timestamps.lastCheck).toLocaleString()
    : 'not yet'
  const lastMaintenance = timestamps.lastMaintenance
    ? new Date(timestamps.lastMaintenance).toLocaleString()
    : 'not yet'

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">Installed agent updates</span>
        <span className="row-sub">
          Installed agents update automatically at startup and every 6 h
        </span>
        <span className="row-sub">Last check: {lastCheck}</span>
        <span className="row-sub">Last maintenance pass: {lastMaintenance}</span>
        {status && !status.enabled && (
          <span className="row-sub error-text">
            Automatic package updates are off: {status.reason || 'disabled by the control plane'}.
          </span>
        )}
        {message && <span className="row-sub">{message}</span>}
      </div>
      <div className="row-side">
        <button
          type="button"
          className="action-button"
          disabled={busy !== null}
          onClick={() => void check()}
        >
          {busy === 'check' ? 'Checking…' : 'Check now'}
        </button>
        <button
          type="button"
          className="action-button primary"
          disabled={busy !== null || status?.enabled === false}
          onClick={() => void run()}
        >
          {busy === 'run' ? 'Starting…' : 'Run maintenance now'}
        </button>
      </div>
    </li>
  )
}

/** Durable home for star / docs / issue links — never a nag. */
function AboutRows() {
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    void window.agentfield.getAppUpdateStatus().then((s) => setVersion(s.currentVersion))
  }, [])

  return (
    <>
      <li className="row">
        <div className="row-main">
          <span className="row-title">
            AgentField Desktop{version ? ` v${version}` : ''}
          </span>
          <span className="row-sub">Free & open source.</span>
        </div>
        <div className="row-side">
          <a
            className="action-button"
            href={COMMUNITY_LINKS.repo}
            target="_blank"
            rel="noreferrer"
          >
            Star on GitHub
          </a>
        </div>
      </li>
      <li className="row">
        <div className="row-main">
          <span className="row-title">Docs</span>
          <span className="row-sub">Guides for installing and authoring agent nodes.</span>
        </div>
        <div className="row-side">
          <a
            className="action-button"
            href={COMMUNITY_LINKS.docs}
            target="_blank"
            rel="noreferrer"
          >
            Open docs
          </a>
        </div>
      </li>
      <li className="row">
        <div className="row-main">
          <span className="row-title">Found a bug?</span>
        </div>
        <div className="row-side">
          <a
            className="action-button"
            href={COMMUNITY_LINKS.issues}
            target="_blank"
            rel="noreferrer"
          >
            Report an issue
          </a>
        </div>
      </li>
    </>
  )
}

/**
 * The app's own release channel: current version, an on-demand check, and
 * the install action — always offered here, even when the user dismissed
 * the banner for this version. Renders as one row of the Updates panel.
 */
function AppUpdateRow() {
  const [status, setStatus] = useState<AppUpdateStatus | null>(null)
  // macOS: the DMG was opened; tell the user what to do with it.
  const [handedOff, setHandedOff] = useState(false)

  useEffect(() => {
    void window.agentfield.getAppUpdateStatus().then(setStatus)
    return window.agentfield.onAppUpdateStatus(setStatus)
  }, [])

  if (!status) {
    return (
      <li className="row">
        <div className="row-main">
          <span className="row-sub">Checking for app updates…</span>
        </div>
      </li>
    )
  }

  const install = async () => {
    setHandedOff(false)
    const next = await window.agentfield.installAppUpdate()
    setStatus(next)
    if (window.agentfield.platform === 'darwin' && !next.error) setHandedOff(true)
  }

  const sub = status.available
    ? handedOff && !status.error
      ? `Installer for v${status.available.version} opened — drag AgentField to Applications, then relaunch.`
      : `Version ${status.available.version} is available.`
    : status.lastCheckedAt
      ? 'You are up to date.'
      : 'Updates come from the AgentField releases on GitHub.'

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">AgentField Desktop v{status.currentVersion}</span>
        <span className="row-sub">{sub}</span>
        {status.error && <span className="row-sub error-text">{status.error}</span>}
      </div>
      <div className="row-side">
        {status.available ? (
          <button
            className="action-button primary"
            disabled={status.downloading}
            onClick={() => void install()}
          >
            {updateActionLabel(status, window.agentfield.platform)}
          </button>
        ) : (
          <button
            className="action-button"
            disabled={status.checking}
            onClick={() => void window.agentfield.checkForAppUpdate().then(setStatus)}
          >
            {status.checking ? 'Checking…' : 'Check for updates'}
          </button>
        )}
      </div>
    </li>
  )
}

/**
 * Which af the app drives, and the one-click path to a good version.
 * Renders as one row of the Updates panel.
 */
function CliRow() {
  const [status, setStatus] = useState<CliStatus | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void window.agentfield.getCliStatus().then(setStatus)
  }, [])

  if (!status) {
    return (
      <li className="row">
        <div className="row-main">
          <span className="row-sub">Checking the AgentField CLI…</span>
        </div>
      </li>
    )
  }

  const runUpdate = async () => {
    setBusy(true)
    setStatus(await window.agentfield.updateCli())
    setBusy(false)
  }

  const SOURCE_LABEL: Record<string, string> = {
    managed: 'installed in ~/.agentfield',
    path: 'from your PATH',
    bundled: 'bundled with the app'
  }

  const versionLabel = status.version ? `v${status.version}` : status.command ? 'dev build' : '—'
  const updateAvailable =
    status.bundledAvailable &&
    status.bundledVersion !== null &&
    status.version !== null &&
    status.bundledVersion !== status.version

  let issue: string | null = null
  let buttonLabel: string | null = null
  if (!status.command) {
    issue = 'No AgentField CLI found.'
    if (status.bundledAvailable) buttonLabel = 'Install AgentField CLI'
  } else if (status.outdated) {
    issue = `Your installed AgentField (v${status.outdated.version}) is older than this app needs (v${status.minVersion}) — the app is using its bundled copy meanwhile.`
    buttonLabel = 'Update AgentField'
  } else if (updateAvailable) {
    buttonLabel = `Update to v${status.bundledVersion}`
  }

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">
          {/* Section header no longer names the CLI — the row must. */}
          AgentField CLI {versionLabel}
          {status.source && (
            <span className="row-meta"> · {SOURCE_LABEL[status.source] ?? status.source}</span>
          )}
        </span>
        {issue && <span className="row-sub error-text">{issue}</span>}
      </div>
      {buttonLabel && (
        <div className="row-side">
          <button
            className="action-button primary"
            disabled={busy}
            onClick={() => void runUpdate()}
          >
            {busy ? 'Updating…' : buttonLabel}
          </button>
        </div>
      )}
    </li>
  )
}

/**
 * Control-plane port choice. Empty = automatic (8080 when free, else the
 * next open port); a number pins the port exactly. Committed on blur/Enter —
 * per-keystroke persistence would save half-typed ports.
 */
function PortRow({
  value,
  onCommit
}: {
  value: number | null
  onCommit: (port: number | null) => void
}) {
  const [text, setText] = useState(value === null ? '' : String(value))

  // Reflect what main actually persisted (it normalizes hostile values).
  useEffect(() => {
    setText(value === null ? '' : String(value))
  }, [value])

  const commit = () => {
    const trimmed = text.trim()
    if (trimmed === '') {
      if (value !== null) onCommit(null)
      return
    }
    const port = Number(trimmed)
    if (Number.isInteger(port) && port >= 1 && port <= 65535) {
      if (port !== value) onCommit(port)
    } else {
      // Invalid input reverts to the last saved value rather than persisting.
      setText(value === null ? '' : String(value))
    }
  }

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">Control plane port</span>
        <span className="row-sub">
          Leave empty to choose automatically — 8080 when free, otherwise the next open port.
          Applies the next time the control plane starts.
        </span>
      </div>
      <div className="row-side">
        <input
          className="env-input port-input"
          type="text"
          inputMode="numeric"
          placeholder="auto"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          }}
        />
      </div>
    </li>
  )
}

/**
 * Credential for a local AgentField server that runs with authentication on
 * (AGENTFIELD_API_KEY). Empty for almost everyone — a server started without a
 * key trusts this app because it connects over loopback. Committed on
 * blur/Enter like the port, so half-typed keys are never persisted or sent.
 */
function LocalApiKeyRow({
  value,
  onCommit
}: {
  value: string
  onCommit: (key: string) => void
}) {
  const [text, setText] = useState(value)
  const [show, setShow] = useState(false)

  // Reflect what main actually persisted (it trims).
  useEffect(() => {
    setText(value)
  }, [value])

  const commit = () => {
    const trimmed = text.trim()
    if (trimmed !== value) onCommit(trimmed)
  }

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">Server API key</span>
        <span className="row-sub">
          Only needed when your AgentField server requires authentication. Stored on this
          computer and sent only to that server.
        </span>
      </div>
      <div className="row-side">
        <input
          className="env-input"
          type={show ? 'text' : 'password'}
          placeholder="none"
          autoComplete="off"
          spellCheck={false}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          }}
        />
        <button className="action-button" type="button" onClick={() => setShow(!show)}>
          {show ? 'Hide' : 'Show'}
        </button>
      </div>
    </li>
  )
}

/**
 * Follows the OS until the user flips the switch. Once overridden, the quiet
 * reset restores system behavior without forcing a third state into a switch.
 */
function AppearanceRow({
  value,
  onChange
}: {
  value: DesktopSettings['appearance']
  onChange: (appearance: DesktopSettings['appearance']) => void
}) {
  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  )

  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const changed = (event: MediaQueryListEvent) => setSystemDark(event.matches)
    setSystemDark(query.matches)
    query.addEventListener('change', changed)
    return () => query.removeEventListener('change', changed)
  }, [])

  const dark = value === 'dark' || (value === 'system' && systemDark)
  const detail =
    value === 'system'
      ? 'Follows your system appearance. Toggle to override it.'
      : `Using ${value} appearance instead of the system setting.`

  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">Dark mode</span>
        <span className="row-sub">{detail}</span>
      </div>
      <div className="row-side">
        {value !== 'system' && (
          <button className="link-button" type="button" onClick={() => onChange('system')}>
            Use system
          </button>
        )}
        <button
          type="button"
          role="switch"
          aria-label="Dark mode"
          aria-checked={dark}
          className={`switch ${dark ? 'on' : ''}`}
          onClick={() => onChange(dark ? 'light' : 'dark')}
        >
          <span className="switch-thumb" aria-hidden="true" />
        </button>
      </div>
    </li>
  )
}

function ToggleRow({
  title,
  sub,
  checked,
  onChange
}: {
  title: string
  sub?: string
  checked: boolean
  onChange: (on: boolean) => void
}) {
  return (
    <li className="row">
      <div className="row-main">
        <span className="row-title">{title}</span>
        {sub && <span className="row-sub">{sub}</span>}
      </div>
      <div className="row-side">
        <button
          role="switch"
          aria-checked={checked}
          className={`switch ${checked ? 'on' : ''}`}
          onClick={() => onChange(!checked)}
        >
          <span className="switch-thumb" aria-hidden="true" />
        </button>
      </div>
    </li>
  )
}
