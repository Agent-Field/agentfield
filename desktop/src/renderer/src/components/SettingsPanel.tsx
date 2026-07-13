import { useEffect, useState } from 'react'
import type { DesktopSettings, InstalledAgent } from '../../../shared/types'

interface SettingsPanelProps {
  agents: InstalledAgent[]
}

/**
 * The "set it and forget it" surface: launch at login, keep the control
 * plane up, and pick which agents come up with it — so everything is already
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

      <div className="panel">
        <ul className="row-list">
          <ToggleRow
            title="Open at login"
            sub="Launch AgentField when you sign in. It starts quietly in the tray."
            checked={settings.openAtLogin}
            onChange={(on) => update({ openAtLogin: on })}
          />
          <ToggleRow
            title="Start the control plane automatically"
            sub="When nothing is listening yet, launch `af server` on app start."
            checked={settings.autostartControlPlane}
            onChange={(on) => update({ autostartControlPlane: on })}
          />
        </ul>
      </div>

      <h2 className="section-title">Auto-start agents</h2>
      <div className="panel">
        {agents.length === 0 ? (
          <div className="empty secondary">
            Install an agent first — then pick which ones start with the app.
          </div>
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
    </>
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
