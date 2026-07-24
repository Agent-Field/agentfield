import { useState } from 'react'
import type { ReactElement } from 'react'
import type { AgentEnvReport, AgentEnvVar, EnvVarStatus } from '../../../shared/types'

interface EnvEditorProps {
  report: AgentEnvReport
  /** Called after a successful set/revoke so statuses refresh. */
  onChanged: () => void
}

const STATUS_LABEL: Record<EnvVarStatus, string> = {
  env: 'From environment',
  stored: 'Stored',
  default: 'Default',
  missing: 'Missing'
}

/**
 * Inline editor for one agent's declared environment variables / API keys.
 * Values are write-only: they go straight into the af CLI's encrypted secret
 * store and are never read back — only the resolution status is shown.
 */
export function EnvEditor({ report, onChanged }: EnvEditorProps): ReactElement {
  const required = report.vars.filter((v) => v.required && !v.group)
  const groups = new Map<string, AgentEnvVar[]>()
  for (const v of report.vars) {
    if (!v.group) continue
    const list = groups.get(v.group) ?? []
    list.push(v)
    groups.set(v.group, list)
  }
  const optional = report.vars.filter((v) => !v.required)

  return (
    <div className="env-editor">
      {report.error && <div className="callout error">{report.error}</div>}
      {required.length > 0 && (
        <EnvSection agent={report.agent} title="Required" vars={required} onChanged={onChanged} />
      )}
      {[...groups.entries()].map(([id, vars]) => (
        <EnvSection
          key={id}
          agent={report.agent}
          title={`One of — ${vars[0]?.groupDescription || id}`}
          vars={vars}
          onChanged={onChanged}
        />
      ))}
      {optional.length > 0 && (
        <EnvSection agent={report.agent} title="Optional" vars={optional} onChanged={onChanged} />
      )}
    </div>
  )
}

interface EnvSectionProps {
  agent: string
  title: string
  vars: AgentEnvVar[]
  onChanged: () => void
}

function EnvSection({ agent, title, vars, onChanged }: EnvSectionProps) {
  return (
    <div className="env-section">
      <div className="env-section-title">{title}</div>
      {vars.map((v) => (
        <EnvRow key={v.name} agent={agent} envVar={v} onChanged={onChanged} />
      ))}
    </div>
  )
}

function EnvRow({
  agent,
  envVar,
  onChanged
}: {
  agent: string
  envVar: AgentEnvVar
  onChanged: () => void
}) {
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)

  const save = async () => {
    setBusy(true)
    setError(null)
    const result = await window.agentfield.setAgentSecret(agent, envVar.name, draft)
    setBusy(false)
    if (!result.ok) {
      setError(result.message)
      return
    }
    setDraft('')
    onChanged()
  }

  const revoke = async () => {
    setBusy(true)
    setError(null)
    const result = await window.agentfield.revokeAgentSecret(agent, envVar.name)
    setBusy(false)
    setConfirming(false)
    if (!result.ok) {
      setError(result.message)
      return
    }
    onChanged()
  }

  const stored = envVar.storedScopes.length > 0
  // A stored global key is shared by every agent that names it — the
  // confirm step says so before a revoke silently breaks a sibling.
  const shared = envVar.storedScopes.includes('global')

  return (
    <div className="env-row">
      <div className="env-row-main">
        <div className="env-row-head">
          <span className="env-name">{envVar.name}</span>
          <span className={`chip ${envVar.status}`}>{STATUS_LABEL[envVar.status]}</span>
          {stored && envVar.scope === 'global' && <span className="env-scope">shared</span>}
        </div>
        {envVar.description && <span className="row-sub">{envVar.description}</span>}
        {confirming && !busy && (
          <span className="row-progress warn-text">
            {shared
              ? 'This key is shared — revoking removes it for every agent that uses it.'
              : 'Revoking removes this key for this agent only.'}
          </span>
        )}
        {error && <span className="row-progress error-text">{error}</span>}
      </div>
      <div className="env-row-controls">
        {confirming ? (
          <>
            <button
              className="action-button danger"
              disabled={busy}
              onClick={() => void revoke()}
            >
              {busy ? 'Revoking…' : shared ? 'Revoke for all agents' : 'Revoke'}
            </button>
            <button
              className="action-button"
              disabled={busy}
              onClick={() => setConfirming(false)}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <input
              className="env-input"
              type={envVar.secret ? 'password' : 'text'}
              placeholder={envVar.status === 'missing' ? 'enter value' : 'replace value'}
              value={draft}
              disabled={busy}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && draft.trim() !== '') void save()
              }}
            />
            <button
              className="action-button primary"
              disabled={busy || draft.trim() === ''}
              onClick={() => void save()}
            >
              {busy ? '…' : 'Set'}
            </button>
            {/* The slot is always rendered so the input column lines up
                across rows whether or not a value is stored. */}
            <button
              className={stored ? 'action-button' : 'action-button ghost-slot'}
              disabled={!stored || busy}
              onClick={() => setConfirming(true)}
            >
              Revoke
            </button>
          </>
        )}
      </div>
    </div>
  )
}
