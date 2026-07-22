import { useCallback, useEffect, useState } from 'react'
import type { ReactElement } from 'react'
import type { SecretsListResult, StoredSecret } from '../../../shared/types'
import { EmptyState } from './EmptyMark'

/**
 * Top-level Secrets view (still reachable via deep link / legacy nav).
 * Prefer composing {@link SecretsSection} inside Settings as "All keys".
 */
export function SecretsPanel(): ReactElement {
  return <SecretsSection />
}

/**
 * Encrypted key store as an embeddable section — no assumption that this is
 * a top-level view. Settings wraps it with an "All keys" title.
 */
export function SecretsSection(): ReactElement {
  const [data, setData] = useState<SecretsListResult | null>(null)

  const load = useCallback(() => {
    window.agentfield
      .listSecrets()
      .then(setData)
      .catch(() => {})
  }, [])
  useEffect(load, [load])

  if (data === null) {
    return (
      <div className="panel">
        <div className="empty secondary">Loading…</div>
      </div>
    )
  }

  return (
    <>
      {data.error && <div className="callout error">{data.error}</div>}
      <div className="panel">
        {data.secrets.length === 0 ? (
          <EmptyState
            variant="orbit"
            title="No keys stored"
            description="Keys you set on an agent appear here. Values stay encrypted and are never shown again."
          />
        ) : (
          <ul className="row-list">
            {data.secrets.map((secret) => (
              <SecretRow
                key={`${secret.scope}${secret.key}`}
                secret={secret}
                onChanged={load}
              />
            ))}
          </ul>
        )}
      </div>
      <p className="footnote">
        Shared keys are stored once and read by every agent that declares the
        variable — revoking one removes it for all of them.
      </p>
    </>
  )
}

function usedByLabel(secret: StoredSecret): string {
  if (secret.usedBy.length === 0) return 'not declared by any installed agent'
  return `used by ${secret.usedBy.join(', ')}`
}

function SecretRow({ secret, onChanged }: { secret: StoredSecret; onChanged: () => void }) {
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const shared = secret.scope === 'global'

  const revoke = async () => {
    setBusy(true)
    setError(null)
    const result = await window.agentfield.revokeSecret(secret.key, secret.scope)
    setBusy(false)
    setConfirming(false)
    if (!result.ok) {
      setError(result.message)
      return
    }
    onChanged()
  }

  return (
    <li className="row-item">
      <div className="row">
        <div className="row-main">
          <div className="env-row-head">
            <span className="env-name">{secret.key}</span>
            <span className={`chip ${shared ? 'stored' : 'default'}`}>
              {shared ? 'Shared — all agents' : `Agent: ${secret.scope}`}
            </span>
          </div>
          <span className="row-sub">{usedByLabel(secret)}</span>
          {confirming && !busy && (
            <span className="row-progress warn-text">
              {shared
                ? 'This key is shared — revoking removes it for every agent that uses it.'
                : `Revoking removes this key for ${secret.scope} only.`}
            </span>
          )}
          {error && <span className="row-progress error-text">{error}</span>}
        </div>
        <div className="row-side">
          <span className="row-meta">••••••••</span>
          <div className="row-actions">
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
                  className="action-button ghost"
                  disabled={busy}
                  onClick={() => setConfirming(false)}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button className="action-button" onClick={() => setConfirming(true)}>
                Revoke
              </button>
            )}
          </div>
        </div>
      </div>
    </li>
  )
}
