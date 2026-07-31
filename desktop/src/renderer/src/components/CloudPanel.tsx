import { useEffect, useState } from 'react'
import type { AgentFieldApi, CloudTestResult, DesktopSettings } from '../../../shared/types'

type CloudDeployApi = AgentFieldApi & {
  cloudDeployRailway(): Promise<void>
}

type Confirmation = {
  mode: 'cloud' | 'local'
  host?: string
}

export function CloudPanel() {
  const [settings, setSettings] = useState<DesktopSettings | null>(null)
  const [serverUrl, setServerUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [testing, setTesting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState<CloudTestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null)

  useEffect(() => {
    void window.agentfield.getSettings().then((next) => {
      setSettings(next)
      setServerUrl(next.cloud?.serverUrl ?? '')
      setApiKey(next.cloud?.apiKey ?? '')
    })
  }, [])

  useEffect(() => {
    if (!confirmation) return
    const timeout = window.setTimeout(() => setConfirmation(null), 4000)
    return () => window.clearTimeout(timeout)
  }, [confirmation])

  const cloud = settings?.cloud
  const enabled = cloud?.enabled ?? false
  const canSubmit = serverUrl.trim() !== '' && apiKey.trim() !== ''
  const busy = testing || saving

  const test = async () => {
    setTesting(true)
    setError(null)
    setConfirmation(null)
    setResult(null)
    try {
      setResult(await window.agentfield.cloudTest(serverUrl.trim(), apiKey.trim()))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTesting(false)
    }
  }

  const saveCloud = async () => {
    if (!result?.ok) {
      const proceed = window.confirm(
        'The connection has not passed its test. Switch to this cloud control plane anyway?'
      )
      if (!proceed) return
    }
    setSaving(true)
    setError(null)
    setConfirmation(null)
    try {
      const next = await window.agentfield.setSettings({
        cloud: {
          enabled: true,
          serverUrl: serverUrl.trim(),
          apiKey: apiKey.trim()
        }
      })
      setSettings(next)
      setServerUrl(next.cloud?.serverUrl ?? serverUrl.trim())
      setApiKey(next.cloud?.apiKey ?? apiKey.trim())
      setConfirmation({
        mode: 'cloud',
        host: displayHost(next.cloud?.serverUrl ?? serverUrl.trim())
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const disconnect = async () => {
    setSaving(true)
    setError(null)
    setConfirmation(null)
    try {
      const next = await window.agentfield.setSettings({
        cloud: {
          enabled: false,
          serverUrl: serverUrl.trim(),
          apiKey: apiKey.trim()
        }
      })
      setSettings(next)
      setConfirmation({ mode: 'local' })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  // TODO(integration): fold into AgentFieldApi
  const deployApi = window.agentfield as CloudDeployApi

  if (!settings) {
    return (
      <div className="panel">
        <div className="empty secondary">Loading…</div>
      </div>
    )
  }

  return (
    <>
      <p className="view-lede">
        Connect AgentField Desktop to a control plane hosted in the cloud.
      </p>

      {error && <div className="callout error">{error}</div>}
      {confirmation && (
        <div className="callout success cloud-confirmation" role="status">
          {confirmation.mode === 'cloud'
            ? `✓ Now managing ${confirmation.host}`
            : '✓ Switched back to the local control plane'}
        </div>
      )}

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Status</h2>
        </div>
        <div className="panel">
          <ul className="row-list">
            <li className="row">
              <span
                className={`cloud-status-dot ${enabled ? 'connected' : ''}`}
                aria-hidden="true"
              />
              <div className="row-main">
                <span className="row-title cloud-status-title">
                  {enabled ? displayHost(cloud?.serverUrl || serverUrl) : 'Local control plane'}
                </span>
                {enabled && (
                  <span className="row-sub">
                    Local server management is disabled while this cloud connection is active.
                  </span>
                )}
              </div>
            </li>
          </ul>
        </div>
      </section>

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Connect</h2>
        </div>
        <div className="panel cloud-form">
          <div className="cloud-field">
            <label className="row-title" htmlFor="cloud-server-url">
              Server URL
            </label>
            <span className="row-sub">The public address of your AgentField control plane.</span>
            <div className="cloud-input-row">
              <input
                id="cloud-server-url"
                className="env-input cloud-input"
                placeholder="https://your-cp.up.railway.app"
                value={serverUrl}
                disabled={busy}
                onChange={(event) => {
                  setServerUrl(event.target.value)
                  setResult(null)
                }}
              />
            </div>
          </div>
          <div className="cloud-field">
            <label className="row-title" htmlFor="cloud-api-key">
              API key
            </label>
            <span className="row-sub">Stored on this computer and sent only to your server.</span>
            <div className="cloud-input-row cloud-key-row">
              <input
                id="cloud-api-key"
                className="env-input cloud-input"
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                disabled={busy}
                onChange={(event) => {
                  setApiKey(event.target.value)
                  setResult(null)
                }}
              />
              <button
                className="action-button cloud-key-toggle"
                type="button"
                disabled={busy}
                onClick={() => setShowKey(!showKey)}
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
        </div>
        <div className="cloud-actions">
          <button
            className={`action-button ${!result || !result.ok || !result.installApi ? 'primary' : ''}`}
            disabled={!canSubmit || busy}
            onClick={() => void test()}
          >
            {testing && <span className="cloud-spinner" aria-hidden="true" />}
            {testing ? 'Testing…' : 'Test connection'}
          </button>
          <button
            className={`action-button ${result?.ok && result.installApi ? 'primary' : ''}`}
            disabled={!canSubmit || busy}
            onClick={() => void saveCloud()}
          >
            {saving ? 'Saving…' : 'Save & switch to cloud'}
          </button>
        </div>
        {result && <CloudTestFeedback result={result} />}
      </section>

      {enabled && (
        <section className="settings-section">
          <div className="subhead">
            <h2 className="section-title">Disconnect</h2>
          </div>
          <div className="panel">
            <div className="row">
              <div className="row-main">
                <span className="row-title">Switch back to local</span>
                <span className="row-sub">Your cloud address and key stay saved for next time.</span>
              </div>
              <div className="row-side">
                <button className="action-button" disabled={saving} onClick={() => void disconnect()}>
                  Switch back to local
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Deploy on Railway</h2>
        </div>
        <div className="panel cloud-railway">
          <div className="cloud-railway-content">
            <span className="row-title">Host your own cloud control plane</span>
            <ol className="cloud-steps">
              <li>Deploy the AgentField control plane template on Railway.</li>
              <li>
                Copy its public URL and the AGENTFIELD_API_KEY value from the service variables.
              </li>
              <li>Paste both above, then Test and Save.</li>
            </ol>
            <button
              className="action-button"
              type="button"
              onClick={() => void deployApi.cloudDeployRailway()}
            >
              Open Railway template
            </button>
          </div>
        </div>
      </section>
    </>
  )
}

function CloudTestFeedback({ result }: { result: CloudTestResult }) {
  const success = result.ok && result.installApi
  const degraded = result.ok && !result.installApi
  const state = success ? 'success' : degraded ? 'warning' : 'error'
  const heading = success
    ? `✓ Connected${result.version ? ` — control plane v${result.version.replace(/^v/, '')}` : ''}`
    : degraded
      ? '⚠ Connected, but this control plane is too old for desktop agent management — update the AgentField server, then test again.'
      : result.message

  const checks: Array<{ label: string; state: 'passed' | 'warning' | 'failed' | 'pending' }> = [
    { label: 'Reachable', state: result.healthy ? 'passed' : 'failed' },
    {
      label: 'API key accepted',
      state: result.authOk ? 'passed' : result.healthy ? 'failed' : 'pending'
    },
    {
      label: 'Agent management available',
      state: result.installApi
        ? 'passed'
        : degraded
          ? 'warning'
          : result.authOk
            ? 'failed'
            : 'pending'
    }
  ]

  return (
    <div className={`callout ${state} cloud-result`} role={success ? 'status' : 'alert'}>
      <div className="cloud-result-heading">{heading}</div>
      <ul className="cloud-checks">
        {checks.map((check) => (
          <li key={check.label} className={check.state}>
            <span className="cloud-check-icon" aria-hidden="true">
              {check.state === 'passed'
                ? '✓'
                : check.state === 'warning'
                  ? '⚠'
                  : check.state === 'failed'
                    ? '✗'
                    : '—'}
            </span>
            <span>{check.label}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function displayHost(serverUrl: string) {
  try {
    return new URL(serverUrl).host
  } catch {
    return serverUrl
  }
}
