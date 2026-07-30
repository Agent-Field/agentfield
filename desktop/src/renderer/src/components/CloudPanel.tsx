import { useEffect, useState } from 'react'
import type { AgentFieldApi, CloudTestResult, DesktopSettings } from '../../../shared/types'

type CloudDeployApi = AgentFieldApi & {
  cloudDeployRailway(): Promise<void>
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

  useEffect(() => {
    void window.agentfield.getSettings().then((next) => {
      setSettings(next)
      setServerUrl(next.cloud?.serverUrl ?? '')
      setApiKey(next.cloud?.apiKey ?? '')
    })
  }, [])

  const cloud = settings?.cloud
  const enabled = cloud?.enabled ?? false
  const canSubmit = serverUrl.trim() !== '' && apiKey.trim() !== ''

  const test = async () => {
    setTesting(true)
    setError(null)
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
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const disconnect = async () => {
    setSaving(true)
    setError(null)
    try {
      const next = await window.agentfield.setSettings({
        cloud: {
          enabled: false,
          serverUrl: serverUrl.trim(),
          apiKey: apiKey.trim()
        }
      })
      setSettings(next)
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

      <section className="settings-section">
        <div className="subhead">
          <h2 className="section-title">Status</h2>
        </div>
        <div className="panel">
          <ul className="row-list">
            <li className="row">
              <div className="row-main">
                <span className="row-title">
                  {enabled ? `Cloud: ${cloud?.serverUrl || serverUrl}` : 'Local control plane'}
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
        <div className="panel">
          <ul className="row-list">
            <li className="row">
              <div className="row-main">
                <label className="row-title" htmlFor="cloud-server-url">Server URL</label>
                <span className="row-sub">The public address of your AgentField control plane.</span>
              </div>
              <div className="row-side">
                <input
                  id="cloud-server-url"
                  className="env-input cloud-input"
                  placeholder="https://your-cp.up.railway.app"
                  value={serverUrl}
                  onChange={(event) => {
                    setServerUrl(event.target.value)
                    setResult(null)
                  }}
                />
              </div>
            </li>
            <li className="row">
              <div className="row-main">
                <label className="row-title" htmlFor="cloud-api-key">API key</label>
                <span className="row-sub">Stored on this computer and sent only to your server.</span>
              </div>
              <div className="row-side env-row-controls">
                <input
                  id="cloud-api-key"
                  className="env-input cloud-input"
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(event) => {
                    setApiKey(event.target.value)
                    setResult(null)
                  }}
                />
                <button className="action-button" type="button" onClick={() => setShowKey(!showKey)}>
                  {showKey ? 'Hide' : 'Show'}
                </button>
              </div>
            </li>
          </ul>
        </div>
        <div className="cloud-actions">
          <button className="action-button" disabled={!canSubmit || testing} onClick={() => void test()}>
            {testing ? 'Testing…' : 'Test connection'}
          </button>
          <button
            className="action-button primary"
            disabled={!canSubmit || saving}
            onClick={() => void saveCloud()}
          >
            {saving ? 'Saving…' : 'Save & switch to cloud'}
          </button>
        </div>
        {result && (
          <div className={`callout ${result.ok ? '' : 'error'}`}>
            <div>
              <div>{result.message}</div>
              <div className="cloud-checks">
                Reachable: {result.healthy ? 'Yes' : 'No'} · Auth: {result.authOk ? 'Passed' : 'Failed'} ·
                Install API: {result.installApi ? 'Available' : 'Unavailable'}
                {result.version ? ` · Version: ${result.version}` : ''}
              </div>
            </div>
          </div>
        )}
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
        <div className="panel">
          <div className="row">
            <div className="row-main">
              <span className="row-title">Host your own cloud control plane</span>
              <span className="row-sub">1. Deploy the AgentField control plane template on Railway.</span>
              <span className="row-sub">
                2. Copy its public URL and the AGENTFIELD_API_KEY value from the service variables.
              </span>
              <span className="row-sub">3. Paste both above, then Test and Save.</span>
            </div>
            <div className="row-side">
              <button
                className="action-button"
                type="button"
                onClick={() => void deployApi.cloudDeployRailway()}
              >
                Open Railway template
              </button>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
