import { useEffect, useState } from 'react'
import type { CloudUpdateStatus } from '../../../shared/types'

export function cloudUpdateBannerVisible(
  status: CloudUpdateStatus,
  dismissedVersion: string | null
): boolean {
  return status.status === 'available' && status.latest !== null && status.latest !== dismissedVersion
}

export function cloudUpdateBannerActionVisible(status: CloudUpdateStatus): boolean {
  return status.canApply
}

export function cloudUpdateBannerText(status: CloudUpdateStatus): string {
  const available = `Control plane v${status.latest} is available`
  return status.canApply ? available : `${available} — update your control plane image`
}

/** Cloud control-plane update strip. It owns a distinct status channel and a
 * per-cloud-version dismissal, independent of AgentField Desktop updates. */
export function CloudUpdateBanner() {
  const [status, setStatus] = useState<CloudUpdateStatus | null>(null)
  const [dismissedVersion, setDismissedVersion] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)

  useEffect(() => {
    void Promise.all([
      window.agentfield.checkCloudUpdate(),
      window.agentfield.getSettings()
    ]).then(([nextStatus, nextSettings]) => {
      setStatus(nextStatus)
      setDismissedVersion(nextSettings.cloud.dismissedUpdateVersion)
    })
    return window.agentfield.onCloudUpdateStatus(setStatus)
  }, [])

  if (
    !status ||
    !cloudUpdateBannerVisible(status, dismissedVersion)
  ) return null

  const apply = async () => {
    setResult(null)
    const next = await window.agentfield.applyCloudUpdate()
    if (!next.ok) setResult(next.message)
  }

  const dismiss = async () => {
    if (!status.latest) return
    await window.agentfield.dismissCloudUpdate(status.latest)
    setDismissedVersion(status.latest)
  }

  return (
    <div className="update-banner" role="status">
      <span className="update-banner-text">
        {cloudUpdateBannerText(status)}
        {result && <span className="error-text"> · {result}</span>}
      </span>
      {cloudUpdateBannerActionVisible(status) && (
        <button
          type="button"
          className="action-button primary"
          disabled={status.applying}
          onClick={() => void apply()}
        >
          {status.applying ? 'Updating control plane…' : 'Update now'}
        </button>
      )}
      <button
        type="button"
        className="update-banner-dismiss"
        aria-label="Hide this control plane version"
        title="Hide this control plane version"
        onClick={() => void dismiss()}
      >
        ×
      </button>
    </div>
  )
}
