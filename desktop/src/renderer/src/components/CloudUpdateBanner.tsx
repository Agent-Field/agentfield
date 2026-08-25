import { useEffect, useState } from 'react'
import type { CloudUpdateApplyResult, CloudUpdateStatus } from '../../../shared/types'

const SUCCESS_VISIBLE_MS = 10_000

interface ApplyFeedback extends CloudUpdateApplyResult {
  shownAt: number
}

export function cloudUpdateApplyResultVisible(
  status: CloudUpdateStatus,
  result: ApplyFeedback | null,
  now = Date.now()
): boolean {
  if (!result) return false
  if (!result.ok) return true
  return status.status !== 'current' && now - result.shownAt < SUCCESS_VISIBLE_MS
}

export function cloudUpdateBannerVisible(
  status: CloudUpdateStatus,
  dismissedVersion: string | null
): boolean {
  return (
    (status.status === 'available' || status.status === 'legacy') &&
    status.latest !== null &&
    status.latest !== dismissedVersion
  )
}

export function cloudUpdateBannerActionVisible(status: CloudUpdateStatus): boolean {
  return status.canApply
}

export function cloudUpdateBannerText(status: CloudUpdateStatus): string {
  const available = `Control plane v${status.latest} is available`
  if (status.status === 'legacy') {
    return `${available} — this one is too old to report its version`
  }
  return status.canApply ? available : `${available} — update your control plane image`
}

/** Cloud control-plane update strip. It owns a distinct status channel and a
 * per-cloud-version dismissal, independent of AgentField Desktop updates. */
export function CloudUpdateBanner() {
  const [status, setStatus] = useState<CloudUpdateStatus | null>(null)
  const [dismissedVersion, setDismissedVersion] = useState<string | null>(null)
  const [result, setResult] = useState<ApplyFeedback | null>(null)

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

  useEffect(() => {
    if (!result?.ok) return
    const remaining = Math.max(0, SUCCESS_VISIBLE_MS - (Date.now() - result.shownAt))
    const timer = window.setTimeout(() => setResult(null), remaining)
    return () => window.clearTimeout(timer)
  }, [result])

  const bannerVisible = status && cloudUpdateBannerVisible(status, dismissedVersion)
  const resultVisible = status && cloudUpdateApplyResultVisible(status, result)
  if (!status || (!bannerVisible && !resultVisible)) return null

  const apply = async () => {
    setResult(null)
    const next = await window.agentfield.applyCloudUpdate()
    setResult({ ...next, shownAt: Date.now() })
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
        {resultVisible && result && (
          <span className={result.ok ? '' : 'error-text'}> · {result.message}</span>
        )}
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
