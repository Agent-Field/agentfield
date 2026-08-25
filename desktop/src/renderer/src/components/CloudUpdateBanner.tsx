import { useEffect, useState } from 'react'
import type { CloudUpdateApplyResult, CloudUpdateStatus } from '../../../shared/types'

const SUCCESS_VISIBLE_MS = 10_000

export interface ApplyFeedback extends CloudUpdateApplyResult {
  shownAt: number
}

type ApplyFeedbackEvent =
  | { type: 'apply'; result: ApplyFeedback }
  | { type: 'dismiss' | 'status' }

/** A failure stays until the user dismisses it or the next status snapshot
 * replaces it. A success is shown for SUCCESS_VISIBLE_MS no matter what the
 * status does meanwhile: the follow-up check flips the status to "current"
 * within a second of a successful update, which is exactly when the
 * "Updated to vX. N agents restored." line must still be readable. */
export function cloudUpdateApplyFeedback(
  current: ApplyFeedback | null,
  event: ApplyFeedbackEvent
): ApplyFeedback | null {
  if (event.type === 'apply') return event.result
  if (event.type === 'status') return current?.ok ? current : null
  return null
}

export function cloudUpdateApplyResultVisible(
  _status: CloudUpdateStatus,
  result: ApplyFeedback | null,
  now = Date.now()
): boolean {
  if (!result) return false
  if (!result.ok) return true
  return now - result.shownAt < SUCCESS_VISIBLE_MS
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
  if (status.latest === null) return status.message
  const available = `Control plane v${status.latest} is available`
  if (status.status === 'legacy') {
    return `${available} — this one is too old to report its version`
  }
  return status.canApply ? available : `${available} — update your control plane image`
}

export function cloudUpdateBannerCopy(
  status: CloudUpdateStatus,
  bannerVisible: boolean,
  resultVisible: boolean,
  result: ApplyFeedback | null
): string {
  const parts: string[] = []
  if (bannerVisible) parts.push(cloudUpdateBannerText(status))
  if (resultVisible && result) parts.push(result.message)
  return parts.join(' · ')
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
    return window.agentfield.onCloudUpdateStatus((next) => {
      setStatus(next)
      setResult((current) => cloudUpdateApplyFeedback(current, { type: 'status' }))
    })
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
    setResult((current) => cloudUpdateApplyFeedback(current, { type: 'dismiss' }))
    const next = await window.agentfield.applyCloudUpdate()
    setResult((current) => cloudUpdateApplyFeedback(current, {
      type: 'apply',
      result: { ...next, shownAt: Date.now() }
    }))
  }

  const dismiss = async () => {
    setResult((current) => cloudUpdateApplyFeedback(current, { type: 'dismiss' }))
    if (status.latest) {
      await window.agentfield.dismissCloudUpdate(status.latest)
      setDismissedVersion(status.latest)
    }
  }

  return (
    <div className="update-banner" role="status">
      <span className="update-banner-text">
        {bannerVisible && cloudUpdateBannerText(status)}
        {resultVisible && result && (
          <span className={result.ok ? '' : 'error-text'}>
            {bannerVisible ? ' · ' : ''}{result.message}
          </span>
        )}
      </span>
      {bannerVisible && cloudUpdateBannerActionVisible(status) && (
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
        aria-label={resultVisible ? 'Dismiss cloud update message' : 'Hide this control plane version'}
        title={resultVisible ? 'Dismiss cloud update message' : 'Hide this control plane version'}
        onClick={() => void dismiss()}
      >
        ×
      </button>
    </div>
  )
}
