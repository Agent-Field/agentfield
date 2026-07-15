import { useEffect, useState } from 'react'
import type { AppUpdateStatus } from '../../../shared/types'

/** Per-platform action label for the install button. An offered update always
 *  carries this platform's installer (check() filters CLI-only releases), so
 *  the label only distinguishes download progress and install mechanics. */
export function updateActionLabel(status: AppUpdateStatus, platform: string): string {
  if (status.downloading) {
    return status.progress !== null ? `Downloading… ${status.progress}%` : 'Downloading…'
  }
  return platform === 'darwin' ? 'Download update' : 'Install update'
}

/**
 * "Update available" strip across the top of the window, fed by the main
 * process's GitHub-releases check. Dismissing hides it for that version only
 * (persisted) — a newer release brings it back, and Settings keeps offering
 * the update either way. On Windows installing quits into the one-click
 * installer; on macOS it opens the downloaded DMG.
 */
export function UpdateBanner() {
  const [status, setStatus] = useState<AppUpdateStatus | null>(null)
  const [dismissed, setDismissed] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  // macOS: the DMG was opened; tell the user what to do with it.
  const [handedOff, setHandedOff] = useState(false)

  useEffect(() => {
    void Promise.all([window.agentfield.getAppUpdateStatus(), window.agentfield.getSettings()])
      .then(([st, settings]) => {
        setStatus(st)
        setDismissed(settings.dismissedUpdateVersion)
      })
      .finally(() => setLoaded(true))
    return window.agentfield.onAppUpdateStatus(setStatus)
  }, [])

  const update = status?.available
  if (!loaded || !status || !update || dismissed === update.version) return null

  const install = async () => {
    setHandedOff(false)
    const next = await window.agentfield.installAppUpdate()
    setStatus(next)
    if (window.agentfield.platform === 'darwin' && !next.error) setHandedOff(true)
  }

  const dismiss = () => {
    setDismissed(update.version)
    void window.agentfield.setSettings({ dismissedUpdateVersion: update.version })
  }

  return (
    <div className="update-banner" role="status">
      <span className="update-banner-text">
        Update available — AgentField {update.version}
        {handedOff && !status.error && (
          <span className="update-banner-note">
            {' '}
            · Installer opened — drag AgentField to Applications, then relaunch.
          </span>
        )}
        {status.error && <span className="error-text"> · {status.error}</span>}
      </span>
      <button
        className="action-button primary"
        disabled={status.downloading}
        onClick={() => void install()}
      >
        {updateActionLabel(status, window.agentfield.platform)}
      </button>
      <button
        className="update-banner-dismiss"
        aria-label="Hide until the next release"
        title="Hide until the next release"
        onClick={dismiss}
      >
        ×
      </button>
    </div>
  )
}
