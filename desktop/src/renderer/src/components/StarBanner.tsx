import { useEffect, useState } from 'react'
import type {
  AgentFieldSnapshot,
  AppUpdateStatus,
  DesktopSettings
} from '../../../shared/types'
import { COMMUNITY_LINKS } from './communityLinks'

/** Once any star-banner action fires this session, never re-show. */
let sessionConsumed = false

const SNOOZE_MS = 7 * 24 * 60 * 60 * 1000

function updateBannerWouldShow(
  status: AppUpdateStatus | null,
  dismissed: string | null
): boolean {
  const update = status?.available
  if (!status || !update) return false
  return dismissed !== update.version
}

function isSnoozed(until: string | null): boolean {
  if (!until) return false
  const t = Date.parse(until)
  return Number.isFinite(t) && t > Date.now()
}

/** True after value was delivered: ≥1 installed agent, or ≥5 succeeded runs. */
function milestoneReached(snapshot: AgentFieldSnapshot | null): boolean {
  if (!snapshot) return false
  if (snapshot.registry.agents.length >= 1) return true
  const running = snapshot.executions?.running ?? []
  const recent = snapshot.executions?.recent ?? []
  const succeeded = [...running, ...recent].filter((r) => r.status === 'succeeded')
  return succeeded.length >= 5
}

interface StarBannerProps {
  snapshot: AgentFieldSnapshot | null
}

/**
 * Quiet milestone ask for a GitHub star. Reuses the update-banner material;
 * never coexists with an undismissed app update (update wins).
 */
export function StarBanner({ snapshot }: StarBannerProps) {
  const [settings, setSettings] = useState<DesktopSettings | null>(null)
  const [updateStatus, setUpdateStatus] = useState<AppUpdateStatus | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [hidden, setHidden] = useState(sessionConsumed)

  useEffect(() => {
    void Promise.all([window.agentfield.getSettings(), window.agentfield.getAppUpdateStatus()])
      .then(([s, st]) => {
        setSettings(s)
        setUpdateStatus(st)
      })
      .finally(() => setLoaded(true))
    return window.agentfield.onAppUpdateStatus(setUpdateStatus)
  }, [])

  if (hidden || sessionConsumed || !loaded || !settings) return null
  if (settings.starPrompt === 'done') return null
  if (isSnoozed(settings.starPromptSnoozedUntil)) return null
  if (!snapshot?.controlPlane.healthy) return null
  if (updateBannerWouldShow(updateStatus, settings.dismissedUpdateVersion)) return null
  if (!milestoneReached(snapshot)) return null

  const consume = (patch: Partial<DesktopSettings>) => {
    sessionConsumed = true
    setHidden(true)
    setSettings((prev) => (prev ? { ...prev, ...patch } : prev))
    void window.agentfield.setSettings(patch)
  }

  return (
    <div className="update-banner community-banner" role="status">
      <span className="update-banner-text">
        ★ Enjoying AgentField? A star on GitHub helps other developers find it.
      </span>
      <a
        className="action-button primary"
        href={COMMUNITY_LINKS.repo}
        target="_blank"
        rel="noreferrer"
        onClick={() => consume({ starPrompt: 'done' })}
      >
        Star on GitHub
      </a>
      <button
        type="button"
        className="action-button"
        onClick={() =>
          consume({ starPromptSnoozedUntil: new Date(Date.now() + SNOOZE_MS).toISOString() })
        }
      >
        Later
      </button>
      <button
        type="button"
        className="action-button ghost"
        onClick={() => consume({ starPrompt: 'done' })}
      >
        Don&apos;t ask again
      </button>
    </div>
  )
}
