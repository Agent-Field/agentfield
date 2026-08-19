import { useCallback, useEffect, useState } from 'react'
import type { View } from '../../../shared/deeplink'
import type { AgentEnvReport, AgentFieldSnapshot } from '../../../shared/types'

/**
 * Installed agents whose required keys do not resolve.
 *
 * Reports carrying `error` are the control-plane-unreachable sentinel
 * (secrets.ts returns one nameless `satisfied: false` row for the whole
 * call) — a transport failure is not a missing key, so it never counts.
 * Agents the control plane cannot describe come back `satisfied: true` on
 * purpose (secrets.ts:246) so an old control plane cannot make the banner
 * cry wolf; that fallback needs no special case here.
 */
export function unsatisfiedAgents(reports: AgentEnvReport[]): string[] {
  return reports.filter((r) => r.agent && !r.error && !r.satisfied).map((r) => r.agent)
}

/** Banner copy, or null when nothing is blocked. One agent gets named. */
export function keysBannerMessage(reports: AgentEnvReport[]): string | null {
  const names = unsatisfiedAgents(reports)
  if (names.length === 0) return null
  if (names.length === 1) {
    return `${names[0]} is installed but needs API keys before it can run.`
  }
  return `${names.length} installed agents need API keys before they can run.`
}

interface KeysBannerProps {
  snapshot: AgentFieldSnapshot | null
  /** Current view — leaving the Agents view is a "done editing keys" signal. */
  view: View
  onNavigate: (view: View) => void
  /**
   * Told whenever this banner starts or stops showing. Only App can order the
   * banner stack, and the star prompt must not sit under a "your agents cannot
   * run" strip — asking for a favour while the product is broken reads badly.
   * Reported upward rather than re-derived in StarBanner because getEnvReports
   * fans out to the control plane per package; one caller is enough.
   */
  onShowingChange?: (showing: boolean) => void
}

/**
 * "Needs API keys" strip across the top of the window. The bundled nodes
 * (desktop/README.md §Bundled agent nodes) are installed but never started,
 * because both want keys the user has not entered — without this banner the
 * first sign of trouble is a coding agent failing at the worst moment.
 *
 * Not dismissible, deliberately. The update banner advertises something
 * optional, so hiding it costs nothing; this one reports that installed
 * agents cannot run at all, and a dismissed copy would leave a first-run
 * user with no proactive signal whatsoever. It is also self-clearing —
 * it goes the moment the keys resolve — so it can never become permanent
 * furniture, and a user who does not want those agents can uninstall them,
 * which removes the banner along with the agents.
 */
export function KeysBanner({ snapshot, view, onNavigate, onShowingChange }: KeysBannerProps) {
  const [message, setMessage] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  // Env/secret statuses are not on the snapshot poll (each call fans out to
  // the control plane per package), so this rides events instead of a timer,
  // mirroring AgentsPanel's load-on-mount-then-on-change pattern. It reloads
  // when the agent roster changes (install, uninstall, or the Start that
  // follows a key being entered) and when the view changes (the banner's own
  // action lands the user in Agents; leaving it again means they are done).
  const roster = (snapshot?.registry.agents ?? []).map((a) => `${a.name}:${a.badge}`).join(',')
  const healthy = snapshot?.controlPlane.healthy ?? false

  const load = useCallback(() => {
    let cancelled = false
    window.agentfield
      .getEnvReports()
      .then((reports) => {
        if (cancelled) return
        setMessage(keysBannerMessage(reports))
        setLoaded(true)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    // No control plane means no truthful answer — App already shows the
    // "server is not running" callout, and stacking a second alarm on it
    // would only add noise.
    if (!healthy) return
    return load()
  }, [healthy, roster, view, load])

  const showing = loaded && healthy && message !== null

  useEffect(() => {
    onShowingChange?.(showing)
  }, [showing, onShowingChange])

  if (!showing) return null

  return (
    <div className="update-banner keys-banner" role="status">
      <span className="update-banner-text">{message}</span>
      <button
        type="button"
        className="action-button primary"
        onClick={() => onNavigate('agents')}
      >
        Add keys
      </button>
    </div>
  )
}
