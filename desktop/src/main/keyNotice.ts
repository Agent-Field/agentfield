// Tell the user, once, when first-launch provisioning finished on agents that
// still cannot run for want of an API key.
//
// The app is designed to open at login, hidden, in the tray (applyLoginItem in
// index.ts). On such a launch bundledAgents.ts installs swe-planner and pr-af,
// deliberately does NOT start them — both need a key the user has not entered —
// and puts a "Needs keys" chip on their Agents rows. Nobody is looking at that
// window. Without a push the two nodes sit there unusable and the user finds
// out later, from a coding agent that could not call them. One native
// notification closes that loop: name the keys, name where to enter them.
//
// Same two-part shape as aforge-companion.ts and bundledAgents.ts:
//   1. planKeyNotice() — pure: given the names this run provisioned, the env
//      reports and what was already announced, decide whether to notify and
//      what the copy says.
//   2. notifyUnresolvedKeys() — the effect, driven by injected deps so tests
//      never construct an Electron Notification.
//
// Best-effort by construction: nothing here throws, every failure resolves to
// a "did not notify" plan, so a missing notification daemon can never delay or
// break startup.
//
// Deliberately does NOT import from 'electron' so it stays unit-testable — the
// Notification call is the caller's ten lines in index.ts, the same split
// tray-model.ts / tray.ts use.

import type { AgentEnvReport } from '../shared/types'

/** Keys named per agent before the copy elides the rest. */
const MAX_LABELS_PER_AGENT = 3
/** Agents named before the copy elides the rest. */
const MAX_AGENTS = 3

export interface KeyNoticeInput {
  /**
   * Bundled node names THIS launch's provisioning run installed. Deliberately
   * not "every provisioned node": the notice belongs to the provisioning
   * event, so a launch that installs nothing never re-raises it. The cost is
   * that a notice lost to a control-plane hiccup is not retried — acceptable,
   * because the Agents row's "Needs keys" chip is the standing affordance and
   * this is only the push that points at it.
   */
  provisioned: readonly string[]
  /** getEnvReports() — see the authority note on `satisfied` below. */
  reports: readonly AgentEnvReport[]
  /** settings.keyNoticeShown — agents already announced on an earlier launch. */
  alreadyNotified: readonly string[]
  /** Notification.isSupported() — false on a desktop with no notification daemon. */
  supported: boolean
}

export interface KeyNoticePlan {
  notify: boolean
  /**
   * The agents this notice speaks for. Recorded in settings on delivery, so
   * they are never announced again.
   */
  agents: string[]
  title: string
  body: string
  /** One line for the log explaining the decision. */
  reason: string
}

const SILENT: Omit<KeyNoticePlan, 'reason'> = { notify: false, agents: [], title: '', body: '' }

/**
 * Which of the just-provisioned names have not been announced yet. Exported so
 * the runner can skip the control-plane round trip when the answer is "none"
 * without duplicating the rule.
 */
export function keyNoticeCandidates(
  provisioned: readonly string[],
  alreadyNotified: readonly string[]
): string[] {
  const seen = new Set(alreadyNotified)
  // Dedupe as well as filter: the caller collects names from a per-install
  // callback, and settings.keyNoticeShown must never gain a duplicate.
  return [...new Set(provisioned)].filter((name) => name !== '' && !seen.has(name))
}

/**
 * The unresolved required keys of one agent, as user-facing labels.
 *
 * A `require_one_of` group yields ONE label listing its alternatives — e.g.
 * "ANTHROPIC_API_KEY or OPENROUTER_API_KEY" — and only when every alternative
 * is missing: telling someone who already stored an Anthropic key that they
 * need an OpenRouter key would be false.
 */
export function missingKeyLabels(report: AgentEnvReport): string[] {
  const resolvedGroups = new Set(
    report.vars
      .filter((variable) => variable.group && variable.status !== 'missing')
      .map((variable) => variable.group as string)
  )
  // Insertion-ordered so the copy follows the manifest's own order. Grouped
  // variables share a slot keyed by group id; ungrouped ones get a private key
  // that cannot collide with a group id.
  const slots = new Map<string, string[]>()
  for (const variable of report.vars) {
    if (!variable.required || variable.status !== 'missing') continue
    if (variable.group && resolvedGroups.has(variable.group)) continue
    const key = variable.group ? `group:${variable.group}` : `var:${variable.name}`
    const slot = slots.get(key)
    if (slot) slot.push(variable.name)
    else slots.set(key, [variable.name])
  }
  return [...slots.values()].map((names) => names.join(' or '))
}

/** "A", "A and B", "A, B and C" — with an elision past MAX_LABELS_PER_AGENT. */
function joinLabels(labels: readonly string[]): string {
  const shown = labels.slice(0, MAX_LABELS_PER_AGENT)
  const hidden = labels.length - shown.length
  const parts = hidden > 0 ? [...shown, `${hidden} more`] : shown
  if (parts.length === 1) return parts[0]
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

export function planKeyNotice(input: KeyNoticeInput): KeyNoticePlan {
  // Degrade silently rather than half-way: no daemon, no notice, and nothing
  // recorded, so a user who later gets one is still told.
  if (!input.supported) {
    return { ...SILENT, reason: 'native notifications unsupported — not notifying' }
  }

  const candidates = keyNoticeCandidates(input.provisioned, input.alreadyNotified)
  if (candidates.length === 0) {
    return { ...SILENT, reason: 'nothing newly provisioned to announce' }
  }

  const unresolved: { agent: string; labels: string[] }[] = []
  for (const agent of candidates) {
    const report = input.reports.find((candidate) => candidate.agent === agent)
    // No report at all, or the error-shaped report getEnvReports() returns when
    // the control plane is unreachable ({ agent: '', vars: [], satisfied: false }).
    // Silence beats guessing: we would be inventing a list of missing keys.
    if (!report || report.error || report.vars.length === 0) continue
    // `satisfied` is the ONLY authority here. It is composed from
    // GET /api/ui/v1/agents/:id/secrets?include=env, which reads the same
    // encrypted store `af run` decrypts — unlike `af doctor` or the package
    // .env file, which are store-blind and call correctly stored keys unset.
    // It is also deliberately `true` when the control plane is too old to
    // report `requirement` metadata (secrets.ts), so an old server can never
    // trigger a notice built on a guess.
    if (report.satisfied) continue
    const labels = missingKeyLabels(report)
    // Unsatisfied but nothing nameable: a shape we do not understand. The copy
    // has to say what is missing, so say nothing instead.
    if (labels.length === 0) continue
    unresolved.push({ agent, labels })
  }

  if (unresolved.length === 0) {
    return { ...SILENT, reason: 'newly provisioned agents have every required key' }
  }

  const shown = unresolved.slice(0, MAX_AGENTS)
  const hidden = unresolved.length - shown.length
  const overflow = hidden > 0 ? ` (and ${hidden} more agent${hidden === 1 ? '' : 's'})` : ''
  const detail = shown.map((row) => `${row.agent} needs ${joinLabels(row.labels)}`).join('; ')
  const oneKey = unresolved.length === 1 && unresolved[0].labels.length === 1
  const title =
    unresolved.length === 1
      ? `${unresolved[0].agent} needs ${oneKey ? 'a key' : 'keys'}`
      : `${unresolved.length} agents need keys`

  return {
    notify: true,
    // Every unresolved agent is recorded, including the ones the copy elided:
    // they were counted in the notice, and re-announcing them later would be
    // the nagging this module exists to avoid.
    agents: unresolved.map((row) => row.agent),
    title,
    body: `${detail}${overflow} — click to add ${oneKey ? 'it' : 'them'} in AgentField → Agents → Keys.`,
    reason: `missing keys for ${unresolved.map((row) => row.agent).join(', ')}`
  }
}

export interface KeyNoticeDeps {
  /** secrets.getEnvReports — resolves to an error-shaped report, never rejects. */
  reports: () => Promise<AgentEnvReport[]>
  /** Notification.isSupported() */
  supported: () => boolean
  /** Show the notification; clicking it must open the app on the Agents view. */
  show: (notice: { title: string; body: string }) => void
  /** Persist the announced names into settings.keyNoticeShown. */
  markNotified: (agents: readonly string[]) => Promise<void>
  log: (message: string) => void
}

/**
 * Announce unresolved keys for the nodes a provisioning run just installed.
 * Resolves to the plan it acted on (tests assert on it); never rejects.
 */
export async function notifyUnresolvedKeys(
  provisioned: readonly string[],
  alreadyNotified: readonly string[],
  deps: KeyNoticeDeps
): Promise<KeyNoticePlan> {
  try {
    // Two cheap gates before the control-plane round trip: an unsupported
    // platform and a run that provisioned nothing new are the common cases,
    // and neither is worth an HTTP call per launch.
    const supported = deps.supported()
    const skip =
      !supported || keyNoticeCandidates(provisioned, alreadyNotified).length === 0
    const reports = skip ? [] : await deps.reports()

    const plan = planKeyNotice({ provisioned, reports, alreadyNotified, supported })
    deps.log(`key notice: ${plan.reason}`)
    if (!plan.notify) return plan

    // Show first, record second: a settings write that fails must not swallow
    // a notification the user is already looking at.
    try {
      deps.show({ title: plan.title, body: plan.body })
    } catch (err) {
      deps.log(`key notice: could not show the notification — ${String(err)}`)
      return { ...plan, notify: false }
    }
    try {
      await deps.markNotified(plan.agents)
    } catch (err) {
      deps.log(`key notice: could not record the notice as shown — ${String(err)}`)
    }
    return plan
  } catch (err) {
    const reason = `key notice aborted — ${String(err)}`
    deps.log(`key notice: ${reason}`)
    return { ...SILENT, reason }
  }
}
