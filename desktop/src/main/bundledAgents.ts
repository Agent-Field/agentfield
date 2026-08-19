// Provision the agent nodes that ship with the app (shared/bundled.ts) on
// first launch, so a fresh install already has swe-planner and pr-af in the
// Agents library instead of an empty view and a marketplace to shop in.
//
// Delivery is "fetch on first launch", not "baked into the installer": the app
// installs them through the same control-plane install API a user-initiated
// install uses, which is why nothing here knows about packaging. What it adds
// is the decision of when to press install, and a status row per node so the
// UI can show the two arriving before they exist on disk.
//
// Same two-part shape as aforge-companion.ts, for the same reason:
//   1. planBundledInstalls() — pure: given the registry, what we already
//      provisioned, the skip env var, the resolved CLI and whether the control
//      plane is up, decide which names to install.
//   2. ensureBundledAgents() — the effect, driven by injected deps so tests
//      never install anything.
//
// Best-effort by construction: every failure is captured into a `failed`
// status row, nothing throws, so a dead network can't break startup. A failed
// node is deliberately NOT marked provisioned, so the next launch retries it.
//
// Deliberately does NOT import from 'electron' so it stays unit-testable.

import { BUNDLED_NODES } from '../shared/bundled'
import type { BundledPhase, BundledStatus, InstallResult } from '../shared/types'

/** Everything the provisioning decision depends on, observed by the caller. */
export interface BundledPlanInput {
  /** Names in ~/.agentfield/installed.yaml right now. */
  installed: readonly string[]
  /** settings.provisionedBundled */
  provisioned: readonly string[]
  /** AGENTFIELD_SKIP_BUNDLED — '1' disables provisioning entirely. */
  skipEnv: string | undefined
  /** The resolved af command, or null when none is usable. */
  cliCommand: string | null
  /** The control plane answered as a recognized AgentField. */
  controlPlaneReachable: boolean
  /** Whether the active control plane is a configured cloud connection. */
  cloudActive: boolean
  /** The installed-agent registry was read successfully. */
  registryReadable: boolean
}

export interface BundledPlan {
  /** Bundled node names to install, in order. */
  install: string[]
  /** Already-installed bundled node names to record as provisioned. */
  adopt: string[]
  /** One line for the log explaining the decision. */
  reason: string
}

export function planBundledInstalls(input: BundledPlanInput): BundledPlan {
  if (input.skipEnv === '1') {
    return { install: [], adopt: [], reason: 'AGENTFIELD_SKIP_BUNDLED=1 — skipping bundled nodes' }
  }
  if (input.cliCommand === null || input.cliCommand.trim() === '') {
    return { install: [], adopt: [], reason: 'no usable af CLI — skipping bundled nodes' }
  }
  if (input.cloudActive) {
    return {
      install: [],
      adopt: [],
      reason: 'cloud control plane active — bundled nodes are provisioned on the local control plane only'
    }
  }
  // The install API lives on the control plane, so there is nothing to talk to
  // until it is up and recognized. Not an error: the next launch retries.
  if (!input.controlPlaneReachable) {
    return { install: [], adopt: [], reason: 'control plane unavailable — skipping bundled nodes' }
  }
  if (!input.registryReadable) {
    return {
      install: [],
      adopt: [],
      reason: 'could not read the installed-agent registry — skipping bundled nodes'
    }
  }

  // Two independent reasons to leave a node alone: it is already in the
  // registry (nothing to do), or we provisioned it once before and the user
  // has since removed it (their choice must stick across launches).
  const installed = new Set(input.installed)
  const provisioned = new Set(input.provisioned)
  const adopt = BUNDLED_NODES.map((entry) => entry.name).filter(
    (name) => installed.has(name) && !provisioned.has(name)
  )
  const install = BUNDLED_NODES.map((entry) => entry.name).filter(
    (name) => !installed.has(name) && !provisioned.has(name)
  )
  if (install.length === 0 && adopt.length === 0) {
    return { install: [], adopt: [], reason: 'bundled nodes already provisioned' }
  }
  const reasons: string[] = []
  if (adopt.length > 0) reasons.push(`adopting already-installed bundled nodes: ${adopt.join(', ')}`)
  if (install.length > 0) reasons.push(`provisioning bundled nodes: ${install.join(', ')}`)
  return { install, adopt, reason: reasons.join('; ') }
}

export interface BundledDeps {
  /** installer.installAgent — resolves, never rejects. */
  install: (name: string, onLine: (line: string) => void) => Promise<InstallResult>
  /** Persist one name into settings.provisionedBundled. */
  markProvisioned: (name: string) => Promise<void>
  /** Whether this control plane supports installing agent packages. */
  hasInstallApi?: () => Promise<boolean>
  /** Called after a node installs successfully, before the next one starts. */
  onInstalled?: (name: string) => Promise<void>
  log: (message: string) => void
}

// Live provisioning rows for the snapshot, module state because the run is a
// launch-time side effect with no owner object to hang it off — same place the
// once-per-launch latch in aforge-companion.ts lives.
//
// Lifetime rule: rows appear when a run starts and `installed` rows are dropped
// when the whole run finishes, so a node the registry now lists never keeps a
// phantom row. `failed` rows survive for the rest of the session — that is the
// only place the user is told the node did not arrive, and the retry does not
// happen until the next launch.
let statuses: BundledStatus[] = []
let running = false

/** Live provisioning state for the snapshot. Empty before/after a run. */
export function bundledStatuses(): BundledStatus[] {
  return statuses.map((status) => ({ ...status }))
}

/** Test hook: clear module state between cases. */
export function resetBundledState(): void {
  statuses = []
  running = false
}

function setPhase(name: string, phase: BundledPhase, message: string): void {
  const row = statuses.find((status) => status.name === name)
  if (!row) return
  row.phase = phase
  row.message = message
}

/** Run the plan sequentially. Resolves when done; never rejects. */
export async function ensureBundledAgents(
  input: BundledPlanInput,
  deps: BundledDeps
): Promise<void> {
  // Re-entrancy guard for the same reason the loop below is sequential: the
  // control-plane install API answers a concurrent install with 409.
  if (running) {
    deps.log('bundled: provisioning already in progress')
    return
  }
  try {
    const plan = planBundledInstalls(input)
    deps.log(`bundled: ${plan.reason}`)
    if (plan.install.length === 0 && plan.adopt.length === 0) return

    running = true
    for (const name of plan.adopt) {
      try {
        await deps.markProvisioned(name)
        deps.log(`bundled: adopted ${name} (already installed)`)
      } catch (err) {
        deps.log(`bundled: could not record ${name} as provisioned — ${String(err)}`)
      }
    }

    if (plan.install.length === 0) return

    if (deps.hasInstallApi) {
      let hasInstallApi = false
      try {
        hasInstallApi = await deps.hasInstallApi()
      } catch {
        // An unreadable capability endpoint is equivalent to no usable API.
      }
      if (!hasInstallApi) {
        deps.log(
          'bundled: control plane has no install API — skipping bundled nodes (update the control plane)'
        )
        return
      }
    }

    // Seed every planned row up front so the Agents view shows both nodes
    // immediately, rather than revealing the second one minutes later.
    statuses = plan.install.map((name) => {
      const entry = BUNDLED_NODES.find((candidate) => candidate.name === name)
      return {
        name,
        description: entry?.description ?? '',
        language: entry?.language,
        phase: 'pending' as BundledPhase,
        message: ''
      }
    })

    for (const name of plan.install) {
      setPhase(name, 'installing', '')
      let result: InstallResult
      try {
        result = await deps.install(name, (line) => setPhase(name, 'installing', line))
      } catch (err) {
        // installAgent is documented never to reject; treat a broken dep as a
        // failed install rather than letting it escape into app startup.
        result = { ok: false, message: String(err) }
      }

      if (!result.ok) {
        setPhase(name, 'failed', result.message)
        deps.log(`bundled: ${name} failed — ${result.message}`)
        continue
      }

      setPhase(name, 'installed', result.message)
      deps.log(`bundled: ${name} installed`)
      // Recorded only on success, so a failure retries on the next launch.
      try {
        await deps.markProvisioned(name)
      } catch (err) {
        deps.log(`bundled: could not record ${name} as provisioned — ${String(err)}`)
      }
      try {
        await deps.onInstalled?.(name)
      } catch (err) {
        deps.log(`bundled: post-install step for ${name} failed — ${String(err)}`)
      }
    }
  } catch (err) {
    deps.log(`bundled: provisioning aborted — ${String(err)}`)
  } finally {
    // Drop the rows the registry now covers; keep the failures visible.
    statuses = statuses.filter((status) => status.phase === 'failed')
    running = false
  }
}
