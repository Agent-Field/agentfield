import type {
  ControlPlaneVersion,
  LocalControlPlaneRestartStatus
} from '../shared/types'
import type { AutostartResult } from './autostart'
import { compareAppVersions } from './updates'

export interface LocalCpUpdateInput {
  managedBinaryReplaced: boolean
  platform: NodeJS.Platform
  cloudEnabled: boolean
  autostart: AutostartResult
  cliVersion: string | null
}

export interface LocalCpUpdateDeps {
  getVersion: () => Promise<ControlPlaneVersion | null>
  /** Optional only when the caller owns a safe stop-and-start mechanism. */
  restartControlPlane?: (port: number) => Promise<{ ok: boolean; message: string }>
  dateNow?: () => Date
  now?: () => number
  sleep?: (milliseconds: number) => Promise<void>
}

const RESTART_VERIFY_INTERVAL_MS = 1_000
const RESTART_VERIFY_TIMEOUT_MS = 30_000

/** Clear the boot-time warning once the polled local server reaches the CLI. */
export function reconcileLocalControlPlaneRestart(
  status: LocalControlPlaneRestartStatus | null,
  running: ControlPlaneVersion | null,
  cloudEnabled = false
): LocalControlPlaneRestartStatus | null {
  if (status?.status !== 'restart_required' || !status.targetVersion) return status
  if (cloudEnabled) return null
  const current = running?.version?.replace(/^v/, '')
  const target = status.targetVersion.replace(/^v/, '')
  if (
    current &&
    /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(current) &&
    compareAppVersions(current, target) >= 0
  ) {
    return null
  }
  return status
}

function record(
  deps: LocalCpUpdateDeps,
  values: Omit<LocalControlPlaneRestartStatus, 'at'>
): LocalControlPlaneRestartStatus {
  return { at: (deps.dateNow?.() ?? new Date()).toISOString(), ...values }
}

/** Restart only an older control plane that this boot adopted. macOS is owned
 * by af-tray install; newly started servers already use the replacement CLI. */
export async function restartAdoptedControlPlaneAfterCliSwap(
  input: LocalCpUpdateInput,
  deps: LocalCpUpdateDeps
): Promise<LocalControlPlaneRestartStatus> {
  if (!input.managedBinaryReplaced) {
    return record(deps, {
      ok: true,
      restarted: false,
      status: 'not_required',
      message: 'Managed CLI was unchanged.'
    })
  }
  if (input.platform === 'darwin') {
    return record(deps, {
      ok: true,
      restarted: false,
      status: 'not_required',
      message: 'macOS control-plane restart is managed by af-tray.'
    })
  }
  if (input.cloudEnabled || input.autostart.kind !== 'adopted') {
    return record(deps, {
      ok: true,
      restarted: false,
      status: 'not_required',
      message: 'No adopted local control plane needs the replacement CLI.'
    })
  }
  if (!input.cliVersion) {
    return record(deps, {
      ok: false,
      restarted: false,
      status: 'failed',
      message: 'The replacement CLI version is unknown; the running control plane was left unchanged.'
    })
  }

  let running: ControlPlaneVersion | null
  try {
    running = await deps.getVersion()
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    return record(deps, {
      ok: false,
      restarted: false,
      status: 'failed',
      message: `Could not check the adopted control-plane version: ${detail}. If it is still running the old build, stop the running "af server" process and start it again (af server).`
    })
  }
  if (
    running?.version &&
    /^v?\d+\.\d+\.\d+$/.test(running.version) &&
    compareAppVersions(running.version, input.cliVersion) >= 0
  ) {
    return record(deps, {
      ok: true,
      restarted: false,
      status: 'not_required',
      message: `Local control plane v${running.version.replace(/^v/, '')} already matches the managed CLI.`
    })
  }

  const restartRequired = () => record(deps, {
    ok: false,
    restarted: false,
    status: 'restart_required' as const,
    message: `AgentField CLI updated to v${input.cliVersion}. Restart the local control plane: stop the running "af server" process and start it again (af server), or restart this machine.`,
    targetVersion: input.cliVersion ?? undefined
  })
  if (!deps.restartControlPlane) return restartRequired()

  const before = running?.version ? running.version.replace(/^v/, '') : null
  const restart = await deps.restartControlPlane(input.autostart.port)
  if (!restart.ok) {
    return record(deps, {
      ok: false,
      restarted: false,
      status: 'failed',
      message: `AgentField CLI updated to v${input.cliVersion}, but the control plane could not restart: ${restart.message}.`
    })
  }
  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? (
    (milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds))
  )
  const deadline = now() + RESTART_VERIFY_TIMEOUT_MS
  for (;;) {
    try {
      const after = await deps.getVersion()
      const afterVersion = after?.version ? after.version.replace(/^v/, '') : null
      if (
        afterVersion &&
        (afterVersion === input.cliVersion || (before !== null && afterVersion !== before))
      ) {
        return record(deps, {
          ok: true,
          restarted: true,
          status: 'restarted',
          message: `Local control plane restarted on port ${input.autostart.port} after the CLI update.`
        })
      }
    } catch {
      // A restarted server is expected to be briefly unreachable.
    }
    if (now() >= deadline) break
    await sleep(RESTART_VERIFY_INTERVAL_MS)
  }
  return restartRequired()
}
