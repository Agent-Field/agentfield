import type {
  CloudAutoUpdateMode,
  CloudUpdateApplyResult,
  CloudUpdateCheck,
  CloudUpdateStatus,
  ControlPlaneVersion
} from '../shared/types'
import { resolveCloudImage } from './deployEngine'
import { compareAppVersions } from './updates'

const APPLY_TIMEOUT_MS = 6 * 60_000
const POLL_INTERVAL_MS = 5_000

function cleanVersion(value: string): string {
  return value.trim().replace(/^v/, '')
}

function imageVersion(image: string | null): string | null {
  const match = /agentfield\/control-plane-cloud:v(\d+\.\d+\.\d+)$/.exec(image ?? '')
  return match?.[1] ?? null
}

function isSemver(value: string | null): value is string {
  return value !== null && /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$/.test(value)
}

function normalizedUrl(value: string | null | undefined): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    return `${url.origin}${url.pathname.replace(/\/+$/, '')}`
  } catch {
    return value.trim().replace(/\/+$/, '') || null
  }
}

function sameUrl(a: string | null | undefined, b: string | null | undefined): boolean {
  const left = normalizedUrl(a)
  return left !== null && left === normalizedUrl(b)
}

export interface CheckCloudUpdateOptions {
  running: ControlPlaneVersion | null
  tfstateImage: string | null
  fetchImpl?: typeof fetch
}

/** Compare the running version (not merely tfstate) with Docker Hub's stable
 * release. Lookup failures stay unknown so an outage never reads as current. */
export async function checkCloudUpdate(options: CheckCloudUpdateOptions): Promise<CloudUpdateCheck> {
  const current = options.running?.version
    ? cleanVersion(options.running.version)
    : imageVersion(options.tfstateImage)
  if (options.running && !isSemver(current)) {
    return {
      status: 'unknown',
      current,
      latest: null,
      message: `This control plane reports a development build (${current ?? 'unknown'}); automatic version checks are unavailable.`
    }
  }
  const latestImage = await resolveCloudImage(options.fetchImpl ?? fetch)
  const latest = imageVersion(latestImage)

  if (!options.running) {
    return {
      status: 'legacy',
      current,
      latest,
      message: 'This control plane is too old to report its running version. Update it to enable automatic version checks.'
    }
  }
  if (!latest) {
    return {
      status: 'unknown',
      current,
      latest: null,
      message: 'Could not check Docker Hub for the latest control plane release. Check your connection and try again.'
    }
  }
  if (current && compareAppVersions(latest, current) > 0) {
    return {
      status: 'available',
      current,
      latest,
      message: `Control plane v${latest} is available.`
    }
  }
  return {
    status: 'current',
    current,
    latest,
    message: `Control plane v${current ?? latest} is up to date.`
  }
}

export interface ApplyCloudUpdateOptions {
  running: ControlPlaneVersion | null
  tfstateImage: string | null
  tfstateServiceId?: string | null
  tfstateUrl?: string | null
  connectedServerUrl?: string | null
}

export type CloudUpdateApplyPath = 'tfstate' | 'railway' | 'none'

/** Select an apply path only when its deployment identity matches the
 * connected control plane. Legacy servers can be matched by their URL. */
export function cloudUpdateApplyPath(
  options: ApplyCloudUpdateOptions
): CloudUpdateApplyPath {
  const hosting = options.running?.hosting
  if (options.tfstateImage) {
    if (
      hosting?.service_id &&
      options.tfstateServiceId &&
      hosting.service_id === options.tfstateServiceId
    ) {
      return 'tfstate'
    }
    if (
      !hosting?.service_id &&
      sameUrl(options.connectedServerUrl, options.tfstateUrl)
    ) {
      return 'tfstate'
    }
  }
  if (
    hosting?.platform === 'railway' &&
    hosting.service_id &&
    hosting.environment_id
  ) {
    return 'railway'
  }
  return 'none'
}

export function cloudUpdateApplyUnavailableMessage(
  options: ApplyCloudUpdateOptions
): string {
  return options.tfstateImage
    ? 'The desktop deployment state belongs to a different control plane, and the connected control plane did not report a usable Railway service identity. Reconnect the matching Railway control plane, then try again.'
    : 'This control plane has no desktop deployment state or Railway service identity. Reconnect it from Remote, then try again.'
}

export interface ApplyCloudUpdateDeps {
  fetchImpl?: typeof fetch
  /** Refresh tfstate, then invoke deployEngine.runDeploy with this exact image. */
  refreshAndDeploy: (targetImage: string) => Promise<{ ok: boolean; message: string }>
  setServiceImage: (serviceId: string, environmentId: string, image: string) => Promise<void>
  redeploy: (serviceId: string, environmentId: string) => Promise<void>
  getVersion: () => Promise<ControlPlaneVersion | null>
  sleep?: (milliseconds: number) => Promise<void>
  now?: () => number
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Apply through desktop tfstate when it exists, otherwise through the
 * running Railway service identity. Never guesses at an unmanaged Docker CP. */
export async function applyCloudUpdate(
  options: ApplyCloudUpdateOptions,
  deps: ApplyCloudUpdateDeps
): Promise<CloudUpdateApplyResult> {
  const targetImage = await resolveCloudImage(deps.fetchImpl ?? fetch)
  const target = imageVersion(targetImage)
  if (!targetImage || !target) {
    return {
      ok: false,
      message: 'Could not resolve the latest stable control plane image from Docker Hub. Check your connection and try again.'
    }
  }

  const current = options.running?.version ? cleanVersion(options.running.version) : null
  if (current && compareAppVersions(target, current) < 0) {
    return {
      ok: false,
      message: `Refusing to downgrade the running control plane from v${current} to v${target}.`
    }
  }

  try {
    const path = cloudUpdateApplyPath(options)
    if (path === 'tfstate') {
      const result = await deps.refreshAndDeploy(targetImage)
      if (!result.ok) return { ok: false, message: result.message }
    } else if (path === 'railway') {
      const hosting = options.running?.hosting
      const serviceId = hosting!.service_id!
      const environmentId = hosting!.environment_id!
      await deps.setServiceImage(serviceId, environmentId, targetImage)
      await deps.redeploy(serviceId, environmentId)
    } else {
      return {
        ok: false,
        message: cloudUpdateApplyUnavailableMessage(options)
      }
    }
  } catch (error) {
    return {
      ok: false,
      message: `Could not start the cloud control-plane update: ${errorText(error)}. Check Railway and try again.`
    }
  }

  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? ((milliseconds: number) => new Promise((resolve) => setTimeout(resolve, milliseconds)))
  const deadline = now() + APPLY_TIMEOUT_MS
  for (;;) {
    try {
      const observed = await deps.getVersion()
      if (
        observed?.version && isSemver(cleanVersion(observed.version)) &&
        compareAppVersions(cleanVersion(observed.version), target) >= 0
      ) {
        return {
          ok: true,
          target,
          message: `Updated to v${target} — agents are being restored by the control plane.`
        }
      }
    } catch {
      // A deployment is expected to be briefly unreachable; keep polling.
    }
    if (now() >= deadline) break
    await sleep(POLL_INTERVAL_MS)
  }
  return {
    ok: false,
    target,
    message: `Railway accepted v${target}, but the control plane did not report that version within 6 minutes. Open Railway deployment logs, then check again.`
  }
}

export interface ApplyCloudUpdateWithRailwayTokenDeps {
  getAccessToken: () => Promise<string | null>
  createApplyDeps: (token: string) => ApplyCloudUpdateDeps
}

/** Resolve deployment identity before requesting Railway auth. Unmanaged
 * Docker/Kubernetes control planes receive their real blocker immediately. */
export async function applyCloudUpdateWithRailwayToken(
  options: ApplyCloudUpdateOptions,
  deps: ApplyCloudUpdateWithRailwayTokenDeps
): Promise<CloudUpdateApplyResult> {
  if (cloudUpdateApplyPath(options) === 'none') {
    return { ok: false, message: cloudUpdateApplyUnavailableMessage(options) }
  }
  const token = await deps.getAccessToken()
  if (!token) {
    return {
      ok: false,
      message: 'Sign in to Railway before updating the cloud control plane.'
    }
  }
  return applyCloudUpdate(options, deps.createApplyDeps(token))
}

export interface CloudAutoUpdateStateIdentity {
  serviceId: string | null
  environmentId: string | null
  url: string | null
}

export interface SetCloudAutoUpdateOptions {
  mode: CloudAutoUpdateMode
  connectedServerUrl: string
  tfstate: CloudAutoUpdateStateIdentity | null
}

export interface SetCloudAutoUpdateDeps {
  getAccessToken: () => Promise<string | null>
  getVersion: () => Promise<ControlPlaneVersion | null>
  setSchedule: (
    token: string,
    serviceId: string,
    environmentId: string,
    mode: CloudAutoUpdateMode
  ) => Promise<void>
}

export interface SetCloudAutoUpdateResult {
  ok: boolean
  message: string
  serviceId?: string
}

const AUTO_UPDATE_LABELS: Record<CloudAutoUpdateMode, string> = {
  off: 'Off',
  nightly: 'Nightly (02:00–06:00 UTC every day)',
  weekends: 'Weekends (Saturday and Sunday, all day UTC)',
  anytime: 'Anytime'
}

/** Apply a Railway schedule without leaking getVersion failures through IPC.
 * A transient version timeout falls back only to URL-matched tfstate. */
export async function setCloudAutoUpdateSchedule(
  options: SetCloudAutoUpdateOptions,
  deps: SetCloudAutoUpdateDeps
): Promise<SetCloudAutoUpdateResult> {
  try {
    const token = await deps.getAccessToken()
    if (!token) {
      return { ok: false, message: 'Sign in to Railway, then choose the schedule again.' }
    }

    let running: ControlPlaneVersion | null = null
    try {
      running = await deps.getVersion()
    } catch {
      // A transient CP timeout must not reject IPC. URL-matched tfstate below
      // remains a safe identity fallback for Desktop-managed deployments.
    }

    let serviceId: string | null = null
    let environmentId: string | null = null
    if (running?.hosting.platform === 'railway') {
      serviceId = running.hosting.service_id ?? null
      environmentId = running.hosting.environment_id ?? null
    } else if (
      !running?.hosting.service_id &&
      options.tfstate &&
      sameUrl(options.connectedServerUrl, options.tfstate.url)
    ) {
      serviceId = options.tfstate.serviceId
      environmentId = options.tfstate.environmentId
    }
    if (!serviceId || !environmentId) {
      return {
        ok: false,
        message: 'The connected control plane could not be matched to a Railway service. Reconnect the matching Railway control plane, then choose the schedule again.'
      }
    }

    await deps.setSchedule(token, serviceId, environmentId, options.mode)
    return {
      ok: true,
      serviceId,
      message: `Railway image auto-updates set to ${AUTO_UPDATE_LABELS[options.mode]}.`
    }
  } catch (error) {
    return {
      ok: false,
      message: `Railway could not save that schedule: ${errorText(error)}. Check your Railway access and try again.`
    }
  }
}

/** First deploys get the Nightly default. Reconciles preserve only a mode
 * already applied to this same service. */
export function autoUpdateModeAfterDeploy(input: {
  firstDeploy: boolean
  serviceId: string
  storedMode: CloudAutoUpdateMode | null
  storedServiceId: string | null
}): CloudAutoUpdateMode | null {
  if (input.firstDeploy) return 'nightly'
  return input.storedServiceId === input.serviceId ? input.storedMode : null
}

export interface CloudUpdateCheckerDeps {
  enabled: () => boolean
  getVersion: () => Promise<ControlPlaneVersion | null>
  getTfstateImage: () => string | null
  canApplyUpdate?: (running: ControlPlaneVersion | null) => boolean
  applyUpdate?: (
    running: ControlPlaneVersion | null,
    tfstateImage: string | null
  ) => Promise<CloudUpdateApplyResult>
  fetchImpl?: typeof fetch
  onStatus?: (status: CloudUpdateStatus) => void
  /** Main-owned settings reconciliation runs only after a completed check. */
  onCompletedCheck?: (running: ControlPlaneVersion | null) => void
}

export class CloudUpdateChecker {
  private readonly deps: CloudUpdateCheckerDeps
  private st: CloudUpdateStatus = {
    status: 'unknown',
    current: null,
    latest: null,
    message: 'Cloud update check has not run yet.',
    checking: false,
    applying: false,
    lastCheckedAt: null,
    canApply: false
  }
  private autoCheckStarted = false

  constructor(deps: CloudUpdateCheckerDeps) {
    this.deps = deps
  }

  status(): CloudUpdateStatus {
    return { ...this.st, hosting: this.st.hosting ? { ...this.st.hosting } : undefined }
  }

  private patch(patch: Partial<CloudUpdateStatus>): CloudUpdateStatus {
    this.st = { ...this.st, ...patch }
    this.deps.onStatus?.(this.status())
    return this.status()
  }

  async check(): Promise<CloudUpdateStatus> {
    if (this.st.checking || this.st.applying) return this.status()
    if (!this.deps.enabled()) {
      return this.patch({
        status: 'unknown',
        message: 'Cloud update checks run when a Remote control plane is enabled.'
      })
    }
    this.patch({ checking: true, hosting: undefined })
    try {
      const running = await this.deps.getVersion()
      const check = await checkCloudUpdate({
        running,
        tfstateImage: this.deps.getTfstateImage(),
        fetchImpl: this.deps.fetchImpl
      })
      const canApply = this.deps.canApplyUpdate?.(running) ?? false
      const message = check.status === 'legacy' && !canApply
        ? check.latest
          ? `This legacy control plane cannot be matched to this desktop deployment. In Railway, set its image to agentfield/control-plane-cloud:v${check.latest} and redeploy it.`
          : 'This legacy control plane cannot be updated automatically. In Railway, redeploy it with agentfield/control-plane-cloud:latest.'
        : check.message
      const status = this.patch({
        ...check,
        message,
        checking: false,
        lastCheckedAt: new Date().toISOString(),
        canApply,
        hosting: running?.hosting
      })
      this.deps.onCompletedCheck?.(running)
      return status
    } catch (error) {
      return this.patch({
        status: 'unknown',
        checking: false,
        canApply: false,
        hosting: undefined,
        message: `Could not read the running control-plane version: ${errorText(error)}. Check the Remote connection and try again.`
      })
    }
  }

  async apply(): Promise<CloudUpdateApplyResult> {
    if (this.st.applying) return { ok: false, message: 'A cloud update is already running.' }
    if (!this.deps.applyUpdate) {
      return { ok: false, message: 'Cloud update effects are not configured.' }
    }
    this.patch({ applying: true })
    try {
      const running = await this.deps.getVersion()
      const result = await this.deps.applyUpdate(running, this.deps.getTfstateImage())
      this.patch({ applying: false, message: result.message })
      if (result.ok) await this.check()
      return result
    } catch (error) {
      const result = {
        ok: false,
        message: `Cloud update failed: ${errorText(error)}. Check Railway and try again.`
      }
      this.patch({ applying: false, message: result.message })
      return result
    }
  }

  startAutoCheck(initialDelayMs = 15_000, intervalMs = 4 * 60 * 60_000): void {
    if (this.autoCheckStarted) return
    this.autoCheckStarted = true
    const tick = () => {
      if (this.deps.enabled()) void this.check()
    }
    setTimeout(tick, initialDelayMs)
    setInterval(tick, intervalMs)
  }
}
