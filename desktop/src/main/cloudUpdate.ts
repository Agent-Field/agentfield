import type {
  CloudAutoUpdateLiveMode,
  CloudAutoUpdateMode,
  CloudAutoUpdatePolicy,
  CloudAutoUpdateStateResult,
  CloudUpdateApplyResult,
  CloudUpdateCheck,
  CloudUpdateStatus,
  ControlPlaneVersion,
  PackageMaintenanceStatus
} from '../shared/types'
import { CpApiError } from './cpClient'
import { resolveCloudImage } from './deployEngine'
import type {
  RailwayEnabledAutoUpdatePolicy,
  RailwayImageAutoUpdates
} from './railwayApi'
import { compareAppVersions } from './updates'
import { stripTrailingSlashes } from '../shared/trimSlashes'

const APPLY_TIMEOUT_MS = 6 * 60_000
const POLL_INTERVAL_MS = 5_000
const MAINTENANCE_TIMEOUT_MS = 2 * 60_000
const RESTORE_FALLBACK_MESSAGE = (target: string) =>
  `Updated to v${target} — agents are being restored by the control plane.`

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

/** `compareAppVersions` predates build metadata and deliberately exposes an
 * invalid dotted core as NaN. Keep that uncertainty explicit at update
 * boundaries so it can never become permission to redeploy. */
function compareComparableVersions(a: string, b: string): number | null {
  const comparison = compareAppVersions(a, b)
  return Number.isFinite(comparison) ? comparison : null
}

function normalizedUrl(value: string | null | undefined): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    return `${url.origin}${stripTrailingSlashes(url.pathname)}`
  } catch {
    return stripTrailingSlashes(value.trim()) || null
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
  const comparison = current ? compareComparableVersions(latest, current) : null
  if (current && comparison === null) {
    return {
      status: 'unknown',
      current,
      latest,
      message: `The running control-plane version v${current} cannot be compared safely with v${latest}; automatic updates are unavailable.`
    }
  }
  if (comparison !== null && comparison > 0) {
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
  tfstateEnvironmentId?: string | null
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

/** Railway-only controls need a concrete service/environment, not merely
 * stale local deployment state. Legacy CPs may use URL-matched tfstate. */
export function cloudUpdateRailwayControlsAvailable(
  options: ApplyCloudUpdateOptions
): boolean {
  const hosting = options.running?.hosting
  if (
    hosting?.platform === 'railway' &&
    hosting.service_id &&
    hosting.environment_id
  ) {
    return true
  }
  return (
    options.running === null &&
    Boolean(options.tfstateServiceId) &&
    Boolean(options.tfstateEnvironmentId) &&
    cloudUpdateApplyPath(options) === 'tfstate'
  )
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
  /** Optional only for old/test integrations; production always supplies it. */
  getMaintenanceStatus?: () => Promise<PackageMaintenanceStatus>
  sleep?: (milliseconds: number) => Promise<void>
  now?: () => number
}

function failedRestoreDetail(error: string): string {
  const match = /^restore\s+([^:]+):\s*(.+)$/.exec(error.trim())
  return match ? `${match[1]} (${match[2]})` : error.trim()
}

function isRestoreError(error: string): boolean {
  return error.startsWith('restore ')
}

/** Turn the boot pass contract into the concise post-update message. */
export function cloudUpdateMaintenanceMessage(
  target: string,
  maintenance: PackageMaintenanceStatus
): string {
  const restored = maintenance.last_run?.restored ?? []
  const errors = maintenance.last_run?.errors ?? []
  const restoreErrors = errors.filter(isRestoreError)
  const warningCount = errors.length - restoreErrors.length
  if (restoreErrors.length === 0) {
    const parts = [
      `Updated to v${target}.`,
      `${restored.length} ${restored.length === 1 ? 'agent' : 'agents'} restored.`
    ]
    if (warningCount > 0) {
      parts.push(`${warningCount} maintenance ${warningCount === 1 ? 'warning' : 'warnings'}.`)
    }
    return parts.join(' ')
  }
  const parts = [`Updated to v${target}.`]
  if (restored.length > 0) parts.push(`Restored: ${restored.join(', ')}.`)
  parts.push(`Failed to restore: ${restoreErrors.map(failedRestoreDetail).join(', ')}.`)
  if (warningCount > 0) {
    parts.push(`${warningCount} maintenance ${warningCount === 1 ? 'warning' : 'warnings'}.`)
  }
  return parts.join(' ')
}

async function waitForBootMaintenance(
  target: string,
  deps: ApplyCloudUpdateDeps,
  now: () => number,
  sleep: (milliseconds: number) => Promise<void>
): Promise<string> {
  if (!deps.getMaintenanceStatus) return RESTORE_FALLBACK_MESSAGE(target)
  const deadline = now() + MAINTENANCE_TIMEOUT_MS
  for (;;) {
    try {
      const maintenance = await deps.getMaintenanceStatus()
      // boot_restore_completed flips as soon as the restore loop ends, but on a
      // fresh container last_run stays empty until the whole pass finishes —
      // wait for a summary to report on rather than announcing "0 restored".
      if (
        maintenance.boot_pass_completed === true ||
        (maintenance.boot_restore_completed === true && maintenance.last_run != null)
      ) {
        return cloudUpdateMaintenanceMessage(target, maintenance)
      }
    } catch (error) {
      if (
        (error instanceof CpApiError && error.status === 404) ||
        (
          typeof error === 'object' &&
          error !== null &&
          (error as { status?: unknown }).status === 404
        )
      ) {
        return RESTORE_FALLBACK_MESSAGE(target)
      }
      // The new container may still be settling; keep polling until bounded.
    }
    if (now() >= deadline) return RESTORE_FALLBACK_MESSAGE(target)
    await sleep(POLL_INTERVAL_MS)
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

interface ResolvedCloudUpdateTarget {
  image: string
  version: string
}

async function resolveCloudUpdateTarget(
  fetchImpl: typeof fetch
): Promise<ResolvedCloudUpdateTarget | CloudUpdateApplyResult> {
  const image = await resolveCloudImage(fetchImpl)
  const version = imageVersion(image)
  if (!image || !version) {
    return {
      ok: false,
      message: 'Could not resolve the latest stable control plane image from Docker Hub. Check your connection and try again.'
    }
  }
  return { image, version }
}

function cloudUpdatePreflight(
  options: ApplyCloudUpdateOptions,
  target: string
): CloudUpdateApplyResult | null {
  const current = options.running?.version ? cleanVersion(options.running.version) : null
  if (!current) return null
  const comparison = compareComparableVersions(target, current)
  if (comparison === null) {
    return {
      ok: false,
      target,
      message: `Cannot update from running control plane v${current} to v${target} because those versions cannot be compared safely.`
    }
  }
  if (comparison === 0) {
    return {
      ok: true,
      target,
      alreadyCurrent: true,
      message: `Control plane is already running v${target}.`
    }
  }
  if (comparison < 0) {
    return {
      ok: false,
      message: `Refusing to downgrade the running control plane from v${current} to v${target}.`
    }
  }
  return null
}

async function applyResolvedCloudUpdate(
  options: ApplyCloudUpdateOptions,
  deps: ApplyCloudUpdateDeps,
  resolved: ResolvedCloudUpdateTarget
): Promise<CloudUpdateApplyResult> {
  const targetImage = resolved.image
  const target = resolved.version
  const preflight = cloudUpdatePreflight(options, target)
  if (preflight) return preflight
  return performResolvedCloudUpdate(options, deps, targetImage, target)
}

/** Apply through desktop tfstate when it exists, otherwise through the
 * running Railway service identity. Never guesses at an unmanaged Docker CP. */
export async function applyCloudUpdate(
  options: ApplyCloudUpdateOptions,
  deps: ApplyCloudUpdateDeps
): Promise<CloudUpdateApplyResult> {
  const resolved = await resolveCloudUpdateTarget(deps.fetchImpl ?? fetch)
  if ('ok' in resolved) return resolved
  return applyResolvedCloudUpdate(options, deps, resolved)
}

async function performResolvedCloudUpdate(
  options: ApplyCloudUpdateOptions,
  deps: ApplyCloudUpdateDeps,
  targetImage: string,
  target: string
): Promise<CloudUpdateApplyResult> {
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
  const previousDeploymentId = options.running?.hosting?.deployment_id ?? undefined
  for (;;) {
    await sleep(POLL_INTERVAL_MS)
    try {
      const observed = await deps.getVersion()
      const observedComparison = observed?.version
        ? compareComparableVersions(cleanVersion(observed.version), target)
        : null
      if (
        observed?.version && isSemver(cleanVersion(observed.version)) &&
        observedComparison !== null && observedComparison >= 0 &&
        (
          previousDeploymentId === undefined ||
          observed.hosting.deployment_id !== previousDeploymentId
        )
      ) {
        return {
          ok: true,
          target,
          message: await waitForBootMaintenance(target, deps, now, sleep)
        }
      }
    } catch {
      // A deployment is expected to be briefly unreachable; keep polling.
    }
    if (now() >= deadline) break
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
  fetchImpl?: typeof fetch
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
  const resolved = await resolveCloudUpdateTarget(deps.fetchImpl ?? fetch)
  if ('ok' in resolved) return resolved
  const preflight = cloudUpdatePreflight(options, resolved.version)
  if (preflight) return preflight
  const token = await deps.getAccessToken()
  if (!token) {
    return {
      ok: false,
      message: 'Sign in to Railway before updating the cloud control plane.'
    }
  }
  return applyResolvedCloudUpdate(options, deps.createApplyDeps(token), resolved)
}

export interface CloudAutoUpdateStateIdentity {
  projectId: string | null
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
  getAutoUpdates: (
    token: string,
    environmentId: string,
    serviceId: string
  ) => Promise<RailwayImageAutoUpdates | null>
  setImageAutoUpdates: (
    token: string,
    environmentId: string,
    serviceId: string,
    mode: CloudAutoUpdateMode,
    policy: RailwayEnabledAutoUpdatePolicy
  ) => Promise<void>
}

export interface SetCloudAutoUpdateResult {
  ok: boolean
  message: string
  serviceId?: string
  settingsUrl?: string
}

/** Main persists a preference only after Railway accepted it and returned the
 * concrete service identity. */
export function cloudAutoUpdatePreferenceAfterSet(
  result: SetCloudAutoUpdateResult,
  mode: CloudAutoUpdateMode
): { mode: CloudAutoUpdateMode; serviceId: string } | null {
  return result.ok && result.serviceId ? { mode, serviceId: result.serviceId } : null
}

const AUTO_UPDATE_LABELS: Record<CloudAutoUpdateMode, string> = {
  off: 'Off',
  nightly: 'Nightly (02:00–06:00 UTC every day)',
  weekends: 'Weekends (Saturday and Sunday, all day UTC)',
  anytime: 'Anytime'
}

interface ResolvedCloudAutoUpdateIdentity {
  projectId: string | null
  serviceId: string
  environmentId: string
}

async function resolveCloudAutoUpdateIdentity(
  options: Pick<SetCloudAutoUpdateOptions, 'connectedServerUrl' | 'tfstate'>,
  getVersion: () => Promise<ControlPlaneVersion | null>
): Promise<ResolvedCloudAutoUpdateIdentity | null> {
  let running: ControlPlaneVersion | null = null
  try {
    running = await getVersion()
  } catch {
    // A transient CP timeout may safely fall back to URL-matched tfstate.
  }

  if (running?.hosting.platform === 'railway') {
    if (running.hosting.service_id && running.hosting.environment_id) {
      const matchingState = options.tfstate?.serviceId === running.hosting.service_id
        ? options.tfstate
        : null
      return {
        projectId: running.hosting.project_id ?? matchingState?.projectId ?? null,
        serviceId: running.hosting.service_id,
        environmentId: running.hosting.environment_id
      }
    }
    return null
  }

  if (
    !running?.hosting.service_id &&
    options.tfstate?.serviceId &&
    options.tfstate.environmentId &&
    sameUrl(options.connectedServerUrl, options.tfstate.url)
  ) {
    return {
      projectId: options.tfstate.projectId,
      serviceId: options.tfstate.serviceId,
      environmentId: options.tfstate.environmentId
    }
  }
  return null
}

export function railwaySettingsUrl(
  projectId: string | null | undefined,
  serviceId: string | null | undefined
): string {
  if (projectId && serviceId) {
    return `https://railway.com/project/${encodeURIComponent(projectId)}/service/${encodeURIComponent(serviceId)}/settings`
  }
  return 'https://railway.com/dashboard'
}

function matchesSchedule(
  schedule: RailwayImageAutoUpdates['schedule'],
  days: number[],
  startHour: number,
  endHour: number
): boolean {
  if (!schedule || schedule.length !== days.length) return false
  const expectedDays = new Set(days)
  const actualDays = new Set(schedule.map((window) => window.day))
  return actualDays.size === expectedDays.size && schedule.every((window) =>
    typeof window.day === 'number' &&
    expectedDays.has(window.day) &&
    window.startHour === startHour &&
    window.endHour === endHour
  )
}

/** Classify only the windows Desktop owns; every other Railway configuration
 * remains visible as custom and can be replaced explicitly by the user. */
export function classifyAutoUpdates(
  autoUpdates: RailwayImageAutoUpdates | null
): { mode: CloudAutoUpdateMode | 'custom' | null; policy: CloudAutoUpdatePolicy } {
  if (!autoUpdates) return { mode: null, policy: null }
  const policy = autoUpdates.type === 'disabled' ||
      autoUpdates.type === 'patch' ||
      autoUpdates.type === 'minor'
    ? autoUpdates.type
    : null
  if (policy === 'disabled') return { mode: 'off', policy }
  if (policy === 'patch' || policy === 'minor') {
    if (matchesSchedule(autoUpdates.schedule, [0, 1, 2, 3, 4, 5, 6], 2, 6)) {
      return { mode: 'nightly', policy }
    }
    if (matchesSchedule(autoUpdates.schedule, [0, 6], 0, 24)) {
      return { mode: 'weekends', policy }
    }
    if (matchesSchedule(autoUpdates.schedule, [0, 1, 2, 3, 4, 5, 6], 0, 24)) {
      return { mode: 'anytime', policy }
    }
  }
  return { mode: 'custom', policy }
}

async function readRailwayAutoUpdateState(
  getAutoUpdates: () => Promise<RailwayImageAutoUpdates | null>,
  settingsUrl: string
): Promise<CloudAutoUpdateStateResult> {
  try {
    const autoUpdates = await getAutoUpdates()
    return { ok: true, ...classifyAutoUpdates(autoUpdates), settingsUrl }
  } catch (error) {
    return {
      ok: false,
      mode: null,
      policy: null,
      message: `Railway could not read image auto-updates: ${errorText(error)}. ${settingsUrl}`,
      settingsUrl
    }
  }
}

export interface GetCloudAutoUpdateOptions {
  connectedServerUrl: string
  tfstate: CloudAutoUpdateStateIdentity | null
}

export interface GetCloudAutoUpdateDeps {
  getAccessToken: () => Promise<string | null>
  getVersion: () => Promise<ControlPlaneVersion | null>
  getAutoUpdates: (
    token: string,
    environmentId: string,
    serviceId: string
  ) => Promise<RailwayImageAutoUpdates | null>
}

/** Read only the sanitized autoUpdates subset from Railway; no environment
 * config document is ever returned to the caller or renderer. */
export async function getCloudAutoUpdateState(
  options: GetCloudAutoUpdateOptions,
  deps: GetCloudAutoUpdateDeps
): Promise<CloudAutoUpdateStateResult> {
  const token = await deps.getAccessToken().catch(() => null)
  const identity = await resolveCloudAutoUpdateIdentity(options, deps.getVersion)
  if (!token) {
    const settingsUrl = railwaySettingsUrl(identity?.projectId, identity?.serviceId)
    return {
      ok: false,
      mode: null,
      policy: null,
      ...(identity ? { serviceId: identity.serviceId } : {}),
      message: `Sign in to Railway to read image auto-updates. ${settingsUrl}`,
      settingsUrl
    }
  }
  if (!identity) {
    const settingsUrl = railwaySettingsUrl(null, null)
    return {
      ok: false,
      mode: null,
      policy: null,
      message: `The connected control plane could not be matched to a Railway service. ${settingsUrl}`,
      settingsUrl
    }
  }
  const settingsUrl = railwaySettingsUrl(identity.projectId, identity.serviceId)
  return {
    ...await readRailwayAutoUpdateState(
      () => deps.getAutoUpdates(
        token,
        identity.environmentId,
        identity.serviceId
      ),
      settingsUrl
    ),
    serviceId: identity.serviceId
  }
}

/** Apply a Railway schedule without leaking getVersion failures through IPC.
 * A transient version timeout falls back only to URL-matched tfstate. */
export async function setCloudAutoUpdateSchedule(
  options: SetCloudAutoUpdateOptions,
  deps: SetCloudAutoUpdateDeps
): Promise<SetCloudAutoUpdateResult> {
  let identity: ResolvedCloudAutoUpdateIdentity | null = null
  try {
    const token = await deps.getAccessToken()
    if (!token) {
      const settingsUrl = railwaySettingsUrl(null, null)
      return {
        ok: false,
        message: `Sign in to Railway, then choose the schedule again. ${settingsUrl}`,
        settingsUrl
      }
    }
    identity = await resolveCloudAutoUpdateIdentity(options, deps.getVersion)
    if (!identity) {
      const settingsUrl = railwaySettingsUrl(null, null)
      return {
        ok: false,
        message: `The connected control plane could not be matched to a Railway service. Reconnect the matching Railway control plane, then choose the schedule again. ${settingsUrl}`,
        settingsUrl
      }
    }

    const {
      projectId,
      environmentId,
      serviceId
    } = identity
    const settingsUrl = railwaySettingsUrl(projectId, serviceId)
    const live = await readRailwayAutoUpdateState(
      () => deps.getAutoUpdates(
        token,
        environmentId,
        serviceId
      ),
      settingsUrl
    )
    const readFailed = !live.ok
    const policy: RailwayEnabledAutoUpdatePolicy = live.policy === 'minor'
      ? 'minor'
      : 'patch'

    await deps.setImageAutoUpdates(
      token,
      environmentId,
      serviceId,
      options.mode,
      policy
    )
    return {
      ok: true,
      serviceId,
      settingsUrl,
      message: `Railway image auto-updates set to ${AUTO_UPDATE_LABELS[options.mode]}.` +
        (readFailed
          ? options.mode === 'off'
            ? " Railway's current policy could not be read first."
            : " Railway's current policy could not be read first; the patch policy was written."
          : live.policy === 'minor' && options.mode === 'off'
            ? " Railway's minor-update policy was replaced by Off; enabling a window again uses the patch policy."
            : '')
    }
  } catch (error) {
    const settingsUrl = railwaySettingsUrl(identity?.projectId, identity?.serviceId)
    return {
      ok: false,
      message: `Railway could not save that schedule: ${errorText(error)}. You can also set it in Railway → Settings → Source → Configure Auto Updates. ${settingsUrl}`,
      settingsUrl
    }
  }
}

/** First deploys and retried deploys without a usable cached preference get the
 * Nightly default. Reconciles preserve a mode already applied to this service. */
export function autoUpdateModeAfterDeploy(input: {
  firstDeploy: boolean
  serviceId: string
  storedMode: CloudAutoUpdateMode | null
  storedServiceId: string | null
}): CloudAutoUpdateMode | null {
  if (
    input.firstDeploy ||
    input.storedServiceId !== input.serviceId ||
    input.storedMode === null
  ) return 'nightly'
  return input.storedMode
}

export interface CloudAutoUpdateReconcileDecision {
  writeMode: CloudAutoUpdateMode | null
  autoUpdateOk: boolean
  autoUpdateMessage: string
}

function classifiedAutoUpdateLabel(mode: Exclude<CloudAutoUpdateLiveMode, null>): string {
  if (mode === 'custom') return 'Custom (set in Railway)'
  return AUTO_UPDATE_LABELS[mode]
}

/** Decide whether post-deploy reconciliation may write. A live Railway policy,
 * including a custom window, is authoritative. A first deploy owns its newly
 * created service, so it can safely write Nightly even when the read failed. */
export function cloudAutoUpdateReconcileDecision(input: {
  firstDeploy: boolean
  serviceId: string
  storedMode: CloudAutoUpdateMode | null
  storedServiceId: string | null
  live: CloudAutoUpdateStateResult
}): CloudAutoUpdateReconcileDecision {
  if (!input.live.ok) {
    if (input.firstDeploy) {
      return {
        writeMode: 'nightly',
        autoUpdateOk: true,
        autoUpdateMessage: `Railway image auto-updates set to ${AUTO_UPDATE_LABELS.nightly}. Railway's current policy could not be read after the first deploy; the patch policy was written.`
      }
    }
    return {
      writeMode: null,
      autoUpdateOk: false,
      autoUpdateMessage: input.live.message ?? 'Railway could not read image auto-updates.'
    }
  }
  if (input.live.mode !== null) {
    const policy = input.live.policy === 'minor' ? ' (minor policy)' : ''
    return {
      writeMode: null,
      autoUpdateOk: true,
      autoUpdateMessage: `Railway image auto-updates: ${classifiedAutoUpdateLabel(input.live.mode)}${policy}.`
    }
  }
  const writeMode = autoUpdateModeAfterDeploy(input)
  return {
    writeMode,
    autoUpdateOk: true,
    autoUpdateMessage: writeMode
      ? `Railway image auto-updates set to ${AUTO_UPDATE_LABELS[writeMode]}.`
      : 'Railway image auto-updates are not set; choose a window below.'
  }
}

export interface CloudDeployAutoUpdateResult {
  appliedMode: CloudAutoUpdateMode | null
  liveMode: CloudAutoUpdateLiveMode
  liveOk: boolean
  autoUpdateOk: boolean
  autoUpdateMessage: string
  autoUpdateSettingsUrl: string
}

/** Convert the reconcile result into Desktop's cached fallback. A successful
 * write wins. A failed or skipped read preserves the same-service fallback;
 * successful custom and unknown live states deliberately clear it. */
export function cloudAutoUpdatePreferenceAfterReconcile(
  result: Pick<CloudDeployAutoUpdateResult, 'appliedMode' | 'liveMode' | 'liveOk'> | null,
  storedModeForService: CloudAutoUpdateMode | null
): CloudAutoUpdateMode | null {
  if (!result) return storedModeForService
  if (result.appliedMode) return result.appliedMode
  if (!result.liveOk) return storedModeForService
  return result.liveMode === 'custom' ? null : result.liveMode
}

export interface ApplyCloudAutoUpdateAfterDeployDeps {
  getAutoUpdates: (
    environmentId: string,
    serviceId: string
  ) => Promise<RailwayImageAutoUpdates | null>
  setImageAutoUpdates: (
    environmentId: string,
    serviceId: string,
    mode: CloudAutoUpdateMode,
    policy: RailwayEnabledAutoUpdatePolicy
  ) => Promise<void>
}

/** Read Railway first, then apply a default/cached policy only when the service
 * has no policy at all. Returns fields that are sent directly to the renderer. */
export async function applyCloudAutoUpdateAfterDeploy(input: {
  firstDeploy: boolean
  projectId: string | null
  environmentId: string
  serviceId: string
  storedMode: CloudAutoUpdateMode | null
  storedServiceId: string | null
}, deps: ApplyCloudAutoUpdateAfterDeployDeps): Promise<CloudDeployAutoUpdateResult> {
  const settingsUrl = railwaySettingsUrl(input.projectId, input.serviceId)
  const live = await readRailwayAutoUpdateState(
    () => deps.getAutoUpdates(input.environmentId, input.serviceId),
    settingsUrl
  )
  const decision = cloudAutoUpdateReconcileDecision({ ...input, live })
  if (!decision.writeMode) {
    return {
      appliedMode: null,
      liveMode: live.mode,
      liveOk: live.ok,
      autoUpdateOk: decision.autoUpdateOk,
      autoUpdateMessage: decision.autoUpdateMessage,
      autoUpdateSettingsUrl: settingsUrl
    }
  }
  try {
    await deps.setImageAutoUpdates(
      input.environmentId,
      input.serviceId,
      decision.writeMode,
      'patch'
    )
    return {
      appliedMode: decision.writeMode,
      liveMode: live.mode,
      liveOk: live.ok,
      autoUpdateOk: true,
      autoUpdateMessage: decision.autoUpdateMessage,
      autoUpdateSettingsUrl: settingsUrl
    }
  } catch (error) {
    return {
      appliedMode: null,
      liveMode: live.mode,
      liveOk: live.ok,
      autoUpdateOk: false,
      autoUpdateMessage: `Automatic updates could not be scheduled: ${errorText(error)}. The deployment is healthy; choose a window below to retry or configure it in Railway → Settings → Source → Configure Auto Updates.`,
      autoUpdateSettingsUrl: settingsUrl
    }
  }
}

export interface CloudUpdateCheckerDeps {
  enabled: () => boolean
  getVersion: () => Promise<ControlPlaneVersion | null>
  getTfstateImage: () => string | null
  canApplyUpdate?: (running: ControlPlaneVersion | null) => boolean
  canManageRailway?: (running: ControlPlaneVersion | null) => boolean
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
    canApply: false,
    canManageRailway: false
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
        message: 'Cloud update checks run when a Remote control plane is enabled.',
        canApply: false,
        canManageRailway: false,
        hosting: undefined
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
      const canManageRailway = this.deps.canManageRailway?.(running) ?? false
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
        canManageRailway,
        hosting: running?.hosting
      })
      this.deps.onCompletedCheck?.(running)
      return status
    } catch (error) {
      return this.patch({
        status: 'unknown',
        checking: false,
        canApply: false,
        canManageRailway: false,
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
      // Let the renderer paint the positive apply result before the follow-up
      // check changes the status to current and intentionally removes it.
      if (result.ok) setTimeout(() => void this.check(), 500)
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
