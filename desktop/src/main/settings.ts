// Persisted app settings. Plain JSON in the app's user-data directory —
// no electron imports here so normalization and IO stay unit-testable; the
// login-item side effect lives in index.ts where `app` is available.

import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'
import type { CloudAutoUpdateMode, DesktopSettings } from '../shared/types'

export const DEFAULT_SETTINGS: DesktopSettings = {
  cloud: {
    enabled: false,
    serverUrl: '',
    apiKey: '',
    autoUpdate: null,
    autoUpdateServiceId: null,
    dismissedUpdateVersion: null
  },
  openAtLogin: false,
  appearance: 'system',
  autostartControlPlane: true,
  controlPlanePort: null,
  localApiKey: '',
  lastControlPlanePort: null,
  autostartAgents: [],
  provisionedBundled: [],
  installSkills: true,
  trayCompanion: true,
  dismissedUpdateVersion: null,
  starPrompt: 'pending',
  starPromptSnoozedUntil: null,
  keyNoticeShown: []
}

/** A usable TCP port, or null for anything else (auto mode / not recorded). */
function normalizePort(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 65535
    ? value
    : null
}

/**
 * Coerce whatever was on disk (old versions, hand edits, corruption) into a
 * valid DesktopSettings. Unknown keys are dropped, wrong types fall back to
 * defaults, agent names are deduped strings.
 */
export function normalizeSettings(raw: unknown): DesktopSettings {
  const obj = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {}
  const cloud =
    typeof obj.cloud === 'object' && obj.cloud !== null
      ? (obj.cloud as Record<string, unknown>)
      : {}
  const agents = Array.isArray(obj.autostartAgents)
    ? [...new Set(obj.autostartAgents.filter((n): n is string => typeof n === 'string'))]
    : DEFAULT_SETTINGS.autostartAgents
  // Same coercion as autostartAgents: a hand-edited or corrupt list must not
  // be able to suppress (or duplicate) first-launch provisioning.
  const provisionedBundled = Array.isArray(obj.provisionedBundled)
    ? [...new Set(obj.provisionedBundled.filter((n): n is string => typeof n === 'string'))]
    : DEFAULT_SETTINGS.provisionedBundled
  // Same again for the once-only key notice: a corrupt list must neither
  // suppress the notification forever nor grow duplicates.
  const keyNoticeShown = Array.isArray(obj.keyNoticeShown)
    ? [...new Set(obj.keyNoticeShown.filter((n): n is string => typeof n === 'string'))]
    : DEFAULT_SETTINGS.keyNoticeShown
  const autoUpdateServiceId =
    typeof cloud.autoUpdateServiceId === 'string' && cloud.autoUpdateServiceId !== ''
      ? cloud.autoUpdateServiceId
      : null
  return {
    cloud: {
      enabled: Boolean(cloud.enabled),
      serverUrl: typeof cloud.serverUrl === 'string' ? cloud.serverUrl.trim() : '',
      apiKey: typeof cloud.apiKey === 'string' ? cloud.apiKey.trim() : '',
      // Older files had a default mode but no record of which Railway
      // service it was applied to. Migrate those to not-yet-applied.
      autoUpdate:
        autoUpdateServiceId !== null &&
        (cloud.autoUpdate === 'off' ||
          cloud.autoUpdate === 'nightly' ||
          cloud.autoUpdate === 'weekends' ||
          cloud.autoUpdate === 'anytime')
          ? cloud.autoUpdate
          : null,
      autoUpdateServiceId,
      dismissedUpdateVersion:
        typeof cloud.dismissedUpdateVersion === 'string' && cloud.dismissedUpdateVersion !== ''
          ? cloud.dismissedUpdateVersion
          : null
    },
    openAtLogin:
      typeof obj.openAtLogin === 'boolean' ? obj.openAtLogin : DEFAULT_SETTINGS.openAtLogin,
    appearance:
      obj.appearance === 'light' || obj.appearance === 'dark' || obj.appearance === 'system'
        ? obj.appearance
        : DEFAULT_SETTINGS.appearance,
    autostartControlPlane:
      typeof obj.autostartControlPlane === 'boolean'
        ? obj.autostartControlPlane
        : DEFAULT_SETTINGS.autostartControlPlane,
    controlPlanePort: normalizePort(obj.controlPlanePort),
    localApiKey: typeof obj.localApiKey === 'string' ? obj.localApiKey.trim() : '',
    lastControlPlanePort: normalizePort(obj.lastControlPlanePort),
    autostartAgents: agents,
    provisionedBundled,
    installSkills:
      typeof obj.installSkills === 'boolean' ? obj.installSkills : DEFAULT_SETTINGS.installSkills,
    trayCompanion:
      typeof obj.trayCompanion === 'boolean' ? obj.trayCompanion : DEFAULT_SETTINGS.trayCompanion,
    dismissedUpdateVersion:
      typeof obj.dismissedUpdateVersion === 'string' && obj.dismissedUpdateVersion !== ''
        ? obj.dismissedUpdateVersion
        : null,
    starPrompt: obj.starPrompt === 'done' || obj.starPrompt === 'pending' ? obj.starPrompt : 'pending',
    starPromptSnoozedUntil:
      typeof obj.starPromptSnoozedUntil === 'string' &&
      obj.starPromptSnoozedUntil !== '' &&
      Number.isFinite(Date.parse(obj.starPromptSnoozedUntil))
        ? obj.starPromptSnoozedUntil
        : null,
    keyNoticeShown
  }
}

/** Merge a partial update (renderer-supplied, so also unvalidated) into base. */
export function mergeSettings(base: DesktopSettings, patch: unknown): DesktopSettings {
  const p = typeof patch === 'object' && patch !== null ? (patch as Record<string, unknown>) : {}
  return normalizeSettings({ ...base, ...p })
}

/** Main-owned field-wise cloud profile merge. The renderer sends only the
 * connection fields it edits, so schedule and dismissal state stay current. */
export function settingsWithCloudProfile(
  base: DesktopSettings,
  profile: { enabled: boolean; serverUrl: string; apiKey: string }
): DesktopSettings {
  return mergeSettings(base, {
    cloud: {
      ...base.cloud,
      enabled: profile.enabled,
      serverUrl: profile.serverUrl,
      apiKey: profile.apiKey
    }
  })
}

/** Reset the applied schedule when Desktop observes a different Railway service. */
export function settingsForCloudService(
  base: DesktopSettings,
  serviceId: string
): DesktopSettings {
  if (base.cloud.autoUpdateServiceId === serviceId) return base
  return mergeSettings(base, {
    cloud: { ...base.cloud, autoUpdate: null, autoUpdateServiceId: serviceId }
  })
}

/** Main-process-only dismissal merge; never accepts a renderer cloud snapshot. */
export function settingsWithDismissedCloudUpdate(
  base: DesktopSettings,
  version: string
): DesktopSettings {
  return mergeSettings(base, {
    cloud: { ...base.cloud, dismissedUpdateVersion: version }
  })
}

/** Persist before returning the replacement settings object so a failed disk
 * write cannot make main's in-memory schedule claim something it did not save. */
export async function persistCloudAutoUpdatePreference(
  base: DesktopSettings,
  mode: CloudAutoUpdateMode,
  serviceId: string,
  persist: (next: DesktopSettings) => Promise<void>
): Promise<DesktopSettings> {
  const next = mergeSettings(base, {
    cloud: {
      ...base.cloud,
      autoUpdate: mode,
      autoUpdateServiceId: serviceId
    }
  })
  await persist(next)
  return next
}

export async function loadSettings(file: string): Promise<DesktopSettings> {
  try {
    return normalizeSettings(JSON.parse(await fs.readFile(file, 'utf8')))
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

export async function saveSettings(file: string, settings: DesktopSettings): Promise<void> {
  await fs.mkdir(dirname(file), { recursive: true })
  await fs.writeFile(file, JSON.stringify(settings, null, 2) + '\n', 'utf8')
}
