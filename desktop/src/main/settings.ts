// Persisted app settings. Plain JSON in the app's user-data directory —
// no electron imports here so normalization and IO stay unit-testable; the
// login-item side effect lives in index.ts where `app` is available.

import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'
import type { DesktopSettings } from '../shared/types'

export const DEFAULT_SETTINGS: DesktopSettings = {
  openAtLogin: false,
  autostartControlPlane: true,
  autostartAgents: [],
  installSkills: true,
  trayCompanion: true,
  dismissedUpdateVersion: null
}

/**
 * Coerce whatever was on disk (old versions, hand edits, corruption) into a
 * valid DesktopSettings. Unknown keys are dropped, wrong types fall back to
 * defaults, agent names are deduped strings.
 */
export function normalizeSettings(raw: unknown): DesktopSettings {
  const obj = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {}
  const agents = Array.isArray(obj.autostartAgents)
    ? [...new Set(obj.autostartAgents.filter((n): n is string => typeof n === 'string'))]
    : DEFAULT_SETTINGS.autostartAgents
  return {
    openAtLogin:
      typeof obj.openAtLogin === 'boolean' ? obj.openAtLogin : DEFAULT_SETTINGS.openAtLogin,
    autostartControlPlane:
      typeof obj.autostartControlPlane === 'boolean'
        ? obj.autostartControlPlane
        : DEFAULT_SETTINGS.autostartControlPlane,
    autostartAgents: agents,
    installSkills:
      typeof obj.installSkills === 'boolean' ? obj.installSkills : DEFAULT_SETTINGS.installSkills,
    trayCompanion:
      typeof obj.trayCompanion === 'boolean' ? obj.trayCompanion : DEFAULT_SETTINGS.trayCompanion,
    dismissedUpdateVersion:
      typeof obj.dismissedUpdateVersion === 'string' && obj.dismissedUpdateVersion !== ''
        ? obj.dismissedUpdateVersion
        : null
  }
}

/** Merge a partial update (renderer-supplied, so also unvalidated) into base. */
export function mergeSettings(base: DesktopSettings, patch: unknown): DesktopSettings {
  const p = typeof patch === 'object' && patch !== null ? (patch as Record<string, unknown>) : {}
  return normalizeSettings({ ...base, ...p })
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
