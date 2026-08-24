import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import {
  DEFAULT_SETTINGS,
  loadSettings,
  mergeSettings,
  normalizeSettings,
  persistCloudAutoUpdatePreference,
  saveSettings,
  settingsWithCloudProfile,
  settingsForCloudService,
  settingsWithDismissedCloudUpdate
} from './settings'

const dir = mkdtempSync(join(tmpdir(), 'af-desktop-settings-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

describe('normalizeSettings', () => {
  it('accepts a valid shape as-is', () => {
    const s = {
      cloud: {
        enabled: true,
        serverUrl: 'https://cloud.example',
        apiKey: 'secret',
        autoUpdate: 'weekends' as const,
        autoUpdateServiceId: 'service-1',
        dismissedUpdateVersion: '0.1.135'
      },
      openAtLogin: true,
      appearance: 'dark' as const,
      autostartControlPlane: false,
      controlPlanePort: 9091,
      localApiKey: 'local-secret',
      lastControlPlanePort: 8081,
      autostartAgents: ['a', 'b'],
      provisionedBundled: ['swe-planner'],
      installSkills: false,
      trayCompanion: false,
      dismissedUpdateVersion: '0.1.110',
      starPrompt: 'done' as const,
      starPromptSnoozedUntil: '2026-08-01T00:00:00.000Z',
      keyNoticeShown: ['swe-planner']
    }
    expect(normalizeSettings(s)).toEqual(s)
  })

  it('coerces bad ports to null (auto)', () => {
    expect(normalizeSettings({}).controlPlanePort).toBeNull()
    expect(normalizeSettings({ controlPlanePort: 8080 }).controlPlanePort).toBe(8080)
    expect(normalizeSettings({ controlPlanePort: 0 }).controlPlanePort).toBeNull()
    expect(normalizeSettings({ controlPlanePort: 65536 }).controlPlanePort).toBeNull()
    expect(normalizeSettings({ controlPlanePort: 8080.5 }).controlPlanePort).toBeNull()
    expect(normalizeSettings({ controlPlanePort: '8080' }).controlPlanePort).toBeNull()
    expect(normalizeSettings({ lastControlPlanePort: -1 }).lastControlPlanePort).toBeNull()
    expect(normalizeSettings({ lastControlPlanePort: 9091 }).lastControlPlanePort).toBe(9091)
  })

  it('keeps a trimmed local API key and drops non-strings', () => {
    expect(normalizeSettings({}).localApiKey).toBe('')
    expect(normalizeSettings({ localApiKey: '  af_local_key  ' }).localApiKey).toBe('af_local_key')
    expect(normalizeSettings({ localApiKey: 42 }).localApiKey).toBe('')
  })

  it('defaults trayCompanion on and coerces non-booleans', () => {
    expect(normalizeSettings({}).trayCompanion).toBe(true)
    expect(normalizeSettings({ trayCompanion: false }).trayCompanion).toBe(false)
    expect(normalizeSettings({ trayCompanion: 'yes' }).trayCompanion).toBe(true)
  })

  it('normalizes appearance overrides', () => {
    expect(normalizeSettings({}).appearance).toBe('system')
    expect(normalizeSettings({ appearance: 'system' }).appearance).toBe('system')
    expect(normalizeSettings({ appearance: 'light' }).appearance).toBe('light')
    expect(normalizeSettings({ appearance: 'dark' }).appearance).toBe('dark')
    expect(normalizeSettings({ appearance: 'sepia' }).appearance).toBe('system')
  })

  it('falls back to defaults for garbage', () => {
    expect(normalizeSettings(null)).toEqual(DEFAULT_SETTINGS)
    expect(normalizeSettings('nope')).toEqual(DEFAULT_SETTINGS)
    expect(normalizeSettings({ openAtLogin: 'yes', autostartAgents: 42 })).toEqual(
      DEFAULT_SETTINGS
    )
  })

  it('normalizes cloud profile values and defaults old settings', () => {
    expect(normalizeSettings({}).cloud).toEqual(DEFAULT_SETTINGS.cloud)
    expect(
      normalizeSettings({
        cloud: { enabled: 'yes', serverUrl: '  https://cp.example/  ', apiKey: ' key ' }
      }).cloud
    ).toEqual({
      enabled: true,
      serverUrl: 'https://cp.example/',
      apiKey: 'key',
      autoUpdate: null,
      autoUpdateServiceId: null,
      dismissedUpdateVersion: null
    })
    expect(normalizeSettings({ cloud: { enabled: 0, serverUrl: 7, apiKey: null } }).cloud).toEqual({
      enabled: false,
      serverUrl: '',
      apiKey: '',
      autoUpdate: null,
      autoUpdateServiceId: null,
      dismissedUpdateVersion: null
    })
  })

  it('migrates cloud schedules to not-set unless their applied service is recorded', () => {
    expect(normalizeSettings({}).cloud.autoUpdate).toBeNull()
    expect(normalizeSettings({ cloud: { autoUpdate: 'anytime' } }).cloud.autoUpdate).toBeNull()
    expect(normalizeSettings({
      cloud: { autoUpdate: 'anytime', autoUpdateServiceId: 'service-1' }
    }).cloud.autoUpdate).toBe('anytime')
    expect(normalizeSettings({
      cloud: { autoUpdate: 'invalid', autoUpdateServiceId: 'service-1' }
    }).cloud.autoUpdate).toBeNull()
    expect(
      normalizeSettings({ cloud: { dismissedUpdateVersion: '0.1.135' } }).cloud.dismissedUpdateVersion
    ).toBe('0.1.135')
  })

  it('drops non-string agent names and dedupes', () => {
    expect(
      normalizeSettings({ autostartAgents: ['a', 7, 'a', null, 'b'] }).autostartAgents
    ).toEqual(['a', 'b'])
  })

  // provisionedBundled is what makes uninstalling a bundled node stick, so a
  // hand-edited or corrupt list must degrade to "provision it again", never to
  // a shape that could suppress or duplicate first-launch provisioning.
  it('coerces provisionedBundled like autostartAgents', () => {
    expect(normalizeSettings({}).provisionedBundled).toEqual([])
    expect(
      normalizeSettings({ provisionedBundled: ['pr-af', 7, 'pr-af', null, 'swe-planner'] })
        .provisionedBundled
    ).toEqual(['pr-af', 'swe-planner'])
    expect(normalizeSettings({ provisionedBundled: 'pr-af' }).provisionedBundled).toEqual([])
  })

  it('coerces a bad dismissed update version to null', () => {
    expect(normalizeSettings({ dismissedUpdateVersion: 42 }).dismissedUpdateVersion).toBeNull()
    expect(normalizeSettings({ dismissedUpdateVersion: '' }).dismissedUpdateVersion).toBeNull()
    expect(normalizeSettings({ dismissedUpdateVersion: '0.2.0' }).dismissedUpdateVersion).toBe(
      '0.2.0'
    )
  })

  it('defaults star prompt fields and coerces unknowns', () => {
    expect(normalizeSettings({}).starPrompt).toBe('pending')
    expect(normalizeSettings({}).starPromptSnoozedUntil).toBeNull()
    expect(normalizeSettings({ starPrompt: 'done' }).starPrompt).toBe('done')
    expect(normalizeSettings({ starPrompt: 'maybe' }).starPrompt).toBe('pending')
    expect(normalizeSettings({ starPrompt: 1 }).starPrompt).toBe('pending')
    expect(normalizeSettings({ starPromptSnoozedUntil: '' }).starPromptSnoozedUntil).toBeNull()
    expect(normalizeSettings({ starPromptSnoozedUntil: 42 }).starPromptSnoozedUntil).toBeNull()
    expect(normalizeSettings({ starPromptSnoozedUntil: 'not-a-date' }).starPromptSnoozedUntil).toBeNull()
    expect(
      normalizeSettings({ starPromptSnoozedUntil: '2026-08-01T12:00:00.000Z' }).starPromptSnoozedUntil
    ).toBe('2026-08-01T12:00:00.000Z')
  })
})

describe('mergeSettings', () => {
  it('applies a partial patch over the base', () => {
    const merged = mergeSettings(DEFAULT_SETTINGS, { openAtLogin: true })
    expect(merged.openAtLogin).toBe(true)
    expect(merged.autostartControlPlane).toBe(DEFAULT_SETTINGS.autostartControlPlane)
  })

  it('sanitizes hostile patches (renderer input is untrusted)', () => {
    const merged = mergeSettings(DEFAULT_SETTINGS, {
      autostartAgents: ['ok', { evil: true }],
      openAtLogin: 'true'
    })
    expect(merged.autostartAgents).toEqual(['ok'])
    expect(merged.openAtLogin).toBe(false)
  })

  it('sanitizes a provisionedBundled patch', () => {
    const merged = mergeSettings(DEFAULT_SETTINGS, {
      provisionedBundled: ['pr-af', { evil: true }, 'pr-af']
    })
    expect(merged.provisionedBundled).toEqual(['pr-af'])
  })

  it('merges star prompt patches', () => {
    const done = mergeSettings(DEFAULT_SETTINGS, { starPrompt: 'done' })
    expect(done.starPrompt).toBe('done')
    const snoozed = mergeSettings(DEFAULT_SETTINGS, {
      starPromptSnoozedUntil: '2026-08-08T00:00:00.000Z'
    })
    expect(snoozed.starPromptSnoozedUntil).toBe('2026-08-08T00:00:00.000Z')
    expect(snoozed.starPrompt).toBe('pending')
  })

  it('resets the applied schedule when the connected Railway service changes', () => {
    const applied = normalizeSettings({
      cloud: {
        enabled: true,
        serverUrl: 'https://cp.example',
        apiKey: 'key',
        autoUpdate: 'off',
        autoUpdateServiceId: 'service-a'
      }
    })
    expect(settingsForCloudService(applied, 'service-a')).toBe(applied)
    expect(settingsForCloudService(applied, 'service-b').cloud).toMatchObject({
      autoUpdate: null,
      autoUpdateServiceId: 'service-b'
    })
  })

  it('dismisses a cloud version without reverting newer cloud settings', () => {
    const current = normalizeSettings({
      cloud: {
        enabled: true,
        serverUrl: 'https://new.example',
        apiKey: 'new-key',
        autoUpdate: 'weekends',
        autoUpdateServiceId: 'service-new'
      }
    })
    expect(settingsWithDismissedCloudUpdate(current, '0.1.136').cloud).toEqual({
      ...current.cloud,
      dismissedUpdateVersion: '0.1.136'
    })
  })

  it('saves a renderer cloud profile without reverting main-owned cloud fields', () => {
    const current = normalizeSettings({
      cloud: {
        enabled: false,
        serverUrl: 'https://old.example',
        apiKey: 'old-key',
        autoUpdate: 'weekends',
        autoUpdateServiceId: 'service-new',
        dismissedUpdateVersion: '0.1.136'
      }
    })

    expect(settingsWithCloudProfile(current, {
      enabled: true,
      serverUrl: 'https://new.example',
      apiKey: 'new-key'
    }).cloud).toEqual({
      enabled: true,
      serverUrl: 'https://new.example',
      apiKey: 'new-key',
      autoUpdate: 'weekends',
      autoUpdateServiceId: 'service-new',
      dismissedUpdateVersion: '0.1.136'
    })
  })

  it('does not publish a Railway schedule preference when persistence fails', async () => {
    const previous = normalizeSettings({
      cloud: {
        autoUpdate: 'nightly',
        autoUpdateServiceId: 'service-a'
      }
    })
    let current = previous

    try {
      current = await persistCloudAutoUpdatePreference(
        current,
        'weekends',
        'service-a',
        async () => { throw new Error('disk full') }
      )
    } catch {
      // The caller retains its previous in-memory value when persistence rejects.
    }

    expect(current).toBe(previous)
    expect(current.cloud.autoUpdate).toBe('nightly')
  })
})

describe('load/save round trip', () => {
  it('persists and reloads settings', async () => {
    const file = join(dir, 'nested', 'settings.json')
    const s = {
      cloud: {
        enabled: true,
        serverUrl: 'https://cloud.example',
        apiKey: 'round-trip-key',
        autoUpdate: 'nightly' as const,
        autoUpdateServiceId: 'service-round-trip',
        dismissedUpdateVersion: null
      },
      openAtLogin: true,
      appearance: 'light' as const,
      autostartControlPlane: true,
      controlPlanePort: null,
      localApiKey: 'round-trip-local-key',
      lastControlPlanePort: 9091,
      autostartAgents: ['swe-planner'],
      provisionedBundled: ['swe-planner', 'pr-af'],
      installSkills: true,
      trayCompanion: true,
      dismissedUpdateVersion: null,
      starPrompt: 'pending' as const,
      starPromptSnoozedUntil: null,
      keyNoticeShown: []
    }
    await saveSettings(file, s)
    expect(await loadSettings(file)).toEqual(s)
  })

  it('missing or corrupt file yields defaults', async () => {
    expect(await loadSettings(join(dir, 'nope.json'))).toEqual(DEFAULT_SETTINGS)
    const bad = join(dir, 'bad.json')
    await saveSettings(bad, DEFAULT_SETTINGS)
    const fs = await import('node:fs')
    fs.writeFileSync(bad, '{not json')
    expect(await loadSettings(bad)).toEqual(DEFAULT_SETTINGS)
  })
})
