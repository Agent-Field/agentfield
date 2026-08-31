import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS, loadSettings, mergeSettings, normalizeSettings, saveSettings } from './settings'

const dir = mkdtempSync(join(tmpdir(), 'af-desktop-settings-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

describe('normalizeSettings', () => {
  it('accepts a valid shape as-is', () => {
    const s = {
      openAtLogin: true,
      appearance: 'dark' as const,
      autostartControlPlane: false,
      controlPlanePort: 9091,
      lastControlPlanePort: 8081,
      autostartAgents: ['a', 'b'],
      installSkills: false,
      trayCompanion: false,
      dismissedUpdateVersion: '0.1.110',
      starPrompt: 'done' as const,
      starPromptSnoozedUntil: '2026-08-01T00:00:00.000Z'
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

  it('drops non-string agent names and dedupes', () => {
    expect(
      normalizeSettings({ autostartAgents: ['a', 7, 'a', null, 'b'] }).autostartAgents
    ).toEqual(['a', 'b'])
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

  it('merges star prompt patches', () => {
    const done = mergeSettings(DEFAULT_SETTINGS, { starPrompt: 'done' })
    expect(done.starPrompt).toBe('done')
    const snoozed = mergeSettings(DEFAULT_SETTINGS, {
      starPromptSnoozedUntil: '2026-08-08T00:00:00.000Z'
    })
    expect(snoozed.starPromptSnoozedUntil).toBe('2026-08-08T00:00:00.000Z')
    expect(snoozed.starPrompt).toBe('pending')
  })
})

describe('load/save round trip', () => {
  it('persists and reloads settings', async () => {
    const file = join(dir, 'nested', 'settings.json')
    const s = {
      openAtLogin: true,
      appearance: 'light' as const,
      autostartControlPlane: true,
      controlPlanePort: null,
      lastControlPlanePort: 9091,
      autostartAgents: ['swe-planner'],
      installSkills: true,
      trayCompanion: true,
      dismissedUpdateVersion: null,
      starPrompt: 'pending' as const,
      starPromptSnoozedUntil: null
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
