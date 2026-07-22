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
      autostartControlPlane: false,
      controlPlanePort: 9091,
      lastControlPlanePort: 8081,
      autostartAgents: ['a', 'b'],
      installSkills: false,
      trayCompanion: false,
      dismissedUpdateVersion: '0.1.110'
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
})

describe('load/save round trip', () => {
  it('persists and reloads settings', async () => {
    const file = join(dir, 'nested', 'settings.json')
    const s = {
      openAtLogin: true,
      autostartControlPlane: true,
      controlPlanePort: null,
      lastControlPlanePort: 9091,
      autostartAgents: ['swe-planner'],
      installSkills: true,
      trayCompanion: true,
      dismissedUpdateVersion: null
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
