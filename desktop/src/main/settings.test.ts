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
      autostartAgents: ['a', 'b'],
      installSkills: false
    }
    expect(normalizeSettings(s)).toEqual(s)
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
      autostartAgents: ['swe-planner'],
      installSkills: true
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
