import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { AppUpdateStatus } from '../shared/types'
import {
  AppUpdater,
  type AppUpdaterDeps,
  compareAppVersions,
  parseLatestRelease,
  pickInstallerAsset
} from './updates'

const dir = mkdtempSync(join(tmpdir(), 'af-desktop-updates-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

// A realistic /releases/latest payload: electron-builder installers mixed in
// with goreleaser CLI archives and checksums, exactly like a real release.
const RELEASE = {
  tag_name: 'v0.2.0',
  html_url: 'https://github.com/Agent-Field/agentfield/releases/tag/v0.2.0',
  assets: [
    { name: 'agentfield_0.2.0_windows_amd64.zip', browser_download_url: 'https://dl/cli.zip', size: 10 },
    { name: 'checksums.txt', browser_download_url: 'https://dl/checksums.txt', size: 1 },
    { name: 'AgentField-Setup-0.2.0.exe', browser_download_url: 'https://dl/setup.exe', size: 40 },
    { name: 'AgentField-0.2.0-arm64.dmg', browser_download_url: 'https://dl/app.dmg', size: 50 },
    { name: 'AgentField-0.2.0-arm64-mac.zip', browser_download_url: 'https://dl/mac.zip', size: 45 }
  ]
}

function deps(overrides: Partial<AppUpdaterDeps>): AppUpdaterDeps {
  return {
    currentVersion: '0.1.0',
    platform: 'win32',
    tempDir: dir,
    openExternal: () => {},
    openPath: () => Promise.resolve(''),
    quitForUpdate: () => {},
    launchInstaller: () => {},
    ...overrides
  }
}

const jsonResponse = (payload: unknown) =>
  (() => Promise.resolve(new Response(JSON.stringify(payload)))) as typeof fetch

describe('compareAppVersions', () => {
  it('orders numeric cores', () => {
    expect(compareAppVersions('0.1.110', '0.1.109')).toBeGreaterThan(0)
    expect(compareAppVersions('0.1.109', '0.1.109')).toBe(0)
    expect(compareAppVersions('v0.2.0', '0.1.999')).toBeGreaterThan(0)
  })

  it('ranks a release above its own prereleases', () => {
    expect(compareAppVersions('0.1.110', '0.1.110-rc.2')).toBeGreaterThan(0)
    expect(compareAppVersions('0.1.110-rc.2', '0.1.110')).toBeLessThan(0)
    // ...but not above a newer prerelease core.
    expect(compareAppVersions('0.1.109', '0.1.110-rc.1')).toBeLessThan(0)
  })

  it('orders prerelease numbers numerically', () => {
    expect(compareAppVersions('0.1.110-rc.10', '0.1.110-rc.2')).toBeGreaterThan(0)
  })
})

describe('pickInstallerAsset', () => {
  it('picks the NSIS installer on Windows, never the CLI zip', () => {
    const asset = pickInstallerAsset(RELEASE.assets, 'win32')
    expect(asset?.name).toBe('AgentField-Setup-0.2.0.exe')
    expect(asset?.url).toBe('https://dl/setup.exe')
  })

  it('picks the DMG on macOS, never the mac zip', () => {
    expect(pickInstallerAsset(RELEASE.assets, 'darwin')?.name).toBe('AgentField-0.2.0-arm64.dmg')
  })

  it('returns null on platforms without a packaged app', () => {
    expect(pickInstallerAsset(RELEASE.assets, 'linux')).toBeNull()
  })
})

describe('parseLatestRelease', () => {
  it('shapes a release payload into update info', () => {
    const info = parseLatestRelease(RELEASE, 'win32')
    expect(info).toEqual({
      version: '0.2.0',
      tagName: 'v0.2.0',
      releaseUrl: RELEASE.html_url,
      assetName: 'AgentField-Setup-0.2.0.exe',
      assetUrl: 'https://dl/setup.exe',
      assetSize: 40
    })
  })

  it('keeps the release usable when no platform asset exists', () => {
    const info = parseLatestRelease(RELEASE, 'linux')
    expect(info?.assetUrl).toBeNull()
    expect(info?.releaseUrl).toBe(RELEASE.html_url)
  })

  it('rejects payloads without a tag', () => {
    expect(parseLatestRelease({}, 'win32')).toBeNull()
    expect(parseLatestRelease(null, 'win32')).toBeNull()
  })
})

describe('AppUpdater.check', () => {
  it('reports a newer release as available', async () => {
    const updater = new AppUpdater(deps({ fetchImpl: jsonResponse(RELEASE) }))
    const status = await updater.check()
    expect(status.available?.version).toBe('0.2.0')
    expect(status.lastCheckedAt).not.toBeNull()
    expect(status.error).toBeNull()
  })

  it('reports up to date when current matches the latest release', async () => {
    const updater = new AppUpdater(
      deps({ currentVersion: '0.2.0', fetchImpl: jsonResponse(RELEASE) })
    )
    expect((await updater.check()).available).toBeNull()
  })

  it('offers the stable release to a matching RC install', async () => {
    const updater = new AppUpdater(
      deps({ currentVersion: '0.2.0-rc.1', fetchImpl: jsonResponse(RELEASE) })
    )
    expect((await updater.check()).available?.version).toBe('0.2.0')
  })

  it('treats 404 (no stable release yet) as up to date, not an error', async () => {
    const updater = new AppUpdater(
      deps({
        fetchImpl: (() =>
          Promise.resolve(new Response('not found', { status: 404 }))) as typeof fetch
      })
    )
    const status = await updater.check()
    expect(status.available).toBeNull()
    expect(status.error).toBeNull()
    expect(status.lastCheckedAt).not.toBeNull()
  })

  it('never rejects — network failures land in status.error', async () => {
    const updater = new AppUpdater(
      deps({ fetchImpl: (() => Promise.reject(new Error('offline'))) as typeof fetch })
    )
    const status = await updater.check()
    expect(status.error).toContain('offline')
    expect(status.checking).toBe(false)
  })

  it('a failed check clears a previously found update only on success', async () => {
    let fail = false
    const fetchImpl = (() =>
      fail
        ? Promise.resolve(new Response('rate limited', { status: 403 }))
        : Promise.resolve(new Response(JSON.stringify(RELEASE)))) as typeof fetch
    const updater = new AppUpdater(deps({ fetchImpl }))
    await updater.check()
    fail = true
    const status = await updater.check()
    // The catch path only sets error; the last known update stays offered.
    expect(status.available?.version).toBe('0.2.0')
    expect(status.error).toContain('403')
  })
})

describe('AppUpdater.install', () => {
  const withUpdate = async (overrides: Partial<AppUpdaterDeps>) => {
    const calls: Record<string, unknown[]> = { launch: [], quit: [], external: [], open: [] }
    const fetchImpl = ((url: unknown) =>
      new URL(String(url)).hostname === 'api.github.com'
        ? Promise.resolve(new Response(JSON.stringify(RELEASE)))
        : Promise.resolve(new Response('installer-bytes'))) as typeof fetch
    const updater = new AppUpdater(
      deps({
        fetchImpl,
        launchInstaller: (p) => calls.launch.push(p),
        quitForUpdate: () => calls.quit.push(true),
        openExternal: (u) => calls.external.push(u),
        openPath: (p) => {
          calls.open.push(p)
          return Promise.resolve('')
        },
        ...overrides
      })
    )
    await updater.check()
    return { updater, calls }
  }

  it('downloads and launches the Windows installer, then quits', async () => {
    const { updater, calls } = await withUpdate({ platform: 'win32' })
    const status = await updater.install()
    expect(status.error).toBeNull()
    expect(calls.launch).toHaveLength(1)
    expect(calls.quit).toHaveLength(1)
    const file = calls.launch[0] as string
    expect(file.endsWith('AgentField-Setup-0.2.0.exe')).toBe(true)
    expect(readFileSync(file, 'utf8')).toBe('installer-bytes')
  })

  it('downloads and opens the DMG on macOS without quitting', async () => {
    const { updater, calls } = await withUpdate({ platform: 'darwin' })
    const status = await updater.install()
    expect(status.error).toBeNull()
    expect(calls.open).toHaveLength(1)
    expect(String(calls.open[0]).endsWith('AgentField-0.2.0-arm64.dmg')).toBe(true)
    expect(calls.quit).toHaveLength(0)
  })

  it('opens the release page when the platform has no installer asset', async () => {
    const { updater, calls } = await withUpdate({ platform: 'linux' })
    await updater.install()
    expect(calls.external).toEqual([RELEASE.html_url])
    expect(calls.launch).toHaveLength(0)
  })

  it('is a no-op without a known update', async () => {
    const calls: string[] = []
    const updater = new AppUpdater(deps({ openExternal: (u) => calls.push(u) }))
    const status = await updater.install()
    expect(status.downloading).toBe(false)
    expect(calls).toHaveLength(0)
  })

  it('a failed download lands in status.error and resets the flow', async () => {
    const fetchImpl = ((url: unknown) =>
      new URL(String(url)).hostname === 'api.github.com'
        ? Promise.resolve(new Response(JSON.stringify(RELEASE)))
        : Promise.reject(new Error('connection reset'))) as typeof fetch
    const updater = new AppUpdater(deps({ fetchImpl }))
    await updater.check()
    const status = await updater.install()
    expect(status.error).toContain('connection reset')
    expect(status.downloading).toBe(false)
    expect(status.progress).toBeNull()
  })

  it('pushes progress updates while downloading', async () => {
    const seen: Array<number | null> = []
    const { updater } = await withUpdate({
      platform: 'win32',
      onStatus: (s: AppUpdateStatus) => seen.push(s.progress)
    })
    await updater.install()
    expect(seen).toContain(100)
  })
})
