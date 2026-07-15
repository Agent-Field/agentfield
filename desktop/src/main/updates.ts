// The app's own update channel. AgentField is a public repo, so the GitHub
// releases feed IS the channel: /releases/latest (stable releases only — RC
// prereleases never appear there) is compared against the running app's
// version, and this platform's installer asset from that release is
// downloaded and handed off on demand.
//
// The release train is the monorepo's, not the desktop app's: releases exist
// that carry only CLI/SDK artifacts (every release before the desktop app
// landed, and any cut from a branch without it). A release is only an *app*
// update when it actually ships this platform's desktop installer — the
// asset is the marker. Version comparison alone would offer "updates" the
// app cannot install (and once mis-offered, the only action left is a
// browser link — exactly the manual flow in-app updates exist to replace).
//
// Builds are unsigned for now, which rules out electron-updater's silent
// flows (Squirrel.Mac refuses unsigned apps). Instead: Windows runs the
// downloaded NSIS one-click installer — it replaces the app in place and
// relaunches it — and macOS opens the downloaded DMG for a drag-install.
//
// No electron imports: the version, platform, temp dir, and every side
// effect (open/launch/quit) are injected by main/index.ts so the check,
// asset selection, and download flow stay unit-testable.

import { spawn } from 'node:child_process'
import { createWriteStream, promises as fs } from 'node:fs'
import { join } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { FetchLike } from './agentfield'
import type { AppUpdateInfo, AppUpdateStatus } from '../shared/types'
import { compareVersions } from './cli'

const RELEASES_LATEST_URL = 'https://api.github.com/repos/Agent-Field/agentfield/releases/latest'
const CHECK_TIMEOUT_MS = 15_000
const DOWNLOAD_TIMEOUT_MS = 10 * 60_000
// GitHub rejects requests without a User-Agent; identify the caller honestly.
const API_HEADERS = { Accept: 'application/vnd.github+json', 'User-Agent': 'agentfield-desktop' }

/** Split "v0.1.110-rc.2" into its numeric core and optional prerelease tag. */
function splitVersion(version: string): { core: string; pre: string | null } {
  const clean = version.replace(/^v/, '')
  const dash = clean.indexOf('-')
  return dash === -1
    ? { core: clean, pre: null }
    : { core: clean.slice(0, dash), pre: clean.slice(dash + 1) }
}

/**
 * Dotted-core compare with prerelease awareness: a release outranks its own
 * prereleases (0.1.110 > 0.1.110-rc.2), so a staging install is offered the
 * stable build of the same version the moment it lands.
 */
export function compareAppVersions(a: string, b: string): number {
  const va = splitVersion(a)
  const vb = splitVersion(b)
  const core = compareVersions(va.core, vb.core)
  if (core !== 0) return core
  if (va.pre === vb.pre) return 0
  if (va.pre === null) return 1
  if (vb.pre === null) return -1
  return va.pre.localeCompare(vb.pre, undefined, { numeric: true })
}

interface InstallerAsset {
  name: string
  url: string
  size: number | null
}

/**
 * This platform's installer among a release's assets. Releases also carry
 * goreleaser CLI archives and checksums, so match exactly what
 * electron-builder produces (see desktop/package.json build): the
 * AgentField-Setup-<version>.exe NSIS installer on Windows, a .dmg on
 * macOS. Anything else (Linux has no packaged app yet, CLI-only releases
 * carry no installers at all) gets null, and check() then treats the
 * release as not-an-update for this platform.
 *
 * macOS is arch-sensitive: electron-builder emits AgentField-<v>-arm64.dmg
 * and AgentField-<v>.dmg (or -x64.dmg). Prefer the exact arch suffix, then a
 * plain .dmg with no arch suffix (a universal build), then any .dmg. The
 * .dmg.blockmap sidecar ends in .blockmap, so `.endsWith('.dmg')` never picks
 * it up.
 */
export function pickInstallerAsset(
  assets: readonly unknown[],
  platform: NodeJS.Platform,
  arch: NodeJS.Architecture = process.arch
): InstallerAsset | null {
  const named: InstallerAsset[] = []
  for (const raw of assets) {
    const asset = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {}
    const name = typeof asset.name === 'string' ? asset.name : null
    const url = typeof asset.browser_download_url === 'string' ? asset.browser_download_url : null
    if (name && url) named.push({ name, url, size: typeof asset.size === 'number' ? asset.size : null })
  }

  if (platform === 'win32') {
    return named.find((a) => /^AgentField-Setup-.+\.exe$/.test(a.name)) ?? null
  }
  if (platform === 'darwin') {
    const dmgs = named.filter((a) => a.name.endsWith('.dmg'))
    const suffix = arch === 'arm64' ? '-arm64.dmg' : '-x64.dmg'
    const exact = dmgs.find((a) => a.name.endsWith(suffix))
    if (exact) return exact
    // No exact arch match: a plain .dmg with no arch suffix is assumed
    // universal; otherwise take whatever .dmg the release does carry.
    const universal = dmgs.find((a) => !/-(arm64|x64)\.dmg$/.test(a.name))
    return universal ?? dmgs[0] ?? null
  }
  return null
}

/** Shape an /releases/latest payload into AppUpdateInfo; null when unusable. */
export function parseLatestRelease(
  payload: unknown,
  platform: NodeJS.Platform,
  arch: NodeJS.Architecture = process.arch
): AppUpdateInfo | null {
  const obj =
    typeof payload === 'object' && payload !== null ? (payload as Record<string, unknown>) : null
  if (!obj || typeof obj.tag_name !== 'string' || obj.tag_name === '') return null
  const tag = obj.tag_name
  const releaseUrl =
    typeof obj.html_url === 'string'
      ? obj.html_url
      : `https://github.com/Agent-Field/agentfield/releases/tag/${tag}`
  const asset = pickInstallerAsset(Array.isArray(obj.assets) ? obj.assets : [], platform, arch)
  return {
    version: tag.replace(/^v/, ''),
    tagName: tag,
    releaseUrl,
    assetName: asset?.name ?? null,
    assetUrl: asset?.url ?? null,
    assetSize: asset?.size ?? null
  }
}

export interface AppUpdaterDeps {
  /** The running app's version (app.getVersion(): release-stamped when packaged). */
  currentVersion: string
  platform: NodeJS.Platform
  /** The app's CPU arch (process.arch) — picks the matching macOS DMG. */
  arch?: NodeJS.Architecture
  /** Where downloads are staged (app.getPath('temp')). */
  tempDir: string
  /** Open a local file with its OS handler (shell.openPath) — mounts the DMG. */
  openPath: (path: string) => Promise<string>
  /** Quit so the Windows installer can replace the app's files. */
  quitForUpdate: () => void
  /** Status pushes for the renderer. Fired on every state change. */
  onStatus?: (status: AppUpdateStatus) => void
  /** Launch the downloaded Windows installer (injectable for tests). */
  launchInstaller?: (path: string) => void
  fetchImpl?: FetchLike
}

function launchWindowsInstaller(file: string): void {
  spawn(file, [], { detached: true, stdio: 'ignore' }).unref()
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export class AppUpdater {
  private readonly deps: AppUpdaterDeps
  private st: AppUpdateStatus
  private autoCheckStarted = false

  constructor(deps: AppUpdaterDeps) {
    this.deps = deps
    this.st = {
      currentVersion: deps.currentVersion,
      checking: false,
      available: null,
      lastCheckedAt: null,
      downloading: false,
      progress: null,
      error: null
    }
  }

  status(): AppUpdateStatus {
    return { ...this.st }
  }

  private patch(p: Partial<AppUpdateStatus>): AppUpdateStatus {
    this.st = { ...this.st, ...p }
    this.deps.onStatus?.(this.status())
    return this.status()
  }

  /** Query /releases/latest. Never rejects — failures land in status.error. */
  async check(): Promise<AppUpdateStatus> {
    if (this.st.checking || this.st.downloading) return this.status()
    this.patch({ checking: true, error: null })
    try {
      const fetchImpl = this.deps.fetchImpl ?? fetch
      const res = await fetchImpl(RELEASES_LATEST_URL, {
        headers: API_HEADERS,
        signal: AbortSignal.timeout(CHECK_TIMEOUT_MS)
      })
      if (res.status === 404) {
        // No stable release exists yet — /releases/latest excludes
        // prereleases and drafts. That is "up to date", not a failure.
        return this.patch({
          checking: false,
          available: null,
          lastCheckedAt: new Date().toISOString()
        })
      }
      if (!res.ok) throw new Error(`GitHub answered ${res.status}`)
      const info = parseLatestRelease(await res.json(), this.deps.platform, this.deps.arch)
      if (!info) throw new Error('unrecognized release payload')
      // Newer AND carrying this platform's installer — a CLI-only release
      // (no desktop assets) is not an app update, whatever its version says.
      const available =
        info.assetUrl !== null &&
        compareAppVersions(info.version, this.deps.currentVersion) > 0
          ? info
          : null
      return this.patch({
        checking: false,
        available,
        lastCheckedAt: new Date().toISOString()
      })
    } catch (err) {
      return this.patch({ checking: false, error: `update check failed: ${errorText(err)}` })
    }
  }

  /**
   * Download the platform installer and hand off to it: Windows launches the
   * NSIS one-click installer and quits (it replaces the app and relaunches);
   * macOS opens the DMG. Never rejects — failures land in status.error.
   */
  async install(): Promise<AppUpdateStatus> {
    const info = this.st.available
    if (!info || this.st.downloading) return this.status()
    // check() only offers releases that carry an installer; belt-and-braces.
    if (!info.assetUrl || !info.assetName) return this.status()
    this.patch({ downloading: true, progress: 0, error: null })
    try {
      const file = await this.download(info.assetUrl, info.assetName, info.assetSize)
      if (this.deps.platform === 'win32') {
        this.patch({ downloading: false, progress: null })
        ;(this.deps.launchInstaller ?? launchWindowsInstaller)(file)
        this.deps.quitForUpdate()
      } else {
        // shell.openPath resolves with an error STRING (empty means success) —
        // it never rejects — so a failure to mount/open the DMG only shows up
        // here. Surface it through the same status.error channel as every
        // other install failure.
        const openError = await this.deps.openPath(file)
        if (openError) throw new Error(openError)
        this.patch({ downloading: false, progress: null })
      }
      return this.status()
    } catch (err) {
      return this.patch({
        downloading: false,
        progress: null,
        error: `update download failed: ${errorText(err)}`
      })
    }
  }

  private async download(url: string, name: string, sizeHint: number | null): Promise<string> {
    const fetchImpl = this.deps.fetchImpl ?? fetch
    const res = await fetchImpl(url, {
      headers: { 'User-Agent': API_HEADERS['User-Agent'] },
      signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS)
    })
    if (!res.ok || !res.body) throw new Error(`asset download answered ${res.status}`)
    const total = Number(res.headers.get('content-length')) || sizeHint || 0
    const dir = await fs.mkdtemp(join(this.deps.tempDir, 'agentfield-update-'))
    const file = join(dir, name)
    let received = 0
    const progress = new Transform({
      transform: (chunk: Buffer, _enc, cb) => {
        received += chunk.length
        if (total > 0) {
          // Cap at 99 until the file is fully flushed to disk.
          const pct = Math.min(99, Math.floor((received / total) * 100))
          if (pct !== this.st.progress) this.patch({ progress: pct })
        }
        cb(null, chunk)
      }
    })
    await pipeline(
      Readable.fromWeb(res.body as unknown as import('node:stream/web').ReadableStream),
      progress,
      createWriteStream(file)
    )
    this.patch({ progress: 100 })
    return file
  }

  /** Check shortly after launch, then on a slow cadence. Packaged apps only —
   *  dev builds report package.json's static version and would always "need"
   *  an update (manual checks from Settings still work there). */
  startAutoCheck(initialDelayMs = 15_000, intervalMs = 4 * 60 * 60_000): void {
    if (this.autoCheckStarted) return
    this.autoCheckStarted = true
    const tick = () => void this.check()
    setTimeout(tick, initialDelayMs)
    setInterval(tick, intervalMs)
  }
}
