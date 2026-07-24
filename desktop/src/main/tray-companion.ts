// Provision and install the af-tray menu-bar companion on macOS, so a
// desktop-app-only install gets the menu-bar icon without ever running the
// curl installer. This mirrors what cli.ts does for the af CLI: both installers
// (this app and scripts/install.sh) converge on ~/.agentfield/bin/af-tray, and
// from there af-tray installs its own ~/Applications/AgentField.app + launchd
// agents (see control-plane/cmd/af-tray).
//
// Two moving parts, kept apart so the decision is unit-testable:
//   1. planTray()  — pure: given the observed state (managed binary version,
//      bundled version, whether the launchd agent is loaded), decide whether to
//      re-stage the binary and/or run `af-tray install`.
//   2. ensureTrayCompanion() / removeTrayCompanion() — the effects, driven by
//      injected deps (fs, a command runner, platform, uid) so tests never touch
//      the real filesystem or launchctl.
//
// Why version-stamp comparison (not mtime/size): af-tray carries a real
// `af-tray version` command stamped via ldflags, and bundle-cli.mjs stamps the
// bundled copy identically to the CLI. That is the only honest "is the bundle
// newer?" signal — byte-size/mtime would churn on every rebuild and reinstall
// pointlessly. When the bundle is an unstamped dev build (version unparseable)
// we can't compare, so we only provision when the managed copy is missing and
// otherwise leave what's there — exactly the trust rule cli.ts uses for `dev`.
//
// Why not run `af-tray install` on every launch: install rewrites the launchd
// agents and reloads them (bootout+bootstrap), which restarts/blinks the tray.
// So we only install when we just (re)staged the binary, or when the tray's
// launchd agent is not currently loaded.
//
// No electron imports: the bundled path is injected by main, so this stays
// unit-testable.

import { promises as fs } from 'node:fs'
import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { managedBinDir } from './cli'
import { childEnv } from './env'

/** launchd label af-tray registers for the menu-bar agent (see af-tray shared.go). */
export const TRAY_LABEL = 'ai.agentfield.tray'

const PROBE_TIMEOUT_MS = 5_000

/** Managed location both installers converge on (darwin: no .exe suffix). */
export function managedTrayPath(): string {
  return join(managedBinDir(), 'af-tray')
}

/** Pull the semver out of `af-tray version` ("af-tray 0.1.110 (abc) 2026-…"). */
export function parseTrayVersion(output: string): string | null {
  const match = /af-tray\s+v?([0-9]+(?:\.[0-9]+)+)/.exec(output)
  return match ? match[1] : null
}

/** Numeric dotted-version compare: negative when a < b. (Same rule as cli.ts.) */
export function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (diff !== 0) return diff
  }
  return 0
}

/** Everything planTray needs, gathered by ensureTrayCompanion from the system. */
export interface TrayState {
  /** The managed binary file exists at managedTrayPath(). */
  managedExists: boolean
  /** Parsed `af-tray version` of the managed copy, or null (missing/dev/unrunnable). */
  managedVersion: string | null
  /** Parsed `af-tray version` of the bundled copy, or null (unstamped dev bundle). */
  bundledVersion: string | null
  /** `launchctl print gui/<uid>/ai.agentfield.tray` succeeds — the agent is loaded. */
  agentLoaded: boolean
}

/** What ensureTrayCompanion should do, decided purely from TrayState. */
export interface TrayPlan {
  /** Re-stage the bundled binary into the managed location. */
  provision: boolean
  /** Run `af-tray install` (provisions the .app bundle + launchd, reloads it). */
  install: boolean
  /** Human-readable why, for logging. */
  reason: string
}

/**
 * Decide whether to (re)provision the managed af-tray and whether to run
 * `af-tray install`. Provision when the binary is missing, or when the bundle
 * is stamped and strictly newer than (or supersedes an unverifiable) managed
 * copy — mirroring cli.ts. Install when we just provisioned, or when the tray's
 * launchd agent is not loaded (so a machine that has the binary but never ran
 * install still gets the menu-bar icon). Otherwise do nothing — running install
 * would reload launchd and blink the tray on every launch.
 */
export function planTray(s: TrayState): TrayPlan {
  let provision: boolean
  let reason: string

  if (!s.managedExists) {
    provision = true
    reason = 'managed af-tray missing — provisioning bundled copy'
  } else if (s.bundledVersion === null) {
    // Unstamped dev bundle: can't tell if newer, so keep what's installed.
    provision = false
    reason = 'bundle unstamped (dev) — keeping installed af-tray'
  } else if (s.managedVersion === null) {
    // Present but dev/unrunnable, and the bundle IS stamped: supersede it, so a
    // stale dev copy this app once staged can't win forever (cf. cli.ts).
    provision = true
    reason = 'installed af-tray is a dev/unverifiable build — replacing with stamped bundle'
  } else if (compareVersions(s.bundledVersion, s.managedVersion) > 0) {
    provision = true
    reason = `bundled af-tray v${s.bundledVersion} is newer than installed v${s.managedVersion}`
  } else {
    provision = false
    reason = `installed af-tray v${s.managedVersion} is up to date`
  }

  if (provision) {
    return { provision, install: true, reason }
  }
  if (!s.agentLoaded) {
    return { provision: false, install: true, reason: `${reason}; launchd agent not loaded` }
  }
  return { provision: false, install: false, reason: `${reason}; agent loaded — nothing to do` }
}

// ---- Effects (injected deps so the orchestration is unit-testable) ----------

/** Outcome of a spawned command: exit code and captured stdout. */
export interface RunResult {
  code: number
  stdout: string
}

export interface TrayDeps {
  platform: NodeJS.Platform
  /** launchd gui domain uid — process.getuid() in production. */
  uid: () => number
  /** Whether a path exists (a regular file, for the binary check). */
  fileExists: (path: string) => Promise<boolean>
  /** Stage the bundled binary into the managed location (copy + chmod + atomic rename). */
  stageBinary: (from: string, to: string) => Promise<void>
  /** Run a command to completion; never rejects (resolves code=-1 on spawn error). */
  run: (command: string, args: string[]) => Promise<RunResult>
}

/** Default fileExists: a readable regular file. */
async function realFileExists(path: string): Promise<boolean> {
  try {
    return (await fs.stat(path)).isFile()
  } catch {
    return false
  }
}

/**
 * Copy `from` to `to.new`, mark it executable, then atomically rename over the
 * target — the same replace-a-possibly-running-binary dance installBundledCli
 * uses. Best-effort de-quarantine after, since a downloaded app bundle may pass
 * quarantine to the copy and launchd would refuse to exec it.
 */
async function realStageBinary(from: string, to: string): Promise<void> {
  await fs.mkdir(managedBinDir(), { recursive: true })
  const staged = `${to}.new`
  await fs.copyFile(from, staged)
  await fs.chmod(staged, 0o755)
  try {
    await fs.rename(to, `${to}.old`)
  } catch {
    // target absent — first install
  }
  await fs.rename(staged, to)
  await fs.rm(`${to}.old`, { force: true }).catch(() => {})
}

/** Default runner: spawn with the app's resolved child env; never rejects. */
function realRun(command: string, args: string[]): Promise<RunResult> {
  return new Promise((resolve) => {
    let stdout = ''
    let settled = false
    const done = (code: number) => {
      if (settled) return
      settled = true
      resolve({ code, stdout })
    }
    const child = spawn(command, args, { windowsHide: true, env: childEnv() })
    const timer = setTimeout(() => {
      child.kill()
      done(-1)
    }, PROBE_TIMEOUT_MS)
    child.stdout?.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8')
    })
    child.on('error', () => {
      clearTimeout(timer)
      done(-1)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code ?? -1)
    })
  })
}

export function defaultTrayDeps(): TrayDeps {
  return {
    platform: process.platform,
    uid: () => (typeof process.getuid === 'function' ? process.getuid() : 0),
    fileExists: realFileExists,
    stageBinary: realStageBinary,
    run: realRun
  }
}

/** Probe `<command> version` → parsed af-tray version, or null. */
async function probeTrayVersion(command: string, deps: TrayDeps): Promise<string | null> {
  const res = await deps.run(command, ['version'])
  return res.code === 0 ? parseTrayVersion(res.stdout) : null
}

/** Result of a tray-companion sync, for logging by the caller. */
export interface TrayCompanionResult {
  ok: boolean
  message: string
}

/**
 * Ensure the af-tray companion is provisioned and installed. Darwin-only; a
 * no-op elsewhere (the app has its own in-app tray there). Errors are captured
 * into the result, never thrown — the caller runs this fire-and-forget.
 */
export async function ensureTrayCompanion(
  bundledPath: string | null,
  deps: TrayDeps = defaultTrayDeps()
): Promise<TrayCompanionResult> {
  if (deps.platform !== 'darwin') {
    return { ok: true, message: 'tray companion is macOS-only' }
  }
  if (!bundledPath || !(await deps.fileExists(bundledPath))) {
    return { ok: false, message: 'no bundled af-tray available to provision' }
  }

  const managed = managedTrayPath()
  const [managedExists, bundledVersion] = await Promise.all([
    deps.fileExists(managed),
    probeTrayVersion(bundledPath, deps)
  ])
  const managedVersion = managedExists ? await probeTrayVersion(managed, deps) : null
  const agentLoaded =
    (await deps.run('launchctl', ['print', `gui/${deps.uid()}/${TRAY_LABEL}`])).code === 0

  const plan = planTray({ managedExists, managedVersion, bundledVersion, agentLoaded })

  try {
    if (plan.provision) {
      await deps.stageBinary(bundledPath, managed)
      // Best-effort: clear quarantine so launchd will exec the staged binary.
      await deps.run('xattr', ['-d', 'com.apple.quarantine', managed])
    }
    if (plan.install) {
      const res = await deps.run(managed, ['install'])
      if (res.code !== 0) {
        return { ok: false, message: `af-tray install failed (exit ${res.code}): ${plan.reason}` }
      }
    }
  } catch (err) {
    return { ok: false, message: `tray companion provisioning failed: ${String(err)}` }
  }

  return { ok: true, message: plan.reason }
}

/**
 * Remove the af-tray companion (user toggled it off): delegate to af-tray's own
 * `uninstall` (bootout tray+server, remove plists + ~/Applications/AgentField.app
 * — see launchd_darwin.go). Prefer the managed binary; fall back to the bundled
 * one, which can drive the same uninstall since it operates on external state,
 * not on itself. Darwin-only; never throws.
 */
export async function removeTrayCompanion(
  bundledPath: string | null,
  deps: TrayDeps = defaultTrayDeps()
): Promise<TrayCompanionResult> {
  if (deps.platform !== 'darwin') {
    return { ok: true, message: 'tray companion is macOS-only' }
  }
  const managed = managedTrayPath()
  const command = (await deps.fileExists(managed)) ? managed : bundledPath
  if (!command) {
    return { ok: false, message: 'no af-tray binary available to run uninstall' }
  }
  const res = await deps.run(command, ['uninstall'])
  return res.code === 0
    ? { ok: true, message: 'af-tray companion uninstalled' }
    : { ok: false, message: `af-tray uninstall failed (exit ${res.code})` }
}

/**
 * Single entry the app calls: install when enabled, uninstall when not.
 * Fire-and-forget from index.ts (darwin only).
 */
export async function syncTrayCompanion(
  enabled: boolean,
  bundledPath: string | null,
  deps: TrayDeps = defaultTrayDeps()
): Promise<TrayCompanionResult> {
  return enabled ? ensureTrayCompanion(bundledPath, deps) : removeTrayCompanion(bundledPath, deps)
}
