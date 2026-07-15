// Which `af` does the app use? Non-technical users install ONLY the desktop
// app — it carries a bundled af CLI so everything works out of the box. Users
// who installed AgentField themselves (curl, dev builds) keep their copy, as
// long as it meets the version this app's features need.
//
// Resolution order (first usable wins):
//   1. managed  — ~/.agentfield/bin (where the curl installer puts it, and
//                 where this app provisions its bundled copy: the SHARED
//                 location both installers converge on)
//   2. PATH     — a developer's own `af`
//   3. bundled  — resources/bin inside the app package
//
// A copy that answers with a semver older than the effective minimum is
// skipped (the UI then offers "Update AgentField", which installs the bundled
// copy into the managed location — never over a newer one). The effective
// minimum is the bundle's own stamped version when it has one — the app needs
// at least the features it shipped with — falling back to MIN_AF_VERSION.
//
// Unparseable versions ("Version: dev") are trusted only while the bundle is
// unstamped too (source builds keep their dev workflow). A release app
// carries a stamped bundle, and a dev-versioned managed/PATH copy is then
// treated as superseded — otherwise a stale dev binary this app once
// provisioned would win forever and no update could ever be offered.
//
// No electron imports: the bundled path is injected by main, so probing,
// selection, and provisioning stay unit-testable.

import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import type { AgentActionResult, CliStatus } from '../shared/types'
import { getAgentFieldHome } from './agentfield'
import { childEnv } from './env'

/** Oldest af this app can drive (needs `af run/stop <name>`, `af skill`). */
export const MIN_AF_VERSION = '0.1.107'

const PROBE_TIMEOUT_MS = 5_000

export type CliSource = 'managed' | 'path' | 'bundled'

export interface CliCandidate {
  command: string
  source: CliSource
}

export interface ProbedCandidate extends CliCandidate {
  /** The command exists and `<cmd> version` exited 0. */
  responds: boolean
  /** Parsed semver like "0.1.107", or null (dev builds, unparseable output). */
  version: string | null
}

function exeName(base: string): string {
  return process.platform === 'win32' ? `${base}.exe` : base
}

export function managedBinDir(): string {
  return join(getAgentFieldHome(), 'bin')
}

/** Candidate list in resolution priority order. bundledPath may not exist. */
export function cliCandidates(bundledPath: string | null): CliCandidate[] {
  const candidates: CliCandidate[] = [
    { command: join(managedBinDir(), exeName('af')), source: 'managed' },
    { command: join(managedBinDir(), exeName('agentfield')), source: 'managed' },
    { command: 'af', source: 'path' }
  ]
  if (bundledPath) candidates.push({ command: bundledPath, source: 'bundled' })
  return candidates
}

/** Pull the semver out of `af version` output ("  Version:    v0.1.107"). */
export function parseAfVersion(output: string): string | null {
  const match = /Version:\s*v?([0-9]+(?:\.[0-9]+)+)/.exec(output)
  return match ? match[1] : null
}

/** Numeric dotted-version compare: negative when a < b. */
export function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (diff !== 0) return diff
  }
  return 0
}

/**
 * The version floor for candidate selection: the bundle's own stamped
 * version when it has one (the catalog and app features are built against
 * exactly that CLI), MIN_AF_VERSION otherwise (unstamped dev bundles).
 */
export function effectiveMinVersion(probed: readonly ProbedCandidate[]): string {
  const bundled = probed.find((c) => c.source === 'bundled' && c.responds)
  if (bundled?.version && compareVersions(bundled.version, MIN_AF_VERSION) > 0) {
    return bundled.version
  }
  return MIN_AF_VERSION
}

/**
 * Pick the CLI to use. Returns the first responding candidate that is not
 * too old, plus the best-ranked copy that WAS skipped — that one drives the
 * "Update AgentField" banner. Unparseable ("dev") versions are trusted only
 * while the bundle is unstamped too; against a stamped bundle they are
 * superseded (see the header comment).
 */
export function selectCli(
  probed: readonly ProbedCandidate[],
  minVersion: string = MIN_AF_VERSION
): { chosen: ProbedCandidate | null; outdated: ProbedCandidate | null } {
  const bundledVersion = probed.find((c) => c.source === 'bundled' && c.responds)?.version ?? null
  let outdated: ProbedCandidate | null = null
  for (const candidate of probed) {
    if (!candidate.responds) continue
    const superseded =
      candidate.version === null && candidate.source !== 'bundled' && bundledVersion !== null
    const tooOld =
      candidate.version !== null && compareVersions(candidate.version, minVersion) < 0
    if (tooOld || superseded) {
      if (!outdated) outdated = candidate
      continue
    }
    return { chosen: candidate, outdated }
  }
  return { chosen: null, outdated }
}

/** Run `<command> version` and parse the answer. Never rejects. */
export function probeCli(candidate: CliCandidate): Promise<ProbedCandidate> {
  return new Promise((resolve) => {
    let output = ''
    let settled = false
    const done = (responds: boolean) => {
      if (settled) return
      settled = true
      resolve({ ...candidate, responds, version: responds ? parseAfVersion(output) : null })
    }

    const child = spawn(candidate.command, ['version'], { windowsHide: true, env: childEnv() })
    const timer = setTimeout(() => {
      child.kill()
      done(false)
    }, PROBE_TIMEOUT_MS)
    child.stdout.on('data', (chunk: Buffer) => {
      output += chunk.toString('utf8')
    })
    child.on('error', () => {
      clearTimeout(timer)
      done(false)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code === 0)
    })
  })
}

// ---- Active command ----------------------------------------------------------
// agents.ts / installer.ts spawn whatever this points at. It starts as bare
// 'af' (PATH) so nothing breaks before initializeCli() has run.

let activeCommand = 'af'

export function getCliCommand(): string {
  return activeCommand
}

function buildStatus(
  chosen: ProbedCandidate | null,
  outdated: ProbedCandidate | null,
  bundled: ProbedCandidate | null,
  minVersion: string
): CliStatus {
  return {
    command: chosen?.command ?? null,
    source: chosen?.source ?? null,
    version: chosen?.version ?? null,
    minVersion,
    outdated:
      outdated && outdated.version
        ? { source: outdated.source, version: outdated.version }
        : null,
    bundledAvailable: bundled?.responds ?? false,
    bundledVersion: bundled?.version ?? null
  }
}

/**
 * Probe everything, pick the CLI, and — when nothing usable exists outside
 * the app package (fresh machine, or every ambient copy too old) — provision
 * the bundled copy into ~/.agentfield/bin so terminals and coding agents get
 * a stable af too, then re-resolve.
 */
export async function initializeCli(bundledPath: string | null): Promise<CliStatus> {
  const probeAll = async () => Promise.all(cliCandidates(bundledPath).map(probeCli))

  let probed = await probeAll()
  let minVersion = effectiveMinVersion(probed)
  let { chosen, outdated } = selectCli(probed, minVersion)
  const bundled = probed.find((c) => c.source === 'bundled') ?? null

  if ((chosen === null || chosen.source === 'bundled') && bundled?.responds) {
    const installed = await installBundledCli(bundledPath as string)
    if (installed.ok) {
      probed = await probeAll()
      minVersion = effectiveMinVersion(probed)
      ;({ chosen, outdated } = selectCli(probed, minVersion))
    }
  }

  if (chosen) activeCommand = chosen.command
  return buildStatus(chosen, outdated, bundled, minVersion)
}

/** Re-resolve without side effects (used after an explicit update). */
export async function refreshCliStatus(bundledPath: string | null): Promise<CliStatus> {
  const probed = await Promise.all(cliCandidates(bundledPath).map(probeCli))
  const minVersion = effectiveMinVersion(probed)
  const { chosen, outdated } = selectCli(probed, minVersion)
  if (chosen) activeCommand = chosen.command
  return buildStatus(
    chosen,
    outdated,
    probed.find((c) => c.source === 'bundled') ?? null,
    minVersion
  )
}

/**
 * Install the bundled af into the managed location (both names the curl
 * installer uses: `agentfield` plus the `af` alias). Copes with a running
 * binary on Windows by renaming it aside first — renaming a running exe is
 * allowed, overwriting it is not.
 */
export async function installBundledCli(bundledPath: string): Promise<AgentActionResult> {
  const binDir = managedBinDir()
  try {
    await fs.mkdir(binDir, { recursive: true })
    for (const base of ['agentfield', 'af']) {
      const target = join(binDir, exeName(base))
      const staged = `${target}.new`
      await fs.copyFile(bundledPath, staged)
      if (process.platform !== 'win32') await fs.chmod(staged, 0o755)
      try {
        await fs.rename(target, `${target}.old`)
      } catch {
        // target absent — first install
      }
      await fs.rename(staged, target)
      await fs.rm(`${target}.old`, { force: true }).catch(() => {})
    }
  } catch (err) {
    return { ok: false, message: `could not install the AgentField CLI: ${String(err)}` }
  }

  if (process.platform === 'win32') registerWindowsUserPath(binDir)
  return { ok: true, message: `AgentField CLI installed to ${binDir}` }
}

/**
 * Best-effort: put ~/.agentfield/bin on the user PATH so `af` also works in
 * terminals (and for coding agents running shell commands). User-scope only,
 * idempotent, and never blocks the caller. macOS/Linux PATH setup remains the
 * curl installer's job — editing shell profiles from a GUI app is too rude.
 */
function registerWindowsUserPath(binDir: string): void {
  const script =
    `$p = [Environment]::GetEnvironmentVariable('Path', 'User'); ` +
    `if (($p -split ';') -notcontains '${binDir}') { ` +
    `[Environment]::SetEnvironmentVariable('Path', "$p;${binDir}", 'User') }`
  spawn('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], {
    windowsHide: true,
    stdio: 'ignore'
  }).on('error', () => {})
}
