import { describe, expect, it, vi } from 'vitest'
import {
  type RunResult,
  type TrayDeps,
  type TrayState,
  compareVersions,
  ensureTrayCompanion,
  managedTrayPath,
  parseTrayVersion,
  planTray,
  removeTrayCompanion,
  syncTrayCompanion
} from './tray-companion'

const BUNDLED = '/app/vendor/af-tray'
const MANAGED = managedTrayPath()

describe('parseTrayVersion', () => {
  it('reads the version out of `af-tray version` output', () => {
    expect(parseTrayVersion('af-tray 0.1.110 (abc123) 2026-07-15T00:00:00Z')).toBe('0.1.110')
  })

  it('strips a pre-release suffix like the CLI parser', () => {
    expect(parseTrayVersion('af-tray 0.1.109-rc.2-55-gf815 (f815) 2026-07-15')).toBe('0.1.109')
  })

  it('returns null for dev/unparseable output', () => {
    expect(parseTrayVersion('af-tray dev (none) unknown')).toBeNull()
    expect(parseTrayVersion('command not found')).toBeNull()
    expect(parseTrayVersion('')).toBeNull()
  })
})

describe('compareVersions', () => {
  it('orders numerically per segment', () => {
    expect(compareVersions('0.1.110', '0.1.109')).toBeGreaterThan(0)
    expect(compareVersions('0.1.109', '0.1.109')).toBe(0)
    expect(compareVersions('0.1.9', '0.1.110')).toBeLessThan(0)
  })
})

describe('planTray', () => {
  const base: TrayState = {
    managedExists: true,
    managedVersion: '0.1.110',
    bundledVersion: '0.1.110',
    agentLoaded: true
  }

  it('provisions and installs when the managed binary is missing', () => {
    const p = planTray({ ...base, managedExists: false, managedVersion: null })
    expect(p).toMatchObject({ provision: true, install: true })
  })

  it('does nothing when up to date and the agent is loaded', () => {
    const p = planTray(base)
    expect(p).toMatchObject({ provision: false, install: false })
  })

  it('installs (only) when up to date but the agent is not loaded', () => {
    const p = planTray({ ...base, agentLoaded: false })
    expect(p).toMatchObject({ provision: false, install: true })
  })

  it('provisions and installs when the bundle is newer', () => {
    const p = planTray({ ...base, bundledVersion: '0.1.120', managedVersion: '0.1.110' })
    expect(p).toMatchObject({ provision: true, install: true })
  })

  it('keeps the installed copy when the bundle is an unstamped dev build', () => {
    const p = planTray({ ...base, bundledVersion: null })
    expect(p).toMatchObject({ provision: false, install: false })
  })

  it('supersedes a dev/unverifiable managed copy when the bundle is stamped', () => {
    const p = planTray({ ...base, managedVersion: null, bundledVersion: '0.1.110' })
    expect(p).toMatchObject({ provision: true, install: true })
  })
})

// ---- Orchestration with injected deps ---------------------------------------

interface FakeConfig {
  platform?: NodeJS.Platform
  existing?: Set<string>
  /** command -> `<cmd> version` stdout */
  versions?: Record<string, string>
  agentLoaded?: boolean
  /** commands whose non-version invocation should fail (code 1) */
  failCommands?: Set<string>
}

function fakeDeps(cfg: FakeConfig = {}) {
  const calls: Array<{ command: string; args: string[] }> = []
  const staged: Array<{ from: string; to: string }> = []
  const existing = cfg.existing ?? new Set<string>([BUNDLED])

  const run = vi.fn(async (command: string, args: string[]): Promise<RunResult> => {
    calls.push({ command, args })
    if (args[0] === 'version') {
      const out = cfg.versions?.[command]
      return { code: out ? 0 : 1, stdout: out ?? '' }
    }
    if (command === 'launchctl' && args[0] === 'print') {
      return { code: cfg.agentLoaded ? 0 : 1, stdout: '' }
    }
    if (cfg.failCommands?.has(command)) return { code: 1, stdout: '' }
    return { code: 0, stdout: '' }
  })

  const deps: TrayDeps = {
    platform: cfg.platform ?? 'darwin',
    uid: () => 501,
    fileExists: vi.fn(async (path: string) => existing.has(path)),
    stageBinary: vi.fn(async (from: string, to: string) => {
      staged.push({ from, to })
      existing.add(to)
    }),
    run
  }
  return { deps, calls, staged, run }
}

const ran = (calls: Array<{ command: string; args: string[] }>, command: string, verb: string) =>
  calls.some((c) => c.command === command && c.args[0] === verb)

describe('ensureTrayCompanion', () => {
  it('is a no-op off macOS', async () => {
    const { deps, calls } = fakeDeps({ platform: 'win32' })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(calls).toHaveLength(0)
  })

  it('fails cleanly when no bundled binary is present', async () => {
    const { deps } = fakeDeps({ existing: new Set() })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(false)
  })

  it('provisions and installs on a fresh machine (managed missing)', async () => {
    const { deps, calls, staged } = fakeDeps({
      existing: new Set([BUNDLED]),
      versions: { [BUNDLED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: false
    })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(staged).toEqual([{ from: BUNDLED, to: MANAGED }])
    expect(ran(calls, MANAGED, 'install')).toBe(true)
  })

  it('does nothing when up to date and the agent is loaded', async () => {
    const { deps, calls, staged } = fakeDeps({
      existing: new Set([BUNDLED, MANAGED]),
      versions: { [BUNDLED]: 'af-tray 0.1.110 (a) 2026', [MANAGED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: true
    })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(staged).toHaveLength(0)
    expect(ran(calls, MANAGED, 'install')).toBe(false)
  })

  it('installs without re-staging when present but the agent is not loaded', async () => {
    const { deps, calls, staged } = fakeDeps({
      existing: new Set([BUNDLED, MANAGED]),
      versions: { [BUNDLED]: 'af-tray 0.1.110 (a) 2026', [MANAGED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: false
    })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(staged).toHaveLength(0)
    expect(ran(calls, MANAGED, 'install')).toBe(true)
  })

  it('re-stages when the bundle is newer than the installed copy', async () => {
    const { deps, staged } = fakeDeps({
      existing: new Set([BUNDLED, MANAGED]),
      versions: { [BUNDLED]: 'af-tray 0.2.0 (a) 2026', [MANAGED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: true
    })
    await ensureTrayCompanion(BUNDLED, deps)
    expect(staged).toEqual([{ from: BUNDLED, to: MANAGED }])
  })

  it('reports failure when `af-tray install` exits non-zero', async () => {
    const { deps } = fakeDeps({
      existing: new Set([BUNDLED]),
      versions: { [BUNDLED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: false,
      failCommands: new Set([MANAGED])
    })
    const res = await ensureTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(false)
  })
})

describe('removeTrayCompanion', () => {
  it('runs `af-tray uninstall` via the managed binary when present', async () => {
    const { deps, calls } = fakeDeps({ existing: new Set([BUNDLED, MANAGED]) })
    const res = await removeTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(ran(calls, MANAGED, 'uninstall')).toBe(true)
  })

  it('falls back to the bundled binary when the managed one is gone', async () => {
    const { deps, calls } = fakeDeps({ existing: new Set([BUNDLED]) })
    const res = await removeTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(ran(calls, BUNDLED, 'uninstall')).toBe(true)
  })

  it('is a no-op off macOS', async () => {
    const { deps, calls } = fakeDeps({ platform: 'linux', existing: new Set([BUNDLED, MANAGED]) })
    const res = await removeTrayCompanion(BUNDLED, deps)
    expect(res.ok).toBe(true)
    expect(calls).toHaveLength(0)
  })
})

describe('syncTrayCompanion', () => {
  it('installs when enabled and uninstalls when disabled', async () => {
    const on = fakeDeps({
      existing: new Set([BUNDLED]),
      versions: { [BUNDLED]: 'af-tray 0.1.110 (a) 2026' },
      agentLoaded: false
    })
    await syncTrayCompanion(true, BUNDLED, on.deps)
    expect(ran(on.calls, MANAGED, 'install')).toBe(true)

    const off = fakeDeps({ existing: new Set([BUNDLED, MANAGED]) })
    await syncTrayCompanion(false, BUNDLED, off.deps)
    expect(ran(off.calls, MANAGED, 'uninstall')).toBe(true)
  })
})
