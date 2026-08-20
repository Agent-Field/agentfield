import { beforeEach, describe, expect, it, vi } from 'vitest'
import { BUNDLED_NODES, bundledEntry, isBundled } from '../shared/bundled'
import { CATALOG, catalogEntry } from '../shared/catalog'
import type { InstallResult } from '../shared/types'
import {
  type BundledDeps,
  type BundledPlanInput,
  bundledStatuses,
  ensureBundledAgents,
  planBundledInstalls,
  resetBundledState
} from './bundledAgents'

const NAMES = BUNDLED_NODES.map((entry) => entry.name)

const baseInput: BundledPlanInput = {
  installed: [],
  provisioned: [],
  skipEnv: undefined,
  cliCommand: '/managed/af',
  cloudActive: false,
  controlPlaneReachable: true,
  registryReadable: true
}

describe('BUNDLED_NODES', () => {
  it('ships the four agent nodes as go nodes sourced at the bare repo', () => {
    expect(NAMES).toEqual(['swe-planner', 'pr-af', 'sec-af', 'cloudsecurity-af'])
    for (const entry of BUNDLED_NODES) {
      expect(entry.name).toMatch(/^[a-z0-9][a-z0-9-]*$/)
      expect(entry.description.length).toBeGreaterThan(0)
      expect(entry.language).toBe('go')
      // The bare repo URL, never the //go selector: the root manifest's
      // superseded_by redirect is what migrates an older install.
      expect(entry.source).toMatch(/^https:\/\/github\.com\/Agent-Field\/[A-Za-z-]+$/)
    }
  })

  it('are not marketplace rows, but stay resolvable for install/update', () => {
    for (const name of NAMES) {
      expect(CATALOG.map((entry) => entry.name)).not.toContain(name)
      expect(catalogEntry(name)).toEqual(bundledEntry(name))
      expect(isBundled(name)).toBe(true)
    }
    expect(isBundled('definitely-not-real')).toBe(false)
    expect(bundledEntry('definitely-not-real')).toBeUndefined()
  })
})

describe('planBundledInstalls', () => {
  it('installs every bundled node on a clean first launch', () => {
    expect(planBundledInstalls(baseInput)).toEqual({
      install: NAMES,
      adopt: [],
      reason: `provisioning bundled nodes: ${NAMES.join(', ')}`
    })
  })

  it('skips only when AGENTFIELD_SKIP_BUNDLED is exactly 1', () => {
    expect(planBundledInstalls({ ...baseInput, skipEnv: '1' })).toEqual({
      install: [],
      adopt: [],
      reason: 'AGENTFIELD_SKIP_BUNDLED=1 — skipping bundled nodes'
    })
    expect(planBundledInstalls({ ...baseInput, skipEnv: '0' }).install).toEqual(NAMES)
    expect(planBundledInstalls({ ...baseInput, skipEnv: '' }).install).toEqual(NAMES)
  })

  it('skips when the CLI is null, empty, or whitespace', () => {
    for (const cliCommand of [null, '', '   ']) {
      expect(planBundledInstalls({ ...baseInput, cliCommand })).toEqual({
        install: [],
        adopt: [],
        reason: 'no usable af CLI — skipping bundled nodes'
      })
    }
  })

  it('skips while the control plane is unavailable', () => {
    expect(planBundledInstalls({ ...baseInput, controlPlaneReachable: false })).toEqual({
      install: [],
      adopt: [],
      reason: 'control plane unavailable — skipping bundled nodes'
    })
  })

  it('applies skip env, missing-CLI, then control-plane precedence', () => {
    expect(
      planBundledInstalls({
        ...baseInput,
        skipEnv: '1',
        cliCommand: null,
        controlPlaneReachable: false
      }).reason
    ).toContain('AGENTFIELD_SKIP_BUNDLED')
    expect(
      planBundledInstalls({ ...baseInput, cliCommand: null, controlPlaneReachable: false }).reason
    ).toBe('no usable af CLI — skipping bundled nodes')
  })

  it('skips cloud control planes before consulting reachability or the registry', () => {
    const plan = planBundledInstalls({
      ...baseInput,
      cloudActive: true,
      controlPlaneReachable: false,
      registryReadable: false
    })
    expect(plan.install).toEqual([])
    expect(plan.adopt).toEqual([])
    expect(plan.reason).toContain('cloud')
  })

  it('skips when the installed-agent registry could not be read', () => {
    const plan = planBundledInstalls({ ...baseInput, registryReadable: false })
    expect(plan.install).toEqual([])
    expect(plan.adopt).toEqual([])
    expect(plan.reason).toContain('registry')
  })

  it('leaves alone nodes already in the registry', () => {
    expect(planBundledInstalls({ ...baseInput, installed: [NAMES[0]] })).toMatchObject({
      adopt: [NAMES[0]],
      install: NAMES.slice(1)
    })
  })

  // Uninstalling a bundled node must stick: it is in provisionedBundled but no
  // longer in the registry, and it must not come back on the next launch.
  it('never re-installs a node already provisioned once', () => {
    expect(planBundledInstalls({ ...baseInput, provisioned: [NAMES[0]] }).install).toEqual(
      NAMES.slice(1)
    )
    expect(planBundledInstalls({ ...baseInput, provisioned: NAMES })).toEqual({
      install: [],
      adopt: [],
      reason: 'bundled nodes already provisioned'
    })
  })

  it('does not reinstall nodes removed after adoption', () => {
    expect(planBundledInstalls({ ...baseInput, provisioned: NAMES })).toMatchObject({
      install: [],
      adopt: []
    })
  })

  it('does not adopt an already-recorded installed node twice', () => {
    expect(
      planBundledInstalls({ ...baseInput, installed: [NAMES[0]], provisioned: [NAMES[0]] })
    ).toMatchObject({ adopt: [], install: NAMES.slice(1) })
  })
})

function fakeDeps(
  results: Record<string, InstallResult> = {}
): BundledDeps & {
  install: ReturnType<typeof vi.fn>
  markProvisioned: ReturnType<typeof vi.fn>
  onInstalled: ReturnType<typeof vi.fn>
  lines: string[]
} {
  const lines: string[] = []
  return {
    install: vi.fn(async (name: string) => results[name] ?? { ok: true, message: `${name} installed` }),
    markProvisioned: vi.fn(async () => {}),
    onInstalled: vi.fn(async () => {}),
    log: (message: string) => lines.push(message),
    lines
  }
}

describe('ensureBundledAgents', () => {
  beforeEach(() => resetBundledState())

  it('installs every planned node in order, sequentially', async () => {
    const order: string[] = []
    const deps = fakeDeps()
    let inFlight = 0
    deps.install.mockImplementation(async (name: string) => {
      // The control-plane install API answers a concurrent install with 409,
      // so the runner must never have two in flight.
      expect(inFlight).toBe(0)
      inFlight += 1
      await Promise.resolve()
      inFlight -= 1
      order.push(name)
      return { ok: true, message: `${name} installed` }
    })

    await ensureBundledAgents(baseInput, deps)

    expect(order).toEqual(NAMES)
    expect(deps.markProvisioned.mock.calls.map((c) => c[0])).toEqual(NAMES)
    expect(deps.onInstalled.mock.calls.map((c) => c[0])).toEqual(NAMES)
    // No phantom rows for nodes the registry now lists.
    expect(bundledStatuses()).toEqual([])
  })

  it('seeds a pending row for every planned node before installing', async () => {
    const deps = fakeDeps()
    const seen: ReturnType<typeof bundledStatuses>[] = []
    deps.install.mockImplementation(async (name: string) => {
      seen.push(bundledStatuses())
      return { ok: true, message: `${name} installed` }
    })

    await ensureBundledAgents(baseInput, deps)

    expect(seen[0].map((s) => s.name)).toEqual(NAMES)
    expect(seen[0][0]).toMatchObject({ phase: 'installing', message: '' })
    expect(seen[0][1]).toMatchObject({ phase: 'pending', message: '' })
    expect(seen[0][0].description).toBe(BUNDLED_NODES[0].description)
    expect(seen[0][0].language).toBe('go')
  })

  it('shows the latest streamed line as the row message while installing', async () => {
    const deps = fakeDeps()
    let mid: ReturnType<typeof bundledStatuses> = []
    deps.install.mockImplementation(async (name: string, onLine: (line: string) => void) => {
      onLine('cloning')
      onLine('building')
      mid = bundledStatuses()
      return { ok: true, message: `${name} installed` }
    })

    await ensureBundledAgents({ ...baseInput, installed: NAMES.slice(1) }, deps)

    expect(mid).toHaveLength(1)
    expect(mid[0]).toMatchObject({ name: NAMES[0], phase: 'installing', message: 'building' })
  })

  it('keeps a failed row for the session and does not mark it provisioned', async () => {
    const deps = fakeDeps({ [NAMES[0]]: { ok: false, message: 'clone failed' } })

    await ensureBundledAgents(baseInput, deps)

    expect(bundledStatuses()).toEqual([
      {
        name: NAMES[0],
        description: BUNDLED_NODES[0].description,
        language: 'go',
        phase: 'failed',
        message: 'clone failed'
      }
    ])
    // The failure must not stop the next node, and must not be recorded —
    // that is what makes the next launch retry it.
    expect(deps.markProvisioned.mock.calls.map((c) => c[0])).toEqual(NAMES.slice(1))
    expect(deps.install).toHaveBeenCalledTimes(NAMES.length)
    expect(deps.lines.some((l) => l.includes('clone failed'))).toBe(true)
  })

  it('treats a rejecting installer as a failed install rather than throwing', async () => {
    const deps = fakeDeps()
    deps.install.mockRejectedValueOnce(new Error('install exploded'))

    await expect(ensureBundledAgents(baseInput, deps)).resolves.toBeUndefined()

    expect(bundledStatuses()).toHaveLength(1)
    expect(bundledStatuses()[0]).toMatchObject({ name: NAMES[0], phase: 'failed' })
    expect(bundledStatuses()[0].message).toContain('install exploded')
  })

  it('still counts an install that persisted or post-install steps could not follow', async () => {
    const deps = fakeDeps()
    deps.markProvisioned.mockRejectedValue(new Error('disk full'))
    deps.onInstalled.mockRejectedValue(new Error('autostart failed'))

    await expect(ensureBundledAgents(baseInput, deps)).resolves.toBeUndefined()

    expect(deps.install).toHaveBeenCalledTimes(NAMES.length)
    expect(bundledStatuses()).toEqual([])
    expect(deps.lines.some((l) => l.includes('disk full'))).toBe(true)
    expect(deps.lines.some((l) => l.includes('autostart failed'))).toBe(true)
  })

  it('does nothing but log when the plan is empty', async () => {
    const deps = fakeDeps()
    await ensureBundledAgents({ ...baseInput, skipEnv: '1' }, deps)
    expect(deps.install).not.toHaveBeenCalled()
    expect(bundledStatuses()).toEqual([])
    expect(deps.lines).toEqual(['bundled: AGENTFIELD_SKIP_BUNDLED=1 — skipping bundled nodes'])
  })

  it('adopts installed nodes without installing or creating status rows', async () => {
    const deps = fakeDeps()
    await ensureBundledAgents({ ...baseInput, installed: NAMES }, deps)

    expect(deps.markProvisioned.mock.calls.map((c) => c[0])).toEqual(NAMES)
    expect(deps.install).not.toHaveBeenCalled()
    expect(bundledStatuses()).toEqual([])
    expect(deps.lines.filter((line) => line.includes('adopted'))).toHaveLength(NAMES.length)
  })

  it('adopts nodes but skips installs when the control plane has no install API', async () => {
    const deps = fakeDeps()
    deps.hasInstallApi = vi.fn(async () => false)

    await ensureBundledAgents({ ...baseInput, installed: [NAMES[0]] }, deps)

    expect(deps.markProvisioned).toHaveBeenCalledWith(NAMES[0])
    expect(deps.install).not.toHaveBeenCalled()
    expect(bundledStatuses()).toEqual([])
    expect(deps.lines.some((line) => line.includes('install API'))).toBe(true)
  })

  it('treats a rejecting install API check as unavailable without throwing', async () => {
    const deps = fakeDeps()
    deps.hasInstallApi = vi.fn(async () => {
      throw new Error('probe failed')
    })

    await expect(ensureBundledAgents(baseInput, deps)).resolves.toBeUndefined()
    expect(deps.install).not.toHaveBeenCalled()
    expect(bundledStatuses()).toEqual([])
  })

  it('works without the optional onInstalled hook', async () => {
    const deps = fakeDeps()
    const { onInstalled: _unused, ...rest } = deps
    await expect(ensureBundledAgents(baseInput, rest)).resolves.toBeUndefined()
    expect(deps.markProvisioned.mock.calls.map((c) => c[0])).toEqual(NAMES)
  })

  it('refuses to run twice concurrently', async () => {
    const deps = fakeDeps()
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0))
    const pending: Array<() => void> = []
    deps.install.mockImplementation(
      (name: string) =>
        new Promise<InstallResult>((resolve) => {
          pending.push(() => resolve({ ok: true, message: `${name} installed` }))
        })
    )

    const first = ensureBundledAgents(baseInput, deps)
    await tick()
    await ensureBundledAgents(baseInput, deps)
    expect(deps.lines).toContain('bundled: provisioning already in progress')
    expect(deps.install).toHaveBeenCalledTimes(1)

    while (pending.length > 0) {
      pending.shift()!()
      await tick()
    }
    await first
    expect(deps.install).toHaveBeenCalledTimes(NAMES.length)
  })

  it('resetBundledState clears rows left by a previous run', async () => {
    const deps = fakeDeps({ [NAMES[0]]: { ok: false, message: 'boom' } })
    await ensureBundledAgents(baseInput, deps)
    expect(bundledStatuses()).toHaveLength(1)
    resetBundledState()
    expect(bundledStatuses()).toEqual([])
  })

  it('hands out copies, so a caller cannot mutate the live rows', async () => {
    const deps = fakeDeps({ [NAMES[0]]: { ok: false, message: 'boom' } })
    await ensureBundledAgents(baseInput, deps)
    bundledStatuses()[0].phase = 'installed'
    expect(bundledStatuses()[0].phase).toBe('failed')
  })
})
