import { describe, expect, it, vi } from 'vitest'
import type { AgentActionResult, ControlPlaneStatus } from '../shared/types'
import {
  type ControlPlaneStartDeps,
  SERVER_LABEL,
  planControlPlaneLaunch,
  startControlPlane
} from './agents'

const HEALTHY: ControlPlaneStatus = {
  reachable: true,
  recognized: true,
  healthy: true
}

/** A deps object whose defaults make startControlPlane resolve on the first
 *  /health poll; individual tests override the pieces they exercise. */
function makeDeps(overrides: Partial<ControlPlaneStartDeps> = {}): ControlPlaneStartDeps {
  return {
    platform: 'darwin',
    uid: () => 501,
    serverAgentLoaded: async () => true,
    run: async () => ({ code: 0, stdout: '' }),
    // A spawn that never errors (stays pending), like the real detached spawn.
    spawnServer: () => new Promise<AgentActionResult>(() => {}),
    checkHealth: async () => HEALTHY,
    now: () => 0,
    delay: async () => {},
    ...overrides
  }
}

describe('planControlPlaneLaunch', () => {
  it('prefers launchd on darwin when the server agent is loaded', () => {
    expect(planControlPlaneLaunch('darwin', true)).toBe('launchd')
  })

  it('direct-spawns on darwin when the server agent is not loaded', () => {
    expect(planControlPlaneLaunch('darwin', false)).toBe('spawn')
  })

  it('direct-spawns off darwin regardless of the agent flag', () => {
    expect(planControlPlaneLaunch('win32', true)).toBe('spawn')
    expect(planControlPlaneLaunch('linux', true)).toBe('spawn')
  })
})

describe('startControlPlane', () => {
  it('kickstarts the launchd agent (and never spawns) when it is loaded', async () => {
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(30_000, deps)

    expect(result).toEqual({ ok: true, message: 'control plane running' })
    expect(run).toHaveBeenCalledWith('launchctl', [
      'kickstart',
      `gui/501/${SERVER_LABEL}`
    ])
    expect(spawnServer).not.toHaveBeenCalled()
  })

  it('direct-spawns when the launchd agent is not loaded', async () => {
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ serverAgentLoaded: async () => false, run, spawnServer })

    const result = await startControlPlane(30_000, deps)

    expect(result).toEqual({ ok: true, message: 'control plane running' })
    expect(spawnServer).toHaveBeenCalledTimes(1)
    // No kickstart attempted.
    expect(run).not.toHaveBeenCalled()
  })

  it('never touches launchctl off darwin', async () => {
    const serverAgentLoaded = vi.fn(async () => true)
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ platform: 'win32', serverAgentLoaded, run, spawnServer })

    const result = await startControlPlane(30_000, deps)

    expect(result.ok).toBe(true)
    expect(serverAgentLoaded).not.toHaveBeenCalled()
    expect(run).not.toHaveBeenCalled()
    expect(spawnServer).toHaveBeenCalledTimes(1)
  })

  it('falls back to a direct spawn when kickstart fails', async () => {
    const run = vi.fn(async () => ({ code: 1, stdout: 'Could not find service' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(30_000, deps)

    expect(result.ok).toBe(true)
    expect(run).toHaveBeenCalledTimes(1) // kickstart attempted once
    expect(spawnServer).toHaveBeenCalledTimes(1) // then fell back
  })

  it('surfaces a spawn error from the fallback path', async () => {
    const run = vi.fn(async () => ({ code: 1, stdout: '' }))
    const spawnServer = vi.fn(async () => ({ ok: false, message: 'af not found on PATH' }))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(30_000, deps)

    expect(result).toEqual({ ok: false, message: 'af not found on PATH' })
  })

  it('reports a foreign service squatting on the port instead of hanging', async () => {
    const deps = makeDeps({
      serverAgentLoaded: async () => false,
      checkHealth: async () => ({
        reachable: true,
        recognized: false,
        healthy: false,
        error: 'another app is using the port'
      })
    })

    const result = await startControlPlane(30_000, deps)

    expect(result).toEqual({ ok: false, message: 'another app is using the port' })
  })

  it('gives up after the deadline when nothing becomes healthy', async () => {
    let clock = 0
    const deps = makeDeps({
      serverAgentLoaded: async () => true,
      checkHealth: async () => ({ reachable: false, recognized: false, healthy: false }),
      now: () => clock,
      delay: async () => {
        clock += 1_000
      }
    })

    const result = await startControlPlane(2_000, deps)

    expect(result).toEqual({
      ok: false,
      message: 'control plane did not become healthy in time'
    })
  })
})
