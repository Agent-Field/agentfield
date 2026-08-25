import { describe, expect, it, vi } from 'vitest'
import type { AgentActionResult, ControlPlaneStatus } from '../shared/types'
import {
  type ControlPlaneStartDeps,
  SERVER_LABEL,
  planControlPlaneLaunch,
  runAgentAction,
  setAgentPackageAutoUpdate,
  uninstallAgent,
  startControlPlane,
  serverSpawnEnv
} from './agents'
import { CpApiError, type CpClient } from './cpClient'

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
    expect(planControlPlaneLaunch('darwin', true, 8080)).toBe('launchd')
  })

  it('direct-spawns on darwin when the server agent is not loaded', () => {
    expect(planControlPlaneLaunch('darwin', false)).toBe('spawn')
  })

  it('direct-spawns off darwin regardless of the agent flag', () => {
    expect(planControlPlaneLaunch('win32', true)).toBe('spawn')
    expect(planControlPlaneLaunch('linux', true)).toBe('spawn')
  })

  it('direct-spawns for a non-default port — launchd only serves 8080', () => {
    expect(planControlPlaneLaunch('darwin', true, 9091)).toBe('spawn')
  })
})

function managementClient(overrides: Partial<CpClient> = {}): CpClient {
  return {
    startAgent: vi.fn(async (agent_id) => ({ agent_id, status: 'running', pid: 1, port: 2, started_at: '', log_file: '', message: 'started', endpoints: { health: '', reasoners: '', skills: '' } })),
    stopAgent: vi.fn(async (agent_id) => ({ agent_id, status: 'stopped', message: 'stopped' })),
    hasInstallApi: vi.fn(async () => true),
    uninstallPackage: vi.fn(async (package_id) => ({ package_id, status: 'uninstalled' })),
    ...overrides
  } as unknown as CpClient
}

describe('HTTP agent management', () => {
  it.each(['start', 'stop'] as const)('%s returns the existing result shape', async (action) => {
    const result = await runAgentAction(action, 'agent', { cpClient: managementClient() })
    expect(result).toEqual({ ok: true, message: action === 'start' ? 'started' : 'stopped' })
  })

  it('restart stops then starts through the client', async () => {
    const client = managementClient()
    expect(await runAgentAction('restart', 'agent', { cpClient: client })).toEqual({ ok: true, message: 'started' })
    expect(client.stopAgent).toHaveBeenCalledBefore(vi.mocked(client.startAgent))
  })

  it('restart does not start when stopping fails', async () => {
    const client = managementClient({
      stopAgent: vi.fn(async () => {
        throw new CpApiError({ status: 500, message: 'could not stop' })
      })
    })

    expect(await runAgentAction('restart', 'agent', { cpClient: client })).toEqual({
      ok: false,
      message: 'could not stop'
    })
    expect(client.startAgent).not.toHaveBeenCalled()
  })

  it('maps server and unreachable errors', async () => {
    const server = managementClient({ startAgent: vi.fn(async () => { throw new CpApiError({ status: 400, message: 'configuration missing' }) }) })
    expect(await runAgentAction('start', 'agent', { cpClient: server })).toEqual({ ok: false, message: 'configuration missing' })
    const offline = managementClient({ startAgent: vi.fn(async () => { throw new TypeError('fetch failed') }) })
    expect((await runAgentAction('start', 'agent', { cpClient: offline })).message).toMatch(/start the control plane/i)
  })

  it('uninstalls without pre-stopping and detects an old control plane', async () => {
    const client = managementClient()
    expect(await uninstallAgent('agent', { cpClient: client })).toEqual({ ok: true, message: 'uninstalled' })
    expect(client.stopAgent).not.toHaveBeenCalled()
    const old = managementClient({ hasInstallApi: vi.fn(async () => false) })
    expect((await uninstallAgent('agent', { cpClient: old })).message).toMatch(/update AgentField CLI/)
  })

  it('maps a mid-flight uninstall 404 to the old-control-plane message', async () => {
    const client = managementClient({
      uninstallPackage: vi.fn(async () => {
        throw new CpApiError({ status: 404, message: 'route not found' })
      })
    })

    expect(await uninstallAgent('agent', { cpClient: client })).toEqual({
      ok: false,
      message: 'Control plane update required — update AgentField CLI'
    })
  })

  it('maps an offline uninstall to the management fallback message', async () => {
    const client = managementClient({
      uninstallPackage: vi.fn(async () => {
        throw new TypeError('fetch failed')
      })
    })

    expect(await uninstallAgent('agent', { cpClient: client })).toEqual({
      ok: false,
      message: 'Could not reach the control plane — start the control plane and try again'
    })
  })

  it('maps a missing package auto-update endpoint to the old-control-plane message', async () => {
    const client = managementClient({
      setPackageAutoUpdate: vi.fn(async () => {
        throw new CpApiError({ status: 404, message: 'route not found' })
      })
    })

    await expect(setAgentPackageAutoUpdate('agent', false, { cpClient: client }))
      .resolves.toEqual({
        ok: false,
        message: 'Control plane update required — update AgentField CLI'
      })
  })
})

describe('startControlPlane', () => {
  it('kickstarts the launchd agent (and never spawns) when it is loaded', async () => {
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(8080, 30_000, deps)

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

    const result = await startControlPlane(8080, 30_000, deps)

    expect(result).toEqual({ ok: true, message: 'control plane running' })
    expect(spawnServer).toHaveBeenCalledTimes(1)
    // No kickstart attempted.
    expect(run).not.toHaveBeenCalled()
  })

  it('direct-spawns on a custom port even with the launchd agent loaded, and pins the port', async () => {
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const checkHealth = vi.fn(async () => HEALTHY)
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer, checkHealth })

    const result = await startControlPlane(9091, 30_000, deps)

    expect(result.ok).toBe(true)
    expect(run).not.toHaveBeenCalled() // launchd path skipped entirely
    expect(spawnServer).toHaveBeenCalledWith(9091)
    expect(checkHealth).toHaveBeenCalledWith('http://localhost:9091')
  })

  it('never touches launchctl off darwin', async () => {
    const serverAgentLoaded = vi.fn(async () => true)
    const run = vi.fn(async () => ({ code: 0, stdout: '' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ platform: 'win32', serverAgentLoaded, run, spawnServer })

    const result = await startControlPlane(8080, 30_000, deps)

    expect(result.ok).toBe(true)
    expect(serverAgentLoaded).not.toHaveBeenCalled()
    expect(run).not.toHaveBeenCalled()
    expect(spawnServer).toHaveBeenCalledTimes(1)
    expect(spawnServer).toHaveBeenCalledWith(8080)
  })

  it('falls back to a direct spawn when kickstart fails', async () => {
    const run = vi.fn(async () => ({ code: 1, stdout: 'Could not find service' }))
    const spawnServer = vi.fn(() => new Promise<AgentActionResult>(() => {}))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(8080, 30_000, deps)

    expect(result.ok).toBe(true)
    expect(run).toHaveBeenCalledTimes(1) // kickstart attempted once
    expect(spawnServer).toHaveBeenCalledTimes(1) // then fell back
  })

  it('surfaces a spawn error from the fallback path', async () => {
    const run = vi.fn(async () => ({ code: 1, stdout: '' }))
    const spawnServer = vi.fn(async () => ({ ok: false, message: 'af not found on PATH' }))
    const deps = makeDeps({ serverAgentLoaded: async () => true, run, spawnServer })

    const result = await startControlPlane(8080, 30_000, deps)

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

    const result = await startControlPlane(8080, 30_000, deps)

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

    const result = await startControlPlane(8080, 2_000, deps)

    expect(result).toEqual({
      ok: false,
      message: 'control plane did not become healthy in time'
    })
  })
})

describe('serverSpawnEnv', () => {
  it('pins the port and tells the server its own URL for the agents it starts', () => {
    expect(serverSpawnEnv(8080)).toEqual({
      AGENTFIELD_PORT: '8080',
      AGENTFIELD_SERVER: 'http://localhost:8080'
    })
    expect(serverSpawnEnv(18480)).toEqual({
      AGENTFIELD_PORT: '18480',
      AGENTFIELD_SERVER: 'http://localhost:18480'
    })
  })
})
