import { describe, expect, it } from 'vitest'
import type { ControlPlaneStatus, DesktopSettings, SnapshotAgent } from '../shared/types'
import {
  autostartAgentPlan,
  controlPlanePortCandidates,
  planControlPlaneBoot,
  type PortProbe
} from './autostart'

function agent(name: string, badge: SnapshotAgent['badge']): SnapshotAgent {
  return {
    name,
    version: '0.1.0',
    description: '',
    status: badge === 'running' ? 'running' : 'stopped',
    path: null,
    port: null,
    pid: null,
    badge
  }
}

function settings(overrides: Partial<DesktopSettings>): DesktopSettings {
  return {
    openAtLogin: false,
    autostartControlPlane: true,
    controlPlanePort: null,
    lastControlPlanePort: null,
    autostartAgents: [],
    installSkills: true,
    trayCompanion: true,
    dismissedUpdateVersion: null,
    ...overrides
  }
}

function probe(port: number, overrides: Partial<ControlPlaneStatus> = {}): PortProbe {
  return {
    port,
    status: { reachable: false, recognized: false, healthy: false, ...overrides }
  }
}

describe('autostartAgentPlan', () => {
  const installed = [agent('a', 'stopped'), agent('b', 'running'), agent('c', 'unknown')]

  it('starts stopped agents and skips running ones', () => {
    expect(autostartAgentPlan(['a', 'b'], installed)).toEqual([{ name: 'a', action: 'start' }])
  })

  it('restarts unknown agents (stale registry after reboot/crash)', () => {
    expect(autostartAgentPlan(['c'], installed)).toEqual([{ name: 'c', action: 'restart' }])
  })

  it('skips selections that are no longer installed', () => {
    expect(autostartAgentPlan(['ghost'], installed)).toEqual([])
  })

  it('preserves selection order', () => {
    expect(autostartAgentPlan(['c', 'a'], installed).map((s) => s.name)).toEqual(['c', 'a'])
  })
})

describe('controlPlanePortCandidates', () => {
  it('probes only the configured port when one is set', () => {
    expect(controlPlanePortCandidates(settings({ controlPlanePort: 9091 }))).toEqual([9091])
    expect(
      controlPlanePortCandidates(settings({ controlPlanePort: 9091, lastControlPlanePort: 8083 }))
    ).toEqual([9091])
  })

  it('probes the default port in automatic mode', () => {
    expect(controlPlanePortCandidates(settings({}))).toEqual([8080])
  })

  it('also probes the last-used port so a restart finds its own server', () => {
    expect(controlPlanePortCandidates(settings({ lastControlPlanePort: 8083 }))).toEqual([
      8080, 8083
    ])
  })

  it('does not probe the default twice when it was also the last-used port', () => {
    expect(controlPlanePortCandidates(settings({ lastControlPlanePort: 8080 }))).toEqual([8080])
  })
})

describe('planControlPlaneBoot', () => {
  it('adopts a recognized control plane, even an unhealthy one', () => {
    expect(
      planControlPlaneBoot(settings({}), [
        probe(8080, { reachable: true, recognized: true, healthy: true })
      ])
    ).toEqual({ kind: 'adopt', port: 8080 })
    expect(
      planControlPlaneBoot(settings({}), [
        probe(8080, { reachable: true, recognized: true, healthy: false })
      ])
    ).toEqual({ kind: 'adopt', port: 8080 })
  })

  it('adopts the last-used port when the default is silent', () => {
    expect(
      planControlPlaneBoot(settings({ lastControlPlanePort: 8083 }), [
        probe(8080),
        probe(8083, { reachable: true, recognized: true, healthy: true })
      ])
    ).toEqual({ kind: 'adopt', port: 8083 })
  })

  it('adopts even when autostart is off — adoption is not starting', () => {
    expect(
      planControlPlaneBoot(settings({ autostartControlPlane: false }), [
        probe(8080, { reachable: true, recognized: true, healthy: true })
      ])
    ).toEqual({ kind: 'adopt', port: 8080 })
  })

  it('never starts when the setting is off', () => {
    expect(planControlPlaneBoot(settings({ autostartControlPlane: false }), [probe(8080)])).toEqual(
      { kind: 'skip', reason: 'autostart is off and no control plane is running' }
    )
  })

  it('starts automatically when nothing answers', () => {
    expect(planControlPlaneBoot(settings({}), [probe(8080)])).toEqual({
      kind: 'start',
      port: null
    })
  })

  it('routes around a foreign service in automatic mode (free-port scan)', () => {
    expect(planControlPlaneBoot(settings({}), [probe(8080, { reachable: true })])).toEqual({
      kind: 'start',
      port: null
    })
  })

  it('starts exactly on a configured port when it is silent', () => {
    expect(planControlPlaneBoot(settings({ controlPlanePort: 9091 }), [probe(9091)])).toEqual({
      kind: 'start',
      port: 9091
    })
  })

  it('never fights a foreign service for a configured port', () => {
    expect(
      planControlPlaneBoot(settings({ controlPlanePort: 9091 }), [
        probe(9091, { reachable: true })
      ])
    ).toEqual({ kind: 'skip', reason: 'port 9091 is in use by another app' })
  })
})
