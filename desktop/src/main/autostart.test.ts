import { describe, expect, it } from 'vitest'
import type { AgentFieldSnapshot, DesktopSettings, SnapshotAgent } from '../shared/types'
import { autostartAgentPlan, shouldStartControlPlane } from './autostart'

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
    autostartAgents: [],
    installSkills: true,
    trayCompanion: true,
    dismissedUpdateVersion: null,
    ...overrides
  }
}

function cp(overrides: Partial<AgentFieldSnapshot['controlPlane']>): AgentFieldSnapshot['controlPlane'] {
  return {
    baseUrl: 'http://localhost:8080',
    reachable: false,
    recognized: false,
    healthy: false,
    ...overrides
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

describe('shouldStartControlPlane', () => {
  it('starts only when enabled and nothing answers', () => {
    expect(shouldStartControlPlane(settings({}), cp({}))).toBe(true)
  })

  it('never starts when the setting is off', () => {
    expect(shouldStartControlPlane(settings({ autostartControlPlane: false }), cp({}))).toBe(false)
  })

  it('leaves a live control plane alone, even an unhealthy one', () => {
    expect(
      shouldStartControlPlane(
        settings({}),
        cp({ reachable: true, recognized: true, healthy: true })
      )
    ).toBe(false)
    expect(
      shouldStartControlPlane(settings({}), cp({ reachable: true, recognized: true }))
    ).toBe(false)
  })

  it('does not fight a foreign service for the port', () => {
    expect(shouldStartControlPlane(settings({}), cp({ reachable: true }))).toBe(false)
  })
})
