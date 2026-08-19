import { describe, expect, it } from 'vitest'
import { canDecideDefaultRoute, controlPlaneStatus, defaultView, shouldRerouteToBundled } from './App'
import type { AgentFieldSnapshot } from '../../shared/types'

describe('defaultView', () => {
  it('lands on the Agents library while bundled nodes are provisioning', () => {
    // First launch: the two bundled rows are the content, so add-mode would
    // hide exactly what the user should be watching.
    expect(defaultView(2, 0)).toBe('agents')
    // Even with a stocked library, an arriving node still wins over Home.
    expect(defaultView(1, 3)).toBe('agents')
  })

  it('opens add-mode only when nothing is installed and nothing is arriving', () => {
    expect(defaultView(0, 0)).toBe('install')
  })

  it('opens Home once the library has agents', () => {
    expect(defaultView(0, 1)).toBe('home')
  })
})

describe('canDecideDefaultRoute', () => {
  it('waits while the registry is unreadable and nothing is provisioning', () => {
    expect(
      canDecideDefaultRoute({ registryExists: false, registryError: 'down', bundledCount: 0 })
    ).toBe(false)
    expect(
      canDecideDefaultRoute({ registryExists: false, registryError: undefined, bundledCount: 0 })
    ).toBe(false)
  })

  it('decides once the registry reads cleanly, even when empty', () => {
    expect(
      canDecideDefaultRoute({ registryExists: true, registryError: undefined, bundledCount: 0 })
    ).toBe(true)
  })

  it('decides as soon as provisioning rows exist', () => {
    expect(
      canDecideDefaultRoute({ registryExists: false, registryError: 'down', bundledCount: 2 })
    ).toBe(true)
  })
})

describe('shouldRerouteToBundled', () => {
  const base = {
    view: 'install' as const,
    bundledCount: 1,
    deepLinkHandled: false,
    userNavigated: false,
    alreadyRerouted: false
  }

  it('reroutes an untouched add-mode launch when bundled rows arrive', () => {
    expect(shouldRerouteToBundled(base)).toBe(true)
  })

  it.each([
    { ...base, view: 'agents' as const },
    { ...base, view: 'home' as const },
    { ...base, bundledCount: 0 },
    { ...base, deepLinkHandled: true },
    { ...base, userNavigated: true },
    { ...base, alreadyRerouted: true }
  ])('does not reroute after another routing decision: $view', (args) => {
    expect(shouldRerouteToBundled(args)).toBe(false)
  })
})

describe('controlPlaneStatus', () => {
  const base = (cp: Partial<AgentFieldSnapshot['controlPlane']>) =>
    ({
      controlPlane: {
        baseUrl: 'http://127.0.0.1:8000',
        reachable: false,
        recognized: false,
        healthy: false,
        ...cp
      }
    }) as AgentFieldSnapshot

  it('is gray until the first snapshot arrives', () => {
    expect(controlPlaneStatus(null).tone).toBe('gray')
  })

  it('reports green when healthy and red when unreachable', () => {
    expect(controlPlaneStatus(base({ reachable: true, recognized: true, healthy: true }))).toEqual({
      tone: 'green',
      label: 'Running'
    })
    expect(controlPlaneStatus(base({})).tone).toBe('red')
  })

  it('separates an unhealthy AgentField from a stranger on the port', () => {
    expect(controlPlaneStatus(base({ reachable: true, recognized: true })).label).toBe('Unhealthy')
    expect(controlPlaneStatus(base({ reachable: true })).label).toBe('Port in use')
  })
})
