import { describe, expect, it } from 'vitest'
import { controlPlaneStatus, defaultView } from './App'
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
