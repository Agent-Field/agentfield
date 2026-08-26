import { describe, expect, it, vi } from 'vitest'
import { classifyAutoUpdates } from './cloudUpdate'
import * as railwayApiModule from './railwayApi'
import {
  autoUpdateSchedule,
  createRailwayApi,
  ENVIRONMENT_CONFIG_QUERY,
  ENVIRONMENT_PATCH_COMMIT_MUTATION,
  REDEPLOY_MUTATION,
  SERVICE_IMAGE_MUTATION
} from './railwayApi'

function graphqlHarness(data: Record<string, unknown> = { ok: true }) {
  const calls: Array<{ query: string; variables: Record<string, unknown> }> = []
  const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
    calls.push(JSON.parse(String(init?.body)) as (typeof calls)[number])
    return new Response(JSON.stringify({ data }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    })
  }) as typeof fetch
  return { calls, api: createRailwayApi('railway-token', fetchImpl), fetchImpl }
}

describe('Railway service mutations', () => {
  it('updates the service image with the contract document and variables', async () => {
    const { api, calls } = graphqlHarness()
    await api.setServiceImage('service', 'environment', 'agentfield/control-plane-cloud:v0.1.135')

    expect(calls).toEqual([{
      query: SERVICE_IMAGE_MUTATION,
      variables: {
        serviceId: 'service',
        environmentId: 'environment',
        image: 'agentfield/control-plane-cloud:v0.1.135'
      }
    }])
  })

  it('redeploys the exact service instance', async () => {
    const { api, calls } = graphqlHarness()
    await api.redeploy('service', 'environment')

    expect(calls).toEqual([{
      query: REDEPLOY_MUTATION,
      variables: { serviceId: 'service', environmentId: 'environment' }
    }])
  })

  it('surfaces Railway GraphQL errors to the caller', async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({
      errors: [{ message: 'service update denied' }]
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as typeof fetch
    const api = createRailwayApi('railway-token', fetchImpl)

    await expect(api.redeploy('service', 'environment')).rejects.toThrow('service update denied')
  })

  it('G7 — a stalled Railway request times out instead of waiting forever', async () => {
    vi.useFakeTimers()
    const timeout = vi.spyOn(AbortSignal, 'timeout').mockImplementation((milliseconds) => {
      const controller = new AbortController()
      setTimeout(() => controller.abort(new DOMException(
        'The operation was aborted due to timeout',
        'TimeoutError'
      )), milliseconds)
      return controller.signal
    })
    try {
      const fetchImpl = vi.fn((_url: string | URL | Request, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal
          if (!signal) {
            reject(new Error('missing abort signal'))
            return
          }
          const abort = () => reject(signal.reason)
          if (signal.aborted) abort()
          else signal.addEventListener('abort', abort, { once: true })
        })) as typeof fetch
      const api = createRailwayApi('railway-token', fetchImpl)

      const request = api.listWorkspaces()
      const rejected = expect(request).rejects.toThrow('timed out')
      await vi.advanceTimersByTimeAsync(20_000)
      await rejected
    } finally {
      timeout.mockRestore()
      vi.useRealTimers()
    }
  })

  it('H8 — an abort while reading the response body is reported as timed out', async () => {
    vi.useFakeTimers()
    const timeout = vi.spyOn(AbortSignal, 'timeout').mockImplementation((milliseconds) => {
      const controller = new AbortController()
      setTimeout(() => controller.abort(new DOMException(
        'The operation was aborted due to timeout',
        'TimeoutError'
      )), milliseconds)
      return controller.signal
    })
    try {
      const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => ({
        ok: true,
        status: 200,
        json: () => new Promise((_resolve, reject) => {
          const signal = init?.signal
          if (!signal) {
            reject(new Error('missing abort signal'))
            return
          }
          const abort = () => reject(signal.reason)
          if (signal.aborted) abort()
          else signal.addEventListener('abort', abort, { once: true })
        })
      }) as Response) as typeof fetch
      const api = createRailwayApi('railway-token', fetchImpl)

      const request = api.listWorkspaces()
      const rejected = expect(request).rejects.toThrow('timed out')
      await vi.advanceTimersByTimeAsync(20_000)
      await rejected
    } finally {
      timeout.mockRestore()
      vi.useRealTimers()
    }
  })
})

describe('Railway image auto-update policy', () => {
  it('C3 — setting nightly sends one exact environment patch commit', async () => {
    const { api, calls } = graphqlHarness()
    await api.setImageAutoUpdates('environment', 'service', 'nightly')

    expect(calls).toEqual([{
      query: ENVIRONMENT_PATCH_COMMIT_MUTATION,
      variables: {
        environmentId: 'environment',
        patch: {
          services: {
            service: {
              source: {
                autoUpdates: {
                  type: 'patch',
                  schedule: Array.from(
                    { length: 7 },
                    (_, day) => ({ day, startHour: 2, endHour: 6 })
                  )
                }
              }
            }
          }
        },
        commitMessage: 'Configure AgentField image auto-updates (nightly)'
      }
    }])
    expect(JSON.stringify(calls[0].variables.patch)).not.toContain('image')
    expect(calls[0].query).not.toContain('serviceInstanceAutoUpdateScheduleUpdate')
  })

  it('C4 / F10 — setting off bypasses the non-empty schedule builder', async () => {
    const { api, calls } = graphqlHarness()
    await api.setImageAutoUpdates('environment', 'service', 'off')

    expect(calls[0].variables.patch).toEqual({
      services: {
        service: {
          source: { autoUpdates: { type: 'disabled' } }
        }
      }
    })
    expect(JSON.stringify(calls[0].variables.patch)).not.toContain('schedule')
  })

  it('C1 — an existing service without autoUpdates reads as not set', async () => {
    const { api } = graphqlHarness({
      environment: {
        config: {
          services: {
            service: { source: { image: 'agentfield/control-plane-cloud:v0.1.135' } }
          }
        }
      }
    })

    const autoUpdates = await api.getEnvironmentConfigAutoUpdates('environment', 'service')
    expect(autoUpdates).toBeNull()
    expect(classifyAutoUpdates(autoUpdates)).toEqual({ mode: null, policy: null })
  })

  it('F8 — a missing environment configuration is a read failure', async () => {
    const { api } = graphqlHarness({ environment: null })

    await expect(api.getEnvironmentConfigAutoUpdates('environment', 'service'))
      .rejects.toThrow('Railway returned no configuration for this service')

    const missingService = graphqlHarness({
      environment: { config: { services: {} } }
    }).api
    await expect(missingService.getEnvironmentConfigAutoUpdates('environment', 'service'))
      .rejects.toThrow('Railway returned no configuration for this service')
  })

  it('C8 — environment config reads drop extra policy and window keys', async () => {
    const railwayAutoUpdates = {
      type: 'patch',
      schedule: [{ day: 0, startHour: 2, endHour: 6, bar: 'drop-me' }],
      foo: 'drop-me'
    }
    const { api, calls } = graphqlHarness({
      environment: {
        config: {
          services: {
            service: {
              source: { image: 'private/image:v1.2.3', autoUpdates: railwayAutoUpdates },
              variables: { SECRET: 'do-not-forward' }
            }
          }
        }
      }
    })

    await expect(api.getEnvironmentConfigAutoUpdates('environment', 'service'))
      .resolves.toEqual({
        type: 'patch',
        schedule: [{ day: 0, startHour: 2, endHour: 6 }]
      })
    expect(calls).toEqual([{
      query: ENVIRONMENT_CONFIG_QUERY,
      variables: { environmentId: 'environment' }
    }])
  })

  it('F9 — an existing minor policy is preserved in the patch', async () => {
    const { api, calls } = graphqlHarness()
    await api.setImageAutoUpdates('environment', 'service', 'weekends', 'minor')

    expect(calls[0].variables.patch).toEqual({
      services: {
        service: {
          source: {
            autoUpdates: {
              type: 'minor',
              schedule: [
                { day: 6, startHour: 0, endHour: 24 },
                { day: 0, startHour: 0, endHour: 24 }
              ]
            }
          }
        }
      }
    })
  })

  it('C10 — removes the old schedule mutation while retaining the window table', () => {
    expect(railwayApiModule).not.toHaveProperty('AUTO_UPDATE_MUTATION')
    expect(railwayApiModule).not.toHaveProperty('setAutoUpdateSchedule')
    expect(autoUpdateSchedule('weekends')).toEqual([
      { day: 6, startHour: 0, endHour: 24 },
      { day: 0, startHour: 0, endHour: 24 }
    ])
  })
})
