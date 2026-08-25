import { describe, expect, it, vi } from 'vitest'
import {
  AUTO_UPDATE_MUTATION,
  createRailwayApi,
  REDEPLOY_MUTATION,
  SERVICE_IMAGE_MUTATION
} from './railwayApi'

function graphqlHarness() {
  const calls: Array<{ query: string; variables: Record<string, unknown> }> = []
  const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
    calls.push(JSON.parse(String(init?.body)) as (typeof calls)[number])
    return new Response(JSON.stringify({ data: { ok: true } }), {
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
})

describe('Railway image auto-update schedules', () => {
  const expected = {
    off: [],
    nightly: Array.from({ length: 7 }, (_, day) => ({ day, startHour: 2, endHour: 6 })),
    weekends: [
      { day: 6, startHour: 0, endHour: 24 },
      { day: 0, startHour: 0, endHour: 24 }
    ],
    anytime: Array.from({ length: 7 }, (_, day) => ({ day, startHour: 0, endHour: 24 }))
  } as const

  it('declares Railway\'s schedule input as a non-null list of non-null windows', () => {
    expect(AUTO_UPDATE_MUTATION).toContain(
      '$schedule: [AutoUpdateScheduleWindowInput!]!'
    )
  })

  it.each(Object.entries(expected))('sends the exact %s schedule', async (mode, schedule) => {
    const { api, calls } = graphqlHarness()
    await api.setAutoUpdateSchedule(
      'service',
      'environment',
      mode as keyof typeof expected
    )

    expect(calls).toEqual([{
      query: AUTO_UPDATE_MUTATION,
      variables: { serviceId: 'service', environmentId: 'environment', schedule }
    }])
    expect(calls[0].variables).toEqual({
      serviceId: 'service',
      environmentId: 'environment',
      schedule
    })
    expect(calls[0].variables.schedule).toEqual(schedule)
  })
})
