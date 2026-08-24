import type { CloudAutoUpdateMode } from '../shared/types'

export interface RailwayWorkspace {
  id: string
  name: string
}

export interface RailwayScheduleWindow {
  day: number
  startHour: number
  endHour: number
}

const RAILWAY_GRAPHQL = 'https://backboard.railway.com/graphql/v2'

export const SERVICE_IMAGE_MUTATION = `mutation ServiceInstanceUpdate($serviceId: String!, $environmentId: String!, $image: String!) {
  serviceInstanceUpdate(serviceId: $serviceId, environmentId: $environmentId, input: {source: {image: $image}})
}`

export const REDEPLOY_MUTATION = `mutation ServiceInstanceRedeploy($serviceId: String!, $environmentId: String!) {
  serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
}`

export const AUTO_UPDATE_MUTATION = `mutation ServiceInstanceAutoUpdateScheduleUpdate($serviceId: String!, $environmentId: String!, $schedule: [AutoUpdateScheduleWindowInput!]!) {
  serviceInstanceAutoUpdateScheduleUpdate(serviceId: $serviceId, environmentId: $environmentId, schedule: $schedule)
}`

export function autoUpdateSchedule(mode: CloudAutoUpdateMode): RailwayScheduleWindow[] {
  if (mode === 'off') return []
  if (mode === 'weekends') {
    // Railway uses the conventional 0=Sunday … 6=Saturday numbering.
    return [
      { day: 6, startHour: 0, endHour: 24 },
      { day: 0, startHour: 0, endHour: 24 }
    ]
  }
  const hours = mode === 'nightly'
    ? { startHour: 2, endHour: 6 }
    : { startHour: 0, endHour: 24 }
  return Array.from({ length: 7 }, (_, day) => ({ day, ...hours }))
}

export interface RailwayApi {
  listWorkspaces(): Promise<RailwayWorkspace[]>
  listVolumes(projectId: string): Promise<unknown[]>
  volumeCreate(
    projectId: string,
    environmentId: string,
    serviceId: string,
    mountPath: string
  ): Promise<void>
  setServiceImage(serviceId: string, environmentId: string, image: string): Promise<void>
  redeploy(serviceId: string, environmentId: string): Promise<void>
  setAutoUpdateSchedule(
    serviceId: string,
    environmentId: string,
    mode: CloudAutoUpdateMode
  ): Promise<void>
}

export function createRailwayApi(
  accessToken: string,
  fetchImpl: typeof fetch = fetch
): RailwayApi {
  async function request(
    query: string,
    variables: Record<string, unknown> = {}
  ): Promise<Record<string, unknown>> {
    const response = await fetchImpl(RAILWAY_GRAPHQL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
        'User-Agent': 'agentfield-desktop'
      },
      body: JSON.stringify({ query, variables })
    })
    const payload = await response.json().catch(() => ({})) as {
      data?: Record<string, unknown>
      errors?: Array<{ message?: string }>
    }
    if (!response.ok || payload.errors?.length) {
      const detail = payload.errors
        ?.map((error) => error.message)
        .filter(Boolean)
        .join('; ')
      throw new Error(detail || `Railway GraphQL request failed (${response.status})`)
    }
    if (!payload.data) throw new Error('Railway GraphQL response contained no data')
    return payload.data
  }

  return {
    async listWorkspaces() {
      const data = await request('query { me { workspaces { id name } } }')
      const me = data.me as { workspaces?: RailwayWorkspace[] } | undefined
      return me?.workspaces ?? []
    },
    async listVolumes(projectId) {
      const data = await request(`query Volumes($projectId: String!) {
  project(id: $projectId) { volumes { edges { node { id name } } } }
}`, { projectId })
      const project = data.project as { volumes?: { edges?: unknown[] } } | undefined
      return project?.volumes?.edges ?? []
    },
    async volumeCreate(projectId, environmentId, serviceId, mountPath) {
      await request(`mutation VolumeCreate($projectId: String!, $environmentId: String!, $serviceId: String!, $mountPath: String!) {
  volumeCreate(input: {projectId: $projectId, environmentId: $environmentId, serviceId: $serviceId, mountPath: $mountPath}) { id name }
}`, { projectId, environmentId, serviceId, mountPath })
    },
    async setServiceImage(serviceId, environmentId, image) {
      await request(SERVICE_IMAGE_MUTATION, { serviceId, environmentId, image })
    },
    async redeploy(serviceId, environmentId) {
      await request(REDEPLOY_MUTATION, { serviceId, environmentId })
    },
    async setAutoUpdateSchedule(serviceId, environmentId, mode) {
      await request(AUTO_UPDATE_MUTATION, {
        serviceId,
        environmentId,
        schedule: autoUpdateSchedule(mode)
      })
    }
  }
}
