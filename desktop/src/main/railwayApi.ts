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

export type RailwayAutoUpdatePolicy = 'disabled' | 'patch' | 'minor'
export type RailwayEnabledAutoUpdatePolicy = Exclude<RailwayAutoUpdatePolicy, 'disabled'>

/** Sanitized subset of Railway's untyped environment config. No other source
 * or service config fields are allowed to leave this module. */
export interface RailwayImageAutoUpdates {
  type: unknown
  schedule?: Array<{
    day: unknown
    startHour: unknown
    endHour: unknown
  }>
}

const RAILWAY_GRAPHQL = 'https://backboard.railway.com/graphql/v2'
const RAILWAY_REQUEST_TIMEOUT_MS = 20_000

export const SERVICE_IMAGE_MUTATION = `mutation ServiceInstanceUpdate($serviceId: String!, $environmentId: String!, $image: String!) {
  serviceInstanceUpdate(serviceId: $serviceId, environmentId: $environmentId, input: {source: {image: $image}})
}`

export const REDEPLOY_MUTATION = `mutation ServiceInstanceRedeploy($serviceId: String!, $environmentId: String!) {
  serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
}`

export const ENVIRONMENT_CONFIG_QUERY = `query EnvironmentConfig($environmentId: String!) {
  environment(id: $environmentId) { config }
}`

export const ENVIRONMENT_PATCH_COMMIT_MUTATION = `mutation EnvironmentPatchCommit($environmentId: String!, $patch: EnvironmentConfig!, $commitMessage: String) {
  environmentPatchCommit(environmentId: $environmentId, patch: $patch, commitMessage: $commitMessage)
}`

export function autoUpdateSchedule(
  mode: Exclude<CloudAutoUpdateMode, 'off'>
): RailwayScheduleWindow[] {
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

export function imageAutoUpdatesPatch(
  serviceId: string,
  mode: CloudAutoUpdateMode,
  policy: RailwayEnabledAutoUpdatePolicy = 'patch'
): Record<string, unknown> {
  const autoUpdates = mode === 'off'
    ? { type: 'disabled' }
    : { type: policy, schedule: autoUpdateSchedule(mode) }
  return {
    services: {
      [serviceId]: {
        source: { autoUpdates }
      }
    }
  }
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
  getEnvironmentConfigAutoUpdates(
    environmentId: string,
    serviceId: string
  ): Promise<RailwayImageAutoUpdates | null>
  setImageAutoUpdates(
    environmentId: string,
    serviceId: string,
    mode: CloudAutoUpdateMode,
    policy?: RailwayEnabledAutoUpdatePolicy
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
    const signal = AbortSignal.timeout(RAILWAY_REQUEST_TIMEOUT_MS)
    let response: Response
    let payload: {
      data?: Record<string, unknown>
      errors?: Array<{ message?: string }>
    }
    try {
      response = await fetchImpl(RAILWAY_GRAPHQL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
          'User-Agent': 'agentfield-desktop'
        },
        body: JSON.stringify({ query, variables }),
        signal
      })
      payload = await response.json().catch((error) => {
        if (signal.aborted) throw error
        return {}
      }) as typeof payload
    } catch (error) {
      if (signal.aborted) {
        throw new Error(`Railway GraphQL request timed out after ${RAILWAY_REQUEST_TIMEOUT_MS / 1000} seconds`)
      }
      throw error
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
    async getEnvironmentConfigAutoUpdates(environmentId, serviceId) {
      const data = await request(ENVIRONMENT_CONFIG_QUERY, { environmentId })
      const environment = data.environment
      if (typeof environment !== 'object' || environment === null) {
        throw new Error('Railway returned no configuration for this service')
      }
      const config = (environment as { config?: unknown }).config
      if (typeof config !== 'object' || config === null) {
        throw new Error('Railway returned no configuration for this service')
      }
      const services = (config as { services?: unknown }).services
      if (typeof services !== 'object' || services === null) {
        throw new Error('Railway returned no configuration for this service')
      }
      const service = (services as Record<string, unknown>)[serviceId]
      if (typeof service !== 'object' || service === null) {
        throw new Error('Railway returned no configuration for this service')
      }
      const source = (service as { source?: unknown }).source
      if (typeof source !== 'object' || source === null) return null
      const value = (source as { autoUpdates?: unknown }).autoUpdates
      if (value === undefined || value === null) return null
      if (typeof value !== 'object') {
        throw new Error('Railway returned an invalid auto-update configuration')
      }

      const raw = value as { type?: unknown; schedule?: unknown }
      const autoUpdates: RailwayImageAutoUpdates = { type: raw.type }
      if (Array.isArray(raw.schedule)) {
        autoUpdates.schedule = raw.schedule.map((window) => {
          const item = typeof window === 'object' && window !== null
            ? window as Record<string, unknown>
            : {}
          return {
            day: item.day,
            startHour: item.startHour,
            endHour: item.endHour
          }
        })
      }
      return autoUpdates
    },
    async setImageAutoUpdates(environmentId, serviceId, mode, policy = 'patch') {
      await request(ENVIRONMENT_PATCH_COMMIT_MUTATION, {
        environmentId,
        patch: imageAutoUpdatesPatch(serviceId, mode, policy),
        commitMessage: `Configure AgentField image auto-updates (${mode})`
      })
    }
  }
}
