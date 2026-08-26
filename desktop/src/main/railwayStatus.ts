import type { RailwayStatus } from '../shared/types'
import type { RailwayWorkspace } from './railwayApi'

export interface RailwayStatusInput {
  token: string | null
  engineAvailable: boolean
  hasDeployment: boolean
  deploymentWorkspaceId: string | null
  deploymentServiceId?: string | null
  listWorkspaces: (token: string) => Promise<RailwayWorkspace[]>
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Build Railway status without ever rejecting over workspace enumeration. A
 * valid OAuth token remains logged in, and the renderer gets recovery text. */
export async function loadRailwayStatus(input: RailwayStatusInput): Promise<RailwayStatus> {
  const base: RailwayStatus = {
    loggedIn: input.token !== null,
    engineAvailable: input.engineAvailable,
    hasDeployment: input.hasDeployment,
    ...(input.deploymentWorkspaceId
      ? { deploymentWorkspaceId: input.deploymentWorkspaceId }
      : {}),
    ...(input.deploymentServiceId
      ? { deploymentServiceId: input.deploymentServiceId }
      : {}),
    workspaces: []
  }
  if (!input.token) return base
  try {
    return { ...base, workspaces: await input.listWorkspaces(input.token) }
  } catch (error) {
    return {
      ...base,
      message: `Signed in, but Railway workspaces could not be loaded: ${errorText(error)}. Check Railway and try again.`
    }
  }
}
