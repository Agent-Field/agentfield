import { describe, expect, it, vi } from 'vitest'
import { loadRailwayStatus } from './railwayStatus'

describe('loadRailwayStatus', () => {
  it('keeps a valid login and returns an actionable message when workspaces fail', async () => {
    await expect(loadRailwayStatus({
      token: 'token',
      engineAvailable: true,
      hasDeployment: true,
      deploymentWorkspaceId: 'workspace-from-state',
      listWorkspaces: vi.fn(async () => { throw new Error('Railway unavailable') })
    })).resolves.toEqual({
      loggedIn: true,
      engineAvailable: true,
      hasDeployment: true,
      deploymentWorkspaceId: 'workspace-from-state',
      workspaces: [],
      message: 'Signed in, but Railway workspaces could not be loaded: Railway unavailable. Check Railway and try again.'
    })
  })

  it('does not call Railway when signed out', async () => {
    const listWorkspaces = vi.fn(async () => [])
    await expect(loadRailwayStatus({
      token: null,
      engineAvailable: true,
      hasDeployment: false,
      deploymentWorkspaceId: null,
      listWorkspaces
    })).resolves.toMatchObject({ loggedIn: false, workspaces: [] })
    expect(listWorkspaces).not.toHaveBeenCalled()
  })
})
