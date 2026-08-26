import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ControlPlaneVersion, PackageMaintenanceStatus } from '../shared/types'
import { CpApiError } from './cpClient'
import {
  applyCloudAutoUpdateAfterDeploy,
  applyCloudUpdate,
  applyCloudUpdateWithRailwayToken,
  autoUpdateModeAfterDeploy,
  checkCloudUpdate,
  classifyAutoUpdates,
  cloudAutoUpdatePreferenceAfterReconcile,
  cloudAutoUpdatePreferenceAfterSet,
  cloudAutoUpdateReconcileDecision,
  cloudUpdateApplyPath,
  cloudUpdateMaintenanceMessage,
  cloudUpdateRailwayControlsAvailable,
  CloudUpdateChecker,
  getCloudAutoUpdateState,
  railwaySettingsUrl,
  setCloudAutoUpdateSchedule
} from './cloudUpdate'
import { imageAutoUpdatesPatch } from './railwayApi'

function running(version: string, hosting: ControlPlaneVersion['hosting'] = { platform: 'railway' }): ControlPlaneVersion {
  return { version, commit: 'abc123', build_date: '2026-08-24T00:00:00Z', hosting, features: [] }
}

function dockerHub(version: string): typeof fetch {
  return vi.fn(async () => new Response(JSON.stringify({ results: [
    { name: 'latest', digest: 'sha256:new' },
    { name: `v${version}`, digest: 'sha256:new' }
  ] }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as typeof fetch
}

describe('checkCloudUpdate', () => {
  it('reports current when the running control plane matches the latest release', async () => {
    await expect(checkCloudUpdate({
      running: running('0.1.135'),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      status: 'current',
      current: '0.1.135',
      latest: '0.1.135',
      message: 'Control plane v0.1.135 is up to date.'
    })
  })

  it('reports available only when Docker Hub has a newer stable release', async () => {
    await expect(checkCloudUpdate({
      running: running('0.1.134'),
      tfstateImage: null,
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      status: 'available',
      current: '0.1.134',
      latest: '0.1.135',
      message: 'Control plane v0.1.135 is available.'
    })
  })

  it('reports a missing version endpoint as legacy rather than current', async () => {
    await expect(checkCloudUpdate({
      running: null,
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.120',
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      status: 'legacy',
      current: '0.1.120',
      latest: '0.1.135',
      message: 'This control plane is too old to report its running version. Update it to enable automatic version checks.'
    })
  })

  it('reports a release lookup failure as unknown, never current', async () => {
    await expect(checkCloudUpdate({
      running: running('0.1.134'),
      tfstateImage: null,
      fetchImpl: vi.fn(async () => { throw new Error('offline') }) as unknown as typeof fetch
    })).resolves.toEqual({
      status: 'unknown',
      current: '0.1.134',
      latest: null,
      message: 'Could not check Docker Hub for the latest control plane release. Check your connection and try again.'
    })
  })

  it('reports development builds as unknown instead of comparing NaN segments', async () => {
    await expect(checkCloudUpdate({
      running: running('dev'),
      tfstateImage: null,
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      status: 'unknown',
      current: 'dev',
      latest: null,
      message: 'This control plane reports a development build (dev); automatic version checks are unavailable.'
    })
  })

  it('compares a running prerelease build with the latest stable release', async () => {
    await expect(checkCloudUpdate({
      running: running('0.1.134-rc.4'),
      tfstateImage: null,
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      status: 'available',
      current: '0.1.134-rc.4',
      latest: '0.1.135',
      message: 'Control plane v0.1.135 is available.'
    })
  })
})

function applyDeps(version = '0.1.135') {
  let clock = 0
  return {
    fetchImpl: dockerHub(version),
    refreshAndDeploy: vi.fn(async () => ({ ok: true, message: 'deployed' })),
    setServiceImage: vi.fn(async () => {}),
    redeploy: vi.fn(async () => {}),
    getVersion: vi.fn(async () => running(version)),
    sleep: vi.fn(async (milliseconds: number) => { clock += milliseconds }),
    now: vi.fn(() => clock)
  }
}

describe('applyCloudUpdate path selection', () => {
  it('H1 — refuses non-comparable versions without any Railway effects', async () => {
    const deps = applyDeps('0.1.136')
    const result = await applyCloudUpdate({
      running: running('0.1.136+abc', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment',
        deployment_id: 'deployment'
      }),
      tfstateImage: null
    }, deps)

    expect(result).toEqual({
      ok: false,
      target: '0.1.136',
      message: 'Cannot update from running control plane v0.1.136+abc to v0.1.136 because those versions cannot be compared safely.'
    })
    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
    expect(deps.setServiceImage).not.toHaveBeenCalled()
    expect(deps.redeploy).not.toHaveBeenCalled()
    expect(deps.getVersion).not.toHaveBeenCalled()
  })

  it('D1 — returns already-current without Railway or tofu effects', async () => {
    const deps = applyDeps()
    const result = await applyCloudUpdate({
      running: running('0.1.135', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment',
        deployment_id: 'deployment'
      }),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.135',
      tfstateServiceId: 'service'
    }, deps)

    expect(result).toEqual({
      ok: true,
      target: '0.1.135',
      alreadyCurrent: true,
      message: 'Control plane is already running v0.1.135.'
    })
    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
    expect(deps.setServiceImage).not.toHaveBeenCalled()
    expect(deps.redeploy).not.toHaveBeenCalled()
    expect(deps.getVersion).not.toHaveBeenCalled()
  })

  it('refreshes and uses the existing deploy engine when tfstate is present', async () => {
    const deps = applyDeps()
    const result = await applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment'
      }),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateServiceId: 'service'
    }, deps)

    expect(deps.refreshAndDeploy).toHaveBeenCalledWith('agentfield/control-plane-cloud:v0.1.135')
    expect(deps.setServiceImage).not.toHaveBeenCalled()
    expect(result).toEqual({
      ok: true,
      target: '0.1.135',
      message: 'Updated to v0.1.135 — agents are being restored by the control plane.'
    })
  })

  it('updates and redeploys a Railway-hosted service when no tfstate exists', async () => {
    const deps = applyDeps()
    const result = await applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)

    expect(deps.setServiceImage).toHaveBeenCalledWith(
      'service',
      'environment',
      'agentfield/control-plane-cloud:v0.1.135'
    )
    expect(deps.redeploy).toHaveBeenCalledWith('service', 'environment')
    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
    expect(result.ok).toBe(true)
  })

  it('never deploys tfstate belonging to a different connected control plane', async () => {
    const deps = applyDeps()
    const result = await applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway',
        service_id: 'connected-service',
        environment_id: 'connected-environment'
      }),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateServiceId: 'state-service'
    }, deps)

    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
    expect(deps.setServiceImage).toHaveBeenCalledWith(
      'connected-service',
      'connected-environment',
      'agentfield/control-plane-cloud:v0.1.135'
    )
    expect(result.ok).toBe(true)
  })

  it('uses URL identity for a legacy control plane and rejects a URL mismatch', async () => {
    expect(cloudUpdateApplyPath({
      running: null,
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateUrl: 'https://state.example/',
      connectedServerUrl: 'https://state.example'
    })).toBe('tfstate')
    expect(cloudUpdateApplyPath({
      running: null,
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateUrl: 'https://state.example',
      connectedServerUrl: 'https://other.example'
    })).toBe('none')
  })

  it('uses URL-matched tfstate when a running Docker version has no Railway service id', () => {
    expect(cloudUpdateApplyPath({
      running: running('0.1.134', { platform: 'docker' }),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateUrl: 'https://state.example/',
      connectedServerUrl: 'https://state.example'
    })).toBe('tfstate')
  })

  it('D9 — exposes Railway controls only for a resolvable connected Railway service', () => {
    expect(cloudUpdateRailwayControlsAvailable({
      running: running('0.1.134', { platform: 'docker' }),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.134',
      tfstateServiceId: 'stale-service',
      tfstateEnvironmentId: 'stale-environment',
      tfstateUrl: 'https://state.example',
      connectedServerUrl: 'https://state.example'
    })).toBe(false)
    expect(cloudUpdateRailwayControlsAvailable({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    })).toBe(true)
  })

  it('explains that deployment identity is missing instead of guessing', async () => {
    const deps = applyDeps()
    const result = await applyCloudUpdate({
      running: running('0.1.134', { platform: 'docker' }),
      tfstateImage: null
    }, deps)

    expect(result).toEqual({
      ok: false,
      message: 'This control plane has no desktop deployment state or Railway service identity. Reconnect it from Remote, then try again.'
    })
    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
    expect(deps.setServiceImage).not.toHaveBeenCalled()
  })

  it('refuses to apply a Docker tag older than the running version', async () => {
    const deps = applyDeps('0.1.135')
    const result = await applyCloudUpdate({
      running: running('0.1.136'),
      tfstateImage: 'agentfield/control-plane-cloud:v0.1.136'
    }, deps)

    expect(result).toEqual({
      ok: false,
      message: 'Refusing to downgrade the running control plane from v0.1.136 to v0.1.135.'
    })
    expect(deps.refreshAndDeploy).not.toHaveBeenCalled()
  })

  it('reports an unmanaged apply blocker before asking Railway to sign in', async () => {
    const getAccessToken = vi.fn(async () => null)
    const createApplyDeps = vi.fn(() => applyDeps())

    await expect(applyCloudUpdateWithRailwayToken({
      running: running('0.1.134', { platform: 'docker' }),
      tfstateImage: null
    }, { getAccessToken, createApplyDeps })).resolves.toEqual({
      ok: false,
      message: 'This control plane has no desktop deployment state or Railway service identity. Reconnect it from Remote, then try again.'
    })
    expect(getAccessToken).not.toHaveBeenCalled()
    expect(createApplyDeps).not.toHaveBeenCalled()
  })

  it('H2 — returns already-current before requiring a Railway token', async () => {
    const getAccessToken = vi.fn(async () => null)
    const createApplyDeps = vi.fn(() => applyDeps('0.1.135'))

    await expect(applyCloudUpdateWithRailwayToken({
      running: running('0.1.135', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment'
      }),
      tfstateImage: null
    }, {
      getAccessToken,
      createApplyDeps,
      fetchImpl: dockerHub('0.1.135')
    })).resolves.toEqual({
      ok: true,
      target: '0.1.135',
      alreadyCurrent: true,
      message: 'Control plane is already running v0.1.135.'
    })
    expect(getAccessToken).not.toHaveBeenCalled()
    expect(createApplyDeps).not.toHaveBeenCalled()
  })
})

describe('CloudUpdateChecker background cadence', () => {
  afterEach(() => vi.useRealTimers())

  it('D8 — publishes apply success before the follow-up check clears available status', async () => {
    vi.useFakeTimers()
    const getVersion = vi
      .fn()
      .mockResolvedValueOnce(running('0.1.134'))
      .mockResolvedValueOnce(running('0.1.134'))
      .mockResolvedValueOnce(running('0.1.135'))
    const checker = new CloudUpdateChecker({
      enabled: () => true,
      getVersion,
      getTfstateImage: () => null,
      canApplyUpdate: () => true,
      applyUpdate: vi.fn(async () => ({
        ok: true,
        target: '0.1.135',
        message: 'Updated to v0.1.135. 1 agent restored.'
      })),
      fetchImpl: dockerHub('0.1.135')
    })
    await checker.check()

    await expect(checker.apply()).resolves.toMatchObject({ ok: true })
    expect(checker.status()).toMatchObject({
      status: 'available',
      message: 'Updated to v0.1.135. 1 agent restored.'
    })

    await vi.advanceTimersByTimeAsync(500)
    expect(checker.status().status).toBe('current')
  })

  it('waits for Remote to be enabled before a background check', async () => {
    vi.useFakeTimers()
    let enabled = false
    const getVersion = vi.fn(async () => running('0.1.135'))
    const checker = new CloudUpdateChecker({
      enabled: () => enabled,
      getVersion,
      getTfstateImage: () => null,
      fetchImpl: dockerHub('0.1.135')
    })
    checker.startAutoCheck(15, 40)

    await vi.advanceTimersByTimeAsync(15)
    expect(getVersion).not.toHaveBeenCalled()
    enabled = true
    await vi.advanceTimersByTimeAsync(40)
    expect(getVersion).toHaveBeenCalledTimes(1)
    expect(checker.status().status).toBe('current')
  })

  it('turns a running-version lookup failure into unknown status', async () => {
    const statuses: ReturnType<CloudUpdateChecker['status']>[] = []
    const completed = vi.fn()
    const getVersion = vi
      .fn()
      .mockResolvedValueOnce(running('0.1.135', {
        platform: 'railway',
        service_id: 'stale-service'
      }))
      .mockRejectedValueOnce(new Error('remote offline'))
    const checker = new CloudUpdateChecker({
      enabled: () => true,
      getVersion,
      getTfstateImage: () => null,
      fetchImpl: dockerHub('0.1.135'),
      onStatus: (status) => statuses.push(status),
      onCompletedCheck: completed
    })
    await checker.check()
    expect(checker.status().hosting?.service_id).toBe('stale-service')
    statuses.length = 0
    completed.mockClear()

    await expect(checker.check()).resolves.toMatchObject({
      status: 'unknown',
      message: expect.stringContaining('remote offline'),
      hosting: undefined
    })
    expect(statuses[0].hosting).toBeUndefined()
    expect(completed).not.toHaveBeenCalled()
  })

  it('marks a URL-matched legacy deployment as actionable', async () => {
    const checker = new CloudUpdateChecker({
      enabled: () => true,
      getVersion: vi.fn(async () => null),
      getTfstateImage: () => 'agentfield/control-plane-cloud:v0.1.120',
      canApplyUpdate: () => true,
      fetchImpl: dockerHub('0.1.135')
    })
    await expect(checker.check()).resolves.toMatchObject({
      status: 'legacy',
      latest: '0.1.135',
      canApply: true
    })
  })

  it('names the manual Railway action for an unmatched legacy deployment', async () => {
    const checker = new CloudUpdateChecker({
      enabled: () => true,
      getVersion: vi.fn(async () => null),
      getTfstateImage: () => 'agentfield/control-plane-cloud:v0.1.120',
      canApplyUpdate: () => false,
      fetchImpl: dockerHub('0.1.135')
    })
    await expect(checker.check()).resolves.toMatchObject({
      status: 'legacy',
      canApply: false,
      message: 'This legacy control plane cannot be matched to this desktop deployment. In Railway, set its image to agentfield/control-plane-cloud:v0.1.135 and redeploy it.'
    })
  })
})

describe('six-minute apply polling', () => {
  it('H1 — treats a non-comparable observed version as not yet updated', async () => {
    const deps = applyDeps('0.1.135')
    deps.getVersion
      .mockResolvedValueOnce(running('0.1.135+abc'))
      .mockResolvedValueOnce(running('0.1.135'))

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({ ok: true, target: '0.1.135' })

    expect(deps.getVersion).toHaveBeenCalledTimes(2)
  })

  it('D2 — rejects the pre-restart deployment and accepts the new deployment id', async () => {
    const deps = applyDeps()
    deps.getVersion
      .mockResolvedValueOnce(running('0.1.135', {
        platform: 'railway', deployment_id: 'old-deployment'
      }))
      .mockResolvedValueOnce(running('0.1.135', {
        platform: 'railway', deployment_id: 'new-deployment'
      }))

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment',
        deployment_id: 'old-deployment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({ ok: true, target: '0.1.135' })

    expect(deps.getVersion).toHaveBeenCalledTimes(2)
    expect(deps.sleep).toHaveBeenCalledTimes(2)
  })

  it('D3 — legacy deployment identity falls back to version after the first sleep', async () => {
    const deps = applyDeps()

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({ ok: true, target: '0.1.135' })

    expect(deps.sleep).toHaveBeenCalledTimes(1)
    expect(deps.getVersion).toHaveBeenCalledTimes(1)
    expect(deps.sleep.mock.invocationCallOrder[0])
      .toBeLessThan(deps.getVersion.mock.invocationCallOrder[0])
  })

  it('D4 — reports restored agents and failures, with a 404 compatibility fallback', async () => {
    const maintenance: PackageMaintenanceStatus = {
      enabled: true,
      reason: '',
      interval: '6h0m0s',
      boot_pass_completed: true,
      hosting: 'railway',
      last_run: {
        started_at: 'start',
        finished_at: 'finish',
        checked: 0,
        updated: [],
        restored: ['a', 'b'],
        skipped: [],
        errors: ['restore c: boom']
      },
      next_run_at: 'later'
    }
    const deps = { ...applyDeps(), getMaintenanceStatus: vi.fn(async () => maintenance) }

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({
      ok: true,
      message: 'Updated to v0.1.135. Restored: a, b. Failed to restore: c (boom).'
    })

    const oldControlPlane = {
      ...applyDeps(),
      getMaintenanceStatus: vi.fn(async (): Promise<PackageMaintenanceStatus> => {
        throw new CpApiError({ status: 404, message: 'missing' })
      })
    }
    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, oldControlPlane)).resolves.toMatchObject({
      ok: true,
      message: 'Updated to v0.1.135 — agents are being restored by the control plane.'
    })
  })

  it('H6 — a restore-complete flag without a summary yet keeps waiting for the pass', async () => {
    const withoutSummary: PackageMaintenanceStatus = {
      enabled: true,
      reason: '',
      interval: '6h0m0s',
      boot_restore_completed: true,
      boot_pass_completed: false,
      hosting: 'railway',
      last_run: null,
      next_run_at: 'later'
    }
    const finished: PackageMaintenanceStatus = {
      ...withoutSummary,
      boot_pass_completed: true,
      last_run: {
        started_at: 'start',
        finished_at: 'end',
        checked: 1,
        updated: [],
        restored: ['a'],
        skipped: [],
        errors: []
      }
    }
    const getMaintenanceStatus = vi.fn()
      .mockResolvedValueOnce(withoutSummary)
      .mockResolvedValueOnce(withoutSummary)
      .mockResolvedValue(finished)
    const deps = { ...applyDeps(), getMaintenanceStatus }

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({
      ok: true,
      message: 'Updated to v0.1.135. 1 agent restored.'
    })
    expect(getMaintenanceStatus).toHaveBeenCalledTimes(3)
  })

  it('H6 — stops on restore completion and separates restore failures from warnings', async () => {
    const maintenance: PackageMaintenanceStatus = {
      enabled: true,
      reason: '',
      interval: '6h0m0s',
      boot_restore_completed: true,
      boot_pass_completed: false,
      hosting: 'railway',
      last_run: {
        started_at: 'start',
        finished_at: '',
        checked: 0,
        updated: [],
        restored: ['a'],
        skipped: [],
        errors: ['check a: remote unreachable']
      },
      next_run_at: 'later'
    }
    const deps = { ...applyDeps(), getMaintenanceStatus: vi.fn(async () => maintenance) }

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({
      ok: true,
      message: 'Updated to v0.1.135. 1 agent restored. 1 maintenance warning.'
    })
    expect(deps.getMaintenanceStatus).toHaveBeenCalledTimes(1)

    expect(cloudUpdateMaintenanceMessage('0.1.135', {
      ...maintenance,
      last_run: {
        ...maintenance.last_run!,
        errors: ['restore c: boom']
      }
    })).toBe('Updated to v0.1.135. Restored: a. Failed to restore: c (boom).')
  })

  it('keeps polling across multiple iterations until the target is reported', async () => {
    let clock = 0
    const deps = applyDeps()
    deps.now.mockImplementation(() => clock)
    deps.sleep.mockImplementation(async (milliseconds) => { clock += milliseconds })
    deps.getVersion
      .mockResolvedValueOnce(running('0.1.134'))
      .mockResolvedValueOnce(running('0.1.134'))
      .mockResolvedValueOnce(running('0.1.135'))

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({ ok: true, target: '0.1.135' })
    expect(deps.getVersion).toHaveBeenCalledTimes(3)
    expect(deps.sleep).toHaveBeenCalledTimes(3)
  })

  it('returns the documented failure after the six-minute deadline', async () => {
    let clock = 0
    const deps = applyDeps()
    deps.now.mockImplementation(() => clock)
    deps.sleep.mockImplementation(async () => { clock = 6 * 60_000 })
    deps.getVersion.mockResolvedValue(running('0.1.134'))

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toEqual({
      ok: false,
      target: '0.1.135',
      message: 'Railway accepted v0.1.135, but the control plane did not report that version within 6 minutes. Open Railway deployment logs, then check again.'
    })
  })

  it('continues when getVersion throws during a redeploy', async () => {
    let clock = 0
    const deps = applyDeps()
    deps.now.mockImplementation(() => clock)
    deps.sleep.mockImplementation(async (milliseconds) => { clock += milliseconds })
    deps.getVersion
      .mockRejectedValueOnce(new Error('restarting'))
      .mockResolvedValueOnce(running('0.1.135'))

    await expect(applyCloudUpdate({
      running: running('0.1.134', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }),
      tfstateImage: null
    }, deps)).resolves.toMatchObject({ ok: true })
    expect(deps.getVersion).toHaveBeenCalledTimes(2)
  })
})

describe('Railway auto-update preference effects', () => {
  it('D4 — falls back to URL-matched tfstate when the version request times out', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    await expect(setCloudAutoUpdateSchedule({
      mode: 'weekends',
      connectedServerUrl: 'https://cp.example/',
      tfstate: {
        projectId: 'state-project',
        serviceId: 'state-service',
        environmentId: 'state-environment',
        url: 'https://cp.example'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => { throw new Error('timeout') }),
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates
    })).resolves.toMatchObject({ ok: true, serviceId: 'state-service' })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'token', 'state-environment', 'state-service', 'weekends', 'patch'
    )
  })

  it('D4 — falls back to URL-matched tfstate for a running non-Railway control plane', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    await expect(setCloudAutoUpdateSchedule({
      mode: 'nightly',
      connectedServerUrl: 'https://cp.example',
      tfstate: {
        projectId: null,
        serviceId: 'state-service',
        environmentId: 'state-environment',
        url: 'https://cp.example/'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', { platform: 'docker' })),
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates
    })).resolves.toMatchObject({ ok: true, serviceId: 'state-service' })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'token', 'state-environment', 'state-service', 'nightly', 'patch'
    )
  })

  it('C1 — a missing Railway autoUpdates object reads as not set', async () => {
    await expect(getCloudAutoUpdateState({
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => null)
    })).resolves.toMatchObject({ ok: true, mode: null, policy: null })
  })

  it('C2 — classifies every owned window, custom windows, and minor policy', () => {
    const nightly = Array.from(
      { length: 7 },
      (_, day) => ({ day, startHour: 2, endHour: 6 })
    )
    const anytime = Array.from(
      { length: 7 },
      (_, day) => ({ day, startHour: 0, endHour: 24 })
    )
    expect(classifyAutoUpdates({ type: 'patch', schedule: nightly }))
      .toEqual({ mode: 'nightly', policy: 'patch' })
    expect(classifyAutoUpdates({ type: 'disabled' }))
      .toEqual({ mode: 'off', policy: 'disabled' })
    expect(classifyAutoUpdates({
      type: 'patch',
      schedule: [
        { day: 6, startHour: 0, endHour: 24 },
        { day: 0, startHour: 0, endHour: 24 }
      ]
    })).toEqual({ mode: 'weekends', policy: 'patch' })
    expect(classifyAutoUpdates({ type: 'patch', schedule: anytime }))
      .toEqual({ mode: 'anytime', policy: 'patch' })
    expect(classifyAutoUpdates({ type: 'patch', schedule: nightly.slice(0, 3) }))
      .toEqual({ mode: 'custom', policy: 'patch' })
    expect(classifyAutoUpdates({ type: 'minor', schedule: nightly }))
      .toEqual({ mode: 'nightly', policy: 'minor' })
  })

  it('C5 — commit failures include Railway detail and the best settings deep link', async () => {
    const withProject = await setCloudAutoUpdateSchedule({
      mode: 'off',
      connectedServerUrl: 'https://cp.example',
      tfstate: {
        projectId: 'project',
        serviceId: 'service',
        environmentId: 'environment',
        url: 'https://cp.example'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates: vi.fn(async () => { throw new Error('mutation denied') })
    })
    const serviceUrl = 'https://railway.com/project/project/service/service/settings'
    expect(withProject).toMatchObject({ ok: false, settingsUrl: serviceUrl })
    expect(withProject.message).toContain('mutation denied')
    expect(withProject.message).toContain(serviceUrl)
    expect(cloudAutoUpdatePreferenceAfterSet(withProject, 'off')).toBeNull()

    const withoutProject = await setCloudAutoUpdateSchedule({
      mode: 'nightly',
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates: vi.fn(async () => { throw new Error('network down') })
    })
    expect(withoutProject.settingsUrl).toBe('https://railway.com/dashboard')
    expect(withoutProject.message).toContain('https://railway.com/dashboard')
  })

  it('C7 / D5 — read failures retain Railway detail and a known service link', async () => {
    const result = await getCloudAutoUpdateState({
      connectedServerUrl: 'https://cp.example',
      tfstate: {
        projectId: 'project',
        serviceId: 'service',
        environmentId: 'environment',
        url: 'https://cp.example'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway',
        service_id: 'service',
        environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => { throw new Error('config query denied') })
    })

    expect(result).toEqual({
      ok: false,
      mode: null,
      policy: null,
      serviceId: 'service',
      message: 'Railway could not read image auto-updates: config query denied. https://railway.com/project/project/service/service/settings',
      settingsUrl: 'https://railway.com/project/project/service/service/settings'
    })
  })

  it('C8 — read IPC state contains no config or variables keys', async () => {
    const result = await getCloudAutoUpdateState({
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway',
        project_id: 'project',
        service_id: 'service',
        environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => ({
        type: 'disabled',
        variables: { SECRET: 'do-not-forward' },
        config: { services: {} }
      } as never))
    })
    expect(result).toEqual({
      ok: true,
      mode: 'off',
      policy: 'disabled',
      serviceId: 'service',
      settingsUrl: railwaySettingsUrl('project', 'service')
    })
    expect(result).not.toHaveProperty('variables')
    expect(result).not.toHaveProperty('config')
  })

  it('G1 / H4 — failed advisory reads report the policy actually written', async () => {
    const patches: Record<string, unknown>[] = []
    const setImageAutoUpdates = vi.fn(async (
      _token: string,
      _environmentId: string,
      serviceId: string,
      mode: 'off' | 'nightly' | 'weekends' | 'anytime',
      policy: 'patch' | 'minor'
    ) => {
      patches.push(imageAutoUpdatesPatch(serviceId, mode, policy))
    })
    const deps = {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => { throw new Error('read timed out') }),
      setImageAutoUpdates
    }

    const nightly = await setCloudAutoUpdateSchedule({
      mode: 'nightly', connectedServerUrl: 'https://cp.example', tfstate: null
    }, deps)
    const off = await setCloudAutoUpdateSchedule({
      mode: 'off', connectedServerUrl: 'https://cp.example', tfstate: null
    }, deps)

    expect(setImageAutoUpdates).toHaveBeenNthCalledWith(
      1, 'token', 'environment', 'service', 'nightly', 'patch'
    )
    expect(setImageAutoUpdates).toHaveBeenNthCalledWith(
      2, 'token', 'environment', 'service', 'off', 'patch'
    )
    expect(patches[0]).toMatchObject({
      services: { service: { source: { autoUpdates: { type: 'patch' } } } }
    })
    expect(patches[1]).toEqual({
      services: { service: { source: { autoUpdates: { type: 'disabled' } } } }
    })
    expect(nightly.ok).toBe(true)
    expect(nightly.message).toContain(
      "Railway's current policy could not be read first; the patch policy was written."
    )
    expect(off.ok).toBe(true)
    expect(off.message).toContain("Railway's current policy could not be read first.")
    expect(off.message).not.toContain('patch policy was written')
  })

  it('F9 — set preserves a live Railway minor policy', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    await expect(setCloudAutoUpdateSchedule({
      mode: 'anytime',
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => ({
        type: 'minor',
        schedule: Array.from(
          { length: 7 },
          (_, day) => ({ day, startHour: 2, endHour: 6 })
        )
      })),
      setImageAutoUpdates
    })).resolves.toMatchObject({ ok: true })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'token', 'environment', 'service', 'anytime', 'minor'
    )
  })

  it("I5 — Off explains that a live minor policy was replaced", async () => {
    const result = await setCloudAutoUpdateSchedule({
      mode: 'off',
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      getAutoUpdates: vi.fn(async () => ({
        type: 'minor',
        schedule: Array.from(
          { length: 7 },
          (_, day) => ({ day, startHour: 2, endHour: 6 })
        )
      })),
      setImageAutoUpdates: vi.fn(async () => {})
    })

    expect(result.ok).toBe(true)
    expect(result.message).toContain(
      "Railway's minor-update policy was replaced by Off; enabling a window again uses the patch policy."
    )
  })

  it('F3 — pure reconcile rule writes only when Railway has no policy', () => {
    const liveNotSet = { ok: true, mode: null, policy: null } as const
    expect(cloudAutoUpdateReconcileDecision({
      firstDeploy: true,
      serviceId: 'new',
      storedMode: 'off',
      storedServiceId: 'old',
      live: liveNotSet
    })).toMatchObject({ writeMode: 'nightly', autoUpdateOk: true })
    expect(cloudAutoUpdateReconcileDecision({
      firstDeploy: false,
      serviceId: 'same',
      storedMode: 'weekends',
      storedServiceId: 'same',
      live: liveNotSet
    })).toMatchObject({ writeMode: 'weekends', autoUpdateOk: true })

    const livePresent = cloudAutoUpdateReconcileDecision({
      firstDeploy: false,
      serviceId: 'same',
      storedMode: 'off',
      storedServiceId: 'same',
      live: { ok: true, mode: 'nightly', policy: 'minor' }
    })
    expect(livePresent).toMatchObject({ writeMode: null, autoUpdateOk: true })
    expect(livePresent.autoUpdateMessage).toContain('Railway image auto-updates: Nightly')

    const readFailure = cloudAutoUpdateReconcileDecision({
      firstDeploy: false,
      serviceId: 'new',
      storedMode: null,
      storedServiceId: null,
      live: {
        ok: false,
        mode: null,
        policy: null,
        message: 'read denied. https://railway.com/project/p/service/s/settings',
        settingsUrl: 'https://railway.com/project/p/service/s/settings'
      }
    })
    expect(readFailure).toEqual({
      writeMode: null,
      autoUpdateOk: false,
      autoUpdateMessage: 'read denied. https://railway.com/project/p/service/s/settings'
    })
  })

  it('G2 — first-deploy read failure writes Nightly while reconcile read failure does not write', async () => {
    const failedLiveRead = {
      ok: false,
      mode: null,
      policy: null,
      message: 'read failed',
      settingsUrl: 'https://railway.com/dashboard'
    } as const
    expect(cloudAutoUpdateReconcileDecision({
      firstDeploy: true,
      serviceId: 'new',
      storedMode: 'off',
      storedServiceId: 'old',
      live: failedLiveRead
    })).toMatchObject({
      writeMode: 'nightly',
      autoUpdateOk: true
    })
    expect(cloudAutoUpdateReconcileDecision({
      firstDeploy: false,
      serviceId: 'existing',
      storedMode: 'weekends',
      storedServiceId: 'existing',
      live: failedLiveRead
    })).toEqual({
      writeMode: null,
      autoUpdateOk: false,
      autoUpdateMessage: 'read failed'
    })

    const setImageAutoUpdates = vi.fn(async () => {})
    const result = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: true,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'new',
      storedMode: null,
      storedServiceId: null
    }, {
      getAutoUpdates: vi.fn(async () => { throw new Error('read timed out') }),
      setImageAutoUpdates
    })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'environment', 'new', 'nightly', 'patch'
    )
    expect(result).toMatchObject({
      appliedMode: 'nightly',
      liveMode: null,
      autoUpdateOk: true
    })
    expect(result.autoUpdateMessage).toContain("current policy could not be read")
  })

  it('F3 — live policy or non-first reconcile read failure never triggers a write', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    const liveResult = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: false,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      storedMode: 'off',
      storedServiceId: 'service'
    }, {
      getAutoUpdates: vi.fn(async () => ({
        type: 'patch',
        schedule: Array.from(
          { length: 7 },
          (_, day) => ({ day, startHour: 0, endHour: 24 })
        )
      })),
      setImageAutoUpdates
    })
    expect(liveResult.autoUpdateMessage).toContain('Railway image auto-updates: Anytime')
    expect(setImageAutoUpdates).not.toHaveBeenCalled()

    const failedResult = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: false,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      storedMode: null,
      storedServiceId: null
    }, {
      getAutoUpdates: vi.fn(async () => { throw new Error('read failed') }),
      setImageAutoUpdates
    })
    expect(failedResult).toMatchObject({
      autoUpdateOk: false,
      autoUpdateSettingsUrl: 'https://railway.com/project/project/service/service/settings'
    })
    expect(failedResult.autoUpdateMessage).toContain('read failed')
    expect(setImageAutoUpdates).not.toHaveBeenCalled()
  })

  it('G5 — post-deploy persistence follows live classification when reconcile did not write', async () => {
    const liveResult = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: false,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      storedMode: 'off',
      storedServiceId: 'service'
    }, {
      getAutoUpdates: vi.fn(async () => ({
        type: 'patch',
        schedule: [
          { day: 6, startHour: 0, endHour: 24 },
          { day: 0, startHour: 0, endHour: 24 }
        ]
      })),
      setImageAutoUpdates: vi.fn(async () => {})
    })
    expect(liveResult).toMatchObject({ appliedMode: null, liveMode: 'weekends' })
    expect(cloudAutoUpdatePreferenceAfterReconcile(liveResult, 'off')).toBe('weekends')
    expect(cloudAutoUpdatePreferenceAfterReconcile({
      appliedMode: null, liveMode: 'custom', liveOk: true
    }, 'off')).toBeNull()
    expect(cloudAutoUpdatePreferenceAfterReconcile({
      appliedMode: null, liveMode: null, liveOk: true
    }, 'off')).toBeNull()
    expect(cloudAutoUpdatePreferenceAfterReconcile({
      appliedMode: 'nightly', liveMode: null, liveOk: false
    }, null)).toBe('nightly')
  })

  it('H2 — a failed or skipped live read preserves the same-service cached preference', async () => {
    const failedRead = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: false,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      storedMode: 'weekends',
      storedServiceId: 'service'
    }, {
      getAutoUpdates: vi.fn(async () => { throw new Error('read timed out') }),
      setImageAutoUpdates: vi.fn(async () => {})
    })

    expect(failedRead).toMatchObject({
      appliedMode: null,
      liveMode: null,
      liveOk: false
    })
    expect(cloudAutoUpdatePreferenceAfterReconcile(failedRead, 'weekends'))
      .toBe('weekends')
    expect(cloudAutoUpdatePreferenceAfterReconcile(null, 'anytime'))
      .toBe('anytime')
    expect(cloudAutoUpdatePreferenceAfterReconcile({
      appliedMode: null,
      liveMode: 'custom',
      liveOk: true
    }, 'weekends')).toBeNull()
    expect(cloudAutoUpdatePreferenceAfterReconcile({
      appliedMode: null,
      liveMode: null,
      liveOk: true
    }, 'weekends')).toBeNull()
  })

  it('F1 — a rejected post-deploy patch returns renderer-facing feedback and link', async () => {
    const result = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: true,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      storedMode: null,
      storedServiceId: null
    }, {
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates: vi.fn(async () => { throw new Error('patch commit rejected') })
    })

    expect(result).toMatchObject({
      appliedMode: null,
      autoUpdateOk: false,
      autoUpdateSettingsUrl: 'https://railway.com/project/project/service/service/settings'
    })
    expect(result.autoUpdateMessage).toContain('patch commit rejected')
  })

  it('C9 — first deploy applies Nightly through the shared patch-commit path', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    await expect(applyCloudAutoUpdateAfterDeploy({
      firstDeploy: true,
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'new',
      storedMode: 'off',
      storedServiceId: 'old'
    }, {
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates
    })).resolves.toMatchObject({ appliedMode: 'nightly', autoUpdateOk: true })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'environment', 'new', 'nightly', 'patch'
    )
  })

  it('I1 / H7 — a retried deploy with no usable stored mode seeds Nightly', async () => {
    const setImageAutoUpdates = vi.fn(async () => {})
    const result = await applyCloudAutoUpdateAfterDeploy({
      firstDeploy: false,
      serviceId: 'new',
      projectId: 'project',
      environmentId: 'environment',
      storedMode: null,
      storedServiceId: 'new'
    }, {
      getAutoUpdates: vi.fn(async () => null),
      setImageAutoUpdates
    })

    expect(autoUpdateModeAfterDeploy({
      firstDeploy: false,
      serviceId: 'new',
      storedMode: null,
      storedServiceId: 'new'
    })).toBe('nightly')
    expect(result).toMatchObject({
      appliedMode: 'nightly',
      liveMode: null,
      liveOk: true,
      autoUpdateOk: true
    })
    expect(setImageAutoUpdates).toHaveBeenCalledWith(
      'environment', 'new', 'nightly', 'patch'
    )
  })

  it('I7 — a skipped reconcile without environmentId keeps the stored preference', () => {
    expect(cloudAutoUpdatePreferenceAfterReconcile(null, 'nightly')).toBe('nightly')
  })
})
