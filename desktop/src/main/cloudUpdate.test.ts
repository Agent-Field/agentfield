import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ControlPlaneVersion, PackageMaintenanceStatus } from '../shared/types'
import { CpApiError } from './cpClient'
import {
  applyCloudUpdate,
  applyCloudUpdateWithRailwayToken,
  autoUpdateModeAfterDeploy,
  checkCloudUpdate,
  cloudUpdateApplyPath,
  cloudUpdateMaintenanceMessage,
  cloudUpdateRailwayControlsAvailable,
  CloudUpdateChecker,
  setCloudAutoUpdateSchedule
} from './cloudUpdate'

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
  it('falls back to URL-matched tfstate when the version request times out', async () => {
    const setSchedule = vi.fn(async () => {})
    await expect(setCloudAutoUpdateSchedule({
      mode: 'weekends',
      connectedServerUrl: 'https://cp.example/',
      tfstate: {
        serviceId: 'state-service',
        environmentId: 'state-environment',
        url: 'https://cp.example'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => { throw new Error('timeout') }),
      setSchedule
    })).resolves.toMatchObject({ ok: true, serviceId: 'state-service' })
    expect(setSchedule).toHaveBeenCalledWith(
      'token', 'state-service', 'state-environment', 'weekends'
    )
  })

  it('falls back to URL-matched tfstate for a running non-Railway control plane', async () => {
    const setSchedule = vi.fn(async () => {})
    await expect(setCloudAutoUpdateSchedule({
      mode: 'nightly',
      connectedServerUrl: 'https://cp.example',
      tfstate: {
        serviceId: 'state-service',
        environmentId: 'state-environment',
        url: 'https://cp.example/'
      }
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', { platform: 'docker' })),
      setSchedule
    })).resolves.toMatchObject({ ok: true, serviceId: 'state-service' })
    expect(setSchedule).toHaveBeenCalledWith(
      'token', 'state-service', 'state-environment', 'nightly'
    )
  })

  it('always returns a failure result when Railway rejects the schedule', async () => {
    await expect(setCloudAutoUpdateSchedule({
      mode: 'off',
      connectedServerUrl: 'https://cp.example',
      tfstate: null
    }, {
      getAccessToken: vi.fn(async () => 'token'),
      getVersion: vi.fn(async () => running('0.1.135', {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      })),
      setSchedule: vi.fn(async () => { throw new Error('mutation denied') })
    })).resolves.toEqual({
      ok: false,
      message: 'Railway could not save that schedule: mutation denied. Check your Railway access and try again.'
    })
  })

  it('defaults only first deploys and re-applies only a stored same-service mode', () => {
    expect(autoUpdateModeAfterDeploy({
      firstDeploy: true,
      serviceId: 'new',
      storedMode: 'off',
      storedServiceId: 'old'
    })).toBe('nightly')
    expect(autoUpdateModeAfterDeploy({
      firstDeploy: false,
      serviceId: 'same',
      storedMode: 'off',
      storedServiceId: 'same'
    })).toBe('off')
    expect(autoUpdateModeAfterDeploy({
      firstDeploy: false,
      serviceId: 'new',
      storedMode: 'weekends',
      storedServiceId: 'old'
    })).toBeNull()
  })
})
