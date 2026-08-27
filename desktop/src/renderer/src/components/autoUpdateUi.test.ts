import { describe, expect, it } from 'vitest'
import type { CloudUpdateStatus, RailwayStatus, SnapshotAgent } from '../../../shared/types'
import { packageToInstalledAgent } from '../../../main/agentfield'
import type { PackageInfo } from '../../../main/cpClient'
import {
  cloudUpdateApplyFeedback,
  cloudUpdateApplyResultVisible,
  cloudUpdateBannerActionVisible,
  cloudUpdateBannerCopy,
  cloudUpdateBannerText,
  cloudUpdateBannerVisible
} from './CloudUpdateBanner'
import {
  CLOUD_AUTO_UPDATE_FALLBACK_HINT,
  CLOUD_AUTO_UPDATE_CUSTOM_HINT,
  CLOUD_AUTO_UPDATE_LOADING_HINT,
  CLOUD_AUTO_UPDATE_MINOR_HINT,
  CLOUD_AUTO_UPDATE_NOT_SET_LABEL,
  CLOUD_AUTO_UPDATE_SAVING_HINT,
  CLOUD_AUTO_UPDATE_UNKNOWN_LABEL,
  cloudAutoUpdateForTarget,
  cloudAutoUpdateReadEnabled,
  cloudAutoUpdateSelectState,
  cloudAutoUpdateSnapshotAfterRead,
  cloudAutoUpdateStoredMode,
  cloudAutoUpdateTargetKey,
  cloudDeployAutoUpdateFeedback,
  cloudUpdateActionLabel,
  cloudUpdateActionVisible,
  cloudUpdateFeedbackForTarget,
  cloudUpdateFeedbackAfterSave,
  cloudUpdateFeedbackAfterTargetChange,
  cloudUpdateFeedbackClass,
  deployedWorkspacePickerDisabled,
  deployedWorkspacePickerVisible,
  deploymentActionWorkspaceId,
  railwayImageUpdatesVisible
} from './CloudPanel'
import {
  agentAutoUpdateActionVisible,
  agentManualUpdateActionVisible,
  agentUpdateChip,
  agentUpdateChipTitle,
  localControlPlaneRestartVisible
} from './AgentsPanel'
import { latestPackageUpdateCheckedAt, packageUpdateTimestamps } from './SettingsPanel'

describe('cloud update banner dismissal', () => {
  const available: CloudUpdateStatus = {
    status: 'available',
    current: '0.1.134',
    latest: '0.1.135',
    message: 'available',
    checking: false,
    applying: false,
    lastCheckedAt: '2026-08-24T00:00:00Z',
    canApply: true,
    hosting: { platform: 'railway' }
  }

  it('dismisses only the version the user dismissed', () => {
    expect(cloudUpdateBannerVisible(available, null)).toBe(true)
    expect(cloudUpdateBannerVisible(available, '0.1.135')).toBe(false)
    expect(cloudUpdateBannerVisible({ ...available, latest: '0.1.136' }, '0.1.135')).toBe(true)
  })

  it('keeps unmanaged updates informational without an Update button', () => {
    const informational = { ...available, canApply: false }
    expect(cloudUpdateBannerVisible(informational, null)).toBe(true)
    expect(cloudUpdateBannerActionVisible(informational)).toBe(false)
    expect(cloudUpdateBannerText(informational)).toBe(
      'Control plane v0.1.135 is available — update your control plane image'
    )
  })

  it('D5 — shows an actionable legacy banner and hides only its dismissed release', () => {
    const legacy = {
      ...available,
      status: 'legacy' as const,
      current: null,
      canApply: true
    }
    expect(cloudUpdateBannerVisible(legacy, null)).toBe(true)
    expect(cloudUpdateBannerText(legacy)).toBe(
      'Control plane v0.1.135 is available — this one is too old to report its version'
    )
    expect(cloudUpdateBannerActionVisible(legacy)).toBe(true)
    expect(cloudUpdateBannerVisible(legacy, '0.1.135')).toBe(false)
  })
})

describe('deployed Railway workspace picker', () => {
  it('remains visible for a deployed project with multiple workspaces', () => {
    const railway: RailwayStatus = {
      loggedIn: true,
      engineAvailable: true,
      hasDeployment: true,
      deploymentWorkspaceId: 'workspace-2',
      workspaces: [
        { id: 'workspace-1', name: 'One' },
        { id: 'workspace-2', name: 'Two' }
      ]
    }
    expect(deployedWorkspacePickerVisible(railway)).toBe(true)
    expect(deploymentActionWorkspaceId(railway, '')).toBe('workspace-2')
    expect(deploymentActionWorkspaceId(railway, 'workspace-1')).toBe('workspace-2')
    expect(deploymentActionWorkspaceId({ ...railway, workspaces: [] }, ''))
      .toBe('workspace-2')
    expect(deployedWorkspacePickerDisabled(railway, false)).toBe(true)
  })

  it('D9 — hides stale Railway controls and locks actions to the recorded workspace', () => {
    const nonRailway: CloudUpdateStatus = {
      status: 'available',
      current: '0.1.134',
      latest: '0.1.135',
      message: '',
      checking: false,
      applying: false,
      lastCheckedAt: '',
      canApply: true,
      canManageRailway: false,
      hosting: { platform: 'docker' }
    }
    expect(railwayImageUpdatesVisible(nonRailway)).toBe(false)
    expect(railwayImageUpdatesVisible({ ...nonRailway, canManageRailway: true })).toBe(true)
    expect(railwayImageUpdatesVisible({
      ...nonRailway,
      canManageRailway: undefined,
      hosting: {
        platform: 'railway', service_id: 'service', environment_id: 'environment'
      }
    })).toBe(true)
    expect(deploymentActionWorkspaceId({
      loggedIn: true,
      engineAvailable: true,
      hasDeployment: true,
      deploymentWorkspaceId: 'recorded',
      workspaces: [{ id: 'recorded', name: 'Recorded' }, { id: 'other', name: 'Other' }]
    }, 'other')).toBe('recorded')
  })
})

describe('Railway live auto-update control', () => {
  it('C1 — shows the not-set choice when Railway has no policy', () => {
    expect(cloudAutoUpdateSelectState({
      ok: true,
      mode: null,
      policy: null
    }, 'nightly')).toEqual({
      value: '',
      placeholder: CLOUD_AUTO_UPDATE_NOT_SET_LABEL,
      hint: null,
      disabled: false
    })
    expect(CLOUD_AUTO_UPDATE_NOT_SET_LABEL).toBe('Not set — choose a window')
  })

  it('C6 — selects the re-read live value instead of the stored preference', () => {
    expect(cloudAutoUpdateSelectState({
      ok: true,
      mode: 'nightly',
      policy: 'patch'
    }, 'weekends')).toEqual({
      value: 'nightly',
      placeholder: CLOUD_AUTO_UPDATE_NOT_SET_LABEL,
      hint: null,
      disabled: false
    })
  })

  it('C7 / H3 — failed reads leave the placeholder selected and disclose the cached mode', () => {
    expect(cloudAutoUpdateSelectState({
      ok: false,
      mode: null,
      policy: null,
      settingsUrl: 'https://railway.com/dashboard'
    }, 'weekends')).toEqual({
      value: '',
      placeholder: CLOUD_AUTO_UPDATE_UNKNOWN_LABEL,
      hint: `Last known window: Weekends. ${CLOUD_AUTO_UPDATE_FALLBACK_HINT}`,
      disabled: false
    })
  })

  it('F2 — a target change hides stale state and reads only an eligible target', () => {
    const live = { ok: true, mode: 'nightly', policy: 'patch' } as const
    expect(cloudAutoUpdateForTarget({ target: 'old', state: live }, 'new')).toBeNull()
    expect(cloudAutoUpdateForTarget({ target: 'new', state: live }, 'new')).toBe(live)
    expect(cloudAutoUpdateReadEnabled(true, true)).toBe(true)
    expect(cloudAutoUpdateReadEnabled(false, true)).toBe(false)
    expect(cloudAutoUpdateReadEnabled(true, false)).toBe(false)
  })

  it('F4 / H5 — loading shows one checking label and only hints at a cached mode', () => {
    expect(cloudAutoUpdateSelectState(null, 'weekends')).toEqual({
      value: 'weekends',
      placeholder: CLOUD_AUTO_UPDATE_LOADING_HINT,
      hint: 'Last known window: Weekends',
      disabled: true
    })
    expect(cloudAutoUpdateSelectState(null, null)).toEqual({
      value: '',
      placeholder: CLOUD_AUTO_UPDATE_LOADING_HINT,
      hint: null,
      disabled: true
    })
  })

  it('F6 / H3 — failed reads combine cached disclosure and cause without the raw URL', () => {
    expect(cloudAutoUpdateSelectState({
      ok: false,
      mode: null,
      policy: null,
      message: 'Railway could not read image auto-updates: network down. https://railway.com/dashboard',
      settingsUrl: 'https://railway.com/dashboard'
    }, 'nightly')).toEqual({
      value: '',
      placeholder: CLOUD_AUTO_UPDATE_UNKNOWN_LABEL,
      hint: 'Last known window: Nightly. Railway could not read image auto-updates: network down.',
      disabled: false
    })
  })

  it('F9 — a live minor policy is explained in the hint', () => {
    expect(cloudAutoUpdateSelectState({
      ok: true,
      mode: 'nightly',
      policy: 'minor'
    }, null)).toEqual({
      value: 'nightly',
      placeholder: CLOUD_AUTO_UPDATE_NOT_SET_LABEL,
      hint: CLOUD_AUTO_UPDATE_MINOR_HINT,
      disabled: false
    })
  })

  it('F11 — select state owns the actual busy disabled expression', () => {
    const live = { ok: true, mode: 'nightly', policy: 'patch' } as const
    expect(cloudAutoUpdateSelectState(live, null, false).disabled).toBe(false)
    expect(cloudAutoUpdateSelectState(live, null, true).disabled).toBe(true)
  })

  it('F1 — deploy auto-update fields become the shared feedback row model', () => {
    expect(cloudDeployAutoUpdateFeedback({
      ok: true,
      message: 'deployed',
      autoUpdateOk: false,
      autoUpdateMessage: 'patch commit rejected',
      autoUpdateSettingsUrl: 'https://railway.com/project/project/service/service/settings'
    })).toEqual({
      ok: false,
      text: 'patch commit rejected',
      settingsUrl: 'https://railway.com/project/project/service/service/settings'
    })
  })

  it('G3 — placeholder distinguishes loading, failed unknown, and successful not-set state', () => {
    expect(cloudAutoUpdateSelectState(null, null).placeholder)
      .toBe('Checking Railway…')
    expect(cloudAutoUpdateSelectState({
      ok: false,
      mode: null,
      policy: null
    }, null).placeholder).toBe('Current window unknown — choose one to set it')
    expect(cloudAutoUpdateSelectState({
      ok: true,
      mode: null,
      policy: null
    }, null).placeholder).toBe('Not set — choose a window')
  })

  it('G4 — target-keyed feedback never exposes a rejected write link on another service', () => {
    const snapshot = {
      target: 'logged-in|enabled|https://service-a.example',
      feedback: {
        ok: false,
        text: 'write rejected',
        settingsUrl: 'https://railway.com/project/p/service/a/settings'
      }
    }
    expect(cloudUpdateFeedbackForTarget(snapshot, snapshot.target))
      .toEqual(snapshot.feedback)
    expect(cloudUpdateFeedbackForTarget(
      snapshot,
      'logged-in|enabled|https://service-b.example'
    )).toBeNull()
  })

  it('H1 — first-deploy feedback keeps its post-deploy target', () => {
    const preDeployTarget = cloudAutoUpdateTargetKey(true, {
      enabled: false,
      serverUrl: ''
    })
    const postDeployTarget = cloudAutoUpdateTargetKey(true, {
      enabled: true,
      serverUrl: 'https://deployed.example'
    })
    const snapshot = {
      target: postDeployTarget,
      feedback: { ok: true, text: 'Railway image auto-updates set to Nightly.' }
    }

    expect(preDeployTarget).toBe('logged-in|disabled|')
    expect(postDeployTarget).toBe('logged-in|enabled|https://deployed.example')
    expect(cloudUpdateFeedbackAfterTargetChange(snapshot, postDeployTarget))
      .toBe(snapshot)
    expect(cloudUpdateFeedbackAfterTargetChange(snapshot, preDeployTarget))
      .toBeNull()
  })

  it('I6 / H6 — feedback clears only after the cloud profile save succeeds', () => {
    const snapshot = {
      target: 'logged-in|enabled|https://service.example',
      feedback: { ok: false, text: 'write rejected' }
    }

    expect(cloudUpdateFeedbackAfterSave(snapshot, false)).toBe(snapshot)
    expect(cloudUpdateFeedbackAfterSave(snapshot, true)).toBeNull()
  })

  it('I2 — a pending selection is shown while Railway saves it', () => {
    expect(cloudAutoUpdateSelectState({
      ok: true,
      mode: 'nightly',
      policy: 'patch',
      serviceId: 'service'
    }, null, true, 'weekends')).toEqual({
      value: 'weekends',
      placeholder: CLOUD_AUTO_UPDATE_NOT_SET_LABEL,
      hint: CLOUD_AUTO_UPDATE_SAVING_HINT,
      disabled: true
    })
  })

  it('I3 — a service B read failure cannot show service A cached state', () => {
    const serviceBFailure = {
      ok: false,
      mode: null,
      policy: null,
      serviceId: 'service-b',
      message: 'read failed'
    } as const
    const scoped = cloudAutoUpdateStoredMode(
      serviceBFailure,
      'nightly',
      'service-a',
      'service-a'
    )

    expect(scoped).toBeNull()
    expect(cloudAutoUpdateSelectState(serviceBFailure, scoped).hint)
      .toBe('read failed')
    expect(cloudAutoUpdateStoredMode(null, 'nightly', 'service-a', 'service-a'))
      .toBe('nightly')
  })

  it('I4 — cached copy describes a neutral last-known window', () => {
    expect(cloudAutoUpdateSelectState({
      ok: false,
      mode: null,
      policy: null
    }, 'nightly')).toMatchObject({
      value: '',
      placeholder: CLOUD_AUTO_UPDATE_UNKNOWN_LABEL,
      hint: `Last known window: Nightly. ${CLOUD_AUTO_UPDATE_FALLBACK_HINT}`
    })
    expect(CLOUD_AUTO_UPDATE_CUSTOM_HINT).toContain(
      'minor policy is preserved unless you choose Off'
    )
  })

  it('G8 — a stale read cannot replace the snapshot for a newly rendered target', () => {
    const current = {
      target: 'service-b',
      state: null
    }
    const staleState = { ok: true, mode: 'nightly', policy: 'patch' } as const
    expect(cloudAutoUpdateSnapshotAfterRead(current, 'service-a', staleState))
      .toBe(current)
    expect(cloudAutoUpdateSnapshotAfterRead(
      { target: 'service-a', state: null },
      'service-a',
      staleState
    )).toEqual({ target: 'service-a', state: staleState })
  })
})

describe('cloud update controls and feedback', () => {
  const status = (overrides: Partial<CloudUpdateStatus>): CloudUpdateStatus => ({
    status: 'unknown',
    current: null,
    latest: null,
    message: '',
    checking: false,
    applying: false,
    lastCheckedAt: null,
    canApply: false,
    ...overrides
  })

  it('offers a legacy update only when a latest release and safe apply path exist', () => {
    expect(cloudUpdateActionVisible(status({
      status: 'legacy', latest: '0.1.135', canApply: true
    }))).toBe(true)
    expect(cloudUpdateActionLabel(status({ status: 'legacy' })))
      .toBe('Update control plane')
    expect(cloudUpdateActionVisible(status({
      status: 'legacy', latest: '0.1.135', canApply: false
    }))).toBe(false)
  })

  it('styles status from the result flag rather than message text', () => {
    expect(cloudUpdateFeedbackClass({ ok: false, text: 'Railway denied it' }))
      .toContain('error-text')
    expect(cloudUpdateFeedbackClass({ ok: true, text: 'Could not be clearer' }))
      .toBe('row-sub')
  })

  it('D8 — keeps success visible until current status or ten seconds', () => {
    const available = status({ status: 'available', latest: '0.1.135', canApply: true })
    const result = { ok: true, target: '0.1.135', message: 'Updated.', shownAt: 1_000 }
    expect(cloudUpdateApplyResultVisible(available, result, 10_999)).toBe(true)
    expect(cloudUpdateApplyResultVisible(available, result, 11_000)).toBe(false)
    // The follow-up check flips the status to current within a second of a
    // successful update; the confirmation must stay readable for its window.
    expect(cloudUpdateApplyResultVisible({ ...available, status: 'current' }, result, 1_001))
      .toBe(true)
    expect(cloudUpdateApplyResultVisible({ ...available, status: 'current' }, result, 11_000))
      .toBe(false)
  })

  it('D8 — a status publication keeps a successful apply result and clears a failed one', () => {
    const success = { ok: true, target: '0.1.136', message: 'Updated to v0.1.136. 2 agents restored.', shownAt: 1_000 }
    const failure = { ok: false, message: 'Sign in to Railway before updating the cloud control plane.', shownAt: 1_000 }
    expect(cloudUpdateApplyFeedback(success, { type: 'status' })).toEqual(success)
    expect(cloudUpdateApplyFeedback(failure, { type: 'status' })).toBeNull()
    expect(cloudUpdateApplyFeedback(success, { type: 'dismiss' })).toBeNull()
  })

  it('H3 — makes failed apply feedback dismissible, status-scoped, and null-safe', () => {
    const unknown = status({
      status: 'unknown',
      latest: null,
      message: 'Could not check for updates.'
    })
    const failure = {
      ok: false,
      message: 'Cloud update failed: Railway denied it.',
      shownAt: 1_000
    }

    expect(cloudUpdateApplyResultVisible(unknown, failure, 2_000)).toBe(true)
    const copy = cloudUpdateBannerCopy(unknown, false, true, failure)
    expect(copy).toBe('Cloud update failed: Railway denied it.')
    expect(copy).not.toContain('vnull')
    expect(cloudUpdateApplyFeedback(failure, { type: 'dismiss' })).toBeNull()
    expect(cloudUpdateApplyFeedback(failure, { type: 'status' })).toBeNull()
  })
})

describe('package update timestamps', () => {
  it('keeps explicit checks separate from maintenance completion', () => {
    expect(packageUpdateTimestamps({
      enabled: true,
      reason: '',
      interval: '6h0m0s',
      boot_pass_completed: true,
      hosting: 'railway',
      next_run_at: '',
      last_run: {
        started_at: 'maintenance-start',
        finished_at: 'maintenance-finish',
        checked: 1,
        updated: [],
        restored: [],
        skipped: [],
        errors: []
      }
    }, 'manual-check')).toEqual({
      lastCheck: 'manual-check',
      lastMaintenance: 'maintenance-finish'
    })
  })

  it('seeds the last check from the newest package snapshot result', () => {
    expect(latestPackageUpdateCheckedAt([
      {
        name: 'older', version: '', description: '', status: '', path: null, port: null, pid: null,
        update: { status: 'current', latestCommit: '', checkedAt: '2026-08-24T01:00:00Z', message: '' }
      },
      {
        name: 'newer', version: '', description: '', status: '', path: null, port: null, pid: null,
        update: { status: 'current', latestCommit: '', checkedAt: '2026-08-24T03:00:00Z', message: '' }
      },
      { name: 'legacy', version: '', description: '', status: '', path: null, port: null, pid: null }
    ])).toBe('2026-08-24T03:00:00Z')
  })
})

describe('agent update chips', () => {
  const agent = {
    name: 'custom-agent',
    badge: 'stopped',
    version: '1',
    description: '',
    status: 'stopped',
    path: null,
    port: null,
    pid: null
  } as SnapshotAgent

  it('labels available, pinned, and paused packages', () => {
    expect(agentUpdateChip({ ...agent, update: { status: 'available', latestCommit: 'new', checkedAt: '', message: '' } })).toBe('Update available')
    expect(agentUpdateChip({ ...agent, update: { status: 'pinned', latestCommit: 'new', checkedAt: '', message: '' } })).toBe('Pinned')
    expect(agentUpdateChip({ ...agent, autoUpdate: false })).toBe('Paused')
    // A deferred or errored check must not make a pending update disappear.
    expect(agentUpdateChip({ ...agent, update: { status: 'deferred', latestCommit: 'new', checkedAt: '', message: 'active executions did not finish' } }))
      .toBe('Update waiting for the node to be idle')
    expect(agentUpdateChip({ ...agent, update: { status: 'error', latestCommit: '', checkedAt: '', message: 'git ls-remote failed' } }))
      .toBe('Update check failed')
    expect(agentUpdateChipTitle({ ...agent, update: { status: 'error', latestCommit: '', checkedAt: '', message: 'git ls-remote failed' } }))
      .toBe('git ls-remote failed')
  })

  it('D10 — shows failed unattended updates while retaining manual update', () => {
    const failed = {
      ...agent,
      update: {
        status: 'failed' as const,
        latestCommit: 'new',
        checkedAt: '',
        message: 'clone failed'
      }
    }
    expect(agentUpdateChip(failed)).toBe('Update failed')
    expect(agentUpdateChip({ ...failed, autoUpdate: false })).toBe('Update failed')
    expect(agentUpdateChipTitle(failed)).toBe('clone failed')
    expect(agentManualUpdateActionVisible(failed)).toBe(true)
  })

  it('hides pause and resume when an older control plane omits autoUpdate', () => {
    const legacyPackage: PackageInfo = {
      id: 'legacy',
      name: 'legacy',
      version: '1',
      status: 'stopped',
      install_status: 'stopped',
      install_path: '/tmp/legacy',
      configuration_required: false,
      configuration_complete: true,
      description: '',
      author: ''
    }
    expect(agentAutoUpdateActionVisible({
      ...packageToInstalledAgent(legacyPackage),
      badge: 'stopped'
    })).toBe(false)
    expect(agentAutoUpdateActionVisible({ ...agent, autoUpdate: true })).toBe(true)
    expect(agentAutoUpdateActionVisible({ ...agent, autoUpdate: false })).toBe(true)
  })

  it('D6 — shows the global restart strip only for restart-required snapshots', () => {
    expect(localControlPlaneRestartVisible(null)).toBe(false)
    expect(localControlPlaneRestartVisible({
      at: '', ok: true, restarted: false, status: 'not_required', message: ''
    })).toBe(false)
    expect(localControlPlaneRestartVisible({
      at: '', ok: false, restarted: false, status: 'restart_required', message: 'Restart it.'
    })).toBe(true)
  })
})
