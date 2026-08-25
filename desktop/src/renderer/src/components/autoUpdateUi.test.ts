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
  cloudUpdateActionLabel,
  cloudUpdateActionVisible,
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
