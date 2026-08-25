import { describe, expect, it } from 'vitest'
import type { CloudUpdateStatus, RailwayStatus, SnapshotAgent } from '../../../shared/types'
import { packageToInstalledAgent } from '../../../main/agentfield'
import type { PackageInfo } from '../../../main/cpClient'
import {
  cloudUpdateBannerActionVisible,
  cloudUpdateBannerText,
  cloudUpdateBannerVisible
} from './CloudUpdateBanner'
import {
  cloudUpdateActionLabel,
  cloudUpdateActionVisible,
  cloudUpdateFeedbackClass,
  deployedWorkspacePickerVisible,
  deploymentActionWorkspaceId
} from './CloudPanel'
import {
  agentAutoUpdateActionVisible,
  agentUpdateChip,
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
    expect(deploymentActionWorkspaceId(railway, 'workspace-1')).toBe('workspace-1')
    expect(deploymentActionWorkspaceId({ ...railway, workspaces: [] }, ''))
      .toBe('workspace-2')
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
})

describe('package update timestamps', () => {
  it('keeps explicit checks separate from maintenance completion', () => {
    expect(packageUpdateTimestamps({
      enabled: true,
      reason: '',
      interval: '6h0m0s',
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

  it('shows the manual restart strip only for restart-required snapshots', () => {
    expect(localControlPlaneRestartVisible(null)).toBe(false)
    expect(localControlPlaneRestartVisible({
      at: '', ok: true, restarted: false, status: 'not_required', message: ''
    })).toBe(false)
    expect(localControlPlaneRestartVisible({
      at: '', ok: false, restarted: false, status: 'restart_required', message: 'Restart it.'
    })).toBe(true)
  })
})
