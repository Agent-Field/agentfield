import { describe, expect, it, vi } from 'vitest'
import type { ControlPlaneVersion } from '../shared/types'
import { restartAdoptedControlPlaneAfterCliSwap } from './localCpUpdate'

function version(value: string): ControlPlaneVersion {
  return { version: value, commit: '', build_date: '', hosting: { platform: 'local' }, features: [] }
}

function deps(reported: ControlPlaneVersion | null) {
  return {
    getVersion: vi.fn(async () => reported)
  }
}

describe('local control-plane restart after a managed CLI swap', () => {
  it.each(['win32', 'linux'] as const)('does not spawn beside an older adopted server on %s', async (platform) => {
    const d = deps(version('0.1.134'))
    const result = await restartAdoptedControlPlaneAfterCliSwap({
      managedBinaryReplaced: true,
      platform,
      cloudEnabled: false,
      autostart: { kind: 'adopted', port: 8083 },
      cliVersion: '0.1.135'
    }, d)

    expect(result).toMatchObject({
      ok: false,
      restarted: false,
      status: 'restart_required',
      message: 'AgentField CLI updated to v0.1.135. Restart the control plane to use it.'
    })
  })

  it('treats a missing version endpoint on an adopted Windows server as restart-required', async () => {
    const d = deps(null)
    await restartAdoptedControlPlaneAfterCliSwap({
      managedBinaryReplaced: true,
      platform: 'win32',
      cloudEnabled: false,
      autostart: { kind: 'adopted', port: 8080 },
      cliVersion: '0.1.135'
    }, d)
    expect(d.getVersion).toHaveBeenCalledTimes(1)
  })

  it('claims a safe restart only after the reported version changes', async () => {
    const getVersion = vi
      .fn()
      .mockResolvedValueOnce(version('0.1.134'))
      .mockResolvedValueOnce(version('0.1.134'))
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce(version('0.1.135'))
    const restartControlPlane = vi.fn(async () => ({ ok: true, message: 'restarted' }))
    let clock = 0
    const sleep = vi.fn(async (milliseconds: number) => { clock += milliseconds })
    const result = await restartAdoptedControlPlaneAfterCliSwap({
      managedBinaryReplaced: true,
      platform: 'linux',
      cloudEnabled: false,
      autostart: { kind: 'adopted', port: 8083 },
      cliVersion: '0.1.135'
    }, { getVersion, restartControlPlane, now: () => clock, sleep })

    expect(restartControlPlane).toHaveBeenCalledWith(8083)
    expect(getVersion).toHaveBeenCalledTimes(4)
    expect(sleep).toHaveBeenCalledTimes(2)
    expect(sleep).toHaveBeenCalledWith(1_000)
    expect(result).toMatchObject({ ok: true, restarted: true, status: 'restarted' })
  })

  it('stops polling after the bounded restart-verification deadline', async () => {
    const getVersion = vi.fn(async () => version('0.1.134'))
    let clock = 0
    const sleep = vi.fn(async (milliseconds: number) => { clock += milliseconds })
    const result = await restartAdoptedControlPlaneAfterCliSwap({
      managedBinaryReplaced: true,
      platform: 'linux',
      cloudEnabled: false,
      autostart: { kind: 'adopted', port: 8083 },
      cliVersion: '0.1.135'
    }, {
      getVersion,
      restartControlPlane: vi.fn(async () => ({ ok: true, message: 'restarted' })),
      now: () => clock,
      sleep
    })

    expect(result).toMatchObject({ restarted: false, status: 'restart_required' })
    expect(clock).toBe(30_000)
    expect(sleep).toHaveBeenCalledTimes(30)
  })

  it.each([
    { label: 'macOS', platform: 'darwin' as const, replaced: true, autostart: { kind: 'adopted' as const, port: 8080 }, reported: version('0.1.134') },
    { label: 'unchanged CLI', platform: 'win32' as const, replaced: false, autostart: { kind: 'adopted' as const, port: 8080 }, reported: version('0.1.134') },
    { label: 'newly started server', platform: 'linux' as const, replaced: true, autostart: { kind: 'started' as const, port: 8080 }, reported: version('0.1.134') },
    { label: 'same server version', platform: 'linux' as const, replaced: true, autostart: { kind: 'adopted' as const, port: 8080 }, reported: version('0.1.135') }
  ])('does not restart for $label', async ({ platform, replaced, autostart, reported }) => {
    const d = deps(reported)
    const result = await restartAdoptedControlPlaneAfterCliSwap({
      managedBinaryReplaced: replaced,
      platform,
      cloudEnabled: false,
      autostart,
      cliVersion: '0.1.135'
    }, d)
    expect(result.restarted).toBe(false)
    expect(result.status).toBe('not_required')
  })
})
