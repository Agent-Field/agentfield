import { describe, expect, it, vi } from 'vitest'
import { runDesktopBootChain } from './bootChain'

function deps() {
  return {
    userPathReady: Promise.resolve(),
    runAutostart: vi.fn(async () => ({ kind: 'started' as const, port: 8080 })),
    recoverAutostartFailure: vi.fn(() => ({ kind: 'skipped' as const })),
    afterAutostart: vi.fn(async () => {}),
    provisionBundledAgents: vi.fn(async () => {}),
    checkPackageUpdates: vi.fn(async () => {}),
    log: vi.fn(),
    warn: vi.fn(),
    error: vi.fn()
  }
}

describe('desktop boot chain', () => {
  it('recovers a rejected autostart and still provisions and checks packages', async () => {
    const d = deps()
    const failure = new Error('autostart rejected')
    d.runAutostart.mockRejectedValue(failure)

    await runDesktopBootChain(d)

    expect(d.recoverAutostartFailure).toHaveBeenCalledWith(failure)
    expect(d.afterAutostart).toHaveBeenCalledWith({ kind: 'skipped' })
    expect(d.provisionBundledAgents).toHaveBeenCalledOnce()
    expect(d.checkPackageUpdates).toHaveBeenCalledOnce()
    expect(d.provisionBundledAgents).toHaveBeenCalledBefore(d.checkPackageUpdates)
  })

  it('logs a provisioning rejection and still performs the package check', async () => {
    const d = deps()
    const failure = new Error('bundle install failed')
    d.provisionBundledAgents.mockRejectedValue(failure)

    await runDesktopBootChain(d)

    expect(d.error).toHaveBeenCalledWith('bundled provisioning failed:', failure)
    expect(d.checkPackageUpdates).toHaveBeenCalledOnce()
  })
})
