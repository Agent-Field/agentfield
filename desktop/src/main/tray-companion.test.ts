import { describe, expect, it, vi } from 'vitest'
import { syncTrayCompanion } from './tray-companion'

const bundled = '/Applications/AgentField.app/Contents/MacOS/af-tray'

describe('Desktop macOS tray lifecycle', () => {
  it('uses the bundled helper and lets it detect the legacy startup setting on upgrade', async () => {
    const run = vi.fn(async () => 'tray-at-login=true\n')
    await expect(syncTrayCompanion(bundled, true, null, run)).resolves.toEqual({ atLogin: true })
    expect(run.mock.calls).toEqual([[bundled, ['desktop-tray', 'show', 'auto']]])
  })

  it('hides the tray through the tray-only command and keeps the login preference', async () => {
    const run = vi.fn(async () => 'tray-at-login=true\n')
    await expect(syncTrayCompanion(bundled, false, true, run)).resolves.toEqual({ atLogin: true })
    expect(run.mock.calls).toEqual([[bundled, ['desktop-tray', 'hide', 'login']]])
  })

  it('allows a visible tray that only runs in the current login session', async () => {
    const run = vi.fn(async () => 'tray-at-login=false\n')
    await expect(syncTrayCompanion(bundled, true, false, run)).resolves.toEqual({ atLogin: false })
    expect(run.mock.calls).toEqual([[bundled, ['desktop-tray', 'show', 'session']]])
  })

  it('reports a helper failure so an incomplete migration can be retried', async () => {
    await expect(syncTrayCompanion(bundled, true, null, async () => '')).rejects.toThrow(
      'The tray helper did not report its login setting'
    )
  })
})
