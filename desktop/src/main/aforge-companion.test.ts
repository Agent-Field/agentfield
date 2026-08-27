import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  type AforgeDeps,
  ensureAforgeCompanion,
  planAforge,
  resetAforgeCompanion
} from './aforge-companion'

const baseState = { cliCommand: '/managed/af', skipEnv: undefined, alreadyRan: false }

describe('planAforge', () => {
  it('skips only when AGENTFIELD_SKIP_AFORGE is exactly 1', () => {
    expect(planAforge({ ...baseState, skipEnv: '1' })).toEqual({
      run: false,
      reason: 'AGENTFIELD_SKIP_AFORGE=1 — skipping aforge provisioning'
    })
    expect(planAforge({ ...baseState, skipEnv: '0' }).run).toBe(true)
    expect(planAforge({ ...baseState, skipEnv: '' }).run).toBe(true)
    expect(planAforge(baseState).run).toBe(true)
  })

  it('skips when the CLI is null, empty, or whitespace', () => {
    for (const cliCommand of [null, '', '   ']) {
      expect(planAforge({ ...baseState, cliCommand })).toEqual({
        run: false,
        reason: 'no usable af CLI — skipping aforge provisioning'
      })
    }
  })

  it('skips when aforge already ran', () => {
    expect(planAforge({ ...baseState, alreadyRan: true })).toEqual({
      run: false,
      reason: 'aforge already provisioned this launch'
    })
  })

  it('runs when no skip condition applies', () => {
    expect(planAforge(baseState)).toEqual({
      run: true,
      reason: 'provisioning aforge via af aforge ensure'
    })
  })

  it('applies skip env, already-ran, then missing-CLI precedence', () => {
    expect(planAforge({ cliCommand: null, skipEnv: '1', alreadyRan: true }).reason).toContain(
      'AGENTFIELD_SKIP_AFORGE'
    )
    expect(planAforge({ cliCommand: null, skipEnv: undefined, alreadyRan: true }).reason).toBe(
      'aforge already provisioned this launch'
    )
  })
})

function fakeDeps(
  result: { code: number; stdout: string; stderr: string } = {
    code: 0,
    stdout: '',
    stderr: ''
  }
): AforgeDeps & { run: ReturnType<typeof vi.fn> } {
  return {
    run: vi.fn(async () => result),
    cliCommand: () => '/managed/af',
    env: () => undefined
  }
}

describe('ensureAforgeCompanion', () => {
  beforeEach(() => resetAforgeCompanion())

  it('runs aforge ensure exactly once on the happy path', async () => {
    const deps = fakeDeps()
    await expect(ensureAforgeCompanion(deps)).resolves.toEqual({
      ok: true,
      message: 'aforge is provisioned'
    })
    expect(deps.run).toHaveBeenCalledExactlyOnceWith('/managed/af', ['aforge', 'ensure'])
  })

  it('reports a non-zero exit with stderr', async () => {
    const deps = fakeDeps({ code: 7, stdout: 'fallback', stderr: ' download failed \n' })
    const result = await ensureAforgeCompanion(deps)
    expect(result.ok).toBe(false)
    expect(result.message).toContain('exit 7')
    expect(result.message).toContain('download failed')
  })

  it('captures a thrown runner error', async () => {
    const deps = fakeDeps()
    deps.run.mockRejectedValueOnce(new Error('spawn exploded'))
    await expect(ensureAforgeCompanion(deps)).resolves.toMatchObject({ ok: false })
  })

  it('does not run again on a second call in the same process', async () => {
    const deps = fakeDeps()
    await ensureAforgeCompanion(deps)
    await expect(ensureAforgeCompanion(deps)).resolves.toEqual({
      ok: true,
      message: 'aforge already provisioned this launch'
    })
    expect(deps.run).toHaveBeenCalledTimes(1)
  })

  it('does not run when the injected environment opts out', async () => {
    const deps = fakeDeps()
    deps.env = (name) => (name === 'AGENTFIELD_SKIP_AFORGE' ? '1' : undefined)
    await expect(ensureAforgeCompanion(deps)).resolves.toMatchObject({ ok: true })
    expect(deps.run).not.toHaveBeenCalled()
  })
})
