import { spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  type CodeafDeps,
  defaultCodeafDeps,
  ensureCodeafCompanion,
  planCodeaf,
  resetCodeafCompanion
} from './codeaf-companion'

vi.mock('node:child_process', () => ({ spawn: vi.fn() }))

const baseState = { cliCommand: '/managed/af', skipEnv: undefined, alreadyRan: false }

describe('planCodeaf', () => {
  it('honors truthy canonical and legacy skip switches', () => {
    for (const value of ['1', 'true', 'TRUE', 'Yes', 'on', ' ON ']) {
      expect(planCodeaf({ ...baseState, skipEnv: value }).run).toBe(false)
      expect(planCodeaf({ ...baseState, legacySkipEnv: value }).run).toBe(false)
    }
    for (const value of ['0', 'false', 'no', 'off', '', undefined]) {
      expect(planCodeaf({ ...baseState, skipEnv: value, legacySkipEnv: value }).run).toBe(true)
    }
  })

  it('skips when the CLI is null, empty, or whitespace', () => {
    for (const cliCommand of [null, '', '   ']) {
      expect(planCodeaf({ ...baseState, cliCommand })).toEqual({
        run: false,
        reason: 'no usable af CLI — skipping codeaf provisioning'
      })
    }
  })

  it('skips when codeaf already ran', () => {
    expect(planCodeaf({ ...baseState, alreadyRan: true })).toEqual({
      run: false,
      reason: 'codeaf already provisioned this launch'
    })
  })

  it('runs when no skip condition applies', () => {
    expect(planCodeaf(baseState)).toEqual({
      run: true,
      reason: 'provisioning codeaf via af codeaf ensure'
    })
  })

  it('applies skip env, already-ran, then missing-CLI precedence', () => {
    expect(planCodeaf({ cliCommand: null, skipEnv: '1', alreadyRan: true }).reason).toContain(
      'AGENTFIELD_SKIP_CODEAF'
    )
    expect(planCodeaf({ cliCommand: null, skipEnv: undefined, alreadyRan: true }).reason).toBe(
      'codeaf already provisioned this launch'
    )
  })
})

function fakeDeps(
  result: { code: number; stdout: string; stderr: string } = {
    code: 0,
    stdout: '',
    stderr: ''
  }
): CodeafDeps & { run: ReturnType<typeof vi.fn> } {
  return {
    run: vi.fn(async () => result),
    cliCommand: () => '/managed/af',
    env: () => undefined
  }
}

describe('ensureCodeafCompanion', () => {
  beforeEach(() => resetCodeafCompanion())

  it('runs codeaf ensure exactly once on the happy path', async () => {
    const deps = fakeDeps()
    await expect(ensureCodeafCompanion(deps)).resolves.toEqual({
      ok: true,
      message: 'codeaf is provisioned'
    })
    expect(deps.run).toHaveBeenCalledExactlyOnceWith('/managed/af', ['codeaf', 'ensure'])
  })

  it('reports a non-zero exit with stderr', async () => {
    const deps = fakeDeps({ code: 7, stdout: 'fallback', stderr: ' download failed \n' })
    const result = await ensureCodeafCompanion(deps)
    expect(result.ok).toBe(false)
    expect(result.message).toContain('exit 7')
    expect(result.message).toContain('download failed')
  })

  it('captures a thrown runner error', async () => {
    const deps = fakeDeps()
    deps.run.mockRejectedValueOnce(new Error('spawn exploded'))
    await expect(ensureCodeafCompanion(deps)).resolves.toMatchObject({ ok: false })
  })

  it('does not run again on a second call in the same process', async () => {
    const deps = fakeDeps()
    await ensureCodeafCompanion(deps)
    await expect(ensureCodeafCompanion(deps)).resolves.toEqual({
      ok: true,
      message: 'codeaf already provisioned this launch'
    })
    expect(deps.run).toHaveBeenCalledTimes(1)
  })

  it.each(['AGENTFIELD_SKIP_CODEAF', 'AGENTFIELD_SKIP_AFORGE'])(
    'does not run when %s opts out',
    async (skipName) => {
      const deps = fakeDeps()
      deps.env = (name) => (name === skipName ? 'YES' : undefined)
      await expect(ensureCodeafCompanion(deps)).resolves.toMatchObject({ ok: true })
      expect(deps.run).not.toHaveBeenCalled()
    }
  )
})

// Exercise the production runner with child-process events, without spawning a CLI.
describe('defaultCodeafDeps runner', () => {
  it('collects output and tolerates duplicate close/error events', async () => {
    const { child, kill } = fakeChild()
    vi.mocked(spawn).mockReturnValueOnce(child as never)
    const result = defaultCodeafDeps().run('/fake/af', ['codeaf', 'ensure'])
    child.stdout.emit('data', Buffer.from('installed'))
    child.stderr.emit('data', Buffer.from('diagnostic'))
    child.emit('close', 0)
    child.emit('error', new Error('late error'))
    await expect(result).resolves.toEqual({ code: 0, stdout: 'installed', stderr: 'diagnostic' })
    expect(kill).not.toHaveBeenCalled()
  })

  it('handles child spawn errors and null exit codes', async () => {
    for (const event of ['error', 'close']) {
      const { child } = fakeChild()
      vi.mocked(spawn).mockReturnValueOnce(child as never)
      const result = defaultCodeafDeps().run('/fake/af', ['codeaf', 'ensure'])
      child.emit(event, event === 'error' ? new Error('spawn error') : null)
      await expect(result).resolves.toMatchObject({ code: -1 })
    }
  })

  it('bounds a stalled download to ten minutes', async () => {
    vi.useFakeTimers()
    try {
      const { child, kill } = fakeChild()
      vi.mocked(spawn).mockReturnValueOnce(child as never)
      const result = defaultCodeafDeps().run('/fake/af', ['codeaf', 'ensure'])
      await vi.advanceTimersByTimeAsync(10 * 60 * 1_000)
      await expect(result).resolves.toMatchObject({ code: -1 })
      expect(kill).toHaveBeenCalledOnce()
      child.emit('close', -1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('turns a synchronous spawn failure into an error result', async () => {
    vi.mocked(spawn).mockImplementationOnce(() => { throw new Error('cannot spawn') })
    await expect(defaultCodeafDeps().run('/fake/af', [])).resolves.toMatchObject({ code: -1 })
  })
})

function fakeChild() {
  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(),
    stderr: new EventEmitter(),
    kill: vi.fn()
  })
  return { child, kill: child.kill }
}
