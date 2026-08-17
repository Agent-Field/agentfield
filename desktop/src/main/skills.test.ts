import { EventEmitter } from 'node:events'
import { spawn } from 'node:child_process'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DesktopSettings, SkillSyncRecord } from '../shared/types'
import {
  SKILL_SYNC_ARGS,
  SkillSync,
  buildSkillSyncRecord,
  defaultSkillSyncDeps,
  formatSkillSyncLogLine,
  lastLine,
  shouldSyncOnCliUpdate,
  shouldSyncOnLaunch,
  shouldSyncOnSettingsChange,
  type SkillSyncDeps,
  type SkillSyncOutcome
} from './skills'

// The default runner is the one seam DI can't cover — it IS the spawn. vi.mock
// is hoisted above the imports, so `spawn` above is the mock.
vi.mock('node:child_process', () => ({ spawn: vi.fn() }))

// ---- test doubles ------------------------------------------------------------

interface Harness {
  deps: SkillSyncDeps
  /** Every (command, args) pair the sync ran. */
  runs: Array<{ command: string; args: readonly string[] }>
  /** Every line appended to the log. */
  logged: string[]
}

function harness(
  outcomes: SkillSyncOutcome | SkillSyncOutcome[],
  overrides: Partial<SkillSyncDeps> = {}
): Harness {
  const queue = Array.isArray(outcomes) ? [...outcomes] : [outcomes]
  const runs: Harness['runs'] = []
  const logged: string[] = []
  let tick = 0
  return {
    runs,
    logged,
    deps: {
      command: () => 'af',
      run: async (command, args) => {
        runs.push({ command, args })
        return queue.length > 1 ? queue.shift()! : queue[0]
      },
      appendLog: async (line) => {
        logged.push(line)
      },
      now: () => new Date(Date.UTC(2026, 0, 1, 0, 0, tick++)),
      ...overrides
    }
  }
}

function ok(output = 'installed 3 skills'): SkillSyncOutcome {
  return { code: 0, output, error: null }
}

// ---- pure helpers ------------------------------------------------------------

describe('lastLine', () => {
  it('takes the last non-empty line, trimmed', () => {
    expect(lastLine('checking targets\n  installed 3 skills  \n\n')).toBe('installed 3 skills')
  })

  it('handles CRLF output and blank input', () => {
    expect(lastLine('a\r\nb\r\n')).toBe('b')
    expect(lastLine('   \n\n')).toBe('')
  })

  it('caps a runaway line with an ellipsis', () => {
    expect(lastLine('x'.repeat(50), 10)).toBe(`${'x'.repeat(9)}…`)
  })
})

describe('buildSkillSyncRecord', () => {
  const at = '2026-01-01T00:00:00.000Z'

  it('exit 0 is a successful sync summarized by the CLI last line', () => {
    expect(buildSkillSyncRecord(at, ok('installed 3 skills for claude-code'))).toEqual({
      at,
      ok: true,
      exitCode: 0,
      message: 'installed 3 skills for claude-code'
    })
  })

  it('exit 0 with no output still reads as up to date', () => {
    expect(buildSkillSyncRecord(at, ok(''))).toEqual({
      at,
      ok: true,
      exitCode: 0,
      message: 'skills up to date'
    })
  })

  it('a non-zero exit is a FAILED sync carrying the code and the CLI message', () => {
    const record = buildSkillSyncRecord(at, {
      code: 1,
      output: 'installing…\nerror: codex target not writable',
      error: null
    })
    expect(record.ok).toBe(false)
    expect(record.exitCode).toBe(1)
    expect(record.message).toBe('af skill install exited 1 — error: codex target not writable')
  })

  it('a spawn error is a failed sync with no exit code', () => {
    const record = buildSkillSyncRecord(at, {
      code: null,
      output: '',
      error: 'spawn af ENOENT'
    })
    expect(record).toEqual({
      at,
      ok: false,
      exitCode: null,
      message: 'could not run af skill install: spawn af ENOENT'
    })
  })
})

describe('formatSkillSyncLogLine', () => {
  it('flattens the run onto one line with timestamp and exit code', () => {
    const record: SkillSyncRecord = {
      at: '2026-01-01T00:00:00.000Z',
      ok: false,
      exitCode: 1,
      message: 'af skill install exited 1 — boom'
    }
    expect(formatSkillSyncLogLine(record, 'starting\n\nboom\n')).toBe(
      '2026-01-01T00:00:00.000Z exit=1 ok=false starting | boom'
    )
  })

  it('falls back to the summary when the run printed nothing', () => {
    const record: SkillSyncRecord = {
      at: '2026-01-01T00:00:00.000Z',
      ok: false,
      exitCode: null,
      message: 'could not run af skill install: spawn af ENOENT'
    }
    expect(formatSkillSyncLogLine(record, '')).toBe(
      '2026-01-01T00:00:00.000Z exit=none ok=false could not run af skill install: spawn af ENOENT'
    )
  })
})

// ---- SkillSync ---------------------------------------------------------------

describe('SkillSync', () => {
  it('reports no sync before the first run', () => {
    expect(new SkillSync(harness(ok()).deps).last()).toBeNull()
  })

  it('runs the non-interactive catalog install and records the success', async () => {
    const h = harness(ok('installed 3 skills'))
    const sync = new SkillSync(h.deps)

    const record = await sync.sync()

    expect(h.runs).toEqual([{ command: 'af', args: SKILL_SYNC_ARGS }])
    expect(record.ok).toBe(true)
    expect(sync.last()).toEqual(record)
    expect(h.logged).toEqual(['2026-01-01T00:00:00.000Z exit=0 ok=true installed 3 skills'])
  })

  it('records a non-zero exit as a failed sync instead of swallowing it', async () => {
    const h = harness({ code: 2, output: 'error: no writable target', error: null })
    const sync = new SkillSync(h.deps)

    const record = await sync.sync()

    expect(record.ok).toBe(false)
    expect(record.exitCode).toBe(2)
    expect(record.message).toContain('error: no writable target')
    expect(sync.last()).toEqual(record)
    expect(h.logged[0]).toContain('exit=2 ok=false')
  })

  it('records a spawn error as a failed sync and never throws', async () => {
    const h = harness({ code: null, output: '', error: 'spawn af ENOENT' })
    const sync = new SkillSync(h.deps)

    const record = await sync.sync()

    expect(record).toMatchObject({ ok: false, exitCode: null })
    expect(record.message).toContain('spawn af ENOENT')
  })

  it('captures a rejecting runner rather than leaking the rejection', async () => {
    const h = harness(ok(), {
      run: async () => {
        throw new Error('runner blew up')
      }
    })

    const record = await new SkillSync(h.deps).sync()

    expect(record.ok).toBe(false)
    expect(record.message).toContain('runner blew up')
  })

  it('still records the sync when the log file cannot be written', async () => {
    const h = harness(ok(), {
      appendLog: async () => {
        throw new Error('EACCES')
      }
    })
    const sync = new SkillSync(h.deps)

    await expect(sync.sync()).resolves.toMatchObject({ ok: true })
    expect(sync.last()?.ok).toBe(true)
  })

  it('serializes concurrent triggers onto a single run', async () => {
    let release: (() => void) | null = null
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const h = harness(ok(), {
      run: async (command, args) => {
        h.runs.push({ command, args })
        await gate
        return ok()
      }
    })
    const sync = new SkillSync(h.deps)

    const first = sync.sync()
    const second = sync.sync()
    expect(sync.isRunning()).toBe(true)
    release!()
    const [a, b] = await Promise.all([first, second])

    expect(h.runs).toHaveLength(1)
    expect(a).toBe(b)
    expect(sync.isRunning()).toBe(false)
  })

  it('runs again once the previous sync finished', async () => {
    const h = harness([ok('first'), ok('second')])
    const sync = new SkillSync(h.deps)

    await sync.sync()
    const record = await sync.sync()

    expect(h.runs).toHaveLength(2)
    expect(record.message).toBe('second')
    expect(sync.last()?.message).toBe('second')
  })

  it('resolves the af command per run, so CLI updates are picked up', async () => {
    let command = 'af'
    const h = harness(ok(), { command: () => command })
    const sync = new SkillSync(h.deps)

    await sync.sync()
    command = '/Users/x/.agentfield/bin/af'
    await sync.sync()

    expect(h.runs.map((r) => r.command)).toEqual(['af', '/Users/x/.agentfield/bin/af'])
  })
})

// ---- default runner (the real spawn seam) ------------------------------------

/** Minimal ChildProcess stand-in: emits on stdout/stderr, then close/error. */
function fakeChild(): EventEmitter & {
  stdout: EventEmitter
  stderr: EventEmitter
  kill: () => void
} {
  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(),
    stderr: new EventEmitter(),
    kill: vi.fn()
  })
  return child
}

describe('defaultSkillSyncDeps().run', () => {
  beforeEach(() => {
    vi.mocked(spawn).mockReset()
  })
  afterEach(() => {
    vi.mocked(spawn).mockReset()
  })

  it('pipes stdio and captures stdout, stderr and the exit code', async () => {
    const child = fakeChild()
    vi.mocked(spawn).mockReturnValue(child as never)

    const pending = defaultSkillSyncDeps('/tmp/does-not-matter.log').run('af', SKILL_SYNC_ARGS)
    child.stdout.emit('data', Buffer.from('installing\n'))
    child.stderr.emit('data', Buffer.from('warn: codex not found\n'))
    child.emit('close', 1)

    await expect(pending).resolves.toEqual({
      code: 1,
      output: 'installing\nwarn: codex not found',
      error: null
    })
    const [, args, options] = vi.mocked(spawn).mock.calls[0]
    expect(args).toEqual([...SKILL_SYNC_ARGS])
    expect(options).toMatchObject({ windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] })
  })

  it('turns a spawn error into an outcome instead of an unhandled event', async () => {
    const child = fakeChild()
    vi.mocked(spawn).mockReturnValue(child as never)

    const pending = defaultSkillSyncDeps('/tmp/does-not-matter.log').run('af', SKILL_SYNC_ARGS)
    child.emit('error', new Error('spawn af ENOENT'))

    await expect(pending).resolves.toEqual({ code: null, output: '', error: 'spawn af ENOENT' })
  })
})

// ---- trigger conditions ------------------------------------------------------

function settings(installSkills: boolean): Pick<DesktopSettings, 'installSkills'> {
  return { installSkills }
}

describe('sync triggers', () => {
  it('launch syncs only when skills are enabled', () => {
    expect(shouldSyncOnLaunch(settings(true))).toBe(true)
    expect(shouldSyncOnLaunch(settings(false))).toBe(false)
  })

  it('a settings update syncs only on the off → on transition', () => {
    expect(shouldSyncOnSettingsChange(settings(false), settings(true))).toBe(true)
    expect(shouldSyncOnSettingsChange(settings(true), settings(true))).toBe(false)
    expect(shouldSyncOnSettingsChange(settings(true), settings(false))).toBe(false)
    expect(shouldSyncOnSettingsChange(settings(false), settings(false))).toBe(false)
  })

  it('a CLI update syncs only after a successful install with skills enabled', () => {
    expect(shouldSyncOnCliUpdate(true, settings(true))).toBe(true)
    expect(shouldSyncOnCliUpdate(false, settings(true))).toBe(false)
    expect(shouldSyncOnCliUpdate(true, settings(false))).toBe(false)
  })
})
