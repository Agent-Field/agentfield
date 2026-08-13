// Keep the AgentField skills present in detected coding agents (Claude Code,
// Codex, Gemini, …) by running `af skill install --non-interactive`, and — the
// point of this module — remember whether that actually worked.
//
// A no-name `af skill install` installs the CLI's ENTIRE skill catalog
// (builder, personal-agent, consumer, and whatever ships next), so new catalog
// skills reach existing installs without a desktop release — hardcoding names
// here is how agentfield-personal was silently missed. It is idempotent
// (skillkit tracks versions in ~/.agentfield/skills/.state.json), so re-running
// it on settings changes and CLI updates is cheap.
//
// It used to be fire-and-forget with stdio:'ignore' and a swallowed 'error'
// handler: a failed sync was indistinguishable from a successful one, and the
// dashboard cheerfully claimed "Installed for coding agents" from a settings
// boolean. Now the run is captured (stdout+stderr, exit code), summarized into
// a SkillSyncRecord the renderer can render honestly, and appended to a log
// file for the times the one-line summary is not enough. `af skill install`
// exits non-zero when any skill/target fails, so a non-zero exit is a failed
// sync, full stop.
//
// One sync at a time (skillkit's state file is not concurrency-safe), enforced
// by an in-flight promise: overlapping triggers all await the same run.
//
// No electron imports — the log-file path is injected by index.ts — so this
// stays unit-testable under plain vitest.

import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'
import type { DesktopSettings, SkillSyncRecord } from '../shared/types'
import { getCliCommand } from './cli'
import { childEnv } from './env'

/** The slice of settings the triggers below care about. */
type SkillSettings = Pick<DesktopSettings, 'installSkills'>

/** App launch: sync whenever the user wants skills kept installed. */
export function shouldSyncOnLaunch(settings: SkillSettings): boolean {
  return settings.installSkills
}

/**
 * A settings update: only the off → on transition. Staying on needs no sync
 * (launch already did one, and every other settings edit would re-trigger it);
 * turning off is not a sync at all — `af skill install` has no uninstall side.
 */
export function shouldSyncOnSettingsChange(prev: SkillSettings, next: SkillSettings): boolean {
  return next.installSkills && !prev.installSkills
}

/**
 * A CLI update: only when the install succeeded and skills are wanted. A newer
 * `af` carries a newer skill catalog, so the skills the coding agents see would
 * otherwise stay a release behind until the next launch.
 */
export function shouldSyncOnCliUpdate(cliUpdateOk: boolean, settings: SkillSettings): boolean {
  return cliUpdateOk && settings.installSkills
}

/** The one command this module runs. Exported so tests assert on it. */
export const SKILL_SYNC_ARGS = ['skill', 'install', '--non-interactive'] as const

/** A sync that hasn't finished by now is hung; kill it and record a failure. */
const SKILL_SYNC_TIMEOUT_MS = 120_000

/** Keep only the tail of a chatty run — the failure is always at the end. */
const MAX_OUTPUT_CHARS = 64 * 1024

/** How much of the output the one-line summary may carry. */
const MAX_SUMMARY_CHARS = 240

/** What one `af skill install` run produced. Never a thrown error. */
export interface SkillSyncOutcome {
  /** Exit code, or null when the process never ran (spawn error / timeout). */
  code: number | null
  /** stdout and stderr interleaved in arrival order. */
  output: string
  /** Spawn-level failure message (missing CLI, timeout), else null. */
  error: string | null
}

/**
 * Last non-empty line of the captured output, collapsed and capped. The CLI's
 * final line is the verdict ("installed 3 skills", "failed for codex: …"), so
 * it makes the most useful one-line summary.
 */
export function lastLine(output: string, max = MAX_SUMMARY_CHARS): string {
  const lines = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '')
  const line = lines.length > 0 ? lines[lines.length - 1] : ''
  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

/**
 * Turn a run into the record the renderer shows. ok is exit code 0 and nothing
 * else: `af skill install` exits non-zero when any skill or target fails, and a
 * spawn error (no usable `af`) is a failed sync, not a missing one.
 */
export function buildSkillSyncRecord(at: string, outcome: SkillSyncOutcome): SkillSyncRecord {
  if (outcome.error !== null) {
    return {
      at,
      ok: false,
      exitCode: outcome.code,
      message: `could not run af skill install: ${outcome.error}`
    }
  }
  const tail = lastLine(outcome.output)
  if (outcome.code === 0) {
    return { at, ok: true, exitCode: 0, message: tail === '' ? 'skills up to date' : tail }
  }
  const suffix = tail === '' ? '' : ` — ${tail}`
  return {
    at,
    ok: false,
    exitCode: outcome.code,
    message: `af skill install exited ${outcome.code ?? 'without a code'}${suffix}`
  }
}

/**
 * One log line per sync: timestamp, exit code, and the run's output flattened
 * onto that line (newlines become " | " so a grep for a failing run returns
 * the whole run).
 */
export function formatSkillSyncLogLine(record: SkillSyncRecord, output: string): string {
  const flattened = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '')
    .join(' | ')
  const exit = record.exitCode === null ? 'none' : String(record.exitCode)
  const body = flattened === '' ? record.message : flattened
  return `${record.at} exit=${exit} ok=${record.ok} ${body}`
}

/** Everything a SkillSync needs from the outside world. */
export interface SkillSyncDeps {
  /** Spawnable af command — read late, so CLI resolution can finish first. */
  command: () => string
  /** Run the sync to completion, capturing output; must never reject. */
  run: (command: string, args: readonly string[]) => Promise<SkillSyncOutcome>
  /** Append one line to the skill-sync log. Best-effort; may reject. */
  appendLog: (line: string) => Promise<void>
  now: () => Date
}

/** Default runner: piped stdio, the app's resolved child env; never rejects. */
function realRun(command: string, args: readonly string[]): Promise<SkillSyncOutcome> {
  return new Promise((resolve) => {
    let output = ''
    let settled = false
    const done = (code: number | null, error: string | null): void => {
      if (settled) return
      settled = true
      resolve({ code, output: output.trim(), error })
    }
    const child = spawn(command, [...args], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: childEnv()
    })
    const timer = setTimeout(() => {
      child.kill()
      done(null, `timed out after ${SKILL_SYNC_TIMEOUT_MS}ms`)
    }, SKILL_SYNC_TIMEOUT_MS)
    const capture = (chunk: Buffer): void => {
      output += chunk.toString('utf8')
      if (output.length > MAX_OUTPUT_CHARS) output = output.slice(-MAX_OUTPUT_CHARS)
    }
    child.stdout?.on('data', capture)
    child.stderr?.on('data', capture)
    child.on('error', (err) => {
      clearTimeout(timer)
      done(null, err.message)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code ?? null, null)
    })
  })
}

/** Append one line to `file`, creating its directory on first write. */
export async function appendLogLine(file: string, line: string): Promise<void> {
  await fs.mkdir(dirname(file), { recursive: true })
  await fs.appendFile(file, `${line}\n`, 'utf8')
}

/**
 * Production deps. `logFile` comes from index.ts (app.getPath('logs')), the
 * only place that may touch electron.
 */
export function defaultSkillSyncDeps(logFile: string): SkillSyncDeps {
  return {
    command: () => getCliCommand(),
    run: realRun,
    appendLog: (line) => appendLogLine(logFile, line),
    now: () => new Date()
  }
}

/**
 * Owns the skill-sync state for the app's lifetime: the last result (served to
 * the renderer inside the snapshot) and the in-flight guard that keeps two
 * triggers from racing on skillkit's state file.
 */
export class SkillSync {
  private readonly deps: SkillSyncDeps
  private lastRecord: SkillSyncRecord | null = null
  private inFlight: Promise<SkillSyncRecord> | null = null

  constructor(deps: SkillSyncDeps) {
    this.deps = deps
  }

  /** The last sync's result, or null when none has finished this session. */
  last(): SkillSyncRecord | null {
    return this.lastRecord
  }

  /** True while a sync is running (a second sync() would join it). */
  isRunning(): boolean {
    return this.inFlight !== null
  }

  /**
   * Run a sync, or join the one already running. Never rejects — every failure
   * mode ends as a recorded, failed SkillSyncRecord.
   */
  sync(): Promise<SkillSyncRecord> {
    if (this.inFlight) return this.inFlight
    const started = this.execute()
    this.inFlight = started
    void started.finally(() => {
      if (this.inFlight === started) this.inFlight = null
    })
    return started
  }

  private async execute(): Promise<SkillSyncRecord> {
    let outcome: SkillSyncOutcome
    try {
      outcome = await this.deps.run(this.deps.command(), SKILL_SYNC_ARGS)
    } catch (err) {
      // A deps.run that rejects is a bug, not a CLI failure — record it anyway
      // rather than letting it escape into an unhandled rejection.
      outcome = { code: null, output: '', error: err instanceof Error ? err.message : String(err) }
    }
    const record = buildSkillSyncRecord(this.deps.now().toISOString(), outcome)
    this.lastRecord = record
    try {
      await this.deps.appendLog(formatSkillSyncLogLine(record, outcome.output))
    } catch {
      // Logging is best-effort: an unwritable log must not fail the sync.
    }
    return record
  }
}
