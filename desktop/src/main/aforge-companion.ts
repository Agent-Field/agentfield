// Provision the pinned aforge coding-harness binary on app launch, so a
// desktop-only install can run harness-backed agents without anyone ever
// running the curl installer. Unlike tray-companion.ts this ships no bundled
// payload: it shells out to `af aforge ensure`, which owns the download,
// checksum verification and upgrade rules (control-plane/internal/aforge).
// One code path, and a version bump in af upgrades every surface at once.
//
// Two moving parts, kept apart so the decision is unit-testable:
//   1. planAforge() — pure: given the resolved af command, the skip env var and
//      whether we already ran, decide whether to shell out at all.
//   2. ensureAforgeCompanion() — the effect, driven by injected deps so tests
//      never spawn anything.
//
// Best-effort by construction: every failure resolves to { ok: false }, never
// throws, so a dead network can't delay or break startup.

import { spawn } from 'node:child_process'
import { getCliCommand } from './cli'
import { childEnv } from './env'

const ENSURE_TIMEOUT_MS = 5 * 60 * 1_000

/** What ensureAforge should do, decided purely from the observed state. */
export interface AforgeState {
  /** The `af` command the app resolved at startup, or null when none is usable. */
  cliCommand: string | null
  /** Value of AGENTFIELD_SKIP_AFORGE in the app's environment. */
  skipEnv: string | undefined
  /** ensureAforge already ran in this process. */
  alreadyRan: boolean
}

export interface AforgePlan {
  run: boolean
  reason: string
}

export function planAforge(s: AforgeState): AforgePlan {
  if (s.skipEnv === '1') {
    return { run: false, reason: 'AGENTFIELD_SKIP_AFORGE=1 — skipping aforge provisioning' }
  }
  if (s.alreadyRan) {
    return { run: false, reason: 'aforge already provisioned this launch' }
  }
  if (s.cliCommand === null || s.cliCommand.trim() === '') {
    return { run: false, reason: 'no usable af CLI — skipping aforge provisioning' }
  }
  return { run: true, reason: 'provisioning aforge via af aforge ensure' }
}

export interface AforgeDeps {
  /** Run a command to completion; must never reject (resolve code=-1 on spawn error). */
  run: (
    command: string,
    args: string[]
  ) => Promise<{ code: number; stdout: string; stderr: string }>
  /** The af command to drive — `getCliCommand()` from './cli' in production. */
  cliCommand: () => string | null
  /** Environment lookup, injected so tests don't mutate process.env. */
  env: (name: string) => string | undefined
}

function realRun(
  command: string,
  args: string[]
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    let stdout = ''
    let stderr = ''
    let settled = false
    const done = (code: number) => {
      if (settled) return
      settled = true
      resolve({ code, stdout, stderr })
    }

    try {
      const child = spawn(command, args, { windowsHide: true, env: childEnv() })
      // aforge is roughly a 35 MB download, so allow slow connections five minutes.
      const timer = setTimeout(() => {
        child.kill()
        done(-1)
      }, ENSURE_TIMEOUT_MS)
      child.stdout?.on('data', (chunk: Buffer) => {
        stdout += chunk.toString('utf8')
      })
      child.stderr?.on('data', (chunk: Buffer) => {
        stderr += chunk.toString('utf8')
      })
      child.on('error', () => {
        clearTimeout(timer)
        done(-1)
      })
      child.on('close', (code) => {
        clearTimeout(timer)
        done(code ?? -1)
      })
    } catch {
      done(-1)
    }
  })
}

export function defaultAforgeDeps(): AforgeDeps {
  return {
    run: realRun,
    cliCommand: getCliCommand,
    env: (name) => process.env[name]
  }
}

export interface AforgeResult {
  ok: boolean
  message: string
}

let alreadyRan = false

export async function ensureAforgeCompanion(
  deps: AforgeDeps = defaultAforgeDeps()
): Promise<AforgeResult> {
  try {
    const cliCommand = deps.cliCommand()
    const plan = planAforge({
      cliCommand,
      skipEnv: deps.env('AGENTFIELD_SKIP_AFORGE'),
      alreadyRan
    })
    if (!plan.run) return { ok: true, message: plan.reason }

    alreadyRan = true
    const result = await deps.run(cliCommand as string, ['aforge', 'ensure'])
    if (result.code === 0) return { ok: true, message: 'aforge is provisioned' }

    const detail = (result.stderr || result.stdout).trim()
    return {
      ok: false,
      message: `af aforge ensure failed (exit ${result.code}): ${detail}`
    }
  } catch (err) {
    return { ok: false, message: `af aforge ensure failed: ${String(err)}` }
  }
}

/** Reset the once-per-launch latch (tests only). */
export function resetAforgeCompanion(): void {
  alreadyRan = false
}
