// Agent + control-plane lifecycle seam: shells out to the af CLI (the single
// contract — the app never reimplements start/stop). No electron imports so
// the module stays unit-testable.
//
// CLI semantics this leans on (control-plane/internal/cli):
//   - `af run <name>` spawns the agent detached, waits for its local /health,
//     and exits — the agent survives the CLI (and this app) exiting.
//   - `af stop <name>` shuts down gracefully (HTTP /shutdown, then signal,
//     then force) and flips the registry entry to stopped.
//   - there is no `af restart` — restart here is stop-then-run.
//   - `af server` always blocks, so the control plane is spawned detached
//     with its output appended to ~/.agentfield/logs/control-plane.log.

import { spawn } from 'node:child_process'
import { closeSync, mkdirSync, openSync } from 'node:fs'
import { join } from 'node:path'
import type { AgentActionResult } from '../shared/types'
import { checkControlPlane, getAgentFieldHome, readInstalledAgents } from './agentfield'
import { getCliCommand } from './cli'
import { sanitizeInstallOutput } from './installer'

export type AgentAction = 'start' | 'stop' | 'restart'

/** How long one `af run`/`af stop` may take (run waits ≤30s for readiness). */
const CLI_TIMEOUT_MS = 90_000

const MISSING_CLI_MESSAGE =
  'The AgentField CLI (af) was not found on PATH. Install it first: https://agentfield.ai/docs'

/** Run one af verb to completion, capturing the last meaningful output line. */
function runCli(args: string[], timeoutMs = CLI_TIMEOUT_MS): Promise<AgentActionResult> {
  return new Promise((resolve) => {
    let lastLine = ''
    let settled = false
    const done = (result: AgentActionResult) => {
      if (!settled) {
        settled = true
        resolve(result)
      }
    }

    const child = spawn(getCliCommand(), args, { windowsHide: true })
    const timer = setTimeout(() => {
      child.kill()
      done({ ok: false, message: `af ${args.join(' ')} timed out` })
    }, timeoutMs)

    const collect = (chunk: Buffer) => {
      const lines = sanitizeInstallOutput(chunk.toString('utf8'))
      if (lines.length > 0) lastLine = lines[lines.length - 1]
    }
    child.stdout.on('data', collect)
    child.stderr.on('data', collect)
    child.on('error', (err: NodeJS.ErrnoException) => {
      clearTimeout(timer)
      done({
        ok: false,
        message: err.code === 'ENOENT' ? MISSING_CLI_MESSAGE : `Failed to run af: ${err.message}`
      })
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(
        code === 0
          ? { ok: true, message: lastLine }
          : { ok: false, message: lastLine || `af ${args.join(' ')} exited with code ${code}` }
      )
    })
  })
}

/**
 * Start / stop / restart an installed agent by registry name. The name is
 * validated against ~/.agentfield/installed.yaml — the renderer only ever
 * supplies names, and unknown ones are refused rather than handed to a shell.
 */
export async function runAgentAction(
  action: AgentAction,
  name: string
): Promise<AgentActionResult> {
  const registry = await readInstalledAgents()
  if (!registry.agents.some((agent) => agent.name === name)) {
    return { ok: false, message: `"${name}" is not an installed agent` }
  }

  switch (action) {
    case 'start':
      return runCli(['run', name])
    case 'stop':
      return runCli(['stop', name])
    case 'restart': {
      // `af stop` exits cleanly when the agent is already stopped, so a
      // restart of a wedged ("unknown") agent degrades to a plain start.
      const stopped = await runCli(['stop', name])
      if (!stopped.ok) return stopped
      return runCli(['run', name])
    }
  }
}

/**
 * Uninstall an installed agent: graceful stop first (a stopped agent's stop
 * is a no-op), then `af uninstall --force`, which removes the package dir,
 * the registry entry, and the node-scoped secrets. Names are validated
 * against the registry like every other verb.
 */
export async function uninstallAgent(name: string): Promise<AgentActionResult> {
  const registry = await readInstalledAgents()
  if (!registry.agents.some((agent) => agent.name === name)) {
    return { ok: false, message: `"${name}" is not an installed agent` }
  }
  await runCli(['stop', name])
  return runCli(['uninstall', name, '--force'])
}

/**
 * Spawn `af server` detached — it outlives the app, matching the "agents on
 * autopilot" model — and wait until /health reports an AgentField control
 * plane. Output goes to ~/.agentfield/logs/control-plane.log (same file the
 * macOS launchd agent uses).
 */
export async function startControlPlane(
  waitMs = 30_000
): Promise<AgentActionResult> {
  let log: number
  try {
    const logsDir = join(getAgentFieldHome(), 'logs')
    mkdirSync(logsDir, { recursive: true })
    log = openSync(join(logsDir, 'control-plane.log'), 'a')
  } catch (err) {
    return { ok: false, message: `could not open control-plane log: ${String(err)}` }
  }

  try {
    const child = spawn(getCliCommand(), ['server'], {
      windowsHide: true,
      detached: true,
      stdio: ['ignore', log, log]
    })
    const spawnError = new Promise<AgentActionResult>((resolve) => {
      child.on('error', (err: NodeJS.ErrnoException) => {
        resolve({
          ok: false,
          message: err.code === 'ENOENT' ? MISSING_CLI_MESSAGE : String(err.message)
        })
      })
    })
    child.unref()

    const deadline = Date.now() + waitMs
    while (Date.now() < deadline) {
      const raced = await Promise.race([
        spawnError,
        new Promise<null>((resolve) => setTimeout(() => resolve(null), 1_000))
      ])
      if (raced) return raced
      const status = await checkControlPlane()
      if (status.healthy) return { ok: true, message: 'control plane running' }
      // A foreign service answering the port will never become healthy.
      if (status.reachable && !status.recognized) {
        return { ok: false, message: status.error ?? 'port in use by another app' }
      }
    }
    return { ok: false, message: 'control plane did not become healthy in time' }
  } finally {
    closeSync(log)
  }
}
