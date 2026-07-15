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
import type { AgentActionResult, ControlPlaneStatus } from '../shared/types'
import { checkControlPlane, getAgentFieldHome, readInstalledAgents } from './agentfield'
import { getCliCommand } from './cli'
import { childEnv } from './env'
import { sanitizeInstallOutput } from './installer'
import type { RunResult } from './tray-companion'

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

    const child = spawn(getCliCommand(), args, { windowsHide: true, env: childEnv() })
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

/** launchd label af-tray registers for the control-plane server agent (see af-tray shared.go). */
export const SERVER_LABEL = 'ai.agentfield.server'

/** How long a launchctl invocation may take before we give up on it. */
const LAUNCHCTL_TIMEOUT_MS = 5_000

/** Which mechanism startControlPlane should use to bring the control plane up. */
export type ControlPlaneLaunch = 'launchd' | 'spawn'

/**
 * Single-owner preference: on macOS, once the tray's launchd server agent is
 * loaded, launchd is the one true owner of the control plane. Kickstart it
 * rather than direct-spawning a second `af server`, which would race launchd's
 * KeepAlive-supervised process for port 8080, lose the bind, and (with
 * KeepAlive={SuccessfulExit:false}) trigger a relaunch loop. Everywhere else —
 * Windows/Linux, or a net-new macOS machine before `af-tray install` has loaded
 * the agent — direct-spawn, which is still the only way to get a server up.
 */
export function planControlPlaneLaunch(
  platform: NodeJS.Platform,
  serverAgentLoaded: boolean
): ControlPlaneLaunch {
  return platform === 'darwin' && serverAgentLoaded ? 'launchd' : 'spawn'
}

/** Everything startControlPlane needs from the outside world (DI so tests never
 *  touch launchctl, spawn a server, or wait in real time). */
export interface ControlPlaneStartDeps {
  platform: NodeJS.Platform
  /** launchd gui domain uid — process.getuid() in production. */
  uid: () => number
  /** True when the tray's launchd server agent is loaded (kickstartable). */
  serverAgentLoaded: () => Promise<boolean>
  /** Run a command to completion (launchctl); never rejects. */
  run: (command: string, args: string[]) => Promise<RunResult>
  /**
   * Direct detached spawn of `af server` (net-new / fallback path). The
   * returned promise resolves ONLY on a spawn-time error (missing CLI, etc.);
   * otherwise it stays pending while the server boots — matching the readiness
   * race the wait loop expects.
   */
  spawnServer: () => Promise<AgentActionResult>
  /** One GET /health probe against the app's control plane. */
  checkHealth: () => Promise<ControlPlaneStatus>
  now: () => number
  /** Resolve after ms (injected so tests advance without real waiting). */
  delay: (ms: number) => Promise<void>
}

/** Run a command to completion, capturing exit code + stdout; never rejects
 *  (resolves code=-1 on spawn error/timeout). Mirrors tray-companion's runner. */
function realRunCommand(command: string, args: string[]): Promise<RunResult> {
  return new Promise((resolve) => {
    let stdout = ''
    let settled = false
    const done = (code: number) => {
      if (settled) return
      settled = true
      resolve({ code, stdout })
    }
    const child = spawn(command, args, { windowsHide: true, env: childEnv() })
    const timer = setTimeout(() => {
      child.kill()
      done(-1)
    }, LAUNCHCTL_TIMEOUT_MS)
    child.stdout?.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8')
    })
    child.on('error', () => {
      clearTimeout(timer)
      done(-1)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code ?? -1)
    })
  })
}

/**
 * Direct detached spawn of `af server` — it outlives the app, matching the
 * "agents on autopilot" model. Output goes to ~/.agentfield/logs/control-plane.log
 * (the same file the macOS launchd agent uses). The returned promise resolves
 * only if the spawn itself errors; otherwise it stays pending.
 */
function defaultSpawnServer(): Promise<AgentActionResult> {
  return new Promise((resolve) => {
    let log: number
    try {
      const logsDir = join(getAgentFieldHome(), 'logs')
      mkdirSync(logsDir, { recursive: true })
      log = openSync(join(logsDir, 'control-plane.log'), 'a')
    } catch (err) {
      resolve({ ok: false, message: `could not open control-plane log: ${String(err)}` })
      return
    }
    // Pin the spawned server to the port this app polls. Without it, an
    // agentfield.yaml that sets a custom port makes `af server` bind there
    // while the app waits on 8080 forever — a healthy server and a spinner
    // that never resolves. Custom-port setups (their own server, agents
    // pointed elsewhere) remain unsupported by the app for now; this only
    // makes the app's own spawn deterministic.
    const child = spawn(getCliCommand(), ['server'], {
      windowsHide: true,
      detached: true,
      stdio: ['ignore', log, log],
      env: childEnv({ AGENTFIELD_PORT: '8080' })
    })
    child.on('error', (err: NodeJS.ErrnoException) => {
      resolve({
        ok: false,
        message: err.code === 'ENOENT' ? MISSING_CLI_MESSAGE : String(err.message)
      })
    })
    child.unref()
    // The detached child dup'd the log fd at spawn; the parent's copy is done.
    closeSync(log)
  })
}

/** Production deps: real launchctl runner, detached spawn, and live /health. */
export function defaultControlPlaneStartDeps(): ControlPlaneStartDeps {
  const uid = () => (typeof process.getuid === 'function' ? process.getuid() : 0)
  return {
    platform: process.platform,
    uid,
    serverAgentLoaded: async () =>
      (await realRunCommand('launchctl', ['print', `gui/${uid()}/${SERVER_LABEL}`])).code === 0,
    run: realRunCommand,
    spawnServer: defaultSpawnServer,
    checkHealth: () => checkControlPlane(),
    now: () => Date.now(),
    delay: (ms) => new Promise((resolve) => setTimeout(resolve, ms))
  }
}

/**
 * Bring the control plane up and wait until /health reports a healthy
 * AgentField. Prefers the tray's launchd server agent on macOS (single owner —
 * see planControlPlaneLaunch); otherwise, or if kickstart fails, direct-spawns
 * `af server`. Either way it then polls /health until healthy, a foreign
 * service is detected on the port, or the deadline passes.
 */
export async function startControlPlane(
  waitMs = 30_000,
  deps: ControlPlaneStartDeps = defaultControlPlaneStartDeps()
): Promise<AgentActionResult> {
  // Only consult launchctl on darwin; elsewhere the launchd path never applies.
  const serverAgentLoaded = deps.platform === 'darwin' ? await deps.serverAgentLoaded() : false
  const launch = planControlPlaneLaunch(deps.platform, serverAgentLoaded)

  // spawnError is watched during the readiness race; it stays null on the
  // launchd path (there is no spawn to fail), and is set when we fall back.
  let spawnError: Promise<AgentActionResult> | null = null
  if (launch === 'launchd') {
    const res = await deps.run('launchctl', ['kickstart', `gui/${deps.uid()}/${SERVER_LABEL}`])
    if (res.code !== 0) {
      // The agent is loaded but kickstart failed (unusual). Fall back to a
      // direct spawn so the app still comes up rather than hanging on a server
      // that never boots.
      spawnError = deps.spawnServer()
    }
  } else {
    spawnError = deps.spawnServer()
  }

  const deadline = deps.now() + waitMs
  while (deps.now() < deadline) {
    if (spawnError) {
      const raced = await Promise.race([spawnError, deps.delay(1_000).then(() => null)])
      if (raced) return raced
    } else {
      await deps.delay(1_000)
    }
    const status = await deps.checkHealth()
    if (status.healthy) return { ok: true, message: 'control plane running' }
    // A foreign service answering the port will never become healthy.
    if (status.reachable && !status.recognized) {
      return { ok: false, message: status.error ?? 'port in use by another app' }
    }
  }
  return { ok: false, message: 'control plane did not become healthy in time' }
}
