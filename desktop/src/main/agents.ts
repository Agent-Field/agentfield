// Agent lifecycle uses the control-plane HTTP API. The only CLI path retained
// here is the detached `af server` bootstrap.

import { spawn } from 'node:child_process'
import { closeSync, mkdirSync, openSync } from 'node:fs'
import { join } from 'node:path'
import type { AgentActionResult, ControlPlaneStatus } from '../shared/types'
import { checkControlPlane, getAgentFieldHome } from './agentfield'
import { getCliCommand } from './cli'
import { childEnv } from './env'
import { DEFAULT_CONTROL_PLANE_PORT, baseUrlForPort } from './ports'
import { CpApiError, createCpClient, type CpClient } from './cpClient'
import type { RunResult } from './tray-companion'

export type AgentAction = 'start' | 'stop' | 'restart'

const MISSING_CLI_MESSAGE =
  'The AgentField CLI (af) was not found on PATH. Install it first: https://agentfield.ai/docs'

export interface AgentManagementDeps {
  cpClient: CpClient
}

function defaultAgentManagementDeps(): AgentManagementDeps {
  return { cpClient: createCpClient() }
}

function managementError(err: unknown): AgentActionResult {
  if (err instanceof CpApiError) return { ok: false, message: err.message }
  return {
    ok: false,
    message: 'Could not reach the control plane — start the control plane and try again'
  }
}

/**
 * Start / stop / restart an installed agent through the control plane.
 */
export async function runAgentAction(
  action: AgentAction,
  name: string,
  deps: AgentManagementDeps = defaultAgentManagementDeps()
): Promise<AgentActionResult> {
  try {
    switch (action) {
      case 'start': {
        const result = await deps.cpClient.startAgent(name)
        return { ok: true, message: result.message }
      }
      case 'stop': {
        const result = await deps.cpClient.stopAgent(name)
        return { ok: true, message: result.message }
      }
      case 'restart': {
        await deps.cpClient.stopAgent(name)
        const result = await deps.cpClient.startAgent(name)
        return { ok: true, message: result.message }
      }
    }
  } catch (err) {
    return managementError(err)
  }
}

/**
 * Uninstall an installed agent. The server owns stopping a running package.
 */
export async function uninstallAgent(
  name: string,
  deps: AgentManagementDeps = defaultAgentManagementDeps()
): Promise<AgentActionResult> {
  try {
    if (!(await deps.cpClient.hasInstallApi())) {
      return {
        ok: false,
        message: 'Control plane update required — update AgentField CLI'
      }
    }
    const result = await deps.cpClient.uninstallPackage(name)
    return { ok: true, message: result.status }
  } catch (err) {
    if (err instanceof CpApiError && err.status === 404) {
      return { ok: false, message: 'Control plane update required — update AgentField CLI' }
    }
    return managementError(err)
  }
}

/** Pause or resume package maintenance while preserving the established
 * old-control-plane message for servers without the additive endpoint. */
export async function setAgentPackageAutoUpdate(
  name: string,
  enabled: boolean,
  deps: AgentManagementDeps = defaultAgentManagementDeps()
): Promise<AgentActionResult> {
  try {
    await deps.cpClient.setPackageAutoUpdate(name, enabled)
    return {
      ok: true,
      message: enabled ? 'Automatic updates resumed.' : 'Automatic updates paused.'
    }
  } catch (err) {
    if (err instanceof CpApiError && err.status === 404) {
      return { ok: false, message: 'Control plane update required — update AgentField CLI' }
    }
    return managementError(err)
  }
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
 *
 * launchd only ever serves the default port (that is what af-tray's plist
 * starts), so a non-default target port always direct-spawns — kickstarting
 * would bring a server up on 8080 while the app waits on the chosen port.
 */
export function planControlPlaneLaunch(
  platform: NodeJS.Platform,
  serverAgentLoaded: boolean,
  port: number = DEFAULT_CONTROL_PLANE_PORT
): ControlPlaneLaunch {
  return platform === 'darwin' && serverAgentLoaded && port === DEFAULT_CONTROL_PLANE_PORT
    ? 'launchd'
    : 'spawn'
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
   * Direct detached spawn of `af server` on the given port (net-new /
   * fallback path). The returned promise resolves ONLY on a spawn-time error
   * (missing CLI, etc.); otherwise it stays pending while the server boots —
   * matching the readiness race the wait loop expects.
   */
  spawnServer: (port: number) => Promise<AgentActionResult>
  /** One GET {baseUrl}/health probe against the target control plane. */
  checkHealth: (baseUrl: string) => Promise<ControlPlaneStatus>
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
/**
 * Environment pinned onto a spawned `af server`.
 *
 * AGENTFIELD_PORT pins the server to the port this app will poll. Without
 * it, an agentfield.yaml that sets its own port makes `af server` bind there
 * while the app waits on the chosen port forever — a healthy server and a
 * spinner that never resolves.
 *
 * AGENTFIELD_SERVER is the URL the server hands to every agent it starts
 * (control-plane resolveServerURL reads its own environment and otherwise
 * falls back to http://localhost:8080 regardless of the port it listens on).
 * Without it, a control plane on any other port — the auto-picked one when
 * 8080 is busy, or a configured one — tells swe-planner, pr-af and every
 * other node to register with localhost:8080, i.e. with whatever else is
 * there, or nothing.
 */
export function serverSpawnEnv(port: number): NodeJS.ProcessEnv {
  return { AGENTFIELD_PORT: String(port), AGENTFIELD_SERVER: `http://localhost:${port}` }
}

function defaultSpawnServer(port: number): Promise<AgentActionResult> {
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
    const child = spawn(getCliCommand(), ['server'], {
      windowsHide: true,
      detached: true,
      stdio: ['ignore', log, log],
      env: childEnv(serverSpawnEnv(port))
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
    checkHealth: (baseUrl) => checkControlPlane(baseUrl),
    now: () => Date.now(),
    delay: (ms) => new Promise((resolve) => setTimeout(resolve, ms))
  }
}

/**
 * Bring the control plane up on the given port and wait until /health there
 * reports a healthy AgentField. Prefers the tray's launchd server agent on
 * macOS for the default port (single owner — see planControlPlaneLaunch);
 * otherwise, or if kickstart fails, direct-spawns `af server` pinned to the
 * port. Either way it then polls /health until healthy, a foreign service is
 * detected on the port, or the deadline passes.
 */
export async function startControlPlane(
  port: number = DEFAULT_CONTROL_PLANE_PORT,
  waitMs = 30_000,
  deps: ControlPlaneStartDeps = defaultControlPlaneStartDeps()
): Promise<AgentActionResult> {
  // Only consult launchctl on darwin; elsewhere the launchd path never applies.
  const serverAgentLoaded = deps.platform === 'darwin' ? await deps.serverAgentLoaded() : false
  const launch = planControlPlaneLaunch(deps.platform, serverAgentLoaded, port)

  // spawnError is watched during the readiness race; it stays null on the
  // launchd path (there is no spawn to fail), and is set when we fall back.
  let spawnError: Promise<AgentActionResult> | null = null
  if (launch === 'launchd') {
    const res = await deps.run('launchctl', ['kickstart', `gui/${deps.uid()}/${SERVER_LABEL}`])
    if (res.code !== 0) {
      // The agent is loaded but kickstart failed (unusual). Fall back to a
      // direct spawn so the app still comes up rather than hanging on a server
      // that never boots.
      spawnError = deps.spawnServer(port)
    }
  } else {
    spawnError = deps.spawnServer(port)
  }

  const baseUrl = baseUrlForPort(port)
  const deadline = deps.now() + waitMs
  while (deps.now() < deadline) {
    if (spawnError) {
      const raced = await Promise.race([spawnError, deps.delay(1_000).then(() => null)])
      if (raced) return raced
    } else {
      await deps.delay(1_000)
    }
    const status = await deps.checkHealth(baseUrl)
    if (status.healthy) return { ok: true, message: 'control plane running' }
    // A foreign service answering the port will never become healthy.
    if (status.reachable && !status.recognized) {
      return { ok: false, message: status.error ?? 'port in use by another app' }
    }
  }
  return { ok: false, message: 'control plane did not become healthy in time' }
}
