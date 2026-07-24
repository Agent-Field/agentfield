// Boot orchestration for the "it's just there" story: when the app launches
// (typically hidden, at login), find the control plane — or bring one up if
// nothing is listening — then start the agents the user selected, so
// everything is already answering by the time Claude/Codex/anything queries
// it.
//
// The control plane is not pinned to 8080 anymore. Discovery probes the
// candidate ports (the configured one, or the default plus the port this app
// last used); a recognized AgentField on any of them is adopted as-is. When
// nothing answers and autostart is on, the app starts its own server: on the
// configured port exactly, or — in automatic mode — on the first free port
// from 8080 upward, so a squatted default never blocks the app (or spawns a
// second control plane over an existing one).
//
// Planning is pure (unit-tested); execution shells out via agents.ts.

import type {
  AgentFieldSnapshot,
  ControlPlaneStatus,
  DesktopSettings,
  SnapshotAgent
} from '../shared/types'
import { checkControlPlane, getSnapshot, setActiveControlPlanePort } from './agentfield'
import { runAgentAction, startControlPlane } from './agents'
import { DEFAULT_CONTROL_PLANE_PORT, baseUrlForPort, pickFreePort } from './ports'

export interface AutostartStep {
  name: string
  action: 'start' | 'restart'
}

/**
 * Decide what to do for each selected agent, from the snapshot's badge
 * (registry × control-plane view):
 *  - running  -> nothing to do
 *  - stopped  -> start
 *  - unknown  -> restart: the registry claims running but the control plane
 *    can't see it — typical after a reboot or crash, where the registry entry
 *    is stale (Windows never reconciles it live). Stop-then-run clears it.
 * Names no longer installed are skipped.
 */
export function autostartAgentPlan(
  selected: readonly string[],
  agents: readonly SnapshotAgent[]
): AutostartStep[] {
  const byName = new Map(agents.map((agent) => [agent.name, agent]))
  const steps: AutostartStep[] = []
  for (const name of selected) {
    const agent = byName.get(name)
    if (!agent || agent.badge === 'running') continue
    steps.push({ name, action: agent.badge === 'unknown' ? 'restart' : 'start' })
  }
  return steps
}

/**
 * Ports worth probing for an already-running control plane, in order. A
 * configured port is authoritative — nothing else is probed, so the app never
 * silently adopts a control plane somewhere the user didn't ask for. In
 * automatic mode: the default port, then the port this app last started a
 * control plane on (so a restart finds its own server again instead of
 * spawning a second one).
 */
export function controlPlanePortCandidates(settings: DesktopSettings): number[] {
  if (settings.controlPlanePort !== null) return [settings.controlPlanePort]
  const candidates = [DEFAULT_CONTROL_PLANE_PORT]
  if (
    settings.lastControlPlanePort !== null &&
    settings.lastControlPlanePort !== DEFAULT_CONTROL_PLANE_PORT
  ) {
    candidates.push(settings.lastControlPlanePort)
  }
  return candidates
}

export interface PortProbe {
  port: number
  status: ControlPlaneStatus
}

/** What to do about the control plane, given the candidate-port probes. */
export type ControlPlaneBootPlan =
  | { kind: 'adopt'; port: number }
  /** port null = automatic: pick the first free port from the default up. */
  | { kind: 'start'; port: number | null }
  | { kind: 'skip'; reason: string }

/**
 * Pure boot decision:
 *  - a recognized AgentField (healthy or not) on any candidate -> adopt it;
 *    an unhealthy control plane is already running and not ours to double-start.
 *  - nothing recognized, autostart off -> skip.
 *  - configured port held by a foreign service -> skip with the reason; a
 *    fixed port is used exactly, never silently traded for another one.
 *  - otherwise -> start: on the configured port, or automatically (null).
 *    In automatic mode a foreign service on 8080 is fine — the free-port
 *    scan routes around it.
 */
export function planControlPlaneBoot(
  settings: DesktopSettings,
  probes: readonly PortProbe[]
): ControlPlaneBootPlan {
  const recognized = probes.find((probe) => probe.status.recognized)
  if (recognized) return { kind: 'adopt', port: recognized.port }
  if (!settings.autostartControlPlane) {
    return { kind: 'skip', reason: 'autostart is off and no control plane is running' }
  }
  if (settings.controlPlanePort !== null) {
    const configured = probes.find((probe) => probe.port === settings.controlPlanePort)
    if (configured?.status.reachable) {
      return {
        kind: 'skip',
        reason: `port ${settings.controlPlanePort} is in use by another app`
      }
    }
    return { kind: 'start', port: settings.controlPlanePort }
  }
  return { kind: 'start', port: null }
}

/**
 * Execute the boot sequence. The effective port is recorded via persistPort
 * (whenever it changed) so the next app start probes it again. Agents are
 * started even when the control plane could not be brought up — SDK agents
 * serve standalone and attach when the control plane appears, which still
 * beats staying down.
 */
export async function runAutostart(
  settings: DesktopSettings,
  log: (message: string) => void,
  persistPort: (port: number) => Promise<void> = async () => {}
): Promise<void> {
  const probes: PortProbe[] = []
  for (const port of controlPlanePortCandidates(settings)) {
    probes.push({ port, status: await checkControlPlane(baseUrlForPort(port)) })
  }

  const plan = planControlPlaneBoot(settings, probes)
  if (plan.kind === 'adopt') {
    setActiveControlPlanePort(plan.port)
    log(`autostart: control plane already running at ${baseUrlForPort(plan.port)}`)
    if (plan.port !== settings.lastControlPlanePort) await persistPort(plan.port)
  } else if (plan.kind === 'skip') {
    log(`autostart: control plane not started — ${plan.reason}`)
  } else {
    const port = plan.port ?? (await pickFreePort())
    // Point the app (and the AGENTFIELD_SERVER handed to `af`) at the target
    // before starting, so snapshot polling tracks the boot on the right port.
    setActiveControlPlanePort(port)
    log(`autostart: starting control plane at ${baseUrlForPort(port)}`)
    const result = await startControlPlane(port)
    log(`autostart: control plane — ${result.message}`)
    if (result.ok && port !== settings.lastControlPlanePort) await persistPort(port)
  }

  const snapshot: AgentFieldSnapshot = await getSnapshot()
  for (const step of autostartAgentPlan(settings.autostartAgents, snapshot.registry.agents)) {
    log(`autostart: ${step.action} ${step.name}`)
    const result = await runAgentAction(step.action, step.name)
    log(`autostart: ${step.name} — ${result.ok ? 'up' : result.message}`)
  }
}
