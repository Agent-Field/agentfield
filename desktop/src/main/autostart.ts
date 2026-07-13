// Boot orchestration for the "it's just there" story: when the app launches
// (typically hidden, at login), bring the control plane up if nothing is
// listening, then start the agents the user selected — so everything is
// already answering by the time Claude/Codex/anything queries it.
//
// Planning is pure (unit-tested); execution shells out via agents.ts.

import type { AgentFieldSnapshot, DesktopSettings, SnapshotAgent } from '../shared/types'
import { getSnapshot } from './agentfield'
import { runAgentAction, startControlPlane } from './agents'

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
 * Start the control plane only when nothing answers at all: an unhealthy but
 * recognized control plane is already running (not ours to double-start), and
 * a foreign service owns the port — starting would just fail to bind.
 */
export function shouldStartControlPlane(
  settings: DesktopSettings,
  cp: AgentFieldSnapshot['controlPlane']
): boolean {
  return settings.autostartControlPlane && !cp.reachable
}

/**
 * Execute the boot sequence. Agents are started even when the control plane
 * could not be brought up — SDK agents serve standalone and attach when the
 * control plane appears, which still beats staying down.
 */
export async function runAutostart(
  settings: DesktopSettings,
  log: (message: string) => void
): Promise<void> {
  let snapshot = await getSnapshot()

  if (shouldStartControlPlane(settings, snapshot.controlPlane)) {
    log('autostart: starting control plane')
    const result = await startControlPlane()
    log(`autostart: control plane — ${result.message}`)
    if (result.ok) snapshot = await getSnapshot()
  }

  for (const step of autostartAgentPlan(settings.autostartAgents, snapshot.registry.agents)) {
    log(`autostart: ${step.action} ${step.name}`)
    const result = await runAgentAction(step.action, step.name)
    log(`autostart: ${step.name} — ${result.ok ? 'up' : result.message}`)
  }
}
