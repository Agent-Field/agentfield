// TODO(af-cli): this module currently reads ~/.agentfield/installed.yaml directly;
// a sibling branch is adding `af list -o json` — swap readInstalledAgents() to shell
// out to that once it lands, so the CLI stays the single source of truth for
// registry parsing.
//
// This is THE single data-access module for AgentField Desktop. Everything that
// touches the AgentField installation (~/.agentfield) or the control plane HTTP
// API lives here and nowhere else. It deliberately does NOT import from
// 'electron' so it stays unit-testable under plain vitest.

import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import yaml from 'js-yaml'
import type {
  AgentBadge,
  AgentFieldSnapshot,
  ControlPlaneStatus,
  DashboardMetrics,
  ExecutionsResult,
  ExecutionSummary,
  InstalledAgent,
  RegistryResult
} from '../shared/types'

import { DEFAULT_CONTROL_PLANE_PORT, baseUrlForPort } from './ports'

export const DEFAULT_BASE_URL = baseUrlForPort(DEFAULT_CONTROL_PLANE_PORT)

// ---- Active control-plane URL --------------------------------------------
// The one base URL the whole app is pointed at. It starts at the default and
// moves when autostart adopts a running control plane on another port or
// starts one there (see autostart.ts). Every consumer — snapshot polling,
// the tray, open-web-ui, AGENTFIELD_SERVER for spawned `af` — reads it via
// getBaseUrl() so nothing in the app hard-codes 8080.

let activeBaseUrl = DEFAULT_BASE_URL

export function getBaseUrl(): string {
  return activeBaseUrl
}

export function setActiveControlPlanePort(port: number): void {
  activeBaseUrl = baseUrlForPort(port)
}

const HTTP_TIMEOUT_MS = 3000

/** Injectable fetch so tests never hit the network. */
export type FetchLike = typeof fetch

/** Root of the local AgentField installation. os.homedir() is platform-aware
 *  (resolves %USERPROFILE% on Windows, $HOME elsewhere). */
export function getAgentFieldHome(): string {
  return path.join(os.homedir(), '.agentfield')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/**
 * Probe GET {baseUrl}/health.
 *  - 200 {"status":"healthy",...}   -> { reachable: true, recognized: true,  healthy: true }
 *  - 503 {"status":"unhealthy",...} -> { reachable: true, recognized: true,  healthy: false }
 *    (an HTTP response — even 503 — still means the control plane is reachable)
 *  - any response whose body is not an AgentField health payload
 *                                   -> { reachable: true, recognized: false, healthy: false, error }
 *    (default port 8080 is popular — an unrelated dev server answering 200 on
 *    /health must not light up the dashboard as a running control plane)
 *  - network error / timeout (3s)   -> { reachable: false, recognized: false, healthy: false, error }
 */
export async function checkControlPlane(
  baseUrl: string = getBaseUrl(),
  fetchImpl: FetchLike = fetch
): Promise<ControlPlaneStatus> {
  try {
    const res = await fetchImpl(`${baseUrl}/health`, {
      signal: AbortSignal.timeout(HTTP_TIMEOUT_MS)
    })
    let raw: unknown
    try {
      raw = await res.json()
    } catch {
      raw = undefined
    }
    const status = isRecord(raw) && typeof raw.status === 'string' ? raw.status : undefined
    const recognized = status === 'healthy' || status === 'unhealthy'
    if (!recognized) {
      return {
        reachable: true,
        recognized: false,
        healthy: false,
        raw,
        error: `A service answered ${baseUrl}/health but does not look like an AgentField control plane — another app may be using the port.`
      }
    }
    return { reachable: true, recognized: true, healthy: res.ok && status === 'healthy', raw }
  } catch (err) {
    return { reachable: false, recognized: false, healthy: false, error: errorMessage(err) }
  }
}

function toInstalledAgent(key: string, entry: unknown): InstalledAgent {
  const record = isRecord(entry) ? entry : {}
  const runtime = isRecord(record.runtime) ? record.runtime : {}
  return {
    name: typeof record.name === 'string' && record.name !== '' ? record.name : key,
    version: typeof record.version === 'string' ? record.version : '',
    description: typeof record.description === 'string' ? record.description : '',
    language: typeof record.language === 'string' ? record.language : undefined,
    status: typeof record.status === 'string' ? record.status : 'unknown',
    path: typeof record.path === 'string' && record.path !== '' ? record.path : null,
    port: typeof runtime.port === 'number' ? runtime.port : null,
    pid: typeof runtime.pid === 'number' ? runtime.pid : null
  }
}

/**
 * Read <homeDir>/installed.yaml (the local agent-node registry).
 *  - Missing file or missing ~/.agentfield dir -> { exists: false, agents: [] }
 *    (graceful empty state, NOT an error).
 *  - Malformed YAML -> error surfaced as a string in the result; never throws,
 *    so nothing blows up across the IPC boundary.
 */
export async function readInstalledAgents(
  homeDir: string = getAgentFieldHome()
): Promise<RegistryResult> {
  const registryPath = path.join(homeDir, 'installed.yaml')
  let text: string
  try {
    text = await fs.readFile(registryPath, 'utf8')
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      return { exists: false, agents: [] }
    }
    return { exists: false, agents: [], error: errorMessage(err) }
  }

  let doc: unknown
  try {
    doc = yaml.load(text)
  } catch (err) {
    return {
      exists: true,
      agents: [],
      error: `Failed to parse ${registryPath}: ${errorMessage(err)}`
    }
  }

  const installed = isRecord(doc) && isRecord(doc.installed) ? doc.installed : {}
  const agents = Object.entries(installed).map(([key, entry]) => toInstalledAgent(key, entry))
  return { exists: true, agents }
}

/**
 * GET {baseUrl}/api/v1/nodes?show_all=true -> {"nodes":[{"id":...,"health_status":...},...]}
 * show_all matters: the endpoint's default filter returns health=active nodes
 * only, and keying "seen on the control plane" off that made the badge flicker
 * running→unknown whenever a node's health dipped for one poll (busy node,
 * post-restart unknown, late lease renewal). Registration presence is stable;
 * health is not — so both are returned and the badge weighs them separately.
 * Returns a map of node id -> health_status, or null on any failure — callers
 * treat null as "control plane view unavailable" and fall back to registry
 * status alone.
 */
export async function fetchControlPlaneNodes(
  baseUrl: string = getBaseUrl(),
  fetchImpl: FetchLike = fetch
): Promise<Map<string, string> | null> {
  try {
    const res = await fetchImpl(`${baseUrl}/api/v1/nodes?show_all=true`, {
      signal: AbortSignal.timeout(HTTP_TIMEOUT_MS)
    })
    if (!res.ok) return null
    const body: unknown = await res.json()
    if (!isRecord(body) || !Array.isArray(body.nodes)) return null
    const health = new Map<string, string>()
    for (const node of body.nodes) {
      if (!isRecord(node) || typeof node.id !== 'string' || node.id === '') continue
      health.set(node.id, typeof node.health_status === 'string' ? node.health_status : 'unknown')
    }
    return health
  } catch {
    return null
  }
}

/**
 * Pure badge derivation. `controlPlaneReachable` here means "we have a usable
 * control-plane node view" (health reachable AND the nodes list fetched).
 * `nodeHealth` is the node's health_status on the control plane, or null when
 * it is not registered there at all.
 *
 * CP view unavailable — trust the registry:
 *   'running' -> 'running' | 'stopped' -> 'stopped' | other/absent -> 'unknown'
 * CP view available — cross-check. Registration presence (not health) proves
 * a running registry entry is live, so transient health dips cannot flicker
 * the badge; health only matters for stopped entries, where an ACTIVE node
 * contradicts the registry:
 *   registry running + registered (any health) -> 'running'
 *   registry running + not registered          -> 'unknown'  (stale registry)
 *   registry stopped + health active           -> 'unknown'  (conflict)
 *   registry stopped + otherwise               -> 'stopped'  (stopped nodes stay
 *                                                  registered as inactive/unknown)
 *   other/absent registry status               -> 'unknown'
 */
export function deriveAgentBadge(
  registryStatus: string | undefined,
  controlPlaneReachable: boolean,
  nodeHealth: string | null
): AgentBadge {
  if (!controlPlaneReachable) {
    if (registryStatus === 'running') return 'running'
    if (registryStatus === 'stopped') return 'stopped'
    return 'unknown'
  }
  if (registryStatus === 'running') {
    return nodeHealth !== null ? 'running' : 'unknown'
  }
  if (registryStatus === 'stopped') {
    return nodeHealth === 'active' ? 'unknown' : 'stopped'
  }
  return 'unknown'
}

const RECENT_EXECUTIONS_LIMIT = 5

function toExecutionSummary(row: Record<string, unknown>): ExecutionSummary | null {
  const runId = typeof row.run_id === 'string' ? row.run_id : ''
  if (!runId) return null
  return {
    runId,
    status: typeof row.status === 'string' ? row.status : 'unknown',
    displayName:
      typeof row.display_name === 'string' && row.display_name !== ''
        ? row.display_name
        : runId,
    agentId: typeof row.agent_id === 'string' ? row.agent_id : '',
    startedAt: typeof row.started_at === 'string' ? row.started_at : '',
    durationMs: typeof row.duration_ms === 'number' ? row.duration_ms : null,
    terminal: row.terminal === true,
    errorMessage:
      typeof row.root_error_message === 'string' && row.root_error_message !== ''
        ? row.root_error_message
        : null
  }
}

/**
 * GET {baseUrl}/api/ui/v2/workflow-runs (newest first) and split the rows into
 * in-flight runs and a short tail of finished ones. Returns null on any
 * failure — callers render "activity unavailable", never an error page.
 */
export async function fetchExecutions(
  baseUrl: string = getBaseUrl(),
  fetchImpl: FetchLike = fetch
): Promise<ExecutionsResult | null> {
  try {
    const res = await fetchImpl(
      `${baseUrl}/api/ui/v2/workflow-runs?page=1&page_size=25&sort_by=updated_at&sort_order=desc`,
      { signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) }
    )
    if (!res.ok) return null
    const body: unknown = await res.json()
    if (!isRecord(body) || !Array.isArray(body.runs)) return null
    const summaries = body.runs
      .filter(isRecord)
      .map(toExecutionSummary)
      .filter((s): s is ExecutionSummary => s !== null)
    return {
      running: summaries.filter((s) => !s.terminal),
      recent: summaries.filter((s) => s.terminal).slice(0, RECENT_EXECUTIONS_LIMIT)
    }
  } catch {
    return null
  }
}

/**
 * GET {baseUrl}/api/ui/v1/dashboard/summary -> headline dashboard numbers.
 * Returns null on any failure; the dashboard renders placeholders instead.
 */
export async function fetchDashboardMetrics(
  baseUrl: string = getBaseUrl(),
  fetchImpl: FetchLike = fetch
): Promise<DashboardMetrics | null> {
  try {
    const res = await fetchImpl(`${baseUrl}/api/ui/v1/dashboard/summary`, {
      signal: AbortSignal.timeout(HTTP_TIMEOUT_MS)
    })
    if (!res.ok) return null
    const body: unknown = await res.json()
    if (!isRecord(body)) return null
    const agents = isRecord(body.agents) ? body.agents : {}
    const executions = isRecord(body.executions) ? body.executions : {}
    return {
      agentsRunning: typeof agents.running === 'number' ? agents.running : 0,
      agentsTotal: typeof agents.total === 'number' ? agents.total : 0,
      executionsToday: typeof executions.today === 'number' ? executions.today : 0,
      executionsYesterday:
        typeof executions.yesterday === 'number' ? executions.yesterday : 0,
      successRate: typeof body.success_rate === 'number' ? body.success_rate : null
    }
  } catch {
    return null
  }
}

export interface SnapshotOptions {
  baseUrl?: string
  homeDir?: string
  fetchImpl?: FetchLike
}

/**
 * Compose everything into the single IPC payload the renderer polls.
 * Options exist only for tests; production callers use the defaults.
 */
export async function getSnapshot(options: SnapshotOptions = {}): Promise<AgentFieldSnapshot> {
  const baseUrl = options.baseUrl ?? getBaseUrl()
  const fetchImpl = options.fetchImpl ?? fetch

  const [controlPlane, registry] = await Promise.all([
    checkControlPlane(baseUrl, fetchImpl),
    readInstalledAgents(options.homeDir)
  ])

  // Only consult a recognized control plane; an unrelated service on the
  // port must not influence badges or show foreign runs as activity.
  const [nodeHealth, executions, metrics] = controlPlane.recognized
    ? await Promise.all([
        fetchControlPlaneNodes(baseUrl, fetchImpl),
        fetchExecutions(baseUrl, fetchImpl),
        fetchDashboardMetrics(baseUrl, fetchImpl)
      ])
    : [null, null, null]
  const hasControlPlaneView = nodeHealth !== null

  const agents = registry.agents.map((agent) => ({
    ...agent,
    badge: deriveAgentBadge(
      agent.status,
      hasControlPlaneView,
      nodeHealth?.get(agent.name) ?? null
    )
  }))

  return {
    controlPlane: { ...controlPlane, baseUrl },
    registry: { exists: registry.exists, agents, error: registry.error },
    executions,
    metrics,
    fetchedAt: new Date().toISOString()
  }
}
