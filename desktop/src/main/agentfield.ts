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
  InstalledAgent,
  RegistryResult
} from '../shared/types'

export const DEFAULT_BASE_URL = 'http://localhost:8080'

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
 *  - 200 {"status":"healthy",...}            -> { reachable: true,  healthy: true }
 *  - 503 {"status":"unhealthy",...}          -> { reachable: true,  healthy: false }
 *    (an HTTP response — even 503 — still means the control plane is reachable)
 *  - network error / timeout (3s)            -> { reachable: false, healthy: false, error }
 */
export async function checkControlPlane(
  baseUrl: string = DEFAULT_BASE_URL,
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
    return { reachable: true, healthy: res.ok, raw }
  } catch (err) {
    return { reachable: false, healthy: false, error: errorMessage(err) }
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
 * GET {baseUrl}/api/v1/nodes -> {"nodes":[{"id":...,"health_status":...},...],"count":N}
 * (the server's default filter returns active nodes only).
 * Returns the list of node ids, or null on any failure — callers treat null as
 * "control plane view unavailable" and fall back to registry status alone.
 */
export async function fetchControlPlaneNodes(
  baseUrl: string = DEFAULT_BASE_URL,
  fetchImpl: FetchLike = fetch
): Promise<string[] | null> {
  try {
    const res = await fetchImpl(`${baseUrl}/api/v1/nodes`, {
      signal: AbortSignal.timeout(HTTP_TIMEOUT_MS)
    })
    if (!res.ok) return null
    const body: unknown = await res.json()
    if (!isRecord(body) || !Array.isArray(body.nodes)) return null
    return body.nodes
      .filter(isRecord)
      .map((node) => (typeof node.id === 'string' ? node.id : ''))
      .filter((id) => id.length > 0)
  } catch {
    return null
  }
}

/**
 * Pure badge derivation. `controlPlaneReachable` here means "we have a usable
 * control-plane node view" (health reachable AND the nodes list fetched).
 *
 * CP view unavailable — trust the registry:
 *   'running' -> 'running' | 'stopped' -> 'stopped' | other/absent -> 'unknown'
 * CP view available — cross-check:
 *   registry running + node seen      -> 'running'
 *   registry running + node NOT seen  -> 'unknown'  (stale registry)
 *   registry stopped + node seen      -> 'unknown'  (conflict)
 *   registry stopped + node NOT seen  -> 'stopped'
 *   other/absent registry status      -> 'unknown'
 */
export function deriveAgentBadge(
  registryStatus: string | undefined,
  controlPlaneReachable: boolean,
  nodeSeenOnControlPlane: boolean
): AgentBadge {
  if (!controlPlaneReachable) {
    if (registryStatus === 'running') return 'running'
    if (registryStatus === 'stopped') return 'stopped'
    return 'unknown'
  }
  if (registryStatus === 'running') {
    return nodeSeenOnControlPlane ? 'running' : 'unknown'
  }
  if (registryStatus === 'stopped') {
    return nodeSeenOnControlPlane ? 'unknown' : 'stopped'
  }
  return 'unknown'
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
  const baseUrl = options.baseUrl ?? DEFAULT_BASE_URL
  const fetchImpl = options.fetchImpl ?? fetch

  const [controlPlane, registry] = await Promise.all([
    checkControlPlane(baseUrl, fetchImpl),
    readInstalledAgents(options.homeDir)
  ])

  const nodeIds = controlPlane.reachable
    ? await fetchControlPlaneNodes(baseUrl, fetchImpl)
    : null
  const hasControlPlaneView = nodeIds !== null
  const seen = new Set(nodeIds ?? [])

  const agents = registry.agents.map((agent) => ({
    ...agent,
    badge: deriveAgentBadge(agent.status, hasControlPlaneView, seen.has(agent.name))
  }))

  return {
    controlPlane: { ...controlPlane, baseUrl },
    registry: { exists: registry.exists, agents, error: registry.error },
    fetchedAt: new Date().toISOString()
  }
}
