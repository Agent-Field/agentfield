// Shared types crossing the main / preload / renderer IPC boundary.
// Import these type-only from every layer — this file must stay runtime-free.

/** Result of probing GET {baseUrl}/health on the control plane. */
export interface ControlPlaneStatus {
  /** An HTTP response came back (any status code, including 503). */
  reachable: boolean
  /**
   * The response body looks like an AgentField control plane health payload
   * (status: "healthy" | "unhealthy"). False when some unrelated service is
   * squatting on the port — its nodes view must not be trusted.
   */
  recognized: boolean
  /** The health endpoint answered 200 with a body reporting "healthy". */
  healthy: boolean
  /** Raw JSON body of the health response, when one was parseable. */
  raw?: unknown
  /** Network/timeout error when unreachable, or why the payload was rejected. */
  error?: string
}

/** One entry parsed from ~/.agentfield/installed.yaml. */
export interface InstalledAgent {
  name: string
  version: string
  description: string
  /** Optional on newer registry entries (python/go); absent on older ones. */
  language?: string
  /** Raw registry status string (e.g. "running", "stopped"). */
  status: string
  port: number | null
  pid: number | null
}

/** Registry read result. Missing file/dir is a graceful empty state, not an error. */
export interface RegistryResult {
  exists: boolean
  agents: InstalledAgent[]
  /** Set when the registry file exists but could not be parsed. */
  error?: string
}

/** Status badge shown in the UI, derived from registry + control-plane view. */
export type AgentBadge = 'running' | 'stopped' | 'unknown'

export interface SnapshotAgent extends InstalledAgent {
  badge: AgentBadge
}

/** The single payload shipped over IPC to the renderer. */
export interface AgentFieldSnapshot {
  controlPlane: ControlPlaneStatus & { baseUrl: string }
  registry: {
    exists: boolean
    agents: SnapshotAgent[]
    error?: string
  }
  /** ISO timestamp of when this snapshot was assembled. */
  fetchedAt: string
}

/** Surface exposed on window.agentfield by the preload script. */
export interface AgentFieldApi {
  getSnapshot(): Promise<AgentFieldSnapshot>
}
