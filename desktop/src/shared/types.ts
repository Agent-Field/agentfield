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
  /** Install dir (~/.agentfield/packages/<name>) — where the manifest lives. */
  path: string | null
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

/** One workflow run parsed from GET /api/ui/v2/workflow-runs. */
export interface ExecutionSummary {
  runId: string
  /** e.g. "running", "succeeded", "failed" */
  status: string
  /** Human-facing name (the root reasoner, e.g. "demo_echo"). */
  displayName: string
  agentId: string
  startedAt: string
  durationMs: number | null
  /** True once the run reached a terminal state. */
  terminal: boolean
}

/** Executions view: in-flight runs plus a short tail of finished ones. */
export interface ExecutionsResult {
  running: ExecutionSummary[]
  recent: ExecutionSummary[]
}

/** One installable node in the curated catalog (see shared/catalog.ts). */
export interface CatalogEntry {
  /** Node name, matches the registry key after install. */
  name: string
  description: string
  /** `af install` source: a git URL or af://registry/<name> reference. */
  source: string
  language?: string
}

/** Terminal states of an install kicked off from the app. */
export interface InstallResult {
  ok: boolean
  message: string
}

/** Outcome of a start/stop/restart issued from the app. */
export interface AgentActionResult {
  ok: boolean
  message: string
}

/**
 * How one declared variable resolves for `af run`, mirroring the CLI's
 * EnvResolver order: process env → encrypted secret store → manifest default.
 */
export type EnvVarStatus = 'env' | 'stored' | 'default' | 'missing'

/** One variable an agent's manifest declares under user_environment. */
export interface AgentEnvVar {
  name: string
  description: string
  /** Manifest type: secret — render a password input, mask everywhere. */
  secret: boolean
  /** Store scope a set writes to: shared "global" (default) or per-node. */
  scope: 'global' | 'node'
  /** Must resolve for `af run` to succeed (required list or a group member). */
  required: boolean
  /** require_one_of group id — any one member resolving satisfies the group. */
  group?: string
  groupDescription?: string
  status: EnvVarStatus
  /** Secret-store scopes currently holding this key ("global" or the node name). */
  storedScopes: string[]
}

/**
 * Everything the renderer needs to show and edit one agent's keys. Values
 * themselves never cross the IPC boundary — only these status flags do.
 */
export interface AgentEnvReport {
  agent: string
  vars: AgentEnvVar[]
  /** Every required variable and group resolves — `af run` won't fail on env. */
  satisfied: boolean
  /** Set when the secret store could not be read (statuses degrade gracefully). */
  error?: string
}

/** Which af CLI the app resolved and whether an installed copy needs updating. */
export interface CliStatus {
  /** Spawnable command (absolute path or bare "af"), null when none usable. */
  command: string | null
  /** Where the resolved CLI came from. */
  source: 'managed' | 'path' | 'bundled' | null
  /** Its version, or null for dev/unparseable builds (trusted as-is). */
  version: string | null
  /** Oldest version this app can drive. */
  minVersion: string
  /** An installed copy that is too old — drives the "Update AgentField" banner. */
  outdated: { source: string; version: string } | null
  /** The app package carries a CLI it can (re)install. */
  bundledAvailable: boolean
  bundledVersion: string | null
}

/**
 * Persisted app settings (settings.json in the app's user-data dir).
 * The goal: the app is "just there" — it boots at login, brings the control
 * plane up, starts the agents you selected, and everything is queryable the
 * moment Claude/Codex/anything asks.
 */
export interface DesktopSettings {
  /** Launch the app when you log in (starts hidden, in the tray). */
  openAtLogin: boolean
  /** Start the control plane on app launch when nothing is listening. */
  autostartControlPlane: boolean
  /** Installed agent names to start once the control plane is healthy. */
  autostartAgents: string[]
  /**
   * Keep the AgentField skills (agentfield: building agents; agentfield-use:
   * calling installed ones) installed in detected coding agents (Claude
   * Code, Codex, …) via `af skill install` — so they know how to use this.
   */
  installSkills: boolean
}

/** Headline numbers from GET /api/ui/v1/dashboard/summary. */
export interface DashboardMetrics {
  agentsRunning: number
  agentsTotal: number
  executionsToday: number
  executionsYesterday: number
  /** Percentage 0-100, or null when the server reports none. */
  successRate: number | null
}

/** The single payload shipped over IPC to the renderer. */
export interface AgentFieldSnapshot {
  controlPlane: ControlPlaneStatus & { baseUrl: string }
  registry: {
    exists: boolean
    agents: SnapshotAgent[]
    error?: string
  }
  /** null when the control plane view is unavailable. */
  executions: ExecutionsResult | null
  /** null when the control plane view is unavailable. */
  metrics: DashboardMetrics | null
  /** ISO timestamp of when this snapshot was assembled. */
  fetchedAt: string
}

/** Surface exposed on window.agentfield by the preload script. */
export interface AgentFieldApi {
  getSnapshot(): Promise<AgentFieldSnapshot>
  getCatalog(): Promise<CatalogEntry[]>
  /** Install a catalog entry by name. Resolves when `af install` exits. */
  install(name: string): Promise<InstallResult>
  /** Start / stop / restart an installed agent by its registry name. */
  agentAction(action: 'start' | 'stop' | 'restart', name: string): Promise<AgentActionResult>
  /** Env/secret status for every installed agent that declares variables. */
  getEnvReports(): Promise<AgentEnvReport[]>
  /** Store a declared variable's value in af's encrypted secret store. */
  setAgentSecret(agent: string, key: string, value: string): Promise<AgentActionResult>
  /** Remove a stored value from every scope relevant to this agent. */
  revokeAgentSecret(agent: string, key: string): Promise<AgentActionResult>
  getSettings(): Promise<DesktopSettings>
  /** Merge a partial update into the settings; returns the result. */
  setSettings(patch: Partial<DesktopSettings>): Promise<DesktopSettings>
  /** Which af CLI the app is using (managed / PATH / bundled) and its version. */
  getCliStatus(): Promise<CliStatus>
  /** Install/refresh the bundled CLI into ~/.agentfield/bin; returns new status. */
  updateCli(): Promise<CliStatus>
  /** Subscribe to install output lines; returns an unsubscribe function. */
  onInstallProgress(listener: (line: string) => void): () => void
  /**
   * Subscribe to deep-link navigation (agentfield://<view>). The view arrives
   * as a plain string over IPC; validate with isView() before trusting it.
   */
  onNavigate(listener: (view: string) => void): () => void
  /**
   * Tell the main process the navigation listener is live. Returns the view
   * of a deep link that arrived before then (e.g. the link that cold-started
   * a hidden app), or null. Call once, after subscribing with onNavigate.
   */
  announceReady(): Promise<string | null>
  /** "darwin" | "win32" | "linux" — for platform-specific chrome (traffic-light inset). */
  platform: string
}
