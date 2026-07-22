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
  /** Root execution's error message when the run failed (null otherwise). */
  errorMessage: string | null
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

/** One entry of the whole secret store (values never leave the store). */
export interface StoredSecret {
  key: string
  /** "global" (shared by every agent) or the node name it is scoped to. */
  scope: string
  /** Installed agents whose manifest declares this variable. */
  usedBy: string[]
}

/** The Secrets view payload: everything `af secrets ls` knows. */
export interface SecretsListResult {
  secrets: StoredSecret[]
  /** Set when the store could not be read (e.g. no usable af CLI). */
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
  /**
   * Fixed control-plane port, or null for automatic (prefer 8080, else the
   * next free port). A fixed port is used exactly: the app starts the control
   * plane there and reports a conflict instead of silently moving elsewhere.
   */
  controlPlanePort: number | null
  /**
   * The port of the control plane this app last started or adopted. App-
   * managed, not a user preference: it lets a restarted app rediscover a
   * control plane it put on a non-default port instead of starting a second
   * one somewhere else.
   */
  lastControlPlanePort: number | null
  /** Installed agent names to start once the control plane is healthy. */
  autostartAgents: string[]
  /**
   * Keep the AgentField skills (agentfield: building agents; agentfield-use:
   * calling installed ones) installed in detected coding agents (Claude
   * Code, Codex, …) via `af skill install` — so they know how to use this.
   */
  installSkills: boolean
  /**
   * macOS only: keep the af-tray menu-bar companion provisioned and installed
   * (~/.agentfield/bin/af-tray → ~/Applications/AgentField.app + launchd), so a
   * desktop-app-only install still gets the menu-bar icon. Meaningless on
   * Windows/Linux, where the app carries its own in-app tray; the setting is
   * present but the toggle is hidden there.
   */
  trayCompanion: boolean
  /**
   * App-update version the user dismissed from the banner (null = none).
   * Hides the banner for that version only — a newer release brings it
   * back, and Settings keeps offering the update regardless.
   */
  dismissedUpdateVersion: string | null
}

/** A newer app release found on GitHub (the desktop app's update channel). */
export interface AppUpdateInfo {
  /** Release version without the v prefix (e.g. "0.1.110"). */
  version: string
  tagName: string
  /** Release page — the fallback when no platform installer asset exists. */
  releaseUrl: string
  /** This platform's installer asset, when the release carries one. */
  assetName: string | null
  assetUrl: string | null
  assetSize: number | null
}

/** State of the app's own update flow (check → download → hand-off). */
export interface AppUpdateStatus {
  currentVersion: string
  checking: boolean
  /** A release newer than currentVersion, else null. */
  available: AppUpdateInfo | null
  /** ISO time of the last successful check, null before the first. */
  lastCheckedAt: string | null
  downloading: boolean
  /** Whole percent 0-100 while downloading, null otherwise. */
  progress: number | null
  error: string | null
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
  /**
   * Install a node from a pasted GitHub repo URL
   * (`https://github.com/<owner>/<repo>` or `…/<repo>//<subdir>`). The main
   * process validates the shape and refuses anything else. Shares the install
   * mutex and progress channel with catalog installs.
   */
  installFromSource(source: string): Promise<InstallResult>
  /** Uninstall an installed agent (stops it first; removes files + secrets). */
  uninstall(name: string): Promise<AgentActionResult>
  /**
   * Update an installed catalog agent to the latest version of its source
   * (reinstall in place; secrets survive). A running agent is stopped for
   * the update and restarted after; a stopped one stays stopped.
   */
  update(name: string): Promise<InstallResult>
  /** Start / stop / restart an installed agent by its registry name. */
  agentAction(action: 'start' | 'stop' | 'restart', name: string): Promise<AgentActionResult>
  /** Env/secret status for every installed agent that declares variables. */
  getEnvReports(): Promise<AgentEnvReport[]>
  /** Store a declared variable's value in af's encrypted secret store. */
  setAgentSecret(agent: string, key: string, value: string): Promise<AgentActionResult>
  /** Remove a stored value from every scope relevant to this agent. */
  revokeAgentSecret(agent: string, key: string): Promise<AgentActionResult>
  /** Every secret in the store, with the agents that declare each one. */
  listSecrets(): Promise<SecretsListResult>
  /** Remove one stored secret from one scope (global = for all agents). */
  revokeSecret(key: string, scope: string): Promise<AgentActionResult>
  getSettings(): Promise<DesktopSettings>
  /** Merge a partial update into the settings; returns the result. */
  setSettings(patch: Partial<DesktopSettings>): Promise<DesktopSettings>
  /** Which af CLI the app is using (managed / PATH / bundled) and its version. */
  getCliStatus(): Promise<CliStatus>
  /** Install/refresh the bundled CLI into ~/.agentfield/bin; returns new status. */
  updateCli(): Promise<CliStatus>
  /** Current state of the app's own update check. */
  getAppUpdateStatus(): Promise<AppUpdateStatus>
  /** Ask GitHub for the latest release now; resolves to the refreshed status. */
  checkForAppUpdate(): Promise<AppUpdateStatus>
  /**
   * Download this platform's installer and hand off to it (Windows quits
   * into the one-click installer; macOS opens the DMG). Falls back to the
   * release page when the release has no installer for this platform.
   */
  installAppUpdate(): Promise<AppUpdateStatus>
  /** Subscribe to update-status pushes (auto-checks, download progress). */
  onAppUpdateStatus(listener: (status: AppUpdateStatus) => void): () => void
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
  /**
   * Open a control-plane web-UI page in the default browser. `path` must be
   * an absolute path on the control plane (e.g. "/ui/runs/run_123"); the main
   * process validates it and joins it to the control-plane base URL.
   */
  openWebUI(path: string): Promise<boolean>
  /** "darwin" | "win32" | "linux" — for platform-specific chrome (traffic-light inset). */
  platform: string
}
