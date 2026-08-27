import { getApiKey, getBaseUrl } from './connection'
import type {
  ControlPlaneVersion,
  PackageMaintenanceStatus,
  PackageUpdateCheckResponse,
  PackageUpdateStatus
} from '../shared/types'
export type FetchLike = typeof fetch

const READ_TIMEOUT_MS = 3_000
const VERSION_TIMEOUT_MS = 10_000
const MUTATION_TIMEOUT_MS = 10_000
// Stopping a node that has stopped answering takes the control plane a
// confirmation window (three probes) plus the graceful-shutdown wait before
// it signals; the request must outlive that.
const LIFECYCLE_TIMEOUT_MS = 60_000
const PACKAGE_MAINTENANCE_TIMEOUT_MS = 120_000
const DEFAULT_WATCH_INTERVAL_MS = 1_000
const DEFAULT_WATCH_TIMEOUT_MS = 15 * 60_000

export type InstallJobStatus = 'pending' | 'running' | 'succeeded' | 'failed'
export type InstallJobKind = 'install' | 'update' | 'uninstall'

export interface InstallJob {
  id: string
  source: string
  kind: InstallJobKind
  status: InstallJobStatus
  package_name?: string
  error?: string
  lines: string[]
  started_at?: string
  finished_at?: string
}

/**
 * True when a listing row represents a package actually present on disk.
 * The listing also contains catalog/marketplace rows; `install_status` is the
 * registry-backed discriminator. Older control planes omit the field — then
 * every row is kept (no way to tell, and hiding real installs is worse).
 */
export function isInstalledPackage(pkg: PackageInfo): boolean {
  if (pkg.install_status === undefined) return true
  return ['installed', 'running', 'stopped'].includes(pkg.install_status)
}

export interface PackageInfo {
  id: string
  name: string
  version: string
  status: string
  /**
   * Raw registry-backed install state ("installed" | "running" | "stopped" |
   * "uninstalled" | ...). Absent on control planes older than the sync
   * reconciliation fix; `status` above is a derived configuration summary,
   * not an install indicator.
   */
  install_status?: string
  installed_at?: string
  install_path: string
  /** Recorded installed.yaml source_path; absent on older control planes. */
  source?: string
  configuration_required: boolean
  configuration_complete: boolean
  running_node_id?: string
  last_started?: string
  process_id?: number
  port?: number
  description: string
  author: string
  installed_commit?: string
  source_ref?: string
  auto_update?: boolean
  update?: {
    status: PackageUpdateStatus
    latest_commit: string
    checked_at: string
    message: string
  }
}

export interface PackageListResponse {
  packages: PackageInfo[]
  total: number
}

export interface AgentEndpoints {
  health: string
  reasoners: string
  skills: string
}

export interface StartAgentResponse {
  agent_id: string
  status: string
  pid: number
  port: number
  started_at: string
  log_file: string
  message: string
  endpoints: AgentEndpoints
}

export interface StopAgentResponse {
  agent_id: string
  status: string
  message: string
}

export interface AgentStatusResponse {
  agent_id: string
  name: string
  is_running: boolean
  status: string
  pid?: number
  port?: number
  uptime?: string
  last_seen?: string
  configuration_required?: boolean
  configuration_status?: string
  endpoints?: AgentEndpoints
  message?: string
}

export interface RunningAgent {
  agent_id: string
  name: string
  status: string
  pid: number
  port: number
  started_at: string
  log_file: string
  package?: {
    name: string
    version: string
    description: string
    author: string
  }
  endpoints?: AgentEndpoints
}

export interface RunningAgentsResponse {
  running_agents: RunningAgent[]
  total_count: number
}

export type SecretScope = 'node' | 'global'

export interface AgentSecretStatus {
  key: string
  is_set: boolean
  env?: boolean
  scope?: SecretScope
  declared_scope?: SecretScope
  description?: string
  secret?: boolean
  default?: string
  requirement?: 'required' | 'one_of' | 'optional' | ''
  group?: string
  group_description?: string
}

export interface SecretReference {
  key: string
  scope: string
}

export interface CpClientOptions {
  baseUrl?: () => string
  apiKey?: () => string | null
  fetchImpl?: FetchLike
  /** Test seam used by job polling. */
  sleep?: (milliseconds: number) => Promise<void>
  /** Test seam used by job polling. */
  now?: () => number
}

export interface WatchInstallJobOptions {
  intervalMs?: number
  timeoutMs?: number
}

export class CpApiError extends Error {
  readonly status: number
  readonly code?: string | number
  readonly activeExecutions?: number

  constructor(details: {
    status: number
    code?: string | number
    message: string
    activeExecutions?: number
  }) {
    super(details.message)
    this.name = 'CpApiError'
    this.status = details.status
    this.code = details.code
    this.activeExecutions = details.activeExecutions
  }
}

export interface CpClient {
  /** GET /api/v1/version. A 404 means a legacy control plane and maps to null. */
  getVersion(): Promise<ControlPlaneVersion | null>
  /** POST /api/ui/v1/agents/packages/install. */
  installPackage(source: string, force?: boolean): Promise<{ job_id: string }>
  /** GET /api/ui/v1/agents/packages/install/jobs/:jobId. */
  getInstallJob(jobId: string): Promise<InstallJob>
  /** GET /api/ui/v1/agents/packages/install/jobs. */
  listInstallJobs(): Promise<InstallJob[]>
  /** POST /api/ui/v1/agents/packages/:packageId/uninstall. */
  uninstallPackage(packageId: string): Promise<{ package_id: string; status: string }>
  /** POST /api/ui/v1/agents/packages/:packageId/update. */
  updatePackage(
    packageId: string,
    options?: { source?: string; force?: boolean }
  ): Promise<{ job_id: string }>
  /** GET /api/ui/v1/agents/packages. */
  listPackages(): Promise<PackageListResponse>
  /** POST /api/ui/v1/agents/packages/check-updates. */
  checkPackageUpdates(): Promise<PackageUpdateCheckResponse>
  /** PUT /api/ui/v1/agents/packages/:id/auto-update. */
  setPackageAutoUpdate(packageId: string, enabled: boolean): Promise<PackageInfo>
  /** GET /api/ui/v1/agents/packages/maintenance. */
  getMaintenanceStatus(): Promise<PackageMaintenanceStatus>
  /** POST /api/ui/v1/agents/packages/maintenance/run. */
  runPackageMaintenance(): Promise<{ started: boolean }>
  /** POST /api/ui/v1/agents/:agentId/start. */
  startAgent(
    agentId: string,
    options?: { port?: number; detach?: boolean }
  ): Promise<StartAgentResponse>
  /** POST /api/ui/v1/agents/:agentId/stop. */
  stopAgent(agentId: string): Promise<StopAgentResponse>
  /** GET /api/ui/v1/agents/:agentId/status. */
  getAgentStatus(agentId: string): Promise<AgentStatusResponse>
  /** GET /api/ui/v1/agents/running. */
  listRunningAgents(): Promise<RunningAgentsResponse>
  /** GET /api/ui/v1/agents/:agentId/secrets?include=env. */
  listAgentSecrets(agentId: string): Promise<{ secrets: AgentSecretStatus[] }>
  /** PUT /api/ui/v1/agents/:agentId/secrets. */
  setAgentSecret(
    agentId: string,
    key: string,
    value: string,
    scope?: SecretScope
  ): Promise<void>
  /** DELETE /api/ui/v1/agents/:agentId/secrets/:key. */
  deleteAgentSecret(agentId: string, key: string, scope?: SecretScope): Promise<void>
  /** GET /api/ui/v1/secrets. */
  listAllSecrets(): Promise<{ secrets: SecretReference[] }>
  /** Probes GET /api/ui/v1/agents/packages/install/jobs for install-route support. */
  hasInstallApi(): Promise<boolean>
  /** Polls GET /api/ui/v1/agents/packages/install/jobs/:jobId until terminal. */
  watchInstallJob(
    jobId: string,
    onLine: (line: string) => void,
    options?: WatchInstallJobOptions
  ): Promise<InstallJob>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function apiError(response: Response): Promise<CpApiError> {
  let body: unknown
  try {
    body = await response.json()
  } catch {
    body = undefined
  }
  const record = isRecord(body) ? body : undefined
  const serverMessage =
    typeof record?.error === 'string'
      ? record.error
      : typeof record?.message === 'string'
        ? record.message
        : undefined
  const code =
    typeof record?.code === 'string' || typeof record?.code === 'number'
      ? record.code
      : undefined
  const activeExecutions =
    typeof record?.active_executions === 'number' ? record.active_executions : undefined
  return new CpApiError({
    status: response.status,
    code,
    activeExecutions,
    message:
      response.status === 401
        ? unauthorizedMessage(record, serverMessage)
        : (serverMessage ?? `Control plane request failed with status ${response.status}`)
  })
}

/**
 * Message for a 401 from the control plane's auth guards, whose body is
 * `{error:"unauthorized", message:"<human readable>", help:{cli, …}}`.
 * `error` there is a machine code, so the generic "prefer error" path above
 * would show the user nothing but "unauthorized" — take the sentence instead,
 * and append the CLI hint when the server offers one.
 */
function unauthorizedMessage(
  record: Record<string, unknown> | undefined,
  fallback: string | undefined
): string {
  const message =
    (typeof record?.message === 'string' && record.message !== '' ? record.message : undefined) ??
    // Older/other responses put a real sentence in `error` instead.
    (fallback !== undefined && fallback !== 'unauthorized' ? fallback : undefined) ??
    'The control plane rejected this request: it requires an API key.'
  const help = record?.help
  const cli =
    isRecord(help) && typeof help.cli === 'string' && help.cli !== '' ? help.cli : undefined
  return cli ? `${message} Run: ${cli}` : message
}

/**
 * Creates a typed client for the control-plane `/api/ui/v1` management API.
 * The base URL and API key providers are evaluated for every request.
 */
export function createCpClient(options: CpClientOptions = {}): CpClient {
  const baseUrl = options.baseUrl ?? getBaseUrl
  const apiKey = options.apiKey ?? getApiKey
  const fetchImpl = options.fetchImpl ?? fetch
  const sleep =
    options.sleep ?? ((milliseconds: number) => new Promise((resolve) => setTimeout(resolve, milliseconds)))
  const now = options.now ?? Date.now

  async function request<T>(
    path: string,
    init: Omit<RequestInit, 'signal'> = {},
    options: { mutation?: boolean; noContent?: boolean; timeoutMs?: number } = {}
  ): Promise<T> {
    const headers = new Headers(init.headers)
    const key = apiKey()
    if (key !== null) headers.set('X-API-Key', key)
    if (init.body !== undefined) headers.set('Content-Type', 'application/json')

    const response = await fetchImpl(`${baseUrl().replace(/\/+$/, '')}${path}`, {
      ...init,
      headers,
      signal: AbortSignal.timeout(
        options.timeoutMs ?? (options.mutation ? MUTATION_TIMEOUT_MS : READ_TIMEOUT_MS)
      )
    })
    if (!response.ok) throw await apiError(response)
    if (options.noContent) return undefined as T
    return (await response.json()) as T
  }

  const client: CpClient = {
    async getVersion() {
      try {
        return await request<ControlPlaneVersion>('/api/v1/version', {}, {
          timeoutMs: VERSION_TIMEOUT_MS
        })
      } catch (error) {
        if (error instanceof CpApiError && error.status === 404) return null
        throw error
      }
    },
    installPackage(source, force) {
      const body = force === undefined ? { source } : { source, force }
      return request('/api/ui/v1/agents/packages/install', {
        method: 'POST',
        body: JSON.stringify(body)
      }, { mutation: true })
    },
    async getInstallJob(jobId) {
      const job = await request<Omit<InstallJob, 'lines'> & { lines: string[] | null }>(
        `/api/ui/v1/agents/packages/install/jobs/${encodeURIComponent(jobId)}`
      )
      return { ...job, lines: job.lines ?? [] }
    },
    async listInstallJobs() {
      const jobs = await request<InstallJob[] | null>('/api/ui/v1/agents/packages/install/jobs')
      return jobs ?? []
    },
    uninstallPackage(packageId) {
      return request(
        `/api/ui/v1/agents/packages/${encodeURIComponent(packageId)}/uninstall`,
        { method: 'POST' },
        { mutation: true }
      )
    },
    updatePackage(packageId, updateOptions) {
      const body = updateOptions && (
        updateOptions.source !== undefined || updateOptions.force !== undefined
      )
        ? JSON.stringify(updateOptions)
        : undefined
      return request(
        `/api/ui/v1/agents/packages/${encodeURIComponent(packageId)}/update`,
        body === undefined ? { method: 'POST' } : { method: 'POST', body },
        { mutation: true }
      )
    },
    async listPackages() {
      const response = await request<
        Omit<PackageListResponse, 'packages'> & { packages: PackageInfo[] | null }
      >(
        '/api/ui/v1/agents/packages'
      )
      return { ...response, packages: response.packages ?? [] }
    },
    checkPackageUpdates() {
      return request('/api/ui/v1/agents/packages/check-updates', { method: 'POST' }, {
        mutation: true,
        timeoutMs: PACKAGE_MAINTENANCE_TIMEOUT_MS
      })
    },
    setPackageAutoUpdate(packageId, enabled) {
      return request(
        `/api/ui/v1/agents/packages/${encodeURIComponent(packageId)}/auto-update`,
        { method: 'PUT', body: JSON.stringify({ enabled }) },
        { mutation: true }
      )
    },
    getMaintenanceStatus() {
      return request('/api/ui/v1/agents/packages/maintenance')
    },
    runPackageMaintenance() {
      return request('/api/ui/v1/agents/packages/maintenance/run', { method: 'POST' }, {
        mutation: true,
        timeoutMs: PACKAGE_MAINTENANCE_TIMEOUT_MS
      })
    },
    startAgent(agentId, startOptions) {
      return request(
        `/api/ui/v1/agents/${encodeURIComponent(agentId)}/start`,
        { method: 'POST', body: JSON.stringify(startOptions ?? {}) },
        { mutation: true, timeoutMs: LIFECYCLE_TIMEOUT_MS }
      )
    },
    stopAgent(agentId) {
      return request(
        `/api/ui/v1/agents/${encodeURIComponent(agentId)}/stop`,
        { method: 'POST' },
        { mutation: true, timeoutMs: LIFECYCLE_TIMEOUT_MS }
      )
    },
    getAgentStatus(agentId) {
      return request(`/api/ui/v1/agents/${encodeURIComponent(agentId)}/status`)
    },
    async listRunningAgents() {
      const response = await request<
        Omit<RunningAgentsResponse, 'running_agents'> & { running_agents: RunningAgent[] | null }
      >('/api/ui/v1/agents/running')
      return { ...response, running_agents: response.running_agents ?? [] }
    },
    async listAgentSecrets(agentId) {
      const response = await request<{ secrets: AgentSecretStatus[] | null }>(
        `/api/ui/v1/agents/${encodeURIComponent(agentId)}/secrets?include=env`
      )
      return { ...response, secrets: response.secrets ?? [] }
    },
    setAgentSecret(agentId, key, value, scope) {
      const body = scope === undefined ? { key, value } : { key, value, scope }
      return request(
        `/api/ui/v1/agents/${encodeURIComponent(agentId)}/secrets`,
        { method: 'PUT', body: JSON.stringify(body) },
        { mutation: true, noContent: true }
      )
    },
    deleteAgentSecret(agentId, key, scope) {
      const query = scope === undefined ? '' : `?scope=${encodeURIComponent(scope)}`
      return request(
        `/api/ui/v1/agents/${encodeURIComponent(agentId)}/secrets/${encodeURIComponent(key)}${query}`,
        { method: 'DELETE' },
        { mutation: true, noContent: true }
      )
    },
    async listAllSecrets() {
      const response = await request<{ secrets: SecretReference[] | null }>('/api/ui/v1/secrets')
      return { ...response, secrets: response.secrets ?? [] }
    },
    async hasInstallApi() {
      try {
        await client.listInstallJobs()
        return true
      } catch (error) {
        if (error instanceof CpApiError && error.status === 404) return false
        throw error
      }
    },
    async watchInstallJob(jobId, onLine, watchOptions = {}) {
      const intervalMs = watchOptions.intervalMs ?? DEFAULT_WATCH_INTERVAL_MS
      const timeoutMs = watchOptions.timeoutMs ?? DEFAULT_WATCH_TIMEOUT_MS
      const startedAt = now()
      let emittedLineCount = 0

      for (;;) {
        if (now() - startedAt >= timeoutMs) {
          throw new Error(`Timed out waiting for install job ${jobId}`)
        }
        let job: InstallJob
        try {
          job = await client.getInstallJob(jobId)
        } catch (error) {
          if (error instanceof CpApiError && error.status === 404) {
            throw new CpApiError({
              status: 404,
              code: error.code,
              message: `Install job ${jobId} disappeared`
            })
          }
          throw error
        }
        for (const line of job.lines.slice(emittedLineCount)) onLine(line)
        emittedLineCount = Math.max(emittedLineCount, job.lines.length)
        if (job.status === 'succeeded' || job.status === 'failed') return job
        await sleep(intervalMs)
      }
    }
  }

  return client
}
