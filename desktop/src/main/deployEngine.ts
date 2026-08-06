import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { delimiter, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export interface DeploySpawnDeps {
  spawnImpl?: typeof import('node:child_process').spawn
  fetchImpl?: typeof fetch
  env?: Record<string, string>
}

export interface DeployEngineOptions {
  railwayToken: string
  workspaceId: string
  projectName?: string
  workspaceDir: string
  binaryDir?: string | null
  onLine?: (line: string) => void
}

export interface DeployResult {
  ok: boolean
  url?: string
  apiKey?: string
  furrowAddress?: string
  message: string
}

export interface CloudImageUpdate {
  current: string | null
  latest: string | null
  updateAvailable: boolean
}

const MODULE = `terraform {
  required_providers {
    railway = { source = "terraform-community-providers/railway", version = "0.6.2" }
  }
}
provider "railway" {}
variable "workspace_id" { type = string }
variable "project_name" {
  type    = string
  default = "agentfield"
}
variable "subdomain" { type = string }
variable "api_key" {
  type      = string
  sensitive = true
}
variable "image" {
  type = string
  # Floating fallback for when the release lookup fails; a fresh deploy still
  # lands on the newest production release at create time.
  default = "agentfield/control-plane-cloud:latest"
}
resource "railway_project" "cp" {
  name         = var.project_name
  workspace_id = var.workspace_id
}
resource "railway_service" "cp" {
  name         = "control-plane"
  project_id   = railway_project.cp.id
  # Pinned to the concrete release tag resolved at deploy time. Railway
  # resolves a floating tag once per deploy and never auto-repulls, and an
  # unchanged image string is a no-op apply — a ":latest" pin here froze
  # existing deployments on whatever "latest" meant the day they were
  # created. With a concrete tag, each new release changes the string, the
  # change is a diff, and the diff is what redeploys the service: re-running
  # deploy IS the upgrade path.
  source_image = var.image
  # The /data volume is created out-of-band (Railway GraphQL, ensureVolume) so
  # this module never declares one — but the provider refreshes it into state,
  # plans the undeclared attribute back to null, and its Update handler treats
  # that as "delete the volume" (volumeDelete — the data, not a detach). Every
  # re-apply would silently wipe the control plane's SQLite/BoltDB, secrets,
  # and installed agents. Ignoring the attribute keeps the refreshed value in
  # the plan so the delete branch can never fire.
  lifecycle {
    ignore_changes = [volume]
  }
}
resource "railway_variable" "api_key" {
  name           = "AGENTFIELD_API_KEY"
  value          = var.api_key
  environment_id = railway_project.cp.default_environment.id
  service_id     = railway_service.cp.id
}
resource "railway_variable" "port" {
  name           = "AGENTFIELD_PORT"
  value          = "8080"
  environment_id = railway_project.cp.default_environment.id
  service_id     = railway_service.cp.id
}
resource "railway_service_domain" "cp" {
  subdomain      = var.subdomain
  environment_id = railway_project.cp.default_environment.id
  service_id     = railway_service.cp.id
}
resource "railway_tcp_proxy" "furrow" {
  application_port = 8802
  environment_id   = railway_project.cp.default_environment.id
  service_id       = railway_service.cp.id
}
resource "railway_variable" "furrow_public_addr" {
  name           = "FURROW_PUBLIC_ADDR"
  value          = "\${trimsuffix(railway_tcp_proxy.furrow.domain, ".")}:\${railway_tcp_proxy.furrow.proxy_port}"
  environment_id = railway_project.cp.default_environment.id
  service_id     = railway_service.cp.id
}
output "url"     { value = "https://\${railway_service_domain.cp.domain}" }
output "furrow_domain" { value = trimsuffix(railway_tcp_proxy.furrow.domain, ".") }
output "furrow_port"   { value = railway_tcp_proxy.furrow.proxy_port }
output "project_id"     { value = railway_project.cp.id }
output "environment_id" { value = railway_project.cp.default_environment.id }
output "service_id"     { value = railway_service.cp.id }
output "api_key" {
  value     = var.api_key
  sensitive = true
}
`

const NO_ENGINE = 'Deploy engine not bundled. Install OpenTofu or rebuild AgentField Desktop with the deploy engine.'

const CLOUD_IMAGE_REPO = 'agentfield/control-plane-cloud'
const CLOUD_IMAGE_LATEST = `${CLOUD_IMAGE_REPO}:latest`
const DOCKER_HUB_TAGS = `https://hub.docker.com/v2/repositories/${CLOUD_IMAGE_REPO}/tags?page_size=100`

/**
 * Resolve the concrete release tag "latest" currently points at (release.yml
 * pushes both on every production release, so they share a digest). Returns
 * null when Docker Hub is unreachable or the digests don't line up — the
 * caller falls back to the pin already recorded in state (so a lookup outage
 * never rewrites a working deployment's image) or, for a fresh deployment,
 * to the floating tag.
 */
export async function resolveCloudImage(fetchImpl: typeof fetch = fetch): Promise<string | null> {
  try {
    const response = await fetchImpl(DOCKER_HUB_TAGS, { signal: AbortSignal.timeout(5000) })
    if (!response.ok) return null
    const payload = (await response.json()) as { results?: Array<{ name?: string; digest?: string }> }
    const tags = payload.results ?? []
    const digest = tags.find((tag) => tag.name === 'latest')?.digest
    const release = digest ? tags.find((tag) => tag.digest === digest && /^v\d+\.\d+\.\d+$/.test(tag.name ?? '')) : undefined
    return release?.name ? `${CLOUD_IMAGE_REPO}:${release.name}` : null
  } catch {
    return null
  }
}

function executableName(): string {
  return process.platform === 'win32' ? 'tofu.exe' : 'tofu'
}

function executableFile(path: string): boolean {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

function findOnPath(name: string): string | null {
  const names = process.platform === 'win32' && !name.endsWith('.exe') ? [`${name}.exe`, name] : [name]
  for (const dir of (process.env.PATH ?? '').split(delimiter)) {
    for (const candidate of names) {
      const path = join(dir, candidate)
      if (dir && executableFile(path)) return path
    }
  }
  return null
}

export function resolveTofuBinary(binaryDir?: string | null): string | null {
  if (binaryDir) {
    const candidate = join(binaryDir, executableName())
    if (executableFile(candidate)) return candidate
  }
  const here = dirname(fileURLToPath(import.meta.url))
  const vendor = resolve(here, '../../vendor/deploy-engine', executableName())
  if (executableFile(vendor)) return vendor
  return findOnPath('tofu') ?? findOnPath('terraform')
}

type TfState = {
  resources?: Array<{ type?: string; name?: string; instances?: Array<{ attributes?: Record<string, unknown> }> }>
  outputs?: Record<string, { value?: unknown }>
}

function readState(workspaceDir: string): TfState | null {
  try {
    return JSON.parse(readFileSync(join(workspaceDir, 'terraform.tfstate'), 'utf8')) as TfState
  } catch {
    return null
  }
}

export function hasDeployment(workspaceDir: string): boolean {
  return (readState(workspaceDir)?.resources?.length ?? 0) > 0
}

export function generateApiKey(): string {
  return randomBytes(24).toString('hex')
}

function stateOutput(state: TfState | null, name: string): string | null {
  const value = state?.outputs?.[name]?.value
  return typeof value === 'string' && value.length > 0 ? value : null
}

function stateSubdomain(state: TfState | null): string | null {
  const resource = state?.resources?.find((item) => item.type === 'railway_service_domain' && item.name === 'cp')
  const attrs = resource?.instances?.[0]?.attributes
  if (typeof attrs?.subdomain === 'string' && attrs.subdomain) return attrs.subdomain
  if (typeof attrs?.domain === 'string' && attrs.domain) return attrs.domain.split('.')[0] || null
  return null
}

function stateWorkspaceId(state: TfState | null): string | null {
  const resource = state?.resources?.find((item) => item.type === 'railway_project')
  const value = resource?.instances?.[0]?.attributes?.workspace_id
  return typeof value === 'string' && value.length > 0 ? value : null
}

function stateSourceImage(state: TfState | null): string | null {
  const resource = state?.resources?.find((item) => item.type === 'railway_service' && item.name === 'cp')
  const value = resource?.instances?.[0]?.attributes?.source_image
  return typeof value === 'string' && value.length > 0 ? value : null
}

export async function checkCloudImageUpdate(
  workspaceDir: string,
  fetchImpl: typeof fetch = fetch
): Promise<CloudImageUpdate> {
  const current = stateSourceImage(readState(workspaceDir))
  if (!current) return { current: null, latest: null, updateAvailable: false }
  const latest = await resolveCloudImage(fetchImpl)
  return {
    current,
    latest,
    updateAvailable: latest !== null && latest !== current
  }
}

function writeConfig(workspaceDir: string, binaryDir?: string | null): string | null {
  if (!binaryDir) return null
  const mirror = join(binaryDir, 'providers')
  if (!existsSync(mirror)) return null
  const config = join(workspaceDir, 'deploy.tfrc')
  writeFileSync(config, `provider_installation {
  filesystem_mirror { path = ${JSON.stringify(mirror)} }
  direct {}
}
`)
  return config
}

type CommandResult = { code: number; stdout: string; lastError?: string }

function runCommand(
  binary: string,
  args: string[],
  cwd: string,
  env: NodeJS.ProcessEnv,
  deps: DeploySpawnDeps,
  streamJson: boolean,
  onLine?: (line: string) => void
): Promise<CommandResult> {
  return new Promise((resolveCommand) => {
    const child = (deps.spawnImpl ?? spawn)(binary, args, { cwd, env, windowsHide: true })
    let stdout = ''
    let stderr = ''
    let pending = ''
    let lastError: string | undefined

    const consume = (text: string, final = false): void => {
      pending += text
      const lines = pending.split(/\r?\n/)
      pending = final ? '' : (lines.pop() ?? '')
      for (const line of lines) {
        if (!line.trim() || !streamJson) continue
        try {
          const event = JSON.parse(line) as Record<string, unknown>
          const type = String(event.type ?? '')
          if (!['apply_progress', 'apply_complete', 'apply_errored', 'diagnostic'].includes(type)) continue
          const diagnostic = event.diagnostic as Record<string, unknown> | undefined
          const summary = typeof diagnostic?.summary === 'string' ? diagnostic.summary : undefined
          const detail = typeof diagnostic?.detail === 'string' ? diagnostic.detail : undefined
          const message = type === 'diagnostic' ? [summary, detail].filter(Boolean).join(': ') : event['@message']
          if (typeof message === 'string' && message) onLine?.(message)
          if (type === 'apply_errored' || (type === 'diagnostic' && diagnostic?.severity === 'error')) {
            lastError = summary ?? detail ?? (typeof event['@message'] === 'string' ? event['@message'] : lastError)
          }
        } catch {
          // OpenTofu may mix non-JSON notices into the JSON stream; ignore them.
        }
      }
    }

    child.stdout?.on('data', (chunk: Buffer | string) => {
      const text = chunk.toString()
      stdout += text
      consume(text)
    })
    child.stderr?.on('data', (chunk: Buffer | string) => { stderr += chunk.toString() })
    child.on('error', (error) => resolveCommand({ code: -1, stdout, lastError: error.message }))
    child.on('close', (code) => {
      consume('', true)
      resolveCommand({ code: code ?? -1, stdout, lastError: lastError ?? (stderr.trim() || undefined) })
    })
  })
}

function deployEnv(opts: DeployEngineOptions, apiKey: string, subdomain: string, cliConfig: string | null, deps: DeploySpawnDeps, image?: string): NodeJS.ProcessEnv {
  return {
    ...process.env,
    ...deps.env,
    RAILWAY_TOKEN: opts.railwayToken,
    TF_VAR_workspace_id: opts.workspaceId,
    TF_VAR_project_name: opts.projectName ?? 'agentfield',
    TF_VAR_subdomain: subdomain,
    TF_VAR_api_key: apiKey,
    TF_IN_AUTOMATION: '1',
    ...(image ? { TF_VAR_image: image } : {}),
    ...(cliConfig ? { TF_CLI_CONFIG_FILE: cliConfig } : {})
  }
}

function providerReady(workspaceDir: string): boolean {
  if (!existsSync(join(workspaceDir, '.terraform'))) return false
  try {
    return /version\s*=\s*"0\.6\.2"/.test(readFileSync(join(workspaceDir, '.terraform.lock.hcl'), 'utf8'))
  } catch {
    return false
  }
}

const RAILWAY_GRAPHQL = 'https://backboard.railway.com/graphql/v2'

async function ensureVolume(
  railwayToken: string,
  projectId: string,
  environmentId: string,
  serviceId: string,
  fetchImpl: typeof fetch
): Promise<void> {
  const request = async (query: string, variables: Record<string, string>): Promise<Record<string, unknown>> => {
    const response = await fetchImpl(RAILWAY_GRAPHQL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${railwayToken}`,
        'Content-Type': 'application/json',
        'User-Agent': 'agentfield-desktop'
      },
      body: JSON.stringify({ query, variables })
    })
    const payload = await response.json() as { data?: Record<string, unknown>; errors?: Array<{ message?: string }> }
    if (!response.ok || payload.errors?.length) {
      throw new Error(payload.errors?.map((error) => error.message).filter(Boolean).join('; ') || `Railway GraphQL request failed (${response.status})`)
    }
    if (!payload.data) throw new Error('Railway GraphQL response contained no data')
    return payload.data
  }

  const query = `query Volumes($projectId: String!) {
  project(id: $projectId) { volumes { edges { node { id name } } } }
}`
  const data = await request(query, { projectId })
  const project = data.project as { volumes?: { edges?: unknown[] } } | undefined
  if ((project?.volumes?.edges?.length ?? 0) > 0) return

  const mutation = `mutation VolumeCreate($projectId: String!, $environmentId: String!, $serviceId: String!, $mountPath: String!) {
  volumeCreate(input: {projectId: $projectId, environmentId: $environmentId, serviceId: $serviceId, mountPath: $mountPath}) { id name }
}`
  await request(mutation, { projectId, environmentId, serviceId, mountPath: '/data' })
}

export async function runDeploy(opts: DeployEngineOptions, deps: DeploySpawnDeps = {}): Promise<DeployResult> {
  const binary = resolveTofuBinary(opts.binaryDir)
  if (!binary) return { ok: false, message: NO_ENGINE }
  mkdirSync(opts.workspaceDir, { recursive: true })
  writeFileSync(join(opts.workspaceDir, 'main.tf'), MODULE)
  const state = readState(opts.workspaceDir)
  const existing = (state?.resources?.length ?? 0) > 0
  // Credentials are deployment identity: only create one for new state and never rotate it on reconciliation.
  const apiKey = existing ? stateOutput(state, 'api_key') : generateApiKey()
  if (!apiKey) return { ok: false, message: 'Existing deployment is missing its API key; refusing to rotate it automatically.' }
  const projectName = opts.projectName ?? 'agentfield'
  const subdomain = existing ? stateSubdomain(state) : `${projectName}-${randomBytes(2).toString('hex')}`
  if (!subdomain) return { ok: false, message: 'Existing deployment is missing its Railway subdomain.' }
  const cliConfig = writeConfig(opts.workspaceDir, opts.binaryDir)
  // Newest release when the lookup succeeds; the deployment's recorded pin
  // when it doesn't (so an outage never rewrites a working image and forces
  // a pointless redeploy); the floating tag only for a brand-new deployment.
  const image = (await resolveCloudImage(deps.fetchImpl ?? fetch)) ?? stateSourceImage(state) ?? CLOUD_IMAGE_LATEST
  opts.onLine?.(`Deploying ${image}`)
  const env = deployEnv(opts, apiKey, subdomain, cliConfig, deps, image)

  if (!providerReady(opts.workspaceDir)) {
    const init = await runCommand(binary, ['init', '-input=false'], opts.workspaceDir, env, deps, false)
    if (init.code !== 0) return { ok: false, message: init.lastError ?? 'OpenTofu initialization failed.' }
  }
  const apply = await runCommand(binary, ['apply', '-auto-approve', '-input=false', '-json'], opts.workspaceDir, env, deps, true, opts.onLine)
  if (apply.code !== 0 || apply.lastError) {
    return { ok: false, message: `${apply.lastError ?? 'Deployment failed'}. State was kept; re-run deploy to reconcile.` }
  }
  const output = await runCommand(binary, ['output', '-json'], opts.workspaceDir, env, deps, false)
  if (output.code !== 0) return { ok: false, message: output.lastError ?? 'Could not read deployment outputs.' }
  try {
    const values = JSON.parse(output.stdout) as Record<string, { value?: unknown }>
    const url = values.url?.value
    const outputKey = values.api_key?.value
    const projectId = values.project_id?.value
    const environmentId = values.environment_id?.value
    const serviceId = values.service_id?.value
    const furrowDomain = values.furrow_domain?.value
    const furrowPort = values.furrow_port?.value
    if (typeof url !== 'string' || !url || typeof outputKey !== 'string' || !outputKey ||
        typeof projectId !== 'string' || !projectId || typeof environmentId !== 'string' || !environmentId ||
        typeof serviceId !== 'string' || !serviceId) throw new Error('missing')
    // Workspace sync is an extra, so its outputs are read separately and never
    // gate the deploy: a control plane that is up and reachable is a success
    // even when no furrow address came back with it.
    const furrowAddress =
      typeof furrowDomain === 'string' && furrowDomain &&
      typeof furrowPort === 'number' && Number.isInteger(furrowPort) && furrowPort > 0
        ? `${furrowDomain}:${furrowPort}`
        : undefined
    opts.onLine?.('Attaching storage volume…')
    try {
      await ensureVolume(opts.railwayToken, projectId, environmentId, serviceId, deps.fetchImpl ?? fetch)
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      return { ok: false, message: `Deployed, but attaching the storage volume failed: ${detail}. Re-run deploy to retry.` }
    }
    opts.onLine?.('Storage volume ready')
    return { ok: true, url, apiKey: outputKey, furrowAddress, message: 'AgentField deployed to Railway.' }
  } catch {
    return { ok: false, message: 'Deployment completed, but required outputs are missing.' }
  }
}

export async function runDestroy(opts: DeployEngineOptions, deps: DeploySpawnDeps = {}): Promise<{ ok: boolean; message: string }> {
  // Destroying the Railway project cascades the GraphQL-created volume server-side.
  const binary = resolveTofuBinary(opts.binaryDir)
  if (!binary) return { ok: false, message: NO_ENGINE }
  const state = readState(opts.workspaceDir)
  const apiKey = stateOutput(state, 'api_key') ?? generateApiKey()
  const subdomain = stateSubdomain(state) ?? (opts.projectName ?? 'agentfield')
  // The provider validates workspace_id (UUID) even on destroy, and callers
  // tearing down don't re-prompt for a workspace — the project's own state
  // records which workspace it was created in.
  const workspaceId = stateWorkspaceId(state) ?? opts.workspaceId
  if (!workspaceId) {
    return { ok: false, message: 'Could not determine the Railway workspace of this deployment — sign in and re-run deploy first.' }
  }
  const cliConfig = writeConfig(opts.workspaceDir, opts.binaryDir)
  const result = await runCommand(binary, ['destroy', '-auto-approve', '-input=false', '-json'], opts.workspaceDir, deployEnv({ ...opts, workspaceId }, apiKey, subdomain, cliConfig, deps), deps, true, opts.onLine)
  if (result.code !== 0 || result.lastError) return { ok: false, message: result.lastError ?? 'Destroy failed.' }
  return { ok: true, message: 'Railway deployment destroyed.' }
}
