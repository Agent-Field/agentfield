// Agent secrets seam: reads each installed node's user_environment spec from
// its agentfield-package.yaml and drives the af CLI's encrypted secret store
// (`af secrets set / ls / rm`) — the exact store `af run` decrypts into the
// agent's process environment. The app never touches the .enc files itself,
// and secret VALUES never cross the IPC boundary: the renderer only ever
// sees per-variable status flags, and values travel renderer → main → af's
// stdin (never argv, which other processes could observe).
//
// Resolution mirrors the CLI's EnvResolver (control-plane/internal/packages/
// env_resolver.go): process env → secret store (node scope, then global) →
// manifest default → missing. `af run` from this app inherits our process
// env, so checking process.env here is faithful to what the spawned agent
// will see.
//
// No electron imports — unit-testable under plain vitest.

import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'
import type {
  AgentActionResult,
  AgentEnvReport,
  AgentEnvVar,
  EnvVarStatus,
  SecretsListResult,
  StoredSecret
} from '../shared/types'
import { getAgentFieldHome, readInstalledAgents } from './agentfield'
import { getCliCommand } from './cli'
import { childEnv } from './env'
import { sanitizeInstallOutput } from './installer'

const SECRETS_TIMEOUT_MS = 15_000

/** The store's shared scope name (control-plane/internal/packages/secrets.go). */
const GLOBAL_SCOPE = 'global'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/** One variable from a manifest's user_environment section. */
export interface EnvSpecVar {
  name: string
  description: string
  /** type: secret — hidden input, masked display. */
  secret: boolean
  /** Store scope `af run` reads and `af secrets set` writes: global unless scope: node. */
  scope: 'global' | 'node'
  /** Manifest default — counts as resolved, matching the EnvResolver. */
  default: string
  /** Optional Go/RE2 regex the value must match (validated on prompt in the CLI). */
  validation: string
}

/** Parsed user_environment block of an agentfield-package.yaml. */
export interface EnvSpec {
  required: EnvSpecVar[]
  /** require_one_of groups: any one resolving option satisfies the group. */
  groups: { id: string; description: string; options: EnvSpecVar[] }[]
  optional: EnvSpecVar[]
}

function toSpecVar(raw: unknown): EnvSpecVar | null {
  if (!isRecord(raw) || typeof raw.name !== 'string' || raw.name === '') return null
  return {
    name: raw.name,
    description: str(raw.description),
    secret: raw.type === 'secret',
    scope: raw.scope === 'node' ? 'node' : 'global',
    default: str(raw.default),
    validation: str(raw.validation)
  }
}

function toSpecVars(raw: unknown): EnvSpecVar[] {
  if (!Array.isArray(raw)) return []
  return raw.map(toSpecVar).filter((v): v is EnvSpecVar => v !== null)
}

/**
 * Extract the user_environment section from a yaml-loaded manifest document.
 * Field shapes follow UserEnvironmentConfig in control-plane/internal/packages/
 * installer.go. Absent or malformed sections yield an empty spec — an agent
 * without declared variables simply has nothing to configure.
 */
export function parseUserEnvironment(doc: unknown): EnvSpec {
  const env = isRecord(doc) && isRecord(doc.user_environment) ? doc.user_environment : {}
  const groups = Array.isArray(env.require_one_of)
    ? env.require_one_of
        .filter(isRecord)
        .map((group) => ({
          id: str(group.id),
          description: str(group.description),
          options: toSpecVars(group.options)
        }))
        .filter((group) => group.options.length > 0)
    : []
  return {
    required: toSpecVars(env.required),
    groups,
    optional: toSpecVars(env.optional)
  }
}

export function specIsEmpty(spec: EnvSpec): boolean {
  return spec.required.length === 0 && spec.groups.length === 0 && spec.optional.length === 0
}

/** Find a declared variable by name anywhere in the spec. */
export function findSpecVar(spec: EnvSpec, name: string): EnvSpecVar | null {
  for (const v of spec.required) if (v.name === name) return v
  for (const group of spec.groups) for (const v of group.options) if (v.name === name) return v
  for (const v of spec.optional) if (v.name === name) return v
  return null
}

/** One row of `af secrets ls`: a stored key and its scope (global or a node name). */
export interface SecretRef {
  key: string
  scope: string
}

/**
 * Parse the `af secrets ls` table (internal/ui.Table — lipgloss rounded
 * borders, no ANSI when piped). Data rows look like:
 *   │ OPENAI_API_KEY │ global │ •••••••• │
 * Everything that isn't a ≥2-cell row (borders, the header row, the empty
 * "No secrets stored" panel) is ignored.
 */
export function parseSecretsTable(output: string): SecretRef[] {
  const refs: SecretRef[] = []
  for (const line of output.split(/\r?\n/)) {
    if (!line.includes('│')) continue
    const cells = line
      .split('│')
      .map((cell) => cell.trim())
      .filter((cell) => cell.length > 0)
    if (cells.length < 2) continue
    if (cells[0] === 'KEY') continue
    refs.push({ key: cells[0], scope: cells[1] })
  }
  return refs
}

/** Store scopes relevant to this agent that currently hold the key. */
function storedScopesFor(refs: SecretRef[], agent: string, key: string): string[] {
  return refs
    .filter((ref) => ref.key === key && (ref.scope === GLOBAL_SCOPE || ref.scope === agent))
    .map((ref) => ref.scope)
}

function statusFor(
  v: EnvSpecVar,
  storedScopes: string[],
  env: Record<string, string | undefined>
): EnvVarStatus {
  const fromEnv = env[v.name]
  if (typeof fromEnv === 'string' && fromEnv !== '') return 'env'
  if (storedScopes.length > 0) return 'stored'
  if (v.default !== '') return 'default'
  return 'missing'
}

function toReportVar(
  v: EnvSpecVar,
  required: boolean,
  group: { id: string; description: string } | null,
  agent: string,
  refs: SecretRef[],
  env: Record<string, string | undefined>
): AgentEnvVar {
  const storedScopes = storedScopesFor(refs, agent, v.name)
  return {
    name: v.name,
    description: v.description,
    secret: v.secret,
    scope: v.scope,
    required,
    group: group?.id || undefined,
    groupDescription: group?.description || undefined,
    status: statusFor(v, storedScopes, env),
    storedScopes
  }
}

/**
 * Assemble the renderer-facing report for one agent. `satisfied` mirrors what
 * the CLI's EnvResolver will conclude headlessly: every required variable and
 * every require_one_of group resolves from env / store / default — so `af run`
 * will not fail with "missing required environment variables".
 */
export function buildEnvReport(
  agent: string,
  spec: EnvSpec,
  refs: SecretRef[],
  env: Record<string, string | undefined> = process.env
): AgentEnvReport {
  const vars: AgentEnvVar[] = []
  for (const v of spec.required) vars.push(toReportVar(v, true, null, agent, refs, env))
  for (const group of spec.groups) {
    for (const v of group.options) vars.push(toReportVar(v, true, group, agent, refs, env))
  }
  for (const v of spec.optional) vars.push(toReportVar(v, false, null, agent, refs, env))

  const requiredOk = vars.every((v) => v.group || !v.required || v.status !== 'missing')
  const groupsOk = spec.groups.every((group) =>
    vars.some((v) => v.group === group.id && v.status !== 'missing')
  )
  return { agent, vars, satisfied: requiredOk && groupsOk }
}

/** Run one `af secrets …` verb, capturing all sanitized output. */
function runSecretsCli(
  args: string[],
  input?: string
): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    const lines: string[] = []
    let settled = false
    const done = (ok: boolean) => {
      if (!settled) {
        settled = true
        resolve({ ok, output: lines.join('\n') })
      }
    }

    const child = spawn(getCliCommand(), ['secrets', ...args], {
      windowsHide: true,
      env: childEnv()
    })
    const timer = setTimeout(() => {
      child.kill()
      lines.push('af secrets timed out')
      done(false)
    }, SECRETS_TIMEOUT_MS)

    const collect = (chunk: Buffer) => {
      lines.push(...sanitizeInstallOutput(chunk.toString('utf8')))
    }
    child.stdout.on('data', collect)
    child.stderr.on('data', collect)
    child.on('error', (err: NodeJS.ErrnoException) => {
      clearTimeout(timer)
      lines.push(
        err.code === 'ENOENT' ? 'The AgentField CLI (af) was not found' : String(err.message)
      )
      done(false)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code === 0)
    })
    if (input !== undefined) {
      // `af secrets set KEY` reads the value from stdin when it is not a TTY
      // (readHiddenValue in internal/cli/secrets.go) — the value never
      // appears in a command line.
      child.stdin.write(`${input}\n`)
    }
    child.stdin.end()
  })
}

/** Read and parse <packageDir>/agentfield-package.yaml; null when unreadable. */
export async function readEnvSpec(packageDir: string): Promise<EnvSpec | null> {
  try {
    const text = await fs.readFile(path.join(packageDir, 'agentfield-package.yaml'), 'utf8')
    return parseUserEnvironment(yaml.load(text))
  } catch {
    return null
  }
}

/**
 * Build env reports for every installed agent that declares environment
 * variables. One `af secrets ls` serves all agents. A store read failure
 * degrades to "store looks empty" with the error attached — statuses from
 * process env and manifest defaults are still real.
 */
export async function getEnvReports(
  homeDir: string = getAgentFieldHome()
): Promise<AgentEnvReport[]> {
  const registry = await readInstalledAgents(homeDir)
  if (registry.agents.length === 0) return []

  const specs: { agent: string; spec: EnvSpec }[] = []
  for (const agent of registry.agents) {
    const dir = agent.path ?? path.join(homeDir, 'packages', agent.name)
    const spec = await readEnvSpec(dir)
    if (spec && !specIsEmpty(spec)) specs.push({ agent: agent.name, spec })
  }
  if (specs.length === 0) return []

  const ls = await runSecretsCli(['ls'])
  const refs = ls.ok ? parseSecretsTable(ls.output) : []
  return specs.map(({ agent, spec }) => {
    const report = buildEnvReport(agent, spec, refs)
    if (!ls.ok) report.error = `could not read the secret store: ${ls.output}`
    return report
  })
}

/** Locate an agent's declared variable; refuse anything the manifest doesn't name. */
async function resolveDeclaredVar(
  agent: string,
  key: string,
  homeDir: string
): Promise<{ spec: EnvSpecVar } | { error: string }> {
  const registry = await readInstalledAgents(homeDir)
  const entry = registry.agents.find((a) => a.name === agent)
  if (!entry) return { error: `"${agent}" is not an installed agent` }
  const spec = await readEnvSpec(entry.path ?? path.join(homeDir, 'packages', agent))
  if (!spec) return { error: `could not read ${agent}'s manifest` }
  const specVar = findSpecVar(spec, key)
  if (!specVar) return { error: `${agent} does not declare ${key}` }
  return { spec: specVar }
}

/**
 * Store a value for one of an agent's declared variables, in the scope the
 * manifest names (global by default — shared across nodes, exactly like the
 * CLI's own prompt-and-store). The renderer only ever supplies names the
 * manifest declares; anything else is refused.
 */
export async function setAgentSecret(
  agent: string,
  key: string,
  value: string,
  homeDir: string = getAgentFieldHome()
): Promise<AgentActionResult> {
  const trimmed = value.trim()
  if (trimmed === '') return { ok: false, message: 'value must not be empty' }

  const resolved = await resolveDeclaredVar(agent, key, homeDir)
  if ('error' in resolved) return { ok: false, message: resolved.error }

  if (resolved.spec.validation) {
    try {
      if (!new RegExp(resolved.spec.validation).test(trimmed)) {
        return {
          ok: false,
          message: `value does not match the required format (${resolved.spec.validation})`
        }
      }
    } catch {
      // Go RE2 pattern that JS cannot compile — let the CLI-side validation
      // (which runs on prompt) be the judge rather than rejecting here.
    }
  }

  const args = ['set', key]
  if (resolved.spec.scope === 'node') args.push('--node', agent)
  const result = await runSecretsCli(args, trimmed)
  return result.ok
    ? { ok: true, message: `${key} stored` }
    : { ok: false, message: result.output || `failed to store ${key}` }
}

/** Names of installed agents whose spec declares the given variable name. */
function agentsDeclaring(
  specs: { agent: string; spec: EnvSpec }[],
  key: string
): string[] {
  return specs.filter(({ spec }) => findSpecVar(spec, key) !== null).map(({ agent }) => agent)
}

/**
 * Join the raw store listing with the installed agents' manifests: which
 * agents can actually read each stored secret. Global secrets are readable
 * by every agent declaring the name; a node-scoped secret only by that node.
 * Ordered global-first, then node scopes, keys alphabetical within a scope.
 */
export function buildSecretsInventory(
  refs: SecretRef[],
  specs: { agent: string; spec: EnvSpec }[]
): StoredSecret[] {
  const rank = (ref: SecretRef) => (ref.scope === GLOBAL_SCOPE ? 0 : 1)
  return [...refs]
    .sort(
      (a, b) =>
        rank(a) - rank(b) || a.scope.localeCompare(b.scope) || a.key.localeCompare(b.key)
    )
    .map((ref) => ({
      key: ref.key,
      scope: ref.scope,
      usedBy:
        ref.scope === GLOBAL_SCOPE
          ? agentsDeclaring(specs, ref.key)
          : agentsDeclaring(
              specs.filter(({ agent }) => agent === ref.scope),
              ref.key
            )
    }))
}

/** The whole secret store (keys and scopes only — never values). */
export async function listStoredSecrets(
  homeDir: string = getAgentFieldHome()
): Promise<SecretsListResult> {
  const ls = await runSecretsCli(['ls'])
  if (!ls.ok) {
    return { secrets: [], error: `could not read the secret store: ${ls.output}` }
  }

  const registry = await readInstalledAgents(homeDir)
  const specs: { agent: string; spec: EnvSpec }[] = []
  for (const agent of registry.agents) {
    const dir = agent.path ?? path.join(homeDir, 'packages', agent.name)
    const spec = await readEnvSpec(dir)
    if (spec && !specIsEmpty(spec)) specs.push({ agent: agent.name, spec })
  }
  return { secrets: buildSecretsInventory(parseSecretsTable(ls.output), specs) }
}

/**
 * Remove one stored secret from one scope. Only (key, scope) pairs that the
 * store actually lists are accepted — the renderer never gets to invent
 * arguments for `af secrets rm`.
 */
export async function revokeStoredSecret(
  key: string,
  scope: string
): Promise<AgentActionResult> {
  const ls = await runSecretsCli(['ls'])
  if (!ls.ok) return { ok: false, message: `could not read the secret store: ${ls.output}` }
  const exists = parseSecretsTable(ls.output).some(
    (ref) => ref.key === key && ref.scope === scope
  )
  if (!exists) return { ok: false, message: `${key} is not stored in the ${scope} scope` }

  const args = ['rm', key]
  if (scope !== GLOBAL_SCOPE) args.push('--node', scope)
  const result = await runSecretsCli(args)
  if (!result.ok) return { ok: false, message: result.output || `failed to remove ${key}` }
  return {
    ok: true,
    message:
      scope === GLOBAL_SCOPE ? `${key} removed for all agents` : `${key} removed for ${scope}`
  }
}

/**
 * Remove a stored value from every scope relevant to this agent that holds
 * it (its node scope and/or global). Revoking a global key affects all
 * agents sharing it — that is the store's sharing model, and the UI labels
 * the scope so the choice is informed.
 */
export async function revokeAgentSecret(
  agent: string,
  key: string,
  homeDir: string = getAgentFieldHome()
): Promise<AgentActionResult> {
  const resolved = await resolveDeclaredVar(agent, key, homeDir)
  if ('error' in resolved) return { ok: false, message: resolved.error }

  const ls = await runSecretsCli(['ls'])
  if (!ls.ok) return { ok: false, message: `could not read the secret store: ${ls.output}` }
  const scopes = storedScopesFor(parseSecretsTable(ls.output), agent, key)
  if (scopes.length === 0) return { ok: true, message: `${key} is not stored` }

  for (const scope of scopes) {
    const args = ['rm', key]
    if (scope !== GLOBAL_SCOPE) args.push('--node', scope)
    const result = await runSecretsCli(args)
    if (!result.ok) {
      return { ok: false, message: result.output || `failed to remove ${key}` }
    }
  }
  return { ok: true, message: `${key} revoked` }
}
