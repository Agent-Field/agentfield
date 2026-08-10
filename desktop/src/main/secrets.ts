import type {
  AgentActionResult,
  AgentEnvReport,
  AgentEnvVar,
  EnvVarStatus,
  SecretsListResult,
  StoredSecret
} from '../shared/types'
import { CpApiError, createCpClient, isInstalledPackage, type CpClient } from './cpClient'

const GLOBAL_SCOPE = 'global'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

export interface EnvSpecVar {
  name: string
  description: string
  secret: boolean
  scope: 'global' | 'node'
  default: string
  validation: string
}

export interface EnvSpec {
  required: EnvSpecVar[]
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
  return Array.isArray(raw)
    ? raw.map(toSpecVar).filter((value): value is EnvSpecVar => value !== null)
    : []
}

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
  return { required: toSpecVars(env.required), groups, optional: toSpecVars(env.optional) }
}

export function specIsEmpty(spec: EnvSpec): boolean {
  return spec.required.length === 0 && spec.groups.length === 0 && spec.optional.length === 0
}

export function findSpecVar(spec: EnvSpec, name: string): EnvSpecVar | null {
  for (const variable of spec.required) if (variable.name === name) return variable
  for (const group of spec.groups) {
    for (const variable of group.options) if (variable.name === name) return variable
  }
  for (const variable of spec.optional) if (variable.name === name) return variable
  return null
}

export interface SecretRef {
  key: string
  scope: string
}

export function parseSecretsTable(output: string): SecretRef[] {
  const refs: SecretRef[] = []
  for (const line of output.split(/\r?\n/)) {
    if (!line.includes('│')) continue
    const cells = line.split('│').map((cell) => cell.trim()).filter(Boolean)
    if (cells.length >= 2 && cells[0] !== 'KEY') refs.push({ key: cells[0], scope: cells[1] })
  }
  return refs
}

function reportVariable(
  variable: EnvSpecVar,
  required: boolean,
  group: EnvSpec['groups'][number] | null,
  agent: string,
  refs: SecretRef[],
  env: Record<string, string | undefined>
): AgentEnvVar {
  const storedScopes = refs
    .filter((ref) =>
      ref.key === variable.name && (ref.scope === GLOBAL_SCOPE || ref.scope === agent)
    )
    .map((ref) => ref.scope)
  const status: EnvVarStatus =
    env[variable.name]
      ? 'env'
      : storedScopes.length > 0
        ? 'stored'
        : variable.default
          ? 'default'
          : 'missing'
  return {
    name: variable.name,
    description: variable.description,
    secret: variable.secret,
    scope: variable.scope,
    required,
    group: group?.id || undefined,
    groupDescription: group?.description || undefined,
    status,
    storedScopes
  }
}

export function buildEnvReport(
  agent: string,
  spec: EnvSpec,
  refs: SecretRef[],
  env: Record<string, string | undefined> = process.env
): AgentEnvReport {
  const vars: AgentEnvVar[] = []
  for (const variable of spec.required) vars.push(reportVariable(variable, true, null, agent, refs, env))
  for (const group of spec.groups) {
    for (const variable of group.options) vars.push(reportVariable(variable, true, group, agent, refs, env))
  }
  for (const variable of spec.optional) vars.push(reportVariable(variable, false, null, agent, refs, env))
  const requiredOk = vars.every((variable) =>
    variable.group || !variable.required || variable.status !== 'missing'
  )
  const groupsOk = spec.groups.every((group) =>
    vars.some((variable) => variable.group === group.id && variable.status !== 'missing')
  )
  return { agent, vars, satisfied: requiredOk && groupsOk }
}

function declaringAgents(
  specs: { agent: string; spec: EnvSpec }[],
  key: string
): string[] {
  return specs.filter(({ spec }) => findSpecVar(spec, key) !== null).map(({ agent }) => agent)
}

export function buildSecretsInventory(
  refs: SecretRef[],
  specs: { agent: string; spec: EnvSpec }[]
): StoredSecret[] {
  return [...refs]
    .sort((a, b) =>
      Number(a.scope !== GLOBAL_SCOPE) - Number(b.scope !== GLOBAL_SCOPE) ||
      a.scope.localeCompare(b.scope) ||
      a.key.localeCompare(b.key)
    )
    .map((ref) => ({
      key: ref.key,
      scope: ref.scope,
      usedBy:
        ref.scope === GLOBAL_SCOPE
          ? declaringAgents(specs, ref.key)
          : declaringAgents(specs.filter(({ agent }) => agent === ref.scope), ref.key)
    }))
}

export interface SecretsDeps {
  cpClient: CpClient
}

function defaultDeps(): SecretsDeps {
  return { cpClient: createCpClient() }
}

function errorMessage(err: unknown): string {
  return err instanceof CpApiError
    ? err.message
    : 'Could not reach the control plane — start the control plane and try again'
}

export async function getEnvReports(
  deps: SecretsDeps = defaultDeps()
): Promise<AgentEnvReport[]> {
  try {
    const packages = await deps.cpClient.listPackages()
    return await Promise.all(packages.packages.filter(isInstalledPackage).map(async (pkg) => {
      const { secrets } = await deps.cpClient.listAgentSecrets(pkg.id)
      const hasEnvMetadata = secrets.some((secret) => Boolean(secret.requirement))
      const vars: AgentEnvVar[] = secrets.map((secret) => {
        if (!hasEnvMetadata) {
          return {
            name: secret.key,
            description: '',
            secret: true,
            scope: secret.scope ?? GLOBAL_SCOPE,
            required: true,
            status: secret.is_set ? 'stored' : 'missing',
            storedScopes: secret.is_set
              ? [secret.scope === 'node' ? pkg.name : secret.scope ?? GLOBAL_SCOPE]
              : []
          }
        }

        const status: EnvVarStatus = secret.is_set
          ? 'stored'
          : secret.default
            ? 'default'
            : 'missing'
        return {
          name: secret.key,
          description: secret.description ?? '',
          secret: secret.secret ?? false,
          scope: secret.declared_scope ?? GLOBAL_SCOPE,
          // Group members are `required: true` by app convention (see
          // AgentEnvVar.required) — EnvEditor's Optional bucket filters on
          // !required, and the gate below exempts grouped vars instead.
          required: secret.requirement === 'required' || secret.requirement === 'one_of',
          group: secret.requirement === 'one_of' ? secret.group || undefined : undefined,
          groupDescription:
            secret.requirement === 'one_of' ? secret.group_description || undefined : undefined,
          status,
          storedScopes: secret.is_set
            ? [secret.scope === 'node' ? pkg.name : secret.scope ?? GLOBAL_SCOPE]
            : []
        }
      })
      const groups = new Set(vars.flatMap((variable) => variable.group ? [variable.group] : []))
      const requiredOk = vars.every((variable) =>
        variable.group || !variable.required || variable.status !== 'missing'
      )
      const groupsOk = [...groups].every((group) =>
        vars.some((variable) => variable.group === group && variable.status !== 'missing')
      )
      // Without metadata, a require_one_of group member is indistinguishable
      // from a hard-required key, so an unset alternative (e.g. one of two
      // LLM-provider keys) must not veto Start. Report the keys as declared,
      // but leave the verdict to the control plane's start-time resolution.
      return {
        agent: pkg.name,
        vars,
        satisfied: hasEnvMetadata ? requiredOk && groupsOk : true
      }
    }))
  } catch (err) {
    return [{ agent: '', vars: [], satisfied: false, error: errorMessage(err) }]
  }
}

export async function setAgentSecret(
  agent: string,
  key: string,
  value: string,
  deps: SecretsDeps = defaultDeps()
): Promise<AgentActionResult> {
  const trimmed = value.trim()
  if (!trimmed) return { ok: false, message: 'value must not be empty' }
  try {
    await deps.cpClient.setAgentSecret(agent, key, trimmed)
    return { ok: true, message: `${key} stored` }
  } catch (err) {
    return { ok: false, message: errorMessage(err) }
  }
}

export async function listStoredSecrets(
  deps: SecretsDeps = defaultDeps()
): Promise<SecretsListResult> {
  try {
    const [{ secrets }, packages] = await Promise.all([
      deps.cpClient.listAllSecrets(),
      deps.cpClient.listPackages()
    ])
    const declarations = await Promise.all(packages.packages.filter(isInstalledPackage).map(async (pkg) => ({
      id: pkg.id,
      name: pkg.name,
      keys: new Set((await deps.cpClient.listAgentSecrets(pkg.id)).secrets.map((secret) => secret.key))
    })))
    return {
      secrets: secrets.map((secret) => ({
        ...secret,
        usedBy: declarations
          .filter(({ id, name, keys }) =>
            keys.has(secret.key) &&
            (secret.scope === GLOBAL_SCOPE || secret.scope === id || secret.scope === name)
          )
          .map(({ name }) => name)
      }))
    }
  } catch (err) {
    return { secrets: [], error: errorMessage(err) }
  }
}

export async function revokeStoredSecret(
  key: string,
  scope: string,
  deps: SecretsDeps = defaultDeps()
): Promise<AgentActionResult> {
  try {
    const { secrets } = await deps.cpClient.listAllSecrets()
    if (!secrets.some((secret) => secret.key === key && secret.scope === scope)) {
      return { ok: false, message: `${key} is not stored in the ${scope} scope` }
    }
    let agent = scope
    if (scope === GLOBAL_SCOPE) {
      const packages = await deps.cpClient.listPackages()
      agent = ''
      for (const pkg of packages.packages.filter(isInstalledPackage)) {
        const declared = await deps.cpClient.listAgentSecrets(pkg.id)
        if (declared.secrets.some((secret) => secret.key === key && secret.scope === GLOBAL_SCOPE)) {
          agent = pkg.id
          break
        }
      }
      if (!agent) return { ok: false, message: `no installed agent can remove ${key}` }
    }
    await deps.cpClient.deleteAgentSecret(agent, key)
    return {
      ok: true,
      message: scope === GLOBAL_SCOPE ? `${key} removed for all agents` : `${key} removed for ${scope}`
    }
  } catch (err) {
    return { ok: false, message: errorMessage(err) }
  }
}

export async function revokeAgentSecret(
  agent: string,
  key: string,
  deps: SecretsDeps = defaultDeps()
): Promise<AgentActionResult> {
  try {
    const { secrets } = await deps.cpClient.listAgentSecrets(agent)
    if (!secrets.find((secret) => secret.key === key)?.is_set) {
      return { ok: true, message: `${key} is not stored` }
    }
    await deps.cpClient.deleteAgentSecret(agent, key)
    return { ok: true, message: `${key} revoked` }
  } catch (err) {
    return { ok: false, message: errorMessage(err) }
  }
}
