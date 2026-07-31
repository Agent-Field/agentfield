import yaml from 'js-yaml'
import { describe, expect, it, vi } from 'vitest'
import type { AgentSecretStatus, CpClient, PackageInfo } from './cpClient'
import {
  buildEnvReport,
  buildSecretsInventory,
  findSpecVar,
  listStoredSecrets,
  parseSecretsTable,
  parseUserEnvironment,
  revokeStoredSecret,
  specIsEmpty
} from './secrets'
import { getEnvReports, revokeAgentSecret, setAgentSecret } from './secrets'

// Abridged from SWE-AF's real agentfield-package.yaml — the exact shapes
// ParsePackageMetadata reads (control-plane/internal/packages/installer.go).
const SWE_MANIFEST = `
config_version: v1
name: swe-planner
version: 0.1.0
user_environment:
  require_one_of:
    - id: llm_provider
      description: an LLM provider key
      options:
        - name: ANTHROPIC_API_KEY
          description: Anthropic API key (Claude)
          type: secret
          scope: global
        - name: OPENROUTER_API_KEY
          description: OpenRouter API key
          type: secret
          scope: global
  required:
    - name: GH_TOKEN
      description: GitHub token (repo scope)
      type: secret
      scope: global
  optional:
    - name: SWE_DEFAULT_MODEL
      description: Override the model id
    - name: AGENTFIELD_SERVER
      description: Control-plane URL
      default: http://localhost:8080
`

// Captured from a real piped `af secrets ls` (lipgloss rounded borders,
// no ANSI when stdout is not a TTY).
const LS_OUTPUT = `Stored secrets (2)
╭───────────────────┬─────────────┬──────────╮
│ KEY               │ SCOPE       │ VALUE    │
├───────────────────┼─────────────┼──────────┤
│ GH_TOKEN          │ global      │ •••••••• │
│ ANTHROPIC_API_KEY │ swe-planner │ •••••••• │
╰───────────────────┴─────────────┴──────────╯`

const LS_EMPTY = `No secrets stored
╭──────────────────────╮
│ Add one with:        │
│   af secrets set KEY │
╰──────────────────────╯`

function sweSpec() {
  return parseUserEnvironment(yaml.load(SWE_MANIFEST))
}

describe('parseUserEnvironment', () => {
  it('reads required, require_one_of, and optional sections', () => {
    const spec = sweSpec()
    expect(spec.required.map((v) => v.name)).toEqual(['GH_TOKEN'])
    expect(spec.groups).toHaveLength(1)
    expect(spec.groups[0].id).toBe('llm_provider')
    expect(spec.groups[0].options.map((v) => v.name)).toEqual([
      'ANTHROPIC_API_KEY',
      'OPENROUTER_API_KEY'
    ])
    expect(spec.optional.map((v) => v.name)).toEqual(['SWE_DEFAULT_MODEL', 'AGENTFIELD_SERVER'])
  })

  it('maps type/scope/default per var', () => {
    const spec = sweSpec()
    const gh = findSpecVar(spec, 'GH_TOKEN')
    expect(gh).toMatchObject({ secret: true, scope: 'global', default: '' })
    const server = findSpecVar(spec, 'AGENTFIELD_SERVER')
    expect(server).toMatchObject({ secret: false, default: 'http://localhost:8080' })
  })

  it('yields an empty spec for manifests without user_environment', () => {
    expect(specIsEmpty(parseUserEnvironment(yaml.load('name: bare-agent')))).toBe(true)
    expect(specIsEmpty(parseUserEnvironment(null))).toBe(true)
    expect(specIsEmpty(parseUserEnvironment({ user_environment: 'garbage' }))).toBe(true)
  })

  it('drops malformed vars and empty groups instead of throwing', () => {
    const spec = parseUserEnvironment({
      user_environment: {
        required: [{ name: 'GOOD' }, { description: 'no name' }, 42],
        require_one_of: [{ id: 'empty', options: [] }, { id: 'ok', options: [{ name: 'A' }] }]
      }
    })
    expect(spec.required.map((v) => v.name)).toEqual(['GOOD'])
    expect(spec.groups.map((g) => g.id)).toEqual(['ok'])
  })
})

describe('parseSecretsTable', () => {
  it('extracts key/scope rows from the bordered table', () => {
    expect(parseSecretsTable(LS_OUTPUT)).toEqual([
      { key: 'GH_TOKEN', scope: 'global' },
      { key: 'ANTHROPIC_API_KEY', scope: 'swe-planner' }
    ])
  })

  it('reads nothing from the empty-store panel', () => {
    expect(parseSecretsTable(LS_EMPTY)).toEqual([])
  })

  it('ignores plain text output', () => {
    expect(parseSecretsTable('some error\nno table here')).toEqual([])
  })
})

describe('buildEnvReport', () => {
  it('is unsatisfied when required and group vars are all missing', () => {
    const report = buildEnvReport('swe-planner', sweSpec(), [], {})
    expect(report.satisfied).toBe(false)
    const byName = Object.fromEntries(report.vars.map((v) => [v.name, v]))
    expect(byName.GH_TOKEN.status).toBe('missing')
    expect(byName.ANTHROPIC_API_KEY.status).toBe('missing')
    // Optional var with a manifest default resolves without any input.
    expect(byName.AGENTFIELD_SERVER.status).toBe('default')
  })

  it('is satisfied once the required var and one group option resolve', () => {
    const refs = parseSecretsTable(LS_OUTPUT)
    const report = buildEnvReport('swe-planner', sweSpec(), refs, {})
    expect(report.satisfied).toBe(true)
    const byName = Object.fromEntries(report.vars.map((v) => [v.name, v]))
    expect(byName.GH_TOKEN).toMatchObject({ status: 'stored', storedScopes: ['global'] })
    expect(byName.ANTHROPIC_API_KEY).toMatchObject({
      status: 'stored',
      storedScopes: ['swe-planner']
    })
    expect(byName.OPENROUTER_API_KEY.status).toBe('missing')
  })

  it('resolves from the process environment first', () => {
    const report = buildEnvReport('swe-planner', sweSpec(), [], {
      GH_TOKEN: 'ghp_x',
      OPENROUTER_API_KEY: 'sk-or-x'
    })
    expect(report.satisfied).toBe(true)
    const byName = Object.fromEntries(report.vars.map((v) => [v.name, v]))
    expect(byName.GH_TOKEN.status).toBe('env')
    expect(byName.OPENROUTER_API_KEY.status).toBe('env')
  })

  it('ignores another node’s scoped secrets', () => {
    const refs = [{ key: 'GH_TOKEN', scope: 'other-agent' }]
    const report = buildEnvReport('swe-planner', sweSpec(), refs, {})
    const gh = report.vars.find((v) => v.name === 'GH_TOKEN')
    expect(gh?.status).toBe('missing')
    expect(gh?.storedScopes).toEqual([])
  })

  it('requires every group to be satisfied independently', () => {
    const spec = parseUserEnvironment({
      user_environment: {
        require_one_of: [
          { id: 'a', options: [{ name: 'A1' }] },
          { id: 'b', options: [{ name: 'B1' }] }
        ]
      }
    })
    const partial = buildEnvReport('x', spec, [{ key: 'A1', scope: 'global' }], {})
    expect(partial.satisfied).toBe(false)
    const both = buildEnvReport(
      'x',
      spec,
      [
        { key: 'A1', scope: 'global' },
        { key: 'B1', scope: 'global' }
      ],
      {}
    )
    expect(both.satisfied).toBe(true)
  })

  it('an empty env value does not count as resolved', () => {
    const report = buildEnvReport('swe-planner', sweSpec(), [], { GH_TOKEN: '' })
    expect(report.vars.find((v) => v.name === 'GH_TOKEN')?.status).toBe('missing')
  })
})

describe('buildSecretsInventory', () => {
  const specs = [{ agent: 'swe-planner', spec: sweSpec() }]

  it('orders global first and joins in the declaring agents', () => {
    const refs = [
      { key: 'ANTHROPIC_API_KEY', scope: 'swe-planner' },
      { key: 'OPENROUTER_API_KEY', scope: 'global' },
      { key: 'GH_TOKEN', scope: 'global' }
    ]
    const inventory = buildSecretsInventory(refs, specs)
    expect(inventory.map((s) => `${s.scope}:${s.key}`)).toEqual([
      'global:GH_TOKEN',
      'global:OPENROUTER_API_KEY',
      'swe-planner:ANTHROPIC_API_KEY'
    ])
    expect(inventory[0].usedBy).toEqual(['swe-planner'])
    expect(inventory[2].usedBy).toEqual(['swe-planner'])
  })

  it('flags secrets no installed agent declares', () => {
    const inventory = buildSecretsInventory([{ key: 'RANDOM_TOKEN', scope: 'global' }], specs)
    expect(inventory[0].usedBy).toEqual([])
  })

  it('a node-scoped secret is only attributed to its own node', () => {
    const refs = [{ key: 'GH_TOKEN', scope: 'other-agent' }]
    const inventory = buildSecretsInventory(refs, specs)
    expect(inventory[0]).toMatchObject({ scope: 'other-agent', usedBy: [] })
  })
})

describe('control-plane secret management', () => {
  const pkg: PackageInfo = {
    id: 'agent-id', name: 'agent', version: '1', status: 'stopped',
    install_path: '/pkg', configuration_required: true, configuration_complete: false,
    description: '', author: ''
  }

  function client(): CpClient {
    return {
      listPackages: vi.fn(async () => ({ packages: [pkg], total: 1 })),
      listAgentSecrets: vi.fn(async () => ({
        secrets: [
          { key: 'SET_KEY', is_set: true, scope: 'global' as const },
          { key: 'UNSET_KEY', is_set: false }
        ]
      })),
      setAgentSecret: vi.fn(async () => {}),
      deleteAgentSecret: vi.fn(async () => {})
    } as unknown as CpClient
  }

  it('reports declared set and unset keys from the API', async () => {
    const reports = await getEnvReports({ cpClient: client() })
    expect(reports[0].vars.map(({ name, status }) => ({ name, status }))).toEqual([
      { name: 'SET_KEY', status: 'stored' },
      { name: 'UNSET_KEY', status: 'missing' }
    ])
    expect(reports[0].vars.every((variable) => variable.required && variable.secret)).toBe(true)
    // Without metadata a missing key may be an unused one-of alternative, so
    // the report must not veto Start — the control plane decides at start time.
    expect(reports[0].satisfied).toBe(true)
  })

  it('maps enriched environment metadata, groups, defaults, and optionals', async () => {
    const cpClient = client()
    const enrichedSecrets: AgentSecretStatus[] = [
        {
          key: 'ANTHROPIC_API_KEY', is_set: false, scope: 'global',
          declared_scope: 'global', description: 'Anthropic API key (Claude)', secret: true,
          default: '', requirement: 'one_of', group: 'llm_provider',
          group_description: 'an LLM provider key'
        },
        {
          key: 'OPENROUTER_API_KEY', is_set: true, scope: 'global',
          declared_scope: 'global', description: 'OpenRouter API key', secret: true,
          default: '', requirement: 'one_of', group: 'llm_provider',
          group_description: 'an LLM provider key'
        },
        {
          key: 'GH_TOKEN', is_set: false, scope: 'global', declared_scope: 'global',
          description: 'GitHub token', secret: true, default: '', requirement: 'required'
        },
        {
          key: 'SWE_DEFAULT_RUNTIME', is_set: false, declared_scope: 'node',
          description: 'Runtime override', secret: false, default: '', requirement: 'optional'
        },
        {
          key: 'AGENTFIELD_SERVER', is_set: false, description: 'Control-plane URL',
          default: 'http://localhost:8080', requirement: 'optional'
        },
        { key: 'UNDECLARED_KEY', is_set: true, scope: 'node' as const }
      ]
    vi.mocked(cpClient.listAgentSecrets).mockResolvedValue({ secrets: enrichedSecrets })

    const [report] = await getEnvReports({ cpClient })

    expect(report.satisfied).toBe(false)
    expect(report.vars).toEqual([
      expect.objectContaining({
        name: 'ANTHROPIC_API_KEY', required: true, group: 'llm_provider',
        groupDescription: 'an LLM provider key', status: 'missing'
      }),
      expect.objectContaining({
        name: 'OPENROUTER_API_KEY', required: true, group: 'llm_provider',
        status: 'stored', storedScopes: ['global']
      }),
      expect.objectContaining({ name: 'GH_TOKEN', required: true, status: 'missing' }),
      expect.objectContaining({
        name: 'SWE_DEFAULT_RUNTIME', required: false, secret: false, scope: 'node',
        status: 'missing'
      }),
      expect.objectContaining({
        name: 'AGENTFIELD_SERVER', required: false, secret: false, scope: 'global',
        status: 'default'
      }),
      expect.objectContaining({
        name: 'UNDECLARED_KEY', required: false, secret: false, scope: 'global',
        group: undefined, status: 'stored', storedScopes: ['agent']
      })
    ])

    vi.mocked(cpClient.listAgentSecrets).mockResolvedValue({
      secrets: enrichedSecrets.map((secret) =>
        secret.key === 'GH_TOKEN' ? { ...secret, is_set: true } : secret
      )
    })
    expect((await getEnvReports({ cpClient }))[0].satisfied).toBe(true)
  })

  it('sets and deletes without sending a scope', async () => {
    const cpClient = client()
    await setAgentSecret('agent-id', 'SET_KEY', 'value', { cpClient })
    await revokeAgentSecret('agent-id', 'SET_KEY', { cpClient })
    expect(cpClient.setAgentSecret).toHaveBeenCalledWith('agent-id', 'SET_KEY', 'value')
    expect(cpClient.deleteAgentSecret).toHaveBeenCalledWith('agent-id', 'SET_KEY')
  })

  it.each(['', '   \t\n'])('rejects an empty secret value without issuing a PUT', async (value) => {
    const cpClient = client()
    expect(await setAgentSecret('agent-id', 'SET_KEY', value, { cpClient })).toEqual({
      ok: false,
      message: 'value must not be empty'
    })
    expect(cpClient.setAgentSecret).not.toHaveBeenCalled()
  })

  it('short-circuits deletion for a declared but unset key', async () => {
    const cpClient = client()
    expect(await revokeAgentSecret('agent-id', 'UNSET_KEY', { cpClient })).toEqual({
      ok: true,
      message: 'UNSET_KEY is not stored'
    })
    expect(cpClient.deleteAgentSecret).not.toHaveBeenCalled()
  })

  it('excludes catalog rows from environment reports', async () => {
    const catalog = { ...pkg, id: 'catalog-id', name: 'catalog', install_status: 'uninstalled' }
    const cpClient = client()
    vi.mocked(cpClient.listPackages).mockResolvedValue({
      packages: [{ ...pkg, install_status: 'installed' }, catalog],
      total: 2
    })

    const reports = await getEnvReports({ cpClient })

    expect(reports.map(({ agent }) => agent)).toEqual(['agent'])
    expect(cpClient.listAgentSecrets).toHaveBeenCalledTimes(1)
    expect(cpClient.listAgentSecrets).not.toHaveBeenCalledWith('catalog-id')
  })

  it('returns the error sentinel when package listing fails', async () => {
    const cpClient = client()
    vi.mocked(cpClient.listPackages).mockRejectedValue(
      new TypeError('fetch failed')
    )

    expect(await getEnvReports({ cpClient })).toEqual([{
      agent: '',
      vars: [],
      satisfied: false,
      error: 'Could not reach the control plane — start the control plane and try again'
    }])
  })
})

describe('stored secret inventory', () => {
  const basePackage: PackageInfo = {
    id: 'base', name: 'base', version: '1', status: 'stopped',
    install_path: '/pkg', configuration_required: true, configuration_complete: false,
    description: '', author: ''
  }

  function inventoryClient(): CpClient {
    const packages: PackageInfo[] = [
      { ...basePackage, id: 'alpha-id', name: 'alpha', install_status: 'installed' },
      { ...basePackage, id: 'beta-id', name: 'beta', install_status: 'running' },
      { ...basePackage, id: 'gamma-id', name: 'gamma' },
      { ...basePackage, id: 'catalog-id', name: 'catalog', install_status: 'uninstalled' }
    ]
    return {
      listPackages: vi.fn(async () => ({ packages, total: packages.length })),
      listAllSecrets: vi.fn(async () => ({
        secrets: [
          { key: 'GLOBAL_KEY', scope: 'global' },
          { key: 'ID_KEY', scope: 'beta-id' },
          { key: 'NAME_KEY', scope: 'gamma' }
        ]
      })),
      listAgentSecrets: vi.fn(async (agent: string) => ({
        secrets: agent === 'alpha-id'
          ? [{ key: 'GLOBAL_KEY', is_set: true, scope: 'global' as const }]
          : agent === 'beta-id'
            ? [{ key: 'ID_KEY', is_set: true, scope: 'node' as const }]
            : [{ key: 'NAME_KEY', is_set: true, scope: 'node' as const }]
      })),
      deleteAgentSecret: vi.fn(async () => {})
    } as unknown as CpClient
  }

  it('lists only installed declarations and joins global, package-id, and package-name scopes', async () => {
    const cpClient = inventoryClient()

    expect(await listStoredSecrets({ cpClient })).toEqual({
      secrets: [
        { key: 'GLOBAL_KEY', scope: 'global', usedBy: ['alpha'] },
        { key: 'ID_KEY', scope: 'beta-id', usedBy: ['beta'] },
        { key: 'NAME_KEY', scope: 'gamma', usedBy: ['gamma'] }
      ]
    })
    expect(cpClient.listAgentSecrets).toHaveBeenCalledTimes(3)
    expect(cpClient.listAgentSecrets).not.toHaveBeenCalledWith('catalog-id')
  })

  it('returns an empty inventory and error when one declaration lookup fails', async () => {
    const cpClient = inventoryClient()
    vi.mocked(cpClient.listAgentSecrets).mockImplementation(async (agent) => {
      if (agent === 'beta-id') throw new TypeError('fetch failed')
      return { secrets: [] }
    })

    expect(await listStoredSecrets({ cpClient })).toEqual({
      secrets: [],
      error: 'Could not reach the control plane — start the control plane and try again'
    })
  })

  it('reports a missing scoped secret without deleting', async () => {
    const cpClient = inventoryClient()
    vi.mocked(cpClient.listAllSecrets).mockResolvedValue({ secrets: [] })

    expect(await revokeStoredSecret('MISSING', 'alpha', { cpClient })).toEqual({
      ok: false,
      message: 'MISSING is not stored in the alpha scope'
    })
    expect(cpClient.deleteAgentSecret).not.toHaveBeenCalled()
  })

  it('deletes a node-scoped secret through its declaring agent', async () => {
    const cpClient = inventoryClient()
    vi.mocked(cpClient.listAllSecrets).mockResolvedValue({
      secrets: [{ key: 'NODE_KEY', scope: 'alpha-id' }]
    })

    expect(await revokeStoredSecret('NODE_KEY', 'alpha-id', { cpClient })).toEqual({
      ok: true,
      message: 'NODE_KEY removed for alpha-id'
    })
    expect(cpClient.deleteAgentSecret).toHaveBeenCalledWith('alpha-id', 'NODE_KEY')
  })

  it('scans installed agents and deletes a global secret through a capable one', async () => {
    const cpClient = inventoryClient()
    vi.mocked(cpClient.listAllSecrets).mockResolvedValue({
      secrets: [{ key: 'GLOBAL_KEY', scope: 'global' }]
    })
    vi.mocked(cpClient.listAgentSecrets).mockImplementation(async (agent) => ({
      secrets: agent === 'beta-id'
        ? [{ key: 'GLOBAL_KEY', is_set: true, scope: 'global' as const }]
        : []
    }))

    expect(await revokeStoredSecret('GLOBAL_KEY', 'global', { cpClient })).toEqual({
      ok: true,
      message: 'GLOBAL_KEY removed for all agents'
    })
    expect(cpClient.deleteAgentSecret).toHaveBeenCalledWith('beta-id', 'GLOBAL_KEY')
    expect(cpClient.listAgentSecrets).not.toHaveBeenCalledWith('catalog-id')
  })

  it('explains when no installed agent can remove a global secret', async () => {
    const cpClient = inventoryClient()
    vi.mocked(cpClient.listAllSecrets).mockResolvedValue({
      secrets: [{ key: 'ORPHANED', scope: 'global' }]
    })
    vi.mocked(cpClient.listAgentSecrets).mockResolvedValue({ secrets: [] })

    expect(await revokeStoredSecret('ORPHANED', 'global', { cpClient })).toEqual({
      ok: false,
      message: 'no installed agent can remove ORPHANED'
    })
    expect(cpClient.deleteAgentSecret).not.toHaveBeenCalled()
    expect(cpClient.listAgentSecrets).not.toHaveBeenCalledWith('catalog-id')
  })
})
