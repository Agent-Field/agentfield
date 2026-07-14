import yaml from 'js-yaml'
import { describe, expect, it } from 'vitest'
import {
  buildEnvReport,
  buildSecretsInventory,
  findSpecVar,
  parseSecretsTable,
  parseUserEnvironment,
  specIsEmpty
} from './secrets'

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
