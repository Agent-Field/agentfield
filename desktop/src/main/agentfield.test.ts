import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  checkControlPlane,
  deriveAgentBadge,
  fetchControlPlaneNodes,
  fetchExecutions,
  getAgentFieldHome,
  getSnapshot,
  readInstalledAgents,
  type FetchLike
} from './agentfield'
import { installCommand, sanitizeInstallOutput } from './installer'
import { CATALOG, catalogEntry } from '../shared/catalog'

const tmpDirs: string[] = []

async function makeHome(installedYaml?: string): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'agentfield-desktop-test-'))
  tmpDirs.push(dir)
  if (installedYaml !== undefined) {
    await fs.writeFile(path.join(dir, 'installed.yaml'), installedYaml, 'utf8')
  }
  return dir
}

afterEach(async () => {
  await Promise.all(
    tmpDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true }))
  )
})

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' }
  })
}

const REGISTRY_FIXTURE = `installed:
  pr-af:
    name: pr-af
    version: 0.1.0
    description: Opens draft pull requests from a task description
    path: /home/abir/.agentfield/packages/pr-af
    source: local
    source_path: ./fix-praf
    installed_at: "2026-07-08T10:35:03-04:00"
    status: running
    language: python
    runtime:
      port: 9001
      pid: 4242
      started_at: "2026-07-08T10:36:00-04:00"
      log_file: /home/abir/.agentfield/logs/pr-af.log
  swe-af:
    version: 0.2.1
    description: Software engineering agent
    status: stopped
    runtime:
      port: null
      pid: null
      started_at: null
      log_file: /home/abir/.agentfield/logs/swe-af.log
`

describe('getAgentFieldHome', () => {
  it('is <homedir>/.agentfield', () => {
    expect(getAgentFieldHome()).toBe(path.join(os.homedir(), '.agentfield'))
  })
})

describe('readInstalledAgents', () => {
  // Contract: registry with running + stopped entries (including null runtime
  // fields and a missing optional language) parses into a correct agents array.
  it('parses running and stopped entries, null runtime fields, optional language', async () => {
    const home = await makeHome(REGISTRY_FIXTURE)
    const result = await readInstalledAgents(home)

    expect(result.exists).toBe(true)
    expect(result.error).toBeUndefined()
    expect(result.agents).toHaveLength(2)

    const prAf = result.agents.find((a) => a.name === 'pr-af')
    expect(prAf).toEqual({
      name: 'pr-af',
      version: '0.1.0',
      description: 'Opens draft pull requests from a task description',
      language: 'python',
      status: 'running',
      path: '/home/abir/.agentfield/packages/pr-af',
      port: 9001,
      pid: 4242
    })

    // Entry without a `name` field falls back to its registry key; nulls stay null.
    const sweAf = result.agents.find((a) => a.name === 'swe-af')
    expect(sweAf).toEqual({
      name: 'swe-af',
      version: '0.2.1',
      description: 'Software engineering agent',
      language: undefined,
      status: 'stopped',
      path: null,
      port: null,
      pid: null
    })
  })

  // Contract: missing installed.yaml (or missing ~/.agentfield entirely) is a
  // graceful empty state, not an error.
  it('returns { exists: false, agents: [] } when installed.yaml is missing', async () => {
    const home = await makeHome() // dir exists, no installed.yaml
    expect(await readInstalledAgents(home)).toEqual({ exists: false, agents: [] })
  })

  it('returns { exists: false, agents: [] } when the home dir itself is missing', async () => {
    const home = await makeHome()
    const missing = path.join(home, 'does-not-exist')
    expect(await readInstalledAgents(missing)).toEqual({ exists: false, agents: [] })
  })

  // Contract: malformed YAML surfaces as an error string — never throws.
  it('surfaces malformed YAML as an error string without throwing', async () => {
    const home = await makeHome('installed:\n  pr-af: [unclosed\n')
    const result = await readInstalledAgents(home)
    expect(result.exists).toBe(true)
    expect(result.agents).toEqual([])
    expect(result.error).toContain('installed.yaml')
  })

  it('treats a YAML doc without an installed map as an empty registry', async () => {
    const home = await makeHome('something_else: true\n')
    const result = await readInstalledAgents(home)
    expect(result).toEqual({ exists: true, agents: [] })
  })
})

describe('deriveAgentBadge', () => {
  // Contract: full truth table. Third column is the node's control-plane
  // health_status, or null when it is not registered there.
  it.each([
    // [registryStatus, cpReachable, nodeHealth, expected]
    // CP view unavailable -> trust the registry
    ['running', false, null, 'running'],
    ['running', false, 'active', 'running'],
    ['stopped', false, null, 'stopped'],
    ['stopped', false, 'active', 'stopped'],
    ['error', false, null, 'unknown'],
    [undefined, false, null, 'unknown'],
    // CP view available -> cross-check
    ['running', true, 'active', 'running'],
    // Registration presence beats a transient health dip — no flicker.
    ['running', true, 'unknown', 'running'],
    ['running', true, 'inactive', 'running'],
    ['running', true, null, 'unknown'], // stale registry
    ['stopped', true, 'active', 'unknown'], // conflict: something is serving
    // Stopped nodes STAY registered (inactive/unknown) — still just stopped.
    ['stopped', true, 'inactive', 'stopped'],
    ['stopped', true, 'unknown', 'stopped'],
    ['stopped', true, null, 'stopped'],
    ['error', true, 'active', 'unknown'],
    [undefined, true, null, 'unknown']
  ] as const)(
    'status=%s reachable=%s health=%s -> %s',
    (status, reachable, health, expected) => {
      expect(deriveAgentBadge(status, reachable, health)).toBe(expected)
    }
  )
})

describe('checkControlPlane', () => {
  // Contract: 200 healthy body -> reachable + healthy.
  it('maps a 200 healthy body to reachable/healthy', async () => {
    const body = {
      status: 'healthy',
      timestamp: '2026-07-10T12:00:00Z',
      version: '0.1.107',
      checks: {}
    }
    const fetchImpl: FetchLike = async () => jsonResponse(body, 200)
    const result = await checkControlPlane('http://localhost:8080', fetchImpl)
    expect(result).toEqual({ reachable: true, recognized: true, healthy: true, raw: body })
  })

  // Contract: 503 with an unhealthy body still means reachable, just not healthy.
  it('maps a 503 unhealthy body to reachable but not healthy', async () => {
    const body = { status: 'unhealthy', checks: { database: 'down' } }
    const fetchImpl: FetchLike = async () => jsonResponse(body, 503)
    const result = await checkControlPlane('http://localhost:8080', fetchImpl)
    expect(result.reachable).toBe(true)
    expect(result.recognized).toBe(true)
    expect(result.healthy).toBe(false)
    expect(result.raw).toEqual(body)
  })

  // Contract: a 200 from something that is NOT an AgentField control plane
  // (default port 8080 is popular) must not read as healthy. Found live on
  // Windows: an unrelated dev server answering {"status":"alive"} on /health
  // lit the dashboard green.
  it('rejects a foreign 200 /health payload as unrecognized', async () => {
    const body = { status: 'alive', uptime_s: 3714 }
    const fetchImpl: FetchLike = async () => jsonResponse(body, 200)
    const result = await checkControlPlane('http://localhost:8080', fetchImpl)
    expect(result.reachable).toBe(true)
    expect(result.recognized).toBe(false)
    expect(result.healthy).toBe(false)
    expect(result.error).toContain('does not look like an AgentField control plane')
  })

  it('rejects a non-JSON 200 response as unrecognized', async () => {
    const fetchImpl: FetchLike = async () =>
      new Response('<html>hi</html>', { status: 200, headers: { 'content-type': 'text/html' } })
    const result = await checkControlPlane('http://localhost:8080', fetchImpl)
    expect(result.reachable).toBe(true)
    expect(result.recognized).toBe(false)
    expect(result.healthy).toBe(false)
  })

  // Contract: network error / timeout -> not reachable, error captured.
  it('maps a rejected fetch to unreachable with an error message', async () => {
    const fetchImpl: FetchLike = async () => {
      throw new TypeError('fetch failed')
    }
    const result = await checkControlPlane('http://localhost:8080', fetchImpl)
    expect(result).toEqual({
      reachable: false,
      recognized: false,
      healthy: false,
      error: 'fetch failed'
    })
  })

  it('probes {baseUrl}/health', async () => {
    let requested = ''
    const fetchImpl: FetchLike = async (input) => {
      requested = String(input)
      return jsonResponse({ status: 'healthy' })
    }
    await checkControlPlane('http://example.test:1234', fetchImpl)
    expect(requested).toBe('http://example.test:1234/health')
  })
})

describe('fetchControlPlaneNodes', () => {
  it('returns node ids from a 200 nodes payload', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({
        nodes: [
          { id: 'pr-af', health_status: 'active' },
          { id: 'swe-af', health_status: 'active' }
        ],
        count: 2
      })
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toEqual(
      new Map([
        ['pr-af', 'active'],
        ['swe-af', 'active']
      ])
    )
  })

  it('returns null on a non-200 response', async () => {
    const fetchImpl: FetchLike = async () => jsonResponse({ error: 'nope' }, 500)
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it('returns null when fetch rejects', async () => {
    const fetchImpl: FetchLike = async () => {
      throw new TypeError('fetch failed')
    }
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it('returns null on an unexpected payload shape', async () => {
    const fetchImpl: FetchLike = async () => jsonResponse({ items: [] })
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it('requests the unfiltered node list (show_all) so health dips cannot flicker badges', async () => {
    let requested = ''
    const fetchImpl: FetchLike = async (url) => {
      requested = String(url)
      return jsonResponse({ nodes: [{ id: 'pr-af', health_status: 'unknown' }], count: 1 })
    }
    // A node whose health momentarily reads "unknown" is still SEEN — its
    // registration is what proves the registry entry is not stale.
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toEqual(
      new Map([['pr-af', 'unknown']])
    )
    expect(requested).toContain('show_all=true')
  })
})

describe('fetchExecutions', () => {
  const runRow = (overrides: Record<string, unknown>) => ({
    run_id: 'run_1',
    status: 'succeeded',
    display_name: 'demo_echo',
    agent_id: 'smoke-agent',
    started_at: '2026-07-13T13:51:39Z',
    duration_ms: 45,
    terminal: true,
    ...overrides
  })

  it('splits rows into running (non-terminal) and recent (terminal)', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({
        runs: [
          runRow({ run_id: 'run_live', status: 'running', terminal: false, duration_ms: null }),
          runRow({ run_id: 'run_done' })
        ],
        total_count: 2
      })
    const result = await fetchExecutions('http://localhost:8080', fetchImpl)
    expect(result).not.toBeNull()
    expect(result!.running.map((r) => r.runId)).toEqual(['run_live'])
    expect(result!.recent.map((r) => r.runId)).toEqual(['run_done'])
    expect(result!.recent[0]).toEqual({
      runId: 'run_done',
      status: 'succeeded',
      displayName: 'demo_echo',
      agentId: 'smoke-agent',
      startedAt: '2026-07-13T13:51:39Z',
      durationMs: 45,
      terminal: true,
      errorMessage: null
    })
  })

  it('surfaces the root error message on failed runs', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({
        runs: [
          runRow({
            run_id: 'run_bad',
            status: 'failed',
            root_error_message: 'review execution failed: CLI command made no progress for 360s'
          }),
          runRow({ run_id: 'run_ok' })
        ],
        total_count: 2
      })
    const result = await fetchExecutions('http://localhost:8080', fetchImpl)
    expect(result!.recent.map((r) => r.errorMessage)).toEqual([
      'review execution failed: CLI command made no progress for 360s',
      null
    ])
  })

  it('caps recent executions at 5', async () => {
    const runs = Array.from({ length: 9 }, (_, i) => runRow({ run_id: `run_${i}` }))
    const fetchImpl: FetchLike = async () => jsonResponse({ runs, total_count: 9 })
    const result = await fetchExecutions('http://localhost:8080', fetchImpl)
    expect(result!.recent).toHaveLength(5)
  })

  it('drops rows without a run_id instead of failing', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({ runs: [runRow({}), { status: 'running' }], total_count: 2 })
    const result = await fetchExecutions('http://localhost:8080', fetchImpl)
    expect(result!.recent).toHaveLength(1)
    expect(result!.running).toHaveLength(0)
  })

  it.each([
    ['non-200 response', async () => jsonResponse({ error: 'nope' }, 500)],
    ['junk payload', async () => jsonResponse({ items: [] })],
    [
      'rejected fetch',
      async () => {
        throw new TypeError('fetch failed')
      }
    ]
  ] as const)('returns null on %s', async (_name, fetchImpl) => {
    expect(await fetchExecutions('http://localhost:8080', fetchImpl)).toBeNull()
  })
})

describe('install catalog', () => {
  it('every entry has a name, description, and an https or af:// source', () => {
    expect(CATALOG.length).toBeGreaterThan(0)
    for (const entry of CATALOG) {
      expect(entry.name).toMatch(/^[a-z0-9][a-z0-9-]*$/)
      expect(entry.description.length).toBeGreaterThan(0)
      expect(entry.source).toMatch(/^(https:\/\/|af:\/\/)/)
    }
  })

  it('entry names are unique', () => {
    const names = CATALOG.map((e) => e.name)
    expect(new Set(names).size).toBe(names.length)
  })

  it('catalogEntry resolves known names and rejects unknown ones', () => {
    expect(catalogEntry(CATALOG[0].name)).toEqual(CATALOG[0])
    expect(catalogEntry('definitely-not-real')).toBeUndefined()
  })
})

describe('installCommand', () => {
  // Contract: the renderer sends catalog *names* over IPC; only vetted
  // sources ever reach spawn, and unknown names are refused.
  it('builds `af install <source>` for a catalog name', () => {
    expect(installCommand(CATALOG[0].name)).toEqual({
      command: 'af',
      args: ['install', CATALOG[0].source]
    })
  })

  it('returns null for names not in the catalog', () => {
    expect(installCommand('evil; rm -rf /')).toBeNull()
    expect(installCommand('')).toBeNull()
  })
})

describe('sanitizeInstallOutput', () => {
  it('strips ANSI color and erase codes and splits spinner frames', () => {
    const esc = String.fromCharCode(27)
    const chunk = `${esc}[32m✓ Dependencies installed${esc}[0m\r${esc}[K✓ Installed swe-af v0.2.0\n`
    expect(sanitizeInstallOutput(chunk)).toEqual([
      '✓ Dependencies installed',
      '✓ Installed swe-af v0.2.0'
    ])
  })

  it('drops empty and whitespace-only lines', () => {
    expect(sanitizeInstallOutput('\r\n  \n\r')).toEqual([])
  })
})

describe('getSnapshot', () => {
  function routedFetch(routes: Record<string, () => Response>): FetchLike {
    return async (input) => {
      const url = String(input)
      const route = Object.keys(routes).find((suffix) => url.endsWith(suffix))
      if (!route) throw new TypeError(`unexpected fetch: ${url}`)
      return routes[route]()
    }
  }

  it('composes control plane + registry with cross-checked badges', async () => {
    const home = await makeHome(REGISTRY_FIXTURE)
    const fetchImpl = routedFetch({
      '/health': () => jsonResponse({ status: 'healthy' }),
      // Control plane sees pr-af but not swe-af.
      '/api/v1/nodes': () => jsonResponse({ nodes: [{ id: 'pr-af' }], count: 1 }),
      'sort_order=desc': () =>
        jsonResponse({
          runs: [
            {
              run_id: 'run_live',
              status: 'running',
              display_name: 'summarize',
              agent_id: 'pr-af',
              started_at: '2026-07-13T13:51:39Z',
              duration_ms: null,
              terminal: false
            }
          ],
          total_count: 1
        }),
      '/dashboard/summary': () =>
        jsonResponse({
          agents: { running: 1, total: 2 },
          executions: { today: 4, yesterday: 2 },
          success_rate: 100,
          packages: { available: 1, installed: 0 }
        })
    })

    const snapshot = await getSnapshot({ homeDir: home, fetchImpl })

    expect(snapshot.controlPlane.baseUrl).toBe('http://localhost:8080')
    expect(snapshot.controlPlane.reachable).toBe(true)
    expect(snapshot.controlPlane.healthy).toBe(true)
    expect(snapshot.registry.exists).toBe(true)
    expect(snapshot.executions?.running.map((r) => r.runId)).toEqual(['run_live'])
    expect(snapshot.metrics).toEqual({
      agentsRunning: 1,
      agentsTotal: 2,
      executionsToday: 4,
      executionsYesterday: 2,
      successRate: 100
    })
    expect(Date.parse(snapshot.fetchedAt)).not.toBeNaN()

    const badges = Object.fromEntries(
      snapshot.registry.agents.map((a) => [a.name, a.badge])
    )
    expect(badges).toEqual({
      'pr-af': 'running', // registry running + seen on CP
      'swe-af': 'stopped' // registry stopped + not seen
    })
  })

  it('falls back to registry status when the nodes endpoint fails', async () => {
    const home = await makeHome(REGISTRY_FIXTURE)
    const fetchImpl = routedFetch({
      '/health': () => jsonResponse({ status: 'healthy' }),
      '/api/v1/nodes': () => jsonResponse({ error: 'boom' }, 500)
    })

    const snapshot = await getSnapshot({ homeDir: home, fetchImpl })
    const badges = Object.fromEntries(
      snapshot.registry.agents.map((a) => [a.name, a.badge])
    )
    // Nodes view unavailable -> trust registry statuses directly.
    expect(badges).toEqual({ 'pr-af': 'running', 'swe-af': 'stopped' })
  })

  it('does not consult the nodes view of an unrecognized service on the port', async () => {
    const home = await makeHome(REGISTRY_FIXTURE)
    const requested: string[] = []
    const fetchImpl: FetchLike = async (input) => {
      requested.push(String(input))
      // A foreign service that would answer BOTH endpoints with junk.
      return jsonResponse({ status: 'alive', nodes: [] })
    }

    const snapshot = await getSnapshot({ homeDir: home, fetchImpl })

    expect(snapshot.controlPlane.recognized).toBe(false)
    // Badges fall back to registry statuses — the foreign 200 on /api/v1/nodes
    // must not flip a running agent to unknown.
    const badges = Object.fromEntries(
      snapshot.registry.agents.map((a) => [a.name, a.badge])
    )
    expect(badges).toEqual({ 'pr-af': 'running', 'swe-af': 'stopped' })
    expect(requested.some((url) => url.includes('/api/v1/nodes'))).toBe(false)
    // Nor may its workflow runs show up as activity.
    expect(snapshot.executions).toBeNull()
    expect(requested.some((url) => url.includes('/workflow-runs'))).toBe(false)
  })

  it('reports an unreachable control plane and an absent registry gracefully', async () => {
    const home = await makeHome()
    const missing = path.join(home, 'nope')
    const fetchImpl: FetchLike = async () => {
      throw new TypeError('fetch failed')
    }

    const snapshot = await getSnapshot({ homeDir: missing, fetchImpl })
    expect(snapshot.controlPlane.reachable).toBe(false)
    expect(snapshot.registry).toEqual({ exists: false, agents: [], error: undefined })
  })
})
