import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CpClient, PackageInfo } from './cpClient'
import {
  DEFAULT_BASE_URL,
  checkControlPlane,
  deriveAgentBadge,
  fetchControlPlaneNodes,
  fetchDashboardMetrics,
  fetchExecutions,
  fetchUsageStats,
  getAgentFieldHome,
  getBaseUrl,
  getSnapshot,
  readInstalledAgents,
  setActiveControlPlanePort,
  type FetchLike
} from './agentfield'
import { DEFAULT_CONTROL_PLANE_PORT } from './ports'
import { installCommand, sanitizeInstallOutput } from './installer'
import { CATALOG, catalogEntry } from '../shared/catalog'
import { setCloudConnection, setLocalApiKey } from './connection'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' }
  })
}

const API_PACKAGES: PackageInfo[] = [
  {
    id: 'pr-af', name: 'pr-af', version: '0.1.0',
    description: 'Opens draft pull requests from a task description',
    status: 'configured', install_status: 'running',
    install_path: '/home/abir/.agentfield/packages/pr-af',
    port: 9001, process_id: 4242, configuration_required: false,
    configuration_complete: true, author: ''
  },
  {
    id: 'swe-af', name: 'swe-af', version: '0.2.1',
    description: 'Software engineering agent', status: 'not_configured',
    install_status: 'stopped', install_path: '',
    configuration_required: false, configuration_complete: true, author: ''
  }
]

function packagesClient(packages = API_PACKAGES): CpClient {
  return { listPackages: vi.fn(async () => ({ packages, total: packages.length })) } as unknown as CpClient
}

describe('active base URL', () => {
  afterEach(() => setActiveControlPlanePort(DEFAULT_CONTROL_PLANE_PORT))

  it('starts at the default and follows the active port', () => {
    expect(getBaseUrl()).toBe(DEFAULT_BASE_URL)
    setActiveControlPlanePort(9091)
    expect(getBaseUrl()).toBe('http://localhost:9091')
  })

  it('drives the default probe target — nothing hard-codes 8080', async () => {
    setActiveControlPlanePort(9091)
    const seen: string[] = []
    const fetchImpl: FetchLike = async (input) => {
      seen.push(String(input))
      return jsonResponse({ status: 'healthy' })
    }
    await checkControlPlane(undefined, fetchImpl)
    expect(seen).toEqual(['http://localhost:9091/health'])
  })
})

describe('raw control-plane fetch authentication', () => {
  afterEach(() => {
    setLocalApiKey(null)
    setActiveControlPlanePort(DEFAULT_CONTROL_PLANE_PORT)
  })

  function readerFetch(): FetchLike {
    return vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/health')) return jsonResponse({ status: 'healthy' })
      if (url.includes('/nodes')) return jsonResponse({ nodes: [] })
      if (url.includes('/workflow-runs')) return jsonResponse({ runs: [] })
      if (url.includes('/dashboard/summary')) return jsonResponse({})
      return jsonResponse({ totals: {} })
    }) as FetchLike
  }

  async function callRawReaders(fetchImpl: FetchLike): Promise<void> {
    await checkControlPlane(undefined, fetchImpl)
    await fetchControlPlaneNodes(undefined, fetchImpl)
    await fetchExecutions(undefined, fetchImpl)
    await fetchDashboardMetrics(undefined, fetchImpl)
    await fetchUsageStats(undefined, fetchImpl)
  }

  it('adds X-API-Key to every raw reader in cloud mode', async () => {
    setCloudConnection('https://cp.example', 'cloud-key')
    const fetchImpl = readerFetch()
    await callRawReaders(fetchImpl)
    expect(vi.mocked(fetchImpl).mock.calls).toHaveLength(5)
    for (const [, init] of vi.mocked(fetchImpl).mock.calls) {
      expect(new Headers(init?.headers).get('X-API-Key')).toBe('cloud-key')
    }
  })

  // Default local mode has no key: the control plane authenticates the app by
  // its loopback address, so nothing must be sent.
  it('omits X-API-Key from every raw reader in local mode', async () => {
    setActiveControlPlanePort(8080)
    const fetchImpl = readerFetch()
    await callRawReaders(fetchImpl)
    for (const [, init] of vi.mocked(fetchImpl).mock.calls) {
      expect(new Headers(init?.headers).has('X-API-Key')).toBe(false)
    }
  })

  it('adds X-API-Key to every raw reader when the local server needs one', async () => {
    setActiveControlPlanePort(8080)
    setLocalApiKey('local-secret')
    const fetchImpl = readerFetch()
    await callRawReaders(fetchImpl)
    expect(vi.mocked(fetchImpl).mock.calls).toHaveLength(5)
    for (const [, init] of vi.mocked(fetchImpl).mock.calls) {
      expect(new Headers(init?.headers).get('X-API-Key')).toBe('local-secret')
    }
  })
})

describe('getAgentFieldHome', () => {
  it('is <homedir>/.agentfield', () => {
    expect(getAgentFieldHome()).toBe(path.join(os.homedir(), '.agentfield'))
  })
})

describe('readInstalledAgents', () => {
  it('maps running and stopped API packages, including absent runtime fields', async () => {
    const result = await readInstalledAgents(packagesClient())

    expect(result.exists).toBe(true)
    expect(result.error).toBeUndefined()
    expect(result.agents).toHaveLength(2)

    const prAf = result.agents.find((a) => a.name === 'pr-af')
    expect(prAf).toEqual({
      name: 'pr-af',
      version: '0.1.0',
      description: 'Opens draft pull requests from a task description',
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
      status: 'stopped',
      path: null,
      port: null,
      pid: null
    })
  })

  it('maps an empty package list to an existing empty registry', async () => {
    const result = await readInstalledAgents(packagesClient([]))
    expect(result).toEqual({ exists: true, agents: [] })
  })

  it('prefers install_status and falls back to status for old servers', async () => {
    const lifecycle = { ...API_PACKAGES[0], status: 'not_configured', install_status: 'installed' }
    const legacy = { ...API_PACKAGES[1], status: 'running', install_status: undefined }

    const result = await readInstalledAgents(packagesClient([lifecycle, legacy]))

    expect(result.agents.map((agent) => agent.status)).toEqual(['installed', 'running'])
  })

  it('surfaces API failures without throwing', async () => {
    const client = packagesClient()
    vi.mocked(client.listPackages).mockRejectedValue(new Error('offline'))
    expect(await readInstalledAgents(client)).toEqual({
      exists: false, agents: [], error: 'offline'
    })
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
    ['installed', false, null, 'unknown'],
    ['installed', false, 'active', 'unknown'],
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
    ['installed', true, 'active', 'running'],
    ['installed', true, 'inactive', 'stopped'],
    ['installed', true, null, 'stopped'],
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

  it('treats a null nodes slice as an empty control-plane view', async () => {
    const fetchImpl: FetchLike = async () => jsonResponse({ nodes: null, count: 0 })
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toEqual(new Map())
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

  it('treats a null runs slice as empty activity', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({ runs: null, total_count: 0 })
    await expect(fetchExecutions('http://localhost:8080', fetchImpl)).resolves.toEqual({
      running: [],
      recent: []
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

describe('fetchUsageStats', () => {
  // Contract: happy path maps totals + by_harness/by_agent, cost_usd → costUsd.
  it('parses cost + by_harness (and by_agent) from a 200 payload', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({
        window: '24h',
        totals: {
          cost_usd: 1.23,
          input_tokens: 1000,
          output_tokens: 2000,
          total_tokens: 3000,
          executions_with_usage: 42
        },
        by_agent: [{ key: 'pr-af', cost_usd: 0.8, total_tokens: 2000, entries: 10 }],
        by_harness: [{ key: 'claude-code', cost_usd: 1.0, total_tokens: 1500, entries: 8 }]
      })
    await expect(fetchUsageStats('http://localhost:8080', fetchImpl)).resolves.toEqual({
      window: '24h',
      costUsd: 1.23,
      totalTokens: 3000,
      byAgent: [{ key: 'pr-af', costUsd: 0.8, totalTokens: 2000 }],
      byHarness: [{ key: 'claude-code', costUsd: 1.0, totalTokens: 1500 }]
    })
  })

  // Contract: null cost_usd stays null; tokens still present (tokens-only UI).
  it('maps null cost_usd to costUsd null while keeping tokens', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({
        window: '24h',
        totals: { cost_usd: null, total_tokens: 4500, executions_with_usage: 3 },
        by_agent: [],
        by_harness: [{ key: 'codex', cost_usd: null, total_tokens: 4500 }]
      })
    const result = await fetchUsageStats('http://localhost:8080', fetchImpl)
    expect(result).toEqual({
      window: '24h',
      costUsd: null,
      totalTokens: 4500,
      byAgent: [],
      byHarness: [{ key: 'codex', costUsd: null, totalTokens: 4500 }]
    })
  })

  // Contract: 404 (older CP without the endpoint) → null so UI hides usage.
  it('returns null on 404', async () => {
    const fetchImpl: FetchLike = async () => jsonResponse({ error: 'not found' }, 404)
    expect(await fetchUsageStats('http://localhost:8080', fetchImpl)).toBeNull()
  })

  // Contract: bad JSON / unexpected body → null, never throws across IPC.
  it('returns null on bad JSON', async () => {
    const fetchImpl: FetchLike = async () =>
      new Response('not json', { status: 200, headers: { 'content-type': 'application/json' } })
    expect(await fetchUsageStats('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it.each([401, 403, 500] as const)('returns null on %s', async (status) => {
    const fetchImpl: FetchLike = async () => jsonResponse({ error: 'nope' }, status)
    expect(await fetchUsageStats('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it('returns null when fetch rejects', async () => {
    const fetchImpl: FetchLike = async () => {
      throw new TypeError('fetch failed')
    }
    expect(await fetchUsageStats('http://localhost:8080', fetchImpl)).toBeNull()
  })

  it('requests usage/stats?window=24h', async () => {
    let requested = ''
    const fetchImpl: FetchLike = async (input) => {
      requested = String(input)
      return jsonResponse({ window: '24h', totals: { total_tokens: 0 } })
    }
    await fetchUsageStats('http://example.test:1234', fetchImpl)
    expect(requested).toBe('http://example.test:1234/api/ui/v1/usage/stats?window=24h')
  })

  it('tolerates missing by_agent / by_harness arrays as empty', async () => {
    const fetchImpl: FetchLike = async () =>
      jsonResponse({ window: '24h', totals: { cost_usd: 0.5, total_tokens: 100 } })
    await expect(fetchUsageStats('http://localhost:8080', fetchImpl)).resolves.toEqual({
      window: '24h',
      costUsd: 0.5,
      totalTokens: 100,
      byAgent: [],
      byHarness: []
    })
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
  it('builds a control-plane install preview for a catalog name', () => {
    expect(installCommand(CATALOG[0].name)).toEqual({
      command: 'control-plane',
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
        }),
      'usage/stats?window=24h': () =>
        jsonResponse({
          window: '24h',
          totals: { cost_usd: 1.23, total_tokens: 3000 },
          by_agent: [],
          by_harness: [{ key: 'claude-code', cost_usd: 1.23, total_tokens: 3000 }]
        })
    })

    const snapshot = await getSnapshot({ cpClient: packagesClient(), fetchImpl })

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
    expect(snapshot.usage).toEqual({
      window: '24h',
      costUsd: 1.23,
      totalTokens: 3000,
      byAgent: [],
      byHarness: [{ key: 'claude-code', costUsd: 1.23, totalTokens: 3000 }]
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
    const fetchImpl = routedFetch({
      '/health': () => jsonResponse({ status: 'healthy' }),
      '/api/v1/nodes': () => jsonResponse({ error: 'boom' }, 500)
    })

    const snapshot = await getSnapshot({ cpClient: packagesClient(), fetchImpl })
    const badges = Object.fromEntries(
      snapshot.registry.agents.map((a) => [a.name, a.badge])
    )
    // Nodes view unavailable -> trust registry statuses directly.
    expect(badges).toEqual({ 'pr-af': 'running', 'swe-af': 'stopped' })
  })

  it('does not consult the nodes view of an unrecognized service on the port', async () => {
    const requested: string[] = []
    const fetchImpl: FetchLike = async (input) => {
      requested.push(String(input))
      // A foreign service that would answer BOTH endpoints with junk.
      return jsonResponse({ status: 'alive', nodes: [] })
    }

    const snapshot = await getSnapshot({ cpClient: packagesClient(), fetchImpl })

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
    // Usage is only fetched against a recognized control plane.
    expect(snapshot.usage).toBeNull()
    expect(requested.some((url) => url.includes('/usage/stats'))).toBe(false)
  })

  it('reports an unreachable control plane and an absent registry gracefully', async () => {
    const fetchImpl: FetchLike = async () => {
      throw new TypeError('fetch failed')
    }

    const unavailable = packagesClient()
    vi.mocked(unavailable.listPackages).mockRejectedValue(new Error('fetch failed'))
    const snapshot = await getSnapshot({ cpClient: unavailable, fetchImpl })
    expect(snapshot.controlPlane.reachable).toBe(false)
    expect(snapshot.registry).toEqual({ exists: false, agents: [], error: 'fetch failed' })
    expect(snapshot.usage).toBeNull()
  })
})

// Catalog rows (install_status uninstalled / other) are filtered from the
// installed-agents listing; absent install_status keeps the row (old CP).
it('readInstalledAgents filters catalog and uninstalled rows', async () => {
  const pkg = (over: Record<string, unknown>) => ({
    id: 'x', name: 'x', version: '1', status: 'not_configured', install_path: '/p',
    configuration_required: false, configuration_complete: true,
    description: '', author: '', ...over
  })
  const cpClient = {
    listPackages: async () => ({
      packages: [
        pkg({ name: 'installed-one', install_status: 'installed' }),
        pkg({ name: 'running-one', install_status: 'running' }),
        pkg({ name: 'stopped-one', install_status: 'stopped' }),
        pkg({ name: 'catalog-one', install_status: 'uninstalled' }),
        pkg({ name: 'weird-one', install_status: 'not_configured' }),
        pkg({ name: 'legacy-cp-row' })
      ],
      total: 6
    })
  } as never
  const result = await readInstalledAgents(cpClient)
  expect(result.agents.map((a) => a.name)).toEqual([
    'installed-one', 'running-one', 'stopped-one', 'legacy-cp-row'
  ])
})
