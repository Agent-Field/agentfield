import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  checkControlPlane,
  deriveAgentBadge,
  fetchControlPlaneNodes,
  getAgentFieldHome,
  getSnapshot,
  readInstalledAgents,
  type FetchLike
} from './agentfield'

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
  // Contract: full truth table.
  it.each([
    // [registryStatus, cpReachable, nodeSeen, expected]
    // CP view unavailable -> trust the registry
    ['running', false, false, 'running'],
    ['running', false, true, 'running'],
    ['stopped', false, false, 'stopped'],
    ['stopped', false, true, 'stopped'],
    ['error', false, false, 'unknown'],
    [undefined, false, false, 'unknown'],
    // CP view available -> cross-check
    ['running', true, true, 'running'],
    ['running', true, false, 'unknown'], // stale registry
    ['stopped', true, true, 'unknown'], // conflict
    ['stopped', true, false, 'stopped'],
    ['error', true, true, 'unknown'],
    [undefined, true, false, 'unknown']
  ] as const)(
    'status=%s reachable=%s seen=%s -> %s',
    (status, reachable, seen, expected) => {
      expect(deriveAgentBadge(status, reachable, seen)).toBe(expected)
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
    expect(await fetchControlPlaneNodes('http://localhost:8080', fetchImpl)).toEqual([
      'pr-af',
      'swe-af'
    ])
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
      '/api/v1/nodes': () => jsonResponse({ nodes: [{ id: 'pr-af' }], count: 1 })
    })

    const snapshot = await getSnapshot({ homeDir: home, fetchImpl })

    expect(snapshot.controlPlane.baseUrl).toBe('http://localhost:8080')
    expect(snapshot.controlPlane.reachable).toBe(true)
    expect(snapshot.controlPlane.healthy).toBe(true)
    expect(snapshot.registry.exists).toBe(true)
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
    expect(requested.some((url) => url.endsWith('/api/v1/nodes'))).toBe(false)
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
