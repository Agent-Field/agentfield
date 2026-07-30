import { describe, expect, it, vi } from 'vitest'
import {
  CpApiError,
  createCpClient,
  isInstalledPackage,
  type FetchLike,
  type InstallJob,
  type PackageInfo
} from './cpClient'

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' }
  })
}

function mockFetch(responses: Response[]): FetchLike {
  return vi.fn(async () => responses.shift() ?? json({})) as FetchLike
}

const job = (status: InstallJob['status'], lines: string[] = []): InstallJob => ({
  id: 'job/1',
  source: 'https://github.com/acme/agent',
  kind: 'install',
  status,
  lines,
  started_at: '2026-07-30T12:00:00Z'
})

describe('createCpClient', () => {
  // Contract 1: every wrapper uses the documented method, path, body, and result.
  it('wraps all management endpoints', async () => {
    const payloads: unknown[] = [
      { job_id: 'i' },
      job('running'),
      [job('succeeded')],
      { package_id: 'pkg', status: 'uninstalled' },
      { job_id: 'u' },
      { packages: [], total: 0 },
      { agent_id: 'agent', status: 'started', pid: 1, port: 2, started_at: 't', log_file: 'l', message: 'ok', endpoints: {} },
      { agent_id: 'agent', status: 'stopped', message: 'ok' },
      { agent_id: 'agent', name: 'Agent', is_running: true, status: 'running' },
      { running_agents: [], total_count: 0 },
      { secrets: [{ key: 'TOKEN', is_set: true, scope: 'global' }] },
      null,
      null,
      { secrets: [{ key: 'TOKEN', scope: 'global' }] }
    ]
    const responses = payloads.map((body, index) =>
      index === 11 || index === 12 ? new Response(null, { status: 204 }) : json(body)
    )
    const fetchImpl = mockFetch(responses)
    const client = createCpClient({ baseUrl: () => 'http://cp', fetchImpl })

    expect(await client.installPackage('https://github.com/acme/agent', true)).toEqual({ job_id: 'i' })
    expect(await client.getInstallJob('job/1')).toEqual(job('running'))
    expect(await client.listInstallJobs()).toEqual([job('succeeded')])
    expect(await client.uninstallPackage('pkg/name')).toEqual({ package_id: 'pkg', status: 'uninstalled' })
    expect(await client.updatePackage('pkg/name')).toEqual({ job_id: 'u' })
    expect(await client.listPackages()).toEqual({ packages: [], total: 0 })
    expect((await client.startAgent('agent/name', { port: 9000, detach: false })).pid).toBe(1)
    expect((await client.stopAgent('agent/name')).status).toBe('stopped')
    expect((await client.getAgentStatus('agent/name')).is_running).toBe(true)
    expect(await client.listRunningAgents()).toEqual({ running_agents: [], total_count: 0 })
    expect((await client.listAgentSecrets('agent/name')).secrets[0].key).toBe('TOKEN')
    await client.setAgentSecret('agent/name', 'TOKEN', 'secret')
    await client.deleteAgentSecret('agent/name', 'TOKEN', 'node')
    expect((await client.listAllSecrets()).secrets[0].scope).toBe('global')

    const calls = vi.mocked(fetchImpl).mock.calls
    expect(calls.map(([url, init]) => [url, init?.method ?? 'GET'])).toEqual([
      ['http://cp/api/ui/v1/agents/packages/install', 'POST'],
      ['http://cp/api/ui/v1/agents/packages/install/jobs/job%2F1', 'GET'],
      ['http://cp/api/ui/v1/agents/packages/install/jobs', 'GET'],
      ['http://cp/api/ui/v1/agents/packages/pkg%2Fname/uninstall', 'POST'],
      ['http://cp/api/ui/v1/agents/packages/pkg%2Fname/update', 'POST'],
      ['http://cp/api/ui/v1/agents/packages', 'GET'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/start', 'POST'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/stop', 'POST'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/status', 'GET'],
      ['http://cp/api/ui/v1/agents/running', 'GET'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets', 'GET'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets', 'PUT'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets/TOKEN?scope=node', 'DELETE'],
      ['http://cp/api/ui/v1/secrets', 'GET']
    ])
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      source: 'https://github.com/acme/agent',
      force: true
    })
    expect(JSON.parse(String(calls[6][1]?.body))).toEqual({ port: 9000, detach: false })
  })

  // Contracts 2 and 3: auth is conditional and late-bound providers are re-read.
  it('late-binds base URL and API key on every call', async () => {
    let host = 'http://one'
    let key: string | null = null
    const fetchImpl = mockFetch([json({ packages: [], total: 0 }), json({ packages: [], total: 0 })])
    const client = createCpClient({ baseUrl: () => host, apiKey: () => key, fetchImpl })
    await client.listPackages()
    host = 'http://two/'
    key = 'cloud-key'
    await client.listPackages()

    const calls = vi.mocked(fetchImpl).mock.calls
    expect(calls[0][0]).toBe('http://one/api/ui/v1/agents/packages')
    expect(new Headers(calls[0][1]?.headers).has('X-API-Key')).toBe(false)
    expect(calls[1][0]).toBe('http://two/api/ui/v1/agents/packages')
    expect(new Headers(calls[1][1]?.headers).get('X-API-Key')).toBe('cloud-key')
  })

  // Contract 4: HTTP errors retain status and parsed server details.
  it.each([
    [409, { error: 'operation busy', code: 'busy' }],
    [400, { error: 'invalid source' }]
  ])('throws CpApiError for status %i', async (status, body) => {
    const client = createCpClient({ fetchImpl: mockFetch([json(body, status)]) })
    const error = await client.installPackage('bad').catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(CpApiError)
    expect(error).toMatchObject({ status, message: body.error })
  })

  // Contract 4: malformed error JSON still produces a status-bearing error.
  it('retains status for malformed error JSON', async () => {
    const fetchImpl = mockFetch([new Response('{', { status: 500 })])
    const error = await createCpClient({ fetchImpl }).listPackages().catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(CpApiError)
    expect(error).toMatchObject({ status: 500 })
  })

  // Contract 5: install API feature detection distinguishes only 404.
  it('detects install API support and rethrows network failures', async () => {
    expect(await createCpClient({ fetchImpl: mockFetch([json([])]) }).hasInstallApi()).toBe(true)
    expect(await createCpClient({ fetchImpl: mockFetch([json({ error: 'missing' }, 404)]) }).hasInstallApi()).toBe(false)
    const failure = new TypeError('offline')
    const fetchImpl = vi.fn(async () => { throw failure }) as FetchLike
    await expect(createCpClient({ fetchImpl }).hasInstallApi()).rejects.toBe(failure)
  })

  // Contract 6: polling emits new lines once and resolves for either terminal state.
  it.each(['succeeded', 'failed'] as const)('watches a job through %s', async (terminal) => {
    let clock = 0
    const fetchImpl = mockFetch([
      json(job('running', ['one'])),
      json(job('running', ['one', 'two'])),
      json(job(terminal, ['one', 'two', 'three']))
    ])
    const lines: string[] = []
    const client = createCpClient({
      fetchImpl,
      now: () => clock,
      sleep: async (ms) => { clock += ms }
    })
    const result = await client.watchInstallJob('job/1', (line) => lines.push(line), {
      intervalMs: 10,
      timeoutMs: 100
    })
    expect(result.status).toBe(terminal)
    expect(lines).toEqual(['one', 'two', 'three'])
  })

  it('reports when a job disappears while it is being watched', async () => {
    let clock = 0
    const client = createCpClient({
      fetchImpl: mockFetch([
        json(job('running', ['one'])),
        json({ error: 'not found' }, 404)
      ]),
      now: () => clock,
      sleep: async (ms) => { clock += ms }
    })

    await expect(
      client.watchInstallJob('job/1', () => undefined, { intervalMs: 10, timeoutMs: 100 })
    ).rejects.toThrow('disappeared')
  })

  it('rethrows non-404 API errors while watching a job', async () => {
    let clock = 0
    const failure = new CpApiError({ status: 500, message: 'server failed' })
    const getInstallJob = vi
      .fn()
      .mockResolvedValueOnce(job('running'))
      .mockRejectedValueOnce(failure)
    const client = createCpClient({
      now: () => clock,
      sleep: async (ms) => { clock += ms }
    })
    client.getInstallJob = getInstallJob

    await expect(
      client.watchInstallJob('job/1', () => undefined, { intervalMs: 10, timeoutMs: 100 })
    ).rejects.toBe(failure)
  })

  it('does not replay lines when the retained job log shrinks', async () => {
    let clock = 0
    const firstLines = Array.from({ length: 10 }, (_, index) => `line ${index + 1}`)
    const retainedLines = firstLines.slice(4)
    const lines: string[] = []
    const client = createCpClient({
      fetchImpl: mockFetch([
        json(job('running', firstLines)),
        json(job('succeeded', retainedLines))
      ]),
      now: () => clock,
      sleep: async (ms) => { clock += ms }
    })

    await client.watchInstallJob('job/1', (line) => lines.push(line), {
      intervalMs: 10,
      timeoutMs: 100
    })
    expect(lines).toEqual(firstLines)
  })

  // Contract 6: polling rejects once its injected deadline is reached.
  it('times out while watching a non-terminal job', async () => {
    let clock = 0
    const client = createCpClient({
      fetchImpl: mockFetch([json(job('running'))]),
      now: () => clock,
      sleep: async (ms) => { clock += ms }
    })
    await expect(
      client.watchInstallJob('job/1', () => undefined, { intervalMs: 10, timeoutMs: 10 })
    ).rejects.toThrow('Timed out')
  })

  // Contract 7: secret scope query is encoded; absent PUT scope is omitted.
  it('serializes optional secret scope exactly', async () => {
    const fetchImpl = mockFetch([
      new Response(null, { status: 204 }),
      new Response(null, { status: 204 })
    ])
    const client = createCpClient({ baseUrl: () => 'http://cp', fetchImpl })
    await client.setAgentSecret('agent', 'TOKEN', 'value')
    await client.deleteAgentSecret('agent', 'TOKEN', 'global')
    const calls = vi.mocked(fetchImpl).mock.calls
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({ key: 'TOKEN', value: 'value' })
    expect(calls[1][0]).toBe('http://cp/api/ui/v1/agents/agent/secrets/TOKEN?scope=global')
  })

  it('omits force from install requests when it is undefined', async () => {
    const fetchImpl = mockFetch([json({ job_id: 'job' })])
    const client = createCpClient({ fetchImpl })
    await client.installPackage('https://github.com/acme/agent')

    expect(JSON.parse(String(vi.mocked(fetchImpl).mock.calls[0][1]?.body))).toEqual({
      source: 'https://github.com/acme/agent'
    })
  })
})

describe('isInstalledPackage', () => {
  const pkg = (install_status?: string): PackageInfo =>
    ({
      id: 'pkg',
      name: 'Package',
      version: '1.0.0',
      status: 'configured',
      install_status,
      install_path: '/tmp/pkg',
      configuration_required: false,
      configuration_complete: true,
      description: '',
      author: ''
    })

  it.each([
    [undefined, true],
    ['installed', true],
    ['running', true],
    ['stopped', true],
    ['uninstalled', false],
    ['not_configured', false],
    ['catalog', false],
    ['', false]
  ])('maps install_status %s to %s', (installStatus, expected) => {
    expect(isInstalledPackage(pkg(installStatus))).toBe(expected)
  })
})
