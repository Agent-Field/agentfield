import { afterEach, describe, expect, it, vi } from 'vitest'
import { setCloudConnection, setLocalPort } from './connection'
import {
  CpApiError,
  createCpClient,
  isInstalledPackage,
  type FetchLike,
  type InstallJob,
  type PackageInfo
} from './cpClient'
import { readInstalledAgents } from './agentfield'

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
  afterEach(() => setLocalPort(8080))
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
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets?include=env', 'GET'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets', 'PUT'],
      ['http://cp/api/ui/v1/agents/agent%2Fname/secrets/TOKEN?scope=node', 'DELETE'],
      ['http://cp/api/ui/v1/secrets', 'GET']
    ])
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      source: 'https://github.com/acme/agent',
      force: true
    })
    expect(calls[4][1]?.body).toBeUndefined()
    expect(JSON.parse(String(calls[6][1]?.body))).toEqual({ port: 9000, detach: false })
  })

  it('sends an update source only when one is provided', async () => {
    const fetchImpl = mockFetch([json({ job_id: 'with-source' }), json({ job_id: 'legacy' })])
    const client = createCpClient({ fetchImpl })

    await expect(
      client.updatePackage('agent', { source: 'https://github.com/Agent-Field/SWE-AF' })
    ).resolves.toEqual({ job_id: 'with-source' })
    await expect(client.updatePackage('agent')).resolves.toEqual({ job_id: 'legacy' })

    const calls = vi.mocked(fetchImpl).mock.calls
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      source: 'https://github.com/Agent-Field/SWE-AF'
    })
    expect(calls[1][1]?.body).toBeUndefined()
  })

  it('D7 — sends force and preserves the active-executions conflict contract', async () => {
    const fetchImpl = mockFetch([
      json({
        error: '1 run is active',
        code: 'executions_active',
        active_executions: 1
      }, 409),
      json({ job_id: 'forced' }, 202)
    ])
    const client = createCpClient({ fetchImpl })

    await expect(client.updatePackage('agent')).rejects.toMatchObject({
      status: 409,
      code: 'executions_active',
      activeExecutions: 1
    })
    await expect(client.updatePackage('agent', { force: true })).resolves.toEqual({
      job_id: 'forced'
    })
    expect(JSON.parse(String(vi.mocked(fetchImpl).mock.calls[1][1]?.body))).toEqual({
      force: true
    })
  })

  it('implements the version and package-maintenance API contracts', async () => {
    const version = {
      version: '0.1.135',
      commit: 'abc123',
      build_date: '2026-08-24T00:00:00Z',
      hosting: { platform: 'railway' },
      features: ['package_updates']
    }
    const check = { checked_at: 'now', packages: [] }
    const maintenance = {
      enabled: true,
      reason: '',
      interval: '6h0m0s',
      boot_pass_completed: false,
      hosting: 'railway',
      last_run: null,
      next_run_at: 'later'
    }
    const fetchImpl = mockFetch([
      json(version),
      json(check),
      json({ id: 'pkg', name: 'pkg', auto_update: false }),
      json(maintenance),
      json({ started: true }, 202)
    ])
    const client = createCpClient({ baseUrl: () => 'http://cp', fetchImpl })

    await expect(client.getVersion()).resolves.toEqual(version)
    await expect(client.checkPackageUpdates()).resolves.toEqual(check)
    await expect(client.setPackageAutoUpdate('pkg/name', false)).resolves.toMatchObject({
      auto_update: false
    })
    await expect(client.getMaintenanceStatus()).resolves.toEqual(maintenance)
    await expect(client.runPackageMaintenance()).resolves.toEqual({ started: true })

    const calls = vi.mocked(fetchImpl).mock.calls
    expect(calls.map(([url, init]) => [url, init?.method ?? 'GET'])).toEqual([
      ['http://cp/api/v1/version', 'GET'],
      ['http://cp/api/ui/v1/agents/packages/check-updates', 'POST'],
      ['http://cp/api/ui/v1/agents/packages/pkg%2Fname/auto-update', 'PUT'],
      ['http://cp/api/ui/v1/agents/packages/maintenance', 'GET'],
      ['http://cp/api/ui/v1/agents/packages/maintenance/run', 'POST']
    ])
    expect(JSON.parse(String(calls[2][1]?.body))).toEqual({ enabled: false })
  })

  it('gives update checks and maintenance runs a 120-second request budget', async () => {
    const originalTimeout = AbortSignal.timeout.bind(AbortSignal)
    const timeout = vi
      .spyOn(AbortSignal, 'timeout')
      .mockImplementation((milliseconds) => originalTimeout(milliseconds))
    const client = createCpClient({
      fetchImpl: mockFetch([
        json({ checked_at: 'now', packages: [] }),
        json({ started: true }, 202)
      ])
    })

    await client.checkPackageUpdates()
    await client.runPackageMaintenance()

    expect(timeout.mock.calls.map(([milliseconds]) => milliseconds)).toEqual([
      120_000,
      120_000
    ])
    timeout.mockRestore()
  })

  it('gives the remote control-plane version read a 10-second request budget', async () => {
    const originalTimeout = AbortSignal.timeout.bind(AbortSignal)
    const timeout = vi
      .spyOn(AbortSignal, 'timeout')
      .mockImplementation((milliseconds) => originalTimeout(milliseconds))
    const client = createCpClient({
      fetchImpl: mockFetch([json({
        version: '0.1.135',
        commit: '',
        build_date: '',
        hosting: { platform: 'railway' },
        features: []
      })])
    })

    await client.getVersion()

    expect(timeout).toHaveBeenCalledWith(10_000)
    timeout.mockRestore()
  })

  it('maps only a 404 version endpoint to legacy null', async () => {
    await expect(createCpClient({
      fetchImpl: mockFetch([json({ error: 'missing' }, 404)])
    }).getVersion()).resolves.toBeNull()
    await expect(createCpClient({
      fetchImpl: mockFetch([json({ error: 'denied' }, 401)])
    }).getVersion()).rejects.toMatchObject({ status: 401 })
  })

  it('normalizes Go nil slices in list responses', async () => {
    const fetchImpl = mockFetch([
      json({ packages: null, total: 0 }),
      json(null),
      json({ running_agents: null, total_count: 0 }),
      json({ secrets: null }),
      json({ secrets: null })
    ])
    const client = createCpClient({ fetchImpl })

    await expect(client.listPackages()).resolves.toEqual({ packages: [], total: 0 })
    await expect(client.listInstallJobs()).resolves.toEqual([])
    await expect(client.listRunningAgents()).resolves.toEqual({
      running_agents: [],
      total_count: 0
    })
    await expect(client.listAgentSecrets('agent')).resolves.toEqual({ secrets: [] })
    await expect(client.listAllSecrets()).resolves.toEqual({ secrets: [] })
  })

  it('maps a null packages wire value to an empty installed-agent registry', async () => {
    const client = createCpClient({
      fetchImpl: mockFetch([json({ packages: null, total: 0 })])
    })

    await expect(readInstalledAgents(client)).resolves.toEqual({ exists: true, agents: [] })
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

  it('uses the default connection-state API key provider', async () => {
    const fetchImpl = mockFetch([json({ packages: [], total: 0 })])
    setCloudConnection('https://cp.example', 'state-key')
    await createCpClient({ fetchImpl }).listPackages()
    const [url, init] = vi.mocked(fetchImpl).mock.calls[0]
    expect(url).toBe('https://cp.example/api/ui/v1/agents/packages')
    expect(new Headers(init?.headers).get('X-API-Key')).toBe('state-key')
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

  // Contract 4a: a 401 from the privileged-route guard reaches the user as the
  // server's sentence plus its CLI hint — never the bare "unauthorized" code.
  it('surfaces the server message and CLI hint for a 401', async () => {
    const client = createCpClient({
      fetchImpl: mockFetch([
        json(
          {
            error: 'unauthorized',
            message:
              'this endpoint installs packages and manages credentials, so it is restricted to local callers while the control plane runs without authentication.',
            help: {
              enable_auth: 'set AGENTFIELD_API_KEY on the control plane',
              cli: 'af auth login --server <control-plane-url>'
            }
          },
          401
        )
      ])
    })
    const error = await client.installPackage('https://github.com/acme/agent').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(CpApiError)
    expect((error as CpApiError).status).toBe(401)
    expect((error as CpApiError).message).toBe(
      'this endpoint installs packages and manages credentials, so it is restricted to local callers while the control plane runs without authentication. Run: af auth login --server <control-plane-url>'
    )
  })

  it('still explains a 401 with no usable body', async () => {
    const client = createCpClient({ fetchImpl: mockFetch([new Response('', { status: 401 })]) })
    const error = await client.listPackages().catch((e: unknown) => e)
    expect(error).toMatchObject({
      status: 401,
      message: 'The control plane rejected this request: it requires an API key.'
    })
  })

  it('keeps a human-readable 401 error field when no message is sent', async () => {
    const client = createCpClient({ fetchImpl: mockFetch([json({ error: 'API key rejected' }, 401)]) })
    const error = await client.listPackages().catch((e: unknown) => e)
    expect(error).toMatchObject({ status: 401, message: 'API key rejected' })
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

  it('normalizes null job lines while watching', async () => {
    const emitted: string[] = []
    const client = createCpClient({
      fetchImpl: mockFetch([json({ ...job('succeeded'), lines: null })])
    })

    const result = await client.watchInstallJob('job/1', (line) => emitted.push(line))

    expect(result.lines).toEqual([])
    expect(emitted).toEqual([])
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
