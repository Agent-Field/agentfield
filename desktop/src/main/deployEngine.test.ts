import { EventEmitter } from 'node:events'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { PassThrough } from 'node:stream'
import { delimiter, join } from 'node:path'
import { tmpdir } from 'node:os'
import { mkdtempSync } from 'node:fs'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import {
  generateApiKey,
  hasDeployment,
  refreshDeploymentState,
  resolveCloudImage,
  resolveTofuBinary,
  runDeploy,
  runDestroy
} from './deployEngine'

type Script = { stdout?: string; stderr?: string; code?: number }

function harness(scripts: Script[], fetchImpl: typeof fetch = vi.fn(async () => new Response(JSON.stringify({
  data: { project: { volumes: { edges: [{ node: { id: 'volume', name: 'data' } }] } } }
}), { status: 200, headers: { 'Content-Type': 'application/json' } }))) {
  const calls: Array<{ args: string[]; env: NodeJS.ProcessEnv }> = []
  const spawnImpl = vi.fn((...raw: unknown[]) => {
    const args = raw[1] as string[]
    const options = raw[2] as { env: NodeJS.ProcessEnv }
    calls.push({ args, env: options.env })
    const script = scripts.shift() ?? {}
    const child = new EventEmitter() as EventEmitter & { stdout: PassThrough; stderr: PassThrough }
    child.stdout = new PassThrough()
    child.stderr = new PassThrough()
    queueMicrotask(() => {
      if (script.stdout) child.stdout.write(script.stdout)
      if (script.stderr) child.stderr.write(script.stderr)
      child.stdout.end()
      child.stderr.end()
      child.emit('close', script.code ?? 0)
    })
    return child
  }) as unknown as typeof import('node:child_process').spawn
  return { calls, deps: { spawnImpl, fetchImpl }, remaining: scripts }
}

function outputs(overrides: Record<string, unknown> = {}) {
  return JSON.stringify({
    url: { value: 'https://cp.test' },
    api_key: { value: 'key' },
    project_id: { value: 'project' },
    environment_id: { value: 'environment' },
    service_id: { value: 'service' },
    furrow_domain: { value: 'furrow.proxy.test' },
    furrow_port: { value: 12345 },
    ...overrides
  })
}

function workspace(mirror = false) {
  const root = mkdtempSync(join(tmpdir(), 'deploy-engine-test-'))
  const binaryDir = join(root, 'bin')
  mkdirSync(binaryDir, { recursive: true })
  writeFileSync(join(binaryDir, process.platform === 'win32' ? 'tofu.exe' : 'tofu'), '')
  if (mirror) mkdirSync(join(binaryDir, 'providers'), { recursive: true })
  return { root, binaryDir, opts: { railwayToken: 'token', workspaceId: 'workspace', workspaceDir: join(root, 'work'), binaryDir } }
}

function deployedState(
  apiKey = 'prior-key',
  subdomain = 'agentfield-dead',
  sourceImage?: string,
  workspaceId?: string
) {
  return JSON.stringify({
    resources: [
      { type: 'railway_service_domain', name: 'cp', instances: [{ attributes: { subdomain } }] },
      ...(workspaceId ? [{ type: 'railway_project', name: 'cp', instances: [{ attributes: { workspace_id: workspaceId } }] }] : []),
      ...(sourceImage ? [{ type: 'railway_service', name: 'cp', instances: [{ attributes: { source_image: sourceImage } }] }] : [])
    ],
    outputs: { api_key: { value: apiKey } }
  })
}

describe('deployment module and execution', () => {
  // Workspace sync is an extra. A control plane that is up and reachable is a
  // successful deploy whether or not a furrow address came back with it, so a
  // missing proxy output must never turn into a failed deployment.
  it('still succeeds when the deployment reports no furrow address', async () => {
    const plain = workspace()
    const fake = harness([
      {},
      { stdout: '{"type":"apply_complete","@message":"Apply complete"}\n' },
      { stdout: outputs({ furrow_domain: undefined, furrow_port: undefined }) }
    ])
    const result = await runDeploy(plain.opts, fake.deps)
    expect(result).toMatchObject({ ok: true, url: 'https://cp.test', apiKey: 'key' })
    expect((result as { furrowAddress?: string }).furrowAddress).toBeUndefined()
  })

  // The Railway provider hands back the proxy domain as an absolute FQDN on
  // create ("altaria.proxy.rlwy.net.") but without the trailing dot on refresh.
  // Interpolating it raw made the very next deploy rewrite FURROW_PUBLIC_ADDR,
  // and a changed service variable restarts the control plane — so a re-deploy
  // that should have been a no-op bounced the server. Normalising both places
  // the domain is read keeps the published address identical across applies.
  it('normalises the proxy domain so redeploys do not rewrite the address', async () => {
    const plain = workspace()
    const fake = harness([
      {},
      { stdout: '{"type":"apply_complete","@message":"Apply complete"}\n' },
      { stdout: outputs({}) }
    ])
    await runDeploy(plain.opts, fake.deps)
    const module = readFileSync(join(plain.opts.workspaceDir, 'main.tf'), 'utf8')
    for (const line of module.split('\n')) {
      if (!line.includes('railway_tcp_proxy.furrow.domain')) continue
      expect(line).toContain('trimsuffix(railway_tcp_proxy.furrow.domain, ".")')
    }
    expect(module).toMatch(/railway_tcp_proxy\.furrow\.domain/)
  })

  it('writes the module and a CLI mirror config only when a mirror exists', async () => {
    const withMirror = workspace(true)
    const fake = harness([
      {},
      { stdout: '{"type":"apply_complete","@message":"Apply complete"}\n' },
      { stdout: outputs() }
    ])
    const result = await runDeploy(withMirror.opts, fake.deps)
    expect(result).toMatchObject({ ok: true, url: 'https://cp.test', apiKey: 'key', furrowAddress: 'furrow.proxy.test:12345' })
    const module = readFileSync(join(withMirror.opts.workspaceDir, 'main.tf'), 'utf8')
    expect(module).toContain('resource "railway_project" "cp"')
    expect(module).toContain('workspace_id = var.workspace_id')
    expect(module).toContain('source_image = var.image')
    expect(module).toContain('default = "agentfield/control-plane-cloud:latest"')
    expect(module).toContain('ignore_changes = [volume]')
    // The default harness fetch is not Docker Hub shaped, so the resolver
    // falls back to the floating tag.
    expect(fake.calls[0].env.TF_VAR_image).toBe('agentfield/control-plane-cloud:latest')
    expect(module).not.toMatch(/\bvolume\s*=/)
    expect(module).toMatch(/resource "railway_tcp_proxy" "furrow" \{[\s\S]*?application_port = 8802[\s\S]*?environment_id\s*= railway_project\.cp\.default_environment\.id[\s\S]*?service_id\s*= railway_service\.cp\.id[\s\S]*?\}/)
    expect(module).toMatch(/resource "railway_variable" "furrow_public_addr" \{[\s\S]*?name\s*= "FURROW_PUBLIC_ADDR"[\s\S]*?value\s*= "\$\{trimsuffix\(railway_tcp_proxy\.furrow\.domain, "\."\)\}:\$\{railway_tcp_proxy\.furrow\.proxy_port\}"[\s\S]*?\}/)
    expect(module).toContain('output "project_id"')
    expect(module).toContain('output "environment_id"')
    expect(module).toContain('output "service_id"')
    expect(readFileSync(join(withMirror.opts.workspaceDir, 'deploy.tfrc'), 'utf8')).toContain('filesystem_mirror')
    expect(fake.calls[0].env.TF_CLI_CONFIG_FILE).toContain('deploy.tfrc')

    const withoutMirror = workspace()
    const other = harness([{}, {}, { stdout: outputs({ url: { value: 'u' }, api_key: { value: 'k' } }) }])
    await runDeploy(withoutMirror.opts, other.deps)
    expect(() => readFileSync(join(withoutMirror.opts.workspaceDir, 'deploy.tfrc'))).toThrow()
    expect(other.calls[0].env.TF_CLI_CONFIG_FILE).toBeUndefined()
  })

  it('forwards supported NDJSON in order and skips malformed lines', async () => {
    const fixture = workspace()
    const lines: string[] = []
    const fake = harness([{}, {
      stdout: [
        'not json',
        '{"type":"apply_progress","@message":"Creating"}',
        '{"type":"other","@message":"hidden"}',
        '{"type":"apply_complete","@message":"Created"}'
      ].join('\n') + '\n'
    }, { stdout: outputs({ url: { value: 'https://x' }, api_key: { value: 'k' } }) }])
    expect((await runDeploy({ ...fixture.opts, onLine: (line) => lines.push(line) }, fake.deps)).ok).toBe(true)
    expect(lines).toEqual(['Deploying agentfield/control-plane-cloud:latest', 'Creating', 'Created', 'Attaching storage volume…', 'Storage volume ready'])
  })

  it('surfaces diagnostic summaries and preserves state for reconciliation', async () => {
    const fixture = workspace()
    const fake = harness([{}, {
      stdout: '{"type":"diagnostic","diagnostic":{"severity":"error","summary":"domain unavailable","detail":"choose another"}}\n', code: 1
    }])
    const result = await runDeploy(fixture.opts, fake.deps)
    expect(result).toEqual({ ok: false, message: 'domain unavailable. State was kept; re-run deploy to reconcile.' })
  })

  it('rejects missing output values', async () => {
    const fixture = workspace()
    const fake = harness([{}, {}, { stdout: '{"url":{"value":"https://x"}}' }])
    expect(await runDeploy(fixture.opts, fake.deps)).toMatchObject({ ok: false, message: expect.stringContaining('outputs are missing') })
  })

  it('generates a fresh 48-hex key and reuses state credentials and subdomain', async () => {
    expect(generateApiKey()).toMatch(/^[a-f0-9]{48}$/)
    const fresh = workspace()
    const first = harness([{}, {}, { stdout: outputs({ url: { value: 'u' }, api_key: { value: 'returned' } }) }])
    expect(await runDeploy(fresh.opts, first.deps)).toMatchObject({
      ok: true,
      firstDeploy: true
    })
    expect(first.calls[0].env.TF_VAR_api_key).toMatch(/^[a-f0-9]{48}$/)
    expect(first.calls[0].env.TF_VAR_subdomain).toMatch(/^agentfield-[a-f0-9]{4}$/)

    const existing = workspace()
    mkdirSync(existing.opts.workspaceDir, { recursive: true })
    writeFileSync(
      join(existing.opts.workspaceDir, 'terraform.tfstate'),
      deployedState('prior-key', 'agentfield-dead', undefined, 'state-workspace')
    )
    const again = harness([{}, {}, { stdout: outputs({ url: { value: 'u' }, api_key: { value: 'prior-key' } }) }])
    expect(await runDeploy(existing.opts, again.deps)).toMatchObject({
      ok: true,
      firstDeploy: false
    })
    expect(again.calls[0].env.TF_VAR_api_key).toBe('prior-key')
    expect(again.calls[0].env.TF_VAR_subdomain).toBe('agentfield-dead')
    expect(again.calls[0].env.TF_VAR_workspace_id).toBe('state-workspace')
    expect(hasDeployment(existing.opts.workspaceDir)).toBe(true)
  })

  it('creates a missing volume with the deployment IDs and Railway-compatible headers', async () => {
    const fixture = workspace()
    const lines: string[] = []
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ results: [
        { name: 'latest', digest: 'sha256:abc' },
        { name: 'v0.1.124', digest: 'sha256:abc' }
      ] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { project: { volumes: { edges: [] } } } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { volumeCreate: { id: 'volume', name: 'data' } } }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)

    expect(await runDeploy({ ...fixture.opts, onLine: (line) => lines.push(line) }, fake.deps)).toEqual({
      ok: true,
      firstDeploy: true,
      url: 'https://cp.test',
      apiKey: 'key',
      furrowAddress: 'furrow.proxy.test:12345',
      projectId: 'project',
      environmentId: 'environment',
      serviceId: 'service',
      message: 'AgentField deployed to Railway.'
    })
    expect(fetchImpl).toHaveBeenCalledTimes(3)
    expect(fake.calls[0].env.TF_VAR_image).toBe('agentfield/control-plane-cloud:v0.1.124')
    const requests = ((fetchImpl as ReturnType<typeof vi.fn>).mock.calls as unknown as Array<[string, RequestInit]>).slice(1)
    for (const [url, init] of requests) {
      expect(url).toBe('https://backboard.railway.com/graphql/v2')
      expect(init.headers).toMatchObject({ Authorization: 'Bearer token', 'Content-Type': 'application/json', 'User-Agent': 'agentfield-desktop' })
    }
    expect(JSON.parse(String(requests[0][1].body))).toMatchObject({ variables: { projectId: 'project' } })
    expect(JSON.parse(String(requests[1][1].body))).toMatchObject({
      variables: { projectId: 'project', environmentId: 'environment', serviceId: 'service', mountPath: '/data' }
    })
    expect(JSON.parse(String(requests[1][1].body)).query).toContain('mutation VolumeCreate')
    expect(lines).toEqual(['Deploying agentfield/control-plane-cloud:v0.1.124', 'Attaching storage volume…', 'Storage volume ready'])
  })

  it('does not create a volume when one already exists', async () => {
    const fixture = workspace()
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({
      data: { project: { volumes: { edges: [{ node: { id: 'existing', name: 'data' } }] } } }
    }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)
    expect((await runDeploy(fixture.opts, fake.deps)).ok).toBe(true)
    // First call resolves the image from Docker Hub; the volume query is the only other one.
    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(JSON.parse(String((fetchImpl as ReturnType<typeof vi.fn>).mock.calls[1][1].body)).query).not.toContain('mutation VolumeCreate')
  })

  it('keeps parsed outputs internal and reports a retryable volume creation failure', async () => {
    const fixture = workspace()
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ results: [] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { project: { volumes: { edges: [] } } } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ errors: [{ message: 'volume denied' }] }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)
    expect(await runDeploy(fixture.opts, fake.deps)).toEqual({
      ok: false,
      message: 'Deployed, but attaching the storage volume failed: volume denied. Re-run deploy to retry.'
    })
  })
})

describe('cloud image updates', () => {
  it('refreshes tfstate before apply and uses the workspace recorded in state', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), JSON.stringify({
      resources: [
        { type: 'railway_project', name: 'cp', instances: [{ attributes: { workspace_id: 'state-workspace' } }] },
        { type: 'railway_service_domain', name: 'cp', instances: [{ attributes: { subdomain: 'agentfield-test' } }] },
        { type: 'railway_service', name: 'cp', instances: [{ attributes: { source_image: 'agentfield/control-plane-cloud:v0.1.134' } }] }
      ],
      outputs: { api_key: { value: 'prior-key' } }
    }))
    const fake = harness([{}, {}])

    await expect(refreshDeploymentState(fixture.opts, fake.deps)).resolves.toEqual({
      ok: true,
      message: 'Railway deployment state refreshed.'
    })
    expect(fake.calls.map((call) => call.args)).toEqual([
      ['init', '-input=false'],
      ['refresh', '-input=false']
    ])
    expect(fake.calls[1].env.TF_VAR_workspace_id).toBe('state-workspace')
    expect(fake.calls[1].env.TF_VAR_image).toBe('agentfield/control-plane-cloud:v0.1.134')
  })

})

describe('cloud image resolution', () => {
  const hub = (results: Array<{ name: string; digest?: string }>, status = 200) =>
    vi.fn(async () => new Response(JSON.stringify({ results }), { status })) as typeof fetch

  it('pins the release tag sharing a digest with latest, ignoring staging tags', async () => {
    const fetchImpl = hub([
      { name: 'latest', digest: 'sha256:abc' },
      { name: 'staging-0.1.125-rc.1', digest: 'sha256:abc' },
      { name: 'v0.1.124', digest: 'sha256:abc' },
      { name: 'v0.1.123', digest: 'sha256:old' }
    ])
    expect(await resolveCloudImage(fetchImpl)).toBe('agentfield/control-plane-cloud:v0.1.124')
    expect((fetchImpl as ReturnType<typeof vi.fn>).mock.calls[0][0]).toContain('hub.docker.com/v2/repositories/agentfield/control-plane-cloud/tags')
  })

  it('returns null when the lookup fails or nothing matches', async () => {
    expect(await resolveCloudImage(vi.fn(async () => { throw new Error('offline') }) as unknown as typeof fetch)).toBeNull()
    expect(await resolveCloudImage(hub([], 500))).toBeNull()
    expect(await resolveCloudImage(hub([{ name: 'v0.1.124', digest: 'sha256:abc' }]))).toBeNull()
    expect(await resolveCloudImage(hub([{ name: 'latest', digest: 'sha256:abc' }, { name: 'staging-0.1.124-rc.9', digest: 'sha256:abc' }]))).toBeNull()
  })

  it('a failed lookup keeps an existing deployment on its recorded pin instead of de-pinning to latest', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), deployedState('prior-key', 'agentfield-dead', 'agentfield/control-plane-cloud:v0.1.124'))
    const offline = harness([{}, {}, { stdout: outputs() }], vi.fn(async () => { throw new Error('offline') }) as unknown as typeof fetch)
    // The volume step also uses the failing fetch, so the deploy reports the
    // retryable volume error — the apply itself, and its pin, still ran.
    await runDeploy(fixture.opts, offline.deps)
    expect(offline.calls[0].env.TF_VAR_image).toBe('agentfield/control-plane-cloud:v0.1.124')

    // A successful lookup still upgrades the same deployment.
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), deployedState('prior-key', 'agentfield-dead', 'agentfield/control-plane-cloud:v0.1.124'))
    const online = harness([{}, {}, { stdout: outputs() }], vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ results: [
        { name: 'latest', digest: 'sha256:new' },
        { name: 'v0.1.125', digest: 'sha256:new' }
      ] }), { status: 200 }))
      .mockResolvedValue(new Response(JSON.stringify({
        data: { project: { volumes: { edges: [{ node: { id: 'volume', name: 'data' } }] } } }
      }), { status: 200 })) as typeof fetch)
    expect((await runDeploy(fixture.opts, online.deps)).ok).toBe(true)
    expect(online.calls[0].env.TF_VAR_image).toBe('agentfield/control-plane-cloud:v0.1.125')
  })

  it('an explicit image override beats both Docker Hub and the state pin', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(
      join(fixture.opts.workspaceDir, 'terraform.tfstate'),
      deployedState(
        'prior-key',
        'agentfield-dead',
        'agentfield/control-plane-cloud:v0.1.120',
        'state-workspace'
      )
    )
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({
      data: { project: { volumes: { edges: [{ node: { id: 'volume', name: 'data' } }] } } }
    }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)

    await expect(runDeploy({
      ...fixture.opts,
      image: 'agentfield/control-plane-cloud:v0.1.140'
    }, fake.deps)).resolves.toMatchObject({ ok: true, firstDeploy: false })
    expect(fake.calls[0].env.TF_VAR_image).toBe(
      'agentfield/control-plane-cloud:v0.1.140'
    )
    // The only fetch is the volume lookup: explicit image selection skipped Hub.
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })
})

describe('binary resolution', () => {
  const originalPath = process.env.PATH
  const bundled = join(process.cwd(), 'vendor', 'deploy-engine', process.platform === 'win32' ? 'tofu.exe' : 'tofu')
  const parked = `${bundled}.resolution-test`
  beforeAll(() => { if (existsSync(bundled)) renameSync(bundled, parked) })
  afterAll(() => { if (existsSync(parked)) renameSync(parked, bundled) })
  afterEach(() => { process.env.PATH = originalPath })

  it('prefers the override and then tofu over terraform on PATH', () => {
    const override = workspace().binaryDir
    const pathDir = mkdtempSync(join(tmpdir(), 'deploy-path-'))
    const suffix = process.platform === 'win32' ? '.exe' : ''
    writeFileSync(join(pathDir, `tofu${suffix}`), '')
    writeFileSync(join(pathDir, `terraform${suffix}`), '')
    process.env.PATH = [pathDir, originalPath].filter(Boolean).join(delimiter)
    expect(resolveTofuBinary(override)).toBe(join(override, `tofu${suffix}`))
    expect(resolveTofuBinary('/does/not/exist')).toBe(join(pathDir, `tofu${suffix}`))
  })

  it('uses terraform as the final fallback and returns null when PATH is empty', () => {
    const pathDir = mkdtempSync(join(tmpdir(), 'deploy-path-'))
    const suffix = process.platform === 'win32' ? '.exe' : ''
    writeFileSync(join(pathDir, `terraform${suffix}`), '')
    process.env.PATH = pathDir
    expect(resolveTofuBinary('/does/not/exist')).toBe(join(pathDir, `terraform${suffix}`))
    process.env.PATH = ''
    expect(resolveTofuBinary('/does/not/exist')).toBeNull()
  })

  it('uses .exe names on Windows', () => {
    const platform = vi.spyOn(process, 'platform', 'get').mockReturnValue('win32')
    const binaryDir = mkdtempSync(join(tmpdir(), 'deploy-windows-'))
    writeFileSync(join(binaryDir, 'tofu.exe'), '')
    expect(resolveTofuBinary(binaryDir)).toBe(join(binaryDir, 'tofu.exe'))
    platform.mockRestore()
  })
})

describe('destroy', () => {
  it('streams a successful destroy and reports failure diagnostics', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), deployedState())
    const lines: string[] = []
    const success = harness([{ stdout: '{"type":"apply_complete","@message":"Destroyed"}\n' }])
    expect(await runDestroy({ ...fixture.opts, onLine: (line) => lines.push(line) }, success.deps)).toEqual({ ok: true, message: 'Railway deployment destroyed.' })
    expect(lines).toEqual(['Destroyed'])
    expect(success.calls[0].args).toEqual(['destroy', '-auto-approve', '-input=false', '-json'])

    const failed = harness([{ stdout: '{"type":"diagnostic","diagnostic":{"severity":"error","summary":"delete denied"}}\n', code: 1 }])
    expect(await runDestroy(fixture.opts, failed.deps)).toEqual({ ok: false, message: 'delete denied' })
  })

  it('recovers the workspace id from state — tear-down has no workspace picker', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), JSON.stringify({
      resources: [
        { type: 'railway_project', name: 'cp', instances: [{ attributes: { workspace_id: 'ws-from-state' } }] },
        { type: 'railway_service_domain', name: 'cp', instances: [{ attributes: { subdomain: 'agentfield-dead' } }] }
      ],
      outputs: { api_key: { value: 'prior-key' } }
    }))
    const fake = harness([{ stdout: '{"type":"apply_complete","@message":"Destroyed"}\n' }])
    expect(await runDestroy({ ...fixture.opts, workspaceId: '' }, fake.deps)).toEqual({
      ok: true,
      message: 'Railway deployment destroyed.'
    })
    expect(fake.calls[0].env.TF_VAR_workspace_id).toBe('ws-from-state')
  })

  it('refuses with a clear message when no workspace id is known', async () => {
    const fixture = workspace()
    mkdirSync(fixture.opts.workspaceDir, { recursive: true })
    writeFileSync(join(fixture.opts.workspaceDir, 'terraform.tfstate'), deployedState())
    const fake = harness([])
    expect(await runDestroy({ ...fixture.opts, workspaceId: '' }, fake.deps)).toEqual({
      ok: false,
      message: 'Could not determine the Railway workspace of this deployment — sign in and re-run deploy first.'
    })
    expect(fake.calls).toHaveLength(0)
  })
})
