import { EventEmitter } from 'node:events'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { PassThrough } from 'node:stream'
import { delimiter, join } from 'node:path'
import { tmpdir } from 'node:os'
import { mkdtempSync } from 'node:fs'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { generateApiKey, hasDeployment, resolveTofuBinary, runDeploy, runDestroy } from './deployEngine'

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

function deployedState(apiKey = 'prior-key', subdomain = 'agentfield-dead') {
  return JSON.stringify({
    resources: [{ type: 'railway_service_domain', name: 'cp', instances: [{ attributes: { subdomain } }] }],
    outputs: { api_key: { value: apiKey } }
  })
}

describe('deployment module and execution', () => {
  it('writes the module and a CLI mirror config only when a mirror exists', async () => {
    const withMirror = workspace(true)
    const fake = harness([
      {},
      { stdout: '{"type":"apply_complete","@message":"Apply complete"}\n' },
      { stdout: outputs() }
    ])
    const result = await runDeploy(withMirror.opts, fake.deps)
    expect(result).toMatchObject({ ok: true, url: 'https://cp.test', apiKey: 'key' })
    const module = readFileSync(join(withMirror.opts.workspaceDir, 'main.tf'), 'utf8')
    expect(module).toContain('resource "railway_project" "cp"')
    expect(module).toContain('workspace_id = var.workspace_id')
    expect(module).toContain('source_image = "agentfield/control-plane-cloud:latest"')
    expect(module).not.toMatch(/\bvolume\s*=/)
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
    expect(lines).toEqual(['Creating', 'Created', 'Attaching storage volume…', 'Storage volume ready'])
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
    await runDeploy(fresh.opts, first.deps)
    expect(first.calls[0].env.TF_VAR_api_key).toMatch(/^[a-f0-9]{48}$/)
    expect(first.calls[0].env.TF_VAR_subdomain).toMatch(/^agentfield-[a-f0-9]{4}$/)

    const existing = workspace()
    mkdirSync(existing.opts.workspaceDir, { recursive: true })
    writeFileSync(join(existing.opts.workspaceDir, 'terraform.tfstate'), deployedState())
    const again = harness([{}, {}, { stdout: outputs({ url: { value: 'u' }, api_key: { value: 'prior-key' } }) }])
    await runDeploy(existing.opts, again.deps)
    expect(again.calls[0].env.TF_VAR_api_key).toBe('prior-key')
    expect(again.calls[0].env.TF_VAR_subdomain).toBe('agentfield-dead')
    expect(hasDeployment(existing.opts.workspaceDir)).toBe(true)
  })

  it('creates a missing volume with the deployment IDs and Railway-compatible headers', async () => {
    const fixture = workspace()
    const lines: string[] = []
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { project: { volumes: { edges: [] } } } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { volumeCreate: { id: 'volume', name: 'data' } } }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)

    expect(await runDeploy({ ...fixture.opts, onLine: (line) => lines.push(line) }, fake.deps)).toEqual({
      ok: true, url: 'https://cp.test', apiKey: 'key', message: 'AgentField deployed to Railway.'
    })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
    const requests = (fetchImpl as ReturnType<typeof vi.fn>).mock.calls as unknown as Array<[string, RequestInit]>
    for (const [url, init] of requests) {
      expect(url).toBe('https://backboard.railway.com/graphql/v2')
      expect(init.headers).toMatchObject({ Authorization: 'Bearer token', 'Content-Type': 'application/json', 'User-Agent': 'agentfield-desktop' })
    }
    expect(JSON.parse(String(requests[0][1].body))).toMatchObject({ variables: { projectId: 'project' } })
    expect(JSON.parse(String(requests[1][1].body))).toMatchObject({
      variables: { projectId: 'project', environmentId: 'environment', serviceId: 'service', mountPath: '/data' }
    })
    expect(JSON.parse(String(requests[1][1].body)).query).toContain('mutation VolumeCreate')
    expect(lines).toEqual(['Attaching storage volume…', 'Storage volume ready'])
  })

  it('does not create a volume when one already exists', async () => {
    const fixture = workspace()
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({
      data: { project: { volumes: { edges: [{ node: { id: 'existing', name: 'data' } }] } } }
    }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)
    expect((await runDeploy(fixture.opts, fake.deps)).ok).toBe(true)
    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(JSON.parse(String((fetchImpl as ReturnType<typeof vi.fn>).mock.calls[0][1].body)).query).not.toContain('mutation VolumeCreate')
  })

  it('keeps parsed outputs internal and reports a retryable volume creation failure', async () => {
    const fixture = workspace()
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { project: { volumes: { edges: [] } } } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ errors: [{ message: 'volume denied' }] }), { status: 200 })) as typeof fetch
    const fake = harness([{}, {}, { stdout: outputs() }], fetchImpl)
    expect(await runDeploy(fixture.opts, fake.deps)).toEqual({
      ok: false,
      message: 'Deployed, but attaching the storage volume failed: volume denied. Re-run deploy to retry.'
    })
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
