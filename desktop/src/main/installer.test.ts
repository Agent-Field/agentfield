import { describe, expect, it, vi } from 'vitest'
import { BUNDLED_NODES } from '../shared/bundled'
import { CpApiError, type CpClient } from './cpClient'
import { installAgent, installCommand, installFromSource, parseRepoSource, sanitizeInstallOutput, updateAgent } from './installer'

// Any vetted installable entry; the marketplace CATALOG is currently empty, so
// the bundled roster is where installCommand's name→source contract lives.
const ENTRY = BUNDLED_NODES[0]

describe('installCommand', () => {
  it('builds a plain install for a catalog entry', () => {
    const cmd = installCommand(ENTRY.name)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', ENTRY.source])
  })

  it('appends --force for updates (reinstall in place, secrets survive)', () => {
    const cmd = installCommand(ENTRY.name, true)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', ENTRY.source, '--force'])
  })

  it('refuses names outside the curated catalog', () => {
    expect(installCommand('rm -rf /', true)).toBeNull()
    expect(installCommand('not-in-catalog')).toBeNull()
  })
})

describe('parseRepoSource', () => {
  const accepted: Array<[string, string]> = [
    // [input, normalized output]
    ['https://github.com/Agent-Field/pr-af', 'https://github.com/Agent-Field/pr-af'],
    ['https://github.com/Agent-Field/pr-af.git', 'https://github.com/Agent-Field/pr-af.git'],
    ['https://github.com/Agent-Field/pr-af/', 'https://github.com/Agent-Field/pr-af'],
    ['  https://github.com/Agent-Field/pr-af  ', 'https://github.com/Agent-Field/pr-af'],
    ['https://github.com/Agent-Field/pr-af//go', 'https://github.com/Agent-Field/pr-af//go'],
    ['https://github.com/Agent-Field/SWE-AF//go/cmd', 'https://github.com/Agent-Field/SWE-AF//go/cmd'],
    ['https://github.com/Agent-Field/pr-af//go/', 'https://github.com/Agent-Field/pr-af//go'],
    ['https://github.com/user_1/repo.name-2', 'https://github.com/user_1/repo.name-2']
  ]
  it.each(accepted)('accepts and normalizes %s', (input, expected) => {
    expect(parseRepoSource(input)).toBe(expected)
  })

  const rejected: Array<[string, string]> = [
    ['http (not https)', 'http://github.com/Agent-Field/pr-af'],
    ['other host', 'https://gitlab.com/Agent-Field/pr-af'],
    ['ssh scp form', 'git@github.com:Agent-Field/pr-af.git'],
    ['ssh url', 'ssh://git@github.com/Agent-Field/pr-af'],
    ['leading-dash flag payload', '--force'],
    ['owner starting with dash', 'https://github.com/-evil/repo'],
    ['dotdot traversal in subdir', 'https://github.com/Agent-Field/pr-af//..%2Fetc'],
    ['literal dotdot in subdir', 'https://github.com/Agent-Field/pr-af//../secret'],
    ['embedded whitespace', 'https://github.com/Agent-Field/pr af'],
    ['empty', ''],
    ['whitespace only', '   '],
    ['query string', 'https://github.com/Agent-Field/pr-af?tab=readme'],
    ['fragment', 'https://github.com/Agent-Field/pr-af#install'],
    ['missing repo', 'https://github.com/Agent-Field'],
    ['empty subdir', 'https://github.com/Agent-Field/pr-af//'],
    ['subdir leading slash', 'https://github.com/Agent-Field/pr-af///go']
  ]
  it.each(rejected)('rejects %s', (_label, input) => {
    expect(parseRepoSource(input)).toBeNull()
  })
})

describe('installFromSource', () => {
  it('refuses a rejected source without spawning', async () => {
    const lines: string[] = []
    const result = await installFromSource('git@github.com:evil/repo.git', (line) =>
      lines.push(line)
    )
    expect(result.ok).toBe(false)
    // Never spawned af install: no progress lines were forwarded.
    expect(lines).toEqual([])
    expect(result.message).toMatch(/github\.com/)
  })

  it('installs a normalized GitHub source without force', async () => {
    const client = installClient()
    const result = await installFromSource(
      '  https://github.com/Agent-Field/pr-af/  ',
      () => {},
      { cpClient: client }
    )

    expect(client.installPackage).toHaveBeenCalledWith(
      'https://github.com/Agent-Field/pr-af',
      undefined
    )
    expect(result).toEqual({
      ok: true,
      message: 'Installed from https://github.com/Agent-Field/pr-af'
    })
  })
})

function installClient(overrides: Partial<CpClient> = {}): CpClient {
  return {
    hasInstallApi: vi.fn(async () => true),
    installPackage: vi.fn(async () => ({ job_id: 'job' })),
    updatePackage: vi.fn(async () => ({ job_id: 'job' })),
    watchInstallJob: vi.fn(async (_id, onLine) => {
      onLine('Cloning')
      onLine('Installed')
      return { id: 'job', source: '', kind: 'install', status: 'succeeded', lines: ['Cloning', 'Installed'] }
    }),
    ...overrides
  } as unknown as CpClient
}

describe('control-plane installs', () => {
  it('preserves progress callbacks and terminal success/failure', async () => {
    const lines: string[] = []
    expect(await installAgent(ENTRY.name, (line) => lines.push(line), false, { cpClient: installClient() })).toEqual({ ok: true, message: `${ENTRY.name} installed` })
    expect(lines).toEqual(['Cloning', 'Installed'])
    const failed = installClient({ watchInstallJob: vi.fn(async () => ({ id: 'job', source: '', kind: 'install' as const, status: 'failed' as const, error: 'clone failed', lines: [] })) })
    expect(await installAgent(ENTRY.name, () => {}, false, { cpClient: failed })).toEqual({ ok: false, message: 'clone failed' })
  })

  it('forwards force for reinstall-in-place updates', async () => {
    const client = installClient()
    await installAgent(ENTRY.name, () => {}, true, { cpClient: client })
    expect(client.installPackage).toHaveBeenCalledWith(ENTRY.source, true)
  })

  it('updates catalog agents from the current catalog source', async () => {
    const client = installClient()
    await updateAgent(ENTRY.name, () => {}, { cpClient: client })
    expect(client.updatePackage).toHaveBeenCalledWith(ENTRY.name, ENTRY.source)
  })

  it('surfaces conflict and old-control-plane errors without fallback', async () => {
    const conflict = installClient({ installPackage: vi.fn(async () => { throw new CpApiError({ status: 409, message: 'another install is running' }) }) })
    expect((await installAgent(ENTRY.name, () => {}, false, { cpClient: conflict })).message).toMatch(/another install/i)
    const old = installClient({ hasInstallApi: vi.fn(async () => false) })
    expect((await installAgent(ENTRY.name, () => {}, false, { cpClient: old })).message).toMatch(/update AgentField CLI/)
    expect((await updateAgent(ENTRY.name, () => {}, { cpClient: old })).message).toMatch(/update AgentField CLI/)
  })

  it('maps mid-request 404s to the update-required result', async () => {
    const missing = new CpApiError({ status: 404, message: 'not found' })
    const install = installClient({
      installPackage: vi.fn(async () => { throw missing })
    })
    expect(await installAgent(ENTRY.name, () => {}, false, { cpClient: install })).toEqual({
      ok: false,
      message: 'Control plane update required — update AgentField CLI'
    })

    const update = installClient({
      updatePackage: vi.fn(async () => { throw missing })
    })
    expect(await updateAgent(ENTRY.name, () => {}, { cpClient: update })).toEqual({
      ok: false,
      message: 'Control plane update required — update AgentField CLI'
    })
  })

  it('reports network failures during install', async () => {
    const client = installClient({
      installPackage: vi.fn(async () => { throw new TypeError('offline') })
    })
    expect(await installAgent(ENTRY.name, () => {}, false, { cpClient: client })).toEqual({
      ok: false,
      message: 'Could not reach the control plane — start the control plane and try again'
    })
  })

  it('falls back from a failed job error to its last line and then a default', async () => {
    const failedWithLine = installClient({
      watchInstallJob: vi.fn(async () => ({
        id: 'job',
        source: '',
        kind: 'install' as const,
        status: 'failed' as const,
        lines: ['clone failed']
      }))
    })
    expect(await installAgent(ENTRY.name, () => {}, false, { cpClient: failedWithLine })).toEqual({
      ok: false,
      message: 'clone failed'
    })

    const failedEmpty = installClient({
      watchInstallJob: vi.fn(async () => ({
        id: 'job',
        source: '',
        kind: 'install' as const,
        status: 'failed' as const,
        lines: []
      }))
    })
    expect(await installAgent(ENTRY.name, () => {}, false, { cpClient: failedEmpty })).toEqual({
      ok: false,
      message: 'Install failed'
    })
  })

  it('rejects unknown catalog names at the async function boundary', async () => {
    await expect(installAgent('not-in-catalog', () => {})).resolves.toEqual({
      ok: false,
      message: '"not-in-catalog" is not in the install catalog'
    })
    await expect(updateAgent('not-in-catalog', () => {})).resolves.toEqual({
      ok: false,
      message: '"not-in-catalog" is not in the install catalog'
    })
  })
})

// A manifest declaring `superseded_by:` redirects the install to a successor,
// which may register under its own name. The control plane reports what it
// actually installed as the job's package_name; the app must repeat that
// rather than the name it asked for.
describe('superseded installs report what actually landed', () => {
  const landedAs = (name: string): CpClient =>
    installClient({
      watchInstallJob: vi.fn(async () => ({
        id: 'job',
        source: '',
        kind: 'install' as const,
        status: 'succeeded' as const,
        package_name: name,
        lines: []
      }))
    })

  it('names the node a pasted repo actually installed, not the URL', async () => {
    expect(
      await installFromSource('https://github.com/Agent-Field/pr-af', () => {}, {
        cpClient: landedAs('pr-af')
      })
    ).toEqual({ ok: true, message: 'pr-af installed' })
  })

  it('falls back to the source when the control plane names nothing', async () => {
    expect(
      await installFromSource('https://github.com/Agent-Field/pr-af', () => {}, {
        cpClient: installClient()
      })
    ).toEqual({ ok: true, message: 'Installed from https://github.com/Agent-Field/pr-af' })
  })

  it('names the successor when a catalog install redirects elsewhere', async () => {
    expect(
      await installAgent(ENTRY.name, () => {}, false, { cpClient: landedAs('successor-node') })
    ).toEqual({ ok: true, message: 'successor-node installed' })
  })

  it('reports an update that renamed the node as a replacement', async () => {
    expect(
      await updateAgent(ENTRY.name, () => {}, { cpClient: landedAs('successor-node') })
    ).toEqual({ ok: true, message: `${ENTRY.name} replaced by successor-node` })
  })

  it('still reads as a plain update when the name is unchanged', async () => {
    expect(
      await updateAgent(ENTRY.name, () => {}, { cpClient: landedAs(ENTRY.name) })
    ).toEqual({ ok: true, message: `${ENTRY.name} updated` })
  })
})

describe('sanitizeInstallOutput', () => {
  it('unwraps zerolog JSON error lines to the underlying error text', () => {
    const line =
      '{"level":"error","error":"invalid package structure: no agentfield-package.yaml found for --path \\"go\\"","time":"2026-07-15T12:30:22-04:00","message":"Error executing root command"}'
    expect(sanitizeInstallOutput(line)).toEqual([
      'invalid package structure: no agentfield-package.yaml found for --path "go"'
    ])
  })

  it('falls back to the zerolog message when there is no error field', () => {
    expect(sanitizeInstallOutput('{"level":"info","message":"cloning repository"}')).toEqual([
      'cloning repository'
    ])
  })

  it('passes non-zerolog JSON and plain lines through untouched', () => {
    expect(sanitizeInstallOutput('{"result":"ok"}')).toEqual(['{"result":"ok"}'])
    expect(sanitizeInstallOutput('✅ Package installed successfully')).toEqual([
      '✅ Package installed successfully'
    ])
    expect(sanitizeInstallOutput('{not json}')).toEqual(['{not json}'])
  })
})
