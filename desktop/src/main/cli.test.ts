import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  type ProbedCandidate,
  cliCandidates,
  compareVersions,
  effectiveMinVersion,
  parseAfVersion,
  posixPathEntry,
  probeCli,
  registerPosixUserPath,
  selectCli
} from './cli'

function probed(overrides: Partial<ProbedCandidate>): ProbedCandidate {
  return { command: 'af', source: 'path', responds: true, version: '0.1.107', ...overrides }
}

describe('parseAfVersion', () => {
  it('reads the Version line of `af version` output', () => {
    const output = 'AgentField Control Plane\n  Version:    v0.1.107\n  Commit:     abc123\n'
    expect(parseAfVersion(output)).toBe('0.1.107')
  })

  it('accepts versions without the v prefix', () => {
    expect(parseAfVersion('Version: 1.2.3')).toBe('1.2.3')
  })

  it('returns null for dev builds and garbage', () => {
    expect(parseAfVersion('AgentField Control Plane\n  Version:    dev\n')).toBeNull()
    expect(parseAfVersion('command not found')).toBeNull()
    expect(parseAfVersion('')).toBeNull()
  })
})

describe('compareVersions', () => {
  it('orders numerically per segment', () => {
    expect(compareVersions('0.1.107', '0.1.107')).toBe(0)
    expect(compareVersions('0.1.99', '0.1.107')).toBeLessThan(0)
    expect(compareVersions('0.2.0', '0.1.999')).toBeGreaterThan(0)
  })

  it('treats missing segments as zero', () => {
    expect(compareVersions('0.1', '0.1.0')).toBe(0)
    expect(compareVersions('1', '0.9.9')).toBeGreaterThan(0)
  })
})

describe('selectCli', () => {
  const MIN = '0.1.107'

  it('prefers the managed copy when it qualifies', () => {
    const { chosen, outdated } = selectCli(
      [
        probed({ command: 'C:\\home\\.agentfield\\bin\\af.exe', source: 'managed' }),
        probed({ command: 'af', source: 'path' }),
        probed({ command: 'bundled/af.exe', source: 'bundled', version: '0.1.108' })
      ],
      MIN
    )
    expect(chosen?.source).toBe('managed')
    expect(outdated).toBeNull()
  })

  it('skips non-responding candidates', () => {
    const { chosen } = selectCli(
      [
        probed({ source: 'managed', responds: false, version: null }),
        probed({ source: 'path', responds: false, version: null }),
        probed({ source: 'bundled', version: '0.1.108' })
      ],
      MIN
    )
    expect(chosen?.source).toBe('bundled')
  })

  it('falls through an outdated install to the bundled copy and reports it', () => {
    const { chosen, outdated } = selectCli(
      [
        probed({ source: 'managed', version: '0.1.90' }),
        probed({ source: 'path', responds: false, version: null }),
        probed({ source: 'bundled', version: '0.1.108' })
      ],
      MIN
    )
    expect(chosen?.source).toBe('bundled')
    expect(outdated?.source).toBe('managed')
    expect(outdated?.version).toBe('0.1.90')
  })

  it('trusts dev builds (unparseable version) while the bundle is unstamped too', () => {
    const { chosen, outdated } = selectCli(
      [probed({ source: 'path', version: null }), probed({ source: 'bundled', version: null })],
      MIN
    )
    expect(chosen?.source).toBe('path')
    expect(outdated).toBeNull()
  })

  it('supersedes dev-versioned copies when the bundle is stamped', () => {
    // A release app must never let a stale dev binary it once provisioned
    // win forever: against a stamped bundle, unparseable managed/PATH copies
    // are skipped so the bundle gets chosen (and then provisioned).
    const { chosen, outdated } = selectCli(
      [
        probed({ source: 'managed', version: null }),
        probed({ source: 'path', version: null }),
        probed({ source: 'bundled', version: '0.1.110' })
      ],
      MIN
    )
    expect(chosen?.source).toBe('bundled')
    expect(outdated?.source).toBe('managed')
  })

  it('never supersedes a parseable copy that meets the minimum', () => {
    const { chosen } = selectCli(
      [
        probed({ source: 'managed', version: '0.1.115' }),
        probed({ source: 'bundled', version: '0.1.110' })
      ],
      MIN
    )
    expect(chosen?.source).toBe('managed')
  })

  it('reports nothing usable when everything is dead or old with no bundle', () => {
    const { chosen, outdated } = selectCli(
      [
        probed({ source: 'managed', version: '0.1.1' }),
        probed({ source: 'path', responds: false, version: null })
      ],
      MIN
    )
    expect(chosen).toBeNull()
    expect(outdated?.version).toBe('0.1.1')
  })
})

describe('effectiveMinVersion', () => {
  it('uses the stamped bundle version as the floor', () => {
    expect(effectiveMinVersion([probed({ source: 'bundled', version: '0.1.120' })])).toBe('0.1.120')
  })

  it('falls back to MIN_AF_VERSION for unstamped or missing bundles', () => {
    expect(effectiveMinVersion([probed({ source: 'bundled', version: null })])).toBe('0.1.107')
    expect(effectiveMinVersion([probed({ source: 'path' })])).toBe('0.1.107')
    // A bundle older than the constant never lowers the floor.
    expect(effectiveMinVersion([probed({ source: 'bundled', version: '0.1.1' })])).toBe('0.1.107')
  })
})

describe('cliCandidates', () => {
  it('orders managed before PATH before bundled', () => {
    const sources = cliCandidates('/tmp/bundle/af').map((c) => c.source)
    expect(sources).toEqual(['managed', 'managed', 'path', 'bundled'])
  })

  it('omits the bundled candidate when the app has none', () => {
    const sources = cliCandidates(null).map((c) => c.source)
    expect(sources).toEqual(['managed', 'managed', 'path'])
  })
})

describe('probeCli', () => {
  it('resolves responds:false when spawn throws synchronously', async () => {
    // An empty command makes spawn() throw before any listeners attach —
    // the same shape as Windows throwing UNKNOWN for a non-PE binary on PATH.
    // probeCli must swallow it: a rejection here fails the whole probeAll and
    // the app never creates its tray or window.
    await expect(probeCli({ command: '', source: 'path' })).resolves.toMatchObject({
      responds: false,
      version: null
    })
  })
})

describe('posixPathEntry', () => {
  const HOME = '/home/u'
  const BIN = '/home/u/.agentfield/bin'
  const EXPORT = `export PATH="${BIN}:$PATH"`

  it('writes ~/.zshrc for zsh with the same export line as the curl installer', () => {
    expect(posixPathEntry('/bin/zsh', HOME, BIN, () => false)).toEqual({
      rcFile: join(HOME, '.zshrc'),
      line: EXPORT
    })
  })

  it('prefers ~/.bashrc for bash, falls back to an existing ~/.bash_profile', () => {
    const bashrc = join(HOME, '.bashrc')
    const profile = join(HOME, '.bash_profile')
    expect(posixPathEntry('/bin/bash', HOME, BIN, (p) => p === bashrc)?.rcFile).toBe(bashrc)
    expect(posixPathEntry('/bin/bash', HOME, BIN, (p) => p === profile)?.rcFile).toBe(profile)
    // Neither exists yet — .bashrc gets created, like the installer.
    expect(posixPathEntry('/bin/bash', HOME, BIN, () => false)?.rcFile).toBe(bashrc)
  })

  it('uses fish syntax in the fish config file', () => {
    expect(posixPathEntry('/usr/bin/fish', HOME, BIN, () => false)).toEqual({
      rcFile: join(HOME, '.config', 'fish', 'config.fish'),
      line: `set -gx PATH ${BIN} $PATH`
    })
  })

  it('returns null for unknown or missing shells', () => {
    expect(posixPathEntry('/bin/tcsh', HOME, BIN, () => false)).toBeNull()
    expect(posixPathEntry(undefined, HOME, BIN, () => false)).toBeNull()
  })
})

describe('registerPosixUserPath', () => {
  const freshHome = () => fs.mkdtemp(join(tmpdir(), 'af-desktop-path-'))

  it('appends a commented PATH entry to the shell startup file', async () => {
    const home = await freshHome()
    const bin = join(home, '.agentfield', 'bin')
    await registerPosixUserPath(bin, '/bin/zsh', home)
    const rc = await fs.readFile(join(home, '.zshrc'), 'utf8')
    expect(rc).toBe(`\n# AgentField CLI\nexport PATH="${bin}:$PATH"\n`)
  })

  it('is idempotent and defers to an entry the curl installer already wrote', async () => {
    const home = await freshHome()
    const bin = join(home, '.agentfield', 'bin')
    const curlEntry = `# AgentField CLI\nexport PATH="${bin}:$PATH"\n`
    await fs.writeFile(join(home, '.zshrc'), curlEntry)
    await registerPosixUserPath(bin, '/bin/zsh', home)
    await registerPosixUserPath(bin, '/bin/zsh', home)
    expect(await fs.readFile(join(home, '.zshrc'), 'utf8')).toBe(curlEntry)
  })

  it('creates the fish config directory when needed', async () => {
    const home = await freshHome()
    const bin = join(home, '.agentfield', 'bin')
    await registerPosixUserPath(bin, '/usr/bin/fish', home)
    const rc = await fs.readFile(join(home, '.config', 'fish', 'config.fish'), 'utf8')
    expect(rc).toContain(`set -gx PATH ${bin} $PATH`)
  })

  it('never rejects for unknown shells or unwritable locations', async () => {
    const home = await freshHome()
    const bin = join(home, '.agentfield', 'bin')
    await expect(registerPosixUserPath(bin, '/bin/tcsh', home)).resolves.toBeUndefined()
    // A regular file where the rc's parent dir should be → mkdir/append fails.
    await fs.writeFile(join(home, '.config'), 'not a directory')
    await expect(registerPosixUserPath(bin, '/usr/bin/fish', home)).resolves.toBeUndefined()
  })
})
