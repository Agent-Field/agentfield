import { describe, expect, it } from 'vitest'
import { type ProbedCandidate, cliCandidates, compareVersions, parseAfVersion, selectCli } from './cli'

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

  it('trusts dev builds (unparseable version) as usable', () => {
    const { chosen, outdated } = selectCli(
      [probed({ source: 'path', version: null }), probed({ source: 'bundled', version: '0.1.108' })],
      MIN
    )
    expect(chosen?.source).toBe('path')
    expect(outdated).toBeNull()
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
