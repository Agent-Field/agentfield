import { describe, expect, it } from 'vitest'
import {
  type ProbedCandidate,
  cliCandidates,
  compareVersions,
  effectiveMinVersion,
  parseAfVersion,
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
