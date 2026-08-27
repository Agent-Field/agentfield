import { describe, expect, it } from 'vitest'
import { installedSourceLabel, sameSourceRepo, sourceRepo } from './catalog'

// Contract C6: the installed card flags an install as coming from somewhere
// else only when the RECORDED source names a different repository than the
// catalog row — never because the registry kept a `//subdir` or `@ref` that the
// catalog row does not carry.
describe('sourceRepo', () => {
  it('reduces a source to its owner/repo identity', () => {
    expect(sourceRepo('https://github.com/Agent-Field/SWE-AF')).toBe('agent-field/swe-af')
    expect(sourceRepo('https://github.com/Agent-Field/SWE-AF//go')).toBe('agent-field/swe-af')
    expect(sourceRepo('https://github.com/Agent-Field/SWE-AF@main//go')).toBe('agent-field/swe-af')
    expect(sourceRepo('https://github.com/Agent-Field/SWE-AF.git/')).toBe('agent-field/swe-af')
    expect(sourceRepo('Agent-Field/SWE-AF@v1.2.3')).toBe('agent-field/swe-af')
  })

  it('is empty for a blank source', () => {
    expect(sourceRepo('')).toBe('')
    expect(sourceRepo('   ')).toBe('')
  })
})

describe('sameSourceRepo', () => {
  const catalog = 'https://github.com/Agent-Field/SWE-AF'

  it('treats the redirect target of a catalog install as the same repo', () => {
    expect(sameSourceRepo('https://github.com/Agent-Field/SWE-AF//go', catalog)).toBe(true)
    expect(sameSourceRepo('https://github.com/agent-field/swe-af@main', catalog)).toBe(true)
    expect(sameSourceRepo(catalog, catalog)).toBe(true)
  })

  it('flags an install recorded from a different repository', () => {
    expect(sameSourceRepo('https://github.com/AbirAbbas/swe-af-furrow-e2e//go', catalog)).toBe(
      false
    )
    expect(sameSourceRepo('https://github.com/Agent-Field/SWE-AF-fork', catalog)).toBe(false)
  })

  it('never matches an unknown origin', () => {
    expect(sameSourceRepo('', catalog)).toBe(false)
    expect(sameSourceRepo('', '')).toBe(false)
  })
})

describe('installedSourceLabel', () => {
  const catalog = 'https://github.com/Agent-Field/SWE-AF'

  it('names the recorded repository when it differs from the catalog row', () => {
    expect(
      installedSourceLabel('https://github.com/AbirAbbas/swe-af-furrow-e2e//go', catalog)
    ).toBe('AbirAbbas/swe-af-furrow-e2e//go')
  })

  it('is silent for a catalog install, including its redirect target', () => {
    expect(installedSourceLabel(catalog, catalog)).toBeNull()
    expect(installedSourceLabel('https://github.com/Agent-Field/SWE-AF//go', catalog)).toBeNull()
  })

  it('is silent when the control plane reported no source', () => {
    expect(installedSourceLabel(undefined, catalog)).toBeNull()
    expect(installedSourceLabel('  ', catalog)).toBeNull()
  })
})
