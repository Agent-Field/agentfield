import { describe, expect, it } from 'vitest'
import type { BundledStatus } from '../../../shared/types'
import { rosterKey, visibleBundledRows } from './AgentsPanel'

const row = (name: string, phase: BundledStatus['phase']): BundledStatus => ({
  name,
  description: `${name} description`,
  phase,
  message: ''
})

describe('visibleBundledRows', () => {
  it('keeps unmatched rows in their original order', () => {
    const bundled = [row('swe-planner', 'pending'), row('pr-af', 'installing')]
    expect(visibleBundledRows(bundled, ['other'])).toEqual(bundled)
  })

  it('drops installed and failed rows that already exist in the registry', () => {
    const bundled = [row('swe-planner', 'installed'), row('pr-af', 'failed')]
    expect(visibleBundledRows(bundled, ['swe-planner', 'pr-af'])).toEqual([])
  })

  it('keeps an installing row that is not yet in the registry', () => {
    const installing = row('pr-af', 'installing')
    expect(visibleBundledRows([installing], ['swe-planner'])).toEqual([installing])
  })

  it('returns an empty list for empty inputs', () => {
    expect(visibleBundledRows([], [])).toEqual([])
  })
})

describe('rosterKey', () => {
  it('is order-insensitive and depends only on the name set', () => {
    expect(rosterKey(['pr-af', 'swe-planner'])).toBe(rosterKey(['swe-planner', 'pr-af']))
    expect(rosterKey(['swe-planner', 'swe-planner'])).toBe(rosterKey(['swe-planner']))
    expect(rosterKey(['swe-planner'])).not.toBe(rosterKey(['pr-af']))
  })
})
