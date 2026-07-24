import { describe, expect, it } from 'vitest'
import {
  formatAbsoluteStarted,
  formatDuration,
  formatElapsed,
  formatFullStarted,
  formatRelativeStarted
} from './timeFormat'

describe('formatDuration', () => {
  it('formats sub-second, seconds, minutes, hours', () => {
    expect(formatDuration(null)).toBe('')
    expect(formatDuration(420)).toBe('420ms')
    expect(formatDuration(3200)).toBe('3.2s')
    expect(formatDuration(12_000)).toBe('12s')
    expect(formatDuration(60_000)).toBe('1m')
    expect(formatDuration(192_000)).toBe('3m 12s')
    expect(formatDuration(3_600_000)).toBe('1h')
    expect(formatDuration(3_960_000)).toBe('1h 6m')
    expect(formatDuration(172_800_000)).toBe('2d')
  })
})

describe('formatRelativeStarted', () => {
  const now = Date.parse('2026-07-22T15:00:00')

  it('uses compact ago phrases, never clock time', () => {
    expect(formatRelativeStarted(new Date(now - 2000).toISOString(), now)).toBe('just now')
    expect(formatRelativeStarted(new Date(now - 12_000).toISOString(), now)).toBe('12s ago')
    expect(formatRelativeStarted(new Date(now - 180_000).toISOString(), now)).toBe('3 mins ago')
    expect(formatRelativeStarted(new Date(now - 7_200_000).toISOString(), now)).toBe('2 hrs ago')
    expect(formatRelativeStarted(new Date(now - 86_400_000).toISOString(), now)).toBe(
      '1 day ago'
    )
    expect(formatRelativeStarted(new Date(now - 3 * 86_400_000).toISOString(), now)).toBe(
      '3 days ago'
    )
    expect(formatRelativeStarted(new Date(now - 40 * 86_400_000).toISOString(), now)).toBe(
      '1 month ago'
    )
  })
})

describe('formatAbsoluteStarted', () => {
  it('is clock-only with no date', () => {
    const label = formatAbsoluteStarted('2026-07-20T14:41:00')
    expect(label).toMatch(/\d/)
    expect(label).not.toMatch(/Jul|2026|Mon|Tue|Wed|Thu|Fri|Sat|Sun/)
  })
})

describe('formatFullStarted', () => {
  const now = Date.parse('2026-07-22T15:00:00')

  it('is clock-only for today', () => {
    const label = formatFullStarted(new Date(now - 600_000).toISOString(), now)
    expect(label).toMatch(/\d/)
    expect(label).not.toMatch(/Jul|2026/)
  })

  it('adds a short date for other days, without weekday/year/seconds', () => {
    const label = formatFullStarted(new Date(now - 86_400_000).toISOString(), now)
    expect(label).toMatch(/Jul/)
    expect(label).toMatch(/\d/)
    expect(label).not.toMatch(/\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b/)
    expect(label).not.toMatch(/2026/)
    expect(label).not.toMatch(/:\d{2}:\d{2}/)
  })
})

describe('formatElapsed', () => {
  it('returns duration since start', () => {
    const now = Date.parse('2026-07-22T15:00:00')
    const iso = new Date(now - 125_000).toISOString()
    expect(formatElapsed(iso, now)).toBe('2m 5s')
  })
})
