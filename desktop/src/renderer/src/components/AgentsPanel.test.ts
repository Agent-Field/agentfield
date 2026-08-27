import { describe, expect, it, vi } from 'vitest'
import type { BundledStatus, InstallResult } from '../../../shared/types'
import {
  activeExecutionsConfirmation,
  rosterKey,
  updateWithExecutionConfirmation,
  visibleBundledRows
} from './AgentsPanel'

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

describe('D7 — active execution update confirmation', () => {
  it('D7 — retries with force after confirmation', async () => {
    const request = vi
      .fn<(force: boolean) => Promise<InstallResult>>()
      .mockResolvedValueOnce({ ok: false, message: 'busy', activeExecutions: 2 })
      .mockResolvedValueOnce({ ok: true, message: 'updated' })
    const confirm = vi.fn(() => true)

    await expect(updateWithExecutionConfirmation('agent', request, confirm)).resolves.toEqual({
      ok: true,
      message: 'updated'
    })
    expect(confirm).toHaveBeenCalledWith(
      activeExecutionsConfirmation(2, 'agent')
    )
    expect(request.mock.calls).toEqual([[false], [true]])
  })

  it('D7 — does not retry when confirmation is cancelled', async () => {
    const request = vi.fn(async (): Promise<InstallResult> => ({
      ok: false,
      message: 'busy',
      activeExecutions: 1
    }))

    await updateWithExecutionConfirmation('agent', request, () => false)
    expect(request).toHaveBeenCalledTimes(1)
  })
})
