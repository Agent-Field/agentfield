import { describe, expect, it } from 'vitest'
import { keysBannerMessage, unsatisfiedAgents } from './KeysBanner'
import type { AgentEnvReport } from '../../../shared/types'

const report = (over: Partial<AgentEnvReport>): AgentEnvReport => ({
  agent: 'swe-planner',
  vars: [],
  satisfied: true,
  ...over
})

describe('unsatisfiedAgents', () => {
  it('ignores agents whose required keys resolve', () => {
    expect(unsatisfiedAgents([report({}), report({ agent: 'pr-af' })])).toEqual([])
  })

  it('never counts the control-plane-unreachable sentinel', () => {
    // secrets.ts returns one nameless satisfied:false row when the whole call
    // fails — a transport error, not a missing key. App already shows the
    // "server is not running" callout for that.
    expect(unsatisfiedAgents([{ agent: '', vars: [], satisfied: false, error: 'boom' }])).toEqual(
      []
    )
    expect(unsatisfiedAgents([report({ satisfied: false, error: 'boom' })])).toEqual([])
  })

  it('lists only the blocked agents, in report order', () => {
    expect(
      unsatisfiedAgents([
        report({ agent: 'swe-planner', satisfied: false }),
        report({ agent: 'hello-world' }),
        report({ agent: 'pr-af', satisfied: false })
      ])
    ).toEqual(['swe-planner', 'pr-af'])
  })
})

describe('keysBannerMessage', () => {
  it('is silent when nothing is blocked', () => {
    expect(keysBannerMessage([])).toBeNull()
    expect(keysBannerMessage([report({})])).toBeNull()
  })

  it('names the agent when exactly one is blocked', () => {
    expect(keysBannerMessage([report({ satisfied: false })])).toBe(
      'swe-planner is installed but needs API keys before it can run.'
    )
  })

  it('counts them once naming every one would be a list, not a sentence', () => {
    expect(
      keysBannerMessage([
        report({ agent: 'swe-planner', satisfied: false }),
        report({ agent: 'pr-af', satisfied: false })
      ])
    ).toBe('2 installed agents need API keys before they can run.')
  })
})
