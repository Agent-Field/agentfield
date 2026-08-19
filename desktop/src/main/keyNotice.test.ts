import { describe, expect, it, vi } from 'vitest'
import type { AgentEnvReport, AgentEnvVar } from '../shared/types'
import {
  type KeyNoticeDeps,
  keyNoticeCandidates,
  missingKeyLabels,
  notifyUnresolvedKeys,
  planKeyNotice
} from './keyNotice'

function variable(partial: Partial<AgentEnvVar> & { name: string }): AgentEnvVar {
  return {
    description: '',
    secret: true,
    scope: 'global',
    required: true,
    status: 'missing',
    storedScopes: [],
    ...partial
  }
}

/** A report shaped like getEnvReports() builds one, satisfied derived by hand. */
function report(agent: string, vars: AgentEnvVar[], satisfied: boolean): AgentEnvReport {
  return { agent, vars, satisfied }
}

const NEEDS_OPENROUTER = report(
  'swe-planner',
  [variable({ name: 'OPENROUTER_API_KEY' })],
  false
)
const NEEDS_TOKEN = report('pr-af', [variable({ name: 'GH_TOKEN' })], false)

describe('keyNoticeCandidates', () => {
  it('drops names already announced and dedupes the rest', () => {
    expect(keyNoticeCandidates(['a', 'b', 'a'], ['b'])).toEqual(['a'])
    expect(keyNoticeCandidates(['a'], ['a'])).toEqual([])
    expect(keyNoticeCandidates([], [])).toEqual([])
    expect(keyNoticeCandidates(['', 'a'], [])).toEqual(['a'])
  })
})

describe('missingKeyLabels', () => {
  it('names each unresolved required variable', () => {
    expect(missingKeyLabels(NEEDS_OPENROUTER)).toEqual(['OPENROUTER_API_KEY'])
  })

  it('ignores optional and already-resolved variables', () => {
    const r = report(
      'x',
      [
        variable({ name: 'SET_IN_ENV', status: 'env' }),
        variable({ name: 'IN_STORE', status: 'stored' }),
        variable({ name: 'HAS_DEFAULT', status: 'default' }),
        variable({ name: 'OPTIONAL_ONE', required: false, status: 'missing' }),
        variable({ name: 'REALLY_MISSING' })
      ],
      false
    )
    expect(missingKeyLabels(r)).toEqual(['REALLY_MISSING'])
  })

  it('collapses a require_one_of group into one "A or B" label', () => {
    const r = report(
      'x',
      [
        variable({ name: 'ANTHROPIC_API_KEY', group: 'llm' }),
        variable({ name: 'OPENROUTER_API_KEY', group: 'llm' })
      ],
      false
    )
    expect(missingKeyLabels(r)).toEqual(['ANTHROPIC_API_KEY or OPENROUTER_API_KEY'])
  })

  it('says nothing about a group one member already satisfies', () => {
    const r = report(
      'x',
      [
        variable({ name: 'ANTHROPIC_API_KEY', group: 'llm', status: 'stored' }),
        variable({ name: 'OPENROUTER_API_KEY', group: 'llm' }),
        variable({ name: 'GH_TOKEN' })
      ],
      false
    )
    expect(missingKeyLabels(r)).toEqual(['GH_TOKEN'])
  })
})

describe('planKeyNotice', () => {
  const base = {
    provisioned: ['swe-planner', 'pr-af'],
    reports: [NEEDS_OPENROUTER, NEEDS_TOKEN],
    alreadyNotified: [] as string[],
    supported: true
  }

  it('names every unresolved agent and what it needs', () => {
    const plan = planKeyNotice(base)
    expect(plan.notify).toBe(true)
    expect(plan.agents).toEqual(['swe-planner', 'pr-af'])
    expect(plan.title).toBe('2 agents need keys')
    expect(plan.body).toBe(
      'swe-planner needs OPENROUTER_API_KEY; pr-af needs GH_TOKEN — click to add them in AgentField → Agents → Keys.'
    )
  })

  it('uses singular copy for a single agent with a single key', () => {
    const plan = planKeyNotice({ ...base, provisioned: ['swe-planner'] })
    expect(plan.title).toBe('swe-planner needs a key')
    expect(plan.body).toBe(
      'swe-planner needs OPENROUTER_API_KEY — click to add it in AgentField → Agents → Keys.'
    )
  })

  it('never names a secret value, only variable names', () => {
    const r = report('x', [variable({ name: 'GH_TOKEN' })], false)
    const plan = planKeyNotice({ ...base, provisioned: ['x'], reports: [r] })
    expect(plan.body).toContain('GH_TOKEN')
    expect(plan.body).not.toMatch(/ghp_|sk-/)
  })

  it('stays silent when every provisioned agent is satisfied', () => {
    const plan = planKeyNotice({
      ...base,
      reports: [
        report('swe-planner', [variable({ name: 'OPENROUTER_API_KEY', status: 'stored' })], true),
        report('pr-af', [variable({ name: 'GH_TOKEN', status: 'stored' })], true)
      ]
    })
    expect(plan.notify).toBe(false)
    expect(plan.reason).toContain('every required key')
  })

  it('trusts satisfied over the variable statuses (old control planes report satisfied: true)', () => {
    // secrets.ts falls back to satisfied: true when the control plane cannot
    // report `requirement` metadata — every var then looks required+missing.
    // Notifying there would be a guess, so the fallback must win.
    const legacy = report(
      'swe-planner',
      [variable({ name: 'OPENROUTER_API_KEY', status: 'missing' })],
      true
    )
    const plan = planKeyNotice({ ...base, provisioned: ['swe-planner'], reports: [legacy] })
    expect(plan.notify).toBe(false)
  })

  it('stays silent on the control-plane error report', () => {
    const err: AgentEnvReport = {
      agent: '',
      vars: [],
      satisfied: false,
      error: 'Could not reach the control plane'
    }
    const plan = planKeyNotice({ ...base, reports: [err] })
    expect(plan.notify).toBe(false)
  })

  it('stays silent when an agent has no report at all', () => {
    expect(planKeyNotice({ ...base, reports: [] }).notify).toBe(false)
  })

  it('stays silent when unsatisfied but nothing is nameable', () => {
    const odd = report('swe-planner', [variable({ name: 'X', required: false })], false)
    const plan = planKeyNotice({ ...base, provisioned: ['swe-planner'], reports: [odd] })
    expect(plan.notify).toBe(false)
  })

  it('skips agents already announced on an earlier launch', () => {
    const plan = planKeyNotice({ ...base, alreadyNotified: ['swe-planner'] })
    expect(plan.agents).toEqual(['pr-af'])
    expect(plan.title).toBe('pr-af needs a key')
  })

  it('does not notify when nothing was provisioned this run', () => {
    const plan = planKeyNotice({ ...base, provisioned: [] })
    expect(plan.notify).toBe(false)
    expect(plan.reason).toContain('nothing newly provisioned')
  })

  it('does not notify when notifications are unsupported', () => {
    const plan = planKeyNotice({ ...base, supported: false })
    expect(plan.notify).toBe(false)
    expect(plan.agents).toEqual([])
  })

  it('elides long lists but still records every agent', () => {
    const many = ['a', 'b', 'c', 'd']
    const plan = planKeyNotice({
      ...base,
      provisioned: many,
      reports: many.map((name) =>
        report(
          name,
          ['K1', 'K2', 'K3', 'K4'].map((key) => variable({ name: `${name}_${key}` })),
          false
        )
      )
    })
    expect(plan.title).toBe('4 agents need keys')
    expect(plan.body).toContain('a needs a_K1, a_K2, a_K3 and 1 more')
    expect(plan.body).toContain('(and 1 more agent)')
    expect(plan.body).not.toContain('d needs')
    // Elided agents are still recorded — the notice counted them.
    expect(plan.agents).toEqual(many)
  })
})

function deps(overrides: Partial<KeyNoticeDeps> = {}): KeyNoticeDeps & {
  shown: { title: string; body: string }[]
  recorded: string[][]
  logs: string[]
} {
  const shown: { title: string; body: string }[] = []
  const recorded: string[][] = []
  const logs: string[] = []
  return {
    shown,
    recorded,
    logs,
    reports: async () => [NEEDS_OPENROUTER, NEEDS_TOKEN],
    supported: () => true,
    show: (notice) => {
      shown.push(notice)
    },
    markNotified: async (agents) => {
      recorded.push([...agents])
    },
    log: (message) => {
      logs.push(message)
    },
    ...overrides
  }
}

describe('notifyUnresolvedKeys', () => {
  it('shows one notification and records the agents', async () => {
    const d = deps()
    const plan = await notifyUnresolvedKeys(['swe-planner', 'pr-af'], [], d)
    expect(plan.notify).toBe(true)
    expect(d.shown).toHaveLength(1)
    expect(d.shown[0].title).toBe('2 agents need keys')
    expect(d.recorded).toEqual([['swe-planner', 'pr-af']])
  })

  it('does not fire again once the agents are recorded', async () => {
    const d = deps()
    await notifyUnresolvedKeys(['swe-planner', 'pr-af'], ['swe-planner', 'pr-af'], d)
    expect(d.shown).toEqual([])
    expect(d.recorded).toEqual([])
  })

  it('skips the control-plane round trip when there is nothing to announce', async () => {
    const reports = vi.fn(async () => [NEEDS_OPENROUTER])
    const d = deps({ reports })
    await notifyUnresolvedKeys([], [], d)
    expect(reports).not.toHaveBeenCalled()
  })

  it('skips the round trip and shows nothing when unsupported', async () => {
    const reports = vi.fn(async () => [NEEDS_OPENROUTER])
    const d = deps({ supported: () => false, reports })
    await notifyUnresolvedKeys(['swe-planner'], [], d)
    expect(reports).not.toHaveBeenCalled()
    expect(d.shown).toEqual([])
    expect(d.recorded).toEqual([])
  })

  it('records nothing when the notification itself fails', async () => {
    const d = deps({
      show: () => {
        throw new Error('no notification daemon')
      }
    })
    const plan = await notifyUnresolvedKeys(['swe-planner'], [], d)
    expect(plan.notify).toBe(false)
    expect(d.recorded).toEqual([])
  })

  it('keeps the notice when persisting it fails', async () => {
    const d = deps({
      markNotified: async () => {
        throw new Error('disk full')
      }
    })
    const plan = await notifyUnresolvedKeys(['swe-planner'], [], d)
    expect(plan.notify).toBe(true)
    expect(d.shown).toHaveLength(1)
    expect(d.logs.some((line) => line.includes('could not record'))).toBe(true)
  })

  it('never rejects when a dependency throws', async () => {
    const d = deps({
      reports: async () => {
        throw new Error('boom')
      }
    })
    const plan = await notifyUnresolvedKeys(['swe-planner'], [], d)
    expect(plan.notify).toBe(false)
    expect(plan.reason).toContain('aborted')
    expect(d.shown).toEqual([])
  })
})
