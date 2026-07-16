import { describe, expect, it } from 'vitest'
import type { ControlPlaneStatus } from '../shared/types'
import { trayIconBase, trayState, trayStatusLabel, trayTooltip } from './tray-model'

function cp(overrides: Partial<ControlPlaneStatus>): ControlPlaneStatus {
  return { reachable: false, recognized: false, healthy: false, ...overrides }
}

describe('trayState', () => {
  it('healthy control plane is running', () => {
    expect(trayState(cp({ reachable: true, recognized: true, healthy: true }))).toBe('running')
  })

  it('recognized but not healthy is unhealthy', () => {
    expect(trayState(cp({ reachable: true, recognized: true }))).toBe('unhealthy')
  })

  it('reachable but unrecognized (another app on the port) is foreign', () => {
    expect(trayState(cp({ reachable: true }))).toBe('foreign')
  })

  it('unreachable is stopped', () => {
    expect(trayState(cp({}))).toBe('stopped')
  })
})

describe('tray labels', () => {
  it('running names the host', () => {
    expect(trayStatusLabel('running', 'localhost:8080')).toBe(
      'Control plane running · localhost:8080'
    )
  })

  it('foreign warns about the squatting app', () => {
    expect(trayStatusLabel('foreign', 'localhost:8080')).toBe(
      'Port in use by another app (localhost:8080)'
    )
  })

  it('stopped and unhealthy are plain statements', () => {
    expect(trayStatusLabel('stopped', 'localhost:8080')).toBe('Control plane not running')
    expect(trayStatusLabel('unhealthy', 'localhost:8080')).toBe('Control plane unhealthy')
  })

  it('tooltip is the brand plus the status', () => {
    expect(trayTooltip('running', 'localhost:8080')).toBe(
      'AgentField — Control plane running · localhost:8080'
    )
  })
})

describe('trayIconBase', () => {
  it('only running earns the active (gold-dot) glyph', () => {
    expect(trayIconBase('running', true)).toBe('tray-active-light')
    expect(trayIconBase('unhealthy', true)).toBe('tray-inactive-light')
    expect(trayIconBase('foreign', true)).toBe('tray-inactive-light')
    expect(trayIconBase('stopped', true)).toBe('tray-inactive-light')
  })

  it('light glyphs for dark taskbars, dark glyphs for light taskbars', () => {
    expect(trayIconBase('running', false)).toBe('tray-active-dark')
    expect(trayIconBase('stopped', false)).toBe('tray-inactive-dark')
  })
})
