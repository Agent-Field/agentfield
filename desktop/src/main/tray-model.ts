// Pure presentation logic for the Windows/Linux tray icon. No Electron
// imports, so it is unit-tested directly (see tray-model.test.ts); the tray
// glue that consumes it lives in tray.ts.
//
// macOS gets no tray from the desktop app on purpose: the menu-bar companion
// there is `af-tray`, installed with AgentField itself (control-plane/cmd/af-tray).

import type { ControlPlaneStatus } from '../shared/types'

export type TrayState = 'running' | 'unhealthy' | 'foreign' | 'stopped'

/** Collapse a health probe into the four states the tray can express. */
export function trayState(cp: ControlPlaneStatus): TrayState {
  if (cp.healthy) return 'running'
  if (cp.reachable && cp.recognized) return 'unhealthy'
  if (cp.reachable) return 'foreign'
  return 'stopped'
}

/** One-line status, used as the disabled menu row. `host` is e.g. "localhost:8080". */
export function trayStatusLabel(state: TrayState, host: string): string {
  switch (state) {
    case 'running':
      return `Control plane running · ${host}`
    case 'unhealthy':
      return 'Control plane unhealthy'
    case 'foreign':
      return `Port in use by another app (${host})`
    case 'stopped':
      return 'Control plane not running'
  }
}

export function trayTooltip(state: TrayState, host: string): string {
  return `AgentField — ${trayStatusLabel(state, host)}`
}

/**
 * Which glyph file the tray wears (relative to resources/tray/, without the
 * -<size>.png suffix). The gold dot doubles as the status light: gold while
 * the control plane is running, gray otherwise. `darkTaskbar` follows the
 * OS system-UI theme — light glyphs for dark taskbars and vice versa.
 */
export function trayIconBase(state: TrayState, darkTaskbar: boolean): string {
  const activity = state === 'running' ? 'active' : 'inactive'
  return `tray-${activity}-${darkTaskbar ? 'light' : 'dark'}`
}
