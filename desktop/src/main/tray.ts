// Windows/Linux tray (taskbar status) icon.
//
// On macOS the menu-bar companion ships with AgentField itself (`af-tray`,
// installed by the curl installer), so the desktop app adds no tray there.
// On Windows there is no curl-installed tray — the desktop app carries it:
// a status glyph (the brand dot goes gold while the control plane runs),
// a small menu, and close-to-tray so the app keeps watching in the background.

import { readFileSync } from 'node:fs'
import { Menu, Tray, nativeImage, nativeTheme, shell } from 'electron'
import { checkControlPlane, getBaseUrl } from './agentfield'
import {
  type TrayState,
  darkTaskbar,
  trayIconBase,
  trayState,
  trayStatusLabel,
  trayTooltip
} from './tray-model'
import trayActiveLightIco from '../../resources/tray/tray-active-light.ico?asset'
import trayActiveDarkIco from '../../resources/tray/tray-active-dark.ico?asset'
import trayInactiveLightIco from '../../resources/tray/tray-inactive-light.ico?asset'
import trayInactiveDarkIco from '../../resources/tray/tray-inactive-dark.ico?asset'
import trayActiveLight16 from '../../resources/tray/tray-active-light-16.png?asset'
import trayActiveLight24 from '../../resources/tray/tray-active-light-24.png?asset'
import trayActiveLight32 from '../../resources/tray/tray-active-light-32.png?asset'
import trayActiveDark16 from '../../resources/tray/tray-active-dark-16.png?asset'
import trayActiveDark24 from '../../resources/tray/tray-active-dark-24.png?asset'
import trayActiveDark32 from '../../resources/tray/tray-active-dark-32.png?asset'
import trayInactiveLight16 from '../../resources/tray/tray-inactive-light-16.png?asset'
import trayInactiveLight24 from '../../resources/tray/tray-inactive-light-24.png?asset'
import trayInactiveLight32 from '../../resources/tray/tray-inactive-light-32.png?asset'
import trayInactiveDark16 from '../../resources/tray/tray-inactive-dark-16.png?asset'
import trayInactiveDark24 from '../../resources/tray/tray-inactive-dark-24.png?asset'
import trayInactiveDark32 from '../../resources/tray/tray-inactive-dark-32.png?asset'

const POLL_MS = 5000

/** Multi-size ico per glyph, for Windows (see scripts/make-icons.mjs). */
const ICOS: Record<string, string> = {
  'tray-active-light': trayActiveLightIco,
  'tray-active-dark': trayActiveDarkIco,
  'tray-inactive-light': trayInactiveLightIco,
  'tray-inactive-dark': trayInactiveDarkIco
}

/** 1x / 1.5x / 2x raster per glyph, for Linux (see scripts/make-icons.mjs). */
const ICONS: Record<string, [string, string, string]> = {
  'tray-active-light': [trayActiveLight16, trayActiveLight24, trayActiveLight32],
  'tray-active-dark': [trayActiveDark16, trayActiveDark24, trayActiveDark32],
  'tray-inactive-light': [trayInactiveLight16, trayInactiveLight24, trayInactiveLight32],
  'tray-inactive-dark': [trayInactiveDark16, trayInactiveDark24, trayInactiveDark32]
}

function trayImage(base: string): Electron.NativeImage {
  // The Windows tray ignores scale-factor representations and upscales the 1x
  // bitmap on >100% displays (electron/electron#33044); Electron only serves a
  // DPI-correct frame when the image is created from an .ico path.
  if (process.platform === 'win32') return nativeImage.createFromPath(ICOS[base])
  const [x1, x15, x2] = ICONS[base]
  const image = nativeImage.createEmpty()
  image.addRepresentation({ scaleFactor: 1, buffer: readFileSync(x1) })
  image.addRepresentation({ scaleFactor: 1.5, buffer: readFileSync(x15) })
  image.addRepresentation({ scaleFactor: 2, buffer: readFileSync(x2) })
  return image
}

/** What the tray needs from the app shell, kept narrow for clarity. */
export interface TrayHost {
  showWindow(): void
  quit(): void
}

/**
 * Create the tray and start polling the control plane. Returns false when the
 * platform has no working status area (e.g. some Linux desktops) — the caller
 * then keeps classic quit-on-close behavior instead of close-to-tray.
 */
export function setupTray(host: TrayHost): boolean {
  // The base URL is re-read on every poll: autostart may adopt (or start)
  // the control plane on a non-default port after the tray already exists.
  let baseUrl = getBaseUrl()
  let state: TrayState = 'stopped'
  let dark = darkTaskbar(nativeTheme)

  let tray: Tray
  try {
    tray = new Tray(trayImage(trayIconBase(state, dark)))
  } catch (err) {
    // e.g. a Linux desktop without a status area — or a packaging bug that
    // lost the glyphs; degrade to quit-on-close but say why.
    console.error('tray unavailable:', err)
    return false
  }

  const apply = (): void => {
    const hostLabel = new URL(baseUrl).host
    tray.setImage(trayImage(trayIconBase(state, dark)))
    tray.setToolTip(trayTooltip(state, hostLabel))
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: 'Open AgentField', click: () => host.showWindow() },
        {
          label: 'Open web UI',
          enabled: state === 'running',
          click: () => void shell.openExternal(`${getBaseUrl()}/ui/`)
        },
        { type: 'separator' },
        { label: trayStatusLabel(state, hostLabel), enabled: false },
        { type: 'separator' },
        { label: 'Quit AgentField', click: () => host.quit() }
      ])
    )
  }
  apply()

  // Re-render only on actual change — replacing the icon/menu every poll
  // would churn native tray APIs (and can dismiss an open menu on Windows).
  const update = (nextState: TrayState, nextDark: boolean): void => {
    const nextUrl = getBaseUrl()
    if (nextState === state && nextDark === dark && nextUrl === baseUrl) return
    state = nextState
    dark = nextDark
    baseUrl = nextUrl
    apply()
  }

  // The poll re-reads the taskbar theme too: Windows does not reliably emit
  // nativeTheme 'updated' when only the system (taskbar) theme flips.
  const refresh = async (): Promise<void> => {
    update(trayState(await checkControlPlane()), darkTaskbar(nativeTheme))
  }
  void refresh()
  setInterval(() => void refresh(), POLL_MS)

  tray.on('click', () => host.showWindow())
  nativeTheme.on('updated', () => update(state, darkTaskbar(nativeTheme)))
  return true
}
