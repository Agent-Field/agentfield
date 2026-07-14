import { join, resolve } from 'node:path'
import { BrowserWindow, Menu, app, ipcMain, shell } from 'electron'
import { CATALOG } from '../shared/catalog'
import { DEEP_LINK_SCHEME, type View, deepLinkFromArgv, parseDeepLink } from '../shared/deeplink'
import type { DesktopSettings } from '../shared/types'
import { spawn } from 'node:child_process'
import { getSnapshot } from './agentfield'
import { type AgentAction, runAgentAction } from './agents'
import { runAutostart } from './autostart'
import { getCliCommand, initializeCli, installBundledCli, refreshCliStatus } from './cli'
import { installAgent } from './installer'
import {
  getEnvReports,
  listStoredSecrets,
  revokeAgentSecret,
  revokeStoredSecret,
  setAgentSecret
} from './secrets'
import { loadSettings, mergeSettings, saveSettings } from './settings'
import { setupTray } from './tray'
import appIcon from '../../resources/icon.png?asset'

const isMac = process.platform === 'darwin'

let mainWindow: BrowserWindow | null = null
/** Deep-link view waiting for a renderer that can show it. */
let pendingView: View | null = null
/**
 * True once the renderer subscribed to navigation (it announces itself via
 * agentfield:renderer-ready on mount). A push before that would be dropped —
 * did-finish-load fires before React mounts, so readiness is the renderer's
 * call, not the page loader's.
 */
let rendererReady = false
/** True once the user chose Quit — lets close-to-tray tell hide from exit. */
let quitting = false
/** True when a tray exists (Windows/Linux) — enables close-to-tray. */
let trayActive = false

// Mac-first chrome: no default File/Edit/View menu bar. On macOS an app menu
// must still exist (it owns Cmd+Q/Cmd+W/Cmd+C…), so build the minimal one;
// on Windows/Linux remove the bar entirely.
function installAppMenu(): void {
  if (!isMac) {
    Menu.setApplicationMenu(null)
    return
  }
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: app.name,
        submenu: [
          { role: 'about' },
          { type: 'separator' },
          { role: 'hide' },
          { role: 'hideOthers' },
          { type: 'separator' },
          { role: 'quit' }
        ]
      },
      // Keeps standard clipboard shortcuts working in text fields.
      { role: 'editMenu' },
      { role: 'windowMenu' }
    ])
  )
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 980,
    height: 700,
    minWidth: 720,
    minHeight: 480,
    title: 'AgentField',
    backgroundColor: '#00000000',
    // Windows/Linux window + taskbar icon; macOS uses the bundle's icns.
    icon: isMac ? undefined : appIcon,
    // Seamless titlebar: traffic lights float over the content on macOS,
    // native window controls overlay on Windows. The renderer reserves a
    // draggable strip at the top (see styles.css .titlebar).
    titleBarStyle: isMac ? 'hiddenInset' : 'hidden',
    trafficLightPosition: isMac ? { x: 18, y: 18 } : undefined,
    titleBarOverlay: isMac
      ? undefined
      : { color: '#00000000', symbolColor: '#8b95a3', height: 48 },
    vibrancy: isMac ? 'sidebar' : undefined,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  mainWindow = win

  // External links (e.g. docs) open in the default browser, never in-app.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })

  // With a tray, closing the window hides it (the app keeps watching from
  // the tray, Docker-Desktop style); Quit lives in the tray menu.
  win.on('close', (event) => {
    if (trayActive && !quitting) {
      event.preventDefault()
      win.hide()
    }
  })
  win.on('closed', () => {
    if (mainWindow === win) {
      mainWindow = null
      rendererReady = false
    }
  })
  // A reload restarts the renderer; it re-announces readiness on mount.
  win.webContents.on('did-start-loading', () => {
    rendererReady = false
  })

  // electron-vite convention: dev server URL in dev, built file in production.
  const devUrl = process.env['ELECTRON_RENDERER_URL']
  if (devUrl) {
    void win.loadURL(devUrl)
  } else {
    void win.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function showMainWindow(): void {
  if (!mainWindow || mainWindow.isDestroyed()) {
    createWindow()
    return
  }
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
}

function flushPendingView(): void {
  // Not ready yet -> keep pendingView; the renderer collects it when it
  // announces readiness (agentfield:renderer-ready returns-and-clears it).
  if (!mainWindow || !pendingView || !rendererReady) return
  mainWindow.webContents.send('agentfield:navigate', pendingView)
  pendingView = null
}

/** Bring the app forward and, when a deep link named a view, switch to it. */
function navigate(view: View | null): void {
  if (view) pendingView = view
  showMainWindow()
  flushPendingView()
}

// Register the agentfield:// scheme (see shared/deeplink.ts). Packaged apps
// register their own executable; in dev the handler must point Electron at
// this app's entry explicitly.
function registerDeepLinks(): void {
  if (process.defaultApp) {
    if (process.argv.length >= 2) {
      app.setAsDefaultProtocolClient(DEEP_LINK_SCHEME, process.execPath, [
        resolve(process.argv[1])
      ])
    }
  } else {
    app.setAsDefaultProtocolClient(DEEP_LINK_SCHEME)
  }
}

let installInFlight = false
let settings: DesktopSettings

function settingsFile(): string {
  return join(app.getPath('userData'), 'settings.json')
}

// The af CLI shipped inside the app package (see build.extraResources). In
// dev, scripts/bundle-cli.mjs drops it into desktop/vendor/ instead.
function bundledCliPath(): string {
  const name = process.platform === 'win32' ? 'af.exe' : 'af'
  return app.isPackaged
    ? join(process.resourcesPath, 'bin', name)
    : join(app.getAppPath(), 'vendor', name)
}

// Keep the AgentField skills present in detected coding agents (Claude Code,
// Codex, Gemini, …): the builder skill (agentfield) and the consumer skill
// (agentfield-use — how to discover and call installed agents). One install
// per skill, sequential so concurrent runs never race on skillkit's state
// file. Idempotent — skillkit tracks versions in ~/.agentfield/skills/
// .state.json — and pure best-effort: an older CLI without agentfield-use in
// its catalog fails that one invocation and nothing else.
function syncSkills(names = ['agentfield', 'agentfield-use']): void {
  const [head, ...rest] = names
  if (!head) return
  spawn(getCliCommand(), ['skill', 'install', head, '--non-interactive'], {
    windowsHide: true,
    stdio: 'ignore'
  })
    .on('error', () => {})
    .on('close', () => syncSkills(rest))
}

// Register (or clear) the OS login item. Dev builds skip it — registering
// electron.exe as a login item would be wrong and confusing.
function applyLoginItem(next: DesktopSettings): void {
  if (!app.isPackaged) return
  app.setLoginItemSettings({
    openAtLogin: next.openAtLogin,
    // Started at login the app stays out of the way: tray only, no window.
    args: ['--hidden']
  })
}

function main(): void {
  registerDeepLinks()

  // Windows/Linux: a relaunch (including one carrying an agentfield:// URL in
  // argv) lands here in the first instance instead of opening a second app.
  app.on('second-instance', (_event, argv) => {
    navigate(deepLinkFromArgv(argv))
  })
  // macOS delivers deep links as open-url events.
  app.on('open-url', (event, url) => {
    event.preventDefault()
    navigate(parseDeepLink(url))
  })
  app.on('before-quit', () => {
    quitting = true
  })

  app.whenReady().then(async () => {
    installAppMenu()
    settings = await loadSettings(settingsFile())
    applyLoginItem(settings)

    // Resolve which af to drive (managed → PATH → bundled); on a machine
    // with no AgentField at all this provisions the bundled CLI, so a
    // desktop-app-only install still gets a working `af`.
    await initializeCli(bundledCliPath())
    if (settings.installSkills) syncSkills()

    ipcMain.handle('agentfield:snapshot', () => getSnapshot())
    ipcMain.handle('agentfield:catalog', () => CATALOG)
    ipcMain.handle('agentfield:install', async (event, name: unknown) => {
      if (typeof name !== 'string') {
        return { ok: false, message: 'invalid install request' }
      }
      if (installInFlight) {
        return { ok: false, message: 'an install is already in progress' }
      }
      installInFlight = true
      try {
        return await installAgent(name, (line) => {
          if (!event.sender.isDestroyed()) {
            event.sender.send('agentfield:install-progress', line)
          }
        })
      } finally {
        installInFlight = false
      }
    })
    ipcMain.handle('agentfield:agent-action', (_event, action: unknown, name: unknown) => {
      if (
        typeof name !== 'string' ||
        (action !== 'start' && action !== 'stop' && action !== 'restart')
      ) {
        return { ok: false, message: 'invalid agent action' }
      }
      return runAgentAction(action as AgentAction, name)
    })
    ipcMain.handle('agentfield:env-reports', () => getEnvReports())
    ipcMain.handle(
      'agentfield:secret-set',
      (_event, agent: unknown, key: unknown, value: unknown) => {
        if (typeof agent !== 'string' || typeof key !== 'string' || typeof value !== 'string') {
          return { ok: false, message: 'invalid secret request' }
        }
        return setAgentSecret(agent, key, value)
      }
    )
    ipcMain.handle('agentfield:secret-revoke', (_event, agent: unknown, key: unknown) => {
      if (typeof agent !== 'string' || typeof key !== 'string') {
        return { ok: false, message: 'invalid secret request' }
      }
      return revokeAgentSecret(agent, key)
    })
    ipcMain.handle('agentfield:secrets-list', () => listStoredSecrets())
    ipcMain.handle('agentfield:secrets-revoke', (_event, key: unknown, scope: unknown) => {
      if (typeof key !== 'string' || typeof scope !== 'string') {
        return { ok: false, message: 'invalid secret request' }
      }
      return revokeStoredSecret(key, scope)
    })
    // The renderer calls this once its navigation listener is live; the
    // return value is the deep-link view (if any) that arrived before then.
    ipcMain.handle('agentfield:renderer-ready', () => {
      rendererReady = true
      const view = pendingView
      pendingView = null
      return view
    })
    ipcMain.handle('agentfield:cli-status', () => refreshCliStatus(bundledCliPath()))
    ipcMain.handle('agentfield:cli-update', async () => {
      const result = await installBundledCli(bundledCliPath())
      if (!result.ok) console.error(result.message)
      return refreshCliStatus(bundledCliPath())
    })
    ipcMain.handle('agentfield:settings-get', () => settings)
    ipcMain.handle('agentfield:settings-set', async (_event, patch: unknown) => {
      settings = mergeSettings(settings, patch)
      applyLoginItem(settings)
      await saveSettings(settingsFile(), settings)
      return settings
    })

    // macOS has its own menu-bar companion (af-tray) — no in-app tray there.
    if (!isMac) {
      trayActive = setupTray({ showWindow: showMainWindow, quit: () => app.quit() })
    }

    // A cold start via deep link (Windows) carries the URL in this argv.
    const initial = deepLinkFromArgv(process.argv)
    if (initial) pendingView = initial

    // Login-item launches pass --hidden: stay in the tray, no window. Without
    // a tray to live in, fall back to showing the window as usual.
    if (!process.argv.includes('--hidden') || !trayActive) {
      createWindow()
    }

    // Bring the control plane and the selected agents up in the background.
    runAutostart(settings, (message) => console.log(message)).catch((err) =>
      console.error('autostart failed:', err)
    )

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
  })

  app.on('window-all-closed', () => {
    // macOS convention keeps apps alive without windows; with a tray the app
    // stays resident too. Only tray-less Windows/Linux quits on last close.
    if (!isMac && !trayActive) app.quit()
  })
}

if (app.requestSingleInstanceLock()) {
  main()
} else {
  app.quit()
}
