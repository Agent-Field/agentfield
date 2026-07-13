import { join, resolve } from 'node:path'
import { BrowserWindow, Menu, app, ipcMain, shell } from 'electron'
import { CATALOG } from '../shared/catalog'
import { DEEP_LINK_SCHEME, type View, deepLinkFromArgv, parseDeepLink } from '../shared/deeplink'
import { getSnapshot } from './agentfield'
import { installAgent } from './installer'
import { setupTray } from './tray'
import appIcon from '../../resources/icon.png?asset'

const isMac = process.platform === 'darwin'

let mainWindow: BrowserWindow | null = null
/** Deep-link view waiting for a window that can render it. */
let pendingView: View | null = null
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
    if (mainWindow === win) mainWindow = null
  })
  win.webContents.on('did-finish-load', () => flushPendingView())

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
  if (!mainWindow || !pendingView) return
  if (mainWindow.webContents.isLoading()) return // did-finish-load will flush
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

  app.whenReady().then(() => {
    installAppMenu()

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

    // macOS has its own menu-bar companion (af-tray) — no in-app tray there.
    if (!isMac) {
      trayActive = setupTray({ showWindow: showMainWindow, quit: () => app.quit() })
    }

    // A cold start via deep link (Windows) carries the URL in this argv.
    const initial = deepLinkFromArgv(process.argv)
    if (initial) pendingView = initial

    createWindow()

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
