import { join } from 'node:path'
import { BrowserWindow, Menu, app, ipcMain, shell } from 'electron'
import { CATALOG } from '../shared/catalog'
import { getSnapshot } from './agentfield'
import { installAgent } from './installer'

const isMac = process.platform === 'darwin'

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

  // External links (e.g. docs) open in the default browser, never in-app.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })

  // electron-vite convention: dev server URL in dev, built file in production.
  const devUrl = process.env['ELECTRON_RENDERER_URL']
  if (devUrl) {
    void win.loadURL(devUrl)
  } else {
    void win.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

let installInFlight = false

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

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
