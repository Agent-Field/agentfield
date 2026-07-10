import { join } from 'node:path'
import { BrowserWindow, app, ipcMain } from 'electron'
import { getSnapshot } from './agentfield'

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1080,
    height: 720,
    title: 'AgentField Desktop',
    backgroundColor: '#111418',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })

  // Read-only dashboard: never open child windows.
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  // electron-vite convention: dev server URL in dev, built file in production.
  const devUrl = process.env['ELECTRON_RENDERER_URL']
  if (devUrl) {
    void win.loadURL(devUrl)
  } else {
    void win.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  ipcMain.handle('agentfield:snapshot', () => getSnapshot())

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
