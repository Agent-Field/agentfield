import { join, resolve } from 'node:path'
import { BrowserWindow, Menu, app, ipcMain, shell } from 'electron'
import { CATALOG } from '../shared/catalog'
import { DEEP_LINK_SCHEME, type View, deepLinkFromArgv, parseDeepLink } from '../shared/deeplink'
import type { DesktopSettings } from '../shared/types'
import { spawn } from 'node:child_process'
import { DEFAULT_BASE_URL, getSnapshot } from './agentfield'
import { type AgentAction, runAgentAction, uninstallAgent } from './agents'
import { runAutostart } from './autostart'
import { getCliCommand, initializeCli, installBundledCli, refreshCliStatus } from './cli'
import { childEnv, initUserPath } from './env'
import { installAgent, installFromSource, updateAgent } from './installer'
import {
  getEnvReports,
  listStoredSecrets,
  revokeAgentSecret,
  revokeStoredSecret,
  setAgentSecret
} from './secrets'
import { loadSettings, mergeSettings, saveSettings } from './settings'
import { setupTray } from './tray'
import { syncTrayCompanion } from './tray-companion'
import { AppUpdater } from './updates'
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
  // Deep links can land before the app is ready — macOS delivers a cold-start
  // agentfield:// URL as an open-url event that can fire ahead of whenReady.
  // Constructing a BrowserWindow before app.whenReady() throws, so just stash
  // the view: the whenReady path builds the window and, once the renderer
  // announces itself, flushes pendingView (agentfield:renderer-ready).
  if (!app.isReady()) return
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

// The af-tray menu-bar companion shipped inside the app package (macOS only;
// see build.extraResources). Same layout as bundledCliPath — resources/bin when
// packaged, desktop/vendor in dev (npm run bundle-cli drops it there).
function bundledTrayPath(): string {
  return app.isPackaged
    ? join(process.resourcesPath, 'bin', 'af-tray')
    : join(app.getAppPath(), 'vendor', 'af-tray')
}

// macOS only: provision + install (or, when toggled off, uninstall) the af-tray
// menu-bar companion. Fire-and-forget like syncSkills — errors are logged, never
// thrown — and safe to call repeatedly (planTray only runs `af-tray install`
// when the binary changed or the launchd agent isn't loaded).
function syncTray(enabled: boolean): void {
  if (!isMac) return
  void syncTrayCompanion(enabled, bundledTrayPath())
    .then((r) => {
      if (!r.ok) console.error('tray companion:', r.message)
    })
    .catch((err) => console.error('tray companion failed:', err))
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
    stdio: 'ignore',
    env: childEnv()
  })
    .on('error', () => {})
    .on('close', () => syncSkills(rest))
}

// Register (or clear) the OS login item. Dev builds skip it — registering
// electron.exe as a login item would be wrong and confusing.
function applyLoginItem(next: DesktopSettings): void {
  if (!app.isPackaged) return
  // Only touch the OS when the desired state differs. Registering is not free:
  // on macOS an unsigned app (or one running outside /Applications) is refused
  // by SMAppService with a logged "Operation not permitted" — calling it with
  // an unchanged openAtLogin=false would emit that noise on every launch.
  if (app.getLoginItemSettings().openAtLogin === next.openAtLogin) return
  app.setLoginItemSettings({
    openAtLogin: next.openAtLogin,
    // Started at login the app stays out of the way: no window on show.
    // macOS launches login items with openAsHidden; Windows/Linux ignore that
    // field and honor the --hidden arg the startup guard reads instead.
    //
    // CAVEAT (macOS 13+): login items are now managed by SMAppService, which
    // treats openAsHidden / wasOpenedAsHidden as legacy and may ignore them —
    // the window can still appear at login on modern macOS. This is a
    // best-effort request; the OS gives no reliable "start hidden" there.
    openAsHidden: isMac,
    args: isMac ? [] : ['--hidden']
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

    // Resolve the user's real login-shell PATH once (Finder/Dock launches
    // inherit launchd's minimal PATH — see main/env.ts). Kicked off here so it
    // runs in parallel with CLI resolution; awaited before autostart, the main
    // spawn path. Until it lands, spawns fall back to process.env.PATH plus the
    // well-known dirs, so nothing breaks in the meantime.
    const userPathReady = initUserPath()

    // Resolve which af to drive (managed → PATH → bundled); on a machine
    // with no AgentField at all this provisions the bundled CLI, so a
    // desktop-app-only install still gets a working `af`.
    await initializeCli(bundledCliPath())
    if (settings.installSkills) syncSkills()

    // macOS only: provision + install the af-tray menu-bar companion so a
    // desktop-app-only install gets the menu-bar icon. Runs after initializeCli
    // (it needs the managed bin dir to exist) and non-blocking, like syncSkills.
    syncTray(settings.trayCompanion)

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
    // Install from a pasted GitHub repo URL. Shares the SAME install mutex and
    // the SAME progress channel as catalog installs — only one install of any
    // kind runs at a time. The source is raw renderer input; installFromSource
    // (via parseRepoSource) is the shape guard that keeps it to github.com
    // https URLs and never a CLI flag.
    ipcMain.handle('agentfield:install-source', async (event, source: unknown) => {
      if (typeof source !== 'string') {
        return { ok: false, message: 'invalid install request' }
      }
      if (installInFlight) {
        return { ok: false, message: 'an install is already in progress' }
      }
      installInFlight = true
      try {
        return await installFromSource(source, (line) => {
          if (!event.sender.isDestroyed()) {
            event.sender.send('agentfield:install-progress', line)
          }
        })
      } finally {
        installInFlight = false
      }
    })
    ipcMain.handle('agentfield:uninstall', (_event, name: unknown) => {
      if (typeof name !== 'string') {
        return { ok: false, message: 'invalid uninstall request' }
      }
      return uninstallAgent(name)
    })
    // Update shares the install mutex and progress channel: it is an install
    // with a stop/restart wrapped around it.
    ipcMain.handle('agentfield:update', async (event, name: unknown) => {
      if (typeof name !== 'string') {
        return { ok: false, message: 'invalid update request' }
      }
      if (installInFlight) {
        return { ok: false, message: 'an install is already in progress' }
      }
      installInFlight = true
      try {
        return await updateAgent(name, (line) => {
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
    // Open a control-plane web-UI page in the default browser. Only absolute
    // paths are accepted (no scheme/host smuggling — "//evil" would be a
    // protocol-relative URL) and they are joined to the known base URL, so
    // the renderer can never send the user to an arbitrary site.
    ipcMain.handle('agentfield:open-web-ui', (_event, path: unknown) => {
      if (typeof path !== 'string' || !path.startsWith('/') || path.startsWith('//')) {
        return false
      }
      void shell.openExternal(`${DEFAULT_BASE_URL}${path}`)
      return true
    })
    ipcMain.handle('agentfield:cli-status', () => refreshCliStatus(bundledCliPath()))
    ipcMain.handle('agentfield:cli-update', async () => {
      const result = await installBundledCli(bundledCliPath())
      if (!result.ok) console.error(result.message)
      return refreshCliStatus(bundledCliPath())
    })

    // The app's own updates, fed by the public GitHub releases (see
    // main/updates.ts). Found updates surface as a banner in the renderer
    // and under Settings; installing hands off to the platform installer.
    const updater = new AppUpdater({
      currentVersion: app.getVersion(),
      platform: process.platform,
      // arch picks the matching macOS DMG (arm64 vs x64) — see updates.ts.
      arch: process.arch,
      tempDir: app.getPath('temp'),
      openPath: (path) => shell.openPath(path),
      // Give the installer a beat to start, then get out of its way — the
      // NSIS one-click installer replaces the app in place and relaunches.
      quitForUpdate: () => setTimeout(() => app.quit(), 500),
      onStatus: (status) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('agentfield:app-update-status', status)
        }
      }
    })
    ipcMain.handle('agentfield:app-update-get', () => updater.status())
    ipcMain.handle('agentfield:app-update-check', () => updater.check())
    ipcMain.handle('agentfield:app-update-install', () => updater.install())
    // Dev builds carry package.json's static version — every release would
    // look like an update — so only packaged apps poll the channel. Manual
    // checks from Settings still work anywhere.
    if (app.isPackaged) updater.startAutoCheck()
    ipcMain.handle('agentfield:settings-get', () => settings)
    ipcMain.handle('agentfield:settings-set', async (_event, patch: unknown) => {
      const prev = settings
      settings = mergeSettings(settings, patch)
      applyLoginItem(settings)
      // macOS: reflect a flipped tray toggle (install ↔ uninstall) right away.
      if (settings.trayCompanion !== prev.trayCompanion) syncTray(settings.trayCompanion)
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

    // Suppress the initial window when we were launched hidden at login. On
    // Windows/Linux that is signalled by the --hidden arg we register; on macOS
    // by wasOpenedAsHidden (best-effort — SMAppService may ignore it on macOS
    // 13+, see applyLoginItem). A windowless macOS app is fine (the Dock and
    // the af-tray companion reopen it); on Windows/Linux only stay hidden when
    // a tray exists to live in, else there would be no way back to the window.
    const openedHidden = isMac
      ? app.getLoginItemSettings().wasOpenedAsHidden
      : process.argv.includes('--hidden')
    if (!openedHidden || (!isMac && !trayActive)) {
      createWindow()
    }

    // Bring the control plane and the selected agents up in the background,
    // once the real PATH is resolved so af's subprocesses (go, uv, …) resolve.
    void userPathReady.finally(() =>
      runAutostart(settings, (message) => console.log(message)).catch((err) =>
        console.error('autostart failed:', err)
      )
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
