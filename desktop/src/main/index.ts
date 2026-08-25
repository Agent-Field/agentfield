import { join, resolve } from 'node:path'
import { existsSync } from 'node:fs'
import { BrowserWindow, Menu, Notification, app, ipcMain, nativeTheme, safeStorage, shell } from 'electron'
import { CATALOG } from '../shared/catalog'
import { BUNDLED_NODES } from '../shared/bundled'
import { RAILWAY_TEMPLATE_URL } from '../shared/cloudLinks'
import { DEEP_LINK_SCHEME, type View, deepLinkFromArgv, parseDeepLink } from '../shared/deeplink'
import type {
  CloudAutoUpdateMode,
  DesktopSettings,
  LocalControlPlaneRestartStatus
} from '../shared/types'
import { getBaseUrl, getSnapshot, setActiveControlPlanePort } from './agentfield'
import {
  type AgentAction,
  runAgentAction,
  setAgentPackageAutoUpdate,
  startControlPlane,
  uninstallAgent
} from './agents'
import { ensureAforgeCompanion } from './aforge-companion'
import { recoverAutostartFailure, runAutostart } from './autostart'
import { runDesktopBootChain } from './bootChain'
import { bundledStatuses, ensureBundledAgents } from './bundledAgents'
import { testCloudConnection, applyConnectionProfile } from './cloud'
import { isCloudActive } from './connection'
import { createCpClient } from './cpClient'
import { getCliCommand, initializeCli, installBundledCli, refreshCliStatus } from './cli'
import { initUserPath } from './env'
import { installAgent, installFromSource, updateAgent } from './installer'
import { notifyUnresolvedKeys } from './keyNotice'
import {
  getEnvReports,
  listStoredSecrets,
  revokeAgentSecret,
  revokeStoredSecret,
  setAgentSecret
} from './secrets'
import {
  loadSettings,
  mergeSettings,
  persistCloudAutoUpdatePreference,
  saveSettings,
  settingsForCloudService,
  settingsWithCloudProfile,
  settingsWithDismissedCloudUpdate
} from './settings'
import {
  SkillSync,
  defaultSkillSyncDeps,
  shouldSyncOnCliUpdate,
  shouldSyncOnLaunch,
  shouldSyncOnSettingsChange
} from './skills'
import { pickFreePort } from './ports'
import { setupTray } from './tray'
import { syncTrayCompanion } from './tray-companion'
import { AppUpdater } from './updates'
import {
  applyCloudUpdateWithRailwayToken,
  autoUpdateModeAfterDeploy,
  cloudUpdateApplyPath,
  cloudUpdateRailwayControlsAvailable,
  CloudUpdateChecker,
  setCloudAutoUpdateSchedule
} from './cloudUpdate'
import {
  reconcileLocalControlPlaneRestart,
  restartAdoptedControlPlaneAfterCliSwap
} from './localCpUpdate'
import {
  getFreshAccessToken,
  isLoggedIn,
  listWorkspaces,
  loginWithRailway,
  logout,
  type RailwayAuthDeps
} from './railwayAuth'
import { createRailwayApi } from './railwayApi'
import { loadRailwayStatus } from './railwayStatus'
import {
  deploymentStateInfo,
  hasDeployment,
  refreshDeploymentState,
  resolveTofuBinary,
  runDeploy,
  runDestroy
} from './deployEngine'
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
    // native window controls overlay on Windows/Linux. The renderer uses
    // Electron's titlebar-area CSS environment variables to keep actions clear.
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

/**
 * The one install mutex for the whole app: the IPC handlers below and
 * first-launch bundled provisioning both go through it, because the control
 * plane answers a second concurrent install with a 409.
 *
 * The two callers want different things when it is taken. A handler is
 * answering a click, so it refuses straight away and the renderer says so.
 * Provisioning is background work nobody is watching, so it waits its turn
 * instead of failing a bundled node over a coincidence of timing.
 */
let installInFlight = false
/** Provisioning calls parked until the current install finishes, in order. */
const installWaiters: Array<() => void> = []

/** Take the mutex, waiting when it is held. Used by provisioning only. */
function acquireInstall(): Promise<void> {
  if (!installInFlight) {
    installInFlight = true
    return Promise.resolve()
  }
  return new Promise((resolve) => installWaiters.push(resolve))
}

/**
 * Release the mutex. A parked waiter is handed the lock directly — the flag
 * stays set across the handoff, so a handler can never slip in during the
 * microtask it takes the waiter to resume.
 */
function releaseInstall(): void {
  const next = installWaiters.shift()
  if (next) {
    next()
    return
  }
  installInFlight = false
}

let cloudDeployInFlight = false
let settings: DesktopSettings
let localControlPlaneRestart: LocalControlPlaneRestartStatus | null = null

function settingsFile(): string {
  return join(app.getPath('userData'), 'settings.json')
}

function authDeps(): RailwayAuthDeps {
  const encrypted = safeStorage.isEncryptionAvailable()
  if (!encrypted) console.warn('Railway token encryption unavailable; token storage is unencrypted')
  return {
    codec: encrypted
      ? {
          encrypt: (plain) => safeStorage.encryptString(plain).toString('base64'),
          decrypt: (blob) => safeStorage.decryptString(Buffer.from(blob, 'base64'))
        }
      : { encrypt: (plain) => plain, decrypt: (blob) => blob },
    storePath: join(app.getPath('userData'), 'railway-auth.json'),
    openUrl: (url) => shell.openExternal(url).then(() => undefined)
  }
}

function deployPaths(): { workspaceDir: string; binaryDir: string | null } {
  const candidate = app.isPackaged
    ? join(process.resourcesPath, 'bin', 'deploy-engine')
    : join(app.getAppPath(), 'vendor', 'deploy-engine')
  return {
    workspaceDir: join(app.getPath('userData'), 'cloud-deploy'),
    binaryDir: existsSync(candidate) ? candidate : null
  }
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

// Where per-run skill-sync lines land: ~/Library/Logs/AgentField/skill-sync.log
// on macOS, the platform equivalent elsewhere. app.getPath('logs') is the
// Electron-blessed spot; fall back under userData if the platform has no logs
// path so a sync is never lost for want of a directory.
function skillSyncLogFile(): string {
  try {
    return join(app.getPath('logs'), 'skill-sync.log')
  } catch {
    return join(app.getPath('userData'), 'logs', 'skill-sync.log')
  }
}

/**
 * Keeps the AgentField skill catalog installed in detected coding agents (see
 * main/skills.ts) and remembers how the last run went. Built once app is
 * ready — skillSyncLogFile() needs the app paths.
 */
let skillSync: SkillSync

/**
 * Fire-and-forget wrapper for the three triggers (launch, the settings toggle
 * flipping on, a successful CLI update). SkillSync never rejects and
 * serializes overlapping calls, so this is safe to call from anywhere.
 */
function syncSkills(reason: string): void {
  void skillSync.sync().then((record) => {
    console.log(`skill sync (${reason}): ${record.ok ? 'ok' : 'FAILED'} — ${record.message}`)
  })
}

/**
 * Install the nodes that ship with the app (see shared/bundled.ts) on the
 * first launch that can reach a control plane. They are not marketplace rows:
 * the app fetches them through the same install API, then shows them in the
 * Agents view, so a brand-new user has a working software factory without
 * choosing anything.
 *
 * Called from the boot chain AFTER runAutostart resolves — installing needs a
 * live control plane, and autostart is what adopts or starts one. Best-effort
 * throughout: ensureBundledAgents never rejects, and a node that fails is left
 * unrecorded so the next launch retries it.
 */
async function provisionBundledAgents(): Promise<void> {
  // One snapshot answers both questions the plan needs: which nodes are
  // already installed, and whether the control plane we just booted actually
  // answered as an AgentField. It is read here rather than before autostart
  // because the active port may have moved (adopted, or freshly picked).
  const snapshot = await getSnapshot()
  // What this run actually installed, collected from onInstalled below —
  // ensureBundledAgents plans internally and reports nothing back, and the
  // key notice must speak only for the nodes that just arrived.
  const justProvisioned: string[] = []
  await ensureBundledAgents(
    {
      installed: snapshot.registry.agents.map((agent) => agent.name),
      provisioned: settings.provisionedBundled,
      skipEnv: process.env.AGENTFIELD_SKIP_BUNDLED,
      cliCommand: getCliCommand(),
      cloudActive: isCloudActive(),
      // recognized, not reachable: an unrelated service holding the port
      // would answer, and installing through it would fail every time.
      controlPlaneReachable: snapshot.controlPlane.recognized,
      registryReadable: snapshot.registry.exists && !snapshot.registry.error
    },
    {
      // Shares the app-wide install mutex, waiting when the user started an
      // install of their own — see acquireInstall/releaseInstall.
      install: async (name, onLine) => {
        await acquireInstall()
        try {
          return await installAgent(name, onLine)
        } finally {
          releaseInstall()
        }
      },
      // Remember the node was provisioned so it is never auto-installed
      // again: uninstalling a bundled node has to stick across launches.
      markProvisioned: async (name) => {
        settings = mergeSettings(settings, {
          provisionedBundled: [...settings.provisionedBundled, name]
        })
        await saveSettings(settingsFile(), settings)
      },
      hasInstallApi: () => createCpClient().hasInstallApi(),
      // Start it from the NEXT launch on, not now. Every bundled node needs an
      // API key the first-launch user has not entered yet, so starting one
      // here would only produce a dead node and an alarming badge; the Agents
      // row's "Needs keys" chip is the affordance that actually helps.
      onInstalled: async (name) => {
        justProvisioned.push(name)
        settings = mergeSettings(settings, {
          autostartAgents: [...settings.autostartAgents, name]
        })
        await saveSettings(settingsFile(), settings)
      },
      // bundledAgents.ts already prefixes every line with "bundled: " — the
      // module owns its own log voice, the way autostart.ts does.
      log: (message) => console.log(message)
    }
  )

  // Nothing was started above, on purpose — the bundled nodes need API keys
  // the first-launch user has not entered. On a login-item launch the app is
  // hidden in the tray, so the Agents row's "Needs keys" chip is telling an
  // empty room. One OS notification is the only thing that reaches the user
  // here. keyNotice.ts decides; this is just the Electron effect.
  await notifyUnresolvedKeys(justProvisioned, settings.keyNoticeShown, {
    // The one authoritative source: composed from the control plane's
    // per-agent secrets endpoint, i.e. the encrypted store `af run` reads.
    reports: () => getEnvReports(),
    supported: () => Notification.isSupported(),
    show: ({ title, body }) => {
      const notice = new Notification({ title, body })
      // The notice names keys; the Keys editor lives on the Agents rows, so
      // that is where a click has to land. navigate() also un-hides the
      // window, which is the whole point on a tray-only launch.
      notice.on('click', () => navigate('agents'))
      notice.show()
    },
    // The at-most-once latch: an announced name is persisted, and keyNotice.ts
    // filters on it, so no launch can raise the same notice twice.
    markNotified: async (agents) => {
      settings = mergeSettings(settings, {
        keyNoticeShown: [...settings.keyNoticeShown, ...agents]
      })
      await saveSettings(settingsFile(), settings)
    },
    log: (message) => console.log(message)
  })
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
    applyConnectionProfile(settings)
    nativeTheme.themeSource = settings.appearance
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
    const cliInitialization = await initializeCli(bundledCliPath())
    // Skills and the furrow client belong to local coding agents even when
    // their control plane and workspaces are remote.
    skillSync = new SkillSync(defaultSkillSyncDeps(skillSyncLogFile()))
    if (shouldSyncOnLaunch(settings)) syncSkills('launch')

    // macOS only: provision + install the af-tray menu-bar companion so a
    // desktop-app-only install gets the menu-bar icon. Runs after initializeCli
    // (it needs the managed bin dir to exist) and non-blocking, like syncSkills.
    syncTray(settings.trayCompanion)

    // The pinned aforge harness binary lands beside af in ~/.agentfield/bin, so a
    // desktop-only install can run harness-backed agents. Fire-and-forget: a failed
    // download must never delay or break app startup.
    void ensureAforgeCompanion().then((r) =>
      console.log(`aforge companion: ${r.ok ? 'ok' : 'FAILED'} — ${r.message}`)
    )

    // The snapshot carries the last skill-sync result along with the control-
    // plane view, so the renderer's existing 5s poll keeps the dashboard's
    // skill state honest without a channel (or a loop) of its own.
    // The snapshot also carries the bundled-node provisioning rows, so the
    // Agents view can show the two nodes arriving on a first launch off the
    // poll it already runs.
    ipcMain.handle('agentfield:snapshot', async () => {
      if (localControlPlaneRestart?.status === 'restart_required') {
        if (settings.cloud.enabled) {
          localControlPlaneRestart = reconcileLocalControlPlaneRestart(
            localControlPlaneRestart,
            null,
            true
          )
        } else {
          try {
            localControlPlaneRestart = reconcileLocalControlPlaneRestart(
              localControlPlaneRestart,
              await createCpClient().getVersion()
            )
          } catch {
            // Keep the actionable warning while the local server is unreachable.
          }
        }
      }
      return getSnapshot({
        skillSync: skillSync.last(),
        bundled: bundledStatuses(),
        localControlPlaneRestart
      })
    })
    // Bundled nodes stay listed so an uninstalled one can be reinstalled from the curated UI.
    ipcMain.handle('agentfield:catalog', () => [...BUNDLED_NODES, ...CATALOG])
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
        releaseInstall()
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
        releaseInstall()
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
    ipcMain.handle('agentfield:update', async (event, name: unknown, updateOptions: unknown) => {
      if (
        typeof name !== 'string' ||
        (
          updateOptions !== undefined &&
          (
            typeof updateOptions !== 'object' ||
            updateOptions === null ||
            Array.isArray(updateOptions) ||
            (
              (updateOptions as { force?: unknown }).force !== undefined &&
              typeof (updateOptions as { force?: unknown }).force !== 'boolean'
            )
          )
        )
      ) {
        return { ok: false, message: 'invalid update request' }
      }
      if (installInFlight) {
        return { ok: false, message: 'an install is already in progress' }
      }
      installInFlight = true
      try {
        return await updateAgent(
          name,
          (line) => {
            if (!event.sender.isDestroyed()) {
              event.sender.send('agentfield:install-progress', line)
            }
          },
          undefined,
          updateOptions as { force?: boolean } | undefined
        )
      } finally {
        releaseInstall()
      }
    })
    ipcMain.handle('agentfield:package-updates-check', () =>
      createCpClient().checkPackageUpdates()
    )
    ipcMain.handle(
      'agentfield:package-auto-update-set',
      async (_event, id: unknown, enabled: unknown) => {
        if (typeof id !== 'string' || typeof enabled !== 'boolean') {
          throw new Error('Invalid package auto-update request.')
        }
        return setAgentPackageAutoUpdate(id, enabled)
      }
    )
    ipcMain.handle('agentfield:package-maintenance-get', () =>
      createCpClient().getMaintenanceStatus()
    )
    ipcMain.handle('agentfield:package-maintenance-run', () =>
      createCpClient().runPackageMaintenance()
    )
    ipcMain.handle('agentfield:agent-action', (_event, action: unknown, name: unknown) => {
      if (
        typeof name !== 'string' ||
        (action !== 'start' && action !== 'stop' && action !== 'restart')
      ) {
        return { ok: false, message: 'invalid agent action' }
      }
      return runAgentAction(action as AgentAction, name)
    })
    ipcMain.handle('agentfield:start-control-plane', async () => {
      if (isCloudActive()) {
        return {
          ok: false,
          message: 'Cloud control plane active — local server management is disabled'
        }
      }
      const port = settings.controlPlanePort ?? (await pickFreePort())
      setActiveControlPlanePort(port)
      const result = await startControlPlane(port)
      if (result.ok && port !== settings.lastControlPlanePort) {
        settings = mergeSettings(settings, { lastControlPlanePort: port })
        await saveSettings(settingsFile(), settings)
      }
      return result
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
      void shell.openExternal(`${getBaseUrl()}${path}`)
      return true
    })
    ipcMain.handle('agentfield:cli-status', () => refreshCliStatus(bundledCliPath()))
    ipcMain.handle('agentfield:cli-update', async () => {
      const result = await installBundledCli(bundledCliPath())
      if (!result.ok) console.error(result.message)
      // A newer af ships a newer skill catalog: re-sync so the skills the
      // coding agents see match the CLI that just landed.
      if (shouldSyncOnCliUpdate(result.ok, settings)) syncSkills('cli update')
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

    const cloudUpdater = new CloudUpdateChecker({
      enabled: () => settings.cloud.enabled,
      getVersion: () => createCpClient().getVersion(),
      getTfstateImage: () => deploymentStateInfo(deployPaths().workspaceDir)?.image ?? null,
      canApplyUpdate: (running) => {
        const state = deploymentStateInfo(deployPaths().workspaceDir)
        return cloudUpdateApplyPath({
          running,
          tfstateImage: state?.image ?? null,
          tfstateServiceId: state?.serviceId,
          tfstateUrl: state?.url,
          connectedServerUrl: settings.cloud.serverUrl
        }) !== 'none'
      },
      canManageRailway: (running) => {
        const state = deploymentStateInfo(deployPaths().workspaceDir)
        return cloudUpdateRailwayControlsAvailable({
          running,
          tfstateImage: state?.image ?? null,
          tfstateServiceId: state?.serviceId,
          tfstateEnvironmentId: state?.environmentId,
          tfstateUrl: state?.url,
          connectedServerUrl: settings.cloud.serverUrl
        })
      },
      applyUpdate: async (running, tfstateImage) => {
        const paths = deployPaths()
        const state = deploymentStateInfo(paths.workspaceDir)
        const options = {
          running,
          tfstateImage,
          tfstateServiceId: state?.serviceId,
          tfstateUrl: state?.url,
          connectedServerUrl: settings.cloud.serverUrl
        }
        return applyCloudUpdateWithRailwayToken(options, {
          getAccessToken: () => getFreshAccessToken(authDeps()),
          createApplyDeps: (token) => {
            const api = createRailwayApi(token)
            return {
              refreshAndDeploy: async (targetImage) => {
                if (!state?.workspaceId) {
                  return {
                    ok: false,
                    message: 'Desktop deployment state is missing its Railway workspace. Choose the deployment workspace and re-run deploy.'
                  }
                }
                const deployOptions = {
                  railwayToken: token,
                  workspaceId: state.workspaceId,
                  ...paths
                }
                const refreshed = await refreshDeploymentState(deployOptions)
                if (!refreshed.ok) return refreshed
                return runDeploy({ ...deployOptions, image: targetImage })
              },
              setServiceImage: (serviceId, environmentId, image) =>
                api.setServiceImage(serviceId, environmentId, image),
              redeploy: (serviceId, environmentId) => api.redeploy(serviceId, environmentId),
              getVersion: () => createCpClient().getVersion(),
              getMaintenanceStatus: () => createCpClient().getMaintenanceStatus()
            }
          }
        })
      },
      onCompletedCheck: (running) => {
        const serviceId = running?.hosting.service_id
        if (serviceId) {
          const next = settingsForCloudService(settings, serviceId)
          if (next !== settings) {
            settings = next
            void saveSettings(settingsFile(), settings).catch((error) => {
              console.error('could not reset cloud auto-update preference:', error)
            })
          }
        }
      },
      onStatus: (status) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('agentfield:cloud-update-status', status)
        }
      }
    })
    ipcMain.handle('agentfield:cloud-update-check', () => cloudUpdater.check())
    ipcMain.handle('agentfield:cloud-update-apply', async () => {
      if (cloudDeployInFlight) {
        return { ok: false, message: 'A cloud deployment is already running.' }
      }
      cloudDeployInFlight = true
      try {
        return await cloudUpdater.apply()
      } finally {
        cloudDeployInFlight = false
      }
    })
    ipcMain.handle('agentfield:cloud-update-dismiss', async (_event, version: unknown) => {
      if (typeof version !== 'string' || version === '') return
      settings = settingsWithDismissedCloudUpdate(settings, version)
      await saveSettings(settingsFile(), settings)
    })
    ipcMain.handle(
      'agentfield:cloud-auto-update-set',
      async (_event, mode: unknown) => {
        if (mode !== 'off' && mode !== 'nightly' && mode !== 'weekends' && mode !== 'anytime') {
          return { ok: false, message: 'Choose Off, Nightly, Weekends, or Anytime.' }
        }
        if (cloudDeployInFlight) {
          return { ok: false, message: 'Wait for the current cloud deployment to finish, then try again.' }
        }
        const state = deploymentStateInfo(deployPaths().workspaceDir)
        const result = await setCloudAutoUpdateSchedule({
          mode: mode as CloudAutoUpdateMode,
          connectedServerUrl: settings.cloud.serverUrl,
          tfstate: state
            ? {
                serviceId: state.serviceId,
                environmentId: state.environmentId,
                url: state.url
              }
            : null
        }, {
          getAccessToken: () => getFreshAccessToken(authDeps()),
          getVersion: () => createCpClient().getVersion(),
          setSchedule: (token, serviceId, environmentId, selectedMode) =>
            createRailwayApi(token).setAutoUpdateSchedule(
              serviceId,
              environmentId,
              selectedMode
            )
        })
        if (result.ok && result.serviceId) {
          try {
            settings = await persistCloudAutoUpdatePreference(
              settings,
              mode as CloudAutoUpdateMode,
              result.serviceId,
              (next) => saveSettings(settingsFile(), next)
            )
          } catch (error) {
            return {
              ok: false,
              message: `Railway saved the schedule, but Desktop could not persist the preference: ${error instanceof Error ? error.message : String(error)}.`
            }
          }
        }
        return { ok: result.ok, message: result.message }
      }
    )
    cloudUpdater.startAutoCheck()
    ipcMain.handle('agentfield:settings-get', () => settings)
    ipcMain.handle('agentfield:cloud-profile-set', async (_event, profile: unknown) => {
      const value = typeof profile === 'object' && profile !== null
        ? profile as Record<string, unknown>
        : null
      if (
        !value ||
        typeof value.enabled !== 'boolean' ||
        typeof value.serverUrl !== 'string' ||
        typeof value.apiKey !== 'string'
      ) {
        throw new Error('Invalid cloud profile request.')
      }
      const next = settingsWithCloudProfile(settings, {
        enabled: value.enabled,
        serverUrl: value.serverUrl,
        apiKey: value.apiKey
      })
      await saveSettings(settingsFile(), next)
      settings = next
      applyConnectionProfile(settings)
      return settings
    })
    ipcMain.handle('agentfield:cloud-test', (_event, url: string, apiKey: string) =>
      testCloudConnection(url, apiKey)
    )
    ipcMain.handle('agentfield:cloud-deploy-railway', () =>
      shell.openExternal(RAILWAY_TEMPLATE_URL)
    )
    ipcMain.handle('agentfield:railway-status', async () => {
      const deps = authDeps()
      const { workspaceDir, binaryDir } = deployPaths()
      const token = isLoggedIn(deps) ? await getFreshAccessToken(deps) : null
      return loadRailwayStatus({
        token,
        engineAvailable: resolveTofuBinary(binaryDir) !== null,
        hasDeployment: hasDeployment(workspaceDir),
        deploymentWorkspaceId: deploymentStateInfo(workspaceDir)?.workspaceId ?? null,
        listWorkspaces: (accessToken) => listWorkspaces(accessToken)
      })
    })
    ipcMain.handle('agentfield:railway-login', async () => {
      const deps = authDeps()
      const result = await loginWithRailway(deps)
      if (!result.ok) return result
      const token = await getFreshAccessToken(deps)
      return { ...result, workspaces: token ? await listWorkspaces(token) : [] }
    })
    ipcMain.handle('agentfield:railway-logout', () => logout(authDeps()))
    ipcMain.handle('agentfield:cloud-deploy', async (event, workspaceId: unknown) => {
      if (typeof workspaceId !== 'string' || workspaceId === '') {
        return { ok: false, message: 'Choose a Railway workspace first' }
      }
      if (cloudDeployInFlight) return { ok: false, message: 'A cloud deployment is already running' }
      cloudDeployInFlight = true
      try {
        const token = await getFreshAccessToken(authDeps())
        if (!token) return { ok: false, message: 'Sign in with Railway first' }
        const result = await runDeploy({
          railwayToken: token,
          workspaceId,
          ...deployPaths(),
          onLine: (line) => {
            if (!event.sender.isDestroyed()) {
              event.sender.send('agentfield:cloud-deploy-progress', line)
            }
          }
        })
        let deployMessage = result.message
        if (result.ok && result.url && result.apiKey) {
          const serviceId = result.serviceId ?? null
          let appliedMode =
            serviceId && settings.cloud.autoUpdateServiceId === serviceId
              ? settings.cloud.autoUpdate
              : null
          if (result.serviceId && result.environmentId) {
            const scheduleMode = autoUpdateModeAfterDeploy({
              firstDeploy: result.firstDeploy === true,
              serviceId: result.serviceId,
              storedMode: settings.cloud.autoUpdate,
              storedServiceId: settings.cloud.autoUpdateServiceId
            })
            if (scheduleMode) {
              try {
                await createRailwayApi(token).setAutoUpdateSchedule(
                  result.serviceId,
                  result.environmentId,
                  scheduleMode
                )
                appliedMode = scheduleMode
                deployMessage = `${deployMessage} Railway image auto-updates are set to ${scheduleMode === 'nightly' ? 'Nightly (02:00–06:00 UTC every day)' : scheduleMode}.`
              } catch (error) {
                deployMessage = `${deployMessage} Automatic updates could not be scheduled: ${error instanceof Error ? error.message : String(error)}. The deployment is healthy; choose a window below to retry.`
              }
            } else {
              deployMessage = `${deployMessage} Railway image auto-updates are not set; choose a window below.`
            }
          } else {
            deployMessage = `${deployMessage} Automatic updates could not be scheduled because Railway service outputs were missing. Re-run deploy to reconcile them.`
          }
          settings = mergeSettings(settings, {
            cloud: {
              ...settings.cloud,
              enabled: true,
              serverUrl: result.url,
              apiKey: result.apiKey,
              autoUpdate: appliedMode,
              autoUpdateServiceId: serviceId
            }
          })
          await saveSettings(settingsFile(), settings)
          applyConnectionProfile(settings)
        }
        return {
          ok: result.ok,
          url: result.url,
          furrowAddress: result.furrowAddress,
          message: deployMessage
        }
      } finally {
        cloudDeployInFlight = false
      }
    })
    ipcMain.handle('agentfield:cloud-destroy', async () => {
      if (cloudDeployInFlight) return { ok: false, message: 'A cloud deployment is already running' }
      cloudDeployInFlight = true
      try {
        const token = await getFreshAccessToken(authDeps())
        if (!token) return { ok: false, message: 'Sign in with Railway first' }
        const result = await runDestroy({ railwayToken: token, workspaceId: '', ...deployPaths() })
        if (result.ok) {
          settings = mergeSettings(settings, {
            cloud: {
              ...settings.cloud,
              enabled: false,
              autoUpdate: null,
              autoUpdateServiceId: null
            }
          })
          await saveSettings(settingsFile(), settings)
          applyConnectionProfile(settings)
        }
        return result
      } finally {
        cloudDeployInFlight = false
      }
    })
    ipcMain.handle('agentfield:settings-set', async (_event, patch: unknown) => {
      const prev = settings
      settings = mergeSettings(settings, patch)
      applyLoginItem(settings)
      if (settings.appearance !== prev.appearance) {
        nativeTheme.themeSource = settings.appearance
      }
      // macOS: reflect a flipped tray toggle (install ↔ uninstall) right away.
      if (settings.trayCompanion !== prev.trayCompanion) syncTray(settings.trayCompanion)
      // Skills toggled on: install them now instead of at the next launch.
      // (Off is not a trigger — `af skill install` has no uninstall side, and
      // the dashboard reports the setting itself as off.)
      if (shouldSyncOnSettingsChange(prev, settings)) syncSkills('settings')
      await saveSettings(settingsFile(), settings)
      applyConnectionProfile(settings)
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
    // The port autostart ends up on (adopted or freshly picked) is persisted
    // so the next app start finds this control plane again instead of
    // spawning a second one somewhere else.
    //
    // Bundled-node provisioning is chained onto the SAME promise rather than
    // started beside it: it installs through the control plane, so it has to
    // wait for the one autostart adopted or brought up.
    void runDesktopBootChain({
      userPathReady,
      runAutostart: () =>
        runAutostart(
          settings,
          (message) => console.log(message),
          async (port) => {
            settings = mergeSettings(settings, { lastControlPlanePort: port })
            await saveSettings(settingsFile(), settings)
          }
        ),
      recoverAutostartFailure,
      afterAutostart: async (autostart) => {
        if (cliInitialization.managedBinaryReplaced) {
          localControlPlaneRestart = await restartAdoptedControlPlaneAfterCliSwap(
            {
              managedBinaryReplaced: true,
              platform: process.platform,
              cloudEnabled: settings.cloud.enabled,
              autostart,
              cliVersion: cliInitialization.status.version
            },
            {
              getVersion: () => createCpClient().getVersion()
            }
          )
          console.log(`local control-plane update: ${localControlPlaneRestart.message}`)
        }
      },
      provisionBundledAgents,
      checkPackageUpdates: () => createCpClient().checkPackageUpdates().then(() => undefined),
      log: (message) => console.log(message),
      warn: (message) => console.warn(message),
      error: (message, error) => console.error(message, error)
    })

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
