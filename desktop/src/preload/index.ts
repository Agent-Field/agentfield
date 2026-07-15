import { contextBridge, ipcRenderer } from 'electron'
import type { AgentFieldApi } from '../shared/types'

// Sandboxed preload: only contextBridge/ipcRenderer are used, no Node APIs.
const api: AgentFieldApi = {
  getSnapshot: () => ipcRenderer.invoke('agentfield:snapshot'),
  getCatalog: () => ipcRenderer.invoke('agentfield:catalog'),
  install: (name) => ipcRenderer.invoke('agentfield:install', name),
  installFromSource: (source) => ipcRenderer.invoke('agentfield:install-source', source),
  uninstall: (name) => ipcRenderer.invoke('agentfield:uninstall', name),
  update: (name) => ipcRenderer.invoke('agentfield:update', name),
  onInstallProgress: (listener) => {
    const wrapped = (_event: Electron.IpcRendererEvent, line: string) => listener(line)
    ipcRenderer.on('agentfield:install-progress', wrapped)
    return () => ipcRenderer.removeListener('agentfield:install-progress', wrapped)
  },
  agentAction: (action, name) => ipcRenderer.invoke('agentfield:agent-action', action, name),
  getEnvReports: () => ipcRenderer.invoke('agentfield:env-reports'),
  setAgentSecret: (agent, key, value) =>
    ipcRenderer.invoke('agentfield:secret-set', agent, key, value),
  revokeAgentSecret: (agent, key) => ipcRenderer.invoke('agentfield:secret-revoke', agent, key),
  listSecrets: () => ipcRenderer.invoke('agentfield:secrets-list'),
  revokeSecret: (key, scope) => ipcRenderer.invoke('agentfield:secrets-revoke', key, scope),
  getSettings: () => ipcRenderer.invoke('agentfield:settings-get'),
  setSettings: (patch) => ipcRenderer.invoke('agentfield:settings-set', patch),
  getCliStatus: () => ipcRenderer.invoke('agentfield:cli-status'),
  updateCli: () => ipcRenderer.invoke('agentfield:cli-update'),
  getAppUpdateStatus: () => ipcRenderer.invoke('agentfield:app-update-get'),
  checkForAppUpdate: () => ipcRenderer.invoke('agentfield:app-update-check'),
  installAppUpdate: () => ipcRenderer.invoke('agentfield:app-update-install'),
  onAppUpdateStatus: (listener) => {
    const wrapped = (
      _event: Electron.IpcRendererEvent,
      status: Parameters<typeof listener>[0]
    ) => listener(status)
    ipcRenderer.on('agentfield:app-update-status', wrapped)
    return () => ipcRenderer.removeListener('agentfield:app-update-status', wrapped)
  },
  onNavigate: (listener) => {
    const wrapped = (_event: Electron.IpcRendererEvent, view: string) => listener(view)
    ipcRenderer.on('agentfield:navigate', wrapped)
    return () => ipcRenderer.removeListener('agentfield:navigate', wrapped)
  },
  announceReady: () => ipcRenderer.invoke('agentfield:renderer-ready'),
  openWebUI: (path) => ipcRenderer.invoke('agentfield:open-web-ui', path),
  platform: process.platform
}

contextBridge.exposeInMainWorld('agentfield', api)
