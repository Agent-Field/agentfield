import { contextBridge, ipcRenderer } from 'electron'
import type { AgentFieldApi } from '../shared/types'

// Sandboxed preload: only contextBridge/ipcRenderer are used, no Node APIs.
const api: AgentFieldApi = {
  getSnapshot: () => ipcRenderer.invoke('agentfield:snapshot'),
  getCatalog: () => ipcRenderer.invoke('agentfield:catalog'),
  install: (name) => ipcRenderer.invoke('agentfield:install', name),
  onInstallProgress: (listener) => {
    const wrapped = (_event: Electron.IpcRendererEvent, line: string) => listener(line)
    ipcRenderer.on('agentfield:install-progress', wrapped)
    return () => ipcRenderer.removeListener('agentfield:install-progress', wrapped)
  },
  onNavigate: (listener) => {
    const wrapped = (_event: Electron.IpcRendererEvent, view: string) => listener(view)
    ipcRenderer.on('agentfield:navigate', wrapped)
    return () => ipcRenderer.removeListener('agentfield:navigate', wrapped)
  },
  platform: process.platform
}

contextBridge.exposeInMainWorld('agentfield', api)
