import { contextBridge, ipcRenderer } from 'electron'
import type { AgentFieldApi } from '../shared/types'

// Sandboxed preload: only contextBridge/ipcRenderer are used, no Node APIs.
const api: AgentFieldApi = {
  getSnapshot: () => ipcRenderer.invoke('agentfield:snapshot')
}

contextBridge.exposeInMainWorld('agentfield', api)
