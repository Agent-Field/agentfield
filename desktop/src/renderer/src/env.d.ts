declare module '*.css'

interface Window {
  /** Exposed by src/preload/index.ts via contextBridge. */
  agentfield: import('../../shared/types').AgentFieldApi
}
