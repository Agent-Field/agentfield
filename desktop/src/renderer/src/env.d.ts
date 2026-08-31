declare module '*.css'
declare module '*.svg' {
  const src: string
  export default src
}

interface Window {
  /** Exposed by src/preload/index.ts via contextBridge. */
  agentfield: import('../../shared/types').AgentFieldApi
}
