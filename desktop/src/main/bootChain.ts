import type { AutostartResult } from './autostart'

export interface DesktopBootChainDeps {
  userPathReady: Promise<unknown>
  runAutostart: () => Promise<AutostartResult>
  recoverAutostartFailure: (error: unknown) => AutostartResult
  afterAutostart: (result: AutostartResult) => Promise<void>
  provisionBundledAgents: () => Promise<void>
  checkPackageUpdates: () => Promise<void>
  log: (message: string) => void
  warn: (message: string) => void
  error: (message: string, error: unknown) => void
}

/** Ordered desktop startup effects with independent best-effort boundaries.
 * Provisioning and the package check must run even when an earlier stage fails. */
export async function runDesktopBootChain(deps: DesktopBootChainDeps): Promise<void> {
  try {
    await deps.userPathReady
  } catch (error) {
    deps.error('login-shell PATH resolution failed:', error)
  }

  let autostart: AutostartResult
  try {
    autostart = await deps.runAutostart()
  } catch (error) {
    autostart = deps.recoverAutostartFailure(error)
  }

  try {
    await deps.afterAutostart(autostart)
  } catch (error) {
    deps.error('post-autostart update check failed:', error)
  }

  try {
    await deps.provisionBundledAgents()
  } catch (error) {
    deps.error('bundled provisioning failed:', error)
  }

  try {
    await deps.checkPackageUpdates()
    deps.log('package update check: complete')
  } catch (error) {
    deps.warn(`package update check: ${error instanceof Error ? error.message : String(error)}`)
  }
}
