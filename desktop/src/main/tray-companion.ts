import { execFile } from 'node:child_process'
import { childEnv } from './env'

export interface TrayCompanionResult {
  atLogin: boolean
}

type Run = (command: string, args: string[]) => Promise<string>

function runHelper(command: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(command, args, { env: childEnv(), timeout: 15_000 }, (error, stdout, stderr) => {
      if (error) reject(new Error(stderr.trim() || error.message))
      else resolve(stdout)
    })
  })
}

// Always call the helper packaged with this version of Desktop. The managed
// CLI copy may be older, and its uninstall command can stop the server.
export async function syncTrayCompanion(
  bundledPath: string,
  visible: boolean,
  atLogin: boolean | null,
  run: Run = runHelper
): Promise<TrayCompanionResult> {
  const login = atLogin === null ? 'auto' : atLogin ? 'login' : 'session'
  const output = await run(bundledPath, ['desktop-tray', visible ? 'show' : 'hide', login])
  const state = /^tray-at-login=(true|false)$/m.exec(output)
  if (!state) throw new Error('The tray helper did not report its login setting')
  return { atLogin: state[1] === 'true' }
}
