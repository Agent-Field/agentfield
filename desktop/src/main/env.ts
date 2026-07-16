// Child-process PATH seam. A macOS app launched from Finder/Dock inherits
// launchd's minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) — not the user's
// shell PATH. The bundled `af` still resolves (we spawn it by absolute path),
// but af's OWN subprocesses (go, uv, python3, claude, codex) are looked up on
// PATH and are then not found. So before spawning af we swap in the real
// login-shell PATH. Windows GUI apps already inherit the user PATH, so this is
// a darwin/linux-only concern and win32 is left untouched.
//
// No electron imports: the shell-spawn and the pure parsing/merging are
// injectable so everything here is unit-testable under plain vitest.

import { spawn } from 'node:child_process'
import { homedir } from 'node:os'
import { posix } from 'node:path'

// Sentinel markers fence the printed PATH off from any banner/motd an
// interactive rc file emits, so we can extract it cleanly.
const PATH_START = '__AF_PATH_START__'
const PATH_END = '__AF_PATH_END__'

/** A login shell that reads its rc files can hang; cap the wait. */
const RESOLVE_TIMEOUT_MS = 3_000

/**
 * Directories AgentField-managed tools and the common package managers live
 * in but a Finder/Dock launch's minimal PATH omits. Merged in unconditionally
 * so `af`'s toolchain resolves even when the shell probe fails.
 */
export function wellKnownBinDirs(home: string = homedir()): string[] {
  return [
    '/opt/homebrew/bin', // Homebrew on Apple silicon
    '/usr/local/bin', // Homebrew on Intel, and manual installs
    posix.join(home, '.agentfield', 'bin'), // where the curl installer / this app put af
    posix.join(home, '.cargo', 'bin'), // rustup (uv, some agent toolchains)
    posix.join(home, '.local', 'bin') // pipx / uv / user-scope python tools
  ]
}

/** Pull the PATH the login shell printed from between the sentinel markers. */
export function extractShellPath(output: string): string | null {
  const start = output.indexOf(PATH_START)
  const end = output.indexOf(PATH_END)
  if (start === -1 || end === -1 || end <= start) return null
  const value = output.slice(start + PATH_START.length, end)
  return value.length > 0 ? value : null
}

/**
 * Merge PATH-like inputs into one string, de-duped with first occurrence
 * winning so priority order is preserved. Empty entries and blank inputs are
 * dropped. Only the darwin/linux branches ever merge (win32 returns its PATH
 * untouched above), so the separator is POSIX regardless of the host.
 */
export function mergePaths(
  inputs: ReadonlyArray<string | null | undefined>,
  sep: string = posix.delimiter
): string {
  const seen = new Set<string>()
  const out: string[] = []
  for (const input of inputs) {
    if (!input) continue
    for (const entry of input.split(sep)) {
      if (entry === '' || seen.has(entry)) continue
      seen.add(entry)
      out.push(entry)
    }
  }
  return out.join(sep)
}

export interface EnvResolveDeps {
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  home?: string
  /**
   * Run the login shell and resolve with its stdout (the marker-wrapped PATH),
   * or null on failure/timeout. Injected in tests; defaults to a real spawn.
   */
  runLoginShell?: () => Promise<string | null>
}

/**
 * The PATH to use before (or instead of) the async login-shell probe:
 * process.env.PATH plus the well-known dirs. No subprocess, so it is safe to
 * call synchronously on the spawn path. On win32 the process PATH is already
 * the user's, so it is returned unchanged.
 */
export function syncResolvedPath(deps: EnvResolveDeps = {}): string {
  const platform = deps.platform ?? process.platform
  const env = deps.env ?? process.env
  if (platform === 'win32') return env.PATH ?? ''
  return mergePaths([env.PATH, ...wellKnownBinDirs(deps.home)])
}

/** Spawn the user's login shell and print its PATH between the markers. */
function spawnLoginShell(env: NodeJS.ProcessEnv): Promise<string | null> {
  return new Promise((resolve) => {
    const shell = env.SHELL || '/bin/zsh'
    let output = ''
    let settled = false
    const done = (value: string | null) => {
      if (settled) return
      settled = true
      resolve(value)
    }
    // -i (interactive) + -l (login) source both the profile and rc files, so
    // whatever the user's PATH edits live in are applied; -c runs the printf.
    const child = spawn(
      shell,
      ['-ilc', `printf "${PATH_START}%s${PATH_END}" "$PATH"`],
      { windowsHide: true }
    )
    const timer = setTimeout(() => {
      child.kill()
      done(null)
    }, RESOLVE_TIMEOUT_MS)
    child.stdout.on('data', (chunk: Buffer) => {
      output += chunk.toString('utf8')
    })
    child.on('error', () => {
      clearTimeout(timer)
      done(null)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      done(code === 0 ? output : null)
    })
  })
}

/**
 * Resolve the user's real PATH: the login-shell PATH (best-effort) merged with
 * the current process PATH and the well-known dirs, de-duped in that priority
 * order. Never rejects — a failed/timed-out shell just falls back to the sync
 * inputs. On win32 returns the process PATH unchanged.
 */
export async function resolveUserPath(deps: EnvResolveDeps = {}): Promise<string> {
  const platform = deps.platform ?? process.platform
  const env = deps.env ?? process.env
  if (platform === 'win32') return env.PATH ?? ''

  const run = deps.runLoginShell ?? (() => spawnLoginShell(env))
  let shellPath: string | null = null
  try {
    const output = await run()
    shellPath = output ? extractShellPath(output) : null
  } catch {
    shellPath = null
  }
  return mergePaths([shellPath, env.PATH, ...wellKnownBinDirs(deps.home)])
}

// ---- Cached child env --------------------------------------------------------
// Resolution is async; the cache lets every spawn site read a PATH
// synchronously. Until initUserPath() lands it holds the sync fallback, so
// nothing breaks if a spawn happens before the shell probe finishes.

let cachedPath: string | null = null

/**
 * Resolve the login-shell PATH once and cache it. Best-effort — on failure the
 * sync fallback stays in effect. Call once at app startup.
 */
export async function initUserPath(deps: EnvResolveDeps = {}): Promise<string> {
  cachedPath = await resolveUserPath(deps)
  return cachedPath
}

/** For tests: forget the cached PATH so the sync fallback is used again. */
export function resetUserPathCache(): void {
  cachedPath = null
}

/**
 * Environment for child processes: inherit ours but with the fuller PATH so
 * `af`'s own subprocesses resolve on a Finder launch. `extra` overrides
 * inherited vars (except PATH, which is always the resolved one).
 */
export function childEnv(extra?: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return { ...process.env, ...extra, PATH: cachedPath ?? syncResolvedPath() }
}
