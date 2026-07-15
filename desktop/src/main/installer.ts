// Install seam: runs `af install <source>` for vetted catalog entries.
// The af CLI is the single contract shared by agents, this app, and
// developers — the app never reimplements install logic, it shells out.
// Deliberately does NOT import from 'electron' so it stays unit-testable.
//
// Security: the renderer may send raw sources on exactly ONE channel — the
// "Install from repository" flow (installFromSource). Every other install
// path takes a curated catalog NAME, never a source, and refuses anything
// unknown. The compensating control for the relaxed channel is strict
// main-process shape validation in parseRepoSource: only an
// https://github.com/<owner>/<repo>[//<subdir>] source survives, and because
// every accepted value begins with "https://github.com/" it can never be read
// as a CLI flag when passed as one argv element to spawn (no shell).

import { spawn } from 'node:child_process'
import { catalogEntry } from '../shared/catalog'
import type { InstallResult } from '../shared/types'
import { readInstalledAgents } from './agentfield'
import { runAgentAction } from './agents'
import { getCliCommand } from './cli'
import { childEnv } from './env'

// CSI sequences (colors, cursor movement, erase-line spinner frames) and OSC
// sequences (terminal titles), per ECMA-48. Written with \u escapes so no
// invisible control characters live in this source file.
const ANSI_PATTERN = new RegExp(
  '\\u001b\\[[0-9;?]*[A-Za-z]|\\u001b\\][^\\u0007\\u001b]*(?:\\u0007|\\u001b\\\\)?',
  'g'
)

/**
 * The af CLI double-reports failures: a human line, then zerolog JSON like
 * {"level":"error","error":"invalid package structure: …","message":"Error
 * executing root command"}. Raw JSON in an install row is unreadable, so
 * unwrap it to the underlying error text. Anything that isn't a zerolog
 * object (no `level`) passes through untouched — agent output may
 * legitimately contain JSON.
 */
function unwrapLogLine(line: string): string {
  if (!line.startsWith('{') || !line.endsWith('}')) return line
  try {
    const obj = JSON.parse(line) as Record<string, unknown>
    if (typeof obj.level !== 'string') return line
    const detail = typeof obj.error === 'string' && obj.error ? obj.error : null
    const message = typeof obj.message === 'string' && obj.message ? obj.message : null
    return detail ?? message ?? line
  } catch {
    return line
  }
}

/**
 * Normalize a chunk of `af install` output into displayable lines: strip
 * ANSI color/spinner escapes, split on newlines and carriage returns
 * (spinner frames), unwrap zerolog JSON lines, drop empties.
 */
export function sanitizeInstallOutput(chunk: string): string[] {
  return chunk
    .replace(ANSI_PATTERN, '')
    .split(/[\r\n]+/)
    .map((line) => unwrapLogLine(line.trim()))
    .filter((line) => line.length > 0)
}

/**
 * Build the argv for installing a catalog entry. Returns null for names not
 * in the curated catalog — the renderer only ever sends names, and anything
 * unknown is refused rather than passed to a shell. `force` maps to
 * `af install --force`, the CLI's reinstall-in-place (package dir and binary
 * are replaced; the registry entry and secrets are untouched).
 */
export function installCommand(
  name: string,
  force = false
): { command: string; args: string[] } | null {
  const entry = catalogEntry(name)
  if (!entry) return null
  // spawn() without a shell; the command is whatever CLI resolution picked
  // (managed copy, PATH `af`, or the app's bundled binary — see main/cli.ts).
  const args = ['install', entry.source]
  if (force) args.push('--force')
  return { command: getCliCommand(), args }
}

/**
 * Spawn `af <args>` (no shell), forward sanitized output lines to onLine as
 * they arrive, and resolve (never reject) with the outcome. `successMessage`
 * is produced lazily so a caller can name whatever it just installed. Shared
 * by installAgent and installFromSource so the spawn/stream/close logic lives
 * in exactly one place.
 */
function runInstall(
  command: string,
  args: string[],
  onLine: (line: string) => void,
  successMessage: () => string
): Promise<InstallResult> {
  return new Promise((resolve) => {
    let lastLine = ''
    const forward = (chunk: Buffer) => {
      for (const line of sanitizeInstallOutput(chunk.toString('utf8'))) {
        // Spinner frames repeat the same text many times a second; only
        // forward changes so the IPC channel stays quiet.
        if (line !== lastLine) {
          lastLine = line
          onLine(line)
        }
      }
    }

    const child = spawn(command, args, { windowsHide: true, env: childEnv() })
    child.stdout.on('data', forward)
    child.stderr.on('data', forward)
    child.on('error', (err: NodeJS.ErrnoException) => {
      resolve({
        ok: false,
        message:
          err.code === 'ENOENT'
            ? 'The AgentField CLI (af) was not found on PATH. Install it first: https://agentfield.ai/docs'
            : `Failed to run af install: ${err.message}`
      })
    })
    child.on('close', (code) => {
      resolve(
        code === 0
          ? { ok: true, message: successMessage() }
          : { ok: false, message: lastLine || `af install exited with code ${code}` }
      )
    })
  })
}

/**
 * Run `af install` for the named catalog entry, forwarding sanitized output
 * lines to onLine as they arrive. Resolves (never rejects) with the outcome.
 */
export function installAgent(
  name: string,
  onLine: (line: string) => void,
  force = false
): Promise<InstallResult> {
  const cmd = installCommand(name, force)
  if (!cmd) {
    return Promise.resolve({ ok: false, message: `"${name}" is not in the install catalog` })
  }
  return runInstall(cmd.command, cmd.args, onLine, () => `${name} installed`)
}

// The one host we install from. Every accepted source starts with this literal
// prefix, which is why a validated value can never be mistaken for a CLI flag.
const GITHUB_PREFIX = 'https://github.com/'
// owner and repo: alphanumerics, underscore, dot, dash — but no *leading* dash,
// so a value can never start with `-` and be read as a flag. A trailing `.git`
// is allowed (it falls out of the dot in the class) and kept as-is; `af`
// accepts it.
const OWNER_REPO = /^[A-Za-z0-9_.][A-Za-z0-9_.-]*\/[A-Za-z0-9_.][A-Za-z0-9_.-]*$/
// //<subdir> selector: slash-separated segments over the same class, no leading
// `-` or `/` (the `..` traversal check is separate).
const SUBDIR = /^[A-Za-z0-9_.][A-Za-z0-9_./-]*$/

/**
 * Validate and normalize a pasted install source. Accepts ONLY a GitHub HTTPS
 * repo URL — `https://github.com/<owner>/<repo>` — optionally followed by the
 * `//<subdir>` selector that picks one node out of a multi-node repo (e.g.
 * `https://github.com/Agent-Field/pr-af//go`). A pasted browser URL of the
 * plain repo is tolerated: a single trailing slash is stripped, a trailing
 * `.git` is kept. Everything else is refused (returns null): http://, other
 * hosts, ssh/git@, query strings, fragments, embedded whitespace, `..`
 * traversal, and anything starting with `-`. The returned string is passed as
 * one argv element to spawn without a shell.
 */
export function parseRepoSource(input: string): string | null {
  const trimmed = input.trim()
  if (!trimmed) return null
  // No embedded whitespace, no browser cruft — these never appear in a bare
  // repo URL and would only smuggle intent.
  if (/\s/.test(trimmed)) return null
  if (trimmed.includes('?') || trimmed.includes('#')) return null
  // Anchors the host: rejects http://, other hosts, ssh/git@ in one check.
  if (!trimmed.startsWith(GITHUB_PREFIX)) return null

  const rest = trimmed.slice(GITHUB_PREFIX.length)
  // Split off the optional //<subdir> selector at its first occurrence.
  const sep = rest.indexOf('//')
  const repoRaw = sep === -1 ? rest : rest.slice(0, sep)
  const subdirRaw = sep === -1 ? null : rest.slice(sep + 2)

  // Tolerate a pasted browser URL: drop one trailing slash from the repo part.
  const repo = repoRaw.replace(/\/$/, '')
  if (!OWNER_REPO.test(repo)) return null

  if (subdirRaw === null) return `${GITHUB_PREFIX}${repo}`

  const subdir = subdirRaw.replace(/\/$/, '')
  if (!subdir || subdir.includes('..') || !SUBDIR.test(subdir)) return null
  return `${GITHUB_PREFIX}${repo}//${subdir}`
}

/**
 * Install a node from a pasted GitHub repository source. Validates and
 * normalizes via parseRepoSource (null → resolve {ok:false} without spawning),
 * then reuses the same `af install` spawn/stream/close path as installAgent.
 * No --force path — this only ever installs, never reinstalls in place.
 */
export function installFromSource(
  source: string,
  onLine: (line: string) => void
): Promise<InstallResult> {
  const normalized = parseRepoSource(source)
  if (!normalized) {
    return Promise.resolve({
      ok: false,
      message: 'Enter a GitHub repository URL, e.g. https://github.com/org/repo (or …/repo//subdir)'
    })
  }
  return runInstall(getCliCommand(), ['install', normalized], onLine, () => `Installed from ${normalized}`)
}

/**
 * Update an installed catalog agent to the latest version of its source:
 * stop it if it is running, `af install <source> --force` (reinstall in
 * place — registry entry and secrets survive), then restore the previous run
 * state: restart only what was running, leave stopped agents stopped. Phase
 * markers ("Stopping…", "Restarting…") ride the same progress channel as the
 * install output. Resolves (never rejects) with the outcome.
 */
export async function updateAgent(
  name: string,
  onLine: (line: string) => void
): Promise<InstallResult> {
  const entry = catalogEntry(name)
  if (!entry) {
    return { ok: false, message: `"${name}" is not in the install catalog` }
  }
  const registry = await readInstalledAgents()
  const installed = registry.agents.find((agent) => agent.name === name)
  if (!installed) {
    return { ok: false, message: `"${name}" is not installed — install it first` }
  }

  // The package binary cannot be replaced while its process runs (Windows
  // locks running executables), so a running agent is stopped first.
  const wasRunning = installed.status === 'running'
  if (wasRunning) {
    onLine(`Stopping ${name}…`)
    const stopped = await runAgentAction('stop', name)
    if (!stopped.ok) {
      return { ok: false, message: `could not stop ${name}: ${stopped.message}` }
    }
  }

  onLine(`Updating ${name}…`)
  const result = await installAgent(name, onLine, true)
  if (!result.ok) {
    // Be explicit about the state we are leaving behind: the agent was
    // stopped for an update that then failed, and nothing restarted it.
    return wasRunning
      ? { ok: false, message: `${result.message} — ${name} was stopped and has not been restarted` }
      : result
  }

  if (wasRunning) {
    onLine(`Restarting ${name}…`)
    const started = await runAgentAction('start', name)
    if (!started.ok) {
      return { ok: false, message: `${name} updated but failed to restart: ${started.message}` }
    }
    return { ok: true, message: `${name} updated and restarted` }
  }
  return { ok: true, message: `${name} updated` }
}
