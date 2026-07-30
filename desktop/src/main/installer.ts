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

import { catalogEntry } from '../shared/catalog'
import type { InstallResult } from '../shared/types'
import { CpApiError, createCpClient, type CpClient } from './cpClient'

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
export interface InstallerDeps {
  cpClient: CpClient
}

/** Pure catalog lookup retained for callers/tests that need to preview an install. */
export function installCommand(
  name: string,
  force = false
): { command: 'control-plane'; args: string[] } | null {
  const entry = catalogEntry(name)
  if (!entry) return null
  return { command: 'control-plane', args: force ? ['install', entry.source, '--force'] : ['install', entry.source] }
}

function defaultInstallerDeps(): InstallerDeps {
  return { cpClient: createCpClient() }
}

const UPDATE_REQUIRED = 'Control plane update required — update AgentField CLI'

async function runInstall(
  source: string,
  onLine: (line: string) => void,
  successMessage: () => string,
  deps: InstallerDeps,
  force?: boolean
): Promise<InstallResult> {
  try {
    if (!(await deps.cpClient.hasInstallApi())) {
      return { ok: false, message: UPDATE_REQUIRED }
    }
    const { job_id } = await deps.cpClient.installPackage(source, force)
    const job = await deps.cpClient.watchInstallJob(job_id, onLine)
    return job.status === 'succeeded'
      ? { ok: true, message: successMessage() }
      : { ok: false, message: job.error || job.lines.at(-1) || 'Install failed' }
  } catch (err) {
    if (err instanceof CpApiError && err.status === 404) {
      return { ok: false, message: UPDATE_REQUIRED }
    }
    if (err instanceof CpApiError && err.status === 409) {
      return { ok: false, message: err.message || 'Another install is already running' }
    }
    return {
      ok: false,
      message:
        err instanceof CpApiError
          ? err.message
          : 'Could not reach the control plane — start the control plane and try again'
    }
  }
}

/**
 * Run `af install` for the named catalog entry, forwarding sanitized output
 * lines to onLine as they arrive. Resolves (never rejects) with the outcome.
 */
export function installAgent(
  name: string,
  onLine: (line: string) => void,
  force = false,
  deps: InstallerDeps = defaultInstallerDeps()
): Promise<InstallResult> {
  const entry = catalogEntry(name)
  if (!entry) {
    return Promise.resolve({ ok: false, message: `"${name}" is not in the install catalog` })
  }
  return runInstall(entry.source, onLine, () => `${name} installed`, deps, force)
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
  onLine: (line: string) => void,
  deps: InstallerDeps = defaultInstallerDeps()
): Promise<InstallResult> {
  const normalized = parseRepoSource(source)
  if (!normalized) {
    return Promise.resolve({
      ok: false,
      message: 'Enter a GitHub repository URL, e.g. https://github.com/org/repo (or …/repo//subdir)'
    })
  }
  return runInstall(normalized, onLine, () => `Installed from ${normalized}`, deps)
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
  onLine: (line: string) => void,
  deps: InstallerDeps = defaultInstallerDeps()
): Promise<InstallResult> {
  const entry = catalogEntry(name)
  if (!entry) {
    return { ok: false, message: `"${name}" is not in the install catalog` }
  }
  try {
    if (!(await deps.cpClient.hasInstallApi())) {
      return { ok: false, message: UPDATE_REQUIRED }
    }
    onLine(`Updating ${name}…`)
    const { job_id } = await deps.cpClient.updatePackage(name)
    const job = await deps.cpClient.watchInstallJob(job_id, onLine)
    return job.status === 'succeeded'
      ? { ok: true, message: `${name} updated` }
      : { ok: false, message: job.error || job.lines.at(-1) || `Failed to update ${name}` }
  } catch (err) {
    if (err instanceof CpApiError && err.status === 404) {
      return { ok: false, message: UPDATE_REQUIRED }
    }
    return {
      ok: false,
      message:
        err instanceof CpApiError
          ? err.message
          : 'Could not reach the control plane — start the control plane and try again'
    }
  }
}
