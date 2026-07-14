// Install seam: runs `af install <source>` for vetted catalog entries.
// The af CLI is the single contract shared by agents, this app, and
// developers — the app never reimplements install logic, it shells out.
// Deliberately does NOT import from 'electron' so it stays unit-testable.

import { spawn } from 'node:child_process'
import { catalogEntry } from '../shared/catalog'
import type { InstallResult } from '../shared/types'
import { readInstalledAgents } from './agentfield'
import { runAgentAction } from './agents'
import { getCliCommand } from './cli'

// CSI sequences (colors, cursor movement, erase-line spinner frames) and OSC
// sequences (terminal titles), per ECMA-48. Written with \u escapes so no
// invisible control characters live in this source file.
const ANSI_PATTERN = new RegExp(
  '\\u001b\\[[0-9;?]*[A-Za-z]|\\u001b\\][^\\u0007\\u001b]*(?:\\u0007|\\u001b\\\\)?',
  'g'
)

/**
 * Normalize a chunk of `af install` output into displayable lines: strip
 * ANSI color/spinner escapes, split on newlines and carriage returns
 * (spinner frames), drop empties.
 */
export function sanitizeInstallOutput(chunk: string): string[] {
  return chunk
    .replace(ANSI_PATTERN, '')
    .split(/[\r\n]+/)
    .map((line) => line.trim())
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

    const child = spawn(cmd.command, cmd.args, { windowsHide: true })
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
          ? { ok: true, message: `${name} installed` }
          : { ok: false, message: lastLine || `af install exited with code ${code}` }
      )
    })
  })
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
