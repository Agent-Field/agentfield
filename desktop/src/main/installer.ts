// Install seam: runs `af install <source>` for vetted catalog entries.
// The af CLI is the single contract shared by agents, this app, and
// developers — the app never reimplements install logic, it shells out.
// Deliberately does NOT import from 'electron' so it stays unit-testable.

import { spawn } from 'node:child_process'
import { catalogEntry } from '../shared/catalog'
import type { InstallResult } from '../shared/types'
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
 * unknown is refused rather than passed to a shell.
 */
export function installCommand(name: string): { command: string; args: string[] } | null {
  const entry = catalogEntry(name)
  if (!entry) return null
  // spawn() without a shell; the command is whatever CLI resolution picked
  // (managed copy, PATH `af`, or the app's bundled binary — see main/cli.ts).
  return { command: getCliCommand(), args: ['install', entry.source] }
}

/**
 * Run `af install` for the named catalog entry, forwarding sanitized output
 * lines to onLine as they arrive. Resolves (never rejects) with the outcome.
 */
export function installAgent(
  name: string,
  onLine: (line: string) => void
): Promise<InstallResult> {
  const cmd = installCommand(name)
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
