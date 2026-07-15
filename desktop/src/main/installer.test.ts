import { describe, expect, it } from 'vitest'
import { CATALOG } from '../shared/catalog'
import { installCommand, installFromSource, parseRepoSource, sanitizeInstallOutput } from './installer'

describe('installCommand', () => {
  it('builds a plain install for a catalog entry', () => {
    const cmd = installCommand(CATALOG[0].name)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', CATALOG[0].source])
  })

  it('appends --force for updates (reinstall in place, secrets survive)', () => {
    const cmd = installCommand(CATALOG[0].name, true)
    expect(cmd).not.toBeNull()
    expect(cmd!.args).toEqual(['install', CATALOG[0].source, '--force'])
  })

  it('refuses names outside the curated catalog', () => {
    expect(installCommand('rm -rf /', true)).toBeNull()
    expect(installCommand('not-in-catalog')).toBeNull()
  })
})

describe('parseRepoSource', () => {
  const accepted: Array<[string, string]> = [
    // [input, normalized output]
    ['https://github.com/Agent-Field/pr-af', 'https://github.com/Agent-Field/pr-af'],
    ['https://github.com/Agent-Field/pr-af.git', 'https://github.com/Agent-Field/pr-af.git'],
    ['https://github.com/Agent-Field/pr-af/', 'https://github.com/Agent-Field/pr-af'],
    ['  https://github.com/Agent-Field/pr-af  ', 'https://github.com/Agent-Field/pr-af'],
    ['https://github.com/Agent-Field/pr-af//go', 'https://github.com/Agent-Field/pr-af//go'],
    ['https://github.com/Agent-Field/SWE-AF//go/cmd', 'https://github.com/Agent-Field/SWE-AF//go/cmd'],
    ['https://github.com/Agent-Field/pr-af//go/', 'https://github.com/Agent-Field/pr-af//go'],
    ['https://github.com/user_1/repo.name-2', 'https://github.com/user_1/repo.name-2']
  ]
  it.each(accepted)('accepts and normalizes %s', (input, expected) => {
    expect(parseRepoSource(input)).toBe(expected)
  })

  const rejected: Array<[string, string]> = [
    ['http (not https)', 'http://github.com/Agent-Field/pr-af'],
    ['other host', 'https://gitlab.com/Agent-Field/pr-af'],
    ['ssh scp form', 'git@github.com:Agent-Field/pr-af.git'],
    ['ssh url', 'ssh://git@github.com/Agent-Field/pr-af'],
    ['leading-dash flag payload', '--force'],
    ['owner starting with dash', 'https://github.com/-evil/repo'],
    ['dotdot traversal in subdir', 'https://github.com/Agent-Field/pr-af//..%2Fetc'],
    ['literal dotdot in subdir', 'https://github.com/Agent-Field/pr-af//../secret'],
    ['embedded whitespace', 'https://github.com/Agent-Field/pr af'],
    ['empty', ''],
    ['whitespace only', '   '],
    ['query string', 'https://github.com/Agent-Field/pr-af?tab=readme'],
    ['fragment', 'https://github.com/Agent-Field/pr-af#install'],
    ['missing repo', 'https://github.com/Agent-Field'],
    ['empty subdir', 'https://github.com/Agent-Field/pr-af//'],
    ['subdir leading slash', 'https://github.com/Agent-Field/pr-af///go']
  ]
  it.each(rejected)('rejects %s', (_label, input) => {
    expect(parseRepoSource(input)).toBeNull()
  })
})

describe('installFromSource', () => {
  it('refuses a rejected source without spawning', async () => {
    const lines: string[] = []
    const result = await installFromSource('git@github.com:evil/repo.git', (line) =>
      lines.push(line)
    )
    expect(result.ok).toBe(false)
    // Never spawned af install: no progress lines were forwarded.
    expect(lines).toEqual([])
    expect(result.message).toMatch(/github\.com/)
  })
})

describe('sanitizeInstallOutput', () => {
  it('unwraps zerolog JSON error lines to the underlying error text', () => {
    const line =
      '{"level":"error","error":"invalid package structure: no agentfield-package.yaml found for --path \\"go\\"","time":"2026-07-15T12:30:22-04:00","message":"Error executing root command"}'
    expect(sanitizeInstallOutput(line)).toEqual([
      'invalid package structure: no agentfield-package.yaml found for --path "go"'
    ])
  })

  it('falls back to the zerolog message when there is no error field', () => {
    expect(sanitizeInstallOutput('{"level":"info","message":"cloning repository"}')).toEqual([
      'cloning repository'
    ])
  })

  it('passes non-zerolog JSON and plain lines through untouched', () => {
    expect(sanitizeInstallOutput('{"result":"ok"}')).toEqual(['{"result":"ok"}'])
    expect(sanitizeInstallOutput('✅ Package installed successfully')).toEqual([
      '✅ Package installed successfully'
    ])
    expect(sanitizeInstallOutput('{not json}')).toEqual(['{not json}'])
  })
})
