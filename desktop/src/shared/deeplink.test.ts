import { describe, expect, it } from 'vitest'
import { deepLinkFromArgv, isView, parseDeepLink } from './deeplink'

describe('parseDeepLink', () => {
  it('maps each view URL to its view', () => {
    expect(parseDeepLink('agentfield://dashboard')).toBe('dashboard')
    expect(parseDeepLink('agentfield://agents')).toBe('agents')
    expect(parseDeepLink('agentfield://activity')).toBe('activity')
    expect(parseDeepLink('agentfield://install')).toBe('install')
  })

  it('is case-insensitive and tolerates trailing slashes and subpaths', () => {
    expect(parseDeepLink('agentfield://Agents')).toBe('agents')
    expect(parseDeepLink('agentfield://agents/')).toBe('agents')
    expect(parseDeepLink('agentfield://agents/some-agent')).toBe('agents')
  })

  it('accepts the no-slash (opaque path) spelling', () => {
    expect(parseDeepLink('agentfield:agents')).toBe('agents')
  })

  it('falls back to dashboard for a bare or unknown target', () => {
    expect(parseDeepLink('agentfield://')).toBe('dashboard')
    expect(parseDeepLink('agentfield://marketplace')).toBe('dashboard')
  })

  it('returns null for other schemes and non-URLs', () => {
    expect(parseDeepLink('https://agentfield.ai')).toBeNull()
    expect(parseDeepLink('http://localhost:8080/ui/agents')).toBeNull()
    expect(parseDeepLink('C:\\Program Files\\AgentField\\AgentField.exe')).toBeNull()
    expect(parseDeepLink('--allow-file-access-from-files')).toBeNull()
    expect(parseDeepLink('')).toBeNull()
  })
})

describe('deepLinkFromArgv', () => {
  it('finds the deep link among ordinary process args', () => {
    const argv = ['C:\\AgentField\\AgentField.exe', '--allow-file-access', 'agentfield://activity']
    expect(deepLinkFromArgv(argv)).toBe('activity')
  })

  it('returns null when no arg is a deep link', () => {
    expect(deepLinkFromArgv(['electron.exe', '.', '--inspect=9229'])).toBeNull()
    expect(deepLinkFromArgv([])).toBeNull()
  })
})

describe('isView', () => {
  it('accepts exactly the app views', () => {
    for (const v of ['dashboard', 'agents', 'activity', 'install', 'secrets', 'settings']) {
      expect(isView(v)).toBe(true)
    }
    expect(isView('marketplace')).toBe(false)
    expect(isView('')).toBe(false)
  })
})
