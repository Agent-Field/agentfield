import { describe, expect, it } from 'vitest'
import { deepLinkFromArgv, isView, parseDeepLink } from './deeplink'

describe('parseDeepLink', () => {
  it('maps each view URL to its view', () => {
    expect(parseDeepLink('agentfield://home')).toBe('home')
    expect(parseDeepLink('agentfield://install')).toBe('install')
    expect(parseDeepLink('agentfield://agents')).toBe('agents')
    expect(parseDeepLink('agentfield://activity')).toBe('activity')
    expect(parseDeepLink('agentfield://settings')).toBe('settings')
  })

  it('migrates legacy dashboard and secrets hosts', () => {
    expect(parseDeepLink('agentfield://dashboard')).toBe('home')
    expect(parseDeepLink('agentfield://secrets')).toBe('settings')
    expect(parseDeepLink('agentfield://secrets#secrets')).toBe('settings')
    expect(parseDeepLink('agentfield:dashboard')).toBe('home')
    expect(parseDeepLink('agentfield:secrets')).toBe('settings')
  })

  it('is case-insensitive and tolerates trailing slashes and subpaths', () => {
    expect(parseDeepLink('agentfield://Agents')).toBe('agents')
    expect(parseDeepLink('agentfield://agents/')).toBe('agents')
    expect(parseDeepLink('agentfield://agents/some-agent')).toBe('agents')
  })

  it('accepts the no-slash (opaque path) spelling', () => {
    expect(parseDeepLink('agentfield:agents')).toBe('agents')
    expect(parseDeepLink('agentfield:home')).toBe('home')
  })

  it('falls back to home for a bare or unknown target', () => {
    expect(parseDeepLink('agentfield://')).toBe('home')
    expect(parseDeepLink('agentfield://marketplace')).toBe('home')
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
    for (const v of ['home', 'install', 'agents', 'activity', 'settings']) {
      expect(isView(v)).toBe(true)
    }
    expect(isView('dashboard')).toBe(false)
    expect(isView('secrets')).toBe(false)
    expect(isView('marketplace')).toBe(false)
    expect(isView('')).toBe(false)
  })
})
