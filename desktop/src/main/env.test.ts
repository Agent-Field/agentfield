import { homedir } from 'node:os'
import { posix } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  childEnv,
  extractShellPath,
  initUserPath,
  mergePaths,
  resetUserPathCache,
  resolveUserPath,
  syncResolvedPath,
  wellKnownBinDirs
} from './env'

// The module works in POSIX PATHs; force the separator so the assertions read
// the same regardless of the host platform running the suite.
const SEP = ':'

describe('extractShellPath', () => {
  it('pulls the PATH from between the sentinel markers', () => {
    const out = 'motd banner\n__AF_PATH_START__/opt/homebrew/bin:/usr/bin__AF_PATH_END__\n'
    expect(extractShellPath(out)).toBe('/opt/homebrew/bin:/usr/bin')
  })

  it('returns null when the markers are missing or empty', () => {
    expect(extractShellPath('no markers here')).toBeNull()
    expect(extractShellPath('__AF_PATH_START____AF_PATH_END__')).toBeNull()
    // End before start — refuse rather than slice a negative range.
    expect(extractShellPath('__AF_PATH_END__x__AF_PATH_START__')).toBeNull()
  })
})

describe('mergePaths', () => {
  it('de-dupes, first occurrence wins, order preserved', () => {
    expect(mergePaths(['/a:/b', '/b:/c', '/a'], SEP)).toBe('/a:/b:/c')
  })

  it('drops empty entries and blank inputs', () => {
    expect(mergePaths(['/a::/b', null, undefined, '', '/c'], SEP)).toBe('/a:/b:/c')
  })
})

describe('wellKnownBinDirs', () => {
  it('includes the AgentField, homebrew and package-manager dirs', () => {
    const dirs = wellKnownBinDirs('/home/me')
    expect(dirs).toContain('/opt/homebrew/bin')
    expect(dirs).toContain('/usr/local/bin')
    expect(dirs).toContain('/home/me/.agentfield/bin')
    expect(dirs).toContain('/home/me/.cargo/bin')
    expect(dirs).toContain('/home/me/.local/bin')
  })
})

describe('syncResolvedPath', () => {
  it('merges process PATH with the well-known dirs on darwin', () => {
    const path = syncResolvedPath({ platform: 'darwin', env: { PATH: '/usr/bin' }, home: '/home/me' })
    expect(path.split(':')).toContain('/usr/bin')
    expect(path.split(':')).toContain('/opt/homebrew/bin')
    expect(path.split(':')).toContain('/home/me/.agentfield/bin')
  })

  it('returns the process PATH unchanged on win32', () => {
    expect(
      syncResolvedPath({ platform: 'win32', env: { PATH: 'C:\\Windows;C:\\bin' } })
    ).toBe('C:\\Windows;C:\\bin')
  })
})

describe('resolveUserPath', () => {
  it('merges the login-shell PATH ahead of process PATH and well-known dirs', async () => {
    const path = await resolveUserPath({
      platform: 'darwin',
      env: { PATH: '/usr/bin' },
      home: '/home/me',
      runLoginShell: async () => '__AF_PATH_START__/shell/bin:/usr/bin__AF_PATH_END__'
    })
    const parts = path.split(':')
    // Shell PATH comes first (highest priority), then the process PATH extras,
    // then the well-known dirs — all de-duped.
    expect(parts[0]).toBe('/shell/bin')
    expect(parts).toContain('/usr/bin')
    expect(parts).toContain('/opt/homebrew/bin')
    expect(parts.filter((p) => p === '/usr/bin')).toHaveLength(1)
  })

  it('falls back to the sync inputs when the shell probe fails', async () => {
    const path = await resolveUserPath({
      platform: 'darwin',
      env: { PATH: '/usr/bin' },
      home: '/home/me',
      runLoginShell: async () => null
    })
    expect(path.split(':')).toContain('/usr/bin')
    expect(path.split(':')).toContain('/opt/homebrew/bin')
    expect(path).not.toContain('undefined')
  })

  it('never throws when the shell runner rejects', async () => {
    const path = await resolveUserPath({
      platform: 'darwin',
      env: { PATH: '/usr/bin' },
      runLoginShell: async () => {
        throw new Error('shell blew up')
      }
    })
    expect(path.split(':')).toContain('/usr/bin')
  })

  it('leaves the PATH untouched on win32 (never spawns a shell)', async () => {
    let ran = false
    const path = await resolveUserPath({
      platform: 'win32',
      env: { PATH: 'C:\\bin' },
      runLoginShell: async () => {
        ran = true
        return null
      }
    })
    expect(path).toBe('C:\\bin')
    expect(ran).toBe(false)
  })
})

describe('childEnv / initUserPath cache', () => {
  it('uses the sync fallback before initUserPath resolves', () => {
    resetUserPathCache()
    const env = childEnv()
    // The real process PATH is present, augmented with the well-known dirs on
    // non-Windows hosts.
    expect(env.PATH).toBeDefined()
    if (process.platform !== 'win32') {
      expect(env.PATH?.split(':')).toContain(posix.join(homedir(), '.agentfield', 'bin'))
    }
  })

  it('serves the resolved PATH once initUserPath has run, and merges extras', async () => {
    await initUserPath({
      platform: 'darwin',
      env: { PATH: '/usr/bin' },
      home: '/home/me',
      runLoginShell: async () => '__AF_PATH_START__/shell/bin__AF_PATH_END__'
    })
    const env = childEnv({ AGENTFIELD_PORT: '8080' })
    expect(env.PATH?.split(':')[0]).toBe('/shell/bin')
    expect(env.AGENTFIELD_PORT).toBe('8080')
    resetUserPathCache()
  })

  it('never lets an extra override the resolved PATH', async () => {
    await initUserPath({
      platform: 'darwin',
      env: { PATH: '/usr/bin' },
      home: '/home/me',
      runLoginShell: async () => '__AF_PATH_START__/shell/bin__AF_PATH_END__'
    })
    const env = childEnv({ PATH: '/attacker/bin' })
    expect(env.PATH?.split(':')[0]).toBe('/shell/bin')
    resetUserPathCache()
  })
})
