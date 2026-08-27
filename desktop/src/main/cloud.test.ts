import { afterEach, describe, expect, it, vi } from 'vitest'
import { getApiKey, getBaseUrl, setLocalApiKey, setLocalPort } from './connection'
import { applyConnectionProfile, normalizeServerUrl, testCloudConnection } from './cloud'
import { DEFAULT_SETTINGS } from './settings'

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' }
  })
}

function fakeFetch(responses: Array<Response | Error>): typeof fetch {
  return vi.fn(async () => {
    const next = responses.shift()
    if (next instanceof Error) throw next
    if (!next) throw new Error('unexpected fetch')
    return next
  }) as unknown as typeof fetch
}

afterEach(() => {
  setLocalApiKey(null)
  setLocalPort(8080)
})

describe('normalizeServerUrl', () => {
  it('normalizes host input and removes paths, queries, and trailing slashes', () => {
    expect(normalizeServerUrl(' cp.example/path/?x=1 ')).toBe('https://cp.example')
    expect(normalizeServerUrl('https://cp.example:8443///')).toBe('https://cp.example:8443')
  })

  it('allows insecure HTTP only on local and private hosts', () => {
    expect(normalizeServerUrl('http://localhost:8080/ui')).toBe('http://localhost:8080')
    expect(normalizeServerUrl('http://127.0.0.1:8080')).toBe('http://127.0.0.1:8080')
    expect(normalizeServerUrl('http://10.2.3.4')).toBe('http://10.2.3.4')
    expect(normalizeServerUrl('http://172.16.1.2')).toBe('http://172.16.1.2')
    expect(normalizeServerUrl('http://192.168.1.2')).toBe('http://192.168.1.2')
    expect(normalizeServerUrl('http://cp.example')).toBeNull()
  })

  it('rejects empty and malformed input', () => {
    expect(normalizeServerUrl('')).toBeNull()
    expect(normalizeServerUrl(':// bad')).toBeNull()
    expect(normalizeServerUrl('ftp://cp.example')).toBeNull()
  })
})

describe('testCloudConnection', () => {
  it('reports healthy authenticated servers, install support, and health version', async () => {
    const fetchImpl = fakeFetch([
      json({ status: 'healthy', version: '1.2.3', furrow_public_addr: 'furrow.example:8802' }),
      json({ packages: [] }),
      json([])
    ])
    await expect(
      testCloudConnection('cp.example/', 'secret', { fetchImpl, furrowProbe: async () => true })
    ).resolves.toEqual({
      ok: true,
      healthy: true,
      authOk: true,
      installApi: true,
      furrowAvailable: true,
      furrowReported: true,
      version: '1.2.3',
      message: 'Connection successful'
    })
    for (const [, init] of vi.mocked(fetchImpl).mock.calls) {
      expect(new Headers(init?.headers).get('X-API-Key')).toBe('secret')
      expect(init?.signal).toBeInstanceOf(AbortSignal)
    }
  })

  it('distinguishes a rejected API key', async () => {
    const fetchImpl = fakeFetch([json({ status: 'healthy' }), json({}, 401)])
    await expect(testCloudConnection('https://cp.example', 'bad', { fetchImpl })).resolves.toEqual({
      ok: false,
      healthy: true,
      authOk: false,
      installApi: false,
      furrowAvailable: false,
      message: 'API key rejected'
    })
  })

  it('distinguishes an unreachable server', async () => {
    const fetchImpl = fakeFetch([new Error('offline')])
    await expect(testCloudConnection('https://cp.example/', 'key', { fetchImpl })).resolves.toEqual({
      ok: false,
      healthy: false,
      authOk: false,
      installApi: false,
      furrowAvailable: false,
      message: 'Could not reach https://cp.example'
    })
  })

  it('reports an older control plane without install API and discovers version separately', async () => {
    const fetchImpl = fakeFetch([
      json({ status: 'healthy' }),
      json({ packages: [] }),
      json({}, 404),
      json({ server_version: '0.9.0' })
    ])
    await expect(
      testCloudConnection('https://cp.example', 'key', { fetchImpl })
    ).resolves.toEqual({
      ok: true,
      healthy: true,
      authOk: true,
      installApi: false,
      furrowAvailable: false,
      version: '0.9.0',
      message: 'Connected; install API unavailable'
    })
  })

  it('omits an unknown version', async () => {
    const fetchImpl = fakeFetch([
      json({ status: 'healthy' }),
      json({ packages: [] }),
      json([]),
      json({}, 404)
    ])
    const result = await testCloudConnection('https://cp.example', 'key', { fetchImpl })
    expect(result.ok).toBe(true)
    expect(result).not.toHaveProperty('version')
  })

  it('keeps a healthy connection when the advertised furrow port is unreachable', async () => {
    const fetchImpl = fakeFetch([
      json({ status: 'healthy', furrow_public_addr: 'furrow.example:8802' }),
      json({ packages: [] }),
      json([]),
      json({}, 404)
    ])
    const result = await testCloudConnection('https://cp.example', 'key', {
      fetchImpl,
      furrowProbe: async () => false
    })
    expect(result).toMatchObject({ ok: true, healthy: true, authOk: true, furrowAvailable: false })
    expect(result.furrowReported).toBe(true)
  })
})

describe('applyConnectionProfile', () => {
  it('applies an enabled normalized profile and clears it when disabled', () => {
    setLocalPort(9091)
    applyConnectionProfile({
      ...DEFAULT_SETTINGS,
      cloud: {
        ...DEFAULT_SETTINGS.cloud,
        enabled: true,
        serverUrl: ' cp.example/path ',
        apiKey: 'secret'
      }
    })
    expect(getBaseUrl()).toBe('https://cp.example')
    expect(getApiKey()).toBe('secret')
    applyConnectionProfile(DEFAULT_SETTINGS)
    expect(getBaseUrl()).toBe('http://localhost:9091')
    expect(getApiKey()).toBeNull()
  })

  it('seeds the local port from the configured or last-used port', () => {
    applyConnectionProfile({ ...DEFAULT_SETTINGS, controlPlanePort: 18480 })
    expect(getBaseUrl()).toBe('http://localhost:18480')
    applyConnectionProfile({ ...DEFAULT_SETTINGS, lastControlPlanePort: 8090 })
    expect(getBaseUrl()).toBe('http://localhost:8090')
    // A configured port beats the remembered one.
    applyConnectionProfile({
      ...DEFAULT_SETTINGS,
      controlPlanePort: 18480,
      lastControlPlanePort: 8090
    })
    expect(getBaseUrl()).toBe('http://localhost:18480')
    // Cloud still wins while enabled, and the seeded port is what a switch
    // back to local returns to.
    applyConnectionProfile({
      ...DEFAULT_SETTINGS,
      controlPlanePort: 18480,
      cloud: {
        ...DEFAULT_SETTINGS.cloud,
        enabled: true,
        serverUrl: 'https://cp.example',
        apiKey: 'k'
      }
    })
    expect(getBaseUrl()).toBe('https://cp.example')
    applyConnectionProfile({ ...DEFAULT_SETTINGS, controlPlanePort: 18480 })
    expect(getBaseUrl()).toBe('http://localhost:18480')
  })

  it('carries a configured local API key on the local profile', () => {
    applyConnectionProfile({ ...DEFAULT_SETTINGS, localApiKey: 'local-secret' })
    expect(getBaseUrl()).toBe('http://localhost:8080')
    expect(getApiKey()).toBe('local-secret')
  })

  it('keeps the local key available for the switch back from cloud', () => {
    applyConnectionProfile({
      ...DEFAULT_SETTINGS,
      localApiKey: 'local-secret',
      cloud: {
        ...DEFAULT_SETTINGS.cloud,
        enabled: true,
        serverUrl: 'https://cp.example',
        apiKey: 'cloud-key'
      }
    })
    expect(getApiKey()).toBe('cloud-key')
    applyConnectionProfile({ ...DEFAULT_SETTINGS, localApiKey: 'local-secret' })
    expect(getApiKey()).toBe('local-secret')
  })
})
