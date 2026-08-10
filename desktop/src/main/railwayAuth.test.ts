// @vitest-environment node
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { get } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  getFreshAccessToken,
  isLoggedIn,
  listWorkspaces,
  loginWithRailway,
  logout,
  railwayAuthTestUtils,
  type RailwayTokens,
  type TokenCodec,
} from './railwayAuth'

const temporaryDirectories: string[] = []

afterEach(() => {
  vi.restoreAllMocks()
  for (const directory of temporaryDirectories.splice(0)) {
    // Tests only create one known token file in each unique temporary directory.
    try { logout({ storePath: join(directory, 'tokens.json') }) } catch { /* noop */ }
  }
})

function tokenPath(): string {
  const directory = mkdtempSync(join(tmpdir(), 'agentfield-railway-auth-'))
  temporaryDirectories.push(directory)
  return join(directory, 'tokens.json')
}

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function request(url: string): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    get(url, (result) => {
      let body = ''
      result.setEncoding('utf8')
      result.on('data', (chunk: string) => { body += chunk })
      result.on('end', () => resolve({ status: result.statusCode ?? 0, body }))
    }).on('error', reject)
  })
}

const reverseCodec: TokenCodec = {
  encrypt: (plain) => [...plain].reverse().join(''),
  decrypt: (blob) => [...blob].reverse().join(''),
}

function writeTokens(path: string, tokens: RailwayTokens, codec: TokenCodec = reverseCodec): void {
  writeFileSync(path, JSON.stringify({ v: 1, blob: codec.encrypt(JSON.stringify(tokens)) }), { mode: 0o600 })
}

describe('PKCE', () => {
  it('creates a 128-character verifier from the RFC 7636 alphabet', () => {
    const verifier = railwayAuthTestUtils.createVerifier()
    expect(verifier).toHaveLength(128)
    expect(verifier).toMatch(/^[A-Za-z0-9._~-]+$/)
  })

  it('computes the RFC 7636 S256 challenge', () => {
    const verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
    expect(railwayAuthTestUtils.createChallenge(verifier)).toBe('E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM')
  })
})

describe('loginWithRailway', () => {
  it('builds the complete auth URL, ignores other paths, accepts a callback, and exchanges the code', async () => {
    const storePath = tokenPath()
    let exchangeBody = ''
    const fetchImpl = vi.fn<typeof fetch>(async (_input, init) => {
      exchangeBody = String(init?.body)
      return response({ access_token: 'access', refresh_token: 'refresh', expires_in: 3600 })
    })
    const result = await loginWithRailway({
      storePath,
      codec: reverseCodec,
      now: () => 1_000,
      fetchImpl,
      openUrl: async (rawUrl) => {
        const url = new URL(rawUrl)
        expect(url.origin + url.pathname).toBe('https://backboard.railway.com/oauth/auth')
        expect(Object.fromEntries(url.searchParams)).toMatchObject({
          response_type: 'code',
          client_id: 'rlwy_oaci_onEklvmksh1hRUiCo7E2zX12',
          scope: 'openid email profile offline_access workspace:admin project:admin',
          code_challenge_method: 'S256',
          prompt: 'consent',
        })
        expect(url.searchParams.get('code_challenge')).toMatch(/^[A-Za-z0-9_-]{43}$/)
        const redirect = new URL(url.searchParams.get('redirect_uri')!)
        const ignored = await request(`${redirect.origin}/unrelated`)
        expect(ignored.status).toBe(404)
        const callback = await request(`${redirect.toString()}?code=auth-code&state=${encodeURIComponent(url.searchParams.get('state')!)}`)
        expect(callback.status).toBe(200)
        expect(callback.body).toContain("You're signed in")
      },
    })

    expect(result).toEqual({ ok: true, message: 'Signed in to Railway.' })
    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(Object.fromEntries(new URLSearchParams(exchangeBody))).toMatchObject({
      grant_type: 'authorization_code',
      code: 'auth-code',
      client_id: 'rlwy_oaci_onEklvmksh1hRUiCo7E2zX12',
    })
    expect(isLoggedIn({ storePath, codec: reverseCodec })).toBe(true)
    if (process.platform !== 'win32') {
      // Windows ignores POSIX modes (ACLs apply instead); assert only where chmod is real.
      expect(statSync(storePath).mode & 0o777).toBe(0o600)
    }
  })

  it('rejects a bad state with a CSRF failure page', async () => {
    const result = await loginWithRailway({
      storePath: tokenPath(),
      openUrl: async (rawUrl) => {
        const redirect = new URL(new URL(rawUrl).searchParams.get('redirect_uri')!)
        const callback = await request(`${redirect.toString()}?code=nope&state=wrong`)
        expect(callback.status).toBe(400)
        expect(callback.body).toContain('CSRF validation failed')
      },
    })
    expect(result.ok).toBe(false)
    expect(result.message).toMatch(/CSRF.*state mismatch/i)
  })

  it('surfaces an OAuth error parameter and its failure page', async () => {
    const result = await loginWithRailway({
      storePath: tokenPath(),
      openUrl: async (rawUrl) => {
        const auth = new URL(rawUrl)
        const redirect = new URL(auth.searchParams.get('redirect_uri')!)
        const callback = await request(`${redirect.toString()}?error=access_denied&error_description=Nope&state=${encodeURIComponent(auth.searchParams.get('state')!)}`)
        expect(callback.status).toBe(400)
        expect(callback.body).toContain('Sign-in failed')
      },
    })
    expect(result).toEqual({ ok: false, message: 'Railway authorization failed: Nope' })
  })

  it('surfaces token exchange HTTP errors', async () => {
    const result = await loginWithRailway({
      storePath: tokenPath(),
      fetchImpl: async () => response({ error: 'invalid_request', error_description: 'Bad exchange' }, 400),
      openUrl: async (rawUrl) => {
        const auth = new URL(rawUrl)
        const redirect = auth.searchParams.get('redirect_uri')!
        await request(`${redirect}?code=code&state=${encodeURIComponent(auth.searchParams.get('state')!)}`)
      },
    })
    expect(result).toEqual({ ok: false, message: 'Railway token request failed: Bad exchange' })
  })
})

describe('stored sessions and refresh', () => {
  it('round-trips through a codec and logout deletes the store', () => {
    const storePath = tokenPath()
    writeTokens(storePath, { accessToken: 'a', refreshToken: 'r', expiresAt: 10 }, reverseCodec)
    expect(readFileSync(storePath, 'utf8')).not.toContain('accessToken')
    expect(isLoggedIn({ storePath, codec: reverseCodec })).toBe(true)
    logout({ storePath })
    expect(isLoggedIn({ storePath, codec: reverseCodec })).toBe(false)
  })

  it('returns a fresh token without making a request', async () => {
    const storePath = tokenPath()
    writeTokens(storePath, { accessToken: 'still-fresh', refreshToken: 'r', expiresAt: 100_000 })
    const fetchImpl = vi.fn<typeof fetch>()
    expect(await getFreshAccessToken({ storePath, codec: reverseCodec, now: () => 1, fetchImpl })).toBe('still-fresh')
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('refreshes near expiry and persists a rotated refresh token', async () => {
    const storePath = tokenPath()
    writeTokens(storePath, { accessToken: 'old', refreshToken: 'old-refresh', expiresAt: 50_000 })
    let requestBody = ''
    const fetchImpl = vi.fn<typeof fetch>(async (_input, init) => {
      requestBody = String(init?.body)
      return response({ access_token: 'new', refresh_token: 'rotated', expires_in: 120 })
    })
    expect(await getFreshAccessToken({ storePath, codec: reverseCodec, now: () => 1_000, fetchImpl })).toBe('new')
    expect(Object.fromEntries(new URLSearchParams(requestBody))).toMatchObject({
      grant_type: 'refresh_token', refresh_token: 'old-refresh',
    })
    const encrypted = JSON.parse(readFileSync(storePath, 'utf8')) as { blob: string }
    expect(JSON.parse(reverseCodec.decrypt(encrypted.blob))).toEqual({
      accessToken: 'new', refreshToken: 'rotated', expiresAt: 121_000,
    })
  })

  it('returns null when logged out', async () => {
    expect(await getFreshAccessToken({ storePath: tokenPath() })).toBeNull()
  })

  it('clears the store when refresh returns invalid_grant', async () => {
    const storePath = tokenPath()
    writeTokens(storePath, { accessToken: 'old', refreshToken: 'bad', expiresAt: 1 })
    expect(await getFreshAccessToken({
      storePath,
      codec: reverseCodec,
      now: () => 10,
      fetchImpl: async () => response({ error: 'invalid_grant' }, 400),
    })).toBeNull()
    expect(isLoggedIn({ storePath, codec: reverseCodec })).toBe(false)
  })
})

describe('listWorkspaces', () => {
  it('uses the Railway GraphQL endpoint and returns workspaces', async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => response({ data: { me: { workspaces: [{ id: 'w', name: 'Team' }] } } }))
    await expect(listWorkspaces('token', { fetchImpl })).resolves.toEqual([{ id: 'w', name: 'Team' }])
    expect(fetchImpl.mock.calls[0]?.[0]).toBe('https://backboard.railway.com/graphql/v2')
    expect(fetchImpl.mock.calls[0]?.[1]?.headers).toMatchObject({ Authorization: 'Bearer token' })
  })
})
