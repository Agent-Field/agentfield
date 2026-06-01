import { createHash, generateKeyPairSync, sign } from 'node:crypto';
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

const { getMock } = vi.hoisted(() => ({
  getMock: vi.fn()
}));

vi.mock('axios', () => ({
  default: { get: getMock },
  get: getMock
}));

import { LocalVerifier } from '../src/verification/LocalVerifier.js';

function makeDidKey(publicKey: Buffer): string {
  return `did:key:z${Buffer.concat([Buffer.from([0xed, 0x01]), publicKey]).toString('base64url')}`;
}

function signPayload(
  privateKey: Parameters<typeof sign>[2],
  timestamp: string,
  body: Buffer,
  nonce?: string
): string {
  const bodyHash = createHash('sha256').update(body).digest('hex');
  const payload = nonce ? `${timestamp}:${nonce}:${bodyHash}` : `${timestamp}:${bodyHash}`;
  return sign(null, Buffer.from(payload, 'utf-8'), privateKey).toString('base64');
}

describe('LocalVerifier additional coverage', () => {
  beforeEach(() => {
    getMock.mockReset();
  });

  it('refresh() populates caches, trims the base URL, and forwards the API key header', async () => {
    const { publicKey } = generateKeyPairSync('ed25519');
    const publicJwk = publicKey.export({ format: 'jwk' }) as { x: string };

    getMock
      .mockResolvedValueOnce({ status: 200, data: { policies: [{ name: 'allow-all', caller_tags: [], target_tags: [], allow_functions: ['*'], deny_functions: [], constraints: {}, action: 'allow', priority: 1 }] } })
      .mockResolvedValueOnce({ status: 200, data: { revoked_dids: ['did:key:revoked'] } })
      .mockResolvedValueOnce({ status: 200, data: { registered_dids: ['did:key:known'] } })
      .mockResolvedValueOnce({ status: 200, data: { public_key_jwk: { x: publicJwk.x }, issuer_did: 'did:web:issuer' } });

    const verifier = new LocalVerifier('http://control-plane.local///', 300, 300, 'api-key-1');

    await expect(verifier.refresh()).resolves.toBe(true);

    expect((getMock as Mock).mock.calls).toEqual([
      ['http://control-plane.local/api/v1/policies', { headers: { 'X-API-Key': 'api-key-1' }, timeout: 10_000 }],
      ['http://control-plane.local/api/v1/revocations', { headers: { 'X-API-Key': 'api-key-1' }, timeout: 10_000 }],
      ['http://control-plane.local/api/v1/registered-dids', { headers: { 'X-API-Key': 'api-key-1' }, timeout: 10_000 }],
      ['http://control-plane.local/api/v1/admin/public-key', { headers: { 'X-API-Key': 'api-key-1' }, timeout: 10_000 }]
    ]);
    expect(verifier.checkRevocation('did:key:revoked')).toBe(true);
    expect(verifier.checkRegistration('did:key:known')).toBe(true);
    expect(verifier.checkRegistration('did:key:unknown')).toBe(false);
    expect(verifier.needsRefresh).toBe(false);
  });

  it('refresh() reports failure when any fetch errors or returns a non-200 status', async () => {
    getMock
      .mockResolvedValueOnce({ status: 500, data: {} })
      .mockRejectedValueOnce(new Error('network'))
      .mockResolvedValueOnce({ status: 200, data: { registered_dids: [] } })
      .mockResolvedValueOnce({ status: 200, data: { public_key_jwk: {} } });

    const verifier = new LocalVerifier('http://control-plane.local');

    await expect(verifier.refresh()).resolves.toBe(false);
    expect(verifier.needsRefresh).toBe(true);
    expect(verifier.checkRegistration('did:key:any')).toBe(true);
  });

  it('verifySignature() accepts valid did:key signatures and rejects stale or malformed inputs', async () => {
    const { publicKey, privateKey } = generateKeyPairSync('ed25519');
    const publicJwk = publicKey.export({ format: 'jwk' }) as { x: string };
    const callerDid = makeDidKey(Buffer.from(publicJwk.x, 'base64url'));
    const body = Buffer.from(JSON.stringify({ prompt: 'hi' }));
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const nonce = 'nonce-1';
    const signature = signPayload(privateKey, timestamp, body, nonce);
    const verifier = new LocalVerifier('http://control-plane.local');

    await expect(verifier.verifySignature(callerDid, signature, timestamp, body, nonce)).resolves.toBe(true);
    await expect(verifier.verifySignature(callerDid, signature, 'not-a-number', body, nonce)).resolves.toBe(false);
    await expect(
      verifier.verifySignature(callerDid, signature, String(Math.floor(Date.now() / 1000) - 1000), body, nonce)
    ).resolves.toBe(false);
    await expect(verifier.verifySignature('did:key:zbad', signature, timestamp, body, nonce)).resolves.toBe(false);
  });

  it('verifySignature() falls back to the cached admin public key for non did:key callers', async () => {
    const { publicKey, privateKey } = generateKeyPairSync('ed25519');
    const publicJwk = publicKey.export({ format: 'jwk' }) as { x: string };
    const verifier = new LocalVerifier('http://control-plane.local');
    (verifier as any).adminPublicKeyBytes = new Uint8Array(Buffer.from(publicJwk.x, 'base64url'));
    const body = Buffer.from('payload');
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const signature = signPayload(privateKey, timestamp, body);

    await expect(
      verifier.verifySignature('did:web:caller.example', signature, timestamp, body)
    ).resolves.toBe(true);
  });

  it('evaluatePolicy() enforces priority, deny rules, constraints, and default fallback behavior', () => {
    const verifier = new LocalVerifier('http://control-plane.local');
    const state = verifier as any;

    expect(verifier.evaluatePolicy([], [], 'any')).toBe(false);

    state.policies = [
      {
        name: 'deny-expensive',
        caller_tags: ['caller'],
        target_tags: ['target'],
        allow_functions: ['*'],
        deny_functions: ['payments.*'],
        constraints: {},
        action: 'allow',
        priority: 10
      },
      {
        name: 'allow-limited',
        caller_tags: ['caller'],
        target_tags: ['target'],
        allow_functions: ['reports.generate', '*summary'],
        deny_functions: [],
        constraints: {
          amount: { operator: '<=', value: 100 }
        },
        action: 'allow',
        priority: 5
      },
      {
        name: 'disabled',
        caller_tags: ['caller'],
        target_tags: ['target'],
        allow_functions: ['reports.generate'],
        deny_functions: [],
        constraints: {},
        action: 'deny',
        priority: 99,
        enabled: false
      }
    ];

    expect(verifier.evaluatePolicy(['caller'], ['target'], 'payments.charge')).toBe(false);
    expect(verifier.evaluatePolicy(['caller'], ['target'], 'reports.generate', { amount: 50 })).toBe(true);
    expect(verifier.evaluatePolicy(['caller'], ['target'], 'daily.summary', { amount: 50 })).toBe(true);
    expect(verifier.evaluatePolicy(['caller'], ['target'], 'reports.generate', { amount: 150 })).toBe(true);
    expect(verifier.evaluatePolicy(['other'], ['target'], 'reports.generate')).toBe(true);

    state.policies = [state.policies[1]];
    expect(verifier.evaluatePolicy(['caller'], ['target'], 'reports.generate', { amount: 150 })).toBe(false);
  });
});
