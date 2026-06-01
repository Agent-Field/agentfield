/**
 * Behavioral invariant tests for DIDAuthenticator.
 *
 * These tests verify structural properties that must always hold regardless of
 * implementation details or refactoring.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import crypto from 'node:crypto';
import {
  DIDAuthenticator,
  HEADER_CALLER_DID,
  HEADER_DID_SIGNATURE,
  HEADER_DID_TIMESTAMP,
  HEADER_DID_NONCE
} from '../src/client/DIDAuthenticator.js';

// ── Test helpers ──────────────────────────────────────────────────────────────

const PKCS8_PREFIX = Buffer.from([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06,
  0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20
]);

function makeKeypair(seed: Buffer = crypto.randomBytes(32)) {
  const privateKey = crypto.createPrivateKey({
    key: Buffer.concat([PKCS8_PREFIX, seed]),
    format: 'der',
    type: 'pkcs8'
  });
  const publicKey = crypto.createPublicKey(privateKey);
  const jwk = JSON.stringify({
    kty: 'OKP',
    crv: 'Ed25519',
    d: seed.toString('base64url')
  });
  return { jwk, privateKey, publicKey };
}

const FIXED_SEED = Buffer.alloc(32, 0x42);
const TEST_DID = 'did:web:test.example.com:agents:invariant-tester';

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Invariants ────────────────────────────────────────────────────────────────

describe('DIDAuthenticator invariants', () => {
  // ── Nonce Uniqueness ────────────────────────────────────────────────────────

  describe('INVARIANT: nonce uniqueness across rapid sign calls', () => {
    it('50 rapid sign calls produce 50 distinct nonces', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      const nonces = new Set<string>();
      for (let i = 0; i < 50; i++) {
        const headers = auth.signRequest(Buffer.from(`{"i":${i}}`));
        nonces.add(headers[HEADER_DID_NONCE]);
      }

      expect(nonces.size).toBe(50);
    });

    it('nonce has expected format: 32 lowercase hex characters', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      for (let i = 0; i < 10; i++) {
        const headers = auth.signRequest(Buffer.from('{}'));
        expect(headers[HEADER_DID_NONCE]).toMatch(/^[0-9a-f]{32}$/);
      }
    });
  });

  // ── Timestamp Monotonicity ──────────────────────────────────────────────────

  describe('INVARIANT: timestamps in sequential sign calls are non-decreasing', () => {
    it('sequential sign calls produce non-decreasing Unix-second timestamps', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      let prev = 0;
      for (let i = 0; i < 20; i++) {
        const headers = auth.signRequest(Buffer.from(`{"seq":${i}}`));
        const ts = Number(headers[HEADER_DID_TIMESTAMP]);
        expect(ts).toBeGreaterThanOrEqual(prev);
        prev = ts;
      }
    });

    it('timestamp is in Unix seconds (not milliseconds) — less than 10^10', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      const headers = auth.signRequest(Buffer.from('{}'));
      const ts = Number(headers[HEADER_DID_TIMESTAMP]);
      expect(ts).toBeLessThan(10_000_000_000);
      expect(ts).toBeGreaterThan(1_000_000_000); // after year 2001
    });

    it('mocked timestamps are reflected accurately', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      const FIXED_TS = 1_700_000_000;
      vi.spyOn(Date, 'now').mockReturnValue(FIXED_TS * 1000);

      const headers = auth.signRequest(Buffer.from('{}'));
      expect(Number(headers[HEADER_DID_TIMESTAMP])).toBe(FIXED_TS);
    });
  });

  // ── Signature Sensitivity ───────────────────────────────────────────────────

  describe('INVARIANT: flip one bit in any input → different signature', () => {
    let auth: DIDAuthenticator;

    beforeEach(() => {
      const { jwk } = makeKeypair(FIXED_SEED);
      auth = new DIDAuthenticator(TEST_DID, jwk);
      // Fix nonce for comparability
      vi.spyOn(crypto, 'randomBytes').mockReturnValue(Buffer.alloc(16, 0xab) as any);
      vi.spyOn(Date, 'now').mockReturnValue(1_700_000_000 * 1000);
    });

    it('changing body content changes the signature', () => {
      const body1 = Buffer.from('{"value":1}');
      const body2 = Buffer.from('{"value":2}');

      const h1 = auth.signRequest(body1);
      const h2 = auth.signRequest(body2);

      expect(h1[HEADER_DID_SIGNATURE]).not.toBe(h2[HEADER_DID_SIGNATURE]);
    });

    it('single-byte mutation in body changes the signature', () => {
      const body = Buffer.from('{"hello":"world","x":42}');
      const mutated = Buffer.from(body);
      mutated[0] = mutated[0] ^ 0x01; // flip one bit

      const h1 = auth.signRequest(body);
      const h2 = auth.signRequest(mutated);

      expect(h1[HEADER_DID_SIGNATURE]).not.toBe(h2[HEADER_DID_SIGNATURE]);
    });

    it('changing the timestamp changes the signature (different nonce released via timestamp mock)', () => {
      const body = Buffer.from('{"same":"body"}');

      vi.spyOn(Date, 'now').mockReturnValueOnce(1_000_000 * 1000);
      const h1 = auth.signRequest(body);

      vi.spyOn(Date, 'now').mockReturnValueOnce(2_000_000 * 1000);
      const h2 = auth.signRequest(body);

      expect(h1[HEADER_DID_SIGNATURE]).not.toBe(h2[HEADER_DID_SIGNATURE]);
    });
  });

  // ── Header Completeness ────────────────────────────────────────────────────

  describe('INVARIANT: every signed request contains all 4 required DID headers', () => {
    const REQUIRED_HEADERS = [
      HEADER_CALLER_DID,
      HEADER_DID_SIGNATURE,
      HEADER_DID_TIMESTAMP,
      HEADER_DID_NONCE
    ] as const;

    it('all 4 headers present on every signed request', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      for (let i = 0; i < 10; i++) {
        const headers = auth.signRequest(Buffer.from(`{"iteration":${i}}`));
        for (const name of REQUIRED_HEADERS) {
          expect(headers, `Missing header ${name} on iteration ${i}`).toHaveProperty(name);
          expect(headers[name]).toBeTruthy();
        }
      }
    });

    it('exactly 4 headers are returned (no extras)', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);
      const headers = auth.signRequest(Buffer.from('{}'));
      expect(Object.keys(headers)).toHaveLength(4);
    });

    it('X-Caller-DID header equals the configured DID', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);
      const headers = auth.signRequest(Buffer.from('{}'));
      expect(headers[HEADER_CALLER_DID]).toBe(TEST_DID);
    });
  });

  // ── Unsigned Request Passthrough ────────────────────────────────────────────

  describe('INVARIANT: unconfigured authenticator adds no headers', () => {
    it('signRequest returns empty object when not configured', () => {
      const auth = new DIDAuthenticator();
      const headers = auth.signRequest(Buffer.from('{"data":"anything"}'));
      expect(headers).toEqual({});
      expect(Object.keys(headers)).toHaveLength(0);
    });

    it('isConfigured is false when no credentials provided', () => {
      const auth = new DIDAuthenticator();
      expect(auth.isConfigured).toBe(false);
    });

    it('after setCredentials, signed request has all 4 headers', () => {
      const { jwk } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator();

      // Still unconfigured
      expect(auth.signRequest(Buffer.from('{}'))).toEqual({});

      // Configure
      auth.setCredentials(TEST_DID, jwk);
      const headers = auth.signRequest(Buffer.from('{}'));
      expect(Object.keys(headers)).toHaveLength(4);
    });
  });

  // ── Signature Correctness ───────────────────────────────────────────────────

  describe('INVARIANT: produced signature is verifiable with the corresponding public key', () => {
    it('every signature verifies with the correct public key', () => {
      const { jwk, publicKey } = makeKeypair(FIXED_SEED);
      const auth = new DIDAuthenticator(TEST_DID, jwk);

      const bodies = [
        Buffer.from('{}'),
        Buffer.from('{"large":"payload","with":"multiple","fields":123}'),
        Buffer.alloc(256, 0xff)
      ];

      for (const body of bodies) {
        const headers = auth.signRequest(body);
        const timestamp = headers[HEADER_DID_TIMESTAMP];
        const nonce = headers[HEADER_DID_NONCE];
        const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
        const payload = `${timestamp}:${nonce}:${bodyHash}`;
        const sigBytes = Buffer.from(headers[HEADER_DID_SIGNATURE], 'base64');

        expect(crypto.verify(null, Buffer.from(payload), publicKey, sigBytes)).toBe(true);
      }
    });
  });
});
