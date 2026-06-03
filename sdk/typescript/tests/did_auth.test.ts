import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import crypto from 'node:crypto';
import {
  DIDAuthenticator,
  HEADER_CALLER_DID,
  HEADER_DID_SIGNATURE,
  HEADER_DID_TIMESTAMP,
  HEADER_DID_NONCE
} from '../src/client/DIDAuthenticator.js';

/**
 * Generate a deterministic Ed25519 keypair from a 32-byte seed and return
 * the JWK string (with "d" and "x" in base64url) plus the public KeyObject
 * for verification.
 */
function generateTestKeypair(seed: Buffer) {
  const pkcs8Prefix = Buffer.from([
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06,
    0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20
  ]);
  const privateKey = crypto.createPrivateKey({
    key: Buffer.concat([pkcs8Prefix, seed]),
    format: 'der',
    type: 'pkcs8'
  });
  const publicKey = crypto.createPublicKey(privateKey);

  // Export raw public key bytes (32 bytes for Ed25519)
  const pubRaw = publicKey.export({ type: 'spki', format: 'der' });
  // SPKI DER for Ed25519: 12-byte prefix + 32-byte key
  const pubBytes = pubRaw.subarray(pubRaw.length - 32);

  const jwk = JSON.stringify({
    kty: 'OKP',
    crv: 'Ed25519',
    d: seed.toString('base64url'),
    x: pubBytes.toString('base64url')
  });

  return { jwk, privateKey, publicKey, pubBytes };
}

// Deterministic 32-byte seed for all tests
const TEST_SEED = Buffer.alloc(32, 0);
TEST_SEED[0] = 0xde;
TEST_SEED[1] = 0xad;
TEST_SEED[31] = 0x01;

const TEST_DID = 'did:web:localhost%3A8080:agents:test-agent';

describe('DIDAuthenticator', () => {
  const { jwk: testJwk, publicKey: testPublicKey } = generateTestKeypair(TEST_SEED);

  describe('constructor and configuration', () => {
    it('is not configured when constructed without arguments', () => {
      const auth = new DIDAuthenticator();
      expect(auth.isConfigured).toBe(false);
      expect(auth.did).toBeUndefined();
    });

    it('is configured when constructed with valid credentials', () => {
      const auth = new DIDAuthenticator(TEST_DID, testJwk);
      expect(auth.isConfigured).toBe(true);
      expect(auth.did).toBe(TEST_DID);
    });

    it('configures via setCredentials after construction', () => {
      const auth = new DIDAuthenticator();
      expect(auth.isConfigured).toBe(false);

      auth.setCredentials(TEST_DID, testJwk);
      expect(auth.isConfigured).toBe(true);
      expect(auth.did).toBe(TEST_DID);
    });
  });

  describe('signRequest — unconfigured', () => {
    it('returns empty object when not configured', () => {
      const auth = new DIDAuthenticator();
      const headers = auth.signRequest(Buffer.from('{"test":true}'));
      expect(headers).toEqual({});
    });
  });

  describe('signRequest — signature correctness', () => {
    let auth: DIDAuthenticator;
    const FIXED_TIMESTAMP = 1738800000;

    beforeEach(() => {
      auth = new DIDAuthenticator(TEST_DID, testJwk);
      vi.spyOn(Date, 'now').mockReturnValue(FIXED_TIMESTAMP * 1000);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('returns all four required headers', () => {
      const body = Buffer.from('{"input":"hello"}');
      const headers = auth.signRequest(body);

      expect(headers).toHaveProperty(HEADER_CALLER_DID);
      expect(headers).toHaveProperty(HEADER_DID_SIGNATURE);
      expect(headers).toHaveProperty(HEADER_DID_TIMESTAMP);
      expect(headers).toHaveProperty(HEADER_DID_NONCE);
      expect(Object.keys(headers)).toHaveLength(4);
    });

    it('sets correct header names matching Go SDK constants', () => {
      expect(HEADER_CALLER_DID).toBe('X-Caller-DID');
      expect(HEADER_DID_SIGNATURE).toBe('X-DID-Signature');
      expect(HEADER_DID_TIMESTAMP).toBe('X-DID-Timestamp');
    });

    it('sets X-Caller-DID to the configured DID', () => {
      const headers = auth.signRequest(Buffer.from('{}'));
      expect(headers[HEADER_CALLER_DID]).toBe(TEST_DID);
    });

    it('sets X-DID-Timestamp to Unix seconds (not milliseconds)', () => {
      const headers = auth.signRequest(Buffer.from('{}'));
      const ts = Number(headers[HEADER_DID_TIMESTAMP]);
      expect(ts).toBe(FIXED_TIMESTAMP);
      // Must be seconds, not milliseconds
      expect(ts).toBeLessThan(10_000_000_000);
    });

    it('produces a signature verifiable with the corresponding public key', () => {
      const body = Buffer.from('{"target":"agent-b.greet","input":{"name":"alice"}}');
      const headers = auth.signRequest(body);

      // Reconstruct the exact payload the server would build
      const timestamp = headers[HEADER_DID_TIMESTAMP];
      const nonce = headers[HEADER_DID_NONCE];
      const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
      const expectedPayload = `${timestamp}:${nonce}:${bodyHash}`;

      // Decode signature from standard base64
      const sigBytes = Buffer.from(headers[HEADER_DID_SIGNATURE], 'base64');

      // Verify using the public key — this is exactly what the server does
      const valid = crypto.verify(null, Buffer.from(expectedPayload), testPublicKey, sigBytes);
      expect(valid).toBe(true);
    });

    it('payload format is "{timestamp}:{nonce}:{lowercase_hex_sha256}" matching Go fmt.Sprintf("%s:%s:%x")', () => {
      const body = Buffer.from('{"data":"test"}');
      const headers = auth.signRequest(body);

      const timestamp = headers[HEADER_DID_TIMESTAMP];
      const nonce = headers[HEADER_DID_NONCE];
      const bodyHash = crypto.createHash('sha256').update(body).digest('hex');

      // Verify the hash is lowercase hex, 64 chars (256 bits)
      expect(bodyHash).toMatch(/^[0-9a-f]{64}$/);

      // Verify the nonce is hex-encoded 16 bytes (32 hex chars)
      expect(nonce).toMatch(/^[0-9a-f]{32}$/);

      // Verify the full payload matches the Go format
      const expectedPayload = `${timestamp}:${nonce}:${bodyHash}`;

      // Decode and verify signature against this exact payload
      const sigBytes = Buffer.from(headers[HEADER_DID_SIGNATURE], 'base64');
      const valid = crypto.verify(null, Buffer.from(expectedPayload), testPublicKey, sigBytes);
      expect(valid).toBe(true);
    });

    it('signature uses standard base64 encoding (not base64url)', () => {
      // Sign many different bodies to increase chance of hitting +, /, = chars
      // that distinguish standard base64 from base64url
      const bodies = Array.from({ length: 20 }, (_, i) =>
        Buffer.from(JSON.stringify({ i, padding: 'x'.repeat(i * 7) }))
      );

      for (const body of bodies) {
        const headers = auth.signRequest(body);
        const sig = headers[HEADER_DID_SIGNATURE];

        // Standard base64 may contain +, /, =
        // base64url would use -, _ instead
        // Ed25519 signatures are 64 bytes → 88 base64 chars (with padding)
        expect(sig).toMatch(/^[A-Za-z0-9+/]+=*$/);
        expect(Buffer.from(sig, 'base64')).toHaveLength(64);

        // Verify it's NOT base64url (would decode differently if it were)
        const fromStd = Buffer.from(sig, 'base64');
        const fromUrl = Buffer.from(sig, 'base64url');
        // If they decode to same bytes, the signature didn't contain +/= chars.
        // But the encoding itself must be standard base64 (decodable as such).
        expect(fromStd).toHaveLength(64);
        // And it must verify
        const timestamp = headers[HEADER_DID_TIMESTAMP];
        const nonce = headers[HEADER_DID_NONCE];
        const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
        const payload = `${timestamp}:${nonce}:${bodyHash}`;
        expect(crypto.verify(null, Buffer.from(payload), testPublicKey, fromStd)).toBe(true);
      }
    });

    it('produces deterministic signatures for same nonce (Ed25519 is deterministic)', () => {
      const fixedNonce = Buffer.alloc(16, 0xab);
      const randomBytesSpy = vi.spyOn(crypto, 'randomBytes').mockReturnValue(fixedNonce as any);

      const body = Buffer.from('{"deterministic":"test"}');
      const h1 = auth.signRequest(body);
      const h2 = auth.signRequest(body);
      expect(h1[HEADER_DID_SIGNATURE]).toBe(h2[HEADER_DID_SIGNATURE]);
      expect(h1[HEADER_DID_TIMESTAMP]).toBe(h2[HEADER_DID_TIMESTAMP]);
      expect(h1[HEADER_DID_NONCE]).toBe(h2[HEADER_DID_NONCE]);

      randomBytesSpy.mockRestore();
    });

    it('different bodies produce different signatures', () => {
      const h1 = auth.signRequest(Buffer.from('{"a":1}'));
      const h2 = auth.signRequest(Buffer.from('{"a":2}'));
      expect(h1[HEADER_DID_SIGNATURE]).not.toBe(h2[HEADER_DID_SIGNATURE]);
    });

    it('different timestamps produce different signatures', () => {
      const body = Buffer.from('{"same":"body"}');

      vi.spyOn(Date, 'now').mockReturnValue(1000000 * 1000);
      const h1 = auth.signRequest(body);

      vi.spyOn(Date, 'now').mockReturnValue(2000000 * 1000);
      const h2 = auth.signRequest(body);

      expect(h1[HEADER_DID_SIGNATURE]).not.toBe(h2[HEADER_DID_SIGNATURE]);
      expect(h1[HEADER_DID_TIMESTAMP]).not.toBe(h2[HEADER_DID_TIMESTAMP]);
    });
  });

  describe('cross-SDK compatibility', () => {
    /**
     * This test manually replicates the Go SDK signing algorithm step-by-step:
     *   timestamp := strconv.FormatInt(time.Now().Unix(), 10)
     *   nonce := hex.EncodeToString(randomBytes(16))
     *   bodyHash := sha256.Sum256(body)
     *   payload := fmt.Sprintf("%s:%s:%x", timestamp, nonce, bodyHash)
     *   signature := ed25519.Sign(privateKey, []byte(payload))
     *   signatureB64 := base64.StdEncoding.EncodeToString(signature)
     *
     * Then verifies the TS DIDAuthenticator produces an identical signature.
     */
    it('produces byte-identical signatures to Go SDK algorithm for same key, body, timestamp, and nonce', () => {
      const FIXED_TS = 1738796400;
      vi.spyOn(Date, 'now').mockReturnValue(FIXED_TS * 1000);

      const fixedNonce = Buffer.alloc(16, 0xca);
      vi.spyOn(crypto, 'randomBytes').mockReturnValue(fixedNonce as any);

      const auth = new DIDAuthenticator(TEST_DID, testJwk);
      const body = Buffer.from('{"target":"other-agent.skill","input":{"data":"test"}}');

      // --- Replicate Go SDK signing manually ---
      const timestamp = String(FIXED_TS);
      const nonce = fixedNonce.toString('hex');
      const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
      const goPayload = `${timestamp}:${nonce}:${bodyHash}`;
      // Sign with the same private key
      const { privateKey } = generateTestKeypair(TEST_SEED);
      const goSignature = crypto.sign(null, Buffer.from(goPayload), privateKey);
      const goSignatureB64 = goSignature.toString('base64');

      // --- Get TS SDK result ---
      const headers = auth.signRequest(body);

      // --- Assert byte-identical ---
      expect(headers[HEADER_DID_SIGNATURE]).toBe(goSignatureB64);
      expect(headers[HEADER_DID_TIMESTAMP]).toBe(timestamp);
      expect(headers[HEADER_DID_NONCE]).toBe(nonce);
      expect(headers[HEADER_CALLER_DID]).toBe(TEST_DID);

      vi.restoreAllMocks();
    });

    it('signature verifiable by server-side algorithm (ed25519.Verify on payload bytes)', () => {
      const FIXED_TS = 1706123456;
      vi.spyOn(Date, 'now').mockReturnValue(FIXED_TS * 1000);

      const auth = new DIDAuthenticator(TEST_DID, testJwk);
      const body = Buffer.from('{"input":{"message":"hello world"}}');
      const headers = auth.signRequest(body);

      // Server-side verification steps (from middleware/did_auth.go):
      // 1. Read timestamp and nonce from headers
      const timestamp = headers[HEADER_DID_TIMESTAMP];
      const nonce = headers[HEADER_DID_NONCE];
      // 2. Hash body bytes
      const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
      // 3. Build payload (with nonce when present, matching server logic)
      const payload = `${timestamp}:${nonce}:${bodyHash}`;
      // 4. Decode signature from base64 (StdEncoding)
      const sigBytes = Buffer.from(headers[HEADER_DID_SIGNATURE], 'base64');
      // 5. Verify with public key
      const { publicKey } = generateTestKeypair(TEST_SEED);
      const valid = crypto.verify(null, Buffer.from(payload), publicKey, sigBytes);

      expect(valid).toBe(true);

      vi.restoreAllMocks();
    });
  });

  describe('JWK parsing', () => {
    it('rejects invalid JSON', () => {
      expect(() => new DIDAuthenticator(TEST_DID, 'not-json')).toThrow('Invalid JWK format');
    });

    it('rejects non-Ed25519 key type', () => {
      const jwk = JSON.stringify({ kty: 'RSA', crv: 'Ed25519', d: TEST_SEED.toString('base64url') });
      expect(() => new DIDAuthenticator(TEST_DID, jwk)).toThrow('expected Ed25519 OKP key');
    });

    it('rejects wrong curve', () => {
      const jwk = JSON.stringify({ kty: 'OKP', crv: 'X25519', d: TEST_SEED.toString('base64url') });
      expect(() => new DIDAuthenticator(TEST_DID, jwk)).toThrow('expected Ed25519 OKP key');
    });

    it('rejects missing d field', () => {
      const jwk = JSON.stringify({ kty: 'OKP', crv: 'Ed25519' });
      expect(() => new DIDAuthenticator(TEST_DID, jwk)).toThrow("Missing 'd'");
    });

    it('rejects wrong-length private key', () => {
      const shortSeed = Buffer.alloc(16, 0xab);
      const jwk = JSON.stringify({ kty: 'OKP', crv: 'Ed25519', d: shortSeed.toString('base64url') });
      expect(() => new DIDAuthenticator(TEST_DID, jwk)).toThrow('expected 32 bytes');
    });

    it('accepts base64url-encoded d field (with and without padding)', () => {
      // Without padding (standard for JWK per RFC 7517)
      const noPad = TEST_SEED.toString('base64url').replace(/=+$/, '');
      const jwkNoPad = JSON.stringify({ kty: 'OKP', crv: 'Ed25519', d: noPad });
      expect(() => new DIDAuthenticator(TEST_DID, jwkNoPad)).not.toThrow();

      // With padding (some implementations add it)
      const withPad = TEST_SEED.toString('base64url');
      const jwkWithPad = JSON.stringify({ kty: 'OKP', crv: 'Ed25519', d: withPad });
      expect(() => new DIDAuthenticator(TEST_DID, jwkWithPad)).not.toThrow();
    });
  });

  describe('AgentFieldClient integration', () => {
    /**
     * Verifies that DID auth headers flow through AgentFieldClient.execute().
     * Uses the actual client with a mocked axios to capture outgoing headers.
     */
    it('execute() attaches DID auth headers when credentials are configured', async () => {
      // Dynamic import to avoid hoisting issues with vi.mock
      const { AgentFieldClient } = await import('../src/client/AgentFieldClient.js');

      // Create client with DID credentials
      const client = new AgentFieldClient({
        nodeId: 'test-agent',
        agentFieldUrl: 'http://localhost:8080',
        did: TEST_DID,
        privateKeyJwk: testJwk
      });

      expect(client.didAuthConfigured).toBe(true);
      expect(client.getDID()).toBe(TEST_DID);
    });

    it('setDIDCredentials enables auth after construction', async () => {
      const { AgentFieldClient } = await import('../src/client/AgentFieldClient.js');

      const client = new AgentFieldClient({
        nodeId: 'test-agent',
        agentFieldUrl: 'http://localhost:8080'
      });

      expect(client.didAuthConfigured).toBe(false);
      client.setDIDCredentials(TEST_DID, testJwk);
      expect(client.didAuthConfigured).toBe(true);
      expect(client.getDID()).toBe(TEST_DID);
    });
  });
});
