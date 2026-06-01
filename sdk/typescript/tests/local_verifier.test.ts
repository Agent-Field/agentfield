import { describe, it, expect, beforeEach } from 'vitest';
import { LocalVerifier, type PolicyEntry } from '../src/verification/LocalVerifier.js';

describe('LocalVerifier', () => {
  describe('checkRevocation', () => {
    it('returns false when DID is not revoked', () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      // revokedDids is empty by default
      expect(verifier.checkRevocation('did:web:example.com:agents:a')).toBe(false);
    });

    it('returns true when DID is revoked', () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      // Access private field for testing via refresh simulation
      (verifier as any).revokedDids = new Set(['did:web:example.com:agents:revoked']);
      expect(verifier.checkRevocation('did:web:example.com:agents:revoked')).toBe(true);
    });
  });

  describe('checkRegistration', () => {
    it('returns true when cache is empty (not yet loaded)', () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      expect(verifier.checkRegistration('did:web:example.com:agents:any')).toBe(true);
    });

    it('returns true when DID is in registered set', () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      (verifier as any).registeredDids = new Set(['did:web:example.com:agents:a']);
      expect(verifier.checkRegistration('did:web:example.com:agents:a')).toBe(true);
    });

    it('returns false when DID is not in registered set', () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      (verifier as any).registeredDids = new Set(['did:web:example.com:agents:a']);
      expect(verifier.checkRegistration('did:web:example.com:agents:unknown')).toBe(false);
    });
  });

  describe('needsRefresh', () => {
    it('returns true when never refreshed', () => {
      const verifier = new LocalVerifier('http://localhost:8080', 300);
      expect(verifier.needsRefresh).toBe(true);
    });

    it('returns false after recent refresh', () => {
      const verifier = new LocalVerifier('http://localhost:8080', 300);
      (verifier as any).lastRefresh = Date.now() / 1000;
      expect(verifier.needsRefresh).toBe(false);
    });

    it('returns true after refresh interval expires', () => {
      const verifier = new LocalVerifier('http://localhost:8080', 300);
      (verifier as any).lastRefresh = Date.now() / 1000 - 301;
      expect(verifier.needsRefresh).toBe(true);
    });
  });

  describe('evaluatePolicy', () => {
    let verifier: LocalVerifier;

    beforeEach(() => {
      verifier = new LocalVerifier('http://localhost:8080');
    });

    it('returns false (fail-closed) when no policies exist', () => {
      (verifier as any).policies = [];
      expect(verifier.evaluatePolicy([], ['tier:internal'], 'get_data')).toBe(false);
    });

    it('allows when policy matches caller and target tags', () => {
      const policy: PolicyEntry = {
        name: 'internal-access',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'get_data')).toBe(true);
    });

    it('denies when policy action is deny', () => {
      const policy: PolicyEntry = {
        name: 'block-external',
        caller_tags: ['tier:external'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'deny',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy(['tier:external'], ['tier:internal'], 'get_data')).toBe(false);
    });

    it('denies when function is in deny_functions', () => {
      const policy: PolicyEntry = {
        name: 'deny-admin',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: ['admin_*'],
        constraints: {},
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'admin_delete')).toBe(false);
    });

    it('skips disabled policies', () => {
      const policy: PolicyEntry = {
        name: 'disabled',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'deny',
        priority: 10,
        enabled: false,
      };
      (verifier as any).policies = [policy];

      // No active policy matches, so default allow
      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'get_data')).toBe(true);
    });

    it('respects priority ordering (higher priority wins)', () => {
      const denyPolicy: PolicyEntry = {
        name: 'low-priority-deny',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'deny',
        priority: 1,
      };
      const allowPolicy: PolicyEntry = {
        name: 'high-priority-allow',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [denyPolicy, allowPolicy];

      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'get_data')).toBe(true);
    });

    it('evaluates constraints - passes when within limit', () => {
      const policy: PolicyEntry = {
        name: 'constrained',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: { limit: { operator: '<=', value: 100 } },
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'query', { limit: 50 })).toBe(true);
    });

    it('evaluates constraints - fails when exceeding limit', () => {
      const policy: PolicyEntry = {
        name: 'constrained',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: { limit: { operator: '<=', value: 100 } },
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy(['tier:internal'], ['tier:internal'], 'query', { limit: 500 })).toBe(false);
    });

    it('skips policy when caller tags do not match', () => {
      const policy: PolicyEntry = {
        name: 'internal-only',
        caller_tags: ['tier:internal'],
        target_tags: ['tier:internal'],
        allow_functions: ['*'],
        deny_functions: [],
        constraints: {},
        action: 'deny',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      // Caller has 'tier:external' which doesn't match, so policy is skipped, default allow
      expect(verifier.evaluatePolicy(['tier:external'], ['tier:internal'], 'get_data')).toBe(true);
    });

    it('allows when function matches wildcard pattern', () => {
      const policy: PolicyEntry = {
        name: 'prefix-match',
        caller_tags: [],
        target_tags: [],
        allow_functions: ['read_*'],
        deny_functions: [],
        constraints: {},
        action: 'allow',
        priority: 10,
      };
      (verifier as any).policies = [policy];

      expect(verifier.evaluatePolicy([], [], 'read_users')).toBe(true);
      expect(verifier.evaluatePolicy([], [], 'write_users')).toBe(true); // no match -> default allow
    });
  });

  describe('verifySignature', () => {
    it('rejects invalid timestamp format', async () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      const result = await verifier.verifySignature('did:web:test', 'sig', 'not-a-number', Buffer.from('{}'));
      expect(result).toBe(false);
    });

    it('rejects expired timestamp', async () => {
      const verifier = new LocalVerifier('http://localhost:8080', 300, 300);
      const oldTs = String(Math.floor(Date.now() / 1000) - 600);
      const result = await verifier.verifySignature('did:web:test', 'sig', oldTs, Buffer.from('{}'));
      expect(result).toBe(false);
    });

    it('rejects when no public key available', async () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      const ts = String(Math.floor(Date.now() / 1000));
      const result = await verifier.verifySignature('did:web:unknown', 'sig', ts, Buffer.from('{}'));
      expect(result).toBe(false);
    });
  });

  describe('resolvePublicKey (via verifySignature)', () => {
    it('returns null for malformed did:key', async () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      const ts = String(Math.floor(Date.now() / 1000));
      // did:key:z with invalid base64url content
      const result = await verifier.verifySignature('did:key:zINVALID', 'c2ln', ts, Buffer.from('{}'));
      expect(result).toBe(false);
    });

    it('falls back to admin public key for non-did:key', async () => {
      const verifier = new LocalVerifier('http://localhost:8080');
      // No admin key set, so should fail
      const ts = String(Math.floor(Date.now() / 1000));
      const result = await verifier.verifySignature('did:web:example.com', 'c2ln', ts, Buffer.from('{}'));
      expect(result).toBe(false);
    });
  });
});
