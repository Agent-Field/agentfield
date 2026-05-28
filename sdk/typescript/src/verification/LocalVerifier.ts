/**
 * Local verification for AgentField SDK (TypeScript).
 *
 * Provides decentralized verification of incoming requests by caching policies,
 * revocation lists, and the admin's Ed25519 public key from the control plane.
 */

import { createHash } from 'node:crypto';
import axios, { type AxiosInstance } from 'axios';

export interface PolicyEntry {
  name: string;
  caller_tags: string[];
  target_tags: string[];
  allow_functions: string[];
  deny_functions: string[];
  constraints: Record<string, ConstraintEntry>;
  action: string;
  priority: number;
  enabled?: boolean;
}

export interface ConstraintEntry {
  operator: string;
  value: number;
}

export class LocalVerifier {
  private readonly agentFieldUrl: string;
  private readonly refreshInterval: number;
  private readonly timestampWindow: number;
  private readonly apiKey?: string;

  private policies: PolicyEntry[] = [];
  private revokedDids: Set<string> = new Set();
  private registeredDids: Set<string> = new Set();
  private adminPublicKeyBytes: Uint8Array | null = null;
  private issuerDid: string | null = null;
  private lastRefresh = 0;
  private initialized = false;

  constructor(
    agentFieldUrl: string,
    refreshInterval = 300,
    timestampWindow = 300,
    apiKey?: string,
  ) {
    this.agentFieldUrl = agentFieldUrl.replace(/\/+$/, '');
    this.refreshInterval = refreshInterval;
    this.timestampWindow = timestampWindow;
    this.apiKey = apiKey;
  }

  get needsRefresh(): boolean {
    return Date.now() / 1000 - this.lastRefresh > this.refreshInterval;
  }

  async refresh(): Promise<boolean> {
    const headers: Record<string, string> = {};
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    let success = true;

    // Fetch policies
    try {
      const resp = await axios.get(`${this.agentFieldUrl}/api/v1/policies`, {
        headers,
        timeout: 10_000,
      });
      if (resp.status !== 200) {
        success = false;
      } else {
        this.policies = resp.data?.policies ?? [];
      }
    } catch {
      success = false;
    }

    // Fetch revocations
    try {
      const resp = await axios.get(`${this.agentFieldUrl}/api/v1/revocations`, {
        headers,
        timeout: 10_000,
      });
      if (resp.status !== 200) {
        success = false;
      } else {
        this.revokedDids = new Set(resp.data?.revoked_dids ?? []);
      }
    } catch {
      success = false;
    }

    // Fetch registered DIDs
    try {
      const resp = await axios.get(`${this.agentFieldUrl}/api/v1/registered-dids`, {
        headers,
        timeout: 10_000,
      });
      if (resp.status !== 200) {
        success = false;
      } else {
        this.registeredDids = new Set(resp.data?.registered_dids ?? []);
      }
    } catch {
      success = false;
    }

    // Fetch admin public key
    try {
      const resp = await axios.get(`${this.agentFieldUrl}/api/v1/admin/public-key`, {
        headers,
        timeout: 10_000,
      });
      if (resp.status !== 200) {
        success = false;
      } else {
        const jwk = resp.data?.public_key_jwk;
        this.issuerDid = resp.data?.issuer_did ?? null;

        if (jwk?.x) {
          // Decode base64url public key (Node 15.7+ supports 'base64url' natively)
          this.adminPublicKeyBytes = new Uint8Array(Buffer.from(jwk.x, 'base64url'));
        }
      }
    } catch {
      success = false;
    }

    if (success) {
      this.lastRefresh = Date.now() / 1000;
      this.initialized = true;
    }

    return success;
  }

  checkRevocation(callerDid: string): boolean {
    return this.revokedDids.has(callerDid);
  }

  /**
   * Check if a caller DID is registered with the control plane.
   * Returns true if registered (known), false if unknown.
   * When the cache is empty (not yet loaded), returns true to avoid
   * blocking requests before the first refresh completes.
   */
  checkRegistration(callerDid: string): boolean {
    if (this.registeredDids.size === 0) {
      return true; // Cache not populated yet — allow
    }
    return this.registeredDids.has(callerDid);
  }

  /**
   * Resolve the public key bytes from a DID.
   *
   * For did:key, the public key is self-contained in the identifier:
   *   did:key:z<base64url(0xed01 + 32-byte-pubkey)>
   *
   * For other DID methods, falls back to the admin public key.
   */
  private resolvePublicKey(callerDid: string): Uint8Array | null {
    if (callerDid.startsWith('did:key:z')) {
      try {
        const encoded = callerDid.slice('did:key:z'.length);
        const decoded = Buffer.from(encoded, 'base64url');
        // Verify Ed25519 multicodec prefix: 0xed, 0x01
        if (decoded.length >= 34 && decoded[0] === 0xed && decoded[1] === 0x01) {
          return new Uint8Array(decoded.subarray(2, 34));
        }
        return null;
      } catch {
        return null;
      }
    }

    // Fallback: use admin public key for non-did:key methods
    return this.adminPublicKeyBytes;
  }

  async verifySignature(
    callerDid: string,
    signatureB64: string,
    timestamp: string,
    body: Buffer,
    nonce?: string,
  ): Promise<boolean> {
    // Validate timestamp window
    const ts = parseInt(timestamp, 10);
    if (isNaN(ts)) return false;

    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - ts) > this.timestampWindow) return false;

    // Resolve public key from the caller's DID
    const publicKeyBytes = this.resolvePublicKey(callerDid);
    if (!publicKeyBytes || publicKeyBytes.length !== 32) {
      return false;
    }

    try {
      const { createPublicKey, verify } = await import('node:crypto');

      // Reconstruct the signed payload: "{timestamp}[:{nonce}]:{sha256(body)}"
      // Must match the format used by SDK signing (DIDAuthenticator)
      const bodyHash = createHash('sha256').update(body).digest('hex');
      const payloadStr = nonce
        ? `${timestamp}:${nonce}:${bodyHash}`
        : `${timestamp}:${bodyHash}`;
      const payload = Buffer.from(payloadStr, 'utf-8');

      // Decode the signature
      const signatureBytes = Buffer.from(signatureB64, 'base64');

      // Create Ed25519 public key object
      const publicKey = createPublicKey({
        key: Buffer.concat([
          // Ed25519 DER prefix for a 32-byte public key
          Buffer.from('302a300506032b6570032100', 'hex'),
          Buffer.from(publicKeyBytes),
        ]),
        format: 'der',
        type: 'spki',
      });

      return verify(null, payload, publicKey, signatureBytes);
    } catch {
      return false;
    }
  }

  evaluatePolicy(
    callerTags: string[],
    targetTags: string[],
    functionName: string,
    inputParams?: Record<string, any>,
  ): boolean {
    if (!this.policies || this.policies.length === 0) {
      return false; // No policies — fail closed
    }

    // Sort by priority descending
    const sorted = [...this.policies].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));

    for (const policy of sorted) {
      if (policy.enabled === false) continue;

      // Check caller tags match
      if (policy.caller_tags?.length > 0) {
        if (!policy.caller_tags.some((t) => callerTags.includes(t))) continue;
      }

      // Check target tags match
      if (policy.target_tags?.length > 0) {
        if (!policy.target_tags.some((t) => targetTags.includes(t))) continue;
      }

      // Check deny functions first
      if (policy.deny_functions?.length > 0 && functionMatches(functionName, policy.deny_functions)) {
        return false;
      }

      // Check allow functions
      if (policy.allow_functions?.length > 0 && !functionMatches(functionName, policy.allow_functions)) {
        continue;
      }

      // Check constraints
      if (policy.constraints && inputParams) {
        if (!evaluateConstraints(policy.constraints, inputParams)) {
          return false;
        }
      }

      const action = policy.action || 'allow';
      return action === 'allow';
    }

    // No matching policy — allow by default.
    // Agent-side verification cannot resolve caller tags, so policies requiring
    // specific caller tags will never match here. The DID signature verification
    // is the primary security gate. The control plane enforces full tag-based
    // policy with caller context.
    return true;
  }
}

function functionMatches(name: string, patterns: string[]): boolean {
  for (const pattern of patterns) {
    if (pattern === '*') return true;
    if (pattern.endsWith('*') && name.startsWith(pattern.slice(0, -1))) return true;
    if (pattern.startsWith('*') && name.endsWith(pattern.slice(1))) return true;
    if (name === pattern) return true;
  }
  return false;
}

function evaluateConstraints(
  constraints: Record<string, ConstraintEntry>,
  inputParams: Record<string, any>,
): boolean {
  for (const [paramName, constraint] of Object.entries(constraints)) {
    if (!(paramName in inputParams)) continue;

    const value = Number(inputParams[paramName]);
    const threshold = Number(constraint.value);
    if (isNaN(value) || isNaN(threshold)) return false;

    switch (constraint.operator) {
      case '<=':
        if (value > threshold) return false;
        break;
      case '>=':
        if (value < threshold) return false;
        break;
      case '<':
        if (value >= threshold) return false;
        break;
      case '>':
        if (value <= threshold) return false;
        break;
      case '==':
        if (Math.abs(value - threshold) > 1e-9) return false;
        break;
    }
  }
  return true;
}
