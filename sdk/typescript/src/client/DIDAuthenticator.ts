import crypto from 'node:crypto';

export const HEADER_CALLER_DID = 'X-Caller-DID';
export const HEADER_DID_SIGNATURE = 'X-DID-Signature';
export const HEADER_DID_TIMESTAMP = 'X-DID-Timestamp';
export const HEADER_DID_NONCE = 'X-DID-Nonce';

/**
 * Ed25519 PKCS#8 DER prefix for wrapping a 32-byte seed into a valid
 * PKCS#8 structure that Node.js `crypto.createPrivateKey` can parse.
 */
const ED25519_PKCS8_PREFIX = Buffer.from([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06,
  0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20
]);

interface EdDSAJWK {
  kty: string;
  crv: string;
  d?: string;
  x?: string;
}

export class DIDAuthenticator {
  private _did?: string;
  private _privateKey?: crypto.KeyObject;

  constructor(did?: string, privateKeyJwk?: string) {
    if (did && privateKeyJwk) {
      this.setCredentials(did, privateKeyJwk);
    }
  }

  get isConfigured(): boolean {
    return this._did !== undefined && this._privateKey !== undefined;
  }

  get did(): string | undefined {
    return this._did;
  }

  signRequest(body: Buffer | Uint8Array): Record<string, string> {
    if (!this.isConfigured) {
      return {};
    }

    const timestamp = Math.floor(Date.now() / 1000).toString();
    const nonce = crypto.randomBytes(16).toString('hex');
    const bodyHash = crypto.createHash('sha256').update(body).digest('hex');
    const payload = `${timestamp}:${nonce}:${bodyHash}`;
    const signature = crypto.sign(null, Buffer.from(payload), this._privateKey!);
    const signatureB64 = signature.toString('base64');

    return {
      [HEADER_CALLER_DID]: this._did!,
      [HEADER_DID_SIGNATURE]: signatureB64,
      [HEADER_DID_TIMESTAMP]: timestamp,
      [HEADER_DID_NONCE]: nonce
    };
  }

  setCredentials(did: string, privateKeyJwk: string): void {
    this._privateKey = parsePrivateKeyJWK(privateKeyJwk);
    this._did = did;
  }
}

function parsePrivateKeyJWK(jwkJSON: string): crypto.KeyObject {
  let key: EdDSAJWK;
  try {
    key = JSON.parse(jwkJSON);
  } catch {
    throw new Error('Invalid JWK format: failed to parse JSON');
  }

  if (key.kty !== 'OKP' || key.crv !== 'Ed25519') {
    throw new Error('Invalid key type: expected Ed25519 OKP key');
  }

  if (!key.d) {
    throw new Error("Missing 'd' (private key) in JWK");
  }

  const seedBytes = Buffer.from(key.d, 'base64url');
  if (seedBytes.length !== 32) {
    throw new Error(`Invalid private key length: expected 32 bytes, got ${seedBytes.length}`);
  }

  return crypto.createPrivateKey({
    key: Buffer.concat([ED25519_PKCS8_PREFIX, seedBytes]),
    format: 'der',
    type: 'pkcs8'
  });
}
