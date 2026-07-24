/**
 * DID-based payload encryption (JWE over X25519 keyAgreement keys).
 *
 * Encrypt a payload *to* an agent's DID so that only that agent — the holder of
 * the matching X25519 private key — can decrypt it. This underpins the
 * discuss/aggregator split: hax-sdk encrypts a scoped payload to the
 * aggregator's DID; the untrusted discuss agent forwards the ciphertext but
 * cannot read it; only the aggregator decrypts.
 *
 * Wire format is standard **JWE compact, `ECDH-ES` + `A256GCM`** over an X25519
 * key (RFC 7518 / RFC 8037), interoperable with the Python SDK's
 * `encrypt_for_did` / `decrypt` (which use `joserfc`). A ciphertext produced
 * here decrypts there and vice-versa.
 *
 * Key-ownership model: the agent owns its keypair. The X25519 private key lives
 * in the agent's environment; the public key is published in the agent's DID
 * document as a `keyAgreement` verification method of type
 * `X25519KeyAgreementKey2020`. The control plane never holds the private key.
 */

import {
  CompactEncrypt,
  compactDecrypt,
  exportJWK,
  generateKeyPair,
  importJWK,
  type JWK,
} from 'jose';

/** JOSE parameters. Must match the Python SDK exactly for interop. */
export const JWE_ALG = 'ECDH-ES' as const;
export const JWE_ENC = 'A256GCM' as const;

/** W3C verification-method type published in the DID document for the X25519 key. */
export const KEY_AGREEMENT_TYPE = 'X25519KeyAgreementKey2020' as const;

export class PayloadEncryptionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PayloadEncryptionError';
  }
}

/** Resolves a DID string to its document (or a control-plane resolve response). */
export type DidResolver = (did: string) => Promise<Record<string, unknown> | null>;

export type Payload = string | Uint8Array | Record<string, unknown>;

const encoder = new TextEncoder();

function payloadToBytes(payload: Payload): Uint8Array {
  if (payload instanceof Uint8Array) return payload;
  if (typeof payload === 'string') return encoder.encode(payload);
  return encoder.encode(JSON.stringify(payload));
}

/**
 * Generate a fresh X25519 keypair for an agent. Persist `privateJwk` into the
 * agent's environment and publish `publicJwk` as the DID `keyAgreement` key.
 */
export async function generateX25519KeyPair(): Promise<{ privateJwk: JWK; publicJwk: JWK }> {
  const { privateKey, publicKey } = await generateKeyPair('ECDH-ES', {
    crv: 'X25519',
    extractable: true,
  });
  const privateJwk = await exportJWK(privateKey);
  const publicJwk = await exportJWK(publicKey);
  return { privateJwk, publicJwk };
}

/**
 * Pull the X25519 `keyAgreement` public JWK out of a resolved DID document.
 * Handles a bare W3C document or a control-plane response wrapping it under
 * `did_document`, and a `keyAgreement` entry that is either inline (with
 * `publicKeyJwk`) or a string reference into `verificationMethod`.
 */
export function extractKeyAgreementJwk(didDocument: Record<string, unknown>): JWK {
  if (typeof didDocument !== 'object' || didDocument === null) {
    throw new PayloadEncryptionError('DID document must be an object');
  }
  const doc = (didDocument['did_document'] as Record<string, unknown>) ?? didDocument;

  // Control-plane resolve responses for did:key return the X25519 public key as a
  // flat `key_agreement` JWK rather than a full W3C keyAgreement array.
  const flat = doc['key_agreement'] as JWK | undefined;
  if (flat && typeof flat === 'object' && flat.crv === 'X25519') {
    return flat;
  }

  const keyAgreement = doc['keyAgreement'];
  if (!keyAgreement || (Array.isArray(keyAgreement) && keyAgreement.length === 0)) {
    throw new PayloadEncryptionError(
      'DID document has no keyAgreement key; the agent has not published an X25519 encryption key',
    );
  }

  const verificationMethods = new Map<string, Record<string, unknown>>();
  for (const vm of (doc['verificationMethod'] as Record<string, unknown>[]) ?? []) {
    if (vm && typeof vm === 'object' && typeof vm['id'] === 'string') {
      verificationMethods.set(vm['id'] as string, vm);
    }
  }

  const entry = Array.isArray(keyAgreement) ? keyAgreement[0] : keyAgreement;
  let vm: Record<string, unknown> | undefined;
  if (typeof entry === 'string') {
    vm = verificationMethods.get(entry);
    if (!vm) {
      throw new PayloadEncryptionError(
        `keyAgreement references unknown verification method ${entry}`,
      );
    }
  } else if (entry && typeof entry === 'object') {
    vm = entry as Record<string, unknown>;
  } else {
    throw new PayloadEncryptionError('unsupported keyAgreement entry shape');
  }

  const jwk = vm['publicKeyJwk'] as JWK | undefined;
  if (!jwk || jwk.crv !== 'X25519') {
    throw new PayloadEncryptionError(
      'keyAgreement verification method does not carry an X25519 publicKeyJwk',
    );
  }
  return jwk;
}

/** Encrypt `payload` to an X25519 public JWK, returning a compact JWE string. */
export async function encryptToJwk(publicJwk: JWK, payload: Payload): Promise<string> {
  if (!publicJwk || publicJwk.crv !== 'X25519') {
    throw new PayloadEncryptionError('publicJwk must be an X25519 OKP JWK');
  }
  try {
    const key = await importJWK(publicJwk, JWE_ALG);
    return await new CompactEncrypt(payloadToBytes(payload))
      .setProtectedHeader({ alg: JWE_ALG, enc: JWE_ENC })
      .encrypt(key);
  } catch (err) {
    if (err instanceof PayloadEncryptionError) throw err;
    throw new PayloadEncryptionError(`encryption failed: ${(err as Error).message}`);
  }
}

/**
 * Resolve `did`, read its X25519 keyAgreement key, and encrypt `payload` to it.
 * Returns a compact JWE that only the DID's owner can decrypt.
 */
export async function encryptForDid(
  did: string,
  payload: Payload,
  resolver: DidResolver,
): Promise<string> {
  const document = await resolver(did);
  if (!document) {
    throw new PayloadEncryptionError(`could not resolve DID ${did}`);
  }
  const publicJwk = extractKeyAgreementJwk(document);
  return encryptToJwk(publicJwk, payload);
}

/**
 * Decrypt a compact JWE produced by {@link encryptForDid} or the Python SDK.
 * Returns the raw plaintext bytes.
 */
export async function decrypt(token: string, privateJwk: JWK): Promise<Uint8Array> {
  try {
    const key = await importJWK(privateJwk, JWE_ALG);
    const { plaintext } = await compactDecrypt(token, key);
    // Normalize to a plain Uint8Array (jose may hand back a Node Buffer subclass).
    return Uint8Array.from(plaintext);
  } catch (err) {
    throw new PayloadEncryptionError(`decryption failed: ${(err as Error).message}`);
  }
}

/** Convenience: decrypt and decode the plaintext as UTF-8 text. */
export async function decryptToString(token: string, privateJwk: JWK): Promise<string> {
  return new TextDecoder().decode(await decrypt(token, privateJwk));
}
