import { describe, expect, it } from 'vitest';
import {
  KEY_AGREEMENT_TYPE,
  PayloadEncryptionError,
  decrypt,
  decryptToString,
  encryptForDid,
  encryptToJwk,
  extractKeyAgreementJwk,
  generateX25519KeyPair,
} from '../src/crypto/didEncryption.js';

function didDocument(publicJwk: Record<string, unknown>, did = 'did:key:zAgg') {
  return {
    id: did,
    verificationMethod: [
      {
        id: `${did}#key-agreement-1`,
        type: KEY_AGREEMENT_TYPE,
        controller: did,
        publicKeyJwk: publicJwk,
      },
    ],
    keyAgreement: [`${did}#key-agreement-1`],
  };
}

describe('DID payload encryption', () => {
  it('round-trips an object payload', async () => {
    const { privateJwk, publicJwk } = await generateX25519KeyPair();
    const payload = { scope: { tier: 'workspace', workspaceId: 'ws_1' } };
    const token = await encryptToJwk(publicJwk, payload);
    expect(JSON.parse(await decryptToString(token, privateJwk))).toEqual(payload);
  });

  it('round-trips string and bytes', async () => {
    const { privateJwk, publicJwk } = await generateX25519KeyPair();
    expect(await decryptToString(await encryptToJwk(publicJwk, 'hi'), privateJwk)).toBe('hi');
    const bytes = new Uint8Array([1, 2, 3]);
    expect(await decrypt(await encryptToJwk(publicJwk, bytes), privateJwk)).toEqual(bytes);
  });

  it('encryptForDid resolves the keyAgreement key from a DID document', async () => {
    const { privateJwk, publicJwk } = await generateX25519KeyPair();
    const doc = didDocument(publicJwk as Record<string, unknown>);
    const token = await encryptForDid('did:key:zAgg', { x: 1 }, async () => doc);
    expect(JSON.parse(await decryptToString(token, privateJwk))).toEqual({ x: 1 });
  });

  it('unwraps a control-plane resolve response (did_document)', async () => {
    const { privateJwk, publicJwk } = await generateX25519KeyPair();
    const resp = { did: 'did:key:zAgg', did_document: didDocument(publicJwk as Record<string, unknown>) };
    const token = await encryptForDid('did:key:zAgg', 'wrapped', async () => resp);
    expect(await decryptToString(token, privateJwk)).toBe('wrapped');
  });

  it('accepts a flat key_agreement JWK (CP did:key resolve response)', async () => {
    const { privateJwk, publicJwk } = await generateX25519KeyPair();
    const resp = { did: 'did:key:zAgg', key_agreement: publicJwk };
    const token = await encryptForDid('did:key:zAgg', 'flat', async () => resp);
    expect(await decryptToString(token, privateJwk)).toBe('flat');
  });

  it('rejects decryption with the wrong key', async () => {
    const { publicJwk } = await generateX25519KeyPair();
    const { privateJwk: otherPriv } = await generateX25519KeyPair();
    const token = await encryptToJwk(publicJwk, 'secret');
    await expect(decrypt(token, otherPriv)).rejects.toBeInstanceOf(PayloadEncryptionError);
  });

  it('throws when the DID document has no keyAgreement key', () => {
    expect(() => extractKeyAgreementJwk({ id: 'did:key:z', verificationMethod: [] })).toThrow(
      /no keyAgreement/,
    );
  });

  it('throws when the DID cannot be resolved', async () => {
    await expect(encryptForDid('did:key:zMissing', 'x', async () => null)).rejects.toThrow(
      /could not resolve/,
    );
  });

  it('rejects a non-X25519 public key', async () => {
    await expect(encryptToJwk({ kty: 'OKP', crv: 'Ed25519', x: 'AAAA' }, 'x')).rejects.toThrow(
      /X25519/,
    );
  });
});
