/**
 * Interop CLI exercising the real TypeScript SDK crypto primitives.
 * Used by run_interop.sh to prove TS<->Python JWE interop end-to-end.
 *
 *   tsx ts_tool.mts genkey            -> prints {privateJwk, publicJwk}
 *   tsx ts_tool.mts encrypt <pubFile> <payload>  -> prints compact JWE
 *   tsx ts_tool.mts decrypt <privFile> <jweFile> -> prints decrypted plaintext
 */
import { readFileSync } from 'node:fs';
import {
  decryptToString,
  encryptToJwk,
  generateX25519KeyPair,
} from '../typescript/src/crypto/didEncryption.js';

const [cmd, a, b] = process.argv.slice(2);

if (cmd === 'genkey') {
  const { privateJwk, publicJwk } = await generateX25519KeyPair();
  process.stdout.write(JSON.stringify({ privateJwk, publicJwk }));
} else if (cmd === 'encrypt') {
  const publicJwk = JSON.parse(readFileSync(a, 'utf8'));
  process.stdout.write(await encryptToJwk(publicJwk, b));
} else if (cmd === 'decrypt') {
  const privateJwk = JSON.parse(readFileSync(a, 'utf8'));
  const token = readFileSync(b, 'utf8').trim();
  process.stdout.write(await decryptToString(token, privateJwk));
} else {
  process.stderr.write(`unknown command: ${cmd}\n`);
  process.exit(2);
}
