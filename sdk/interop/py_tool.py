"""Interop CLI exercising the real Python SDK crypto primitives.

Used by run_interop.sh to prove TS<->Python JWE interop end-to-end.

    python py_tool.py genkey                       -> prints {privateJwk, publicJwk}
    python py_tool.py encrypt <pubFile> <payload>  -> prints compact JWE
    python py_tool.py decrypt <privFile> <jweFile> -> prints decrypted plaintext
"""

import json
import sys

from agentfield import crypto


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "genkey":
        priv, pub = crypto.generate_x25519_keypair()
        sys.stdout.write(json.dumps({"privateJwk": priv, "publicJwk": pub}))
    elif cmd == "encrypt":
        public_jwk = json.load(open(sys.argv[2]))
        sys.stdout.write(crypto.encrypt_to_jwk(public_jwk, sys.argv[3]))
    elif cmd == "decrypt":
        private_jwk = json.load(open(sys.argv[2]))
        token = open(sys.argv[3]).read().strip()
        sys.stdout.write(crypto.decrypt(token, private_jwk).decode("utf-8"))
    else:
        sys.stderr.write(f"unknown command: {cmd}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
