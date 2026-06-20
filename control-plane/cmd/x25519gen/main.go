// Command x25519gen is a standalone interop fixture: it derives an X25519
// keyAgreement keypair using the same HKDF derivation and JWK encoding the
// DID service uses, and prints the public/private JWKs as JSON. It exists so the
// control-plane-derived JWKs can be cross-checked against the SDK crypto layer
// (encrypt to the public JWK, decrypt with the private JWK).
package main

import (
	"crypto/ecdh"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"golang.org/x/crypto/hkdf"
)

// deriveX25519PrivateKeyAtEpoch mirrors DIDService.deriveX25519PrivateKeyAtEpoch:
// HKDF-SHA256 with the keyAgreement-specific salt and `<path>/enc/<epoch>` as
// info, so the derived key changes per rotation epoch.
func deriveX25519PrivateKeyAtEpoch(masterSeed []byte, derivationPath string, epoch int) (*ecdh.PrivateKey, error) {
	salt := []byte("agentfield-did-keyagreement-v1")
	info := []byte(derivationPath + "/enc/" + strconv.Itoa(epoch))

	hkdfReader := hkdf.New(sha256.New, masterSeed, salt, info)
	derivedSeed := make([]byte, 32)
	if _, err := io.ReadFull(hkdfReader, derivedSeed); err != nil {
		return nil, fmt.Errorf("HKDF X25519 key derivation failed: %w", err)
	}
	return ecdh.X25519().NewPrivateKey(derivedSeed)
}

// keypairAtEpoch derives the keypair at epoch and returns its JWK pair as a map.
func keypairAtEpoch(masterSeed []byte, derivationPath string, epoch int) (map[string]interface{}, error) {
	priv, err := deriveX25519PrivateKeyAtEpoch(masterSeed, derivationPath, epoch)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"privateJwk": privateJWK(priv),
		"publicJwk":  publicJWK(priv.PublicKey()),
	}, nil
}

func publicJWK(pub *ecdh.PublicKey) map[string]interface{} {
	return map[string]interface{}{
		"kty": "OKP",
		"crv": "X25519",
		"x":   base64.RawURLEncoding.EncodeToString(pub.Bytes()),
		"use": "enc",
		"alg": "ECDH-ES",
	}
}

func privateJWK(priv *ecdh.PrivateKey) map[string]interface{} {
	return map[string]interface{}{
		"kty": "OKP",
		"crv": "X25519",
		"x":   base64.RawURLEncoding.EncodeToString(priv.PublicKey().Bytes()),
		"d":   base64.RawURLEncoding.EncodeToString(priv.Bytes()),
		"use": "enc",
		"alg": "ECDH-ES",
	}
}

func main() {
	// Hardcoded fixture inputs (32-byte master seed + a derivation path).
	masterSeed := []byte("0123456789abcdef0123456789abcdef")
	derivationPath := "m/44'/12345'/0'"

	epoch0, err := keypairAtEpoch(masterSeed, derivationPath, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "derive epoch0 error:", err)
		os.Exit(1)
	}
	epoch1, err := keypairAtEpoch(masterSeed, derivationPath, 1)
	if err != nil {
		fmt.Fprintln(os.Stderr, "derive epoch1 error:", err)
		os.Exit(1)
	}

	// Emit both epochs so the rotation acceptance gate can encrypt to epoch0 and
	// confirm epoch1's private key cannot decrypt it. Epoch-0 keys are also
	// surfaced at the top level for backwards compatibility with the original
	// single-epoch fixture consumer.
	out := map[string]interface{}{
		"privateJwk": epoch0["privateJwk"],
		"publicJwk":  epoch0["publicJwk"],
		"epoch0":     epoch0,
		"epoch1":     epoch1,
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, "encode error:", err)
		os.Exit(1)
	}
}
