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

	"golang.org/x/crypto/hkdf"
)

// deriveX25519PrivateKey mirrors DIDService.deriveX25519PrivateKey: HKDF-SHA256
// with the keyAgreement-specific salt and the derivation path as info.
func deriveX25519PrivateKey(masterSeed []byte, derivationPath string) (*ecdh.PrivateKey, error) {
	salt := []byte("agentfield-did-keyagreement-v1")
	info := []byte(derivationPath)

	hkdfReader := hkdf.New(sha256.New, masterSeed, salt, info)
	derivedSeed := make([]byte, 32)
	if _, err := io.ReadFull(hkdfReader, derivedSeed); err != nil {
		return nil, fmt.Errorf("HKDF X25519 key derivation failed: %w", err)
	}
	return ecdh.X25519().NewPrivateKey(derivedSeed)
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

	priv, err := deriveX25519PrivateKey(masterSeed, derivationPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "derive error:", err)
		os.Exit(1)
	}

	out := map[string]interface{}{
		"privateJwk": privateJWK(priv),
		"publicJwk":  publicJWK(priv.PublicKey()),
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, "encode error:", err)
		os.Exit(1)
	}
}
