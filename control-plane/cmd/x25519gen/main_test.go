package main

import (
	"crypto/ecdh"
	"encoding/base64"
	"testing"
)

var (
	testSeed = []byte("0123456789abcdef0123456789abcdef")
	testPath = "m/44'/12345'/0'"
)

// TestMainRuns invokes main() end-to-end. It derives both epochs from the
// hardcoded fixture and JSON-encodes the result to stdout; the success path must
// not panic or os.Exit. (Output goes to the real stdout, which is acceptable for
// this interop fixture command.)
func TestMainRuns(t *testing.T) {
	main()
}

// TestDeriveX25519PrivateKeyAtEpoch_Deterministic asserts the HKDF derivation is
// deterministic for the same (seed, path, epoch) and yields a valid X25519 key.
func TestDeriveX25519PrivateKeyAtEpoch_Deterministic(t *testing.T) {
	a, err := deriveX25519PrivateKeyAtEpoch(testSeed, testPath, 0)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	b, err := deriveX25519PrivateKeyAtEpoch(testSeed, testPath, 0)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	if string(a.Bytes()) != string(b.Bytes()) {
		t.Fatal("derivation is not deterministic for the same (seed, path, epoch)")
	}
	if len(a.PublicKey().Bytes()) != 32 {
		t.Fatalf("X25519 public key must be 32 bytes, got %d", len(a.PublicKey().Bytes()))
	}
}

// TestDeriveX25519PrivateKeyAtEpoch_EpochDivergence asserts different epochs
// derive cryptographically distinct keys (rotation retires the prior key).
func TestDeriveX25519PrivateKeyAtEpoch_EpochDivergence(t *testing.T) {
	e0, err := deriveX25519PrivateKeyAtEpoch(testSeed, testPath, 0)
	if err != nil {
		t.Fatalf("derive epoch0: %v", err)
	}
	e1, err := deriveX25519PrivateKeyAtEpoch(testSeed, testPath, 1)
	if err != nil {
		t.Fatalf("derive epoch1: %v", err)
	}
	if string(e0.Bytes()) == string(e1.Bytes()) {
		t.Fatal("epoch 0 and epoch 1 must derive different private keys")
	}
	if string(e0.PublicKey().Bytes()) == string(e1.PublicKey().Bytes()) {
		t.Fatal("epoch 0 and epoch 1 must derive different public keys")
	}
}

// TestKeypairAtEpoch_JWKShape asserts the keypair map carries a public and
// private JWK whose `x` components agree, and the private JWK exposes `d`.
func TestKeypairAtEpoch_JWKShape(t *testing.T) {
	kp, err := keypairAtEpoch(testSeed, testPath, 0)
	if err != nil {
		t.Fatalf("keypairAtEpoch: %v", err)
	}

	pub, ok := kp["publicJwk"].(map[string]interface{})
	if !ok {
		t.Fatal("publicJwk must be a map")
	}
	priv, ok := kp["privateJwk"].(map[string]interface{})
	if !ok {
		t.Fatal("privateJwk must be a map")
	}

	if pub["crv"] != "X25519" || pub["kty"] != "OKP" {
		t.Errorf("publicJwk crv/kty = %v/%v, want X25519/OKP", pub["crv"], pub["kty"])
	}
	if _, hasD := pub["d"]; hasD {
		t.Error("publicJwk must NOT carry the private `d` component")
	}
	d, ok := priv["d"].(string)
	if !ok || d == "" {
		t.Error("privateJwk must carry a non-empty `d` component")
	}
	// The public `x` in both JWKs must match (same keypair).
	if pub["x"] != priv["x"] {
		t.Errorf("publicJwk x (%v) must equal privateJwk x (%v)", pub["x"], priv["x"])
	}
	// `x` must be valid base64url and decode to a 32-byte X25519 public key.
	xStr, _ := pub["x"].(string)
	raw, err := base64.RawURLEncoding.DecodeString(xStr)
	if err != nil {
		t.Fatalf("x must be base64url: %v", err)
	}
	if len(raw) != 32 {
		t.Fatalf("decoded x must be 32 bytes, got %d", len(raw))
	}
}

// TestPrivateJWKMatchesDerivedKey asserts privateJWK/publicJWK encode the
// underlying ecdh key correctly (round-trips the public key bytes).
func TestPrivateJWKMatchesDerivedKey(t *testing.T) {
	priv, err := deriveX25519PrivateKeyAtEpoch(testSeed, testPath, 0)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	pj := privateJWK(priv)
	pubj := publicJWK(priv.PublicKey())

	wantX := base64.RawURLEncoding.EncodeToString(priv.PublicKey().Bytes())
	if pj["x"] != wantX {
		t.Errorf("privateJWK x = %v, want %v", pj["x"], wantX)
	}
	if pubj["x"] != wantX {
		t.Errorf("publicJWK x = %v, want %v", pubj["x"], wantX)
	}
	wantD := base64.RawURLEncoding.EncodeToString(priv.Bytes())
	if pj["d"] != wantD {
		t.Errorf("privateJWK d = %v, want %v", pj["d"], wantD)
	}

	// Sanity: re-import the encoded private scalar yields the same public key.
	dBytes, err := base64.RawURLEncoding.DecodeString(pj["d"].(string))
	if err != nil {
		t.Fatalf("decode d: %v", err)
	}
	reimported, err := ecdh.X25519().NewPrivateKey(dBytes)
	if err != nil {
		t.Fatalf("reimport private key: %v", err)
	}
	if string(reimported.PublicKey().Bytes()) != string(priv.PublicKey().Bytes()) {
		t.Fatal("re-imported private key must produce the same public key")
	}
}
