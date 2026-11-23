package nodelink

import (
	"bytes"
	"crypto/hkdf"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

const (
	ScopeOwn  = "own"
	ScopeRead = "read"
	ScopeAct  = "act"

	proofVersion    = "afo1"
	defaultProofTTL = time.Hour
	maxProofTTL     = 24 * time.Hour
	proofLeeway     = time.Minute
	proofKeyFile    = "run_owner_proof.key"
	proofKeySize    = 32
)

type ownerProofPayload struct {
	KeyID   string   `json:"kid"`
	Owner   string   `json:"o"`
	Scopes  []string `json:"s"`
	Run     string   `json:"r,omitempty"`
	Issued  int64    `json:"iat"`
	Expires int64    `json:"exp"`
	ID      string   `json:"jti"`
}

// ProofSigner signs proofs of owner consent. A proof shows that a link with
// the named node ID consented. The link uses the control plane's shared API
// key or internal token, so a proof cannot identify a person or machine.
type ProofSigner struct {
	key []byte
	kid string
	Now func() time.Time
}

func NewProofSigner(secret []byte) *ProofSigner {
	digest := sha256.Sum256(secret)
	key, err := hkdf.Key(sha256.New, secret, nil, "agentfield/run-owner-proof/v1", sha256.Size)
	if err != nil {
		panic(err) // The fixed size is within the HKDF limit for SHA-256.
	}
	return &ProofSigner{key: key, kid: hex.EncodeToString(digest[:8])}
}

func (s *ProofSigner) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s *ProofSigner) mac(payload []byte) []byte {
	mac := hmac.New(sha256.New, s.key)
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (s *ProofSigner) Mint(owner string, scopes []string, run string, ttl time.Duration) (string, time.Time, error) {
	if owner == "" || len(scopes) == 0 {
		return "", time.Time{}, refuse(http.StatusBadRequest, "delegate requires an owner and at least one scope")
	}
	for _, scope := range scopes {
		switch scope {
		case ScopeOwn, ScopeRead, ScopeAct:
		default:
			return "", time.Time{}, refuse(http.StatusBadRequest, "delegate scopes must be own, read, or act")
		}
	}
	scopes = canonicalProofScopes(scopes)
	if ttl < 0 {
		return "", time.Time{}, refuse(http.StatusBadRequest, "delegate ttl_seconds must be non-negative")
	}
	ttl = effectiveProofTTL(ttl)
	now := s.now().Truncate(time.Second)
	expires := now.Add(ttl).Truncate(time.Second)
	var id [12]byte
	rand.Read(id[:])
	payload, err := json.Marshal(ownerProofPayload{
		KeyID: s.kid, Owner: owner, Scopes: scopes, Run: run,
		Issued: now.Unix(), Expires: expires.Unix(), ID: hex.EncodeToString(id[:]),
	})
	if err != nil {
		return "", time.Time{}, err
	}
	base64url := base64.RawURLEncoding
	return proofVersion + "." + base64url.EncodeToString(payload) + "." + base64url.EncodeToString(s.mac(payload)), expires, nil
}

func canonicalProofScopes(scopes []string) []string {
	out := slices.Clone(scopes)
	slices.Sort(out)
	return slices.Compact(out)
}

func effectiveProofTTL(ttl time.Duration) time.Duration {
	if ttl == 0 {
		return defaultProofTTL
	}
	return min(ttl, maxProofTTL)
}

func (s *ProofSigner) Verify(token, owner, scope, run string) error {
	invalid := refuse(http.StatusForbidden, "run_owner_proof_invalid: invalid proof of owner consent")
	parts := strings.SplitN(token, ".", 4)
	if len(parts) != 3 || parts[0] != proofVersion || strings.ContainsAny(token, "\r\n") {
		return invalid
	}
	base64url := base64.RawURLEncoding.Strict()
	body, err := base64url.DecodeString(parts[1])
	if err != nil {
		return invalid
	}
	sig, err := base64url.DecodeString(parts[2])
	if err != nil || len(sig) != sha256.Size {
		return invalid
	}
	var payload ownerProofPayload
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if err := d.Decode(&payload); err != nil || d.Decode(new(any)) != io.EOF {
		return invalid
	}
	kid, err := hex.DecodeString(payload.KeyID)
	if err != nil || len(kid) != 8 {
		return invalid
	}
	if payload.KeyID != s.kid {
		return refuse(http.StatusForbidden, "run_owner_proof_key_unknown: ask your node link for a new proof")
	}
	if !hmac.Equal(sig, s.mac(body)) || payload.Owner != owner || !slices.Contains(payload.Scopes, scope) ||
		(payload.Run != "" && payload.Run != run) {
		return invalid
	}
	if s.now().After(time.Unix(payload.Expires, 0).Add(proofLeeway)) {
		return refuse(http.StatusForbidden, "run_owner_proof_expired: ask your node link for a new proof")
	}
	return nil
}

var processProofSecret = sync.OnceValue(func() []byte {
	logger.Logger.Warn().Msg("run owner proofs will not survive a restart because the signing key could not be persisted")
	key := make([]byte, proofKeySize)
	rand.Read(key)
	return key
})

// LoadProofSecret uses the configured key verbatim, or keeps a generated key
// in the data directory so existing proofs can survive a restart.
func LoadProofSecret(configured, dir string) ([]byte, string, error) {
	if configured != "" {
		if len(configured) < 16 {
			return nil, "", fmt.Errorf("run owner proof key must be at least 16 bytes (agentfield.run_owner.proof_key or AGENTFIELD_RUN_OWNER_PROOF_KEY)")
		}
		return []byte(configured), "configured", nil
	}
	path := filepath.Join(dir, proofKeyFile)
	key, err := os.ReadFile(path)
	if err == nil && len(key) == proofKeySize {
		return key, "file", nil
	}

	if err == nil || errors.Is(err, os.ErrNotExist) {
		generated := make([]byte, proofKeySize)
		rand.Read(generated)
		if err := persistProofSecret(dir, path, generated, err == nil); err == nil || errors.Is(err, os.ErrExist) {
			// A concurrent startup can publish its key first. Read the winner.
			if key, err := os.ReadFile(path); err == nil && len(key) == proofKeySize {
				return key, "file", nil
			}
		}
	}
	return slices.Clone(processProofSecret()), "process", nil
}

func persistProofSecret(dir, path string, key []byte, replace bool) error {
	f, err := os.CreateTemp(dir, ".run_owner_proof-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, writeErr := f.Write(key)
	closeErr := f.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		return err
	}
	if replace {
		return os.Rename(f.Name(), path)
	}
	// Publish the complete key without replacing a concurrent startup's key.
	return os.Link(f.Name(), path)
}
