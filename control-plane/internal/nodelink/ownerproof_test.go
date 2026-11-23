package nodelink

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

func TestOwnerProofContractA5A7A7bA11A12(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	signer := NewProofSigner([]byte("example signing key"))
	signer.Now = func() time.Time { return now }
	token, expires, err := signer.Mint("my-app", []string{ScopeRead, ScopeOwn, ScopeRead}, "run-1", 0)
	require.NoError(t, err)
	require.Equal(t, now.Add(time.Hour), expires)
	require.NoError(t, signer.Verify(token, "my-app", ScopeOwn, "run-1"))
	require.NoError(t, signer.Verify(token, "my-app", ScopeRead, "run-1"))

	_, capped, err := signer.Mint("my-app", []string{ScopeOwn}, "", 72*time.Hour)
	require.NoError(t, err)
	require.Equal(t, now.Add(24*time.Hour), capped)

	parts := strings.Split(token, ".")
	require.Len(t, parts, 3)
	require.Equal(t, "afo1", parts[0])
	require.NotContains(t, token, "=")
	enc := base64.RawURLEncoding
	raw, err := enc.DecodeString(parts[1])
	require.NoError(t, err)
	var claims struct {
		KeyID   string   `json:"kid"`
		Owner   string   `json:"o"`
		Scopes  []string `json:"s"`
		Run     string   `json:"r"`
		Issued  int64    `json:"iat"`
		Expires int64    `json:"exp"`
		ID      string   `json:"jti"`
	}
	require.NoError(t, json.Unmarshal(raw, &claims))
	require.Regexp(t, "^[a-f0-9]{16}$", claims.KeyID)
	require.Regexp(t, "^[a-f0-9]{24}$", claims.ID)
	require.Equal(t, []string{"own", "read"}, claims.Scopes)
	require.Equal(t, now.Unix(), claims.Issued)
	require.Equal(t, expires.Unix(), claims.Expires)
	require.Equal(t, "my-app", claims.Owner)
	require.Equal(t, "run-1", claims.Run)

	sig, err := enc.DecodeString(parts[2])
	require.NoError(t, err)
	sig[0] ^= 1
	modified := bytes.Replace(raw, []byte("my-app"), []byte("other-app"), 1)
	unknownField := append(bytes.Clone(raw[:len(raw)-1]), []byte(`,"other":true}`)...)
	extraJSON := append(bytes.Clone(raw), []byte(` {}`)...)
	for _, tc := range []struct {
		name, token, owner, scope, run string
	}{
		{"tampered signature", "afo1." + parts[1] + "." + enc.EncodeToString(sig), "my-app", ScopeOwn, "run-1"},
		{"tampered payload", "afo1." + enc.EncodeToString(modified) + "." + parts[2], "other-app", ScopeOwn, "run-1"},
		{"wrong owner", token, "other-app", ScopeOwn, "run-1"},
		{"missing scope", token, "my-app", ScopeAct, "run-1"},
		{"wrong run", token, "my-app", ScopeOwn, "run-2"},
		{"missing run", token, "my-app", ScopeOwn, ""},
		{"unknown field", "afo1." + enc.EncodeToString(unknownField) + "." + enc.EncodeToString(signer.mac(unknownField)), "my-app", ScopeOwn, "run-1"},
		{"multiple payloads", "afo1." + enc.EncodeToString(extraJSON) + "." + enc.EncodeToString(signer.mac(extraJSON)), "my-app", ScopeOwn, "run-1"},
		{"no dots", "afo1", "my-app", ScopeOwn, "run-1"},
		{"bad base64", "afo1.!.!", "my-app", ScopeOwn, "run-1"},
		{"empty", "", "my-app", ScopeOwn, "run-1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requireOwnerProofRefusal(t, signer.Verify(tc.token, tc.owner, tc.scope, tc.run), "run_owner_proof_invalid")
		})
	}
	rotated := NewProofSigner([]byte("a different signing key"))
	requireOwnerProofRefusal(t, rotated.Verify(token, "my-app", ScopeOwn, "run-1"), "run_owner_proof_key_unknown")

	now = expires.Add(time.Minute)
	require.NoError(t, signer.Verify(token, "my-app", ScopeOwn, "run-1"))
	now = now.Add(time.Nanosecond)
	requireOwnerProofRefusal(t, signer.Verify(token, "my-app", ScopeOwn, "run-1"), "run_owner_proof_expired")
}

func requireOwnerProofRefusal(t *testing.T, err error, code string) {
	t.Helper()
	var refusal *PipeError
	require.ErrorAs(t, err, &refusal)
	require.Equal(t, http.StatusForbidden, refusal.Code)
	require.True(t, strings.HasPrefix(refusal.Words, code+": "), refusal.Words)
}

func TestOwnerProofSecretContractA11(t *testing.T) {
	dir := t.TempDir()
	key, source, err := LoadProofSecret("", dir)
	require.NoError(t, err)
	require.Equal(t, "file", source)
	require.Len(t, key, 32)
	info, err := os.Stat(filepath.Join(dir, "run_owner_proof.key"))
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0600), info.Mode().Perm())
	reloaded, source, err := LoadProofSecret("", dir)
	require.NoError(t, err)
	require.Equal(t, "file", source)
	require.Equal(t, key, reloaded)
	proof, _, err := NewProofSigner(key).Mint("my-app", []string{ScopeOwn}, "", 0)
	require.NoError(t, err)
	require.NoError(t, NewProofSigner(reloaded).Verify(proof, "my-app", ScopeOwn, ""))

	const configured = "  example signing key  "
	key, source, err = LoadProofSecret(configured, dir)
	require.NoError(t, err)
	require.Equal(t, "configured", source)
	require.Equal(t, configured, string(key))
	_, _, err = LoadProofSecret("too short", dir)
	require.ErrorContains(t, err, "at least 16 bytes")

	// A regular file cannot contain a signing key even when run by root.
	impossibleDir := filepath.Join(dir, "run_owner_proof.key")
	logs := captureOwnerWarnings(t)
	key, source, err = LoadProofSecret("", impossibleDir)
	require.NoError(t, err)
	require.Equal(t, "process", source)
	require.Len(t, key, 32)
	require.Equal(t, 1, strings.Count(logs.String(), "will not survive a restart"))
	another, _, err := LoadProofSecret("", impossibleDir)
	require.NoError(t, err)
	require.Equal(t, key, another)
	require.Equal(t, 1, strings.Count(logs.String(), "will not survive a restart"))
}

func captureOwnerWarnings(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	old := logger.Logger
	logger.Logger = zerolog.New(&logs).Level(zerolog.WarnLevel)
	t.Cleanup(func() { logger.Logger = old })
	return &logs
}

func TestOwnerPolicyContractA3A4A5A6A15(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	signer := NewProofSigner([]byte("example signing key"))
	token, _, err := signer.Mint("my-app", []string{ScopeRead}, "", 0)
	require.NoError(t, err)
	logs := captureOwnerWarnings(t)
	policy := &OwnerPolicy{Signer: signer, Now: func() time.Time { return now }}
	strict := &OwnerPolicy{Signer: signer, RequireProof: true}

	results := []string{"proven", "unproven_accepted", "refused_missing_proof", "refused_invalid_proof"}
	initial := make(map[string]float64)
	for _, result := range results {
		initial[result] = testutil.ToFloat64(runOwnerClaims.WithLabelValues(result))
	}
	var unwired *OwnerPolicy
	require.NoError(t, unwired.Check("my-app", "", ScopeRead, ""))
	require.NoError(t, strict.Check("", "invalid", ScopeRead, ""))
	require.Empty(t, logs.String())
	for _, result := range results {
		require.Equal(t, initial[result], testutil.ToFloat64(runOwnerClaims.WithLabelValues(result)))
	}

	require.NoError(t, policy.Check("my-app", token, ScopeRead, ""))
	require.Empty(t, logs.String())
	requireOwnerProofRefusal(t, policy.Check("my-app", token, ScopeAct, ""), "run_owner_proof_invalid")
	requireOwnerProofRefusal(t, strict.Check("my-app", "", ScopeRead, ""), "run_owner_proof_required")
	require.Empty(t, logs.String())

	require.NoError(t, policy.Check("my-app", "", ScopeRead, ""))
	now = now.Add(9 * time.Minute)
	require.NoError(t, policy.Check("my-app", "", ScopeRead, ""))
	require.Equal(t, 1, strings.Count(logs.String(), "unproven run owner claim"))
	require.Contains(t, logs.String(), `"owner":"my-app"`)
	require.Contains(t, logs.String(), `"scope":"read"`)
	require.NoError(t, policy.Check("my-app", "", ScopeAct, ""))
	require.NoError(t, policy.Check("other-app", "", ScopeRead, ""))
	now = now.Add(time.Minute)
	require.NoError(t, policy.Check("my-app", "", ScopeRead, ""))
	require.Equal(t, 4, strings.Count(logs.String(), "unproven run owner claim"))
	for _, result := range results {
		delta := float64(1)
		if result == "unproven_accepted" {
			delta = 5
		}
		require.Equal(t, initial[result]+delta, testutil.ToFloat64(runOwnerClaims.WithLabelValues(result)))
	}
}
