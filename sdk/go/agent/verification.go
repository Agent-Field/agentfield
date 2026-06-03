package agent

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// LocalVerifier verifies incoming requests locally using cached policies,
// revocation lists, registered DIDs, and the admin's Ed25519 public key.
// Periodically refreshes caches from the control plane.
type LocalVerifier struct {
	agentFieldURL   string
	refreshInterval time.Duration
	timestampWindow int64
	apiKey          string

	mu              sync.RWMutex
	policies        []PolicyEntry
	revokedDIDs     map[string]struct{}
	registeredDIDs  map[string]struct{}
	adminPublicKey  ed25519.PublicKey
	issuerDID       string
	lastRefresh     time.Time
	initialized     bool
}

// PolicyEntry represents a cached access policy for local evaluation.
type PolicyEntry struct {
	Name           string                       `json:"name"`
	CallerTags     []string                     `json:"caller_tags"`
	TargetTags     []string                     `json:"target_tags"`
	AllowFunctions []string                     `json:"allow_functions"`
	DenyFunctions  []string                     `json:"deny_functions"`
	Constraints    map[string]ConstraintEntry    `json:"constraints"`
	Action         string                       `json:"action"`
	Priority       int                          `json:"priority"`
	Enabled        *bool                        `json:"enabled"`
}

// ConstraintEntry represents a parameter constraint in a policy.
type ConstraintEntry struct {
	Operator string  `json:"operator"`
	Value    float64 `json:"value"`
}

// NewLocalVerifier creates a new local verifier.
func NewLocalVerifier(agentFieldURL string, refreshInterval time.Duration, apiKey string) *LocalVerifier {
	return &LocalVerifier{
		agentFieldURL:   strings.TrimRight(agentFieldURL, "/"),
		refreshInterval: refreshInterval,
		timestampWindow: 300,
		apiKey:          apiKey,
		revokedDIDs:     make(map[string]struct{}),
		registeredDIDs:  make(map[string]struct{}),
	}
}

// Refresh fetches policies, revocations, registered DIDs, and admin public key from the control plane.
func (v *LocalVerifier) Refresh() error {
	client := &http.Client{Timeout: 10 * time.Second}

	// Fetch policies
	policies, err := v.fetchPolicies(client)
	if err != nil {
		return fmt.Errorf("fetch policies: %w", err)
	}

	// Fetch revocations
	revoked, err := v.fetchRevocations(client)
	if err != nil {
		return fmt.Errorf("fetch revocations: %w", err)
	}

	// Fetch registered DIDs
	registered, err := v.fetchRegisteredDIDs(client)
	if err != nil {
		return fmt.Errorf("fetch registered DIDs: %w", err)
	}

	// Fetch admin public key
	pubKey, issuerDID, err := v.fetchAdminPublicKey(client)
	if err != nil {
		return fmt.Errorf("fetch admin public key: %w", err)
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	v.policies = policies
	v.revokedDIDs = revoked
	v.registeredDIDs = registered
	v.adminPublicKey = pubKey
	v.issuerDID = issuerDID
	v.lastRefresh = time.Now()
	v.initialized = true
	return nil
}

// NeedsRefresh returns true if the cache is stale.
func (v *LocalVerifier) NeedsRefresh() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return time.Since(v.lastRefresh) > v.refreshInterval
}

// CheckRevocation returns true if the DID is revoked.
func (v *LocalVerifier) CheckRevocation(callerDID string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	_, revoked := v.revokedDIDs[callerDID]
	return revoked
}

// CheckRegistration returns true if the caller DID is registered with the control plane.
// When the cache is empty (not yet loaded), returns true to avoid blocking requests
// before the first refresh completes.
func (v *LocalVerifier) CheckRegistration(callerDID string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if len(v.registeredDIDs) == 0 {
		return true // Cache not populated yet — allow
	}
	_, registered := v.registeredDIDs[callerDID]
	return registered
}

// resolvePublicKey resolves the public key bytes from a DID.
// For did:key, the public key is self-contained in the identifier:
//
//	did:key:z<base64url(0xed01 + 32-byte-pubkey)>
//
// For other DID methods, falls back to the admin public key.
func (v *LocalVerifier) resolvePublicKey(callerDID string) ed25519.PublicKey {
	if strings.HasPrefix(callerDID, "did:key:z") {
		encoded := callerDID[len("did:key:z"):]
		decoded, err := base64.RawURLEncoding.DecodeString(encoded)
		if err != nil {
			return nil
		}
		// Verify Ed25519 multicodec prefix: 0xed, 0x01
		if len(decoded) >= 34 && decoded[0] == 0xed && decoded[1] == 0x01 {
			return ed25519.PublicKey(decoded[2:34])
		}
		return nil
	}

	// Fallback: use admin public key for non-did:key methods
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.adminPublicKey
}

// VerifySignature verifies an Ed25519 DID signature on an incoming request.
// Resolves the caller's public key from their DID (did:key embeds the key
// directly; other methods fall back to the admin public key).
func (v *LocalVerifier) VerifySignature(callerDID, signatureB64, timestamp string, body []byte, nonce string) bool {
	// Validate timestamp window
	ts, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return false
	}
	now := time.Now().Unix()
	if abs64(now-ts) > v.timestampWindow {
		return false
	}

	// Resolve public key from the caller's DID
	pubKey := v.resolvePublicKey(callerDID)
	if len(pubKey) == 0 {
		return false
	}

	// Decode signature
	sigBytes, err := base64.StdEncoding.DecodeString(signatureB64)
	if err != nil {
		return false
	}

	// Reconstruct the signed payload: "{timestamp}[:{nonce}]:{sha256(body)}"
	// Must match the format used by SDK signing (DIDAuthenticator)
	bodyHash := sha256.Sum256(body)
	var payload string
	if nonce != "" {
		payload = fmt.Sprintf("%s:%s:%x", timestamp, nonce, bodyHash)
	} else {
		payload = fmt.Sprintf("%s:%x", timestamp, bodyHash)
	}

	return ed25519.Verify(pubKey, []byte(payload), sigBytes)
}

func (v *LocalVerifier) doRequest(client *http.Client, path string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, v.agentFieldURL+path, nil)
	if err != nil {
		return nil, err
	}
	if v.apiKey != "" {
		req.Header.Set("X-API-Key", v.apiKey)
	}
	return client.Do(req)
}

func (v *LocalVerifier) fetchPolicies(client *http.Client) ([]PolicyEntry, error) {
	resp, err := v.doRequest(client, "/api/v1/policies")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var result struct {
		Policies []PolicyEntry `json:"policies"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Policies, nil
}

func (v *LocalVerifier) fetchRevocations(client *http.Client) (map[string]struct{}, error) {
	resp, err := v.doRequest(client, "/api/v1/revocations")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var result struct {
		RevokedDIDs []string `json:"revoked_dids"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	revoked := make(map[string]struct{}, len(result.RevokedDIDs))
	for _, d := range result.RevokedDIDs {
		revoked[d] = struct{}{}
	}
	return revoked, nil
}

func (v *LocalVerifier) fetchRegisteredDIDs(client *http.Client) (map[string]struct{}, error) {
	resp, err := v.doRequest(client, "/api/v1/registered-dids")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var result struct {
		RegisteredDIDs []string `json:"registered_dids"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	registered := make(map[string]struct{}, len(result.RegisteredDIDs))
	for _, d := range result.RegisteredDIDs {
		registered[d] = struct{}{}
	}
	return registered, nil
}

func (v *LocalVerifier) fetchAdminPublicKey(client *http.Client) (ed25519.PublicKey, string, error) {
	resp, err := v.doRequest(client, "/api/v1/admin/public-key")
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	var result struct {
		IssuerDID    string                 `json:"issuer_did"`
		PublicKeyJWK map[string]interface{} `json:"public_key_jwk"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, "", err
	}

	// Parse Ed25519 public key from JWK
	xValue, ok := result.PublicKeyJWK["x"].(string)
	if !ok {
		return nil, "", fmt.Errorf("missing 'x' in public key JWK")
	}
	pubKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return nil, "", fmt.Errorf("decode public key: %w", err)
	}
	if len(pubKeyBytes) != ed25519.PublicKeySize {
		return nil, "", fmt.Errorf("invalid public key size: %d", len(pubKeyBytes))
	}
	return ed25519.PublicKey(pubKeyBytes), result.IssuerDID, nil
}

func abs64(x int64) int64 {
	if x < 0 {
		neg := -x
		if neg < 0 {
			// Overflow: -math.MinInt64 overflows back to negative.
			return math.MaxInt64
		}
		return neg
	}
	return x
}

// EvaluatePolicy evaluates access policies locally.
func (v *LocalVerifier) EvaluatePolicy(callerTags, targetTags []string, functionName string, inputParams map[string]any) bool {
	v.mu.RLock()
	policies := make([]PolicyEntry, len(v.policies))
	copy(policies, v.policies)
	v.mu.RUnlock()

	if len(policies) == 0 {
		return false // No policies — fail closed
	}

	// Sort by priority descending so highest-priority policies are evaluated first.
	sort.Slice(policies, func(i, j int) bool {
		return policies[i].Priority > policies[j].Priority
	})

	for _, policy := range policies {
		if policy.Enabled != nil && !*policy.Enabled {
			continue
		}

		// Check caller tags match
		if len(policy.CallerTags) > 0 && !anyTagMatch(callerTags, policy.CallerTags) {
			continue
		}

		// Check target tags match
		if len(policy.TargetTags) > 0 && !anyTagMatch(targetTags, policy.TargetTags) {
			continue
		}

		// Check deny functions first
		if len(policy.DenyFunctions) > 0 && functionMatches(functionName, policy.DenyFunctions) {
			return false
		}

		// Check allow functions
		if len(policy.AllowFunctions) > 0 && !functionMatches(functionName, policy.AllowFunctions) {
			continue
		}

		// Check constraints
		if len(policy.Constraints) > 0 && inputParams != nil {
			if !evaluateConstraints(policy.Constraints, functionName, inputParams) {
				return false
			}
		}

		action := policy.Action
		if action == "" {
			action = "allow"
		}
		return action == "allow"
	}

	return true // No matching policy — allow by default
}

func anyTagMatch(have, want []string) bool {
	for _, w := range want {
		for _, h := range have {
			if h == w {
				return true
			}
		}
	}
	return false
}

func functionMatches(name string, patterns []string) bool {
	for _, p := range patterns {
		if matchWildcard(name, p) {
			return true
		}
	}
	return false
}

func matchWildcard(name, pattern string) bool {
	if pattern == "*" {
		return true
	}
	if strings.HasSuffix(pattern, "*") {
		return strings.HasPrefix(name, strings.TrimSuffix(pattern, "*"))
	}
	if strings.HasPrefix(pattern, "*") {
		return strings.HasSuffix(name, strings.TrimPrefix(pattern, "*"))
	}
	return name == pattern
}

func evaluateConstraints(constraints map[string]ConstraintEntry, functionName string, inputParams map[string]any) bool {
	for paramName, constraint := range constraints {
		val, ok := inputParams[paramName]
		if !ok {
			return false // Fail closed: constrained parameter missing from input
		}
		numVal, err := toFloat64(val)
		if err != nil {
			return false // Fail closed: cannot convert constrained parameter to numeric
		}
		switch constraint.Operator {
		case "<=":
			if numVal > constraint.Value {
				return false
			}
		case ">=":
			if numVal < constraint.Value {
				return false
			}
		case "<":
			if numVal >= constraint.Value {
				return false
			}
		case ">":
			if numVal <= constraint.Value {
				return false
			}
		case "==":
			if math.Abs(numVal-constraint.Value) > 1e-9 {
				return false
			}
		}
	}
	return true
}

func toFloat64(v any) (float64, error) {
	switch val := v.(type) {
	case float64:
		return val, nil
	case float32:
		return float64(val), nil
	case int:
		return float64(val), nil
	case int64:
		return float64(val), nil
	case json.Number:
		return val.Float64()
	case string:
		return strconv.ParseFloat(val, 64)
	default:
		return 0, fmt.Errorf("unsupported type %T", v)
	}
}
