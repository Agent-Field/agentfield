package services

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"golang.org/x/crypto/hkdf"
)

// rootDerivationPath is the path the af server root DID is derived from.
const rootDerivationPath = "m/44'/0'"

// deriveEd25519Key derives an Ed25519 key from the master seed using HKDF
// (RFC 5869) with the derivation path as info. It is the single source of truth
// for identity key derivation.
func deriveEd25519Key(masterSeed []byte, derivationPath string) (ed25519.PrivateKey, error) {
	salt := []byte("agentfield-did-key-derivation-v1")
	hkdfReader := hkdf.New(sha256.New, masterSeed, salt, []byte(derivationPath))
	derivedSeed := make([]byte, ed25519.SeedSize)
	if _, err := io.ReadFull(hkdfReader, derivedSeed); err != nil {
		return nil, fmt.Errorf("HKDF key derivation failed: %w", err)
	}
	return ed25519.NewKeyFromSeed(derivedSeed), nil
}

// didKeyFromPublicKey encodes an Ed25519 public key as the did:key form used by
// the control plane: "did:key:z" + base64url(0xed01 || publicKey).
func didKeyFromPublicKey(publicKey ed25519.PublicKey) string {
	multicodecKey := append([]byte{0xed, 0x01}, publicKey...)
	return "did:key:z" + base64.RawURLEncoding.EncodeToString(multicodecKey)
}

// publicKeyFromDIDKey decodes the Ed25519 public key embedded in a control-plane
// did:key identifier. The DID itself is the public key, so this needs no state.
func publicKeyFromDIDKey(did string) (ed25519.PublicKey, error) {
	const prefix = "did:key:z"
	if !strings.HasPrefix(did, prefix) {
		return nil, fmt.Errorf("not a did:key identifier")
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(did, prefix))
	if err != nil {
		return nil, fmt.Errorf("invalid did:key encoding: %w", err)
	}
	if len(raw) != 2+ed25519.PublicKeySize || raw[0] != 0xed || raw[1] != 0x01 {
		return nil, fmt.Errorf("did:key does not encode an Ed25519 public key")
	}
	return ed25519.PublicKey(raw[2:]), nil
}

// publicKeyJWKFromDIDKey returns the public JWK for a control-plane did:key.
func publicKeyJWKFromDIDKey(did string) (string, error) {
	publicKey, err := publicKeyFromDIDKey(did)
	if err != nil {
		return "", err
	}
	return ed25519PublicKeyJWK(publicKey)
}

func ed25519PublicKeyJWK(publicKey ed25519.PublicKey) (string, error) {
	jwk := map[string]interface{}{
		"kty": "OKP",
		"crv": "Ed25519",
		"x":   base64.RawURLEncoding.EncodeToString(publicKey),
		"use": "sig",
		"alg": "EdDSA",
	}
	jwkBytes, err := json.Marshal(jwk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal JWK: %w", err)
	}
	return string(jwkBytes), nil
}

// didForPath returns the DID derived from masterSeed at derivationPath.
func didForPath(masterSeed []byte, derivationPath string) (string, error) {
	privateKey, err := deriveEd25519Key(masterSeed, derivationPath)
	if err != nil {
		return "", err
	}
	return didKeyFromPublicKey(privateKey.Public().(ed25519.PublicKey)), nil
}

// identityMatchesPath reports whether derivationPath re-derives exactly did.
func identityMatchesPath(masterSeed []byte, derivationPath, did string) bool {
	if len(masterSeed) == 0 || derivationPath == "" || did == "" {
		return false
	}
	derived, err := didForPath(masterSeed, derivationPath)
	return err == nil && derived == did
}

// serverPathSegment folds an af server ID into the BIP32-style path segment
// used for all agent and component derivation paths under that server.
func serverPathSegment(agentfieldServerID string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(agentfieldServerID))
	return h.Sum32() % (1 << 31)
}

func agentDerivationPath(serverSegment uint32, agentIndex int) string {
	return fmt.Sprintf("m/44'/%d'/%d'", serverSegment, agentIndex)
}

// componentBranch is the path branch for a component type: 0 for reasoners and
// 1 for skills.
func componentBranch(componentType string) int {
	if componentType == "skill" {
		return 1
	}
	return 0
}

func componentDerivationPath(agentPath string, branch, componentIndex int) string {
	return fmt.Sprintf("%s/%d'/%d'", agentPath, branch, componentIndex)
}

// hardenedSegments parses "m/44'/a'/b'" into [44 a b]. ok is false for any path
// that is not entirely hardened numeric segments.
func hardenedSegments(path string) ([]int, bool) {
	parts := strings.Split(path, "/")
	if len(parts) < 2 || parts[0] != "m" {
		return nil, false
	}
	segments := make([]int, 0, len(parts)-1)
	for _, part := range parts[1:] {
		if !strings.HasSuffix(part, "'") {
			return nil, false
		}
		value, err := strconv.Atoi(strings.TrimSuffix(part, "'"))
		if err != nil || value < 0 {
			return nil, false
		}
		segments = append(segments, value)
	}
	return segments, true
}

// agentIndexFromPath extracts the agent index from an agent path under the
// given server segment.
func agentIndexFromPath(path string, serverSegment uint32) (int, bool) {
	segments, ok := hardenedSegments(path)
	if !ok || len(segments) != 3 || segments[0] != 44 || segments[1] != int(serverSegment) {
		return 0, false
	}
	return segments[2], true
}

// componentIndexFromPath extracts the component index when path is a component
// of agentPath on the given branch.
func componentIndexFromPath(path, agentPath string, branch int) (int, bool) {
	prefix := fmt.Sprintf("%s/%d'/", agentPath, branch)
	if !strings.HasPrefix(path, prefix) {
		return 0, false
	}
	rest := strings.TrimPrefix(path, prefix)
	if !strings.HasSuffix(rest, "'") {
		return 0, false
	}
	value, err := strconv.Atoi(strings.TrimSuffix(rest, "'"))
	if err != nil || value < 0 {
		return 0, false
	}
	return value, true
}

// LegacyAgentFieldServerID is the historical server ID: the first 16 hex chars
// of SHA-256 over the absolute AGENTFIELD_HOME path. It is still used to find
// an existing registry and to name a brand-new one, but it is no longer the
// only way a server finds its identity.
func LegacyAgentFieldServerID(agentfieldHome string, absPath func(string) (string, error)) string {
	if absPath == nil {
		absPath = filepath.Abs
	}
	resolved, err := absPath(agentfieldHome)
	if err != nil {
		resolved = agentfieldHome
	}
	hash := sha256.Sum256([]byte(resolved))
	return hex.EncodeToString(hash[:])[:16]
}

// NewAgentFieldServerID mints a server ID for a brand-new DID registry: 8 random
// bytes, hex encoded, giving the same 16-character shape as the legacy
// home-path IDs. The ID is folded into every derivation path, so generating it
// randomly keeps a control plane's identity independent of where
// AGENTFIELD_HOME happened to live when the registry was created.
func NewAgentFieldServerID() (string, error) {
	raw := make([]byte, 8)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("failed to generate a random af server ID: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

// SelectAgentFieldServerID picks which persisted DID registry this control plane
// owns, so that relocating AGENTFIELD_HOME does not silently mint a new identity.
//
//  1. An explicitly configured server ID always wins.
//  2. A registry already stored under the legacy home-path ID is used, so
//     installs created before random IDs keep their identity.
//  3. With no registries at all, a fresh random ID names the new registry; the
//     next boot adopts it through rule 4.
//  4. With exactly one registry, it is adopted even if the home path changed.
//  5. Otherwise the choice is ambiguous and startup fails closed.
func SelectAgentFieldServerID(registry *DIDRegistry, configuredID, legacyID string) (string, error) {
	if configured := strings.TrimSpace(configuredID); configured != "" {
		return configured, nil
	}
	if registry == nil {
		return NewAgentFieldServerID()
	}

	existing, err := registry.ListRegistries()
	if err != nil {
		return "", fmt.Errorf("failed to list DID registries: %w", err)
	}
	ids := make([]string, 0, len(existing))
	for _, reg := range existing {
		if reg.AgentFieldServerID == legacyID {
			return legacyID, nil
		}
		ids = append(ids, reg.AgentFieldServerID)
	}
	sort.Strings(ids)

	switch len(ids) {
	case 0:
		// Brand-new install: name the registry with random bytes rather than a
		// hash of AGENTFIELD_HOME. The next boot finds exactly one registry and
		// adopts it.
		return NewAgentFieldServerID()
	case 1:
		logger.Logger.Warn().
			Str("stored_server_id", ids[0]).
			Str("home_derived_server_id", legacyID).
			Msg("AGENTFIELD_HOME path changed since the DID registry was created; keeping the existing server identity")
		return ids[0], nil
	default:
		return "", fmt.Errorf(
			"found %d DID registries (%s) and none matches this AGENTFIELD_HOME (%s); "+
				"refusing to guess which server identity to use. Set AGENTFIELD_DID_SERVER_ID "+
				"(or features.did.server_id) to the ID this control plane should keep",
			len(ids), strings.Join(ids, ", "), legacyID)
	}
}

// legacyPathIndex recovers derivation paths for identities persisted by
// versions that stored a placeholder path. Candidate paths are derived lazily
// and matched against the DID, which encodes the public key.
type legacyPathIndex struct {
	masterSeed     []byte
	serverSegment  uint32
	maxAgentIndex  int
	maxComponentIx int
	agents         map[string]string
	componentsByIx map[int]map[string]string
}

func newLegacyPathIndex(registry *types.DIDRegistry) *legacyPathIndex {
	maxComponents := 0
	for _, agent := range registry.AgentNodes {
		if n := len(agent.Reasoners); n > maxComponents {
			maxComponents = n
		}
		if n := len(agent.Skills); n > maxComponents {
			maxComponents = n
		}
	}
	return &legacyPathIndex{
		masterSeed:     registry.MasterSeed,
		serverSegment:  serverPathSegment(registry.AgentFieldServerID),
		maxAgentIndex:  maxInt(64, 2*len(registry.AgentNodes)+16),
		maxComponentIx: maxInt(32, 2*maxComponents+8),
		componentsByIx: make(map[int]map[string]string),
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (ix *legacyPathIndex) agentPath(did string) (string, bool) {
	if ix.agents == nil {
		ix.agents = make(map[string]string, ix.maxAgentIndex+1)
		for i := 0; i <= ix.maxAgentIndex; i++ {
			path := agentDerivationPath(ix.serverSegment, i)
			if derived, err := didForPath(ix.masterSeed, path); err == nil {
				ix.agents[derived] = path
			}
		}
	}
	path, ok := ix.agents[did]
	return path, ok
}

func (ix *legacyPathIndex) componentsUnderAgent(agentIndex int) map[string]string {
	if table, ok := ix.componentsByIx[agentIndex]; ok {
		return table
	}
	agentPath := agentDerivationPath(ix.serverSegment, agentIndex)
	table := make(map[string]string, 2*(ix.maxComponentIx+1))
	for branch := 0; branch <= 1; branch++ {
		for j := 0; j <= ix.maxComponentIx; j++ {
			path := componentDerivationPath(agentPath, branch, j)
			if derived, err := didForPath(ix.masterSeed, path); err == nil {
				table[derived] = path
			}
		}
	}
	ix.componentsByIx[agentIndex] = table
	return table
}

// componentPath looks under the owning agent first, then under every other
// agent index, because older partial registrations picked the agent index from
// unordered map iteration.
func (ix *legacyPathIndex) componentPath(did string, preferredAgentIndex int, hasPreferred bool) (string, bool) {
	if hasPreferred {
		if path, ok := ix.componentsUnderAgent(preferredAgentIndex)[did]; ok {
			return path, true
		}
	}
	for i := 0; i <= ix.maxAgentIndex; i++ {
		if hasPreferred && i == preferredAgentIndex {
			continue
		}
		if path, ok := ix.componentsUnderAgent(i)[did]; ok {
			return path, true
		}
	}
	return "", false
}

// verifyRegistryIdentities checks that every identity in the registry
// re-derives to its DID. Stale placeholder paths are replaced with the
// recovered real path. Identities that cannot be recovered are returned as
// quarantined, keyed by DID, and must never be given keys or resolved.
func verifyRegistryIdentities(registry *types.DIDRegistry) (repaired int, quarantined map[string]string, rootErr error) {
	quarantined = make(map[string]string)

	if !identityMatchesPath(registry.MasterSeed, rootDerivationPath, registry.RootDID) {
		rootErr = fmt.Errorf("root DID %s for af server %s does not match its master seed (wrong encryption passphrase or corrupted seed)",
			registry.RootDID, registry.AgentFieldServerID)
		for _, agent := range registry.AgentNodes {
			quarantined[agent.DID] = rootErr.Error()
			for _, reasoner := range agent.Reasoners {
				quarantined[reasoner.DID] = rootErr.Error()
			}
			for _, skill := range agent.Skills {
				quarantined[skill.DID] = rootErr.Error()
			}
		}
		return 0, quarantined, rootErr
	}

	var index *legacyPathIndex
	lazyIndex := func() *legacyPathIndex {
		if index == nil {
			index = newLegacyPathIndex(registry)
		}
		return index
	}
	serverSegment := serverPathSegment(registry.AgentFieldServerID)

	nodeIDs := make([]string, 0, len(registry.AgentNodes))
	for nodeID := range registry.AgentNodes {
		nodeIDs = append(nodeIDs, nodeID)
	}
	sort.Strings(nodeIDs)

	for _, nodeID := range nodeIDs {
		agent := registry.AgentNodes[nodeID]

		if !identityMatchesPath(registry.MasterSeed, agent.DerivationPath, agent.DID) {
			path, found := lazyIndex().agentPath(agent.DID)
			if !found {
				reason := fmt.Sprintf("agent %s DID %s cannot be re-derived from the stored master seed", nodeID, agent.DID)
				quarantined[agent.DID] = reason
				for _, reasoner := range agent.Reasoners {
					quarantined[reasoner.DID] = reason
				}
				for _, skill := range agent.Skills {
					quarantined[skill.DID] = reason
				}
				logger.Logger.Error().Str("agent_node_id", nodeID).Str("did", agent.DID).
					Msg("DID identity is inconsistent with persisted key material; refusing to use it")
				continue
			}
			logger.Logger.Warn().Str("agent_node_id", nodeID).Str("stored_path", agent.DerivationPath).
				Str("recovered_path", path).Msg("Recovered DID derivation path for agent")
			agent.DerivationPath = path
			repaired++
		}
		if normalizePublicKeyJWK(&agent.PublicKeyJWK, agent.DID) {
			repaired++
		}

		agentIndex, hasAgentIndex := agentIndexFromPath(agent.DerivationPath, serverSegment)

		for id, reasoner := range agent.Reasoners {
			fixed, ok := verifyComponentPath(registry.MasterSeed, reasoner.DID, reasoner.DerivationPath, lazyIndex, agentIndex, hasAgentIndex)
			if !ok {
				quarantined[reasoner.DID] = fmt.Sprintf("reasoner %s DID %s cannot be re-derived from the stored master seed", id, reasoner.DID)
				logger.Logger.Error().Str("agent_node_id", nodeID).Str("reasoner", id).Str("did", reasoner.DID).
					Msg("Component DID identity is inconsistent with persisted key material; refusing to use it")
				continue
			}
			changed := fixed != reasoner.DerivationPath
			reasoner.DerivationPath = fixed
			if normalizePublicKeyJWK(&reasoner.PublicKeyJWK, reasoner.DID) {
				changed = true
			}
			if changed {
				agent.Reasoners[id] = reasoner
				repaired++
			}
		}
		for id, skill := range agent.Skills {
			fixed, ok := verifyComponentPath(registry.MasterSeed, skill.DID, skill.DerivationPath, lazyIndex, agentIndex, hasAgentIndex)
			if !ok {
				quarantined[skill.DID] = fmt.Sprintf("skill %s DID %s cannot be re-derived from the stored master seed", id, skill.DID)
				logger.Logger.Error().Str("agent_node_id", nodeID).Str("skill", id).Str("did", skill.DID).
					Msg("Component DID identity is inconsistent with persisted key material; refusing to use it")
				continue
			}
			changed := fixed != skill.DerivationPath
			skill.DerivationPath = fixed
			if normalizePublicKeyJWK(&skill.PublicKeyJWK, skill.DID) {
				changed = true
			}
			if changed {
				agent.Skills[id] = skill
				repaired++
			}
		}

		registry.AgentNodes[nodeID] = agent
	}

	return repaired, quarantined, nil
}

func verifyComponentPath(masterSeed []byte, did, storedPath string, lazyIndex func() *legacyPathIndex, agentIndex int, hasAgentIndex bool) (string, bool) {
	if identityMatchesPath(masterSeed, storedPath, did) {
		return storedPath, true
	}
	return lazyIndex().componentPath(did, agentIndex, hasAgentIndex)
}

// normalizePublicKeyJWK sets jwk to the public key encoded in did when it is
// missing or disagrees with the DID. It reports whether it changed anything.
func normalizePublicKeyJWK(jwk *json.RawMessage, did string) bool {
	expected, err := publicKeyJWKFromDIDKey(did)
	if err != nil {
		return false
	}
	if len(*jwk) > 0 {
		var stored, want struct {
			X string `json:"x"`
		}
		if json.Unmarshal(*jwk, &stored) == nil && json.Unmarshal([]byte(expected), &want) == nil && stored.X == want.X && stored.X != "" {
			return false
		}
	}
	*jwk = json.RawMessage(expected)
	return true
}
