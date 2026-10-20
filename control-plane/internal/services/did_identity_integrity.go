package services

import (
	"crypto/ed25519"
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

// SelectAgentFieldServerID picks which persisted DID registry this control plane
// owns, so that relocating AGENTFIELD_HOME does not silently mint a new identity.
//
//  1. An explicitly configured server ID is used when a registry with that ID
//     is stored, or when no registries exist yet. A configured ID that matches
//     none of the stored registries fails, so a typo cannot create a new
//     identity.
//  2. A registry already stored under the home-path ID is used.
//  3. With no registries at all, the home-path ID names the new registry. It is
//     deterministic, so replicas that share a home path and start together
//     converge on one registry.
//  4. With exactly one registry, it is adopted even if the home path changed.
//  5. Otherwise the choice is ambiguous and startup fails closed.
func SelectAgentFieldServerID(registry *DIDRegistry, configuredID, legacyID string) (string, error) {
	configured := strings.TrimSpace(configuredID)
	if registry == nil {
		if configured != "" {
			return configured, nil
		}
		return legacyID, nil
	}

	existing, err := registry.ListRegistries()
	if err != nil {
		return "", fmt.Errorf("failed to list DID registries: %w", err)
	}
	ids := make([]string, 0, len(existing))
	for _, reg := range existing {
		ids = append(ids, reg.AgentFieldServerID)
	}
	sort.Strings(ids)
	has := func(id string) bool {
		for _, candidate := range ids {
			if candidate == id {
				return true
			}
		}
		return false
	}

	if configured != "" {
		if len(ids) == 0 || has(configured) {
			return configured, nil
		}
		return "", fmt.Errorf(
			"AGENTFIELD_DID_SERVER_ID is set to %q but no stored DID registry has that ID (stored: %s); "+
				"refusing to create a new server identity. Set it to one of the stored IDs or unset it",
			configured, strings.Join(ids, ", "))
	}
	if has(legacyID) {
		return legacyID, nil
	}

	switch len(ids) {
	case 0:
		return legacyID, nil
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
	nextAgentIndex int
	componentsByIx map[int]map[string]string
	nextComponent  map[int]int
	derivations    int // Candidate derivations, excluding direct integrity checks.
}

// newLegacyPathIndex bounds the agent-index search by every agent row stored
// in the database, not just this server's: older versions loaded all agents
// into every registry, so their agent indices counted agents of other servers.
func newLegacyPathIndex(registry *types.DIDRegistry, storedAgentRows int) *legacyPathIndex {
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
		maxAgentIndex:  maxInt(64, 2*maxInt(storedAgentRows, len(registry.AgentNodes))+16),
		maxComponentIx: maxInt(32, 2*maxComponents+8),
		componentsByIx: make(map[int]map[string]string),
		nextComponent:  make(map[int]int),
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (ix *legacyPathIndex) derive(path string, table map[string]string) {
	ix.derivations++
	if did, err := didForPath(ix.masterSeed, path); err == nil {
		table[did] = path
	}
}

func (ix *legacyPathIndex) agentPath(did string) (string, bool) {
	if ix.agents == nil {
		ix.agents = make(map[string]string)
	}
	if path, ok := ix.agents[did]; ok {
		return path, true
	}
	for ix.nextAgentIndex <= ix.maxAgentIndex {
		path := agentDerivationPath(ix.serverSegment, ix.nextAgentIndex)
		ix.nextAgentIndex++
		ix.derive(path, ix.agents)
		if path, ok := ix.agents[did]; ok {
			return path, true
		}
	}
	return "", false
}

// componentUnderAgent expands one candidate at a time and caches all work.
// Interleave branches so a low-index skill does not require deriving the
// entire reasoner branch first. The inclusive bounds are unchanged.
func (ix *legacyPathIndex) componentUnderAgent(did string, agentIndex int) (string, bool) {
	table, ok := ix.componentsByIx[agentIndex]
	if !ok {
		table = make(map[string]string)
		ix.componentsByIx[agentIndex] = table
	}
	if path, ok := table[did]; ok {
		return path, true
	}
	agentPath := agentDerivationPath(ix.serverSegment, agentIndex)
	for ix.nextComponent[agentIndex] < 2*(ix.maxComponentIx+1) {
		n := ix.nextComponent[agentIndex]
		ix.nextComponent[agentIndex]++
		ix.derive(componentDerivationPath(agentPath, n%2, n/2), table)
		if path, ok := table[did]; ok {
			return path, true
		}
	}
	return "", false
}

// componentPath looks under the owning agent first, then under every other
// agent index, because older partial registrations picked the agent index from
// unordered map iteration.
func (ix *legacyPathIndex) componentPath(did string, preferredAgentIndex int, hasPreferred bool) (string, bool) {
	if hasPreferred {
		if path, ok := ix.componentUnderAgent(did, preferredAgentIndex); ok {
			return path, true
		}
	}
	for i := 0; i <= ix.maxAgentIndex; i++ {
		if hasPreferred && i == preferredAgentIndex {
			continue
		}
		if path, ok := ix.componentUnderAgent(did, i); ok {
			return path, true
		}
	}
	return "", false
}

// verifyRegistryIdentities checks that every identity in the registry
// re-derives to its DID. Stale placeholder paths are replaced with the
// recovered real path. Identities that cannot be recovered are returned as
// quarantined, keyed by DID, and must never be given keys or resolved.
//
// An unrecoverable component of an agent that did verify is removed from the
// registry and returned in retire (agent DID to component DIDs), so the agent
// keeps working and the component gets a fresh identity when it registers
// again. An unrecoverable agent stays quarantined until an operator revokes it.
func verifyRegistryIdentities(registry *types.DIDRegistry, storedAgentRows int) (repaired int, quarantined map[string]string, retire map[string][]string, rootErr error) {
	quarantined = make(map[string]string)
	retire = make(map[string][]string)

	if !identityMatchesPath(registry.MasterSeed, rootDerivationPath, registry.RootDID) {
		rootErr = fmt.Errorf("root DID %s for af server %s does not match its master seed (wrong encryption passphrase or corrupted seed); see docs/ENVIRONMENT_VARIABLES.md, Lost-root recovery",
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
		return 0, quarantined, retire, rootErr
	}

	var index *legacyPathIndex
	lazyIndex := func() *legacyPathIndex {
		if index == nil {
			index = newLegacyPathIndex(registry, storedAgentRows)
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
					Msg("Agent DID cannot be re-derived from persisted key material; refusing to issue its keys. " +
						"To give the agent a new identity, set status = 'revoked' on its agent_dids row and restart")
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
					Msg("Component DID cannot be re-derived from persisted key material; retiring it so the component gets a new identity")
				delete(agent.Reasoners, id)
				retire[agent.DID] = append(retire[agent.DID], reasoner.DID)
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
					Msg("Component DID cannot be re-derived from persisted key material; retiring it so the component gets a new identity")
				delete(agent.Skills, id)
				retire[agent.DID] = append(retire[agent.DID], skill.DID)
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

	return repaired, quarantined, retire, nil
}

// usedIdentities is every DID and derivation path that must never be issued
// again: identities in memory, quarantined ones, and every agent and component
// row in storage (any server, any status, including retired and revoked).
type usedIdentities struct {
	dids  map[string]bool
	paths map[string]bool
}

func newUsedIdentities() *usedIdentities {
	return &usedIdentities{dids: make(map[string]bool), paths: make(map[string]bool)}
}

func (u *usedIdentities) add(did, path string) {
	if did != "" {
		u.dids[did] = true
	}
	if path != "" {
		u.paths[path] = true
	}
}

func (u *usedIdentities) addRegistry(registry *types.DIDRegistry) {
	u.add(registry.RootDID, "")
	for _, agent := range registry.AgentNodes {
		u.add(agent.DID, agent.DerivationPath)
		for _, reasoner := range agent.Reasoners {
			u.add(reasoner.DID, reasoner.DerivationPath)
		}
		for _, skill := range agent.Skills {
			u.add(skill.DID, skill.DerivationPath)
		}
	}
}

// taken reports whether path, or the DID it derives, was ever issued.
func (u *usedIdentities) taken(masterSeed []byte, path string) (bool, string, error) {
	did, err := didForPath(masterSeed, path)
	if err != nil {
		return true, "", err
	}
	return u.paths[path] || u.dids[did], did, nil
}

// nextAgentPathFor returns the lowest agent path above every agent index used
// under this server whose path and DID were never issued.
func nextAgentPathFor(registry *types.DIDRegistry, used *usedIdentities) (string, error) {
	segment := serverPathSegment(registry.AgentFieldServerID)
	next := 0
	for path := range used.paths {
		if index, ok := agentIndexFromPath(path, segment); ok && index+1 > next {
			next = index + 1
		}
	}
	for ; ; next++ {
		path := agentDerivationPath(segment, next)
		taken, did, err := used.taken(registry.MasterSeed, path)
		if err != nil {
			return "", err
		}
		if !taken {
			used.add(did, path)
			return path, nil
		}
	}
}

// componentPathAllocator hands out component paths under one agent path.
// Indices start above every index ever used under that agent path, by any
// agent, and skip anything whose path or DID was already issued.
type componentPathAllocator struct {
	masterSeed []byte
	agentPath  string
	used       *usedIdentities
	next       map[int]int
}

func newComponentPathAllocator(masterSeed []byte, agentPath string, used *usedIdentities) *componentPathAllocator {
	alloc := &componentPathAllocator{masterSeed: masterSeed, agentPath: agentPath, used: used, next: map[int]int{0: 0, 1: 0}}
	for path := range used.paths {
		for branch := 0; branch <= 1; branch++ {
			if index, ok := componentIndexFromPath(path, agentPath, branch); ok && index+1 > alloc.next[branch] {
				alloc.next[branch] = index + 1
			}
		}
	}
	return alloc
}

func (a *componentPathAllocator) allocate(componentType string) (string, error) {
	branch := componentBranch(componentType)
	for {
		index := a.next[branch]
		a.next[branch] = index + 1
		path := componentDerivationPath(a.agentPath, branch, index)
		taken, did, err := a.used.taken(a.masterSeed, path)
		if err != nil {
			return "", err
		}
		if !taken {
			a.used.add(did, path)
			return path, nil
		}
	}
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
