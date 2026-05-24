package services

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// TriggerEventInput is the data the dispatcher hands the VC service to mint a
// trigger event credential. Everything except Verification is required.
type TriggerEventInput struct {
	TriggerID   string
	SourceName  string
	EventType   string
	EventID     string                  // provider's idempotency key (or our own UUID for cron)
	Payload     []byte                  // raw inbound body — used to compute the payload hash
	Verification types.VCTriggerVerification
	ReceivedAt  time.Time
}

// GenerateTriggerEventVC mints a CP-rooted credential attesting that an
// external signed payload arrived at a trigger and was about to be dispatched.
// Returns nil (with no error) when the DID system is disabled — callers should
// proceed with dispatch in that case so trigger functionality still works
// without DID enabled.
//
// The credential is stored in the execution_vcs table with kind='trigger_event'
// so the existing chain walker can use it as a chain root for the resulting
// execution VC produced by the dispatched reasoner.
func (s *VCService) GenerateTriggerEventVC(ctx context.Context, in TriggerEventInput) (*types.ExecutionVC, error) {
	if s == nil || s.config == nil || !s.config.Enabled {
		return nil, nil
	}
	if !s.config.VCRequirements.RequireVCForExecution {
		// Same policy as GenerateExecutionVC — opt-in via DID config.
		return nil, nil
	}

	// CP issues this credential under its own root DID. If we can't resolve
	// the CP server, we can't sign — return nil-and-error so the caller logs
	// it but still dispatches (caller's responsibility to swallow nil-VC).
	cpServerID, err := s.didService.GetAgentFieldServerID()
	if err != nil {
		return nil, fmt.Errorf("trigger VC: get af server ID: %w", err)
	}
	registry, err := s.didService.GetRegistry(cpServerID)
	if err != nil {
		return nil, fmt.Errorf("trigger VC: get af server registry: %w", err)
	}
	issuerDID := registry.RootDID
	if issuerDID == "" {
		return nil, nil
	}
	issuerIdentity, err := s.didService.ResolveDID(issuerDID)
	if err != nil {
		return nil, fmt.Errorf("trigger VC: resolve issuer DID: %w", err)
	}

	receivedAt := in.ReceivedAt
	if receivedAt.IsZero() {
		receivedAt = time.Now().UTC()
	}

	// Hash the raw payload — the caller should pass the exact bytes the
	// signature was computed against so the hash is reproducible at audit.
	hash := sha256.Sum256(in.Payload)
	payloadHash := base64.RawURLEncoding.EncodeToString(hash[:])

	subject := types.TriggerEventVCSubject{
		TriggerID:   in.TriggerID,
		SourceName:  in.SourceName,
		EventType:   in.EventType,
		EventID:     in.EventID,
		PayloadHash: payloadHash,
		ReceivedAt:  receivedAt.UTC().Format(time.RFC3339),
		Verification: in.Verification,
	}

	vcID := s.generateVCID()
	vcDoc := map[string]any{
		"@context": []string{
			"https://www.w3.org/2018/credentials/v1",
			"https://agentfield.example.com/contexts/trigger-event/v1",
		},
		"type": []string{
			"VerifiableCredential",
			"AgentFieldTriggerEventCredential",
		},
		"id":                fmt.Sprintf("urn:agentfield:trigger-event-vc:%s", vcID),
		"issuer":            issuerDID,
		"issuanceDate":      time.Now().UTC().Format(time.RFC3339),
		"credentialSubject": subject,
	}

	canonical, err := json.Marshal(vcDoc)
	if err != nil {
		return nil, fmt.Errorf("trigger VC: marshal canonical: %w", err)
	}
	signature, err := s.signCanonical(canonical, issuerIdentity)
	if err != nil {
		return nil, fmt.Errorf("trigger VC: sign: %w", err)
	}
	vcDoc["proof"] = types.VCProof{
		Type:               "Ed25519Signature2020",
		Created:            time.Now().UTC().Format(time.RFC3339),
		VerificationMethod: fmt.Sprintf("%s#key-1", issuerDID),
		ProofPurpose:       "assertionMethod",
		ProofValue:         signature,
	}
	docBytes, err := json.Marshal(vcDoc)
	if err != nil {
		return nil, fmt.Errorf("trigger VC: marshal final: %w", err)
	}

	triggerID := in.TriggerID
	sourceName := in.SourceName
	eventType := in.EventType
	eventID := in.EventID
	vc := &types.ExecutionVC{
		VCID:         vcID,
		ExecutionID:  in.EventID,         // event ID populates the execution_id slot for uniqueness
		WorkflowID:   "",
		SessionID:    "",
		IssuerDID:    issuerDID,
		TargetDID:    "",
		CallerDID:    issuerDID,
		VCDocument:   json.RawMessage(docBytes),
		Signature:    signature,
		StorageURI:   "",
		DocumentSize: int64(len(docBytes)),
		InputHash:    payloadHash,
		OutputHash:   "",
		Status:       string(types.ExecutionStatusSucceeded),
		CreatedAt:    receivedAt,
		Kind:         types.ExecutionVCKindTriggerEvent,
		TriggerID:    &triggerID,
		SourceName:   &sourceName,
		EventType:    &eventType,
		EventID:      &eventID,
	}

	if s.ShouldPersistExecutionVC() {
		if err := s.vcStorage.StoreExecutionVC(ctx, vc); err != nil {
			return nil, fmt.Errorf("trigger VC: persist: %w", err)
		}
	} else {
		logger.Logger.Debug().Str("trigger_id", in.TriggerID).Msg("trigger event VC persistence skipped by policy")
	}

	return vc, nil
}

// signCanonical signs canonical bytes using the supplied identity's Ed25519
// private key. Shared by trigger-event and (eventually) other non-execution
// credentials so signing logic doesn't drift across files.
func (s *VCService) signCanonical(canonical []byte, identity *types.DIDIdentity) (string, error) {
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(identity.PrivateKeyJWK), &jwk); err != nil {
		return "", fmt.Errorf("parse issuer JWK: %w", err)
	}
	dValue, ok := jwk["d"].(string)
	if !ok {
		return "", fmt.Errorf("issuer JWK missing 'd'")
	}
	seed, err := base64.RawURLEncoding.DecodeString(dValue)
	if err != nil {
		return "", fmt.Errorf("decode issuer seed: %w", err)
	}
	if len(seed) != ed25519.SeedSize {
		return "", fmt.Errorf("invalid issuer seed length: got %d want %d", len(seed), ed25519.SeedSize)
	}
	priv := ed25519.NewKeyFromSeed(seed)
	sig := ed25519.Sign(priv, canonical)
	return base64.RawURLEncoding.EncodeToString(sig), nil
}
