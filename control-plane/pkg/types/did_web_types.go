package types

import (
	"encoding/json"
	"time"
)

// DIDMethod represents the DID method type.
type DIDMethod string

const (
	DIDMethodKey DIDMethod = "did:key"
	DIDMethodWeb DIDMethod = "did:web"
)

// DIDWebDocument represents a W3C DID Document for did:web method.
// See: https://www.w3.org/TR/did-core/
type DIDWebDocument struct {
	Context            []string             `json:"@context"`
	ID                 string               `json:"id"`
	Controller         string               `json:"controller,omitempty"`
	VerificationMethod []VerificationMethod `json:"verificationMethod"`
	Authentication     []string             `json:"authentication"`
	AssertionMethod    []string             `json:"assertionMethod,omitempty"`
	KeyAgreement       []string             `json:"keyAgreement,omitempty"`
	Service            []DIDService         `json:"service,omitempty"`
}

// VerificationMethod represents a verification method in a DID Document.
type VerificationMethod struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Controller   string          `json:"controller"`
	PublicKeyJwk json.RawMessage `json:"publicKeyJwk"`
}

// DIDService represents a service endpoint in a DID Document.
type DIDService struct {
	ID              string `json:"id"`
	Type            string `json:"type"`
	ServiceEndpoint string `json:"serviceEndpoint"`
}

// DIDDocumentRecord represents the database record for a DID document.
type DIDDocumentRecord struct {
	DID          string          `json:"did" db:"did"`
	AgentID      string          `json:"agent_id" db:"agent_id"`
	DIDDocument  json.RawMessage `json:"did_document" db:"did_document"`
	PublicKeyJWK string          `json:"public_key_jwk" db:"public_key_jwk"`
	RevokedAt    *time.Time      `json:"revoked_at,omitempty" db:"revoked_at"`
	CreatedAt    time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at" db:"updated_at"`
}

// IsRevoked returns true if the DID has been revoked.
func (d *DIDDocumentRecord) IsRevoked() bool {
	return d.RevokedAt != nil
}

// DIDResolutionResult represents the result of resolving a DID.
type DIDResolutionResult struct {
	DIDDocument      *DIDWebDocument `json:"didDocument,omitempty"`
	DIDResolutionMetadata DIDResolutionMetadata `json:"didResolutionMetadata"`
	DIDDocumentMetadata   DIDDocumentMetadata   `json:"didDocumentMetadata"`
}

// DIDResolutionMetadata contains metadata about the resolution process.
type DIDResolutionMetadata struct {
	ContentType string `json:"contentType,omitempty"`
	Error       string `json:"error,omitempty"`
}

// DIDDocumentMetadata contains metadata about the DID document.
type DIDDocumentMetadata struct {
	Created     string `json:"created,omitempty"`
	Updated     string `json:"updated,omitempty"`
	Deactivated bool   `json:"deactivated,omitempty"`
}

// DIDWebConfig holds configuration for did:web generation.
type DIDWebConfig struct {
	Domain   string `json:"domain" yaml:"domain" mapstructure:"domain"`
	BasePath string `json:"base_path" yaml:"base_path" mapstructure:"base_path"`
}

// GenerateDIDWeb creates a did:web identifier for an agent.
// Format: did:web:{domain}:agents:{agentID}
func GenerateDIDWeb(domain, agentID string) string {
	return "did:web:" + domain + ":agents:" + agentID
}

// GenerateDIDWebVerificationMethodID creates the verification method ID for a did:web.
// Format: {did}#key-1
func GenerateDIDWebVerificationMethodID(did string) string {
	return did + "#key-1"
}

// NewDIDWebDocument creates a new DID Document for did:web method.
func NewDIDWebDocument(did string, publicKeyJWK json.RawMessage) *DIDWebDocument {
	verificationMethodID := GenerateDIDWebVerificationMethodID(did)

	return &DIDWebDocument{
		Context: []string{
			"https://www.w3.org/ns/did/v1",
			"https://w3id.org/security/suites/jws-2020/v1",
		},
		ID: did,
		VerificationMethod: []VerificationMethod{
			{
				ID:           verificationMethodID,
				Type:         "JsonWebKey2020",
				Controller:   did,
				PublicKeyJwk: publicKeyJWK,
			},
		},
		Authentication:  []string{verificationMethodID},
		AssertionMethod: []string{verificationMethodID},
	}
}

// DIDWebConstants holds constants for did:web implementation.
var DIDWebConstants = struct {
	VerificationMethodType string
	Context                []string
}{
	VerificationMethodType: "JsonWebKey2020",
	Context: []string{
		"https://www.w3.org/ns/did/v1",
		"https://w3id.org/security/suites/jws-2020/v1",
	},
}
