package types

// TriggerEventVCSubject is the credentialSubject embedded inside the VCDocument
// of a trigger-event-kind ExecutionVC. It records what the control plane
// observed when an external signed payload arrived: which Source plugin
// processed it, what the provider's event identifier was, and whether the
// signature verification step passed. The dispatched reasoner's execution VC
// chains back to this credential via parent_vc_id, so af verify can walk a
// chain from a reasoner all the way back to "Stripe really sent this."
type TriggerEventVCSubject struct {
	TriggerID    string `json:"trigger_id"`
	SourceName   string `json:"source_name"`
	EventType    string `json:"event_type,omitempty"`
	EventID      string `json:"event_id,omitempty"`
	PayloadHash  string `json:"payload_hash"`
	ReceivedAt   string `json:"received_at"`
	Verification VCTriggerVerification `json:"verification"`
}

// VCTriggerVerification captures the Source plugin's signature/auth check
// outcome at ingest time. Recorded inside the trigger event VC so an auditor
// can see post-hoc that the inbound payload's signature was checked, by which
// algorithm, and whether it passed — without trusting the control plane's logs.
type VCTriggerVerification struct {
	Passed     bool   `json:"passed"`
	Algorithm  string `json:"algorithm,omitempty"`  // e.g. "stripe-v1", "github-hmac-sha256"
	HeaderName string `json:"header_name,omitempty"` // header the signature came from
	Detail     string `json:"detail,omitempty"`      // free-form note ("ts within 5m skew", etc.)
}
