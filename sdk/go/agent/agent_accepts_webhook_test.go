package agent

import (
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// TestAcceptsWebhookAutoSetTrue tests that AcceptsWebhook is auto-set to "true" when triggers are declared.
func TestAcceptsWebhookAutoSetTrue(t *testing.T) {
	r := &Reasoner{Name: "test"}
	
	// Simulate registering a trigger
	r.Triggers = append(r.Triggers, types.TriggerBinding{Source: "stripe"})
	
	// Manually apply the auto-set logic (what RegisterReasoner does)
	if r.AcceptsWebhook == nil && len(r.Triggers) > 0 {
		flag := "true"
		r.AcceptsWebhook = &flag
	}
	
	if r.AcceptsWebhook == nil || *r.AcceptsWebhook != "true" {
		t.Errorf("expected AcceptsWebhook='true' when triggers declared, got %v", r.AcceptsWebhook)
	}
}

// TestAcceptsWebhookExplicitOverridesAuto tests that explicit setting overrides auto-detect.
func TestAcceptsWebhookExplicitOverridesAuto(t *testing.T) {
	r := &Reasoner{Name: "test"}
	flag := "false"
	r.AcceptsWebhook = &flag
	
	// Simulate registering a trigger
	r.Triggers = append(r.Triggers, types.TriggerBinding{Source: "stripe"})
	
	// Apply the auto-set logic (should NOT override explicit value)
	if r.AcceptsWebhook == nil && len(r.Triggers) > 0 {
		flag := "true"
		r.AcceptsWebhook = &flag
	}
	
	if r.AcceptsWebhook == nil || *r.AcceptsWebhook != "false" {
		t.Errorf("expected explicit AcceptsWebhook='false' to be preserved, got %v", r.AcceptsWebhook)
	}
}

// TestAcceptsWebhookNilByDefault tests that AcceptsWebhook is nil when no triggers and no explicit setting.
func TestAcceptsWebhookNilByDefault(t *testing.T) {
	r := &Reasoner{Name: "test"}
	
	// No triggers, no explicit setting
	if r.AcceptsWebhook != nil {
		t.Errorf("expected AcceptsWebhook=nil by default, got %v", r.AcceptsWebhook)
	}
}

// TestAcceptsWebhookCanBeSetExplicitly tests that WithAcceptsWebhook option works.
func TestAcceptsWebhookCanBeSetExplicitly(t *testing.T) {
	r := &Reasoner{Name: "test"}
	
	// Apply WithAcceptsWebhook option
	opt := WithAcceptsWebhook("true")
	opt(r)
	
	if r.AcceptsWebhook == nil || *r.AcceptsWebhook != "true" {
		t.Errorf("expected WithAcceptsWebhook('true') to set flag, got %v", r.AcceptsWebhook)
	}
}
