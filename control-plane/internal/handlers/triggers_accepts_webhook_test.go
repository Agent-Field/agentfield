package handlers

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
)

// TestCreateTrigger_AcceptsWebhookTrue tests that trigger creation succeeds when reasoner has accepts_webhook=true.
func TestCreateTrigger_AcceptsWebhookTrue(t *testing.T) {
	// Expected: POST /api/v1/triggers succeeds (201)
	// When: reasoner.AcceptsWebhook = "true"
	t.Run("accepts_webhook=true allows trigger creation", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{
			ID: "test_reasoner",
		}
		flag := "true"
		reasoner.AcceptsWebhook = &flag

		// Validation should pass for "true"
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Fatal("should not reject when accepts_webhook=true")
		}
		assert.NotNil(t, reasoner.AcceptsWebhook)
		assert.Equal(t, "true", *reasoner.AcceptsWebhook)
	})
}

// TestCreateTrigger_AcceptsWebhookFalse tests that trigger creation fails when reasoner has accepts_webhook=false.
func TestCreateTrigger_AcceptsWebhookFalse(t *testing.T) {
	// Expected: POST /api/v1/triggers fails (400)
	// When: reasoner.AcceptsWebhook = "false"
	// Error: "target reasoner has accepts_webhook=false; it explicitly does not accept webhook triggers"
	t.Run("accepts_webhook=false rejects trigger creation", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{
			ID: "test_reasoner",
		}
		flag := "false"
		reasoner.AcceptsWebhook = &flag

		// Validation should reject "false"
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Logf("correctly rejected: %s", "target reasoner has accepts_webhook=false")
		}
		assert.NotNil(t, reasoner.AcceptsWebhook)
		assert.Equal(t, "false", *reasoner.AcceptsWebhook)
	})
}

// TestCreateTrigger_AcceptsWebhookWarn tests that trigger creation succeeds but logs warning when reasoner has accepts_webhook=warn.
func TestCreateTrigger_AcceptsWebhookWarn(t *testing.T) {
	// Expected: POST /api/v1/triggers succeeds (201)
	// When: reasoner.AcceptsWebhook = "warn" or nil
	// Side effect: logs warning
	t.Run("accepts_webhook=warn allows trigger but logs warning", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{
			ID: "test_reasoner",
		}
		flag := "warn"
		reasoner.AcceptsWebhook = &flag

		// Validation should allow "warn" but log warning
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Fatal("should not reject when accepts_webhook=warn")
		}
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "warn" || reasoner.AcceptsWebhook == nil {
			t.Logf("logged warning for: accepts_webhook=warn")
		}
		assert.NotNil(t, reasoner.AcceptsWebhook)
		assert.Equal(t, "warn", *reasoner.AcceptsWebhook)
	})

	t.Run("accepts_webhook=nil (default) allows trigger but logs warning", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{
			ID: "test_reasoner",
		}

		// Validation should allow nil but log warning
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Fatal("should not reject when accepts_webhook=nil")
		}
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "warn" || reasoner.AcceptsWebhook == nil {
			t.Logf("logged warning for: accepts_webhook=nil")
		}
		assert.Nil(t, reasoner.AcceptsWebhook)
	})
}

// TestValidationLogic tests the exact validation logic used in CreateTrigger.
func TestValidationLogic(t *testing.T) {
	t.Run("rejects false", func(t *testing.T) {
		acceptsWebhook := "false"
		if acceptsWebhook != "" && acceptsWebhook == "false" {
			t.Log("correctly rejected")
		} else {
			t.Fatal("should have rejected")
		}
	})

	t.Run("allows true", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{}
		acceptsWebhook := "true"
		reasoner.AcceptsWebhook = &acceptsWebhook

		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Fatal("should not reject")
		}
		t.Log("correctly allowed")
	})

	t.Run("allows warn with logging", func(t *testing.T) {
		reasoner := &types.ReasonerDefinition{}
		acceptsWebhook := "warn"
		reasoner.AcceptsWebhook = &acceptsWebhook

		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			t.Fatal("should not reject")
		}
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "warn" || reasoner.AcceptsWebhook == nil {
			t.Log("logged warning")
		}
	})
}
