package handlers

import (
	"testing"
)

func TestSetRedactPayloads(t *testing.T) {
	// Save original and restore after test
	original := defaultRedactPayloads
	defer func() { defaultRedactPayloads = original }()

	// Default should be true (safe)
	if !defaultRedactPayloads {
		t.Fatal("expected defaultRedactPayloads to be true initially")
	}

	// SetRedactPayloads(false) should disable redaction
	SetRedactPayloads(false)
	if defaultRedactPayloads {
		t.Fatal("expected defaultRedactPayloads to be false after SetRedactPayloads(false)")
	}

	// SetRedactPayloads(true) should re-enable redaction
	SetRedactPayloads(true)
	if !defaultRedactPayloads {
		t.Fatal("expected defaultRedactPayloads to be true after SetRedactPayloads(true)")
	}
}

func TestNewExecutionControllerInheritsRedactPayloads(t *testing.T) {
	original := defaultRedactPayloads
	defer func() { defaultRedactPayloads = original }()

	store := newTestExecutionStorage(nil)

	// When defaultRedactPayloads is true
	SetRedactPayloads(true)
	ctrl := newExecutionController(store, nil, nil, 0, "")
	if !ctrl.redactPayloads {
		t.Fatal("expected controller to inherit redactPayloads=true")
	}

	// When defaultRedactPayloads is false
	SetRedactPayloads(false)
	ctrl = newExecutionController(store, nil, nil, 0, "")
	if ctrl.redactPayloads {
		t.Fatal("expected controller to inherit redactPayloads=false")
	}
}
