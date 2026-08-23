package triggers

import (
	"embed"
	"encoding/json"
	"path/filepath"
	"testing"
)

// fixtureFS embeds the captured provider payloads into the package so
// LoadFixture works from any caller's working directory — including when the
// SDK is consumed from the module cache, where a relative "testdata" path
// would not resolve.
//
// The files are byte-identical copies of
// sdk/python/agentfield/fixtures/triggers/ so the Go, Python, and TypeScript
// SDKs prove behavioural parity against the same JSON.
//
//go:embed testdata/*.json
var fixtureFS embed.FS

// LoadFixture reads a captured provider payload from the embedded fixture
// library and returns it as a decoded map.
//
// The name may be given with or without the .json suffix, so both "stripe"
// and "stripe.json" resolve to the same fixture.
//
//	fixture := triggers.LoadFixture(t, "stripe")
//	result, err := triggers.SimulateEvent(t, handlePayment, triggers.SimulateEventOpts{
//	    Source: "stripe",
//	    Body:   fixture,
//	})
//
// LoadFixture calls t.Fatalf when the fixture is missing or malformed, so a
// typo surfaces as a clear test failure rather than a nil-map panic later.
func LoadFixture(t *testing.T, name string) map[string]any {
	t.Helper()

	filename := name
	if filepath.Ext(filename) != ".json" {
		filename += ".json"
	}
	// embed.FS always uses forward slashes, regardless of host OS.
	path := "testdata/" + filename

	raw, err := fixtureFS.ReadFile(path)
	if err != nil {
		t.Fatalf("LoadFixture(%q): %v (available: %v)", name, err, FixtureNames())
	}

	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("LoadFixture(%q): fixture is not a JSON object: %v", name, err)
	}
	return payload
}

// RawFixture returns the undecoded bytes of a fixture. Useful for tests that
// need to assert on the exact JSON (for example, verifying byte-for-byte
// parity with another SDK's copy).
func RawFixture(t *testing.T, name string) []byte {
	t.Helper()

	filename := name
	if filepath.Ext(filename) != ".json" {
		filename += ".json"
	}
	raw, err := fixtureFS.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("RawFixture(%q): %v", name, err)
	}
	return raw
}

// FixtureNames returns the source names of every fixture in the library, for
// table-driven tests that want to exercise all providers.
func FixtureNames() []string {
	return []string{
		"stripe",
		"github",
		"slack",
		"cron",
		"generic_hmac",
		"generic_bearer",
	}
}
