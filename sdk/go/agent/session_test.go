package agent

import "testing"

func TestAgentRegisterSessionStoresExplicitDefinition(t *testing.T) {
	a, err := New(Config{NodeID: "support", Version: "v1"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := a.RegisterSession(
		"voice",
		"openai",
		"webrtc",
		WithSessionModel("gpt-realtime-2"),
		WithSessionVoice("marin"),
		WithSessionTools("support.resolve_voice_turn"),
		WithSessionTags("voice", "pii"),
	); err != nil {
		t.Fatalf("RegisterSession returned error: %v", err)
	}

	sessions := a.SessionDefinitions()
	if len(sessions) != 1 {
		t.Fatalf("len(sessions) = %d, want 1", len(sessions))
	}
	if sessions[0].Provider != "openai" || sessions[0].Transport != "webrtc" {
		t.Fatalf("session provider/transport = %s/%s", sessions[0].Provider, sessions[0].Transport)
	}
	if sessions[0].Tools[0] != "support.resolve_voice_turn" {
		t.Fatalf("tool target = %q", sessions[0].Tools[0])
	}
	if len(sessions[0].Tags) != 2 || sessions[0].Tags[0] != "voice" || sessions[0].ProposedTags[1] != "pii" {
		t.Fatalf("session tags = %#v proposed=%#v", sessions[0].Tags, sessions[0].ProposedTags)
	}
}

func TestAgentRegisterSessionRejectsInvalidTransport(t *testing.T) {
	a, err := New(Config{NodeID: "support", Version: "v1"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := a.RegisterSession("voice", "openrouter", "webrtc"); err == nil {
		t.Fatal("expected invalid provider/transport error")
	}
}

func TestAgentSessionDefinitionsReturnsDefensiveSnapshots(t *testing.T) {
	a, err := New(Config{NodeID: "support", Version: "v1"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	metadata := map[string]any{
		"origin": "registry",
		"nested": map[string]any{
			"labels": []any{"original", map[string]any{"owner": "support"}},
		},
		"typedLabels": []string{"voice"},
	}
	if err := a.RegisterSession(
		"voice",
		"openai",
		"webrtc",
		WithSessionModalities("audio"),
		WithSessionTools("support.resolve_voice_turn"),
		WithSessionTags("voice"),
		WithSessionMetadata(metadata),
	); err != nil {
		t.Fatalf("RegisterSession returned error: %v", err)
	}
	registered := a.sessions["voice"]
	registered.ApprovedTags = []string{"approved"}
	a.sessions["voice"] = registered

	snapshot := a.SessionDefinitions()
	snapshot[0].Tools[0] = "mutated-tool"
	snapshot[0].Modalities[0] = "mutated-modality"
	snapshot[0].Tags[0] = "mutated-tag"
	snapshot[0].ProposedTags[0] = "mutated-proposed-tag"
	snapshot[0].ApprovedTags[0] = "mutated-approved-tag"
	snapshot[0].Metadata["origin"] = "mutated-origin"
	nested := snapshot[0].Metadata["nested"].(map[string]any)
	labels := nested["labels"].([]any)
	labels[0] = "mutated-label"
	labels[1].(map[string]any)["owner"] = "mutated-owner"
	snapshot[0].Metadata["typedLabels"].([]string)[0] = "mutated-typed-label"

	secondSnapshot := a.SessionDefinitions()
	session := secondSnapshot[0]
	if session.Tools[0] != "support.resolve_voice_turn" {
		t.Errorf("tools = %#v, want original values", session.Tools)
	}
	if session.Modalities[0] != "audio" {
		t.Errorf("modalities = %#v, want original values", session.Modalities)
	}
	if session.Tags[0] != "voice" || session.ProposedTags[0] != "voice" || session.ApprovedTags[0] != "approved" {
		t.Errorf("tags = %#v proposed=%#v approved=%#v, want original values", session.Tags, session.ProposedTags, session.ApprovedTags)
	}
	if session.Metadata["origin"] != "registry" {
		t.Errorf("metadata origin = %#v, want registry", session.Metadata["origin"])
	}
	gotNested := session.Metadata["nested"].(map[string]any)
	gotLabels := gotNested["labels"].([]any)
	if gotLabels[0] != "original" || gotLabels[1].(map[string]any)["owner"] != "support" {
		t.Errorf("nested metadata = %#v, want original values", gotNested)
	}
	if session.Metadata["typedLabels"].([]string)[0] != "voice" {
		t.Errorf("typed metadata slice = %#v, want original values", session.Metadata["typedLabels"])
	}
}

func TestAgentSessionDefinitionsPreservesOverlappingSliceLengths(t *testing.T) {
	a, err := New(Config{NodeID: "support", Version: "v1"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	shared := []any{"a", "b", "c"}
	if err := a.RegisterSession(
		"voice",
		"openai",
		"webrtc",
		WithSessionMetadata(map[string]any{
			"all":  shared,
			"head": shared[:1],
		}),
	); err != nil {
		t.Fatalf("RegisterSession returned error: %v", err)
	}

	metadata := a.SessionDefinitions()[0].Metadata
	if got := len(metadata["all"].([]any)); got != 3 {
		t.Errorf("len(all) = %d, want 3", got)
	}
	if got := len(metadata["head"].([]any)); got != 1 {
		t.Errorf("len(head) = %d, want 1", got)
	}
}
