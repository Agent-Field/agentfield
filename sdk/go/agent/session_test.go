package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAgentSessionDefinitionsDetachesTurnDetection(t *testing.T) {
	for _, tc := range []struct {
		name, provider, transport string
		config                    *TurnDetection
	}{
		{name: "defaults", provider: "openai", transport: "webrtc"},
		{name: "explicit zero and false", provider: "openai", transport: "websocket", config: &TurnDetection{
			Type: "server_vad", Threshold: turnDetectionTestPtr(0.0),
			PrefixPaddingMS: turnDetectionTestPtr(0), SilenceDurationMS: turnDetectionTestPtr(0),
			CreateResponse: turnDetectionTestPtr(false), InterruptResponse: turnDetectionTestPtr(false),
		}},
		{name: "semantic", provider: "openai", transport: "webrtc", config: &TurnDetection{Type: "semantic_vad", Eagerness: "low"}},
		{name: "no turn detection", provider: "openrouter", transport: "audio_turns"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := New(Config{NodeID: "support", Version: "v1"})
			if err != nil {
				t.Fatal(err)
			}
			var opts []SessionOption
			if tc.config != nil {
				opts = append(opts, WithSessionTurnDetection(*tc.config))
			}
			if err := a.RegisterSession("voice", tc.provider, tc.transport, opts...); err != nil {
				t.Fatal(err)
			}
			want, err := json.Marshal(a.sessions["voice"].TurnDetection)
			if err != nil {
				t.Fatal(err)
			}
			snapshot := a.SessionDefinitions()[0]
			if config := snapshot.TurnDetection; config != nil {
				config.Type = "mutated"
				config.Eagerness = "mutated"
				if config.Threshold != nil {
					*config.Threshold = 1
				}
				if config.PrefixPaddingMS != nil {
					*config.PrefixPaddingMS = 999
				}
				if config.SilenceDurationMS != nil {
					*config.SilenceDurationMS = 999
				}
				*config.CreateResponse = !*config.CreateResponse
				*config.InterruptResponse = !*config.InterruptResponse
			}
			got, err := json.Marshal(a.SessionDefinitions()[0].TurnDetection)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(want) {
				t.Fatalf("snapshot mutation changed registered session: got %s, want %s", got, want)
			}
		})
	}
}

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

func TestAgentRegisterSessionTurnDetection(t *testing.T) {
	a, err := New(Config{NodeID: "support", Version: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	err = a.RegisterSession("voice", "OpenAI", "WebRTC", WithSessionTurnDetection(TurnDetection{
		Type: "server_vad", Threshold: turnDetectionTestPtr(0.0), InterruptResponse: turnDetectionTestPtr(false),
	}))
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(a.SessionDefinitions())
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{`"turn_detection":`, `"threshold":0`, `"interrupt_response":false`} {
		if !strings.Contains(string(body), value) {
			t.Fatalf("missing %s in %s", value, body)
		}
	}
	err = a.RegisterSession("invalid", "openai", "webrtc", WithSessionTurnDetection(TurnDetection{
		Type: "semantic_vad", Threshold: turnDetectionTestPtr(0.5),
	}))
	if err == nil || len(a.SessionDefinitions()) != 1 {
		t.Fatal("invalid registration must not change registry")
	}
	err = a.RegisterSession("invalid", "openrouter", "audio_turns", WithSessionTurnDetection(TurnDetection{Type: "server_vad"}))
	if err == nil {
		t.Fatal("expected unsupported provider error")
	}
}
