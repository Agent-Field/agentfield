package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

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
