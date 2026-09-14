package types

import (
	"encoding/json"
	"math"
	"testing"
)

func turnDetectionTestPtr[T any](v T) *T { return &v }

func TestNormalizeTurnDetectionDefaultsAndOverrides(t *testing.T) {
	for _, transport := range []string{"webrtc", "websocket"} {
		config, err := NormalizeTurnDetection("openai", transport, nil)
		if err != nil {
			t.Fatal(err)
		}
		if config.Type != "server_vad" || *config.Threshold != 0.5 || *config.PrefixPaddingMS != 300 ||
			*config.SilenceDurationMS != 500 || !*config.CreateResponse || !*config.InterruptResponse {
			t.Fatalf("unexpected defaults: %+v", config)
		}
	}
	input := &TurnDetection{Type: "server_vad", Threshold: turnDetectionTestPtr(0.0),
		PrefixPaddingMS: turnDetectionTestPtr(0), SilenceDurationMS: turnDetectionTestPtr(750),
		CreateResponse: turnDetectionTestPtr(false), InterruptResponse: turnDetectionTestPtr(false)}
	config, err := NormalizeTurnDetection("openai", "webrtc", input)
	if err != nil {
		t.Fatal(err)
	}
	*input.Threshold = 1
	*input.InterruptResponse = true
	serialized, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"type":"server_vad","threshold":0,"prefix_padding_ms":0,"silence_duration_ms":750,"create_response":false,"interrupt_response":false}`
	if string(serialized) != expected {
		t.Fatalf("got %s, want %s", serialized, expected)
	}
	semantic, err := NormalizeTurnDetection("openai", "webrtc", &TurnDetection{Type: "semantic_vad"})
	if err != nil {
		t.Fatal(err)
	}
	serialized, err = json.Marshal(semantic)
	if err != nil {
		t.Fatal(err)
	}
	expected = `{"type":"semantic_vad","create_response":true,"interrupt_response":true,"eagerness":"auto"}`
	if string(serialized) != expected {
		t.Fatalf("got %s, want %s", serialized, expected)
	}
}

func TestNormalizeTurnDetectionRejectsInvalidOptions(t *testing.T) {
	for _, input := range []TurnDetection{
		{}, {Type: "client_vad"},
		{Type: "server_vad", Threshold: turnDetectionTestPtr(1.1)},
		{Type: "server_vad", Threshold: turnDetectionTestPtr(math.NaN())},
		{Type: "server_vad", Threshold: turnDetectionTestPtr(math.Inf(1))},
		{Type: "server_vad", PrefixPaddingMS: turnDetectionTestPtr(-1)},
		{Type: "server_vad", SilenceDurationMS: turnDetectionTestPtr(-1)},
		{Type: "server_vad", Eagerness: "low"},
		{Type: "semantic_vad", Threshold: turnDetectionTestPtr(0.0)},
		{Type: "semantic_vad", PrefixPaddingMS: turnDetectionTestPtr(0)},
		{Type: "semantic_vad", SilenceDurationMS: turnDetectionTestPtr(0)},
		{Type: "semantic_vad", Eagerness: "urgent"},
	} {
		if _, err := NormalizeTurnDetection("openai", "webrtc", &input); err == nil {
			t.Fatalf("accepted invalid config: %+v", input)
		}
	}
	if _, err := NormalizeTurnDetection("openrouter", "audio_turns", &TurnDetection{Type: "server_vad"}); err == nil {
		t.Fatal("accepted VAD for openrouter")
	}
	if config, err := NormalizeTurnDetection("openrouter", "audio_turns", nil); err != nil || config != nil {
		t.Fatalf("changed openrouter defaults: %+v, %v", config, err)
	}
}
