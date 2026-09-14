package types

import (
	"encoding/json"
	"math"
	"strings"
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

func TestParseSessionTurnDetection(t *testing.T) {
	for _, tc := range []struct {
		name, provider, transport, raw, expected, wantError string
	}{
		{name: "omitted", provider: "openai", transport: "webrtc", expected: `{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}`},
		{name: "null", provider: "openai", transport: "webrtc", raw: " null ", expected: `{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}`},
		{name: "server explicit zero and false", provider: "openai", transport: "webrtc", raw: `{"type":"server_vad","threshold":0,"prefix_padding_ms":0,"silence_duration_ms":0,"create_response":false,"interrupt_response":false}`, expected: `{"type":"server_vad","threshold":0,"prefix_padding_ms":0,"silence_duration_ms":0,"create_response":false,"interrupt_response":false}`},
		{name: "semantic", provider: "openai", transport: "websocket", raw: `{"type":"semantic_vad","eagerness":"low"}`, expected: `{"type":"semantic_vad","create_response":true,"interrupt_response":true,"eagerness":"low"}`},
		{name: "other provider omitted", provider: "openrouter", transport: "audio_turns", expected: `null`},
		{name: "other provider configured", provider: "openrouter", transport: "audio_turns", raw: `{"type":"server_vad"}`, wantError: "requires provider=openai"},
		{name: "malformed JSON", provider: "openai", transport: "webrtc", raw: `{`, wantError: "must be an object"},
		{name: "array", provider: "openai", transport: "webrtc", raw: `[]`, wantError: "must be an object"},
		{name: "missing type", provider: "openai", transport: "webrtc", raw: `{}`, wantError: "turn_detection.type"},
		{name: "unknown field", provider: "openai", transport: "webrtc", raw: `{"type":"server_vad","typo":true}`, wantError: "invalid turn_detection"},
		{name: "incorrect case", provider: "openai", transport: "webrtc", raw: `{"Type":"server_vad"}`, wantError: "unknown turn_detection field"},
		{name: "wrong boolean type", provider: "openai", transport: "webrtc", raw: `{"type":"server_vad","create_response":"false"}`, wantError: "invalid turn_detection"},
		{name: "null field", provider: "openai", transport: "webrtc", raw: `{"type":"server_vad","threshold":null}`, wantError: "threshold must not be null"},
		{name: "server with eagerness", provider: "openai", transport: "webrtc", raw: `{"type":"server_vad","eagerness":"low"}`, wantError: "eagerness is unsupported"},
		{name: "empty eagerness", provider: "openai", transport: "webrtc", raw: `{"type":"semantic_vad","eagerness":""}`, wantError: "eagerness must be"},
		{name: "mixed mode fields", provider: "openai", transport: "webrtc", raw: `{"type":"semantic_vad","threshold":0.5}`, wantError: "unsupported for semantic_vad"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config, err := ParseSessionTurnDetection(tc.provider, tc.transport, json.RawMessage(tc.raw))
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("expected error containing %q, got config=%+v err=%v", tc.wantError, config, err)
				}
				if config != nil {
					t.Fatalf("invalid input returned usable config: %+v", config)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(config)
			if err != nil {
				t.Fatal(err)
			}
			if string(body) != tc.expected {
				t.Fatalf("got %s, want %s", body, tc.expected)
			}
		})
	}
}
