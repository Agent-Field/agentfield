package agent

import (
	"fmt"
	"math"
)

// TurnDetection configures OpenAI Realtime input audio. Nil options use defaults.
// Pointer fields preserve explicit false and zero values during JSON serialization.
type TurnDetection struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	PrefixPaddingMS   *int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMS *int     `json:"silence_duration_ms,omitempty"`
	CreateResponse    *bool    `json:"create_response,omitempty"`
	InterruptResponse *bool    `json:"interrupt_response,omitempty"`
	Eagerness         string   `json:"eagerness,omitempty"`
}

// NormalizeTurnDetection validates options and returns an independent config with
// automatic responses and barge-in enabled unless explicitly disabled.
func NormalizeTurnDetection(provider, transport string, config *TurnDetection) (*TurnDetection, error) {
	if provider != "openai" || (transport != "webrtc" && transport != "websocket") {
		if config != nil {
			return nil, fmt.Errorf("turn_detection requires provider=openai and transport=webrtc or websocket")
		}
		return nil, nil
	}
	result := TurnDetection{Type: "server_vad"}
	if config != nil {
		result = *config
	}
	switch result.Type {
	case "server_vad":
		if result.Eagerness != "" {
			return nil, fmt.Errorf("turn_detection.eagerness is unsupported for server_vad")
		}
		threshold := 0.5
		if result.Threshold != nil {
			threshold = *result.Threshold
		}
		if math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold < 0 || threshold > 1 {
			return nil, fmt.Errorf("turn_detection.threshold must be a finite number between 0 and 1")
		}
		padding, silence := 300, 500
		if result.PrefixPaddingMS != nil {
			padding = *result.PrefixPaddingMS
		}
		if result.SilenceDurationMS != nil {
			silence = *result.SilenceDurationMS
		}
		if padding < 0 || silence < 0 {
			return nil, fmt.Errorf("turn_detection durations must be non-negative integers")
		}
		result.Threshold, result.PrefixPaddingMS, result.SilenceDurationMS = &threshold, &padding, &silence
	case "semantic_vad":
		if result.Threshold != nil || result.PrefixPaddingMS != nil || result.SilenceDurationMS != nil {
			return nil, fmt.Errorf("turn_detection threshold and durations are unsupported for semantic_vad")
		}
		if result.Eagerness == "" {
			result.Eagerness = "auto"
		}
		switch result.Eagerness {
		case "auto", "low", "medium", "high":
		default:
			return nil, fmt.Errorf("turn_detection.eagerness must be auto, low, medium, or high")
		}
	default:
		return nil, fmt.Errorf("turn_detection.type must be server_vad or semantic_vad")
	}
	create, interrupt := true, true
	if result.CreateResponse != nil {
		create = *result.CreateResponse
	}
	if result.InterruptResponse != nil {
		interrupt = *result.InterruptResponse
	}
	result.CreateResponse, result.InterruptResponse = &create, &interrupt
	return &result, nil
}
