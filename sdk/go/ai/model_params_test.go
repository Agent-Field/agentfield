package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNeedsMaxCompletionTokens(t *testing.T) {
	tests := []struct {
		model    string
		expected bool
	}{
		// o1 series
		{"o1-mini", true},
		{"o1-preview", true},
		{"o1", true},
		// o3 series
		{"o3-mini", true},
		{"o3", true},
		// o4 series
		{"o4-mini", true},
		// Future o-series
		{"o5", true},
		{"o5-mini", true},
		{"o10-preview", true},
		// gpt-4o series
		{"gpt-4o", true},
		{"gpt-4o-mini", true},
		{"gpt-4o-2024-05-13", true},
		// gpt-4.x dot releases
		{"gpt-4.1", true},
		{"gpt-4.5-preview", true},
		// gpt-5 and future gpt families
		{"gpt-5", true},
		{"gpt-5-mini", true},
		{"gpt-5.1", true},
		{"gpt-6", true},
		// With provider prefix
		{"openai/gpt-4o", true},
		{"openai/o1-mini", true},
		{"openai/gpt-5", true},
		{"openrouter/openai/gpt-4o-mini", true},
		// Case insensitive
		{"GPT-4O", true},
		{"O1-Mini", true},
		// Legacy models — should NOT rewrite
		{"gpt-4", false},
		{"gpt-4-turbo", false},
		{"gpt-4-32k", false},
		{"gpt-3.5-turbo", false},
		{"gpt-4-0613", false},
		// Non-OpenAI models
		{"claude-3-opus", false},
		{"mistral-large", false},
		{"llama-3-70b", false},
		// "o"-prefixed names without a digit — should NOT match the o-series
		{"omni-moderation-latest", false},
		{"openchat-7b", false},
		// Empty
		{"", false},
	}

	for _, tt := range tests {
		got := needsMaxCompletionTokens(tt.model)
		if got != tt.expected {
			t.Errorf("needsMaxCompletionTokens(%q) = %v, want %v", tt.model, got, tt.expected)
		}
	}
}

func TestMarshalRequest_RewritesMaxTokensForNewModels(t *testing.T) {
	maxTokens := 1000
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "gpt-4o"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent for gpt-4o model")
	}
	if val, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present for gpt-4o model")
	} else if int(val.(float64)) != 1000 {
		t.Errorf("expected max_completion_tokens=1000, got %v", val)
	}
}

func TestMarshalRequest_KeepsMaxTokensForLegacyModels(t *testing.T) {
	maxTokens := 2000
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "gpt-3.5-turbo"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "gpt-3.5-turbo",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_completion_tokens"]; ok {
		t.Error("expected max_completion_tokens to be absent for gpt-3.5-turbo model")
	}
	if val, ok := raw["max_tokens"]; !ok {
		t.Error("expected max_tokens to be present for gpt-3.5-turbo model")
	} else if int(val.(float64)) != 2000 {
		t.Errorf("expected max_tokens=2000, got %v", val)
	}
}

func TestMarshalRequest_O1MiniUsesMaxCompletionTokens(t *testing.T) {
	maxTokens := 500
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "o1-mini"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "o1-mini",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent for o1-mini")
	}
	if val, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present for o1-mini")
	} else if int(val.(float64)) != 500 {
		t.Errorf("expected max_completion_tokens=500, got %v", val)
	}
}

func TestMarshalRequest_Gpt5UsesMaxCompletionTokens(t *testing.T) {
	maxTokens := 1200
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "gpt-5-mini"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "gpt-5-mini",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent for gpt-5-mini")
	}
	if val, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present for gpt-5-mini")
	} else if int(val.(float64)) != 1200 {
		t.Errorf("expected max_completion_tokens=1200, got %v", val)
	}
}

func TestMarshalRequest_NilMaxTokensOmitted(t *testing.T) {
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "gpt-4o"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: nil,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent when nil")
	}
	if _, ok := raw["max_completion_tokens"]; ok {
		t.Error("expected max_completion_tokens to be absent when MaxTokens is nil")
	}
}

func TestMarshalRequest_UsesRequestModelOverClientModel(t *testing.T) {
	maxTokens := 800
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "gpt-3.5-turbo" // client default is legacy

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "o1-preview", // request overrides to new model
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent for o1-preview")
	}
	if _, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present for o1-preview")
	}
}

func TestIsVouchedRewriteEndpoint(t *testing.T) {
	tests := []struct {
		baseURL  string
		expected bool
	}{
		// OpenAI
		{"https://api.openai.com/v1", true},
		{"https://api.openai.com/v1/", true},
		// Azure OpenAI
		{"https://my-resource.openai.azure.com/openai/deployments/gpt-4o", true},
		// OpenRouter
		{"https://openrouter.ai/api/v1", true},
		// Known non-OpenAI providers — should NOT rewrite
		{"https://api.anthropic.com/v1", false},
		{"https://api.cohere.ai/v1", false},
		{"https://api.cohere.com/v1/chat", false},
		// Custom/self-hosted (Ollama, LM Studio, vLLM, proxies) — keep
		// max_tokens: unknown hosts may silently drop unknown fields.
		{"http://localhost:8080/v1", false},
		{"http://localhost:11434/v1", false},
		{"https://my-proxy.example.com/v1", false},
		{"https://vllm.internal.corp/v1", false},
		// Lookalike hosts must not match by substring
		{"https://notopenai.com/v1", false},
		{"https://api.openai.com.evil.example/v1", false},
		// Edge cases: no scheme means no host, and requests can't be sent
		// there anyway
		{"api.openai.com/v1", false},
		{"", false},
	}

	for _, tt := range tests {
		got := isVouchedRewriteEndpoint(tt.baseURL)
		if got != tt.expected {
			t.Errorf("isVouchedRewriteEndpoint(%q) = %v, want %v", tt.baseURL, got, tt.expected)
		}
	}
}

func TestMarshalRequest_NonOpenAIEndpointKeepsMaxTokens(t *testing.T) {
	// Even for a new model, non-OpenAI endpoints should keep max_tokens
	maxTokens := 1000
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.anthropic.com/v1"
	cfg.Model = "gpt-4o" // model name matches but endpoint is Anthropic

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_completion_tokens"]; ok {
		t.Error("expected max_completion_tokens to be absent for non-OpenAI endpoint")
	}
	if _, ok := raw["max_tokens"]; !ok {
		t.Error("expected max_tokens to be present for non-OpenAI endpoint")
	}
}

func TestMarshalRequest_SelfHostedEndpointKeepsMaxTokens(t *testing.T) {
	// Self-hosted OpenAI-compatible servers (Ollama, LM Studio, llama.cpp,
	// vLLM) may silently drop unknown fields, so max_tokens must be kept
	// even when the model is named like a newer OpenAI model.
	maxTokens := 700
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "http://localhost:11434/v1"
	cfg.Model = "gpt-4o"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "o1-mini",
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, ok := raw["max_completion_tokens"]; ok {
		t.Error("expected max_completion_tokens to be absent for self-hosted endpoint")
	}
	if val, ok := raw["max_tokens"]; !ok {
		t.Error("expected max_tokens to be present for self-hosted endpoint")
	} else if int(val.(float64)) != 700 {
		t.Errorf("expected max_tokens=700, got %v", val)
	}
}

func TestMarshalRequest_StreamingPathRewritesMaxTokens(t *testing.T) {
	// Verify that marshalRequest (used by both streaming and non-streaming paths)
	// correctly rewrites for streaming requests too.
	maxTokens := 2048
	cfg := DefaultConfig()
	cfg.APIKey = "test-key"
	cfg.BaseURL = "https://api.openai.com/v1"
	cfg.Model = "o3-mini"

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	req := &Request{
		Model:     "o3-mini",
		MaxTokens: &maxTokens,
		Stream:    true, // streaming request
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}},
		},
	}

	body, err := client.marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	// Verify rewrite happened
	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent for o3-mini streaming request")
	}
	if val, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present for o3-mini streaming request")
	} else if int(val.(float64)) != 2048 {
		t.Errorf("expected max_completion_tokens=2048, got %v", val)
	}
	// Verify stream flag is preserved
	if val, ok := raw["stream"]; !ok || val != true {
		t.Error("expected stream=true to be preserved in rewritten request")
	}
}

// recordingTransport captures the request body without hitting the network
// and replies with a minimal valid chat-completion response. It lets tests
// point BaseURL at a real provider hostname (which gates the rewrite) while
// still observing the exact JSON that would go on the wire.
type recordingTransport struct {
	body []byte
}

func (rt *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	rt.body = b

	resp := Response{
		Choices: []Choice{
			{
				Message: Message{
					Role:    "assistant",
					Content: []ContentPart{{Type: "text", Text: "ok"}},
				},
			},
		},
	}
	payload, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}

	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(string(payload))),
	}, nil
}

func TestComplete_WireBody_VouchedEndpointRewrites(t *testing.T) {
	// End-to-end through Complete(): the JSON body sent to api.openai.com
	// must carry max_completion_tokens (not max_tokens) for a newer model.
	cfg := &Config{
		APIKey:  "test-key",
		BaseURL: "https://api.openai.com/v1",
		Model:   "gpt-5",
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	rt := &recordingTransport{}
	client.httpClient.Transport = rt

	if _, err := client.Complete(context.Background(), "hello", WithMaxTokens(321)); err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(rt.body, &raw); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}

	if _, ok := raw["max_tokens"]; ok {
		t.Error("expected max_tokens to be absent in wire body for gpt-5 on api.openai.com")
	}
	if val, ok := raw["max_completion_tokens"]; !ok {
		t.Error("expected max_completion_tokens to be present in wire body for gpt-5 on api.openai.com")
	} else if int(val.(float64)) != 321 {
		t.Errorf("expected max_completion_tokens=321, got %v", val)
	}
}

func TestComplete_WireBody_UnknownHostKeepsMaxTokens(t *testing.T) {
	// End-to-end over real HTTP: a self-hosted (httptest) endpoint must
	// receive max_tokens even for a model named like a newer OpenAI model.
	var captured []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		captured = b

		resp := Response{
			Choices: []Choice{
				{
					Message: Message{
						Role:    "assistant",
						Content: []ContentPart{{Type: "text", Text: "ok"}},
					},
				},
			},
		}
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	defer server.Close()

	cfg := &Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "gpt-4o",
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	if _, err := client.Complete(context.Background(), "hello", WithMaxTokens(654)); err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(captured, &raw); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}

	if _, ok := raw["max_completion_tokens"]; ok {
		t.Error("expected max_completion_tokens to be absent in wire body for unknown host")
	}
	if val, ok := raw["max_tokens"]; !ok {
		t.Error("expected max_tokens to be present in wire body for unknown host")
	} else if int(val.(float64)) != 654 {
		t.Errorf("expected max_tokens=654, got %v", val)
	}
}
