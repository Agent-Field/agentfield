package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockProvider implements MediaProvider for testing.
type mockProvider struct {
	name       string
	modalities []string
}

func (m *mockProvider) Name() string                { return m.name }
func (m *mockProvider) SupportedModalities() []string { return m.modalities }
func (m *mockProvider) GenerateImage(_ context.Context, _ ImageRequest) (*MediaResponse, error) {
	return &MediaResponse{Text: m.name + ":image"}, nil
}
func (m *mockProvider) GenerateAudio(_ context.Context, _ AudioRequest) (*MediaResponse, error) {
	return &MediaResponse{Text: m.name + ":audio"}, nil
}
func (m *mockProvider) GenerateVideo(_ context.Context, _ VideoRequest) (*MediaResponse, error) {
	return &MediaResponse{Text: m.name + ":video"}, nil
}

func TestMediaRouterResolve(t *testing.T) {
	router := NewMediaRouter()

	or := &mockProvider{name: "openrouter", modalities: []string{"image", "audio", "video"}}
	other := &mockProvider{name: "other", modalities: []string{"image"}}

	router.Register("openrouter/", or)
	router.Register("other/", other)

	tests := []struct {
		model      string
		capability string
		wantName   string
		wantErr    bool
	}{
		{"openrouter/kling", "video", "openrouter", false},
		{"openrouter/gpt-image-1", "image", "openrouter", false},
		{"other/some-model", "image", "other", false},
		{"other/some-model", "video", "", true},  // other doesn't support video
		{"unknown/model", "image", "", true},       // no matching prefix
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s_%s", tt.model, tt.capability), func(t *testing.T) {
			p, err := router.Resolve(tt.model, tt.capability)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantName, p.Name())
			}
		})
	}
}

func TestMediaRouterLongestPrefixMatch(t *testing.T) {
	router := NewMediaRouter()

	general := &mockProvider{name: "general", modalities: []string{"image"}}
	specific := &mockProvider{name: "specific", modalities: []string{"image"}}

	router.Register("openrouter/", general)
	router.Register("openrouter/dall-e", specific)

	p, err := router.Resolve("openrouter/dall-e-3", "image")
	require.NoError(t, err)
	assert.Equal(t, "specific", p.Name(), "longer prefix should match first")

	p, err = router.Resolve("openrouter/kling-v2", "image")
	require.NoError(t, err)
	assert.Equal(t, "general", p.Name(), "shorter prefix should match as fallback")
}

func TestMediaRouterEmpty(t *testing.T) {
	router := NewMediaRouter()
	_, err := router.Resolve("any/model", "image")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no provider")
}

func TestStripPrefix(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"openrouter/kling-video/v2", "kling-video/v2"},
		{"openrouter/gpt-image-1", "gpt-image-1"},
		{"plain-model", "plain-model"},
		{"openrouter/", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, stripPrefix(tt.input))
	}
}

func TestOpenRouterMediaProviderName(t *testing.T) {
	p, err := NewOpenRouterMediaProvider("test-key")
	require.NoError(t, err)
	assert.Equal(t, "openrouter", p.Name())
	assert.Equal(t, []string{"image", "audio", "video"}, p.SupportedModalities())
}

func TestOpenRouterMediaProviderDefaultKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "env-key")
	p, err := NewOpenRouterMediaProvider("")
	require.NoError(t, err)
	assert.Equal(t, "env-key", p.APIKey)
}

func TestOpenRouterMediaProviderEmptyKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	_, err := NewOpenRouterMediaProvider("")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "API key required")
}

func TestOpenRouterGenerateImage(t *testing.T) {
	// Mock server returning a chat completion with image content
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		assert.Equal(t, "gpt-image-1", payload["model"])

		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": []map[string]any{
							{"type": "text", "text": "Here is your image"},
							{"type": "image_url", "b64_json": "aW1hZ2VkYXRh"},
						},
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	resp, err := p.GenerateImage(context.Background(), ImageRequest{
		Prompt: "a cat",
		Model:  "openrouter/gpt-image-1",
	})
	require.NoError(t, err)
	assert.Equal(t, "Here is your image", resp.Text)
	require.Len(t, resp.Images, 1)
	assert.Equal(t, "aW1hZ2VkYXRh", resp.Images[0].B64JSON)
}

func TestOpenRouterGenerateImageError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": "bad request"}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	_, err := p.GenerateImage(context.Background(), ImageRequest{Prompt: "test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "400")
}

func TestOpenRouterGenerateAudio(t *testing.T) {
	// Mock SSE streaming server
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)

		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		assert.Equal(t, true, payload["stream"])

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		// Send text chunk
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"Hello"}}]}`)
		flusher.Flush()

		// Send audio chunk (base64 of "audio")
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"audio":{"data":"YXVkaW8="}}}]}`)
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:  "Say hello",
		Voice: "nova",
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello", resp.Text)
	require.NotNil(t, resp.Audio)
	assert.Equal(t, "pcm16", resp.Audio.Format)
	assert.NotEmpty(t, resp.Audio.Data)
}

func TestOpenRouterGenerateVideoSubmitError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(`{"error":"forbidden"}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt: "test",
		Model:  "openrouter/kling",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "403")
}

func TestOpenRouterGenerateVideoFullLifecycle(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/videos":
			json.NewEncoder(w).Encode(map[string]string{"id": "job-123"})
		case r.Method == http.MethodGet && r.URL.Path == "/videos/job-123":
			callCount++
			if callCount == 1 {
				json.NewEncoder(w).Encode(map[string]any{
					"id": "job-123", "status": "processing",
				})
			} else {
				json.NewEncoder(w).Encode(map[string]any{
					"id":           "job-123",
					"status":       "completed",
					"unsigned_url": "https://example.com/video.mp4",
					"duration":     5.0,
					"cost_usd":     0.05,
				})
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	resp, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:       "test video",
		Model:        "openrouter/kling",
		PollInterval: 10 * time.Millisecond,
		Timeout:      5 * time.Second,
	})
	require.NoError(t, err)
	require.Len(t, resp.Videos, 1)
	assert.Equal(t, "https://example.com/video.mp4", resp.Videos[0].URL)
	assert.Equal(t, "generated_video.mp4", resp.Videos[0].Filename)
	assert.Equal(t, 5.0, resp.Videos[0].Duration)
	assert.Equal(t, 0.05, resp.Videos[0].CostUSD)
	assert.Equal(t, "video/mp4", resp.Videos[0].MimeType)
}

func TestOpenRouterGenerateVideoEmptyPrompt(t *testing.T) {
	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: "http://localhost",
		Client:  &http.Client{},
	}
	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt: "",
		Model:  "openrouter/kling",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "prompt must not be empty")
}

func TestOpenRouterGenerateAudioEmptyText(t *testing.T) {
	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: "http://localhost",
		Client:  &http.Client{},
	}
	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "  ",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "text input must not be empty")
}
