package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ===========================================================================
// 1. MediaRouter integration tests
// ===========================================================================

func TestIntegrationMediaRouterRegisterResolve(t *testing.T) {
	router := NewMediaRouter()

	fal := &mockProvider{name: "fal", modalities: []string{"image", "audio", "video"}}
	or := &mockProvider{name: "openrouter", modalities: []string{"image", "audio", "video"}}
	litellm := &mockProvider{name: "litellm", modalities: []string{"image", "audio"}}

	router.Register("fal-ai/", fal)
	router.Register("openrouter/", or)
	router.Register("", litellm)

	tests := []struct {
		model      string
		capability string
		wantName   string
		wantErr    bool
	}{
		{"fal-ai/flux/dev", "image", "fal", false},
		{"fal-ai/minimax-video/image-to-video", "video", "fal", false},
		{"openrouter/google/veo-3", "video", "openrouter", false},
		{"openrouter/openai/gpt-image-1", "image", "openrouter", false},
		{"dall-e-3", "image", "litellm", false},
		{"tts-1", "audio", "litellm", false},
		{"dall-e-3", "video", "", true}, // litellm doesn't support video
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

func TestIntegrationMediaRouterLongestPrefixFirst(t *testing.T) {
	router := NewMediaRouter()

	general := &mockProvider{name: "general", modalities: []string{"image", "video"}}
	specific := &mockProvider{name: "specific", modalities: []string{"image", "video"}}

	// Register in any order — longest should always win
	router.Register("openrouter/", general)
	router.Register("openrouter/google/", specific)

	p, err := router.Resolve("openrouter/google/veo-3", "video")
	require.NoError(t, err)
	assert.Equal(t, "specific", p.Name(), "longer prefix should match first")

	p, err = router.Resolve("openrouter/openai/dall-e", "image")
	require.NoError(t, err)
	assert.Equal(t, "general", p.Name(), "shorter prefix should match as fallback")
}

func TestIntegrationMediaRouterEmptyPrefixCatchAll(t *testing.T) {
	router := NewMediaRouter()
	fallback := &mockProvider{name: "fallback", modalities: []string{"image", "audio"}}
	router.Register("", fallback)

	p, err := router.Resolve("any-model", "image")
	require.NoError(t, err)
	assert.Equal(t, "fallback", p.Name())
}

func TestIntegrationMediaRouterNoMatch(t *testing.T) {
	router := NewMediaRouter()
	_, err := router.Resolve("unknown/model", "video")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no provider")
}

// ===========================================================================
// 2. OpenRouterMediaProvider: httptest video lifecycle
// ===========================================================================

func TestIntegrationVideoSubmitPollDownload(t *testing.T) {
	pollCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/videos":
			// Validate request body
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			assert.Equal(t, "google/veo-3", payload["model"])
			assert.Equal(t, "A sunset timelapse", payload["prompt"])
			assert.Equal(t, float64(5), payload["duration"])

			json.NewEncoder(w).Encode(map[string]string{"id": "job-42"})

		case r.Method == http.MethodGet && r.URL.Path == "/videos/job-42":
			pollCount++
			if pollCount < 2 {
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "job-42",
					"status": "processing",
				})
			} else {
				json.NewEncoder(w).Encode(map[string]any{
					"id":           "job-42",
					"status":       "completed",
					"unsigned_url": "https://cdn.example.com/video.mp4",
					"duration":     5.0,
					"cost_usd":     0.08,
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
		Prompt:       "A sunset timelapse",
		Model:        "openrouter/google/veo-3",
		Duration:     5,
		PollInterval: 10 * time.Millisecond,
		Timeout:      5 * time.Second,
	})

	require.NoError(t, err)
	require.Len(t, resp.Videos, 1)

	video := resp.Videos[0]
	assert.Equal(t, "https://cdn.example.com/video.mp4", video.URL)
	assert.Equal(t, "video/mp4", video.MimeType)
	assert.Equal(t, "generated_video.mp4", video.Filename)
	assert.Equal(t, 5.0, video.Duration)
	assert.Equal(t, 0.08, video.CostUSD)
	assert.True(t, pollCount >= 2, "should have polled at least twice")
}

func TestIntegrationVideoSubmitError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		w.Write([]byte(`{"error":"insufficient credits"}`))
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
	assert.Contains(t, err.Error(), "402")
}

func TestIntegrationVideoJobFailed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			json.NewEncoder(w).Encode(map[string]string{"id": "job-fail"})
		case r.Method == http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "job-fail",
				"status": "failed",
				"error":  "content policy violation",
			})
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:       "test",
		Model:        "openrouter/kling",
		PollInterval: 10 * time.Millisecond,
		Timeout:      2 * time.Second,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed")
	assert.Contains(t, err.Error(), "content policy")
}

func TestIntegrationVideoInvalidJobID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"id": "../../../etc/passwd"})
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
	assert.Contains(t, err.Error(), "invalid job ID")
}

// ===========================================================================
// 2b. OpenRouterMediaProvider: audio SSE lifecycle
// ===========================================================================

func TestIntegrationAudioSSEStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		// Validate payload
		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		assert.Equal(t, true, payload["stream"])
		assert.Equal(t, "openai/gpt-audio-mini", payload["model"])

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		audioB64 := base64.StdEncoding.EncodeToString([]byte("audio-chunk-1"))
		audioB64_2 := base64.StdEncoding.EncodeToString([]byte("audio-chunk-2"))

		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"Hello world"}}]}`)
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"%s\"}}}]}\n\n", audioB64)
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"%s\"}}}]}\n\n", audioB64_2)
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
	// Pre-seed metadata so routing picks the chat-completions path
	// without hitting the real OpenRouter `/models/.../endpoints` endpoint.
	p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})

	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:   "Say hello",
		Model:  "openrouter/openai/gpt-audio-mini",
		Voice:  "nova",
		Format: "mp3", // avoid pcm→wav rewrap so we can compare raw bytes
	})

	require.NoError(t, err)
	assert.Equal(t, "Hello world", resp.Text)
	require.NotNil(t, resp.Audio)
	assert.Equal(t, "mp3", resp.Audio.Format)
	assert.NotEmpty(t, resp.Audio.Data)

	// Decode and verify audio bytes
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, "audio-chunk-1audio-chunk-2", string(decoded))
}

func TestIntegrationAudioWithCustomFormat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)

		audioConf := payload["audio"].(map[string]any)
		assert.Equal(t, "mp3", audioConf["format"])
		assert.Equal(t, "echo", audioConf["voice"])

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}
	p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})

	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:   "test",
		Model:  "openai/gpt-audio-mini",
		Voice:  "echo",
		Format: "mp3",
	})
	require.NoError(t, err)
	assert.Equal(t, "mp3", resp.Audio.Format)
}

func TestIntegrationAudioSpeechEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/audio/speech", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		assert.Equal(t, "hexgrad/kokoro-82m", payload["model"])
		assert.Equal(t, "af_bella", payload["voice"])
		assert.Equal(t, "pcm", payload["response_format"]) // wav → pcm on wire

		// Return 1KB of fake PCM16
		pcm := bytes.Repeat([]byte{0x00, 0x01}, 500)
		w.Header().Set("Content-Type", "audio/pcm")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(pcm)
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "test-key", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("hexgrad/kokoro-82m", []string{"speech"}, []string{"text"})

	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:   "hello",
		Model:  "openrouter/hexgrad/kokoro-82m",
		Voice:  "af_bella",
		Format: "wav",
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Audio)
	assert.Equal(t, "wav", resp.Audio.Format)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, []byte("RIFF"), decoded[:4])
	assert.Equal(t, []byte("WAVE"), decoded[8:12])
}

func TestIntegrationImageGeneration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		assert.Equal(t, "gpt-image-1", payload["model"])

		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": []map[string]any{
							{"type": "text", "text": "Here is a cat"},
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
		Prompt:  "a cat",
		Model:   "openrouter/gpt-image-1",
		Size:    "1024x1024",
		Quality: "standard",
	})
	require.NoError(t, err)
	assert.Equal(t, "Here is a cat", resp.Text)
	require.Len(t, resp.Images, 1)
	assert.Equal(t, "aW1hZ2VkYXRh", resp.Images[0].B64JSON)
}

// ===========================================================================
// 3. Input validation
// ===========================================================================

func TestIntegrationEmptyPromptRejected(t *testing.T) {
	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: "http://unused",
		Client:  &http.Client{},
	}

	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt: "",
		Model:  "openrouter/kling",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "prompt must not be empty")
}

func TestIntegrationEmptyTextRejected(t *testing.T) {
	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: "http://unused",
		Client:  &http.Client{},
	}

	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "   ",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "text input must not be empty")
}

func TestIntegrationWhitespaceOnlyPromptRejected(t *testing.T) {
	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: "http://unused",
		Client:  &http.Client{},
	}

	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt: "  \t\n  ",
		Model:  "openrouter/kling",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "prompt must not be empty")
}

func TestIntegrationJobIDValidation(t *testing.T) {
	// validJobID rejects path traversal and special characters
	assert.True(t, validJobID.MatchString("job-123"))
	assert.True(t, validJobID.MatchString("abc_def"))
	assert.False(t, validJobID.MatchString("../etc/passwd"))
	assert.False(t, validJobID.MatchString("job 123"))
	assert.False(t, validJobID.MatchString("job/123"))
	assert.False(t, validJobID.MatchString(""))
}

func TestIntegrationNoAPIKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	_, err := NewOpenRouterMediaProvider("")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "API key required")
}

// ===========================================================================
// 4. Context cancellation
// ===========================================================================

func TestIntegrationContextCancelStopsPollLoop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			json.NewEncoder(w).Encode(map[string]string{"id": "job-cancel"})
		case r.Method == http.MethodGet:
			// Always return pending — context cancel must stop us
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "job-cancel",
				"status": "pending",
			})
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  srv.Client(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := p.GenerateVideo(ctx, VideoRequest{
		Prompt:       "test",
		Model:        "openrouter/kling",
		PollInterval: 50 * time.Millisecond,
		Timeout:      30 * time.Second, // large timeout — context cancel should win
	})

	elapsed := time.Since(start)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context")
	// Should have stopped quickly (well under the 30s video timeout)
	assert.Less(t, elapsed, 2*time.Second, "context cancellation should stop poll quickly")
}

func TestIntegrationContextCancelDuringSubmit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Delay so context expires during submit
		time.Sleep(500 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"id": "job-x"})
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Client:  &http.Client{Timeout: 5 * time.Second},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := p.GenerateVideo(ctx, VideoRequest{
		Prompt: "test",
		Model:  "openrouter/kling",
	})
	assert.Error(t, err)
}
