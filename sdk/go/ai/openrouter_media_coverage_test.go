package ai

import (
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

// baseURL() should fall back to default when BaseURL is empty.
func TestBaseURLEmptyUsesDefault(t *testing.T) {
	p := &OpenRouterMediaProvider{APIKey: "k"}
	assert.Equal(t, defaultOpenRouterBaseURL, p.baseURL())
}

// GenerateVideo exercises every optional payload field.
func TestGenerateVideoAllOptionalFields(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			require.NoError(t, json.NewDecoder(r.Body).Decode(&captured))
			json.NewEncoder(w).Encode(map[string]string{"id": "job-opt"})
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]any{
				"id":           "job-opt",
				"status":       "completed",
				"unsigned_url": "https://cdn/x.mp4",
				"duration":     3.0,
				"cost_usd":     0.02,
			})
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}

	gen := true
	seed := 7
	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:          "x",
		Model:           "openrouter/m",
		Duration:        4,
		Resolution:      "1080p",
		AspectRatio:     "16:9",
		GenerateAudio:   &gen,
		Seed:            &seed,
		FrameImages:     []map[string]any{{"url": "a"}},
		InputReferences: []map[string]any{{"url": "b"}},
		Extra:           map[string]any{"style": "cinema"},
		PollInterval:    5 * time.Millisecond,
		Timeout:         2 * time.Second,
	})
	require.NoError(t, err)

	assert.Equal(t, "1080p", captured["resolution"])
	assert.Equal(t, "16:9", captured["aspect_ratio"])
	assert.Equal(t, true, captured["generate_audio"])
	assert.Equal(t, float64(7), captured["seed"])
	assert.NotNil(t, captured["frame_images"])
	assert.NotNil(t, captured["input_references"])
	assert.Equal(t, "cinema", captured["style"])
}

// Submit returns empty id → error.
func TestGenerateVideoMissingJobID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateVideo(context.Background(), VideoRequest{Prompt: "x", Model: "m"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no job ID")
}

// Submit returns malformed JSON → parse error.
func TestGenerateVideoSubmitBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not-json`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateVideo(context.Background(), VideoRequest{Prompt: "x", Model: "m"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse submit response")
}

// Poll keeps returning a 500 → trips the transient-error retry limit.
func TestGenerateVideoPollTransientErrorsExceeded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost {
			json.NewEncoder(w).Encode(map[string]string{"id": "job-boom"})
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"upstream"}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:       "x",
		Model:        "m",
		PollInterval: 5 * time.Millisecond,
		Timeout:      3 * time.Second,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "video poll failed")
}

// Poll returns malformed JSON → parse error counted as transient.
func TestGenerateVideoPollBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost {
			json.NewEncoder(w).Encode(map[string]string{"id": "job-bad"})
			return
		}
		w.Write([]byte(`garbage`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:       "x",
		Model:        "m",
		PollInterval: 5 * time.Millisecond,
		Timeout:      3 * time.Second,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "video poll failed")
}

// Image request with ImageConfig is forwarded to the API.
func TestGenerateImageWithImageConfig(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":null}}]}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateImage(context.Background(), ImageRequest{
		Prompt:      "x",
		ImageConfig: &ImageConfig{AspectRatio: "1:1", ImageSize: "512"},
	})
	require.NoError(t, err)
	assert.NotNil(t, got["image_config"])
}

// Image response with content as plain string containing inline base64.
func TestGenerateImageStringContentInlineBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{"choices":[{"message":{"content":"Here is it: data:image/png;base64,QUJD and more"}}]}`
		w.Write([]byte(body))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateImage(context.Background(), ImageRequest{Prompt: "x"})
	require.NoError(t, err)
	assert.Contains(t, resp.Text, "Here is it")
	require.Len(t, resp.Images, 1)
	assert.Equal(t, "QUJD", resp.Images[0].B64JSON)
}

// Image response where Gemini-style images[] contains both data: URL and a regular URL.
func TestGenerateImageGeminiStyleImages(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{
			"choices":[{
				"message":{
					"content": null,
					"images":[
						{"type":"image_url","image_url":{"url":"data:image/png;base64,WFhY"}},
						{"type":"image_url","image_url":{"url":"https://cdn/pic.png"}},
						{"type":"image_url","image_url":{"url":""}}
					]
				}
			}]
		}`
		w.Write([]byte(body))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateImage(context.Background(), ImageRequest{Prompt: "x"})
	require.NoError(t, err)
	require.Len(t, resp.Images, 2)
	assert.Equal(t, "WFhY", resp.Images[0].B64JSON)
	assert.Equal(t, "https://cdn/pic.png", resp.Images[1].URL)
}

// Image response with malformed JSON → parse error.
func TestGenerateImageBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not-json`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateImage(context.Background(), ImageRequest{Prompt: "x"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse image response")
}

// Audio with no voice sets default "alloy".
func TestGenerateAudioDefaultVoice(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi"})
	require.NoError(t, err)
	audioConf := got["audio"].(map[string]any)
	assert.Equal(t, "alloy", audioConf["voice"])
}

// Audio response with 4xx returns a descriptive error.
func TestGenerateAudioHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"bad key"}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

// Audio SSE with invalid JSON chunk → silently skipped, valid chunks still captured.
func TestGenerateAudioSkipsInvalidSSELines(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, ": heartbeat\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: not-json\n\n")
		flusher.Flush()
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`+"\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi"})
	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Text)
}

// Audio chunk that's not valid base64 in either std or raw encoding → decode error.
func TestGenerateAudioInvalidBase64Chunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, `data: {"choices":[{"delta":{"audio":{"data":"!!!not-base64!!!"}}}]}`+"\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	_, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "decode audio chunk")
}

// Audio chunk without base64 padding is decoded via RawStdEncoding fallback.
func TestGenerateAudioRawStdBase64Fallback(t *testing.T) {
	raw := []byte("hello-audio")
	unpadded := base64.RawStdEncoding.EncodeToString(raw)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, `data: {"choices":[{"delta":{"audio":{"data":"%s"}}}]}`+"\n\n", unpadded)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi"})
	require.NoError(t, err)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, raw, decoded)
}
