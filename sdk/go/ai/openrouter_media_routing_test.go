package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// fetchModelMeta — happy path + cache reuse + network failures
// =============================================================================

func TestFetchModelMetaCachesResult(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		assert.True(t, strings.HasSuffix(r.URL.Path, "/models/openai/gpt-audio-mini/endpoints"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"id":"openai/gpt-audio-mini","architecture":{"output_modalities":["text","audio"],"input_modalities":["text"]}}}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}

	meta1 := p.fetchModelMeta(context.Background(), "openrouter/openai/gpt-audio-mini")
	meta2 := p.fetchModelMeta(context.Background(), "openai/gpt-audio-mini") // same model, no prefix

	assert.Equal(t, 1, calls, "cache should prevent second HTTP call")
	assert.Equal(t, []string{"text", "audio"}, meta1.OutputModalities)
	assert.Equal(t, meta1, meta2)
}

func TestFetchModelMetaReturnsEmptyOnHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	meta := p.fetchModelMeta(context.Background(), "unknown/model")
	assert.Empty(t, meta.OutputModalities)
	assert.Empty(t, meta.InputModalities)
}

func TestFetchModelMetaReturnsEmptyOnMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not-json`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	meta := p.fetchModelMeta(context.Background(), "x/y")
	assert.Empty(t, meta.OutputModalities)
}

func TestFetchModelMetaTriggersAutoRoutingViaSpeechEndpoint(t *testing.T) {
	// Server handles BOTH the metadata GET and the /audio/speech POST so the
	// provider can self-discover routing end-to-end.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/models/hexgrad/kokoro-82m/endpoints"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":{"id":"hexgrad/kokoro-82m","architecture":{"output_modalities":["speech"],"input_modalities":["text"]}}}`))
		case r.URL.Path == "/audio/speech":
			w.Header().Set("Content-Type", "audio/pcm")
			_, _ = w.Write(make([]byte, 240)) // raw PCM
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:   "hi",
		Model:  "hexgrad/kokoro-82m",
		Voice:  "af_bella",
		Format: "wav",
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Audio)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, []byte("RIFF"), decoded[:4])
}

// =============================================================================
// generateAudioViaSpeechEndpoint — non-WAV format, error paths
// =============================================================================

func TestGenerateAudioSpeechMP3PassthroughNoWavWrap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/audio/speech", r.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, "mp3", body["response_format"])
		assert.Equal(t, 1.5, body["speed"])
		assert.Equal(t, "fr-FR", body["language"]) // from Extra
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("FAKE_MP3_BYTES_ABCDEF"))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("hexgrad/kokoro-82m", []string{"speech"}, []string{"text"})

	speed := 1.5
	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text:   "bonjour",
		Model:  "hexgrad/kokoro-82m",
		Voice:  "af_bella",
		Format: "mp3",
		Speed:  &speed,
		Extra:  map[string]any{"language": "fr-FR"},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Audio)
	assert.Equal(t, "mp3", resp.Audio.Format)
	// MP3 is returned as-is (no WAV wrap)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, "FAKE_MP3_BYTES_ABCDEF", string(decoded))
}

func TestGenerateAudioSpeechErrorBubblesUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"bad key"}`))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("hexgrad/kokoro-82m", []string{"speech"}, []string{"text"})

	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "x", Model: "hexgrad/kokoro-82m", Voice: "af_bella", Format: "wav",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "audio/speech error")
	assert.Contains(t, err.Error(), "401")
}

func TestGenerateAudioSpeechDefaultsVoiceToAlloy(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&seen)
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("x"))
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("openai/gpt-4o-mini-tts", []string{"speech"}, []string{"text"})
	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-4o-mini-tts", Format: "mp3",
	})
	require.NoError(t, err)
	assert.Equal(t, "alloy", seen["voice"])
}

// =============================================================================
// Video — first/last frame + auth-aware download
// =============================================================================

func TestGenerateVideoSendsFrameImagesAndImageURL(t *testing.T) {
	var submit map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/videos":
			require.NoError(t, json.NewDecoder(r.Body).Decode(&submit))
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"id":"jobABC"}`))
		case "/videos/jobABC":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"jobABC","status":"completed",
				"unsigned_urls":["` + r.Host + `/blob/jobABC"],
				"usage":{"cost":0.42}
			}`))
		}
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}

	resp, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt:   "test",
		Model:    "openrouter/google/veo-3.1-lite",
		Duration: 4,
		ImageURL: "https://example.com/seed.jpg",
		FrameImages: []map[string]any{
			{"type": "image_url", "image_url": map[string]string{"url": "https://x/first.jpg"}, "frame_type": "first_frame"},
			{"type": "image_url", "image_url": map[string]string{"url": "https://x/last.jpg"}, "frame_type": "last_frame"},
		},
		InputReferences: []map[string]any{
			{"type": "image_url", "image_url": map[string]string{"url": "https://x/ref.jpg"}},
		},
		Extra:        map[string]any{"personGeneration": "allow_all"},
		PollInterval: 10 * time.Millisecond,
		Timeout:      30 * time.Second,
	})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Videos, 1)
	assert.Equal(t, 0.42, resp.Videos[0].CostUSD)

	// Verify submit payload included our params.
	assert.Equal(t, "google/veo-3.1-lite", submit["model"])
	assert.Equal(t, "https://example.com/seed.jpg", submit["image_url"])
	frames := submit["frame_images"].([]any)
	require.Len(t, frames, 2)
	assert.Equal(t, "first_frame", frames[0].(map[string]any)["frame_type"])
	assert.Equal(t, "last_frame", frames[1].(map[string]any)["frame_type"])
	assert.Equal(t, "allow_all", submit["personGeneration"])
}

func TestGenerateVideoDownloadsWithoutAuthFromNonOpenRouterHost(t *testing.T) {
	// Use a fixed URL string captured after server start so the poll response
	// can embed an absolute http:// URL.
	var apiSrv *httptest.Server
	apiSrv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/videos":
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"id":"job1"}`))
		case "/videos/job1":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"id":"job1","status":"completed","unsigned_urls":["%s/blob/job1"],"usage":{"cost":0.1}}`,
				apiSrv.URL)))
		case "/blob/job1":
			// Non-openrouter host → no Authorization header.
			assert.Empty(t, r.Header.Get("Authorization"))
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("FAKE_MP4"))
		}
	}))
	defer apiSrv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: apiSrv.URL, Client: apiSrv.Client()}
	resp, err := p.GenerateVideo(context.Background(), VideoRequest{
		Prompt: "t", Model: "x/y",
		PollInterval: 10 * time.Millisecond, Timeout: 30 * time.Second,
	})
	require.NoError(t, err)
	require.Len(t, resp.Videos, 1)
	decoded, err := base64.StdEncoding.DecodeString(resp.Videos[0].Data)
	require.NoError(t, err)
	assert.Equal(t, "FAKE_MP4", string(decoded))
}

// =============================================================================
// Image — ImageURLs reference images + image_config conversion
// =============================================================================

func TestGenerateImageWithReferenceImagesAndConfig(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&captured))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":null,"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QUJD"}}]}}]}`))
	}))
	defer srv.Close()

	strength := 0.7
	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	resp, err := p.GenerateImage(context.Background(), ImageRequest{
		Prompt:    "a fox in watercolor",
		Model:     "openrouter/x-ai/grok-imagine-image-quality",
		ImageURLs: []string{"https://x/ref1.png", "https://x/ref2.png"},
		ImageConfig: &ImageConfig{
			AspectRatio: "16:9",
			Strength:    &strength,
			RgbColors:   [][3]int{{255, 100, 50}},
		},
		Extra: map[string]any{"high_quality": true},
	})
	require.NoError(t, err)
	require.Len(t, resp.Images, 1)
	assert.Equal(t, "QUJD", resp.Images[0].B64JSON)

	// Modalities sent as ["image"] only.
	mods := captured["modalities"].([]any)
	require.Len(t, mods, 1)
	assert.Equal(t, "image", mods[0])

	// User content is a multi-part array (text + 2 image_url entries).
	messages := captured["messages"].([]any)
	userMsg := messages[0].(map[string]any)
	content := userMsg["content"].([]any)
	require.Len(t, content, 3)
	assert.Equal(t, "text", content[0].(map[string]any)["type"])
	assert.Equal(t, "image_url", content[1].(map[string]any)["type"])
	assert.Equal(t, "image_url", content[2].(map[string]any)["type"])

	// Extra passthrough.
	assert.Equal(t, true, captured["high_quality"])
}

// =============================================================================
// wrapPCM16AsWAV — header correctness
// =============================================================================

func TestWrapPCM16AsWAVHeader(t *testing.T) {
	pcm := make([]byte, 240) // 240 bytes = 120 16-bit samples = 5ms @ 24kHz
	wav := wrapPCM16AsWAV(pcm, 24000)

	assert.Equal(t, []byte("RIFF"), wav[:4])
	assert.Equal(t, []byte("WAVE"), wav[8:12])
	assert.Equal(t, []byte("fmt "), wav[12:16])
	assert.Equal(t, []byte("data"), wav[36:40])
	// total file size = 44 (header) + 240 (data)
	assert.Len(t, wav, 44+240)
}

// =============================================================================
// SeedModelMeta — chat-audio path verification on a separate model
// =============================================================================

func TestSeedModelMetaRoutesToChatCompletions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})

	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.NoError(t, err)
}
