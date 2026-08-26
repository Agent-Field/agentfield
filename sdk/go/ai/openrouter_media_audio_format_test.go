package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Chat-completions audio format support (issue #584).
//
// OpenRouter's chat-completions audio modality only ever delivers pcm16:
// OpenAI rejects a non-pcm16 audio.format while streaming, and the OpenRouter
// gateway rejects an audio completion that is not streamed at all. So pcm16
// (and wav, which is pcm16 wrapped into a RIFF/WAVE container client-side) are
// the only formats this route can serve, and anything else must fail before a
// request goes out.
// =============================================================================

// chatAudioProvider returns a provider wired to srv with the gpt-audio family
// metadata pre-seeded, so GenerateAudio takes the chat-completions path and
// issues no model-metadata request of its own.
func chatAudioProvider(srv *httptest.Server) *OpenRouterMediaProvider {
	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})
	return p
}

// wav is wired as pcm16 over the stream and re-wrapped into a RIFF/WAVE
// container client-side.
func TestChatAudioWAVStreamsPCM16AndWrapsRIFF(t *testing.T) {
	var payload map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		pcm := base64.StdEncoding.EncodeToString(make([]byte, 240))
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"%s\"}}}]}\n\n", pcm)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	resp, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Voice: "nova", Format: "wav",
	})
	require.NoError(t, err)

	assert.Equal(t, true, payload["stream"], "wav is wired as pcm16 and must stream")
	assert.Equal(t, "pcm16", payload["audio"].(map[string]any)["format"])

	require.NotNil(t, resp.Audio)
	assert.Equal(t, "wav", resp.Audio.Format)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	require.Greater(t, len(decoded), 44)
	assert.Equal(t, []byte("RIFF"), decoded[:4])
	assert.Equal(t, []byte("WAVE"), decoded[8:12])
}

// pcm16 goes out on the streaming path with the format untouched.
func TestChatAudioPCM16KeepsStreamingRequestShape(t *testing.T) {
	var payload map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		data := base64.StdEncoding.EncodeToString([]byte("pcmbytes"))
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\",\"audio\":{\"data\":\"%s\"}}}]}\n\n", data)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	resp, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Voice: "nova", Format: "pcm16",
	})
	require.NoError(t, err)

	assert.Equal(t, true, payload["stream"])
	assert.Equal(t, "pcm16", payload["audio"].(map[string]any)["format"])
	assert.Equal(t, "nova", payload["audio"].(map[string]any)["voice"])

	require.NotNil(t, resp.Audio)
	assert.Equal(t, "pcm16", resp.Audio.Format)
	assert.Equal(t, "Hi", resp.Text)
	decoded, err := base64.StdEncoding.DecodeString(resp.Audio.Data)
	require.NoError(t, err)
	assert.Equal(t, "pcmbytes", string(decoded))
}

// mp3 / flac / opus are refused locally, naming the format and the one the
// provider actually delivers — and without spending a request on a 400.
func TestChatAudioUnsupportedFormatIsRefusedBeforeAnyRequest(t *testing.T) {
	for _, format := range []string{"mp3", "flac", "opus"} {
		t.Run(format, func(t *testing.T) {
			var requests int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&requests, 1)
				t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
			}))
			defer srv.Close()

			_, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
				Text: "hi", Model: "openai/gpt-audio-mini", Voice: "nova", Format: format,
			})
			require.Error(t, err)
			assert.Contains(t, err.Error(), format, "the error must name the requested format")
			assert.Contains(t, err.Error(), "pcm16", "the error must name the format OpenRouter delivers")
			assert.Contains(t, err.Error(), "wav", "the error must point at the wav alternative")
			assert.Equal(t, int32(0), atomic.LoadInt32(&requests),
				"an unsupported format must not reach the provider")
		})
	}
}
