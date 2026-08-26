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
// Chat-completions audio: OpenRouter only streams pcm16 deltas, so every other
// wire format must be requested with stream=false and read from the plain JSON
// response body (issue #584).
// =============================================================================

// chatAudioProvider returns a provider wired to srv with the gpt-audio family
// metadata pre-seeded, so GenerateAudio takes the chat-completions path.
func chatAudioProvider(srv *httptest.Server) *OpenRouterMediaProvider {
	p := &OpenRouterMediaProvider{APIKey: "k", BaseURL: srv.URL, Client: srv.Client()}
	p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})
	return p
}

// wav is wired as pcm16 over a stream and re-wrapped into a RIFF/WAVE container
// client-side.
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

// mp3 / flac / opus are sent with stream=false and parsed out of the single
// JSON document.
func TestChatAudioNonPCM16UsesNonStreamJSON(t *testing.T) {
	for _, format := range []string{"mp3", "flac", "opus"} {
		t.Run(format, func(t *testing.T) {
			data := base64.StdEncoding.EncodeToString([]byte("audio-" + format))
			var payload map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/chat/completions", r.URL.Path)
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w,
					`{"choices":[{"message":{"role":"assistant","content":null,`+
						`"audio":{"data":"%s","transcript":"spoken words"}}}]}`, data)
			}))
			defer srv.Close()

			resp, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
				Text: "hi", Model: "openai/gpt-audio-mini", Voice: "nova", Format: format,
			})
			require.NoError(t, err)

			assert.Equal(t, false, payload["stream"], "%s must not be streamed", format)
			assert.Equal(t, format, payload["audio"].(map[string]any)["format"])

			require.NotNil(t, resp.Audio)
			assert.Equal(t, format, resp.Audio.Format)
			assert.Equal(t, data, resp.Audio.Data)
			assert.Equal(t, "spoken words", resp.Text)
		})
	}
}

// A non-stream response that carries plain string content and no transcript
// still surfaces the text.
func TestChatAudioNonStreamFallsBackToMessageContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"plain text","audio":{"data":"YQ=="}}}]}`)
	}))
	defer srv.Close()

	resp, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.NoError(t, err)
	assert.Equal(t, "plain text", resp.Text)
	assert.Equal(t, "YQ==", resp.Audio.Data)
}

// An upstream failure on the non-stream path surfaces the status code and an
// excerpt of the response body.
func TestChatAudioNonStreamHTTPErrorCarriesStatusAndBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"audio.format does not support mp3"}}`))
	}))
	defer srv.Close()

	_, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "400")
	assert.Contains(t, err.Error(), "audio.format does not support mp3")
}

// A non-JSON body on the non-stream path is reported as a parse failure rather
// than silently returning empty audio.
func TestChatAudioNonStreamInvalidJSONErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html>gateway timeout</html>"))
	}))
	defer srv.Close()

	_, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "flac",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse audio response")
}

// An empty completion yields empty audio without an error — the SSE path
// behaves the same way for a stream with no audio deltas.
func TestChatAudioNonStreamEmptyChoicesReturnsEmptyAudio(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	resp, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "opus",
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Audio)
	assert.Empty(t, resp.Audio.Data)
	assert.Equal(t, "opus", resp.Audio.Format)
	assert.Empty(t, resp.Text)
}

// A declared Content-Length above the cap is refused before the body is read.
func TestChatAudioNonStreamRejectsOversizedDeclaredBody(t *testing.T) {
	body := `{"choices":[{"message":{"audio":{"data":"` + strings.Repeat("A", 128) + `"}}}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body)) // httptest sets Content-Length for us
	}))
	defer srv.Close()

	prev := maxAudioResponseBytes
	maxAudioResponseBytes = 32
	defer func() { maxAudioResponseBytes = prev }()

	_, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "audio response too large")
}

// A chunked body (no Content-Length) that runs past the cap is refused too,
// so an oversized response is never buffered without bound.
func TestChatAudioNonStreamRejectsOversizedChunkedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"audio":{"data":"`))
		flusher.Flush() // forces chunked encoding: Content-Length is unknown
		for i := 0; i < 20; i++ {
			_, _ = w.Write([]byte(strings.Repeat("A", 64)))
			flusher.Flush()
		}
		_, _ = w.Write([]byte(`"}}}]}`))
	}))
	defer srv.Close()

	prev := maxAudioResponseBytes
	maxAudioResponseBytes = 64
	defer func() { maxAudioResponseBytes = prev }()

	_, err := chatAudioProvider(srv).GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "audio response too large")
}

// A non-streaming request returns nothing until the whole clip is synthesised,
// so it must not inherit the provider's short whole-request client timeout —
// the 60s default aborts a paragraph of mp3 mid-synthesis.
func TestChatAudioNonStreamOutlastsShortClientTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond) // synthesis takes longer than the client timeout
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"audio":{"data":"bXAz","transcript":"spoken"}}}]}`))
	}))
	defer srv.Close()

	p := chatAudioProvider(srv)
	p.Client = &http.Client{Transport: srv.Client().Transport, Timeout: 30 * time.Millisecond}

	resp, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Audio)
	assert.Equal(t, "bXAz", resp.Audio.Data)
	assert.Equal(t, "spoken", resp.Text)
}

// Raising the cap is scoped to the non-streaming request: the streaming path
// still runs on the caller's configured client timeout.
func TestChatAudioStreamKeepsShortClientTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := chatAudioProvider(srv)
	p.Client = &http.Client{Transport: srv.Client().Transport, Timeout: 30 * time.Millisecond}

	_, err := p.GenerateAudio(context.Background(), AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "pcm16",
	})
	require.Error(t, err)
}

// The longer budget is a client-timeout floor, not an escape from the caller:
// a cancelled context still aborts the non-streaming request.
func TestChatAudioNonStreamHonoursContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, err := chatAudioProvider(srv).GenerateAudio(ctx, AudioRequest{
		Text: "hi", Model: "openai/gpt-audio-mini", Format: "mp3",
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}
