package ai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGenerateAudioSSEOptionalSpace(t *testing.T) {
	for _, prefix := range []string{"data: ", "data:"} {
		t.Run(prefix, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "%s{\"choices\":[{\"delta\":{\"content\":\"hello\",\"audio\":{\"data\":\"QUJD\"}}}]}\n\n%s[DONE]\n\n", prefix, prefix)
			}))
			defer srv.Close()
			p := &OpenRouterMediaProvider{APIKey: "local", BaseURL: srv.URL, Client: srv.Client()}
			p.SeedModelMeta("openai/gpt-audio-mini", []string{"text", "audio"}, []string{"text"})
			got, err := p.GenerateAudio(context.Background(), AudioRequest{Text: "hi", Model: "openai/gpt-audio-mini", Format: "pcm16"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != "hello" || got.Audio == nil || got.Audio.Data != "QUJD" {
				t.Fatalf("lost SSE output: %+v audio=%+v", got, got.Audio)
			}
		})
	}
}
