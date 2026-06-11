package cli

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRunSessionOfferPostsSDPAndWritesRawAnswer(t *testing.T) {
	var gotBody string
	var gotContentType string
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/sessions/sess-1/realtime-offer", r.URL.Path)
		require.Equal(t, "openai", r.URL.Query().Get("provider"))
		require.Equal(t, "webrtc", r.URL.Query().Get("transport"))
		gotContentType = r.Header.Get("Content-Type")
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/sdp")
		_, _ = w.Write([]byte("v=0\r\nanswer\r\n"))
	})

	var stdout bytes.Buffer
	err := runSessionOffer(context.Background(), "sess-1", &sessionOfferOptions{
		provider:     "openai",
		transport:    "webrtc",
		sdpSource:    "v=0\r\noffer\r\n",
		outputFormat: "raw",
		stdout:       &stdout,
	})
	require.NoError(t, err)
	require.Equal(t, "application/sdp", gotContentType)
	require.Equal(t, "v=0\r\noffer\r\n", gotBody)
	require.Equal(t, "v=0\r\nanswer\r\n", stdout.String())
}

func TestRunSessionOfferReadsSDPFromStdinAndCanWrapJSON(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, "v=0\nstdin-offer\n", string(body))
		w.Header().Set("Content-Type", "application/sdp")
		_, _ = w.Write([]byte("v=0\nstdin-answer\n"))
	})

	var stdout bytes.Buffer
	err := runSessionOffer(context.Background(), "sess-stdin", &sessionOfferOptions{
		provider:     "openai",
		transport:    "webrtc",
		outputFormat: "json",
		stdin:        strings.NewReader("v=0\nstdin-offer\n"),
		stdout:       &stdout,
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"answer_sdp":"v=0\nstdin-answer\n"}`, stdout.String())
}

func TestRunSessionOfferRequiresSDP(t *testing.T) {
	var stdout bytes.Buffer
	err := runSessionOffer(context.Background(), "sess-empty", &sessionOfferOptions{
		provider:  "openai",
		transport: "webrtc",
		stdin:     strings.NewReader(" "),
		stdout:    &stdout,
	})
	require.ErrorContains(t, err, "SDP offer required")
}
