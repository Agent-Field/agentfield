package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewSessionCommandRegistersSubcommands(t *testing.T) {
	cmd := NewSessionCommand()
	var uses []string
	for _, sub := range cmd.Commands() {
		uses = append(uses, sub.Use)
	}

	require.Contains(t, uses, "start <node>.<session>")
	require.Contains(t, uses, "offer <session_id>")
	require.Contains(t, uses, "tool <session_id> <tool>")
	require.Contains(t, uses, "workflows <session_id>")
}

func TestRunSessionStartPostsPayloadAndWritesResponse(t *testing.T) {
	var gotBody map[string]interface{}
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/session-targets/support.voice/start", r.URL.Path)
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"session_id":"sess-1","offer_url":"/api/v1/session-instances/sess-1/realtime-offer"}`))
	})

	var stdout bytes.Buffer
	err := runSessionStart(context.Background(), "support.voice", &sessionStartOptions{
		provider:     "openai",
		transport:    "webrtc",
		model:        "gpt-realtime-2",
		voice:        "marin",
		outputFormat: "json",
		stdout:       &stdout,
	})

	require.NoError(t, err)
	require.Equal(t, map[string]interface{}{
		"provider":  "openai",
		"transport": "webrtc",
		"model":     "gpt-realtime-2",
		"voice":     "marin",
	}, gotBody)
	require.JSONEq(t, `{"session_id":"sess-1","offer_url":"/api/v1/session-instances/sess-1/realtime-offer"}`, stdout.String())
}

func TestRunSessionStartReturnsStatusErrors(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad session"}`, http.StatusBadRequest)
	})

	err := runSessionStart(context.Background(), "support.voice", &sessionStartOptions{stdout: &bytes.Buffer{}})

	require.ErrorContains(t, err, "session start failed with status 400")
}

func TestRunSessionOfferPostsSDPAndWritesRawAnswer(t *testing.T) {
	var gotBody string
	var gotContentType string
	var gotAPIKey string
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/session-instances/sess-1/realtime-offer", r.URL.Path)
		require.Equal(t, "openai", r.URL.Query().Get("provider"))
		require.Equal(t, "webrtc", r.URL.Query().Get("transport"))
		gotContentType = r.Header.Get("Content-Type")
		gotAPIKey = r.Header.Get("X-API-Key")
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/sdp")
		_, _ = w.Write([]byte("v=0\r\nanswer\r\n"))
	})
	apiKey = "test-key"

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
	require.Equal(t, "test-key", gotAPIKey)
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

func TestRunSessionOfferReturnsStatusErrors(t *testing.T) {
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "provider rejected", http.StatusBadGateway)
	})

	err := runSessionOffer(context.Background(), "sess-1", &sessionOfferOptions{
		provider:  "openai",
		transport: "webrtc",
		sdpSource: "v=0",
		stdout:    &bytes.Buffer{},
	})

	require.ErrorContains(t, err, "session offer failed with status 502")
}

func TestReadSessionSDPFromFileAndInline(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "offer.sdp")
	require.NoError(t, os.WriteFile(path, []byte("v=0\nfile-offer\n"), 0o600))

	got, err := readSessionSDP("@"+path, nil)
	require.NoError(t, err)
	require.Equal(t, "v=0\nfile-offer\n", got)

	got, err = readSessionSDP("v=0\ninline\n", nil)
	require.NoError(t, err)
	require.Equal(t, "v=0\ninline\n", got)
}

func TestReadSessionSDPRejectsBadSources(t *testing.T) {
	dir := t.TempDir()
	emptyPath := filepath.Join(dir, "empty.sdp")
	require.NoError(t, os.WriteFile(emptyPath, []byte(" "), 0o600))

	_, err := readSessionSDP("@", nil)
	require.ErrorContains(t, err, "SDP file path is required")

	_, err = readSessionSDP("@"+filepath.Join(dir, "missing.sdp"), nil)
	require.ErrorContains(t, err, "read SDP file")

	_, err = readSessionSDP("@"+emptyPath, nil)
	require.ErrorContains(t, err, "is empty")

	_, err = readSessionSDP("-", errReader{})
	require.ErrorContains(t, err, "read SDP from stdin")
}

func TestRunSessionToolPostsPayloadAndWritesResponse(t *testing.T) {
	var gotBody map[string]interface{}
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/session-instances/sess-1/tools/resolve", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"execution_id":"exec-1","status":"queued"}`))
	})

	var stdout bytes.Buffer
	err := runSessionTool(context.Background(), "sess-1", "resolve", &sessionToolOptions{
		target:       "support.resolve",
		inputSource:  `{"topic":"billing"}`,
		outputFormat: "pretty",
		stdout:       &stdout,
	})

	require.NoError(t, err)
	require.Equal(t, map[string]interface{}{
		"target": "support.resolve",
		"input":  map[string]interface{}{"topic": "billing"},
	}, gotBody)
	require.Contains(t, stdout.String(), `"execution_id": "exec-1"`)
}

func TestRunSessionToolReadsInputFromStdin(t *testing.T) {
	var gotBody map[string]interface{}
	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	})

	err := runSessionTool(context.Background(), "sess-1", "support.resolve", &sessionToolOptions{
		stdin:  strings.NewReader(`{"topic":"shipping"}`),
		stdout: &bytes.Buffer{},
	})

	require.NoError(t, err)
	require.Equal(t, map[string]interface{}{
		"target": "",
		"input":  map[string]interface{}{"topic": "shipping"},
	}, gotBody)
}

func TestRunSessionToolReturnsInputAndStatusErrors(t *testing.T) {
	err := runSessionTool(context.Background(), "sess-1", "resolve", &sessionToolOptions{
		inputSource: "{",
		stdout:      &bytes.Buffer{},
	})
	require.ErrorContains(t, err, "parse json")

	err = runSessionTool(context.Background(), "sess-1", "resolve", &sessionToolOptions{
		stdin:  strings.NewReader("{"),
		stdout: &bytes.Buffer{},
	})
	require.ErrorContains(t, err, "parse stdin JSON")

	withTriggerTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad tool"}`, http.StatusBadRequest)
	})
	err = runSessionTool(context.Background(), "sess-1", "resolve", &sessionToolOptions{
		stdout: &bytes.Buffer{},
	})
	require.ErrorContains(t, err, "session tool failed with status 400")
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) {
	return 0, errors.New("boom")
}
