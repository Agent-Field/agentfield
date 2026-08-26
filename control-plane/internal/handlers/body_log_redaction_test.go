package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// captureHandlerLogs points the package logger at an in-memory buffer at the
// requested level and restores the previous logger afterwards.
func captureHandlerLogs(t *testing.T, level zerolog.Level) *bytes.Buffer {
	t.Helper()
	prev := logger.Logger
	var buf bytes.Buffer
	logger.Logger = zerolog.New(&buf).With().Timestamp().Logger().Level(level)
	t.Cleanup(func() { logger.Logger = prev })
	return &buf
}

// withRedaction sets the payload-redaction switch for the duration of a test.
func withRedaction(t *testing.T, redact bool) {
	t.Helper()
	prev := defaultRedactPayloads
	SetRedactPayloads(redact)
	t.Cleanup(func() { SetRedactPayloads(prev) })
}

// findLogEntry returns the first structured log line whose message matches.
func findLogEntry(t *testing.T, buf *bytes.Buffer, message string) map[string]interface{} {
	t.Helper()
	for _, line := range bytes.Split(bytes.TrimSpace(buf.Bytes()), []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var entry map[string]interface{}
		if err := json.Unmarshal(line, &entry); err != nil {
			continue
		}
		if entry["message"] == message {
			return entry
		}
	}
	t.Fatalf("no log line with message %q in:\n%s", message, buf.String())
	return nil
}

// secretBody is a plausible agent response that is not valid JSON. It carries a
// recognisable marker so a test can assert it never reaches the log.
const secretBody = `<html><body>customer-ssn-078-05-1120 order#4711</body></html>`

// runReasonerAgainstAgent drives ExecuteReasonerHandler against an agent that
// replies with the given content type and body.
func runReasonerAgainstAgent(t *testing.T, contentType, body string) *httptest.ResponseRecorder {
	t.Helper()
	gin.SetMode(gin.TestMode)

	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(agent.Close)

	store := newReasonerHandlerStorage(newReasonerAgent(agent.URL))
	router := gin.New()
	router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

// TestUndecodableAgentResponseIsNotLoggedWhenRedacting covers contract C7: with
// redaction on (the default) an agent response that fails to decode is
// described, not reproduced, in the log.
func TestUndecodableAgentResponseIsNotLoggedWhenRedacting(t *testing.T) {
	withRedaction(t, true)
	buf := captureHandlerLogs(t, zerolog.DebugLevel)

	resp := runReasonerAgainstAgent(t, "text/html", secretBody)
	require.Equal(t, http.StatusInternalServerError, resp.Code)

	require.NotContains(t, buf.String(), "customer-ssn", "agent response body must not reach the log")
	require.NotContains(t, buf.String(), "order#4711", "agent response body must not reach the log")

	entry := findLogEntry(t, buf, "failed to decode agent response")
	require.Equal(t, "text/html", entry["content_type"])
	require.Equal(t, float64(len(secretBody)), entry["body_bytes"])
	require.Equal(t, true, entry["body_redacted"])
	require.Len(t, entry["body_sha256"], 8, "digest prefix should be 8 hex characters")
}

// TestUndecodableAgentResponseIsLoggedWhenRedactionDisabled covers contract C8:
// operators who explicitly turn redaction off still get the body preview.
func TestUndecodableAgentResponseIsLoggedWhenRedactionDisabled(t *testing.T) {
	withRedaction(t, false)
	buf := captureHandlerLogs(t, zerolog.DebugLevel)

	resp := runReasonerAgainstAgent(t, "text/html", secretBody)
	require.Equal(t, http.StatusInternalServerError, resp.Code)

	entry := findLogEntry(t, buf, "failed to decode agent response")
	require.Equal(t, secretBody, entry["body"])
	require.Nil(t, entry["body_redacted"])
}

// callServerlessAgent drives callAgent against a serverless-deployed agent that
// replies with the given content type and body.
func callServerlessAgent(t *testing.T, contentType, body string) {
	t.Helper()
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(agentServer.Close)

	agent := &types.AgentNode{
		ID:             "node-1",
		BaseURL:        agentServer.URL,
		DeploymentType: "serverless",
		Reasoners:      []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	controller := newExecutionController(newTestExecutionStorage(agent), nil, nil, 90*time.Second, "")
	plan := &preparedExecution{
		exec:        &types.Execution{ExecutionID: "test-exec", RunID: "test-run"},
		requestBody: []byte(`{"input":{}}`),
		agent:       agent,
		target:      &parsedTarget{NodeID: "node-1", TargetName: "reasoner-a"},
	}

	_, _, _, err := controller.callAgent(context.Background(), plan)
	require.NoError(t, err)
}

// TestServerlessResponseIsNotLoggedWhenRedacting covers contract C9 for the
// serverless debug line with redaction on.
func TestServerlessResponseIsNotLoggedWhenRedacting(t *testing.T) {
	withRedaction(t, true)
	buf := captureHandlerLogs(t, zerolog.DebugLevel)

	callServerlessAgent(t, "application/json", `{"secret":"customer-ssn-078-05-1120"}`)

	require.NotContains(t, buf.String(), "customer-ssn", "serverless response body must not reach the log")

	entry := findLogEntry(t, buf, "serverless response")
	require.Equal(t, "application/json", entry["content_type"])
	require.Equal(t, float64(len(`{"secret":"customer-ssn-078-05-1120"}`)), entry["body_bytes"])
	require.Equal(t, true, entry["body_redacted"])
	require.Len(t, entry["body_sha256"], 8)
}

// TestServerlessResponseIsLoggedWhenRedactionDisabled covers contract C9 for
// the serverless debug line with redaction off.
func TestServerlessResponseIsLoggedWhenRedactionDisabled(t *testing.T) {
	withRedaction(t, false)
	buf := captureHandlerLogs(t, zerolog.DebugLevel)

	callServerlessAgent(t, "application/json", `{"secret":"customer-ssn-078-05-1120"}`)

	entry := findLogEntry(t, buf, "serverless response")
	require.Equal(t, `{"secret":"customer-ssn-078-05-1120"}`, entry["body"])
}

// digestFor renders one redacted body annotation and returns its digest prefix.
func digestFor(t *testing.T, body string) string {
	t.Helper()
	buf := captureHandlerLogs(t, zerolog.DebugLevel)
	annotateBodyForLog(logger.Logger.Debug(), "text/plain", []byte(body)).Msg("probe")
	return findLogEntry(t, buf, "probe")["body_sha256"].(string)
}

// TestRedactedDigestIdentifiesTheBody covers contract C10: the digest prefix is
// stable for the same body and different for a different one, so operators can
// still tell "same failure again" from "a new failure".
func TestRedactedDigestIdentifiesTheBody(t *testing.T) {
	withRedaction(t, true)

	first := digestFor(t, "alpha response")
	again := digestFor(t, "alpha response")
	other := digestFor(t, "beta response")

	require.Equal(t, first, again, "the same body must produce the same digest prefix")
	require.NotEqual(t, first, other, "different bodies must produce different digest prefixes")
}

// TestAnnotateBodyForLogOnDisabledEvent guards the no-op path: annotating a
// suppressed event neither panics nor emits anything.
func TestAnnotateBodyForLogOnDisabledEvent(t *testing.T) {
	withRedaction(t, true)
	buf := captureHandlerLogs(t, zerolog.InfoLevel)

	require.NotPanics(t, func() {
		annotateBodyForLog(logger.Logger.Debug(), "text/plain", []byte("hidden")).Msg("probe")
	})
	require.Empty(t, buf.String())
}

// TestAnnotateBodyForLogOmitsUnknownContentType checks the empty-content-type
// edge case: the field is left out rather than logged blank.
func TestAnnotateBodyForLogOmitsUnknownContentType(t *testing.T) {
	withRedaction(t, true)
	buf := captureHandlerLogs(t, zerolog.DebugLevel)

	annotateBodyForLog(logger.Logger.Debug(), "", nil).Msg("probe")

	entry := findLogEntry(t, buf, "probe")
	require.Nil(t, entry["content_type"])
	require.Equal(t, float64(0), entry["body_bytes"])
	require.Len(t, entry["body_sha256"], 8)
}
