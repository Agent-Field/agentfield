package cli

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// withUnreachableServer points the CLI at a real ephemeral port that has just
// been closed, so every control-plane request fails fast with "connection
// refused" at the transport layer — deterministic across environments (unlike
// dialing a privileged port, which some hosts hang on rather than refuse).
func withUnreachableServer(t *testing.T) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := server.URL
	server.Close() // frees the port; subsequent dials get connection refused

	originalServerURL := serverURL
	originalAPIKey := apiKey
	originalTimeout := requestTimeout
	serverURL = url
	apiKey = ""
	requestTimeout = 2
	t.Cleanup(func() {
		serverURL = originalServerURL
		apiKey = originalAPIKey
		requestTimeout = originalTimeout
	})
}

func TestControlPlaneUnreachableErrorMentionsServer(t *testing.T) {
	withUnreachableServer(t)
	err := controlPlaneUnreachableError(context.DeadlineExceeded)
	require.ErrorContains(t, err, "Control plane not reachable at "+GetServerURL())
	require.ErrorContains(t, err, "af server")
	require.ErrorContains(t, err, "desktop app")
}

// TestUnreachableControlPlaneHintAcrossCommands is the contract test: when the
// control plane can't be reached, call/ls/tail/wait all surface the shared
// "start it with `af server`" hint instead of a bare Go dial error.
func TestUnreachableControlPlaneHintAcrossCommands(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cases := []struct {
		name string
		run  func() error
	}{
		{
			name: "call",
			run: func() error {
				return runCall(ctx, "node.echo", &callOptions{
					inputSource:  `{"message":"hi"}`,
					async:        true,
					outputFormat: "json",
					stdin:        bytes.NewBuffer(nil),
					stdout:       bytes.NewBuffer(nil),
					stderr:       bytes.NewBuffer(nil),
				})
			},
		},
		{
			name: "ls",
			run: func() error {
				return runReasonerList(ctx, "", &lsOptions{outputFormat: "json", stdout: bytes.NewBuffer(nil)})
			},
		},
		{
			name: "tail",
			run: func() error {
				return streamExecutionEvents(ctx, "run-1", 0, "json", bytes.NewBuffer(nil))
			},
		},
		{
			name: "wait",
			run: func() error {
				return runWait(ctx, "run-1", &waitOptions{
					timeout:      2 * time.Second,
					pollInterval: 5 * time.Millisecond,
					outputFormat: "json",
					stdout:       bytes.NewBuffer(nil),
					stderr:       bytes.NewBuffer(nil),
				})
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			withUnreachableServer(t)
			err := tc.run()
			require.Error(t, err)
			require.ErrorContains(t, err, "af server")
		})
	}
}

// TestMakeRequestCancelledContextPassthrough pins that a request failing because
// the caller cancelled the context is surfaced as-is — NOT dressed up as an
// unreachable-control-plane error, which would mislead a harness that
// deliberately cancelled (Ctrl-C / its own timeout).
func TestMakeRequestCancelledContextPassthrough(t *testing.T) {
	withUnreachableServer(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := makeRequest(ctx, http.MethodGet, "/api/v1/agentic/run/x", nil, "application/json")
	require.Error(t, err)
	require.NotContains(t, err.Error(), "af server")
}
