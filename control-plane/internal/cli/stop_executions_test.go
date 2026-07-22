package cli

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// executionsQueryServer stands up a control plane that answers the
// /api/v1/agentic/query endpoint with the given running executions and records
// how many times it was called.
func executionsQueryServer(t *testing.T, execs []runningExecution, calls *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls != nil {
			*calls++
		}
		if r.URL.Path != "/api/v1/agentic/query" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["resource"] != "executions" {
			t.Errorf("expected resource=executions, got %v", body["resource"])
		}
		resp := executionsQueryResponse{OK: true}
		resp.Data.Results = execs
		resp.Data.Total = len(execs)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

func twoRunningExecutions() []runningExecution {
	now := time.Now()
	return []runningExecution{
		{ExecutionID: "exec-aaa", AgentNodeID: "swe-af", Status: "running", StartedAt: now.Add(-75 * time.Minute)},
		{ExecutionID: "exec-bbb", AgentNodeID: "swe-af", Status: "running", StartedAt: now.Add(-30 * time.Second)},
	}
}

// Contract: non-TTY stop with running executions warns, includes the count, and
// proceeds (returns true) without prompting.
func TestConfirmStop_NonInteractive_WarnsWithCount(t *testing.T) {
	srv := executionsQueryServer(t, twoRunningExecutions(), nil)
	defer srv.Close()

	var out bytes.Buffer
	stopper := &AgentNodeStopper{
		ServerURL:   srv.URL,
		Interactive: false,
		ErrOut:      &out,
	}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); !proceed {
		t.Fatalf("non-interactive stop must proceed, got proceed=false")
	}

	got := out.String()
	if !strings.Contains(got, "2 running execution") {
		t.Errorf("warning should mention the count of 2, got:\n%s", got)
	}
	if !strings.Contains(got, "exec-aaa") || !strings.Contains(got, "exec-bbb") {
		t.Errorf("warning should list the execution ids, got:\n%s", got)
	}
	if strings.Contains(got, "[y/N]") {
		t.Errorf("non-interactive stop must not prompt, got:\n%s", got)
	}
}

// Contract: --force skips the check entirely — no warning is printed and the
// control plane is never queried.
func TestConfirmStop_Force_NoWarnNoQuery(t *testing.T) {
	calls := 0
	srv := executionsQueryServer(t, twoRunningExecutions(), &calls)
	defer srv.Close()

	var out bytes.Buffer
	stopper := &AgentNodeStopper{
		ServerURL:   srv.URL,
		Force:       true,
		Interactive: false,
		ErrOut:      &out,
	}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); !proceed {
		t.Fatalf("--force stop must proceed, got proceed=false")
	}
	if out.Len() != 0 {
		t.Errorf("--force must not warn, got:\n%s", out.String())
	}
	if calls != 0 {
		t.Errorf("--force must not query the control plane, got %d calls", calls)
	}
}

// Contract: with no running executions the stop proceeds silently.
func TestConfirmStop_NoRunningExecutions_Silent(t *testing.T) {
	srv := executionsQueryServer(t, nil, nil)
	defer srv.Close()

	var out bytes.Buffer
	stopper := &AgentNodeStopper{ServerURL: srv.URL, ErrOut: &out}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); !proceed {
		t.Fatalf("stop with no running executions must proceed")
	}
	if out.Len() != 0 {
		t.Errorf("no executions should produce no warning, got:\n%s", out.String())
	}
}

// Contract: an unreachable control plane notes the failure and proceeds.
func TestConfirmStop_ControlPlaneUnreachable_Proceeds(t *testing.T) {
	var out bytes.Buffer
	stopper := &AgentNodeStopper{
		// A port nothing is listening on.
		ServerURL: "http://127.0.0.1:1",
		ErrOut:    &out,
	}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); !proceed {
		t.Fatalf("unreachable control plane must proceed (best-effort)")
	}
	if !strings.Contains(out.String(), "Could not check") {
		t.Errorf("unreachable control plane should note it, got:\n%s", out.String())
	}
}

// Contract: on a terminal the operator is prompted; a "no" aborts.
func TestConfirmStop_Interactive_DeclineAborts(t *testing.T) {
	srv := executionsQueryServer(t, twoRunningExecutions(), nil)
	defer srv.Close()

	var out bytes.Buffer
	stopper := &AgentNodeStopper{
		ServerURL:   srv.URL,
		Interactive: true,
		In:          strings.NewReader("n\n"),
		ErrOut:      &out,
	}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); proceed {
		t.Fatalf("declining the prompt must abort (proceed=false)")
	}
	if !strings.Contains(out.String(), "[y/N]") {
		t.Errorf("interactive stop should prompt, got:\n%s", out.String())
	}
}

func TestConfirmStop_Interactive_AcceptProceeds(t *testing.T) {
	srv := executionsQueryServer(t, twoRunningExecutions(), nil)
	defer srv.Close()

	stopper := &AgentNodeStopper{
		ServerURL:   srv.URL,
		Interactive: true,
		In:          strings.NewReader("y\n"),
		ErrOut:      io.Discard,
	}

	if proceed := stopper.confirmStopWithRunningExecutions("swe-af"); !proceed {
		t.Fatalf("accepting the prompt must proceed")
	}
}

func TestQueryRunningExecutions_ParsesEnvelope(t *testing.T) {
	srv := executionsQueryServer(t, twoRunningExecutions(), nil)
	defer srv.Close()

	execs, err := queryRunningExecutions(srv.URL, "swe-af")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(execs) != 2 {
		t.Fatalf("expected 2 executions, got %d", len(execs))
	}
	if execs[0].ExecutionID != "exec-aaa" {
		t.Errorf("expected first execution exec-aaa, got %s", execs[0].ExecutionID)
	}
}

func TestQueryRunningExecutions_EmptyServerURL(t *testing.T) {
	if _, err := queryRunningExecutions("", "swe-af"); err == nil {
		t.Fatalf("empty server URL should error")
	}
}

func TestReadAffirmative(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"y\n", true},
		{"Y\n", true},
		{"yes\n", true},
		{"YES\n", true},
		{" y \n", true},
		{"n\n", false},
		{"no\n", false},
		{"\n", false},
		{"", false},
		{"maybe\n", false},
	}
	for _, tc := range cases {
		if got := readAffirmative(strings.NewReader(tc.in)); got != tc.want {
			t.Errorf("readAffirmative(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestFormatExecutionAge(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{-5 * time.Second, "0s"},
		{45 * time.Second, "45s"},
		{90 * time.Second, "1m30s"},
		{75 * time.Minute, "1h15m"},
		{2 * time.Hour, "2h0m"},
	}
	for _, tc := range cases {
		if got := formatExecutionAge(tc.d); got != tc.want {
			t.Errorf("formatExecutionAge(%v) = %q, want %q", tc.d, got, tc.want)
		}
	}
}
