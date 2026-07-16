package packages

import (
	"fmt"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestHealthNodeID(t *testing.T) {
	cases := []struct {
		body string
		want string
	}{
		{`{"status":"healthy","node_id":"swe-planner"}`, "swe-planner"},
		{`{"status":"healthy"}`, ""},
		{`not json`, ""},
		{``, ""},
	}
	for _, c := range cases {
		if got := HealthNodeID([]byte(c.body)); got != c.want {
			t.Errorf("HealthNodeID(%q) = %q, want %q", c.body, got, c.want)
		}
	}
}

func TestNodeIDsEquivalent(t *testing.T) {
	if !NodeIDsEquivalent("swe-planner", "swe_planner") {
		t.Error("hyphen/underscore drift should be equivalent")
	}
	if !NodeIDsEquivalent("Agent", "agent") {
		t.Error("case drift should be equivalent")
	}
	if NodeIDsEquivalent("swe-planner", "smoke-agent") {
		t.Error("different nodes must not be equivalent")
	}
}

// Contract: a healthy response from a DIFFERENT node on the expected port is
// not readiness — it is a squatter, and the error must say so. This is the
// live-caught Windows failure where the port probe misses an existing
// listener and a second agent's readiness poll hits the first agent.
func TestWaitForAgentNode_RejectsSquatter(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"status":"healthy","node_id":"squatter-agent"}`)
	})
	server := &http.Server{Handler: mux}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })

	ar := &AgentNodeRunner{}
	err = ar.waitForAgentNode(port, "/health", "real-agent", 1200*time.Millisecond)
	if err == nil {
		t.Fatal("expected squatter rejection, got success")
	}
	if !strings.Contains(err.Error(), "squatter-agent") || !strings.Contains(err.Error(), "real-agent") {
		t.Fatalf("error should name both nodes, got: %v", err)
	}
}

// Contract: a matching node_id (including hyphen/underscore drift) and a
// payload without node_id both count as ready — custom healthchecks are not
// required to identify themselves.
func TestWaitForAgentNode_AcceptsMatchAndAnonymous(t *testing.T) {
	for _, body := range []string{
		`{"status":"healthy","node_id":"real_agent"}`,
		`{"status":"healthy"}`,
	} {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		port := listener.Addr().(*net.TCPAddr).Port
		payload := body
		mux := http.NewServeMux()
		mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
			fmt.Fprint(w, payload)
		})
		server := &http.Server{Handler: mux}
		go func() { _ = server.Serve(listener) }()

		ar := &AgentNodeRunner{}
		err = ar.waitForAgentNode(port, "/health", "real-agent", 2*time.Second)
		_ = server.Close()
		if err != nil {
			t.Fatalf("body %q: expected ready, got %v", payload, err)
		}
	}
}

// Contract: a port with an active listener is not available, even when a
// probe bind would succeed (the Windows no-SO_EXCLUSIVEADDRUSE case) — the
// dial probe must catch it.
func TestIsPortAvailable_DetectsActiveListener(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port

	ar := &AgentNodeRunner{}
	if ar.isPortAvailable(port) {
		t.Fatalf("port %d has an active listener but was reported available", port)
	}
}
