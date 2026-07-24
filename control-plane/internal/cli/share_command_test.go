package cli

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/share"
)

func TestRunShareDemoWritesRedactedHTML(t *testing.T) {
	out := filepath.Join(t.TempDir(), "demo.html")
	err := runShare(context.Background(), "", &shareOptions{
		output: out,
		demo:   true,
		redact: true,
		title:  "Demo Review",
	})
	if err != nil {
		t.Fatalf("runShare returned error: %v", err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("expected share artifact to be written: %v", err)
	}
	text := string(body)
	if !strings.Contains(text, "Demo Review") || !strings.Contains(text, "[redacted]") {
		t.Fatalf("artifact did not include title/redaction: %s", text)
	}
}

func TestRunShareFetchesRunFromControlPlane(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/ui/v1/workflows/run-remote/dag":
			_, _ = fmt.Fprint(w, `{
				"root_workflow_id":"run-remote",
				"workflow_status":"succeeded",
				"timeline":[{
					"execution_id":"exec-root",
					"agent_node_id":"orchestrator",
					"reasoner_id":"run_audit",
					"status":"succeeded",
					"started_at":"2026-04-08T09:00:00Z",
					"completed_at":"2026-04-08T09:00:02Z",
					"duration_ms":2000
				},{
					"execution_id":"exec-child",
					"agent_node_id":"worker",
					"reasoner_id":"inspect",
					"status":"failed",
					"started_at":"2026-04-08T09:00:01Z",
					"parent_execution_id":"exec-root"
				}]
			}`)
		case "/api/ui/v1/executions/exec-root/details":
			_, _ = fmt.Fprint(w, `{"input_data":{"target":"acme"},"output_data":{"ok":true}}`)
		case "/api/ui/v1/executions/exec-child/details":
			_, _ = fmt.Fprint(w, `{"input_data":{"url":"/admin"},"error_message":"blocked"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	oldServer := serverURL
	serverURL = server.URL
	defer func() { serverURL = oldServer }()

	out := filepath.Join(t.TempDir(), "remote.html")
	err := runShare(context.Background(), "run-remote", &shareOptions{output: out})
	if err != nil {
		t.Fatalf("runShare returned error: %v", err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("expected share artifact to be written: %v", err)
	}
	text := string(body)
	for _, want := range []string{"run of orchestrator.run_audit", "exec-child", "blocked", "acme"} {
		if !strings.Contains(text, want) {
			t.Fatalf("artifact missing %q: %s", want, text)
		}
	}
}

func TestShareHelpers(t *testing.T) {
	duration := int64(500)
	nodes := []share.BundleNode{
		{ID: "child", Agent: "worker", Func: "step", StartedAt: "2026-04-08T09:00:01Z", DurationMS: &duration},
		{ID: "root", Agent: "orchestrator", Func: "run", StartedAt: "2026-04-08T09:00:00Z", EndedAt: "2026-04-08T09:00:03Z"},
	}
	bundle := &share.Bundle{
		Nodes: nodes,
		Edges: []share.BundleEdge{{From: "root", To: "child"}},
	}
	if got := deriveTitle(bundle); got != "run of orchestrator.run" {
		t.Fatalf("deriveTitle = %q", got)
	}
	if got := computeWallClock(nodes); got != 3000 {
		t.Fatalf("computeWallClock = %d", got)
	}
	if got := rawJSONPreview([]byte(`{"b":2}`)); !strings.Contains(got, "\n") || !strings.Contains(got, `"b": 2`) {
		t.Fatalf("rawJSONPreview did not pretty-print JSON: %q", got)
	}
	if got := sanitizeFilename("../run:a b"); got != "run-a-b" {
		t.Fatalf("sanitizeFilename = %q", got)
	}
}

func TestPublishBundleUsesConfiguredShareHost(t *testing.T) {
	var sawAPIKey bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/shares" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		sawAPIKey = r.Header.Get("X-API-Key") == "test-key"
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"url":"https://share.example/s/abc"}`))
	}))
	defer server.Close()

	t.Setenv("AGENTFIELD_SHARE_URL", server.URL)
	t.Setenv("AGENTFIELD_API_KEY", "test-key")
	err := publishBundle(context.Background(), &share.Bundle{Version: share.BundleVersion, WorkflowID: "run-1"})
	if err != nil {
		t.Fatalf("publishBundle returned error: %v", err)
	}
	if !sawAPIKey {
		t.Fatalf("expected publishBundle to forward X-API-Key")
	}
}
