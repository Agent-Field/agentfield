package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// A RUN A TRIGGER STARTED KEEPS ITS TREE. The control plane's trigger
// dispatch names the workflow but no run id; a call the reasoner makes must
// join that workflow's run, or every child becomes a run of its own with no
// parent and the job's rail shows nothing under the root.
func TestACallFromATriggerStartedRunJoinsItsWorkflow(t *testing.T) {
	var got atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/execute/async/support.search_docs":
			got.Store(r.Header.Get("X-Run-ID") + "|" + r.Header.Get("X-Parent-Execution-ID"))
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{"execution_id": "exec-child", "run_id": "wf-trigger", "status": "queued"})
		case r.Method == http.MethodGet && r.URL.Path == "/api/v1/executions/exec-child":
			_ = json.NewEncoder(w).Encode(map[string]any{"execution_id": "exec-child", "status": "succeeded", "result": map[string]any{"ok": true}})
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()
	a, err := New(Config{NodeID: "support", Version: "1", AgentFieldURL: server.URL, Logger: log.New(io.Discard, "", 0)})
	if err != nil {
		t.Fatal(err)
	}
	ctx := contextWithExecution(context.Background(), ExecutionContext{WorkflowID: "wf-trigger", ExecutionID: "exec-investigate"})
	if _, err := a.Call(ctx, "support.search_docs", map[string]any{"q": "negative totals"}); err != nil {
		t.Fatal(err)
	}
	if v, _ := got.Load().(string); v != "wf-trigger|exec-investigate" {
		t.Fatalf("the child was sent run|parent %q; want it in the trigger's workflow under its caller", v)
	}
}
