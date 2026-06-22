package capabilities

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/databricks"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/prompts"
)

func TestHandleDatabricksEventIsThinWebhookEntry(t *testing.T) {
	rt := Runtime{Config: config.Config{NodeID: "databricks-test"}}
	out, err := rt.handleDatabricksEvent(context.Background(), map[string]any{
		"event": map[string]any{
			"event_id":   "evt-1",
			"event_type": "TERMINATED",
			"databricks": map[string]any{
				"workspace": "workspace",
			},
		},
	})
	if err != nil {
		t.Fatalf("handleDatabricksEvent: %v", err)
	}
	got := out.(map[string]any)
	if got["event_id"] != "evt-1" || got["event_type"] != "TERMINATED" || got["recommended_call"] != "query_readonly" {
		t.Fatalf("bad event output: %#v", got)
	}
}

func TestExplainResultUsesDatabricksAIQuery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"statement_id":"ai-1",
			"status":{"state":"SUCCEEDED"},
			"manifest":{"schema":{"columns":[{"name":"response"}]}},
			"result":{"data_array":[["Revenue dipped because refunds increased."]]}
		}`))
	}))
	defer server.Close()
	client, err := databricks.NewClient(databricks.Config{WorkspaceURL: server.URL, Token: "pat", WarehouseID: "wh", AIEndpoint: "model", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	store := &prompts.Store{}
	store, err = prompts.Load("../../../prompts/default-prompts.yaml", "")
	if err != nil {
		t.Fatalf("load prompts: %v", err)
	}
	rt := Runtime{Databricks: client, Prompts: store}
	out, err := rt.explainResult(context.Background(), map[string]any{"question": "why?", "result": map[string]any{"rows": []any{1}}})
	if err != nil {
		t.Fatalf("explainResult: %v", err)
	}
	got := out.(map[string]any)
	if got["answer"] != "Revenue dipped because refunds increased." {
		t.Fatalf("bad answer: %#v", got)
	}
}
